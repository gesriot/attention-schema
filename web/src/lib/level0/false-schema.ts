import { createBatch, forward, type Mlp } from "../nn/mlp.ts";
import { appendFrame, greedyAction } from "./dqn.ts";
import { attentionMoves, createWorld, decodeAction, stepWorld, type WorldConfig } from "./world.ts";

export const falseShifts = [-3, -2, -1, 1, 2, 3] as const;
export const falseScopes = ["history", "latest", "cleaned"] as const;
export type FalseScope = (typeof falseScopes)[number];
export type ShiftResult = {
  scope: FalseScope;
  shift: number;
  decisions: number;
  // Mean change of the column command against the shift; 0 means the schema does not steer.
  direction: number;
  changed: number;
};
export type ConsequenceResult = {
  shift: number;
  episodes: number;
  focusClean: number;
  focusFalse: number;
  trackingClean: number;
  trackingFalse: number;
};
export type FalseSchemaResult = { shifts: ShiftResult[]; consequences: ConsequenceResult[] };
export type Act = (observation: Uint8Array) => number;

// The last decision whose result is scored by tracking, and the step of the one-step lie.
const lastScoredDecision = 8;
const falseStep = 5;
const frameSize = 200;
const firstSeed = 4100000000;

export function modelPolicy(model: Mlp): Act {
  const batch = createBatch(model, 1);
  return (observation) => {
    batch.activations[0]!.set(observation);
    return greedyAction(forward(model, batch));
  };
}

// Moves the window drawn in the field and nothing else. False if it would leave the grid.
// "cleaned" also blanks the visual cells under the false square, except row 9 with the paddle,
// so the noise there cannot pass for a stimulus. A ball that left the real window for that
// area goes too, which is rare in a trained network.
export function shiftField(
  source: Uint8Array,
  target: Uint8Array,
  shift: number,
  scope: FalseScope,
): boolean {
  target.set(source);
  const first = scope === "latest" ? source.length - frameSize : 0;
  for (let offset = first; offset < source.length; offset += frameSize) {
    target.fill(0, offset + 100, offset + frameSize);
    for (let cell = 0; cell < 100; cell++) {
      if (source[offset + 100 + cell] === 0) continue;
      const col = cell % 10;
      if (col + shift < 0 || col + shift > 9) return false;
      target[offset + 100 + cell + shift] = 1;
    }
    if (scope !== "cleaned") continue;
    for (let cell = 0; cell < 90; cell++) {
      if (target[offset + 100 + cell] === 1 && source[offset + 100 + cell] === 0) {
        target[offset + cell] = 0;
      }
    }
  }
  return true;
}

function columnMove(config: WorldConfig, action: number): number {
  return attentionMoves(config.variant)[decodeAction(config, action).attention]![0];
}

function playEpisode(
  act: Act,
  config: WorldConfig,
  seed: number,
  shift: number | null,
): { focus: number; tracking: number } | null {
  const world = createWorld(config, seed);
  const observation = new Uint8Array(frameSize * 10);
  const shifted = new Uint8Array(frameSize * 10);
  let focus = 0;
  let tracking = 0;
  appendFrame(observation, world.frame, true);
  while (world.t < 10) {
    let input = observation;
    if (world.t === falseStep && shift !== null) {
      if (!shiftField(observation, shifted, shift, "history")) return null;
      input = shifted;
    }
    const step = world.t;
    stepWorld(world, act(input));
    appendFrame(observation, world.frame, false);
    if (step === falseStep) focus = world.reward > 0 ? 1 : 0;
    if (step >= falseStep && world.t < 10) tracking += world.reward;
  }
  return { focus, tracking };
}

export function probeFalseSchema(
  act: Act,
  config: WorldConfig,
  episodes: number,
): FalseSchemaResult {
  const shifts = falseScopes.flatMap((scope) =>
    falseShifts.map((shift) => ({ scope, shift, decisions: 0, direction: 0, changed: 0 })),
  );
  const observation = new Uint8Array(frameSize * 10);
  const shifted = new Uint8Array(frameSize * 10);
  for (let episode = 1; episode <= episodes; episode++) {
    const world = createWorld(config, firstSeed + episode);
    appendFrame(observation, world.frame, true);
    while (world.t < 10) {
      const action = act(observation);
      if (world.t <= lastScoredDecision) {
        for (const entry of shifts) {
          if (!shiftField(observation, shifted, entry.shift, entry.scope)) continue;
          const change = columnMove(config, act(shifted)) - columnMove(config, action);
          entry.decisions++;
          entry.direction -= Math.sign(entry.shift) * change;
          if (change !== 0) entry.changed++;
        }
      }
      stepWorld(world, action);
      appendFrame(observation, world.frame, false);
    }
  }
  for (const entry of shifts) {
    entry.direction /= entry.decisions;
    entry.changed /= entry.decisions;
  }
  const consequences = falseShifts.map((shift) => {
    const sum = {
      shift,
      episodes: 0,
      focusClean: 0,
      focusFalse: 0,
      trackingClean: 0,
      trackingFalse: 0,
    };
    for (let episode = 1; episode <= episodes; episode++) {
      const lied = playEpisode(act, config, firstSeed + episode, shift);
      if (lied === null) continue;
      const clean = playEpisode(act, config, firstSeed + episode, null);
      if (clean === null) throw new Error("Эпизод без подмены всегда завершается");
      sum.episodes++;
      sum.focusClean += clean.focus;
      sum.focusFalse += lied.focus;
      sum.trackingClean += clean.tracking;
      sum.trackingFalse += lied.tracking;
    }
    sum.focusClean /= sum.episodes;
    sum.focusFalse /= sum.episodes;
    sum.trackingClean /= sum.episodes;
    sum.trackingFalse /= sum.episodes;
    return sum;
  });
  return { shifts, consequences };
}
