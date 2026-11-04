import { createRng, nextInt } from "../rng.ts";
import {
  actionCount,
  attentionMoves,
  encodeAction,
  movePaddle,
  moveWindow,
  type ActionSpace,
  type Position,
  type Variant,
} from "./world.ts";

export type Policy = (frame: Uint8Array, t: number) => number;

export function chebyshev(a: Position, b: Position): number {
  return Math.max(Math.abs(a.row - b.row), Math.abs(a.col - b.col));
}

export function chooseAttention(variant: Variant, center: Position, target: Position): number {
  let best = 0;
  let distance = Infinity;
  for (let action = 0; action < attentionMoves(variant).length; action++) {
    const nextDistance = chebyshev(moveWindow(center, variant, action), target);
    if (nextDistance < distance) {
      best = action;
      distance = nextDistance;
    }
  }
  return best;
}

export function createRandomPolicy(config: ActionSpace, seed: number): Policy {
  // Policy draws use stream 2, independent of world and noise draws.
  const rng = createRng(seed, 2);
  const count = actionCount(config);
  return (_frame, _t) => nextInt(rng, count);
}

function initialWindow(frame: Uint8Array): Position {
  for (let col = 1; col <= 8; col++) {
    let pixels = 0;
    let pixelRow = -1;
    for (let row = 0; row <= 2; row++) {
      for (let x = col - 1; x <= col + 1; x++) {
        if (frame[row * 10 + x] === 1) {
          pixels++;
          pixelRow = row;
        }
      }
    }
    if (pixels === 1 && pixelRow === 0) return { row: 1, col };
  }
  return { row: 1, col: 4 };
}

function observeBall(frame: Uint8Array, center: Position, forecast: Position): Position | null {
  let observation: Position | null = null;
  let distance = Infinity;
  for (let row = center.row - 1; row <= Math.min(8, center.row + 1); row++) {
    for (let col = center.col - 1; col <= center.col + 1; col++) {
      if (frame[row * 10 + col] !== 1) continue;
      const candidate = { row, col };
      const nextDistance = chebyshev(candidate, forecast);
      if (nextDistance < distance) {
        observation = candidate;
        distance = nextDistance;
      }
    }
  }
  return observation;
}

export function createTrackerPolicy(config: ActionSpace): Policy {
  let window = { row: 1, col: 4 };
  let paddle = 4;
  let nextTie: 0 | 1 = 1;
  let predictedCol = 4;
  let lastObservation: Position | null = null;
  let shift = 0;

  return (frame, t) => {
    if (t === 1) {
      window = initialWindow(frame);
      predictedCol = window.col;
    }
    const observation = observeBall(frame, window, { row: t - 1, col: predictedCol });
    if (observation !== null) {
      // Use the raw difference between successive observations, without smoothing.
      if (lastObservation !== null) shift = observation.col - lastObservation.col;
      lastObservation = observation;
      predictedCol = observation.col;
    }
    predictedCol += shift;
    const attention = chooseAttention(config.variant, window, { row: t, col: predictedCol });
    let direction: 0 | 1;
    if (paddle === predictedCol) {
      direction = nextTie;
      nextTie = nextTie === 1 ? 0 : 1;
    } else {
      direction = paddle < predictedCol ? 1 : 0;
    }
    window = moveWindow(window, config.variant, attention);
    paddle = movePaddle(paddle, direction);
    return encodeAction(config, {
      paddle: direction,
      attention,
      resource: config.schema === "free" ? attention : 0,
    });
  };
}
