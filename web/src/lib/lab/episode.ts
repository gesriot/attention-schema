import {
  appendFrame,
  clearObservationField,
  conditionConfig,
  greedyAction,
} from "../level0/dqn.ts";
import { shiftField } from "../level0/false-schema.ts";
import { createWorld, stepWorld, type World } from "../level0/world.ts";
import { createBatch, forward, type Batch, type Mlp } from "../nn/mlp.ts";

export const labConditions = ["schema", "none", "commands"] as const;
export type LabCondition = (typeof labConditions)[number];
export type Intervention = "normal" | "windowOff" | "fieldOff" | "falseSchema";
export const noiseLevels = [0, 0.25, 0.5, 0.75, 1] as const;
export type LabSettings = {
  condition: LabCondition;
  noise: number;
  intervention: Intervention;
  falseShift: number;
  falseStep: number;
};
export const defaultSettings: LabSettings = {
  condition: "schema",
  noise: 0.5,
  intervention: "normal",
  falseShift: 2,
  falseStep: 5,
};
export type FalseStatus = "none" | "pending" | "applied" | "outside" | "unavailable";
export type LabEpisode = {
  settings: LabSettings;
  seed: number;
  model: Mlp;
  batch: Batch;
  world: World;
  observation: Uint8Array;
  input: Uint8Array;
  values: Float64Array;
  action: number | null;
  falseStatus: FalseStatus;
};
export type LabView = {
  step: number;
  frame: Uint8Array;
  values: Float64Array;
  action: number | null;
  tracking: number;
  catching: number;
  falseStatus: FalseStatus;
  truth: {
    ball: { row: number; col: number };
    window: { row: number; col: number };
    paddle: number;
    landingCol: number;
    windowActive: boolean;
  };
};

// This function accepts observation bytes only. Evaluator truth never enters inference.
export function predictAction(
  model: Mlp,
  batch: Batch,
  observation: Uint8Array,
): { values: Float64Array; action: number } {
  batch.activations[0]!.set(observation);
  const values = forward(model, batch).slice();
  return { values, action: greedyAction(values) };
}

function prepareDecision(episode: LabEpisode): void {
  const { settings, world, observation } = episode;
  if (settings.intervention === "fieldOff") clearObservationField(observation);
  episode.input = observation.slice();
  if (settings.intervention === "falseSchema") {
    if (settings.condition !== "schema") episode.falseStatus = "unavailable";
    else if (world.t === settings.falseStep) {
      const shifted = new Uint8Array(observation.length);
      if (shiftField(observation, shifted, settings.falseShift, "history")) {
        episode.input = shifted;
        episode.falseStatus = "applied";
      } else {
        // The bench excludes this case. Keep the original input and display the skip.
        episode.falseStatus = "outside";
      }
    }
  }
  if (world.t === 10) {
    episode.action = null;
    episode.values = new Float64Array(0);
  } else Object.assign(episode, predictAction(episode.model, episode.batch, episode.input));
}

export function createLabEpisode(model: Mlp, settings: LabSettings, seed: number): LabEpisode {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 4294967295)
    throw new Error("Зерно должно быть целым числом от 0 до 4294967295.");
  if (model.sizes[0] !== 2000 || model.sizes.at(-1) !== 16)
    throw new Error("Для лаборатории нужна сеть с 2000 входами и 16 выходами.");
  if (
    !labConditions.includes(settings.condition) ||
    !noiseLevels.some((noise) => noise === settings.noise) ||
    !["normal", "windowOff", "fieldOff", "falseSchema"].includes(settings.intervention) ||
    !Number.isInteger(settings.falseStep) ||
    settings.falseStep < 1 ||
    settings.falseStep > 9 ||
    !Number.isInteger(settings.falseShift) ||
    Math.abs(settings.falseShift) > 3
  )
    throw new Error("Недопустимые настройки лаборатории.");
  const world = createWorld(
    {
      ...conditionConfig(settings.condition, settings.intervention !== "windowOff"),
      noise: settings.noise,
    },
    seed,
  );
  const observation = new Uint8Array(model.sizes[0]);
  appendFrame(observation, world.frame, true);
  const episode: LabEpisode = {
    settings: { ...settings },
    seed,
    model,
    batch: createBatch(model, 1),
    world,
    observation,
    input: observation.slice(),
    values: new Float64Array(0),
    action: null,
    falseStatus: settings.intervention === "falseSchema" ? "pending" : "none",
  };
  prepareDecision(episode);
  return episode;
}

export function advanceLabEpisode(episode: LabEpisode): void {
  if (episode.action === null) throw new Error("Эпизод завершён. Начните новый эпизод.");
  stepWorld(episode.world, episode.action);
  appendFrame(episode.observation, episode.world.frame, false);
  prepareDecision(episode);
}

export function labView(episode: LabEpisode): LabView {
  const { world } = episode;
  return {
    step: world.t,
    frame: episode.input.slice(-200),
    values: episode.values.slice(),
    action: episode.action,
    tracking: world.trackingScore,
    catching: world.catchScore,
    falseStatus: episode.falseStatus,
    truth: {
      ball: { ...world.ball },
      window: { ...world.window },
      paddle: world.paddle,
      landingCol: world.landingCol,
      windowActive: world.config.windowActive,
    },
  };
}
