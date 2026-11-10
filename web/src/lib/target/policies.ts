import { createRng, nextInt } from "../rng.ts";
import { streams, type Config } from "./config.ts";
import { clamp, resourceShare } from "./attention.ts";
import { isOccluded } from "./world.ts";
import { measurementSigma } from "./measurement.ts";
import type { Action, Move, Observation, Policy, Truth } from "./episode.ts";

export function createNeutralPolicy(config: Config, seed: number): Policy {
  const rng = createRng(seed, streams.policy);
  let step = 0;
  return (_observation) => {
    step++;
    // Per decision: command index (row-major), effort, paddle, then final answer only.
    const command = nextInt(rng, 9);
    return {
      command: { row: (Math.floor(command / 3) - 1) as Move, col: ((command % 3) - 1) as Move },
      effort: nextInt(rng, 2) as 0 | 1,
      paddleMove: (nextInt(rng, 3) - 1) as Move,
      ...(step === config.episodeSteps ? { answer: nextInt(rng, config.hueCategories) } : {}),
    };
  };
}

export type IdealPolicy = (observation: Observation, truth: Truth) => Action;

export function hueCategory(config: Config, hue: number): number {
  return clamp(Math.floor(hue * config.hueCategories), 0, config.hueCategories - 1);
}

export function idealAttentionAction(
  config: Config,
  truth: Truth,
): Pick<Action, "command" | "effort"> {
  const { world, attention } = truth;
  const target = world.target;
  // The specified forecast is position + velocity, without sampling the next motion.
  // Math.round resolves half-integer ties toward +Infinity, before clamping the command.
  const row = clamp(Math.round(target.row + target.velocity.row - attention.center.row), -1, 1);
  const col = clamp(Math.round(target.col + target.velocity.col - attention.center.col), -1, 1);
  return {
    command: { row: row as Move, col: col as Move },
    effort: resourceShare(config, attention, target) < config.effortThreshold ? 1 : 0,
  };
}

export function createIdealPolicy(config: Config, enhanced = true): IdealPolicy {
  let weightedHue = 0;
  let weightSum = 0;
  let exactHue = 0;
  let exactCount = 0;
  return (observation, truth) => {
    const { world, attention } = truth;
    const target = world.target;
    if (world.t >= world.jumpStep && !isOccluded(config, world.occluder, target)) {
      const cell = target.row * config.fieldSize + target.col;
      const hue = observation.measurement[config.fieldSize * config.fieldSize + cell]!;
      const sigma = measurementSigma(config, attention.resource[cell]!, enhanced);
      if (sigma === 0) {
        // Zero-variance observations dominate finite weights without Infinity / Infinity.
        exactHue += hue;
        exactCount++;
      } else {
        const weight = 1 / (sigma * sigma);
        weightedHue += weight * hue;
        weightSum += weight;
      }
    }
    const action: Action = {
      ...idealAttentionAction(config, truth),
      paddleMove: Math.sign(target.col - observation.paddle) as Move,
    };
    if (world.t === config.episodeSteps) {
      // With no eligible measurement, choose category zero; never use the true hue.
      action.answer =
        exactCount > 0
          ? hueCategory(config, exactHue / exactCount)
          : weightSum > 0
            ? hueCategory(config, weightedHue / weightSum)
            : 0;
    }
    return action;
  };
}
