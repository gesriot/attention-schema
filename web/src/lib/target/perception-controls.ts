import { firstSeed, type Config } from "./config.ts";
import { applyAction, createEpisode, type Action, type Move, type Observation } from "./episode.ts";
import { hueCategory, idealAttentionAction } from "./policies.ts";
import {
  createPerception,
  poolPerception,
  updatePerception,
  type Perception,
} from "./perception.ts";
import { handC6Action } from "./hand-c6.ts";

export type PerceptionControl = { correctRate: number; catchRate: number; trackingRate: number };
export type PerceptionControls = { F2: PerceptionControl; C0: PerceptionControl };

export function perceptionAction(
  config: Config,
  observation: Observation,
  perception: Perception,
  attentionAction: Pick<Action, "command" | "effort">,
  final: boolean,
): Action {
  const slot = perception.slots[0];
  return {
    ...attentionAction,
    paddleMove: Math.sign(slot.col - observation.paddle) as Move,
    ...(final ? { answer: hueCategory(config, slot.hue) } : {}),
  };
}

export function checkPerception(config: Config, episodes: number): PerceptionControls {
  const results: PerceptionControls = {
    F2: { correctRate: 0, catchRate: 0, trackingRate: 0 },
    C0: { correctRate: 0, catchRate: 0, trackingRate: 0 },
  };
  for (let index = 0; index < episodes; index++) {
    for (const enhanced of [true, false]) {
      const episode = createEpisode(config, firstSeed + index, enhanced);
      let perception = createPerception();
      const result = enhanced ? results.F2 : results.C0;
      for (let step = 1; step <= config.episodeSteps; step++) {
        perception = updatePerception(config, perception, episode.observation.measurement);
        const slot = perception.slots[0];
        const target = episode.truth.world.target;
        // Truth is used here only for evaluation and the prescribed oracle attention command.
        if (
          slot.active &&
          Math.max(Math.abs(slot.row - target.row), Math.abs(slot.col - target.col)) <= 1
        ) {
          result.trackingRate++;
        }
        applyAction(
          episode,
          perceptionAction(
            config,
            episode.observation,
            perception,
            idealAttentionAction(config, episode.truth),
            step === config.episodeSteps,
          ),
        );
      }
      result.correctRate += Number(episode.score!.correct);
      result.catchRate += Number(episode.score!.caught);
    }
  }
  for (const result of [results.F2, results.C0]) {
    result.correctRate /= episodes;
    result.catchRate /= episodes;
    result.trackingRate /= episodes * config.episodeSteps;
  }
  return results;
}

export type HandC6Control = { correctRate: number; catchRate: number; focusRate: number };

export function checkHandC6(config: Config, episodes: number): HandC6Control {
  const result: HandC6Control = { correctRate: 0, catchRate: 0, focusRate: 0 };
  // Ascending control seeds, then all decision steps, including the first and last.
  for (let index = 0; index < episodes; index++) {
    const episode = createEpisode(config, firstSeed + index);
    let perception = createPerception();
    for (let step = 1; step <= config.episodeSteps; step++) {
      perception = updatePerception(config, perception, episode.observation.measurement);
      const action = handC6Action(
        config,
        {
          perception: poolPerception(config, perception),
          paddle: episode.observation.paddle,
          resource: episode.truth.attention.resource,
        },
        step === config.episodeSteps,
      );
      // World truth and the stored attention center are read only by this evaluator.
      // Focus is measured before applying the current action, at the observed step.
      const { center } = episode.truth.attention;
      const target = episode.truth.world.target;
      if (Math.max(Math.abs(center.row - target.row), Math.abs(center.col - target.col)) <= 1)
        result.focusRate++;
      applyAction(episode, action);
    }
    result.correctRate += Number(episode.score!.correct);
    result.catchRate += Number(episode.score!.caught);
  }
  result.correctRate /= episodes;
  result.catchRate /= episodes;
  result.focusRate /= episodes * config.episodeSteps;
  return result;
}
