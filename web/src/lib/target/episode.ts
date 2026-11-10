import { createRng, type Rng } from "../rng.ts";
import { streams, validateConfig, type Config } from "./config.ts";
import { advanceWorld, createWorld, type Position, type World } from "./world.ts";
import { advanceAttention, clamp, initialAttention, type Attention } from "./attention.ts";
import { measure, telemetry } from "./measurement.ts";

export type Move = -1 | 0 | 1;
export type Command = { row: Move; col: Move };
export type Action = { command: Command; effort: 0 | 1; paddleMove: Move; answer?: number };
export type Observation = {
  measurement: Float64Array;
  previousCommand: Position;
  previousEffort: number;
  previousPaddleMove: number;
  paddle: number;
  telemetry: Float64Array;
};
export type Truth = { world: World; attention: Attention };
export type Score = { caught: boolean; correct: boolean; effortSteps: number; total: number };
export type Episode = {
  config: Config;
  enhanced: boolean;
  rng: { world: Rng; flash: Rng; attention: Rng; measurement: Rng; telemetry: Rng };
  truth: Truth;
  observation: Observation;
  paddle: number;
  effortSteps: number;
  score: Score | null;
};
export type Policy = (observation: Observation) => Action;

function observe(episode: Episode, previous: Action): Observation {
  const { config, truth, rng } = episode;
  return {
    measurement: measure(
      config,
      truth.world.frame,
      truth.attention.resource,
      episode.enhanced,
      rng.measurement,
    ),
    previousCommand: { ...previous.command },
    previousEffort: previous.effort,
    previousPaddleMove: previous.paddleMove,
    paddle: episode.paddle,
    telemetry: telemetry(config, truth.attention.resource, rng.telemetry),
  };
}

export function createEpisode(config: Config, seed: number, enhanced = true): Episode {
  validateConfig(config);
  const rng = {
    world: createRng(seed, streams.world),
    flash: createRng(seed, streams.flash),
    attention: createRng(seed, streams.attention),
    measurement: createRng(seed, streams.measurement),
    telemetry: createRng(seed, streams.telemetry),
  };
  const world = createWorld(config, rng.world, rng.flash);
  const episode: Episode = {
    config,
    enhanced,
    rng,
    truth: { world, attention: initialAttention(config, world.target) },
    observation: {} as Observation,
    paddle: config.paddleStart,
    effortSteps: 0,
    score: null,
  };
  // No preceding action at t = 1: rest command, no effort, no paddle move.
  episode.observation = observe(episode, { command: { row: 0, col: 0 }, effort: 0, paddleMove: 0 });
  return episode;
}

export function applyAction(episode: Episode, action: Action): void {
  if (episode.score !== null) throw new Error("Эпизод уже завершён");
  const { config, truth, rng } = episode;
  for (const move of [action.command.row, action.command.col, action.paddleMove]) {
    if (move !== -1 && move !== 0 && move !== 1) throw new RangeError("Недопустимый ход");
  }
  if (action.effort !== 0 && action.effort !== 1) throw new RangeError("Недопустимое усилие");
  if (
    truth.world.t === config.episodeSteps &&
    (!Number.isInteger(action.answer) ||
      action.answer! < 0 ||
      action.answer! >= config.hueCategories)
  ) {
    throw new RangeError("На последнем шаге нужна категория ответа");
  }
  // Paddle moves immediately after the decision, including the final decision.
  // Every chosen effort is charged, including the last, though it has no next measurement.
  episode.paddle = clamp(episode.paddle + action.paddleMove, config.paddleMin, config.paddleMax);
  episode.effortSteps += action.effort;
  if (truth.world.t === config.episodeSteps) {
    const caught = Math.abs(episode.paddle - truth.world.target.col) <= config.catchRadius;
    const correct = action.answer === truth.world.target.category;
    episode.score = {
      caught,
      correct,
      effortSteps: episode.effortSteps,
      total:
        (caught ? config.catchReward : -config.catchReward) +
        (correct ? config.answerReward : -config.answerReward) -
        config.effortCost * episode.effortSteps,
    };
    return;
  }
  advanceWorld(config, truth.world, rng.world, rng.flash);
  truth.attention = advanceAttention(
    config,
    truth.attention,
    action.command,
    action.effort,
    truth.world.flash,
    rng.attention,
  );
  episode.observation = observe(episode, action);
}

export function runEpisode(episode: Episode, policy: Policy): Score {
  while (episode.score === null) applyAction(episode, policy(episode.observation));
  return episode.score;
}
