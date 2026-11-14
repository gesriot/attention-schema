import { createAdam, adamStep, type Adam } from "../nn/adam.ts";
import { backwardGru } from "../nn/gru.ts";
import { createRng, nextInt, type Rng } from "../rng.ts";
import { streams, validateConfig, type Config } from "./config.ts";
import {
  advanceController,
  c6InputWidth,
  controllerAction,
  createController,
  createControllerBatch,
  distribution,
  encodeC6,
  forwardController,
  type Controller,
  type ControllerBatch,
} from "./controller.ts";
import { applyAction, createEpisode, type Action, type Score } from "./episode.ts";
import { createPerception, updatePerception } from "./perception.ts";

export const c6TrainSeed = 7000000001;
export const c6EvalSeed = 8000000001;
export const c6RunSpan = 10000000;

export type Rollout = {
  size: number;
  steps: number;
  input: number;
  inputs: Float64Array;
  actions: Int32Array;
  logProbs: Float64Array;
  values: Float64Array;
  rewards: Float64Array;
  advantages: Float64Array;
  targets: Float64Array;
};

export function createRollout(size: number, steps: number, input: number): Rollout {
  const samples = size * steps;
  return {
    size,
    steps,
    input,
    inputs: new Float64Array(samples * input),
    actions: new Int32Array(samples * 4).fill(-1),
    logProbs: new Float64Array(samples),
    values: new Float64Array(samples),
    rewards: new Float64Array(samples),
    advantages: new Float64Array(samples),
    targets: new Float64Array(samples),
  };
}

export function trainingSeed(config: Config, run: number, update: number, episode: number): number {
  if (
    !Number.isSafeInteger(run) ||
    run < 1 ||
    run > 99 ||
    !Number.isSafeInteger(update) ||
    update < 1 ||
    update > config.ppoUpdates ||
    !Number.isSafeInteger(episode) ||
    episode < 0 ||
    episode >= config.ppoEpisodes ||
    config.ppoUpdates * config.ppoEpisodes > c6RunSpan
  )
    throw new RangeError("Недопустимые индексы обучающего эпизода C6");
  return c6TrainSeed + (run - 1) * c6RunSpan + (update - 1) * config.ppoEpisodes + episode;
}

export function stepReward(config: Config, action: Action, score: Score | null): number {
  let reward = -config.effortCost * action.effort;
  if (score !== null) {
    reward += score.caught ? config.catchReward : -config.catchReward;
    reward += score.correct ? config.answerReward : -config.answerReward;
  }
  return reward;
}

// Raw targets are saved before global population-variance normalization.
// GAE: ascending episodes, descending time; reductions: ascending stored samples.
export function computeAdvantages(data: Rollout, lambda: number): void {
  for (let episode = 0; episode < data.size; episode++) {
    let following = 0;
    for (let step = data.steps - 1; step >= 0; step--) {
      const sample = episode * data.steps + step;
      const next = step + 1 === data.steps ? 0 : data.values[sample + 1]!;
      const delta = data.rewards[sample]! + next - data.values[sample]!;
      following = delta + lambda * following;
      data.advantages[sample] = following;
      data.targets[sample] = following + data.values[sample]!;
    }
  }
  let mean = 0;
  for (const advantage of data.advantages) mean += advantage;
  mean /= data.advantages.length;
  let variance = 0;
  for (const advantage of data.advantages) variance += (advantage - mean) * (advantage - mean);
  const denominator = Math.sqrt(variance / data.advantages.length) + 1e-8;
  for (let i = 0; i < data.advantages.length; i++)
    data.advantages[i] = (data.advantages[i]! - mean) / denominator;
}

export function collectC6(
  config: Config,
  model: Controller,
  run: number,
  update: number,
  rng: Rng,
): Rollout {
  const data = createRollout(config.ppoEpisodes, config.episodeSteps, model.gru.input);
  const batch = createControllerBatch(model, data.size, data.steps);
  batch.recurrent.inputs = data.inputs;
  const episodes = Array.from({ length: data.size }, (_, i) =>
    createEpisode(config, trainingSeed(config, run, update, i)),
  );
  const perceptions = episodes.map(() => createPerception());
  for (let step = 0; step < data.steps; step++) {
    for (let i = 0; i < data.size; i++) {
      const episode = episodes[i]!;
      perceptions[i] = updatePerception(config, perceptions[i]!, episode.observation.measurement);
      data.inputs.set(
        encodeC6(
          config,
          perceptions[i]!,
          episode.observation,
          episode.truth.attention.resource,
          step === 0,
        ),
        (i * data.steps + step) * data.input,
      );
    }
    advanceController(model, batch, step);
    for (let i = 0; i < data.size; i++) {
      const sample = i * data.steps + step;
      const chosen = controllerAction(model, batch.logits, sample, step + 1 === data.steps, rng);
      data.actions.set(chosen.choices, sample * 4);
      data.logProbs[sample] = chosen.logProb;
      data.values[sample] = batch.logits[sample * model.outputs + model.heads[4]!.output]!;
      applyAction(episodes[i]!, chosen.action);
      data.rewards[sample] = stepReward(config, chosen.action, episodes[i]!.score);
    }
  }
  computeAdvantages(data, config.ppoLambda);
  return data;
}

export type PpoLossConfig = Pick<Config, "ppoClip" | "ppoValueCoef" | "ppoEntropyCoef">;

// The batch contains full episodes from data in the supplied order; all sums are
// sequence-major, then time, head, output, hidden coordinate. GRU uses its own
// documented reverse-time order. Equality at a clip boundary takes the active branch.
export function ppoGradient(
  config: PpoLossConfig,
  model: Controller,
  batch: ControllerBatch,
  data: Rollout,
  episodes: readonly number[],
): number {
  if (
    batch.recurrent.size !== episodes.length ||
    batch.recurrent.steps !== data.steps ||
    model.gru.input !== data.input
  )
    throw new RangeError("Размеры мини-батча PPO не совпадают");
  const { hidden } = model.gru;
  const count = episodes.length * data.steps;
  const inverse = 1 / count;
  for (let i = 0; i < episodes.length; i++) {
    const begin = episodes[i]! * data.steps * data.input;
    batch.recurrent.inputs.set(
      data.inputs.subarray(begin, begin + data.steps * data.input),
      i * data.steps * data.input,
    );
  }
  forwardController(model, batch);
  batch.delta.fill(0);
  batch.gradient.fill(0);
  batch.outputGradient.fill(0);
  let loss = 0;
  for (let i = 0; i < episodes.length; i++) {
    for (let step = 0; step < data.steps; step++) {
      const sample = i * data.steps + step;
      const source = episodes[i]! * data.steps + step;
      const base = sample * model.outputs;
      const heads = model.heads.slice(0, step + 1 === data.steps ? 4 : 3);
      const distributions = heads.map((head) =>
        distribution(batch.logits, base + head.output, head.width),
      );
      let logProb = 0;
      let entropy = 0;
      for (let h = 0; h < heads.length; h++) {
        logProb += distributions[h]!.logs[data.actions[source * 4 + h]!]!;
        entropy += distributions[h]!.entropy;
      }
      const ratio = Math.exp(logProb - data.logProbs[source]!);
      const advantage = data.advantages[source]!;
      const clipped = Math.max(1 - config.ppoClip, Math.min(1 + config.ppoClip, ratio));
      const inactive =
        (advantage >= 0 && ratio > 1 + config.ppoClip) ||
        (advantage < 0 && ratio < 1 - config.ppoClip);
      const coefficient = inactive ? 0 : -advantage * ratio;
      const valueIndex = base + model.heads[4]!.output;
      const error = batch.logits[valueIndex]! - data.targets[source]!;
      loss +=
        -Math.min(ratio * advantage, clipped * advantage) +
        config.ppoValueCoef * error * error -
        config.ppoEntropyCoef * entropy;
      for (let h = 0; h < heads.length; h++) {
        const head = heads[h]!;
        const values = distributions[h]!;
        for (let j = 0; j < head.width; j++) {
          const p = values.probabilities[j]!;
          batch.delta[base + head.output + j] =
            (coefficient * ((data.actions[source * 4 + h] === j ? 1 : 0) - p) +
              config.ppoEntropyCoef * p * (values.logs[j]! + values.entropy)) *
            inverse;
        }
      }
      batch.delta[valueIndex] = 2 * config.ppoValueCoef * error * inverse;
      const state = (i * (data.steps + 1) + step + 1) * hidden;
      for (const head of model.heads) {
        for (let j = 0; j < head.width; j++) {
          const d = batch.delta[base + head.output + j]!;
          const bias = head.offset + head.width * hidden + j;
          batch.gradient[bias] = batch.gradient[bias]! + d;
          for (let k = 0; k < hidden; k++) {
            const w = head.offset + j * hidden + k;
            batch.gradient[w] = batch.gradient[w]! + d * batch.recurrent.states[state + k]!;
            const dh = sample * hidden + k;
            batch.outputGradient[dh] =
              batch.outputGradient[dh]! + d * head.weights[j * hidden + k]!;
          }
        }
      }
    }
  }
  loss *= inverse;
  if (!Number.isFinite(loss)) throw new Error("Неконечные потери PPO");
  backwardGru(model.gru, batch.recurrent, batch.outputGradient, batch.gradient);
  return loss;
}

export function clipGradient(gradient: Float64Array, maximum: number): number {
  let squared = 0;
  for (const value of gradient) squared += value * value;
  const norm = Math.sqrt(squared);
  if (!Number.isFinite(norm)) throw new Error("Неконечная норма градиента PPO");
  if (norm > maximum) {
    const scale = maximum / norm;
    for (let i = 0; i < gradient.length; i++) gradient[i] = gradient[i]! * scale;
  }
  return norm;
}

// Restart from ascending episode indices each epoch; descending Fisher-Yates
// consumes exactly size-1 draws (nextInt(i+1)). Last minibatch may be smaller.
export function shuffleEpisodes(size: number, rng: Rng): number[] {
  const order = Array.from({ length: size }, (_, i) => i);
  for (let i = size - 1; i > 0; i--) {
    const j = nextInt(rng, i + 1);
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  return order;
}

export function optimizePpo(
  config: Config,
  model: Controller,
  data: Rollout,
  adam: Adam,
  rate: number,
  rng: Rng,
): void {
  const batches = new Map<number, ControllerBatch>();
  for (let epoch = 0; epoch < config.ppoEpochs; epoch++) {
    const order = shuffleEpisodes(data.size, rng);
    for (let begin = 0; begin < data.size; begin += config.ppoMinibatch) {
      const episodes = order.slice(begin, begin + config.ppoMinibatch);
      let batch = batches.get(episodes.length);
      if (batch === undefined) {
        batch = createControllerBatch(model, episodes.length, data.steps);
        batches.set(episodes.length, batch);
      }
      ppoGradient(config, model, batch, data, episodes);
      clipGradient(batch.gradient, config.ppoMaxGradNorm);
      adamStep(model.parameters, batch.gradient, adam, rate);
      for (const value of model.parameters)
        if (!Number.isFinite(value)) throw new Error("Неконечные параметры PPO");
    }
  }
}

export type C6Metrics = {
  reward: number;
  caught: number;
  correct: number;
  effort: number;
  focus: number;
};
export type Checkpoint = C6Metrics & { update: number };
export type C6Result = { run: number; rate: number; checkpoints: Checkpoint[] };
export type C6Progress =
  | { type: "updateStart" | "updateEnd"; update: number }
  | { type: "checkpoint"; point: Checkpoint; collectionReward: number | null };

export function evaluateC6(config: Config, model: Controller): C6Metrics {
  const metrics: C6Metrics = { reward: 0, caught: 0, correct: 0, effort: 0, focus: 0 };
  // Evaluation chunks bound memory; no sampling or optimizer generators are used.
  // Reward totals are reduced in ascending episode order, independent of chunk size.
  for (let begin = 0; begin < config.ppoEvalEpisodes; begin += config.ppoMinibatch) {
    const size = Math.min(config.ppoMinibatch, config.ppoEvalEpisodes - begin);
    const episodes = Array.from({ length: size }, (_, i) =>
      createEpisode(config, c6EvalSeed + begin + i),
    );
    const perceptions = episodes.map(() => createPerception());
    const batch = createControllerBatch(model, size, config.episodeSteps);
    for (let step = 0; step < config.episodeSteps; step++) {
      for (let i = 0; i < size; i++) {
        const episode = episodes[i]!;
        perceptions[i] = updatePerception(config, perceptions[i]!, episode.observation.measurement);
        batch.recurrent.inputs.set(
          encodeC6(
            config,
            perceptions[i]!,
            episode.observation,
            episode.truth.attention.resource,
            step === 0,
          ),
          (i * config.episodeSteps + step) * model.gru.input,
        );
        const { center } = episode.truth.attention;
        const target = episode.truth.world.target;
        if (Math.max(Math.abs(center.row - target.row), Math.abs(center.col - target.col)) <= 1)
          metrics.focus++;
      }
      advanceController(model, batch, step);
      for (let i = 0; i < size; i++) {
        const { action } = controllerAction(
          model,
          batch.logits,
          i * config.episodeSteps + step,
          step + 1 === config.episodeSteps,
        );
        metrics.effort += action.effort;
        applyAction(episodes[i]!, action);
      }
    }
    for (const episode of episodes) {
      metrics.reward += episode.score!.total;
      metrics.caught += Number(episode.score!.caught);
      metrics.correct += Number(episode.score!.correct);
    }
  }
  metrics.reward /= config.ppoEvalEpisodes;
  metrics.caught /= config.ppoEvalEpisodes;
  metrics.correct /= config.ppoEvalEpisodes;
  metrics.effort /= config.ppoEvalEpisodes * config.episodeSteps;
  metrics.focus /= config.ppoEvalEpisodes * config.episodeSteps;
  return metrics;
}

export function trainC6(
  config: Config,
  run: number,
  rate: number,
  progress?: (event: C6Progress) => void,
): C6Result {
  validateConfig(config);
  trainingSeed(config, run, 1, 0);
  if (!Number.isFinite(rate) || rate <= 0)
    throw new RangeError("Скорость обучения должна быть положительной и конечной");
  const rng = createRng(run, streams.controller);
  const actionRng = createRng(run, streams.controllerAction);
  const model = createController(
    c6InputWidth(config),
    config.controllerHidden,
    config.hueCategories,
    rng,
  );
  const adam = createAdam(model.parameters.length);
  const result: C6Result = { run, rate, checkpoints: [] };
  let collectionSum = 0;
  let collectionEpisodes = 0;
  const checkpoint = (update: number) => {
    const point = { update, ...evaluateC6(config, model) };
    result.checkpoints.push(point);
    progress?.({
      type: "checkpoint",
      point,
      collectionReward: collectionEpisodes === 0 ? null : collectionSum / collectionEpisodes,
    });
    collectionSum = 0;
    collectionEpisodes = 0;
  };
  checkpoint(0);
  for (let update = 1; update <= config.ppoUpdates; update++) {
    progress?.({ type: "updateStart", update });
    const data = collectC6(config, model, run, update, actionRng);
    for (let episode = 0; episode < data.size; episode++) {
      let reward = 0;
      for (let step = 0; step < data.steps; step++)
        reward += data.rewards[episode * data.steps + step]!;
      collectionSum += reward;
    }
    collectionEpisodes += data.size;
    optimizePpo(config, model, data, adam, rate, rng);
    progress?.({ type: "updateEnd", update });
    if (update % config.ppoEvalEvery === 0 || update === config.ppoUpdates) checkpoint(update);
  }
  return result;
}
