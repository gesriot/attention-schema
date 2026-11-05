import { createAdam, adamStep, type Adam } from "../nn/adam.ts";
import { createBatch, createMlp, forward, huberGradient, type Batch, type Mlp } from "../nn/mlp.ts";
import { createRng, nextInt, nextUnit, type Rng } from "../rng.ts";
import {
  createReplay,
  createReplayBatch,
  FIRST,
  LAST,
  MID,
  pushReplay,
  sampleReplay,
  type Replay,
  type ReplayBatch,
} from "./replay.ts";
import { createWorld, stepWorld, type World, type WorldConfig } from "./world.ts";

export type Condition = "schema" | "none" | "commands";
export type Check = "normal" | "windowOff" | "fieldOff" | "resourceRandom" | "resourceStill";
export const conditions: Condition[] = ["schema", "none", "commands"];
export type TrainingConfig = {
  sizes: number[];
  collectSteps: number;
  updates: number;
  batchSize: number;
  evalEpisodes: number;
};
export const publishedTraining: TrainingConfig = {
  sizes: [2000, 200, 200, 200, 16],
  collectSteps: 1000,
  updates: 100,
  batchSize: 100,
  evalEpisodes: 200,
};
export type TrainJob = {
  condition: Condition;
  subject: number;
  iterations: number;
  evalEvery: number;
  config: TrainingConfig;
};
export type CurvePoint = {
  iteration: number;
  check: Check;
  tracking: number;
  catching: number;
};
export type TrainResult = { condition: Condition; subject: number; points: CurvePoint[] };
export type DqnEnvironment = { world: WorldConfig; frames: number; discount: number };
export type DqnJob = {
  subject: number;
  iterations: number;
  evalEvery: number;
  config: TrainingConfig;
  environment: DqnEnvironment;
  checks: Check[];
};
export type Phase = "iteration" | "collect" | "update" | "evaluate";
export type TrainingEvent = {
  iteration: number;
  phase: Phase;
  stage: "start" | "end";
  loss: number;
};
export type Collector = {
  config: WorldConfig;
  subject: number;
  episode: number;
  world: World;
  observation: Uint8Array;
  replay: Replay;
  environmentSteps: number;
  boundarySteps: number;
  discount: number;
};

export function conditionConfig(condition: Condition, windowActive = true): WorldConfig {
  return {
    variant: "wilterson",
    schema: condition === "schema" ? "given" : condition === "none" ? "off" : "commands",
    noise: 0.5,
    windowActive,
  };
}

export function appendFrame(observation: Uint8Array, frame: Uint8Array, first: boolean): void {
  if (first) observation.fill(0);
  else observation.copyWithin(0, 200);
  observation.set(frame, observation.length - 200);
}

export function clearObservationField(observation: Uint8Array): void {
  for (let offset = 0; offset < observation.length; offset += 200) {
    observation.fill(0, offset + 100, offset + 200);
  }
}

export function greedyAction(values: Float64Array, offset = 0, count = values.length): number {
  let action = 0;
  for (let a = 1; a < count; a++) {
    if (values[offset + a]! > values[offset + action]!) action = a;
  }
  return action;
}

export function createEpsilonPolicy(model: Mlp, rng: Rng): (observation: Uint8Array) => number {
  const batch = createBatch(model, 1);
  const actions = model.sizes[model.sizes.length - 1]!;
  return (observation) => {
    // One exploration draw per step, plus one action draw only when exploring.
    if (nextUnit(rng) < 0.2) return nextInt(rng, actions);
    batch.activations[0]!.set(observation);
    return greedyAction(forward(model, batch));
  };
}

export function createCollector(
  condition: Condition | WorldConfig,
  subject: number,
  frames = 10,
  discount = 0.4,
): Collector {
  const config = typeof condition === "string" ? conditionConfig(condition) : condition;
  const world = createWorld(config, subject * 1000000 + 1);
  const observation = new Uint8Array(frames * 200);
  appendFrame(observation, world.frame, true);
  return {
    config,
    subject,
    episode: 1,
    world,
    observation,
    replay: createReplay(10000, observation.length),
    environmentSteps: 0,
    boundarySteps: 0,
    discount,
  };
}

export function collectSteps(
  collector: Collector,
  policy: (observation: Uint8Array) => number,
  count = 1000,
): void {
  let collected = 0;
  while (collected < count) {
    const action = policy(collector.observation);
    if (collector.world.t === 10) {
      // The driver calls the policy on LAST too; this action is ignored by reset.
      pushReplay(collector.replay, collector.observation, LAST, action, 0, collector.discount);
      collector.episode++;
      collector.world = createWorld(
        collector.config,
        collector.subject * 1000000 + collector.episode,
      );
      appendFrame(collector.observation, collector.world.frame, true);
      collector.boundarySteps++;
      continue;
    }
    const type = collector.world.t === 1 ? FIRST : MID;
    stepWorld(collector.world, action);
    pushReplay(
      collector.replay,
      collector.observation,
      type,
      action,
      collector.world.reward,
      collector.world.t === 10 ? 0 : collector.discount,
    );
    appendFrame(collector.observation, collector.world.frame, false);
    collector.environmentSteps++;
    collected++;
  }
}

export function updateDqn(
  model: Mlp,
  adam: Adam,
  batch: Batch,
  nextBatch: Batch,
  sample: ReplayBatch,
  targets: Float64Array,
): number {
  nextBatch.activations[0]!.set(sample.nextObservations);
  const nextValues = forward(model, nextBatch);
  const outputs = model.sizes[model.sizes.length - 1]!;
  for (let i = 0; i < sample.size; i++) {
    const offset = i * outputs;
    const best = greedyAction(nextValues, offset, outputs);
    targets[i] = sample.rewards[i]! + sample.discounts[i]! * nextValues[offset + best]!;
  }
  batch.activations[0]!.set(sample.observations);
  forward(model, batch);
  const loss = huberGradient(model, batch, sample.actions, targets, sample.masks);
  if (!Number.isFinite(loss)) throw new Error("Потери обучения перестали быть конечными");
  adamStep(model.parameters, batch.gradient, adam);
  return loss;
}

export function checksFor(condition: Condition): Check[] {
  return condition === "none" ? ["normal", "windowOff"] : ["normal", "windowOff", "fieldOff"];
}

export function evaluateDqn(
  model: Mlp,
  condition: Condition,
  check: Check,
  episodes = 200,
): { tracking: number; catching: number } {
  return evaluatePolicy(model, conditionConfig(condition), check, episodes, 1);
}

export function randomizeResource(action: number, rng: Rng): number {
  return Math.floor(action / 3) * 3 + nextInt(rng, 3);
}

// Resource action 0 moves straight down, so the square keeps its starting column.
export function holdResource(action: number): number {
  return Math.floor(action / 3) * 3;
}

export function evaluatePolicy(
  model: Mlp,
  config: WorldConfig,
  check: Check,
  episodes: number,
  subject: number,
): { tracking: number; catching: number } {
  if ((check === "resourceRandom" || check === "resourceStill") && config.schema !== "free") {
    throw new Error("Подмена действий ресурса доступна только в свободном режиме");
  }
  const batch = createBatch(model, 1);
  const observation = new Uint8Array(model.sizes[0]!);
  const resourceRng = createRng(subject, 6);
  let tracking = 0;
  let catching = 0;
  for (let episode = 1; episode <= episodes; episode++) {
    const world = createWorld(
      { ...config, windowActive: check !== "windowOff" },
      4000000000 + episode,
    );
    appendFrame(observation, world.frame, true);
    while (world.t < 10) {
      if (check === "fieldOff") clearObservationField(observation);
      batch.activations[0]!.set(observation);
      const action = greedyAction(forward(model, batch));
      stepWorld(
        world,
        check === "resourceRandom"
          ? randomizeResource(action, resourceRng)
          : check === "resourceStill"
            ? holdResource(action)
            : action,
      );
      appendFrame(observation, world.frame, false);
    }
    tracking += world.trackingScore;
    catching += world.catchScore;
  }
  return { tracking: tracking / episodes, catching: catching / episodes };
}

export function trainSubject(
  job: TrainJob,
  onEvent?: (event: TrainingEvent) => void,
): { model: Mlp; result: TrainResult } {
  const { model, points } = trainDqn(
    {
      subject: job.subject,
      iterations: job.iterations,
      evalEvery: job.evalEvery,
      config: job.config,
      environment: { world: conditionConfig(job.condition), frames: 10, discount: 0.4 },
      checks: checksFor(job.condition),
    },
    onEvent,
  );
  return { model, result: { condition: job.condition, subject: job.subject, points } };
}

export function trainDqn(
  job: DqnJob,
  onEvent?: (event: TrainingEvent) => void,
): { model: Mlp; points: CurvePoint[] } {
  const model = createMlp(job.config.sizes, createRng(job.subject, 3));
  const adam = createAdam(model.parameters.length);
  const replayRng = createRng(job.subject, 5);
  const collector = createCollector(
    job.environment.world,
    job.subject,
    job.environment.frames,
    job.environment.discount,
  );
  const policy = createEpsilonPolicy(model, createRng(job.subject, 4));
  const batch = createBatch(model, job.config.batchSize);
  const nextBatch = createBatch(model, job.config.batchSize);
  const sample = createReplayBatch(job.config.batchSize, collector.observation.length);
  const targets = new Float64Array(job.config.batchSize);
  const points: CurvePoint[] = [];
  let loss = 0;
  for (let iteration = 1; iteration <= job.iterations; iteration++) {
    const event = (phase: Phase, stage: "start" | "end") =>
      onEvent?.({ iteration, phase, stage, loss });
    event("iteration", "start");
    event("collect", "start");
    collectSteps(collector, policy, job.config.collectSteps);
    event("collect", "end");
    event("update", "start");
    for (let update = 0; update < job.config.updates; update++) {
      sampleReplay(collector.replay, replayRng, sample);
      loss = updateDqn(model, adam, batch, nextBatch, sample, targets);
    }
    event("update", "end");
    event("evaluate", "start");
    if (iteration % job.evalEvery === 0 || iteration === job.iterations) {
      for (const check of job.checks) {
        points.push({
          iteration,
          check,
          ...evaluatePolicy(
            model, job.environment.world, check, job.config.evalEpisodes, job.subject,
          ),
        });
      }
    }
    event("evaluate", "end");
    event("iteration", "end");
  }
  return { model, points };
}
