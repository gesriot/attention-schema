import { adamStep, createAdam } from "../nn/adam.ts";
import { createBatch, createMlp, crossEntropyGradient, forward, type Mlp } from "../nn/mlp.ts";
import { createRng, nextInt } from "../rng.ts";
import { greedyAction, holdResource, randomizeResource } from "./dqn.ts";
import { createWorld, stepWorld, type WorldConfig } from "./world.ts";

export const probeSources = [
  "resource",
  "visual",
  "both",
  "resourceRandom",
  "resourceStill",
  "untrained",
  "context",
] as const;
export type ProbeSource = (typeof probeSources)[number];
export type ProbeConfig = {
  episodes: number;
  trainEpisodes: number;
  hidden: number;
  batchSize: number;
  updates: number;
};
export const publishedProbes: ProbeConfig = {
  episodes: 2000,
  trainEpisodes: 1600,
  hidden: 200,
  batchSize: 100,
  updates: 2000,
};
export type ProbeFrames = {
  frames: Uint8Array;
  labels: Uint8Array;
  steps: Uint8Array;
  episodes: Uint32Array;
};
export type ProbeData = {
  width: number;
  inputs: Uint8Array;
  labels: Uint8Array;
  episodes: Uint32Array;
};
export type ProbeResult = { source: ProbeSource; accuracy: number; majority: number };

export function collectProbeFrames(
  model: Mlp,
  config: WorldConfig,
  subject: number,
  episodes = 2000,
  resource: "policy" | "random" | "still" = "policy",
): ProbeFrames {
  if (config.variant !== "piefke" || config.schema !== "free" || model.sizes[0] !== 200) {
    throw new Error("Зонд требует свободный ресурс и наблюдение из одного кадра");
  }
  const data: ProbeFrames = {
    frames: new Uint8Array(episodes * 8 * 200),
    labels: new Uint8Array(episodes * 8),
    steps: new Uint8Array(episodes * 8),
    episodes: new Uint32Array(episodes * 8),
  };
  const batch = createBatch(model, 1);
  const resourceRng = createRng(subject, 6);
  let index = 0;
  for (let episode = 1; episode <= episodes; episode++) {
    const world = createWorld(config, 3000000000 + episode);
    while (world.t < 10) {
      batch.activations[0]!.set(world.frame);
      const action = greedyAction(forward(model, batch));
      stepWorld(
        world,
        resource === "random"
          ? randomizeResource(action, resourceRng)
          : resource === "still"
            ? holdResource(action)
            : action,
      );
      if (world.t === 10) continue;
      data.frames.set(world.frame, index * 200);
      // Truth supplies only the evaluator's target, never an input feature.
      data.labels[index] = world.window.col - 1;
      data.steps[index] = world.t;
      data.episodes[index] = episode;
      index++;
    }
  }
  return data;
}

export function probeData(frames: ProbeFrames, source: ProbeSource): ProbeData {
  const width = source === "both" ? 200 : source === "context" ? 10 : 100;
  const inputs = new Uint8Array(frames.labels.length * width);
  for (let sample = 0; sample < frames.labels.length; sample++) {
    if (source === "context") {
      inputs[sample * width + frames.steps[sample]! - 1] = 1;
    } else {
      const offset = sample * 200 + (source === "both" || source === "visual" ? 0 : 100);
      inputs.set(frames.frames.subarray(offset, offset + width), sample * width);
    }
  }
  return { width, inputs, labels: frames.labels, episodes: frames.episodes };
}

export function splitProbeData(data: ProbeData, trainEpisodes: number): {
  train: number[];
  test: number[];
} {
  const train: number[] = [];
  const test: number[] = [];
  for (let i = 0; i < data.labels.length; i++) {
    (data.episodes[i]! <= trainEpisodes ? train : test).push(i);
  }
  if (train.length === 0 || test.length === 0) {
    throw new Error("Обе части данных зонда должны быть непустыми");
  }
  return { train, test };
}

export function fitProbe(
  data: ProbeData,
  subject: number,
  config: ProbeConfig = publishedProbes,
): { accuracy: number; majority: number } {
  const { train, test } = splitProbeData(data, config.trainEpisodes);
  // Reset stream 7 for each source; initialization is followed by batch draws.
  const rng = createRng(subject, 7);
  const model = createMlp([data.width, config.hidden, 8], rng);
  const adam = createAdam(model.parameters.length);
  const batch = createBatch(model, config.batchSize);
  const labels = new Uint8Array(config.batchSize);
  for (let update = 0; update < config.updates; update++) {
    for (let row = 0; row < config.batchSize; row++) {
      const index = train[nextInt(rng, train.length)]!;
      batch.activations[0]!.set(
        data.inputs.subarray(index * data.width, (index + 1) * data.width),
        row * data.width,
      );
      labels[row] = data.labels[index]!;
    }
    forward(model, batch);
    const loss = crossEntropyGradient(model, batch, labels);
    if (!Number.isFinite(loss)) throw new Error("Потери зонда перестали быть конечными");
    adamStep(model.parameters, batch.gradient, adam);
  }
  const evaluation = createBatch(model, 1);
  const counts = new Uint32Array(8);
  let correct = 0;
  for (const index of test) {
    evaluation.activations[0]!.set(
      data.inputs.subarray(index * data.width, (index + 1) * data.width),
    );
    const label = data.labels[index]!;
    counts[label] = counts[label]! + 1;
    if (greedyAction(forward(model, evaluation)) === label) correct++;
  }
  return { accuracy: correct / test.length, majority: Math.max(...counts) / test.length };
}

export function runProbes(
  model: Mlp,
  world: WorldConfig,
  subject: number,
  config: ProbeConfig = publishedProbes,
  onProgress?: (source: ProbeSource, stage: "start" | "end") => void,
): ProbeResult[] {
  const normal = collectProbeFrames(model, world, subject, config.episodes);
  const randomized = collectProbeFrames(model, world, subject, config.episodes, "random");
  const still = collectProbeFrames(model, world, subject, config.episodes, "still");
  const initial = createMlp(model.sizes, createRng(subject, 3));
  const untrained = collectProbeFrames(initial, world, subject, config.episodes);
  return probeSources.map((source) => {
    onProgress?.(source, "start");
    const frames =
      source === "resourceRandom"
        ? randomized
        : source === "resourceStill"
          ? still
          : source === "untrained"
            ? untrained
            : normal;
    const result = { source, ...fitProbe(probeData(frames, source), subject, config) };
    onProgress?.(source, "end");
    return result;
  });
}
