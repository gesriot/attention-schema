import { adamStep, createAdam } from "../nn/adam.ts";
import {
  bindGru,
  createGru,
  createGruBatch,
  forwardGru,
  backwardGru,
  type Gru,
  type GruBatch,
} from "../nn/gru.ts";
import { createRng, nextInt, nextUnit, type Rng } from "../rng.ts";
import { streams, type Config } from "./config.ts";
import { applyAction, createEpisode, type Observation } from "./episode.ts";
import { createNeutralPolicy } from "./policies.ts";
import { createPerception, poolPerception, updatePerception } from "./perception.ts";

export const f4FirstSeed = 6000000001;
export const sources = [
  "frames",
  "perception",
  "perceptionStep",
  "commands",
  "telemetry",
  "telemetryStep",
  "schema",
  "context",
  "measurement",
] as const;
export type Source = (typeof sources)[number];
export type Corpus = {
  episodes: number;
  steps: number;
  cells: number;
  telemetryWidth: number;
  frames: Float64Array;
  perception: Float64Array;
  commands: Float64Array;
  telemetry: Float64Array;
  measurement: Float64Array;
  centers: Float64Array;
  rowLabels: Uint8Array;
  colLabels: Uint8Array;
};

function sharedFloats(length: number): Float64Array {
  return new Float64Array(new SharedArrayBuffer(length * Float64Array.BYTES_PER_ELEMENT));
}

export function encodeCommands(observation: Observation, first: boolean): Float64Array {
  const values = new Float64Array(10);
  if (first) return values;
  const { row, col } = observation.previousCommand;
  values[(row + 1) * 3 + col + 1] = 1;
  values[9] = observation.previousEffort;
  return values;
}

export function splitEpisodes(config: Config, episodes: number): { train: number; test: number } {
  if (
    !Number.isSafeInteger(episodes) ||
    episodes < 2 ||
    f4FirstSeed + episodes > Number.MAX_SAFE_INTEGER
  ) {
    throw new Error("Для F4 нужны хотя бы два эпизода");
  }
  const train = Math.floor(episodes * config.f4TrainFraction);
  if (train < 1 || train >= episodes) throw new Error("Обе части данных F4 должны быть непустыми");
  return { train, test: episodes - train };
}

export function collectCorpus(
  config: Config,
  episodes: number,
  selected: readonly Source[] = sources,
): Corpus {
  splitEpisodes(config, episodes);
  if (config.fieldSize > 256) throw new Error("Метки зондов допускают не более 256 классов");
  const steps = config.episodeSteps;
  const cells = config.fieldSize * config.fieldSize;
  const telemetryWidth = (config.fieldSize / config.k) ** 2;
  const samples = episodes * steps;
  // Store individual frames once. Read-only shared buffers avoid copying the corpus per worker.
  const corpus: Corpus = {
    episodes,
    steps,
    cells,
    telemetryWidth,
    frames: sharedFloats(selected.includes("frames") ? samples * cells * 2 : 0),
    perception: sharedFloats(samples * 18),
    commands: sharedFloats(samples * 10),
    telemetry: sharedFloats(samples * telemetryWidth),
    measurement: sharedFloats(selected.includes("measurement") ? samples * cells * 2 : 0),
    centers: sharedFloats(samples * 2),
    rowLabels: new Uint8Array(new SharedArrayBuffer(samples)),
    colLabels: new Uint8Array(new SharedArrayBuffer(samples)),
  };
  for (let index = 0; index < episodes; index++) {
    const seed = f4FirstSeed + index;
    const episode = createEpisode(config, seed);
    const policy = createNeutralPolicy(config, seed);
    let perception = createPerception();
    for (let step = 0; step < steps; step++) {
      const sample = index * steps + step;
      const observation = episode.observation;
      perception = updatePerception(config, perception, observation.measurement);
      if (corpus.frames.length > 0)
        corpus.frames.set(episode.truth.world.frame, sample * cells * 2);
      if (corpus.measurement.length > 0)
        corpus.measurement.set(observation.measurement, sample * cells * 2);
      corpus.perception.set(poolPerception(config, perception), sample * 18);
      corpus.commands.set(encodeCommands(observation, step === 0), sample * 10);
      corpus.telemetry.set(observation.telemetry, sample * telemetryWidth);
      // Evaluator truth supplies labels and continuous scoring coordinates, never features.
      const center = episode.truth.attention.center;
      corpus.centers[sample * 2] = center.row;
      corpus.centers[sample * 2 + 1] = center.col;
      corpus.rowLabels[sample] = Math.round(center.row);
      corpus.colLabels[sample] = Math.round(center.col);
      applyAction(episode, policy(observation));
    }
  }
  return corpus;
}

type Block = { values: Float64Array; width: number };
export type Inputs = {
  blocks: Block[];
  steps: number;
  samples: number;
  width: number;
  context: boolean;
};

export function sourceInputs(corpus: Corpus, source: Source): Inputs {
  const block = (
    key: "frames" | "perception" | "commands" | "telemetry" | "measurement",
  ): Block => {
    const width =
      key === "perception"
        ? 18
        : key === "commands"
          ? 10
          : key === "telemetry"
            ? corpus.telemetryWidth
            : corpus.cells * 2;
    if (corpus[key].length !== corpus.episodes * corpus.steps * width)
      throw new Error(`Источник не сохранён в данных: ${key}`);
    return { values: corpus[key], width };
  };
  const blocks =
    source === "context"
      ? []
      : source === "schema"
        ? [block("perception"), block("commands"), block("telemetry")]
        : [
            block(
              source === "perceptionStep"
                ? "perception"
                : source === "telemetryStep"
                  ? "telemetry"
                  : source,
            ),
          ];
  return {
    blocks,
    steps: corpus.steps,
    samples: corpus.episodes * corpus.steps,
    width: source === "context" ? corpus.steps : blocks.reduce((sum, part) => sum + part.width, 0),
    context: source === "context",
  };
}

export function writeInput(inputs: Inputs, sample: number, output: Float64Array, offset = 0): void {
  if (inputs.context) {
    output.fill(0, offset, offset + inputs.width);
    output[offset + (sample % inputs.steps)] = 1;
    return;
  }
  // Only the current step, in source order. The recurrent state carries the history.
  for (const block of inputs.blocks) {
    const start = sample * block.width;
    output.set(block.values.subarray(start, start + block.width), offset);
    offset += block.width;
  }
}

export type ProbeData = {
  inputs: Inputs;
  rowLabels: Uint8Array;
  colLabels: Uint8Array;
  centers: Float64Array;
  trainEpisodes: number;
};
export type ProbeResult = {
  source: Source;
  meanError: number;
  withinOne: number;
  bestError: number;
};
export type ProbeProgress = { update: number; loss: number; meanError: number };
export const checkpointInterval = 500;

export function probeData(config: Config, corpus: Corpus, source: Source): ProbeData {
  return {
    inputs: sourceInputs(corpus, source),
    rowLabels: corpus.rowLabels,
    colLabels: corpus.colLabels,
    centers: corpus.centers,
    trainEpisodes: splitEpisodes(config, corpus.episodes).train,
  };
}

export type Probe = {
  gru: Gru;
  classes: number;
  parameters: Float64Array;
  weights: Float64Array;
  biases: Float64Array;
};
export type ProbeBatch = {
  recurrent: GruBatch;
  logits: Float64Array;
  delta: Float64Array;
  outputGradient: Float64Array;
  gradient: Float64Array;
};

export function createProbe(input: number, hidden: number, classes: number, rng: Rng): Probe {
  const initial = createGru(input, hidden, rng);
  const count = initial.parameters.length;
  const parameters = new Float64Array(count + 2 * classes * (hidden + 1));
  parameters.set(initial.parameters);
  const gru = bindGru(input, hidden, parameters.subarray(0, count));
  const weights = parameters.subarray(count, count + 2 * classes * hidden);
  const biases = parameters.subarray(count + weights.length);
  const scale = 1 / Math.sqrt(hidden);
  // Continue the GRU generator: readout output-major weights, then zero biases (no draws).
  for (let index = 0; index < weights.length; index++)
    weights[index] = -scale + 2 * scale * nextUnit(rng);
  return { gru, classes, parameters, weights, biases };
}

export function createProbeBatch(model: Probe, size: number, steps: number): ProbeBatch {
  return {
    recurrent: createGruBatch(model.gru, size, steps),
    logits: new Float64Array(size * steps * 2 * model.classes),
    delta: new Float64Array(size * steps * 2 * model.classes),
    outputGradient: new Float64Array(size * steps * model.gru.hidden),
    gradient: new Float64Array(model.parameters.length),
  };
}

export function createSourceBatch(
  model: Probe,
  episodes: number,
  steps: number,
  source: Source,
): ProbeBatch {
  // Keep episode-major sample order; only the recurrent sequence boundaries change.
  const singleStep = source === "perceptionStep" || source === "telemetryStep";
  return createProbeBatch(model, singleStep ? episodes * steps : episodes, singleStep ? 1 : steps);
}

export function sampleProbeBatch(
  data: ProbeData,
  batch: ProbeBatch,
  rows: Uint8Array,
  cols: Uint8Array,
  rng: Rng,
): void {
  const { steps, width } = data.inputs;
  // One draw per episode, then all its steps in ascending order, for both layouts.
  const episodes = rows.length / steps;
  for (let sequence = 0; sequence < episodes; sequence++) {
    const episode = nextInt(rng, data.trainEpisodes);
    for (let step = 0; step < steps; step++) {
      const sample = episode * steps + step;
      const destination = sequence * steps + step;
      writeInput(data.inputs, sample, batch.recurrent.inputs, destination * width);
      rows[destination] = data.rowLabels[sample]!;
      cols[destination] = data.colLabels[sample]!;
    }
  }
}

export function forwardProbe(model: Probe, batch: ProbeBatch): Float64Array {
  const { hidden } = model.gru;
  const { size, steps, states } = batch.recurrent;
  forwardGru(model.gru, batch.recurrent);
  const outputs = 2 * model.classes;
  // Sequence, time, output (row classes then column classes), hidden coordinate.
  for (let sequence = 0; sequence < size; sequence++) {
    for (let step = 0; step < steps; step++) {
      const base = (sequence * (steps + 1) + step + 1) * hidden;
      const destination = (sequence * steps + step) * outputs;
      for (let j = 0; j < outputs; j++) {
        let sum = 0;
        for (let k = 0; k < hidden; k++) sum += model.weights[j * hidden + k]! * states[base + k]!;
        batch.logits[destination + j] = sum + model.biases[j]!;
      }
    }
  }
  return batch.logits;
}

export function probeGradient(
  model: Probe,
  batch: ProbeBatch,
  rows: Uint8Array,
  cols: Uint8Array,
): number {
  const { hidden } = model.gru;
  const { size, steps, states } = batch.recurrent;
  const samples = size * steps;
  const outputs = 2 * model.classes;
  const readoutOffset = model.gru.parameters.length;
  const biasOffset = readoutOffset + model.weights.length;
  const { logits, delta, outputGradient, gradient } = batch;
  gradient.fill(0);
  outputGradient.fill(0);
  let loss = 0;
  // Sum cross-entropies in sequence/time/axis/class order; divide by B*T, not by 2*B*T.
  for (let sample = 0; sample < samples; sample++) {
    for (let axis = 0; axis < 2; axis++) {
      const base = sample * outputs + axis * model.classes;
      const label = (axis === 0 ? rows : cols)[sample]!;
      let maximum = -Infinity;
      for (let j = 0; j < model.classes; j++) maximum = Math.max(maximum, logits[base + j]!);
      let sum = 0;
      for (let j = 0; j < model.classes; j++) {
        const value = Math.exp(logits[base + j]! - maximum);
        delta[base + j] = value;
        sum += value;
      }
      loss += Math.log(sum) - (logits[base + label]! - maximum);
      for (let j = 0; j < model.classes; j++)
        delta[base + j] = (delta[base + j]! / sum - (j === label ? 1 : 0)) / samples;
    }
    const sequence = Math.floor(sample / steps);
    const stateBase = (sample + sequence + 1) * hidden;
    for (let j = 0; j < outputs; j++) {
      const d = delta[sample * outputs + j]!;
      gradient[biasOffset + j] = gradient[biasOffset + j]! + d;
      for (let k = 0; k < hidden; k++) {
        const w = readoutOffset + j * hidden + k;
        gradient[w] = gradient[w]! + d * states[stateBase + k]!;
        outputGradient[sample * hidden + k] =
          outputGradient[sample * hidden + k]! + d * model.weights[j * hidden + k]!;
      }
    }
  }
  backwardGru(model.gru, batch.recurrent, outputGradient, gradient);
  return loss / samples;
}

function argmax(values: Float64Array, offset: number, length: number): number {
  let best = 0;
  for (let index = 1; index < length; index++)
    if (values[offset + index]! > values[offset + best]!) best = index;
  return best;
}

export function scorePredictions(
  data: { centers: Float64Array; trainSamples: number },
  rows: Uint8Array,
  cols: Uint8Array,
): { meanError: number; withinOne: number } {
  let error = 0;
  let withinOne = 0;
  for (let index = 0; index < rows.length; index++) {
    const sample = index + data.trainSamples;
    const dr = rows[index]! - data.centers[sample * 2]!;
    const dc = cols[index]! - data.centers[sample * 2 + 1]!;
    const distance = Math.sqrt(dr * dr + dc * dc);
    error += distance;
    if (distance <= 1) withinOne++;
  }
  return { meanError: error / rows.length, withinOne: withinOne / rows.length };
}

function evaluateProbe(model: Probe, data: ProbeData, batch: ProbeBatch) {
  const { steps, samples, width } = data.inputs;
  const trainSamples = data.trainEpisodes * steps;
  const rows = new Uint8Array(samples - trainSamples);
  const cols = new Uint8Array(rows.length);
  for (let first = trainSamples; first < samples; first += steps) {
    for (let step = 0; step < steps; step++)
      writeInput(data.inputs, first + step, batch.recurrent.inputs, step * width);
    const logits = forwardProbe(model, batch);
    for (let step = 0; step < steps; step++) {
      const base = step * 2 * model.classes;
      rows[first + step - trainSamples] = argmax(logits, base, model.classes);
      cols[first + step - trainSamples] = argmax(logits, base + model.classes, model.classes);
    }
  }
  return scorePredictions({ centers: data.centers, trainSamples }, rows, cols);
}

export function fitProbe(
  config: Config,
  data: ProbeData,
  source: Source,
  onProgress?: (progress: ProbeProgress) => void,
): ProbeResult {
  const { steps, samples, width } = data.inputs;
  if (
    !Number.isSafeInteger(data.trainEpisodes) ||
    data.trainEpisodes < 1 ||
    data.trainEpisodes >= samples / steps ||
    samples % steps !== 0
  )
    throw new Error("Обе части данных зонда должны быть непустыми");
  // Each source resets seed 1, stream 22: GRU, readout, then B episode draws per update.
  // This makes a selected source independent of which other sources run or finish first.
  const rng = createRng(1, streams.probe);
  const model = createProbe(width, config.f4Hidden, config.fieldSize, rng);
  const adam = createAdam(model.parameters.length);
  const batch = createSourceBatch(model, config.f4BatchSize, steps, source);
  const evaluation = createSourceBatch(model, 1, steps, source);
  const rows = new Uint8Array(config.f4BatchSize * steps);
  const cols = new Uint8Array(rows.length);
  const recentLosses = new Float64Array(checkpointInterval);
  let bestError = Infinity;
  let final = { meanError: Infinity, withinOne: 0 };
  for (let update = 1; update <= config.f4Updates; update++) {
    sampleProbeBatch(data, batch, rows, cols, rng);
    forwardProbe(model, batch);
    const loss = probeGradient(model, batch, rows, cols);
    if (!Number.isFinite(loss)) throw new Error("Потери зонда перестали быть конечными");
    adamStep(model.parameters, batch.gradient, adam);
    recentLosses[(update - 1) % checkpointInterval] = loss;
    if (update % checkpointInterval === 0 || update === config.f4Updates) {
      final = evaluateProbe(model, data, evaluation);
      if (!Number.isFinite(final.meanError))
        throw new Error("Ошибка зонда перестала быть конечной");
      bestError = Math.min(bestError, final.meanError);
      const count = Math.min(update, checkpointInterval);
      const first = update < checkpointInterval ? 0 : update % checkpointInterval;
      let sum = 0;
      // Even a final partial interval reports the most recent 500 updates, in time order.
      for (let index = 0; index < count; index++)
        sum += recentLosses[(first + index) % checkpointInterval]!;
      onProgress?.({ update, loss: sum / count, meanError: final.meanError });
    }
  }
  return { source, ...final, bestError };
}
