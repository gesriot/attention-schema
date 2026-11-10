import { adamStep, createAdam } from "../nn/adam.ts";
import { createBatch, createMlp, crossEntropyGradient, forward } from "../nn/mlp.ts";
import { createRng, nextInt } from "../rng.ts";
import { streams, type Config } from "./config.ts";
import { applyAction, createEpisode, type Observation } from "./episode.ts";
import { createNeutralPolicy } from "./policies.ts";
import { createPerception, poolPerception, updatePerception } from "./perception.ts";

export const f4FirstSeed = 6000000001;
export const sources = [
  "frames",
  "perception",
  "commands",
  "telemetry",
  "schema",
  "context",
  "measurement",
] as const;
export type Source = (typeof sources)[number];
export type Axis = "row" | "col";
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

export function collectCorpus(config: Config, episodes: number): Corpus {
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
    frames: sharedFloats(samples * cells * 2),
    perception: sharedFloats(samples * 18),
    commands: sharedFloats(samples * 10),
    telemetry: sharedFloats(samples * telemetryWidth),
    measurement: sharedFloats(samples * cells * 2),
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
      corpus.frames.set(episode.truth.world.frame, sample * cells * 2);
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

type Block = { values: Float64Array; width: number; history: boolean };
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
  ): Block => ({
    values: corpus[key],
    width:
      key === "perception"
        ? 18
        : key === "commands"
          ? 10
          : key === "telemetry"
            ? corpus.telemetryWidth
            : corpus.cells * 2,
    history: key !== "measurement",
  });
  const blocks =
    source === "context"
      ? []
      : source === "schema"
        ? [block("perception"), block("commands"), block("telemetry")]
        : [block(source)];
  const width =
    source === "context"
      ? corpus.steps
      : blocks.reduce((sum, part) => sum + part.width * (part.history ? corpus.steps : 1), 0);
  return {
    blocks,
    steps: corpus.steps,
    samples: corpus.episodes * corpus.steps,
    width,
    context: source === "context",
  };
}

export function writeInput(inputs: Inputs, sample: number, output: Float64Array, offset = 0): void {
  output.fill(0, offset, offset + inputs.width);
  const step = sample % inputs.steps;
  if (inputs.context) {
    output[offset + step] = 1;
    return;
  }
  // Source histories are concatenated, each newest first and padded only before this episode.
  for (const block of inputs.blocks) {
    const history = block.history ? step + 1 : 1;
    for (let lag = 0; lag < history; lag++) {
      const start = (sample - lag) * block.width;
      output.set(block.values.subarray(start, start + block.width), offset + lag * block.width);
    }
    offset += block.width * (block.history ? inputs.steps : 1);
  }
}

export type ProbeData = {
  inputs: Inputs;
  rowLabels: Uint8Array;
  colLabels: Uint8Array;
  centers: Float64Array;
  trainSamples: number;
};
export type ProbeResult = { source: Source; meanError: number; withinOne: number };
export type ProbeProgress = { axis: Axis; update: number; loss: number };

export function probeData(config: Config, corpus: Corpus, source: Source): ProbeData {
  return {
    inputs: sourceInputs(corpus, source),
    rowLabels: corpus.rowLabels,
    colLabels: corpus.colLabels,
    centers: corpus.centers,
    trainSamples: splitEpisodes(config, corpus.episodes).train * corpus.steps,
  };
}

function argmax(values: Float64Array): number {
  let best = 0;
  for (let index = 1; index < values.length; index++)
    if (values[index]! > values[best]!) best = index;
  return best;
}

function fitAxis(
  config: Config,
  data: ProbeData,
  axis: Axis,
  onProgress?: (progress: ProbeProgress) => void,
): Uint8Array {
  // Reset seed 1 and the axis stream for every source. Consume initialization first,
  // then one uniform draw per batch row, sampling training examples with replacement.
  const rng = createRng(1, axis === "row" ? streams.probeRow : streams.probeCol);
  const model = createMlp([data.inputs.width, config.f4Hidden, config.fieldSize], rng);
  const adam = createAdam(model.parameters.length);
  const batch = createBatch(model, config.f4BatchSize);
  const labels = new Uint8Array(config.f4BatchSize);
  const allLabels = axis === "row" ? data.rowLabels : data.colLabels;
  for (let update = 1; update <= config.f4Updates; update++) {
    for (let row = 0; row < config.f4BatchSize; row++) {
      const sample = nextInt(rng, data.trainSamples);
      writeInput(data.inputs, sample, batch.activations[0]!, row * data.inputs.width);
      labels[row] = allLabels[sample]!;
    }
    forward(model, batch);
    const loss = crossEntropyGradient(model, batch, labels);
    if (!Number.isFinite(loss)) throw new Error("Потери зонда перестали быть конечными");
    adamStep(model.parameters, batch.gradient, adam);
    onProgress?.({ axis, update, loss });
  }
  const evaluation = createBatch(model, 1);
  const predictions = new Uint8Array(data.inputs.samples - data.trainSamples);
  for (let sample = data.trainSamples; sample < data.inputs.samples; sample++) {
    writeInput(data.inputs, sample, evaluation.activations[0]!);
    predictions[sample - data.trainSamples] = argmax(forward(model, evaluation));
  }
  return predictions;
}

export function scorePredictions(
  data: Pick<ProbeData, "centers" | "trainSamples">,
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

export function fitProbe(
  config: Config,
  data: ProbeData,
  source: Source,
  onProgress?: (progress: ProbeProgress) => void,
): ProbeResult {
  if (data.trainSamples < 1 || data.trainSamples >= data.inputs.samples) {
    throw new Error("Обе части данных зонда должны быть непустыми");
  }
  const rows = fitAxis(config, data, "row", onProgress);
  const cols = fitAxis(config, data, "col", onProgress);
  return { source, ...scorePredictions(data, rows, cols) };
}
