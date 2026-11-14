import { nextUnit, type Rng } from "../rng.ts";
import {
  bindGru,
  createGru,
  createGruBatch,
  forwardGru,
  type Gru,
  type GruBatch,
} from "../nn/gru.ts";
import type { Config } from "./config.ts";
import type { Action, Move, Observation } from "./episode.ts";
import { encodeCommands } from "./f4.ts";
import { poolPerception, type Perception } from "./perception.ts";

export type Head = {
  offset: number;
  output: number;
  width: number;
  weights: Float64Array;
  biases: Float64Array;
};
export type Controller = {
  gru: Gru;
  heads: Head[];
  outputs: number;
  parameters: Float64Array;
};
export type ControllerBatch = {
  recurrent: GruBatch;
  logits: Float64Array;
  delta: Float64Array;
  outputGradient: Float64Array;
  gradient: Float64Array;
};

export function c6InputWidth(config: Config): number {
  return 32 + config.fieldSize * config.fieldSize;
}

// The only oracle input is resource. No world state or attention center is accepted.
// Perception has already assimilated the current measurement before this call.
export function encodeC6(
  config: Config,
  perception: Perception,
  observation: Observation,
  resource: Float64Array,
  first: boolean,
): Float64Array {
  if (resource.length !== config.fieldSize * config.fieldSize)
    throw new RangeError("Неверная ширина карты ресурса C6");
  const input = new Float64Array(c6InputWidth(config));
  input.set(poolPerception(config, perception));
  input.set(encodeCommands(observation, first), 18);
  if (!first) input[28 + observation.previousPaddleMove + 1] = 1;
  // A fixed paddle has no positional degree of freedom; encode it as zero.
  input[31] =
    config.paddleMax === config.paddleMin
      ? 0
      : (observation.paddle - config.paddleMin) / (config.paddleMax - config.paddleMin);
  input.set(resource, 32);
  return input;
}

export function createController(
  input: number,
  hidden: number,
  classes: number,
  rng: Rng,
): Controller {
  if (!Number.isSafeInteger(classes) || classes < 2)
    throw new RangeError("Нужно не меньше двух категорий ответа");
  const initial = createGru(input, hidden, rng);
  const widths = [9, 2, 3, classes, 1];
  const outputs = widths.reduce((sum, width) => sum + width, 0);
  const parameters = new Float64Array(initial.parameters.length + outputs * (hidden + 1));
  parameters.set(initial.parameters);
  const gru = bindGru(input, hidden, parameters.subarray(0, initial.parameters.length));
  const heads: Head[] = [];
  let offset = initial.parameters.length;
  let output = 0;
  const scale = 1 / Math.sqrt(hidden);
  // After GRU: command, effort, paddle, answer, value; each W output-major then b.
  // One draw per weight in storage order, none for zero biases.
  for (const width of widths) {
    const weights = parameters.subarray(offset, offset + width * hidden);
    const biases = parameters.subarray(offset + width * hidden, offset + width * (hidden + 1));
    for (let i = 0; i < weights.length; i++) weights[i] = -scale + 2 * scale * nextUnit(rng);
    heads.push({ offset, output, width, weights, biases });
    offset += width * (hidden + 1);
    output += width;
  }
  return { gru, heads, outputs, parameters };
}

export function createControllerBatch(
  model: Controller,
  size: number,
  steps: number,
): ControllerBatch {
  return {
    recurrent: createGruBatch(model.gru, size, steps),
    logits: new Float64Array(size * steps * model.outputs),
    delta: new Float64Array(size * steps * model.outputs),
    outputGradient: new Float64Array(size * steps * model.gru.hidden),
    gradient: new Float64Array(model.parameters.length),
  };
}

function readHeads(
  model: Controller,
  batch: ControllerBatch,
  sequence: number,
  step: number,
): void {
  const { hidden } = model.gru;
  const { steps, states } = batch.recurrent;
  const state = (sequence * (steps + 1) + step + 1) * hidden;
  const base = (sequence * steps + step) * model.outputs;
  for (const head of model.heads) {
    for (let j = 0; j < head.width; j++) {
      let sum = 0;
      for (let k = 0; k < hidden; k++) sum += head.weights[j * hidden + k]! * states[state + k]!;
      batch.logits[base + head.output + j] = sum + head.biases[j]!;
    }
  }
}

export function forwardController(model: Controller, batch: ControllerBatch): void {
  forwardGru(model.gru, batch.recurrent);
  for (let step = 0; step < batch.recurrent.steps; step++)
    for (let sequence = 0; sequence < batch.recurrent.size; sequence++)
      readHeads(model, batch, sequence, step);
}

// Online equivalent of forwardGru, retaining its arithmetic and sparse-input order.
// A fresh batch supplies h_0 = 0; calls must visit steps 0..T-1 once in order.
// Keeping this adapter here leaves the shared GRU and earlier probes unchanged.
export function advanceController(model: Controller, batch: ControllerBatch, step: number): void {
  const { input, hidden, gates } = model.gru;
  const { inputs, states, steps, reset, update, candidate, resetState, indices, counts, size } =
    batch.recurrent;
  for (let sequence = 0; sequence < size; sequence++) {
    const sample = sequence * steps + step;
    const xBase = sample * input;
    const base = sample * hidden;
    const previous = (sequence * (steps + 1) + step) * hidden;
    const current = previous + hidden;
    let count = 0;
    for (let k = 0; k < input; k++) if (inputs[xBase + k] !== 0) indices[xBase + count++] = k;
    counts[sample] = count;
    for (let gateIndex = 0; gateIndex < 3; gateIndex++) {
      const { weights, recurrent, biases } = gates[gateIndex]!;
      const destination = gateIndex === 0 ? reset : gateIndex === 1 ? update : candidate;
      const recurrentValues = gateIndex === 2 ? resetState : states;
      const recurrentBase = gateIndex === 2 ? base : previous;
      for (let j = 0; j < hidden; j++) {
        let sum = 0;
        const wBase = j * input;
        const uBase = j * hidden;
        for (let n = 0; n < count; n++) {
          const k = indices[xBase + n]!;
          sum += weights[wBase + k]! * inputs[xBase + k]!;
        }
        for (let k = 0; k < hidden; k++)
          sum += recurrent[uBase + k]! * recurrentValues[recurrentBase + k]!;
        sum += biases[j]!;
        destination[base + j] =
          gateIndex === 2 ? 1 - 2 / (1 + Math.exp(2 * sum)) : 1 / (1 + Math.exp(-sum));
      }
      if (gateIndex === 0) {
        for (let j = 0; j < hidden; j++)
          resetState[base + j] = reset[base + j]! * states[previous + j]!;
      }
    }
    for (let j = 0; j < hidden; j++) {
      const z = update[base + j]!;
      states[current + j] = (1 - z) * candidate[base + j]! + z * states[previous + j]!;
    }
    readHeads(model, batch, sequence, step);
  }
}

export function distribution(
  logits: Float64Array,
  base: number,
  width: number,
): {
  probabilities: Float64Array;
  logs: Float64Array;
  entropy: number;
} {
  let maximum = -Infinity;
  for (let j = 0; j < width; j++) maximum = Math.max(maximum, logits[base + j]!);
  const probabilities = new Float64Array(width);
  const logs = new Float64Array(width);
  let sum = 0;
  for (let j = 0; j < width; j++) {
    probabilities[j] = Math.exp(logits[base + j]! - maximum);
    sum += probabilities[j]!;
  }
  const normalizer = Math.log(sum);
  let entropy = 0;
  for (let j = 0; j < width; j++) {
    probabilities[j] = probabilities[j]! / sum;
    logs[j] = logits[base + j]! - maximum - normalizer;
    entropy -= probabilities[j]! * logs[j]!;
  }
  return { probabilities, logs, entropy };
}

// Draws: time, episode, command/effort/paddle, and answer only at terminal time.
// Each sampled head consumes exactly one draw. Greedy ties keep the first index.
export function controllerAction(
  model: Controller,
  logits: Float64Array,
  sample: number,
  terminal: boolean,
  rng?: Rng,
): {
  choices: Int32Array;
  action: Action;
  logProb: number;
  entropy: number;
} {
  const choices = new Int32Array([0, 0, 0, -1]);
  let logProb = 0;
  let entropy = 0;
  for (let h = 0; h < (terminal ? 4 : 3); h++) {
    const head = model.heads[h]!;
    const base = sample * model.outputs + head.output;
    const values = distribution(logits, base, head.width);
    let choice = 0;
    if (rng === undefined) {
      for (let j = 1; j < head.width; j++)
        if (logits[base + j]! > logits[base + choice]!) choice = j;
    } else {
      const draw = nextUnit(rng);
      let cumulative = 0;
      choice = head.width - 1;
      for (let j = 0; j < head.width; j++) {
        cumulative += values.probabilities[j]!;
        if (draw < cumulative) {
          choice = j;
          break;
        }
      }
    }
    choices[h] = choice;
    logProb += values.logs[choice]!;
    entropy += values.entropy;
  }
  if (!Number.isFinite(logProb) || !Number.isFinite(entropy))
    throw new Error("Неконечное распределение действий");
  return {
    choices,
    logProb,
    entropy,
    action: {
      command: {
        row: (Math.floor(choices[0]! / 3) - 1) as Move,
        col: ((choices[0]! % 3) - 1) as Move,
      },
      effort: choices[1] as 0 | 1,
      paddleMove: (choices[2]! - 1) as Move,
      ...(terminal ? { answer: choices[3]! } : {}),
    },
  };
}
