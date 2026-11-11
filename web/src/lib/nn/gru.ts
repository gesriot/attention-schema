import { nextUnit, type Rng } from "../rng.ts";

export type Gate = {
  offset: number;
  weights: Float64Array;
  recurrent: Float64Array;
  biases: Float64Array;
};
export type Gru = { input: number; hidden: number; parameters: Float64Array; gates: Gate[] };
export type GruBatch = {
  size: number;
  steps: number;
  inputs: Float64Array;
  states: Float64Array;
  reset: Float64Array;
  update: Float64Array;
  candidate: Float64Array;
  resetState: Float64Array;
  indices: Int32Array;
  counts: Int32Array;
  stateGradient: Float64Array;
  resetGradient: Float64Array;
  gateGradient: Float64Array;
};

export function gruParameterCount(input: number, hidden: number): number {
  if (![input, hidden].every((size) => Number.isSafeInteger(size) && size > 0))
    throw new RangeError("Размеры GRU должны быть положительными целыми числами");
  return 3 * hidden * (input + hidden + 1);
}

// Bind views so a caller can append its own parameters in a single optimizer vector.
export function bindGru(input: number, hidden: number, parameters: Float64Array): Gru {
  if (parameters.length !== gruParameterCount(input, hidden))
    throw new RangeError("Неверное число параметров GRU");
  const gates: Gate[] = [];
  const width = hidden * (input + hidden + 1);
  // Gates r, z, n; within each gate W, U, b. Matrices are output-major.
  for (let gate = 0; gate < 3; gate++) {
    const offset = gate * width;
    const recurrentStart = offset + hidden * input;
    const biasStart = recurrentStart + hidden * hidden;
    gates.push({
      offset,
      weights: parameters.subarray(offset, recurrentStart),
      recurrent: parameters.subarray(recurrentStart, biasStart),
      biases: parameters.subarray(biasStart, offset + width),
    });
  }
  return { input, hidden, parameters, gates };
}

export function createGru(input: number, hidden: number, rng: Rng): Gru {
  const model = bindGru(input, hidden, new Float64Array(gruParameterCount(input, hidden)));
  const scale = 1 / Math.sqrt(hidden);
  // Draw in parameter order, skipping zero biases without consuming the generator.
  for (const gate of model.gates) {
    for (const weights of [gate.weights, gate.recurrent]) {
      for (let index = 0; index < weights.length; index++)
        weights[index] = -scale + 2 * scale * nextUnit(rng);
    }
  }
  return model;
}

export function createGruBatch(model: Gru, size: number, steps: number): GruBatch {
  if (![size, steps].every((value) => Number.isSafeInteger(value) && value > 0))
    throw new RangeError("Размер батча и длина последовательности должны быть положительными");
  const samples = size * steps;
  return {
    size,
    steps,
    inputs: new Float64Array(samples * model.input),
    // Sequence-major, with a separate zero initial state for each sequence.
    states: new Float64Array(size * (steps + 1) * model.hidden),
    reset: new Float64Array(samples * model.hidden),
    update: new Float64Array(samples * model.hidden),
    candidate: new Float64Array(samples * model.hidden),
    resetState: new Float64Array(samples * model.hidden),
    indices: new Int32Array(samples * model.input),
    counts: new Int32Array(samples),
    stateGradient: new Float64Array(size * (steps + 1) * model.hidden),
    resetGradient: new Float64Array(model.hidden),
    gateGradient: new Float64Array(3 * model.hidden),
  };
}

export function forwardGru(model: Gru, batch: GruBatch, skipZeros = true): Float64Array {
  const { input, hidden, gates } = model;
  const { inputs, states, steps, reset, update, candidate, resetState, indices, counts } = batch;
  for (let sample = 0; sample < batch.size * steps; sample++) {
    const base = sample * input;
    let count = 0;
    for (let k = 0; k < input; k++) {
      if (!skipZeros || inputs[base + k] !== 0) indices[base + count++] = k;
    }
    counts[sample] = count;
  }
  // Time, sequence, gate, output, then ascending inputs and recurrent coordinates.
  // Each dot product starts at zero; W x precedes U h and the bias is added last.
  for (let sequence = 0; sequence < batch.size; sequence++)
    states.fill(0, sequence * (steps + 1) * hidden, (sequence * (steps + 1) + 1) * hidden);
  for (let step = 0; step < steps; step++) {
    for (let sequence = 0; sequence < batch.size; sequence++) {
      const sample = sequence * steps + step;
      const xBase = sample * input;
      const base = sample * hidden;
      const previous = (sequence * (steps + 1) + step) * hidden;
      const current = previous + hidden;
      const count = counts[sample]!;
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
    }
  }
  return states;
}

// Output derivatives are sequence-major [sequence, step, hidden], without h_0.
// Accumulate into the supplied parameter gradient; the caller owns clearing it.
export function backwardGru(
  model: Gru,
  batch: GruBatch,
  outputGradient: Float64Array,
  gradient: Float64Array,
): void {
  const { input, hidden, gates } = model;
  const {
    inputs,
    states,
    steps,
    reset,
    update,
    candidate,
    resetState,
    indices,
    counts,
    stateGradient,
    resetGradient,
    gateGradient,
  } = batch;
  stateGradient.fill(0);
  // Reverse time, ascending sequences. Branch derivatives: direct state, z, n, r.
  for (let step = steps - 1; step >= 0; step--) {
    for (let sequence = 0; sequence < batch.size; sequence++) {
      const sample = sequence * steps + step;
      const base = sample * hidden;
      const xBase = sample * input;
      const previous = (sequence * (steps + 1) + step) * hidden;
      const current = previous + hidden;
      resetGradient.fill(0);
      for (let j = 0; j < hidden; j++) {
        const dh = outputGradient[base + j]! + stateGradient[current + j]!;
        const z = update[base + j]!;
        const n = candidate[base + j]!;
        stateGradient[previous + j] = dh * z;
        gateGradient[hidden + j] = dh * (states[previous + j]! - n) * z * (1 - z);
        gateGradient[2 * hidden + j] = dh * (1 - z) * (1 - n * n);
      }
      // U_n acts on r * h, so its transpose must precede the reset derivative.
      const un = gates[2]!.recurrent;
      for (let j = 0; j < hidden; j++) {
        const d = gateGradient[2 * hidden + j]!;
        for (let k = 0; k < hidden; k++)
          resetGradient[k] = resetGradient[k]! + d * un[j * hidden + k]!;
      }
      for (let j = 0; j < hidden; j++) {
        const r = reset[base + j]!;
        const d = resetGradient[j]!;
        gateGradient[j] = d * states[previous + j]! * r * (1 - r);
        stateGradient[previous + j] = stateGradient[previous + j]! + d * r;
      }
      // Parameter sums: descending time, ascending sequences, gates r/z/n, outputs/inputs.
      for (let gateIndex = 0; gateIndex < 3; gateIndex++) {
        const { offset, recurrent } = gates[gateIndex]!;
        const uOffset = offset + hidden * input;
        const bOffset = uOffset + hidden * hidden;
        const recurrentValues = gateIndex === 2 ? resetState : states;
        const recurrentBase = gateIndex === 2 ? base : previous;
        for (let j = 0; j < hidden; j++) {
          const d = gateGradient[gateIndex * hidden + j]!;
          if (d === 0) continue;
          gradient[bOffset + j] = gradient[bOffset + j]! + d;
          const wBase = offset + j * input;
          for (let n = 0; n < counts[sample]!; n++) {
            const k = indices[xBase + n]!;
            gradient[wBase + k] = gradient[wBase + k]! + d * inputs[xBase + k]!;
          }
          for (let k = 0; k < hidden; k++) {
            const w = uOffset + j * hidden + k;
            gradient[w] = gradient[w]! + d * recurrentValues[recurrentBase + k]!;
          }
          if (gateIndex < 2) {
            for (let k = 0; k < hidden; k++)
              stateGradient[previous + k] =
                stateGradient[previous + k]! + d * recurrent[j * hidden + k]!;
          }
        }
      }
    }
  }
}
