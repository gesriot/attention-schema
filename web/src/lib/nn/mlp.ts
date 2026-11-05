import { nextNormal, nextUnit, type Rng } from "../rng.ts";

export type Layer = {
  input: number;
  output: number;
  offset: number;
  weights: Float64Array;
  biases: Float64Array;
};
export type Mlp = { sizes: number[]; parameters: Float64Array; layers: Layer[] };
export type Batch = {
  size: number;
  activations: Float64Array[];
  deltas: Float64Array[];
  gradient: Float64Array;
  inputIndices: Int32Array;
  inputCounts: Int32Array;
};

export function createMlp(sizes: readonly number[], rng: Rng): Mlp {
  if (sizes.length < 2 || sizes.some((size) => !Number.isInteger(size) || size < 1)) {
    throw new RangeError("Размеры слоёв должны быть положительными целыми числами");
  }
  let count = 0;
  for (let i = 1; i < sizes.length; i++) count += (sizes[i - 1]! + 1) * sizes[i]!;
  const parameters = new Float64Array(count);
  const layers: Layer[] = [];
  let offset = 0;
  for (let i = 1; i < sizes.length; i++) {
    const input = sizes[i - 1]!;
    const output = sizes[i]!;
    const weights = parameters.subarray(offset, offset + input * output);
    const biases = parameters.subarray(offset + input * output, offset + (input + 1) * output);
    const hidden = i < sizes.length - 1;
    const stddev = Math.sqrt(2 / input) / 0.87962566103423978;
    // Output-major weights, ascending layers/outputs/inputs, then biases.
    for (let j = 0; j < weights.length; j++) {
      if (hidden) {
        let normal: number;
        do normal = nextNormal(rng); while (Math.abs(normal) > 2);
        weights[j] = normal * stddev;
      } else {
        weights[j] = -0.03 + 0.06 * nextUnit(rng);
      }
    }
    biases.fill(hidden ? 0 : -0.2);
    layers.push({ input, output, offset, weights, biases });
    offset += weights.length + biases.length;
  }
  return { sizes: [...sizes], parameters, layers };
}

export function createBatch(model: Mlp, size: number): Batch {
  return {
    size,
    activations: model.sizes.map((width) => new Float64Array(size * width)),
    deltas: model.sizes.map((width, i) => new Float64Array(i === 0 ? 0 : size * width)),
    gradient: new Float64Array(model.parameters.length),
    inputIndices: new Int32Array(size * model.sizes[0]!),
    inputCounts: new Int32Array(size),
  };
}

export function forward(model: Mlp, batch: Batch): Float64Array {
  const width = model.sizes[0]!;
  const inputs = batch.activations[0]!;
  // Omit zero products without changing the order of any nonzero addends.
  for (let sample = 0; sample < batch.size; sample++) {
    const base = sample * width;
    let count = 0;
    for (let k = 0; k < width; k++) {
      if (inputs[base + k] !== 0) batch.inputIndices[base + count++] = k;
    }
    batch.inputCounts[sample] = count;
  }
  for (let l = 0; l < model.layers.length; l++) {
    const { input, output, weights, biases } = model.layers[l]!;
    const source = batch.activations[l]!;
    const destination = batch.activations[l + 1]!;
    const hidden = l < model.layers.length - 1;
    for (let sample = 0; sample < batch.size; sample++) {
      const base = sample * input;
      for (let j = 0; j < output; j++) {
        const weightBase = j * input;
        let sum = 0;
        // Sum inputs in increasing order, adding the bias after the dot product.
        if (l === 0) {
          const count = batch.inputCounts[sample]!;
          for (let n = 0; n < count; n++) {
            const k = batch.inputIndices[base + n]!;
            sum += source[base + k]! * weights[weightBase + k]!;
          }
        } else {
          for (let k = 0; k < input; k++) sum += source[base + k]! * weights[weightBase + k]!;
        }
        sum += biases[j]!;
        destination[sample * output + j] = hidden ? Math.max(0, sum) : sum;
      }
    }
  }
  return batch.activations[model.layers.length]!;
}

export function huberGradient(
  model: Mlp,
  batch: Batch,
  actions: Uint8Array,
  targets: Float64Array,
  masks: Float64Array,
): number {
  batch.gradient.fill(0);
  for (const delta of batch.deltas) delta.fill(0);
  const last = model.layers.length;
  const output = model.sizes[last]!;
  const values = batch.activations[last]!;
  const outputDelta = batch.deltas[last]!;
  let loss = 0;
  for (let sample = 0; sample < batch.size; sample++) {
    const index = sample * output + actions[sample]!;
    const error = values[index]! - targets[sample]!;
    const absolute = Math.abs(error);
    loss += masks[sample]! * (absolute <= 1 ? 0.5 * error * error : absolute - 0.5);
    outputDelta[index] = (masks[sample]! * Math.max(-1, Math.min(1, error))) / batch.size;
  }
  backward(model, batch);
  return loss / batch.size;
}

export function crossEntropyGradient(model: Mlp, batch: Batch, labels: Uint8Array): number {
  batch.gradient.fill(0);
  for (const delta of batch.deltas) delta.fill(0);
  const last = model.layers.length;
  const output = model.sizes[last]!;
  const values = batch.activations[last]!;
  const delta = batch.deltas[last]!;
  let loss = 0;
  for (let sample = 0; sample < batch.size; sample++) {
    const base = sample * output;
    let maximum = -Infinity;
    for (let j = 0; j < output; j++) maximum = Math.max(maximum, values[base + j]!);
    let sum = 0;
    for (let j = 0; j < output; j++) {
      const value = Math.exp(values[base + j]! - maximum);
      delta[base + j] = value;
      sum += value;
    }
    loss += Math.log(sum) - (values[base + labels[sample]!]! - maximum);
    for (let j = 0; j < output; j++) {
      delta[base + j] = (delta[base + j]! / sum - (j === labels[sample] ? 1 : 0)) / batch.size;
    }
  }
  backward(model, batch);
  return loss / batch.size;
}

function backward(model: Mlp, batch: Batch): void {
  const last = model.layers.length;
  for (let l = last - 1; l >= 0; l--) {
    const { input, output, weights, offset } = model.layers[l]!;
    const source = batch.activations[l]!;
    const activations = batch.activations[l + 1]!;
    const delta = batch.deltas[l + 1]!;
    const previous = batch.deltas[l]!;
    const gradient = batch.gradient.subarray(offset, offset + input * output);
    const biasGradient = batch.gradient.subarray(offset + input * output);
    for (let sample = 0; sample < batch.size; sample++) {
      const base = sample * input;
      for (let j = 0; j < output; j++) {
        const index = sample * output + j;
        const d = l < last - 1 && activations[index]! <= 0 ? 0 : delta[index]!;
        if (d === 0) continue;
        biasGradient[j] = biasGradient[j]! + d;
        const weightBase = j * input;
        if (l === 0) {
          const count = batch.inputCounts[sample]!;
          for (let n = 0; n < count; n++) {
            const k = batch.inputIndices[base + n]!;
            const w = weightBase + k;
            gradient[w] = gradient[w]! + d * source[base + k]!;
          }
        } else {
          for (let k = 0; k < input; k++) {
            const w = weightBase + k;
            gradient[w] = gradient[w]! + d * source[base + k]!;
          }
        }
        if (l > 0) {
          for (let k = 0; k < input; k++) {
            previous[base + k] = previous[base + k]! + d * weights[weightBase + k]!;
          }
        }
      }
    }
  }
}
