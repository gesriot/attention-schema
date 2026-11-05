import assert from "node:assert/strict";
import test from "node:test";
import { createRng, nextNormal, nextUnit } from "../rng.ts";
import { createBatch, createMlp, forward, huberGradient } from "./mlp.ts";

test("Box-Muller consumes exactly two draws and discards the second normal", () => {
  const rng = createRng(17, 3);
  const reference = createRng(17, 3);
  for (let i = 0; i < 20; i++) {
    const expected =
      Math.sqrt(-2 * Math.log(1 - nextUnit(reference))) *
      Math.cos(2 * Math.PI * nextUnit(reference));
    assert.equal(nextNormal(rng), expected);
    assert.deepEqual(rng, reference);
  }
});

test("initialization uses truncated normals, uniform output weights and specified biases", () => {
  const model = createMlp([17, 31, 23, 16], createRng(1, 3));
  for (let i = 0; i < model.layers.length; i++) {
    const layer = model.layers[i]!;
    const hidden = i < model.layers.length - 1;
    const bound = hidden ? (2 * Math.sqrt(2 / layer.input)) / 0.87962566103423978 : 0.03;
    for (const weight of layer.weights) {
      assert.ok(weight >= -bound && (hidden ? weight <= bound : weight < bound));
    }
    for (const bias of layer.biases) assert.equal(bias, hidden ? 0 : -0.2);
  }
  const rng = createRng(9, 3);
  const reference = createRng(9, 3);
  const small = createMlp([1, 20, 1], rng);
  const scale = Math.sqrt(2) / 0.87962566103423978;
  for (const weight of small.layers[0]!.weights) {
    let normal: number;
    do normal = nextNormal(reference); while (Math.abs(normal) > 2);
    assert.equal(weight, normal * scale);
  }
  for (const weight of small.layers[1]!.weights) {
    assert.equal(weight, -0.03 + 0.06 * nextUnit(reference));
  }
  assert.deepEqual(rng, reference);
  assert.deepEqual(model.parameters, createMlp([17, 31, 23, 16], createRng(1, 3)).parameters);
});

test("forward is row-major and applies ReLU only to hidden layers", () => {
  const model = createMlp([2, 2, 1], createRng(1, 3));
  model.parameters.set([1, -2, -1, 3, 0.5, -0.5, 2, -1, -0.2]);
  const batch = createBatch(model, 2);
  batch.activations[0]!.set([1, 2, -1, 0]);
  const output = forward(model, batch);
  assert.deepEqual([...batch.activations[1]!], [0, 4.5, 0, 0.5]);
  assert.deepEqual([...output], [-4.7, -0.7]);
});

test("omitting zero inputs preserves the exact ordered dense forward result", () => {
  const model = createMlp([31, 7, 5, 3], createRng(23, 3));
  const batch = createBatch(model, 9);
  const rng = createRng(27, 4);
  for (let i = 0; i < batch.activations[0]!.length; i++) {
    batch.activations[0]![i] = i % 3 === 0 ? nextUnit(rng) - 0.5 : 0;
  }
  let source = batch.activations[0]!.slice();
  for (let l = 0; l < model.layers.length; l++) {
    const layer = model.layers[l]!;
    const output = new Float64Array(batch.size * layer.output);
    for (let sample = 0; sample < batch.size; sample++) {
      for (let j = 0; j < layer.output; j++) {
        let sum = 0;
        for (let k = 0; k < layer.input; k++) {
          sum += source[sample * layer.input + k]! * layer.weights[j * layer.input + k]!;
        }
        sum += layer.biases[j]!;
        output[sample * layer.output + j] = l < model.layers.length - 1 ? Math.max(0, sum) : sum;
      }
    }
    source = output;
  }
  assert.deepEqual(forward(model, batch), source);
});

test("Huber gradients match finite differences with a linear branch and a masked row", () => {
  const model = createMlp([2, 3, 2], createRng(5, 3));
  model.parameters.set([
    0.6, -0.2, -0.5, 0.3, 0.4, 0.8,
    0.7, 0.9, -1,
    0.4, -0.2, 0.3, -0.6, 0.5, 0.1,
    -0.2, 0.1,
  ]);
  const batch = createBatch(model, 3);
  batch.activations[0]!.set([0.4, 0.7, -0.3, 0.2, 0.8, -0.6]);
  const actions = new Uint8Array([0, 1, 0]);
  const targets = new Float64Array([0.1, 4.3, -100]);
  const masks = new Float64Array([1, 1, 0]);
  const outputs = forward(model, batch);
  const firstError = outputs[0]! - targets[0]!;
  const secondError = outputs[3]! - targets[1]!;
  assert.ok(Math.abs(firstError) < 1);
  assert.ok(Math.abs(secondError) > 1);
  const loss = huberGradient(model, batch, actions, targets, masks);
  assert.equal(loss, (0.5 * firstError * firstError + Math.abs(secondError) - 0.5) / 3);
  const analytical = batch.gradient.slice();
  const epsilon = 1e-5;
  for (let i = 0; i < model.parameters.length; i++) {
    const original = model.parameters[i]!;
    model.parameters[i] = original + epsilon;
    forward(model, batch);
    const plus = huberGradient(model, batch, actions, targets, masks);
    model.parameters[i] = original - epsilon;
    forward(model, batch);
    const minus = huberGradient(model, batch, actions, targets, masks);
    model.parameters[i] = original;
    assert.ok(Math.abs(analytical[i]! - (plus - minus) / (2 * epsilon)) < 1e-8, `parameter ${i}`);
  }
  masks.fill(0);
  forward(model, batch);
  assert.equal(huberGradient(model, batch, actions, targets, masks), 0);
  assert.ok(batch.gradient.every((value) => value === 0));
});
