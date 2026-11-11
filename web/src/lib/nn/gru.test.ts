import assert from "node:assert/strict";
import test from "node:test";
import { createRng, nextUnit } from "../rng.ts";
import { backwardGru, createGru, createGruBatch, forwardGru } from "./gru.ts";

function close(actual: number, expected: number, tolerance = 1e-8): void {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
}

test("GRU parameter order and generator draws skip all biases", () => {
  const rng = createRng(23, 8);
  const expected = createRng(23, 8);
  const model = createGru(3, 4, rng);
  assert.equal(model.parameters.length, 3 * 4 * (3 + 4 + 1));
  for (let gate = 0; gate < 3; gate++) {
    const offset = gate * 32;
    assert.equal(model.gates[gate]!.offset, offset);
    for (let index = 0; index < 28; index++)
      assert.equal(model.parameters[offset + index], -0.5 + nextUnit(expected));
    assert.deepEqual([...model.parameters.slice(offset + 28, offset + 32)], [0, 0, 0, 0]);
  }
  assert.deepEqual(rng, expected);
  assert.throws(() => createGru(0, 4, rng));
});

test("GRU starts each sequence at zero and applies reset before the recurrent candidate matrix", () => {
  const model = createGru(1, 2, createRng(1, 1));
  model.parameters.set([
    0.2, -0.3, 0.1, 0.4, -0.2, 0.5, 0.05, -0.1, -0.4, 0.2, 0.3, -0.5, 0.7, 0.2, -0.02, 0.03, 0.5,
    -0.7, -0.6, 0.8, 0.4, -0.9, 0.1, 0.2,
  ]);
  const batch = createGruBatch(model, 2, 2);
  batch.inputs.set([0.8, -0.4, -0.2, 0.6]);
  batch.states.fill(999);
  forwardGru(model, batch);
  const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
  const tanh = (x: number) => 1 - 2 / (1 + Math.exp(2 * x));
  for (let sequence = 0; sequence < 2; sequence++) {
    let h = [0, 0];
    assert.deepEqual([...batch.states.slice(sequence * 6, sequence * 6 + 2)], h);
    for (let step = 0; step < 2; step++) {
      const x = batch.inputs[sequence * 2 + step]!;
      const r = [
        sigmoid(0.2 * x + 0.1 * h[0]! + 0.4 * h[1]! + 0.05),
        sigmoid(-0.3 * x - 0.2 * h[0]! + 0.5 * h[1]! - 0.1),
      ];
      const z = [
        sigmoid(-0.4 * x + 0.3 * h[0]! - 0.5 * h[1]! - 0.02),
        sigmoid(0.2 * x + 0.7 * h[0]! + 0.2 * h[1]! + 0.03),
      ];
      const n = [
        tanh(0.5 * x - 0.6 * r[0]! * h[0]! + 0.8 * r[1]! * h[1]! + 0.1),
        tanh(-0.7 * x + 0.4 * r[0]! * h[0]! - 0.9 * r[1]! * h[1]! + 0.2),
      ];
      h = h.map((previous, j) => (1 - z[j]!) * n[j]! + z[j]! * previous);
      for (let j = 0; j < 2; j++) close(batch.states[sequence * 6 + (step + 1) * 2 + j]!, h[j]!);
    }
  }
  const first = batch.states.slice();
  forwardGru(model, batch);
  assert.deepEqual(batch.states, first);
});

test("exp-based candidate saturates without NaN for large positive and negative arguments", () => {
  const model = createGru(1, 2, createRng(1, 1));
  model.parameters.fill(0);
  model.gates[2]!.biases.set([1e300, -1e300]);
  const batch = createGruBatch(model, 1, 1);
  forwardGru(model, batch);
  assert.deepEqual([...batch.candidate], [1, -1]);
  assert.deepEqual([...batch.states], [0, 0, 0.5, -0.5]);
});

test("GRU BPTT matches every central difference and accumulates parameter gradients", () => {
  const rng = createRng(7, 4);
  const model = createGru(3, 4, rng);
  const batch = createGruBatch(model, 2, 5);
  for (let i = 0; i < batch.inputs.length; i++)
    batch.inputs[i] = i % 3 === 0 ? 0 : nextUnit(rng) - 0.5;
  const derivative = Float64Array.from({ length: 2 * 5 * 4 }, () => nextUnit(rng) - 0.5);
  const loss = () => {
    forwardGru(model, batch);
    let sum = 0;
    for (let b = 0; b < 2; b++)
      for (let t = 0; t < 5; t++)
        for (let j = 0; j < 4; j++)
          sum += batch.states[(b * 6 + t + 1) * 4 + j]! * derivative[(b * 5 + t) * 4 + j]!;
    return sum;
  };
  loss();
  const gradient = new Float64Array(model.parameters.length);
  backwardGru(model, batch, derivative, gradient);
  const initial = gradient.slice();
  backwardGru(model, batch, derivative, gradient);
  for (let i = 0; i < gradient.length; i++) close(gradient[i]!, 2 * initial[i]!);
  for (let i = 0; i < model.parameters.length; i++) {
    const value = model.parameters[i]!;
    model.parameters[i] = value + 1e-5;
    const plus = loss();
    model.parameters[i] = value - 1e-5;
    const minus = loss();
    model.parameters[i] = value;
    close(initial[i]!, (plus - minus) / 2e-5);
  }
});

test("skipping zero inputs preserves forward and backward bits", () => {
  const model = createGru(5, 4, createRng(1, 2));
  const dense = createGruBatch(model, 2, 5);
  const sparse = createGruBatch(model, 2, 5);
  for (let i = 0; i < dense.inputs.length; i++) dense.inputs[i] = i % 3 === 0 ? i / 50 : 0;
  sparse.inputs.set(dense.inputs);
  forwardGru(model, dense, false);
  forwardGru(model, sparse);
  assert.deepEqual(Buffer.from(dense.states.buffer), Buffer.from(sparse.states.buffer));
  const derivative = new Float64Array(40).fill(0.3);
  const dg = new Float64Array(model.parameters.length);
  const sg = new Float64Array(model.parameters.length);
  backwardGru(model, dense, derivative, dg);
  backwardGru(model, sparse, derivative, sg);
  assert.deepEqual(Buffer.from(dg.buffer), Buffer.from(sg.buffer));
});
