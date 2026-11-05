import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { createRng } from "../rng.ts";
import { createBatch, createMlp, crossEntropyGradient, forward, huberGradient } from "./mlp.ts";

test("cross entropy gradients match central differences", () => {
  const model = createMlp([3, 4, 3], createRng(6, 7));
  for (let i = 0; i < model.parameters.length; i++) model.parameters[i] = 0.1 + (i % 7) * 0.03;
  const batch = createBatch(model, 2);
  batch.activations[0]!.set([1, 0.3, 0, 0.2, 0.9, 1]);
  const labels = new Uint8Array([0, 2]);
  forward(model, batch);
  crossEntropyGradient(model, batch, labels);
  const expected = batch.gradient.slice();
  const epsilon = 1e-5;
  for (let i = 0; i < model.parameters.length; i++) {
    const value = model.parameters[i]!;
    model.parameters[i] = value + epsilon;
    forward(model, batch);
    const plus = crossEntropyGradient(model, batch, labels);
    model.parameters[i] = value - epsilon;
    forward(model, batch);
    const minus = crossEntropyGradient(model, batch, labels);
    model.parameters[i] = value;
    assert.ok(Math.abs(expected[i]! - (plus - minus) / (2 * epsilon)) < 1e-8, `parameter ${i}`);
  }
});

test("softmax remains finite for large logits and its deltas sum to zero", () => {
  const model = createMlp([1, 3], createRng(1, 7));
  model.parameters.set([0, 0, 0, 1000, 1001, 999]);
  const batch = createBatch(model, 1);
  forward(model, batch);
  const loss = crossEntropyGradient(model, batch, new Uint8Array([1]));
  assert.ok(Math.abs(loss - Math.log(1 + Math.exp(-1) + Math.exp(-2))) < 1e-14);
  assert.ok(Math.abs(batch.deltas[1]!.reduce((sum, value) => sum + value, 0)) < 1e-14);
});

test("Huber loss and gradient bytes retain the pre-generalization reference", () => {
  const model = createMlp([4, 5, 3], createRng(12, 3));
  const batch = createBatch(model, 3);
  batch.activations[0]!.set([0, 1, 0, 1, 1, 0, 0.25, 0, 0.5, 0, 1, 1]);
  forward(model, batch);
  const loss = huberGradient(model, batch, new Uint8Array([1, 2, 0]),
    new Float64Array([0.3, 4, -10]), new Float64Array([1, 1, 0]));
  assert.equal(loss, 1.2823665214785427);
  const digest = createHash("sha256").update(Buffer.from(batch.gradient.buffer)).digest("hex");
  assert.equal(digest, "1aa94a50184f2dc7966b1cee66ddd6fe2c5cea31a0fd55e0a6d10b94884e778f");
});
