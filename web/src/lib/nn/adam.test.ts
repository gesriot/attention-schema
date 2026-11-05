import assert from "node:assert/strict";
import test from "node:test";
import { adamStep, createAdam } from "./adam.ts";

test("Adam matches two independently calculated TensorFlow-style steps", () => {
  const parameters = new Float64Array([1, -2]);
  const adam = createAdam(2);
  const expected = [
    {
      weights: [0.9990000063245154, -1.9990000126489507],
      m: [0.05, -0.025],
      v: [0.00025, 0.0000625],
    },
    {
      weights: [0.9984889834180863, -1.9994942005053975],
      m: [0.035, 0.052500000000000005],
      v: [0.00025975000000000005, 0.0006249375000000001],
    },
  ];
  const gradients = [
    [0.5, -0.25],
    [-0.1, 0.75],
  ];
  for (let step = 0; step < 2; step++) {
    adamStep(parameters, new Float64Array(gradients[step]!), adam);
    assert.equal(adam.step, step + 1);
    for (let i = 0; i < 2; i++) {
      assert.ok(Math.abs(parameters[i]! - expected[step]!.weights[i]!) < 1e-15);
      assert.ok(Math.abs(adam.m[i]! - expected[step]!.m[i]!) < 1e-15);
      assert.ok(Math.abs(adam.v[i]! - expected[step]!.v[i]!) < 1e-15);
    }
  }
});
