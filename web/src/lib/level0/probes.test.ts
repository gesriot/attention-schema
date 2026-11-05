import assert from "node:assert/strict";
import test from "node:test";
import { createBatch, createMlp, forward } from "../nn/mlp.ts";
import { createRng } from "../rng.ts";
import { greedyAction, holdResource, randomizeResource } from "./dqn.ts";
import { makeEmergenceJob } from "./emergence.ts";
import {
  collectProbeFrames, fitProbe, probeData, probeSources, runProbes, splitProbeData,
  type ProbeData,
} from "./probes.ts";
import { createWorld, stepWorld } from "./world.ts";

test("probe labels are true window columns and features contain only declared pixels or time", () => {
  const config = makeEmergenceJob("free", 0.5, 1, 1, 1).environment.world;
  const model = createMlp([200, 4, 18], createRng(201, 3));
  for (const resource of ["policy", "random", "still"] as const) {
    const data = collectProbeFrames(model, config, 201, 3, resource);
    const batch = createBatch(model, 1);
    const rng = createRng(201, 6);
    let sample = 0;
    for (let episode = 1; episode <= 3; episode++) {
      const world = createWorld(config, 3000000000 + episode);
      while (world.t < 10) {
        batch.activations[0]!.set(world.frame);
        const action = greedyAction(forward(model, batch));
        stepWorld(
          world,
          resource === "random"
            ? randomizeResource(action, rng)
            : resource === "still"
              ? holdResource(action)
              : action,
        );
        if (world.t === 10) continue;
        assert.equal(data.labels[sample], world.window.col - 1);
        assert.equal(data.steps[sample], world.t);
        assert.equal(data.episodes[sample], episode);
        assert.deepEqual(data.frames.slice(sample * 200, (sample + 1) * 200), world.frame);
        sample++;
      }
    }
    assert.equal(sample, 24);
    for (const source of probeSources) {
      const selected = probeData(data, source);
      const otherLabels = probeData({ ...data, labels: new Uint8Array(24).fill(7) }, source);
      assert.deepEqual(selected.inputs, otherLabels.inputs);
      for (let i = 0; i < sample; i++) {
        const input = selected.inputs.slice(i * selected.width, (i + 1) * selected.width);
        if (source === "context") {
          assert.equal(input.length, 10);
          assert.equal(input.reduce((sum, value) => sum + value, 0), 1);
          assert.equal(input[data.steps[i]! - 1], 1);
        } else {
          const offset = i * 200 + (source === "both" || source === "visual" ? 0 : 100);
          assert.deepEqual(input, data.frames.slice(offset, offset + selected.width));
        }
      }
    }
  }
});

test("the probe split keeps every episode entirely on one side", () => {
  const data: ProbeData = {
    width: 1,
    inputs: new Uint8Array(16000),
    labels: new Uint8Array(16000),
    episodes: Uint32Array.from({ length: 16000 }, (_, i) => i % 2000 + 1),
  };
  const split = splitProbeData(data, 1600);
  assert.equal(split.train.length, 12800);
  assert.equal(split.test.length, 3200);
  const trainEpisodes = new Set(split.train.map((i) => data.episodes[i]));
  const testEpisodes = new Set(split.test.map((i) => data.episodes[i]));
  assert.equal(trainEpisodes.size, 1600);
  assert.equal(testEpisodes.size, 400);
  for (const episode of trainEpisodes) assert.ok(!testEpisodes.has(episode));
});

test("a probe learns an encoded class while a featureless input stays at the majority baseline", () => {
  const samples = 240;
  const data: ProbeData = {
    width: 8,
    inputs: new Uint8Array(samples * 8),
    labels: Uint8Array.from({ length: samples }, (_, i) => i % 8),
    episodes: Uint32Array.from({ length: samples }, (_, i) => Math.floor(i / 8) + 1),
  };
  for (let i = 0; i < samples; i++) data.inputs[i * 8 + data.labels[i]!] = 1;
  const config = { episodes: 30, trainEpisodes: 24, hidden: 16, batchSize: 32, updates: 1000 };
  const result = fitProbe(data, 1, config);
  assert.ok(result.accuracy > 0.98);
  assert.equal(result.majority, 0.125);
  const blank = fitProbe({ ...data, inputs: new Uint8Array(samples * 8) }, 1, config);
  assert.equal(blank.accuracy, blank.majority);
});

test("probes leave agent weights intact and untrained means the original initialization", () => {
  const subject = 201;
  const model = createMlp([200, 4, 18], createRng(subject, 3));
  model.parameters.fill(0);
  model.layers.at(-1)!.biases[17] = 1;
  const before = model.parameters.slice();
  const config = { episodes: 8, trainEpisodes: 6, hidden: 4, batchSize: 4, updates: 3 };
  const world = makeEmergenceJob("free", 0.5, 1, 1, 1).environment.world;
  const results = runProbes(model, world, subject, config);
  assert.deepEqual(model.parameters, before);
  assert.deepEqual(results.map((result) => result.source), [...probeSources]);
  const initial = createMlp(model.sizes, createRng(subject, 3));
  const expected = fitProbe(probeData(collectProbeFrames(initial, world, subject, 8), "untrained"), subject, config);
  assert.deepEqual(results.find((result) => result.source === "untrained"), { source: "untrained", ...expected });
});
