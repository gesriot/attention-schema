import assert from "node:assert/strict";
import test from "node:test";
import { createAdam } from "../nn/adam.ts";
import { createBatch, createMlp, forward } from "../nn/mlp.ts";
import { createRng } from "../rng.ts";
import {
  appendFrame,
  clearObservationField,
  collectSteps,
  createCollector,
  evaluateDqn,
  greedyAction,
  trainSubject,
  updateDqn,
  type TrainJob,
} from "./dqn.ts";
import { createReplayBatch, LAST, MID } from "./replay.ts";
import { createWorld, stepWorld, type WorldConfig } from "./world.ts";

test("commands encode only the previous joint action and start blank", () => {
  const config: WorldConfig = {
    variant: "wilterson",
    schema: "commands",
    noise: 0.5,
    windowActive: true,
  };
  for (let action = 0; action < 16; action++) {
    const first = createWorld(config, 1);
    const second = createWorld(config, 12);
    second.window = { row: 8, col: 1 };
    assert.ok(first.frame.slice(100).every((pixel) => pixel === 0));
    stepWorld(first, action);
    stepWorld(second, action);
    assert.notDeepEqual(first.window, second.window);
    for (let i = 0; i < 100; i++) assert.equal(first.frame[100 + i], i === action ? 1 : 0);
    assert.deepEqual(first.frame.slice(100), second.frame.slice(100));
    stepWorld(first, (action + 1) % 16);
    assert.equal(first.frame[100 + action], 0);
    assert.equal(first.frame[100 + ((action + 1) % 16)], 1);
    const given = createWorld({ ...config, schema: "given" }, 1);
    stepWorld(given, action);
    stepWorld(given, (action + 1) % 16);
    assert.deepEqual(first.frame.slice(0, 100), given.frame.slice(0, 100));
    assert.equal(first.reward, given.reward);
    assert.deepEqual(first.rng, given.rng);
  }
});

test("observation appends frames, drops the oldest, and resets at episode start", () => {
  const observation = new Uint8Array(2000);
  appendFrame(observation, new Uint8Array(200).fill(1), true);
  assert.ok(observation.slice(0, 1800).every((pixel) => pixel === 0));
  for (let i = 2; i <= 12; i++) appendFrame(observation, new Uint8Array(200).fill(i), false);
  for (let frame = 0; frame < 10; frame++) {
    assert.ok(
      observation.slice(frame * 200, frame * 200 + 200).every((pixel) => pixel === frame + 3),
    );
  }
  appendFrame(observation, new Uint8Array(200).fill(13), true);
  assert.ok(observation.slice(0, 1800).every((pixel) => pixel === 0));
  assert.ok(observation.slice(1800).every((pixel) => pixel === 13));
});

test("field ablation clears only the lower half of every frame", () => {
  const observation = Uint8Array.from({ length: 2000 }, (_, i) => i % 251);
  const before = observation.slice();
  clearObservationField(observation);
  for (let i = 0; i < 2000; i++) assert.equal(observation[i], i % 200 < 100 ? before[i] : 0);
});

test("collection counts 1000 real steps and resumes the same episode across iterations", () => {
  const collector = createCollector("schema", 2);
  let calls = 0;
  const policy = (...args: [Uint8Array]) => {
    assert.equal(args.length, 1);
    assert.equal(args[0].length, 2000);
    calls++;
    return 8;
  };
  collectSteps(collector, policy);
  assert.equal(collector.environmentSteps, 1000);
  assert.equal(collector.boundarySteps, 111);
  assert.equal(collector.episode, 112);
  assert.equal(collector.world.t, 2);
  assert.equal(collector.replay.size, 1111);
  assert.equal(calls, 1111);
  const frame = collector.world.frame.slice();
  const last = collector.replay.size;
  collectSteps(collector, policy, 1);
  assert.equal(collector.episode, 112);
  assert.equal(collector.world.t, 3);
  assert.deepEqual(
    collector.replay.observations.slice(last * 2000 + 1800, (last + 1) * 2000),
    frame,
  );
  for (let episode = 0; episode < 111; episode++) {
    const terminalTransition = episode * 10 + 8;
    const boundary = episode * 10 + 9;
    assert.equal(collector.replay.types[terminalTransition], MID);
    assert.equal(collector.replay.discounts[terminalTransition], 0);
    assert.equal(Math.abs(collector.replay.rewards[terminalTransition]!), 2);
    assert.equal(collector.replay.types[boundary], LAST);
    assert.equal(collector.replay.rewards[boundary], 0);
  }
  collectSteps(collector, policy, 999);
  assert.equal(collector.environmentSteps, 2000);
  assert.equal(collector.episode, 223);
  assert.equal(collector.world.t, 3);
});

test("a three-state chain converges to analytical discounted Q values", () => {
  const model = createMlp([3, 1], createRng(1, 3));
  const adam = createAdam(model.parameters.length);
  const batch = createBatch(model, 3);
  const nextBatch = createBatch(model, 3);
  const sample = createReplayBatch(3, 3);
  sample.observations.set([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  sample.nextObservations.set([0, 1, 0, 0, 0, 1, 0, 0, 0]);
  sample.rewards.set([0, 0, 1]);
  sample.discounts.set([0.4, 0.4, 0]);
  sample.masks.fill(1);
  const targets = new Float64Array(3);
  for (let i = 0; i < 6000; i++) updateDqn(model, adam, batch, nextBatch, sample, targets);
  const values = forward(model, batch);
  for (const [i, expected] of [0.16, 0.4, 1].entries()) {
    assert.ok(Math.abs(values[i]! - expected) < 1e-6, `${values[i]} versus ${expected}`);
  }
});

test("targets use current weights, stop gradients, and mask the reset transition", () => {
  const model = createMlp([1, 2], createRng(1, 3));
  model.parameters.set([2, 3, 0, 0]);
  const batch = createBatch(model, 2);
  const next = createBatch(model, 2);
  const sample = createReplayBatch(2, 1);
  sample.observations.set([1, 100]);
  sample.nextObservations.set([2, 200]);
  sample.actions.set([0, 1]);
  sample.rewards.set([0.5, -100]);
  sample.discounts.fill(0.4);
  sample.masks.set([1, 0]);
  const targets = new Float64Array(2);
  updateDqn(model, createAdam(4), batch, next, sample, targets);
  assert.equal(targets[0], 0.5 + 0.4 * 6);
  assert.ok(Math.abs(batch.gradient[0]! - (-0.9 / 2)) < 1e-15);
  assert.equal(batch.gradient[1], 0);
  assert.equal(batch.gradient[3], 0);
  assert.equal(model.parameters[1], 3);
  assert.equal(greedyAction(new Float64Array([4, 4, 1])), 0);
});

test("training repeats exactly and evaluation leaves weights and training streams intact", () => {
  const job: TrainJob = {
    subject: 3,
    condition: "commands",
    iterations: 2,
    evalEvery: 2,
    config: { sizes: [2000, 3, 16], collectSteps: 12, updates: 2, batchSize: 4, evalEpisodes: 2 },
  };
  const first = trainSubject(job);
  const second = trainSubject(job);
  assert.deepEqual(first, second);
  const moreEvaluation = trainSubject({ ...job, evalEvery: 1 });
  assert.deepEqual(first.model.parameters, moreEvaluation.model.parameters);
  const before = first.model.parameters.slice();
  const normal = evaluateDqn(first.model, "commands", "normal", 3);
  assert.deepEqual(normal, evaluateDqn(first.model, "commands", "normal", 3));
  assert.deepEqual(first.model.parameters, before);
});
