import assert from "node:assert/strict";
import test from "node:test";
import { createRng } from "../rng.ts";
import {
  clearObservationField,
  collectSteps,
  createCollector,
  holdResource,
  randomizeResource,
  trainDqn,
} from "./dqn.ts";
import { effectiveSubject, makeEmergenceJob, noiseGrid, type EmergenceSettings } from "./emergence.ts";
import { createWorld, decodeAction, stepWorld } from "./world.ts";

const small: EmergenceSettings = {
  training: { sizes: [200, 4, 18], collectSteps: 12, updates: 2, batchSize: 4, evalEpisodes: 2 },
  probes: { episodes: 8, trainEpisodes: 6, hidden: 4, batchSize: 4, updates: 3 },
};

test("free uses a single frame, eighteen outputs and the requested transition discounts", () => {
  const job = makeEmergenceJob("free", 0.5, 1, 2, 1);
  assert.deepEqual(job.config.sizes, [200, 200, 200, 200, 18]);
  assert.equal(makeEmergenceJob("given", 0.5, 1, 2, 1).config.sizes.at(-1), 6);
  assert.equal(makeEmergenceJob("off", 0.5, 1, 2, 1).config.sizes.at(-1), 6);
  const collector = createCollector(job.environment.world, 201, 1, 0.9);
  collectSteps(collector, (observation) => {
    assert.deepEqual(observation, collector.world.frame);
    return 17;
  }, 20);
  assert.equal(collector.replay.width, 200);
  assert.equal(collector.environmentSteps, 20);
  assert.equal(collector.boundarySteps, 2);
  for (let i = 0; i < collector.replay.size; i++) {
    assert.equal(collector.replay.discounts[i], i % 10 === 8 ? 0 : 0.9);
    assert.equal(collector.replay.actions[i], 17);
  }
});

test("random resource actions preserve policy paddle and window commands and world noise", () => {
  const config = makeEmergenceJob("free", 0.5, 1, 1, 1).environment.world;
  const rng = createRng(201, 6);
  let changed = false;
  for (let action = 0; action < 18; action++) {
    const world = createWorld(config, 71);
    const randomized = createWorld(config, 71);
    const replacement = randomizeResource(action, rng);
    const original = decodeAction(config, action);
    const replaced = decodeAction(config, replacement);
    assert.equal(original.paddle, replaced.paddle);
    assert.equal(original.attention, replaced.attention);
    stepWorld(world, action);
    stepWorld(randomized, replacement);
    assert.deepEqual(world.window, randomized.window);
    assert.deepEqual(world.ball, randomized.ball);
    assert.equal(world.paddle, randomized.paddle);
    assert.equal(world.reward, randomized.reward);
    assert.deepEqual(world.rng, randomized.rng);
    assert.deepEqual(world.frame.slice(0, 100), randomized.frame.slice(0, 100));
    changed ||= world.resource!.col !== randomized.resource!.col;
  }
  assert.ok(changed);
});

test("a held resource keeps its column while paddle and window follow the policy", () => {
  const config = makeEmergenceJob("free", 0.5, 1, 1, 1).environment.world;
  const held = createWorld(config, 7);
  const policy = createWorld(config, 7);
  const start = held.resource!.col;
  for (let t = 1; t < 10; t++) {
    const action = (t * 11) % 18;
    stepWorld(held, holdResource(action));
    stepWorld(policy, action);
    assert.equal(held.resource!.col, start);
    assert.deepEqual(held.window, policy.window);
    assert.equal(held.paddle, policy.paddle);
  }
});

test("resource ablation clears only the second half of a single-frame input", () => {
  const world = createWorld(makeEmergenceJob("free", 0.5, 1, 1, 1).environment.world, 31);
  const observation = world.frame.slice();
  clearObservationField(observation);
  assert.deepEqual(observation.slice(0, 100), world.frame.slice(0, 100));
  assert.ok(observation.slice(100).every((pixel) => pixel === 0));
  assert.equal(world.frame.slice(100).reduce((sum, pixel) => sum + pixel, 0), 9);
});

test("noise levels use disjoint seed subjects and identical jobs repeat exact weights", () => {
  const subjects = new Set<number>();
  for (const noise of noiseGrid) {
    for (let subject = 1; subject <= 100; subject++) {
      const effective = effectiveSubject(subject, noise);
      assert.ok(!subjects.has(effective));
      subjects.add(effective);
    }
  }
  assert.throws(() => effectiveSubject(101, 0), /от 1 до 100/);
  assert.throws(() => effectiveSubject(1, 0.1), /сетки/);
  const job = makeEmergenceJob("free", 0.5, 1, 2, 1, small);
  const settings = { ...job, subject: effectiveSubject(job.subject, job.noise), checks: ["normal"] as const };
  const first = trainDqn({ ...settings, checks: [...settings.checks] });
  const second = trainDqn({ ...settings, checks: [...settings.checks] });
  assert.deepEqual(first, second);
  const other = trainDqn({ ...settings, subject: effectiveSubject(1, 0.75), checks: [...settings.checks] });
  assert.notDeepEqual(first.model.parameters, other.model.parameters);
});
