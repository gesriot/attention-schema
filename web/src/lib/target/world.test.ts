import assert from "node:assert/strict";
import test from "node:test";
import { createRng } from "../rng.ts";
import { configure, defaults, firstSeed, streams } from "./config.ts";
import { advanceWorld, createWorld, isOccluded, moveObject, renderWorld } from "./world.ts";
import { applyAction, createEpisode, runEpisode, type Action, type Policy } from "./episode.ts";
import { createIdealPolicy, createNeutralPolicy } from "./policies.ts";
import { initialAttention } from "./attention.ts";

const rest: Action = { command: { row: 0, col: 0 }, effort: 0, paddleMove: 0, answer: 0 };

test("objects stay inside, start apart and the target hue jumps once to a different category", () => {
  for (let seed = firstSeed; seed < firstSeed + 100; seed++) {
    const rng = createRng(seed, streams.world);
    const flashRng = createRng(seed, streams.flash);
    const world = createWorld(defaults, rng, flashRng);
    assert.ok(
      Math.max(
        Math.abs(world.target.row - world.distractor.row),
        Math.abs(world.target.col - world.distractor.col),
      ) >= 4,
    );
    assert.ok(world.jumpStep >= 16 && world.jumpStep <= 27);
    const distractorHue = world.distractor.hue;
    let jumps = 0;
    let previousHue = world.target.hue;
    let previousCategory = world.target.category;
    for (let t = 1; t <= 30; t++) {
      assert.equal(world.t, t);
      for (const object of [world.target, world.distractor]) {
        assert.ok(object.row >= 0 && object.row < 16 && object.col >= 0 && object.col < 16);
        assert.ok(Math.abs(object.hue - (object.category + 0.5) / 4) <= 0.05 + 1e-15);
      }
      assert.equal(world.distractor.hue, distractorHue);
      if (world.target.hue !== previousHue) {
        jumps++;
        assert.equal(t, world.jumpStep);
        assert.notEqual(world.target.category, previousCategory);
      }
      previousHue = world.target.hue;
      previousCategory = world.target.category;
      if (t < 30) advanceWorld(defaults, world, rng, flashRng);
    }
    assert.equal(jumps, 1);
    assert.throws(() => advanceWorld(defaults, world, rng, flashRng), /завершён/);
  }
});

test("reflection changes each offending velocity component before moving", () => {
  const config = configure({ pMove: 1, pTurn: 0 });
  const object = { row: 0, col: 15, velocity: { row: -1, col: 1 }, hue: 0.5, category: 2 };
  moveObject(config, object, createRng(2, streams.world));
  assert.deepEqual(object, {
    row: 1,
    col: 14,
    velocity: { row: 1, col: -1 },
    hue: 0.5,
    category: 2,
  });
  const frozen = structuredClone(object);
  moveObject(configure({ pMove: 0, pTurn: 1 }), object, createRng(2, streams.world));
  assert.deepEqual(object, frozen);
});

test("rendering hides objects under the occluder and draws target then additive flash", () => {
  const { world } = createEpisode(defaults, 31).truth;
  world.occluder = { row: 4, col: 5 };
  world.target.row = 4;
  world.target.col = 5;
  world.distractor.row = 5;
  world.distractor.col = 6;
  world.flash = { row: 4, col: 5, strength: 0.8 };
  let frame = renderWorld(defaults, world);
  assert.ok(isOccluded(defaults, world.occluder, world.target));
  assert.equal(frame[69], 0.3 + 0.8);
  assert.equal(frame[256 + 69], 0);
  assert.equal(frame[86], 0.3);
  assert.equal(frame[256 + 86], 0);
  world.target.row = world.distractor.row = 0;
  world.target.col = world.distractor.col = 0;
  world.flash = { row: 0, col: 0, strength: 1.5 };
  frame = renderWorld(defaults, world);
  assert.equal(frame[0], 2.5);
  assert.equal(frame[256], world.target.hue);
  assert.equal(frame[1], 0);
  assert.equal(frame[257], 0);
});

test("the observation contains exactly the permitted fields with no truth aliases", () => {
  const episode = createEpisode(defaults, 61);
  const { observation, truth } = episode;
  assert.deepEqual(Object.keys(observation).sort(), [
    "measurement",
    "paddle",
    "previousCommand",
    "previousEffort",
    "previousPaddleMove",
    "telemetry",
  ]);
  assert.deepEqual(Object.keys(observation.previousCommand).sort(), ["col", "row"]);
  assert.equal(observation.measurement.length, 512);
  assert.equal(observation.telemetry.length, 16);
  assert.deepEqual(observation.previousCommand, { row: 0, col: 0 });
  const before = structuredClone(truth);
  observation.measurement.fill(100);
  observation.telemetry.fill(200);
  observation.previousCommand.row = 99;
  assert.deepEqual(truth, before);
  let calls = 0;
  const policy: Policy = (...args) => {
    assert.equal(args.length, 1);
    assert.deepEqual(Object.keys(args[0]).sort(), Object.keys(observation).sort());
    calls++;
    return rest;
  };
  runEpisode(episode, policy);
  assert.equal(calls, 30);
  const checkSignature = (ordinary: Policy) => {
    // @ts-expect-error Evaluator truth is not an observation.
    ordinary(truth);
    // @ts-expect-error The policy interface accepts no truth argument.
    ordinary(observation, truth);
  };
  void checkSignature;
});

test("repeated seeds give identical full traces and policies cannot change world events", () => {
  for (let seed = firstSeed; seed < firstSeed + 8; seed++) {
    const first = createEpisode(defaults, seed);
    const repeat = createEpisode(defaults, seed);
    const idealEpisode = createEpisode(defaults, seed);
    const neutral = createNeutralPolicy(defaults, seed);
    const repeatedNeutral = createNeutralPolicy(defaults, seed);
    const ideal = createIdealPolicy(defaults);
    for (let t = 1; t <= 30; t++) {
      assert.deepEqual(first, repeat);
      assert.deepEqual(first.truth.world, idealEpisode.truth.world);
      const action = neutral(first.observation);
      assert.deepEqual(action, repeatedNeutral(repeat.observation));
      applyAction(first, action);
      applyAction(repeat, action);
      applyAction(idealEpisode, ideal(idealEpisode.observation, idealEpisode.truth));
    }
    assert.deepEqual(first.score, repeat.score);
  }
});

test("after initialization only flashes carry world information into priority", () => {
  const config = configure({ pFlash: 0 });
  const first = createEpisode(config, 12);
  const changed = createEpisode(config, 12);
  changed.truth.world.target.row = 15 - first.truth.world.target.row;
  changed.truth.world.target.hue = -10;
  changed.truth.world.distractor.col = 15 - first.truth.world.distractor.col;
  changed.truth.world.occluder = { row: 0, col: 0 };
  for (let t = 1; t <= 30; t++) {
    assert.deepEqual(first.truth.attention, changed.truth.attention);
    applyAction(first, rest);
    applyAction(changed, rest);
  }
});

test("attention uses the previous decision and the final paddle move precedes scoring", () => {
  const config = configure({ pMove: 0, pFlash: 1, sigmaEpsilon: 0 });
  const episode = createEpisode(config, 5);
  assert.deepEqual(episode.truth.attention, initialAttention(config, episode.truth.world.target));
  const action: Action = { command: { row: 1, col: -1 }, effort: 1, paddleMove: 1, answer: 0 };
  const previousObservation = episode.observation;
  applyAction(episode, action);
  assert.equal(episode.truth.world.t, 2);
  assert.equal(episode.observation.paddle, 8);
  assert.deepEqual(episode.observation.previousCommand, action.command);
  assert.notEqual(episode.observation.previousCommand, action.command);
  assert.equal(episode.observation.previousEffort, 1);
  assert.equal(episode.observation.previousPaddleMove, 1);
  assert.equal(previousObservation.paddle, 7);
  while (episode.truth.world.t < 30) applyAction(episode, rest);
  episode.paddle = 7;
  episode.truth.world.target.col = 9;
  episode.truth.world.target.category = 2;
  const before = structuredClone(episode);
  assert.throws(() => applyAction(episode, { ...rest, answer: undefined }), /категория/);
  assert.deepEqual(episode, before);
  applyAction(episode, { ...action, answer: 2 });
  assert.deepEqual(episode.score, { caught: true, correct: true, effortSteps: 2, total: 1.96 });
  assert.equal(episode.truth.world.t, 30);
  assert.throws(() => applyAction(episode, rest), /завершён/);
});
