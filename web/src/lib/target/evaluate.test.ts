import assert from "node:assert/strict";
import test from "node:test";
import { configure, defaults } from "./config.ts";
import { applyAction, createEpisode, type Action } from "./episode.ts";
import { createIdealPolicy } from "./policies.ts";
import { createEventTracker, recordEvents, summarizeEvents } from "./evaluate.ts";

test("the ideal report uses inverse measurement variance and C0 removes enhancement from weights", () => {
  for (const enhanced of [true, false]) {
    const episode = createEpisode(defaults, 19, enhanced);
    const { world, attention } = episode.truth;
    world.target.row = world.target.col = 7;
    world.occluder = { row: 0, col: 0 };
    world.jumpStep = world.t = 16;
    // True category and hue deliberately disagree with the supplied measurement.
    world.target.category = 0;
    world.target.hue = 0.1;
    const cell = 7 * 16 + 7;
    const ideal = createIdealPolicy(defaults, enhanced);
    attention.resource[cell] = 0;
    episode.observation.measurement[256 + cell] = 0.1;
    assert.equal(ideal(episode.observation, episode.truth).answer, undefined);
    world.t = 30;
    attention.resource[cell] = 1;
    episode.observation.measurement[256 + cell] = 0.8;
    assert.equal(ideal(episode.observation, episode.truth).answer, enhanced ? 3 : 1);
  }
});

test("the ideal report excludes pre-jump and occluded samples and defines missing or exact data", () => {
  const episode = createEpisode(defaults, 17);
  const { world, attention } = episode.truth;
  world.target.row = world.target.col = 7;
  world.jumpStep = 16;
  const cell = 7 * 16 + 7;
  const ideal = createIdealPolicy(defaults);
  world.occluder = { row: 0, col: 0 };
  world.t = 15;
  episode.observation.measurement[256 + cell] = 100;
  ideal(episode.observation, episode.truth);
  world.t = 16;
  world.occluder = { row: 7, col: 7 };
  ideal(episode.observation, episode.truth);
  world.t = 17;
  world.occluder = { row: 0, col: 0 };
  episode.observation.measurement[256 + cell] = 0.4;
  ideal(episode.observation, episode.truth);
  world.t = 30;
  world.occluder = { row: 7, col: 7 };
  episode.observation.measurement[256 + cell] = 100;
  assert.equal(ideal(episode.observation, episode.truth).answer, 1);
  assert.equal(createIdealPolicy(defaults)(episode.observation, episode.truth).answer, 0);

  const exact = createIdealPolicy(configure({ alpha: 1 }));
  world.occluder = { row: 0, col: 0 };
  world.t = 17;
  attention.resource[cell] = 1;
  episode.observation.measurement[256 + cell] = 0.65;
  exact(episode.observation, episode.truth);
  world.t = 30;
  attention.resource[cell] = 0;
  episode.observation.measurement[256 + cell] = -100;
  assert.equal(exact(episode.observation, episode.truth).answer, 2);
});

test("the ideal command uses the specified forecast without a future draw or true hue", () => {
  const episode = createEpisode(defaults, 23);
  const { world, attention } = episode.truth;
  world.target.row = 15;
  world.target.col = 5;
  world.target.velocity = { row: 1, col: -1 };
  attention.center = { row: 15, col: 4.5 };
  attention.resource.fill(0);
  attention.resource[0] = attention.total = 1;
  const state = structuredClone(episode);
  const action = createIdealPolicy(defaults)(episode.observation, episode.truth);
  assert.deepEqual(action.command, { row: 1, col: -0 });
  assert.equal(action.effort, 1);
  assert.equal(action.paddleMove, -1);
  assert.deepEqual(episode, state);
});

test("C0 changes only measurement, answer and score, preserving attention and paddle actions", () => {
  const enhanced = createEpisode(defaults, 5000000001);
  const uniform = createEpisode(defaults, 5000000001, false);
  const firstPolicy = createIdealPolicy(defaults);
  const secondPolicy = createIdealPolicy(defaults, false);
  for (let t = 1; t <= 30; t++) {
    assert.deepEqual(enhanced.truth, uniform.truth);
    assert.deepEqual(enhanced.observation.telemetry, uniform.observation.telemetry);
    assert.deepEqual(enhanced.rng, uniform.rng);
    const { answer: firstAnswer, ...firstAction } = firstPolicy(
      enhanced.observation,
      enhanced.truth,
    );
    const { answer: secondAnswer, ...secondAction } = secondPolicy(
      uniform.observation,
      uniform.truth,
    );
    assert.deepEqual(firstAction, secondAction);
    applyAction(enhanced, { ...firstAction, answer: firstAnswer });
    applyAction(uniform, { ...secondAction, answer: secondAnswer });
  }
  assert.equal(enhanced.score!.caught, uniform.score!.caught);
  assert.equal(enhanced.score!.effortSteps, uniform.score!.effortSteps);
});

test("event counts follow each flash location, terminate on loss and censor at episode end", () => {
  const episode = createEpisode(defaults, 17);
  const { world, attention } = episode.truth;
  world.occluder = { row: 0, col: 0 };
  const tracker = createEventTracker();
  const frames = [
    { center: [2, 2], target: [2, 2], flash: false, command: 1, effort: 0 },
    { center: [2, 3], target: [9, 9], flash: true, command: 0, effort: 1 },
    { center: [2, 3], target: [2, 3], flash: true, command: 0, effort: 0 },
    { center: [8, 8], target: [2, 3], flash: false, command: 0, effort: 0 },
    { center: [8, 8], target: [2, 3], flash: true, command: 0, effort: 1 },
  ];
  for (const frame of frames) {
    attention.center = { row: frame.center[0]!, col: frame.center[1]! };
    attention.resource.fill(0);
    attention.resource[attention.center.row * 16 + attention.center.col] = 1;
    attention.total = 1;
    attention.entropy = 0;
    world.target.row = frame.target[0]!;
    world.target.col = frame.target[1]!;
    world.flash = frame.flash ? { ...attention.center, strength: 1 } : null;
    recordEvents(defaults, tracker, episode.truth, {
      command: { row: 0, col: frame.command },
      effort: frame.effort,
      paddleMove: 0,
    } as Action);
  }
  assert.equal(tracker.counts.captures, 3);
  assert.equal(tracker.counts.captureSteps, 4);
  assert.deepEqual(summarizeEvents(tracker.counts), {
    flashes: 3,
    captureRate: 1,
    captureLength: 4 / 3,
    shiftsPerEpisode: 2,
    exitsPerEpisode: 2,
    occludedRate: 1 / 5,
    commandGap: Math.sqrt(61) / 4,
    focusedRate: 2 / 5,
    targetShare: 2 / 5,
    entropy: 0,
    effortRate: 2 / 5,
  });
});
