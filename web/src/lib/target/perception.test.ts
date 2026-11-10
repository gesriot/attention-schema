import assert from "node:assert/strict";
import test from "node:test";
import { defaults } from "./config.ts";
import { createEpisode } from "./episode.ts";
import { createPerception, localNoise, poolPerception, updatePerception } from "./perception.ts";
import { perceptionAction } from "./perception-controls.ts";

function frame(pixels: [number, number, number, number][] = []): Float64Array {
  const values = new Float64Array(512);
  for (const [row, col, brightness, hue] of pixels) {
    values[row * 16 + col] = brightness;
    values[256 + row * 16 + col] = hue;
  }
  return values;
}

test("local noise excludes the center, uses absolute values and averages even median pairs", () => {
  const flat = new Float64Array(512).fill(2);
  flat[5 * 16 + 5] = 1000;
  assert.equal(localNoise(flat, 16)[5 * 16 + 5], 1.4826 * 2);
  const values = new Float64Array([1, -2, 3, -4, 1000, 5, -6, 7, -8]);
  const noise = localNoise(values, 3);
  assert.equal(noise[4], 1.4826 * 4.5);
  assert.equal(noise[0], 1.4826 * 4);
  assert.equal(noise[1], 1.4826 * 4);
});

test("initialization takes the first maximum even below threshold and leaves slot one inactive", () => {
  const state = updatePerception(
    defaults,
    createPerception(),
    frame([
      [4, 3, 0.1, 0.2],
      [4, 5, 0.1, 0.8],
    ]),
  );
  const slot = state.slots[0];
  assert.equal(slot.row, 4);
  assert.equal(slot.col, 3);
  assert.equal(slot.evidence, 2);
  assert.equal(slot.hue, 0.2);
  assert.equal(slot.hueWeight, 1 / (0.05 * 0.05));
  assert.equal(slot.velocityRow, 0);
  assert.equal(slot.predictionSupport, 0);
  assert.equal(slot.uncertainty, 0);
  assert.equal(state.slots[1].active, false);
});

test("detection respects gates, threshold, row-major ties and smoothed velocity", () => {
  let state = updatePerception(defaults, createPerception(), frame([[5, 5, 1, 0.2]]));
  const original = structuredClone(state);
  state = updatePerception(
    defaults,
    state,
    frame([
      [5, 6, 0.2, 0.4],
      [12, 12, 10, 0.8],
    ]),
  );
  assert.equal(original.slots[0].col, 5);
  assert.equal(state.slots[0].col, 6);
  assert.equal(state.slots[0].velocityCol, 0.5);
  assert.equal(state.slots[0].predictionSupport, 0.5);
  assert.equal(state.slots[0].evidence, 4);
  assert.equal(state.slots[1].active, true);
  assert.equal(state.slots[1].row, 12);
  state = updatePerception(defaults, state, frame([[5, 7, 0.199, 0.5]]));
  assert.equal(state.slots[0].col, 7);
  assert.equal(state.slots[0].evidence, 0);
  assert.equal(state.slots[0].predictionSupport, 0);
  assert.equal(state.slots[0].missed, 1);
  assert.equal(state.slots[0].velocityCol, 0.5);
  const tie = updatePerception(
    defaults,
    original,
    frame([
      [3, 4, 1, 0.1],
      [3, 5, 1, 0.9],
    ]),
  );
  assert.equal(tie.slots[0].row, 3);
  assert.equal(tie.slots[0].col, 4);
});

test("slot one activates outside the gate, excludes slot zero and expires after consecutive misses", () => {
  let state = updatePerception(defaults, createPerception(), frame([[5, 5, 1, 0.2]]));
  state = updatePerception(
    defaults,
    state,
    frame([
      [5, 5, 1, 0.2],
      [5, 7, 0.8, 0.7],
    ]),
  );
  assert.equal(state.slots[1].active, false);
  state = updatePerception(
    defaults,
    state,
    frame([
      [5, 5, 1, 0.2],
      [5, 8, 0.8, 0.7],
    ]),
  );
  assert.equal(state.slots[1].active, true);
  state.slots[1].col = 6;
  state = updatePerception(
    defaults,
    state,
    frame([
      [5, 5, 1, 0.2],
      [5, 6, 0.8, 0.7],
    ]),
  );
  assert.equal(state.slots[0].col, 5);
  assert.equal(state.slots[1].col, 6);
  for (let missed = 1; missed <= 4; missed++) {
    state = updatePerception(defaults, state, frame());
    assert.equal(state.slots[0].active, true);
    assert.equal(state.slots[0].missed, missed);
    assert.equal(state.slots[1].active, missed < 3);
  }
  assert.deepEqual([...poolPerception(defaults, state).slice(9)], new Array(9).fill(0));
});

test("the hue estimate follows repeated new observations without knowing a jump time", () => {
  let state = updatePerception(defaults, createPerception(), frame([[5, 5, 1, 0.1]]));
  for (let i = 0; i < 10; i++) {
    state = updatePerception(defaults, state, frame([[5, 5, 1, 0.9]]));
  }
  assert.ok(Math.abs(state.slots[0].hue - 0.9) < 0.001);
  assert.equal(state.slots[0].predictionSupport, 1);
  const pooled = poolPerception(defaults, state);
  assert.equal(pooled.length, 18);
  assert.deepEqual([...pooled.slice(0, 9)], [1, 5 / 15, 5 / 15, 0, 0, state.slots[0].hue, 2, 1, 0]);
  const noisy = new Float64Array(512).fill(0.2);
  state = updatePerception(defaults, state, noisy);
  assert.equal(state.slots[0].uncertainty, 1.4826 * 0.2);
  assert.equal(state.slots[0].evidence, 0);
});

test("perception depends only on measurement and prior slots with no mutation or truth input", () => {
  const first = createEpisode(defaults, 11);
  const second = createEpisode(defaults, 12);
  second.observation.measurement = first.observation.measurement.slice();
  second.truth.attention.resource.fill(1);
  second.observation.telemetry.fill(100);
  second.truth.world.target.hue = -100;
  const previous = createPerception();
  const snapshot = structuredClone(previous);
  const a = updatePerception(defaults, previous, first.observation.measurement);
  const b = updatePerception(defaults, previous, second.observation.measurement);
  assert.deepEqual(a, b);
  assert.deepEqual(previous, snapshot);
  const checkSignature = () => {
    // @ts-expect-error Truth cannot be passed in place of measurement.
    updatePerception(defaults, previous, first.truth);
    // @ts-expect-error No extra argument can carry telemetry or truth.
    updatePerception(defaults, previous, first.observation.measurement, first.truth);
  };
  void checkSignature;
});

test("the perception control uses slot position and hue for paddle and report", () => {
  const episode = createEpisode(defaults, 5);
  const perception = updatePerception(defaults, createPerception(), frame([[2, 1, 1, 0.8]]));
  const attention = { command: { row: 1 as const, col: -1 as const }, effort: 1 as const };
  assert.deepEqual(perceptionAction(defaults, episode.observation, perception, attention, true), {
    ...attention,
    paddleMove: -1,
    answer: 3,
  });
});
