import assert from "node:assert/strict";
import test from "node:test";
import { createRng, nextNormal } from "../rng.ts";
import { defaults, streams } from "./config.ts";
import { noiseMaxima, summarizeFalseAlarms } from "./false-alarms.ts";
import { localNoise } from "./perception.ts";

test("noise maxima use the perception estimator and exactly 512 normal draws per frame", () => {
  const rng = createRng(1, streams.falseAlarm);
  const expectedRng = createRng(1, 24);
  const frames = 5;
  const maxima = noiseMaxima(defaults, frames, rng);
  const expected = new Float64Array(frames);
  for (let frame = 0; frame < frames; frame++) {
    const measurement = Float64Array.from({ length: 512 }, () => nextNormal(expectedRng));
    const noise = localNoise(measurement, 16);
    let maximum = -Infinity;
    for (let cell = 0; cell < 256; cell++) {
      maximum = Math.max(maximum, measurement[cell]! / Math.max(noise[cell]!, defaults.sigmaFloor));
    }
    expected[frame] = maximum;
  }
  assert.deepEqual(maxima, expected);
  assert.deepEqual(rng, expectedRng);
  assert.deepEqual(maxima, noiseMaxima(defaults, frames, createRng(1, streams.falseAlarm)));
  // Neither brightness gating nor the world's measurement level belongs in this control.
  const changed = { ...defaults, minBrightness: 1e9, sigmaEta: 0 };
  assert.deepEqual(maxima, noiseMaxima(changed, frames, createRng(1, streams.falseAlarm)));
});

test("false alarms include threshold ties and use the specified percentile index", () => {
  const values = Float64Array.from({ length: 40 }, (_, index) => 39 - index);
  assert.deepEqual(summarizeFalseAlarms(values, 38), {
    frames: 40,
    rate: 2 / 40,
    percentile95: 38,
  });
  assert.equal(values[0], 39);
  assert.throws(() => summarizeFalseAlarms(new Float64Array(0), 10));
  assert.throws(() => noiseMaxima(defaults, 0, createRng(1, streams.falseAlarm)));
});
