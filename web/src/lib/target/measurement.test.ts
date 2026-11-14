import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { createRng, nextNormal } from "../rng.ts";
import { configure, defaults, streams } from "./config.ts";
import { applyAction, createEpisode } from "./episode.ts";
import { telemetry, telemetryWidth } from "./measurement.ts";
import { createPerception, poolPerception, updatePerception } from "./perception.ts";
import { createNeutralPolicy } from "./policies.ts";

function assertBits(actual: Float64Array, expected: Float64Array): void {
  assert.deepEqual(
    Buffer.from(actual.buffer, actual.byteOffset, actual.byteLength),
    Buffer.from(expected.buffer, expected.byteOffset, expected.byteLength),
  );
}

test("period one preserves complete observation traces recorded before periodic telemetry", () => {
  // Pre-change fixtures: all observation fields at all steps, encoded as little-endian binary64.
  const references = [
    [61, "0e8c7398dddbe2480942fd3b4541cae2387cccda7f44287f47372df566c8f30f"],
    [5000000001, "2d2f27500f52be990cc817d4918e3a175a151ea98144bbc60d83ce4881baffbf"],
    [5000000002, "f02079f5ac80d6fb07cf877310cdfd6192983187cd90a43c4075a8742b0a7d2b"],
    [6000000007, "e39bc164963710634641de090bb604a8a715013301d15b2356eaa9ff83494fef"],
  ] as const;
  for (const [seed, expected] of references) {
    const episode = createEpisode(defaults, seed);
    const policy = createNeutralPolicy(defaults, seed);
    const hash = createHash("sha256");
    for (let t = 1; t <= defaults.episodeSteps; t++) {
      const o = episode.observation;
      const values = [
        ...o.measurement,
        o.previousCommand.row,
        o.previousCommand.col,
        o.previousEffort,
        o.previousPaddleMove,
        o.paddle,
        ...o.telemetry,
      ];
      const bytes = Buffer.alloc(values.length * 8);
      values.forEach((value, index) => bytes.writeDoubleLE(value, index * 8));
      hash.update(bytes);
      assert.equal(o.telemetry.length, 16);
      applyAction(episode, policy(o));
    }
    assert.equal(hash.digest("hex"), expected);
  }
});

test("period three masks only telemetry and leaves every stream and neutral trajectory unchanged", () => {
  const config = configure({ telemetryPeriod: 3 });
  for (const seed of [61, 5000000001, 5000000002, 6000000007]) {
    const every = createEpisode(defaults, seed);
    const periodic = createEpisode(config, seed);
    const policy = createNeutralPolicy(defaults, seed);
    const periodicPolicy = createNeutralPolicy(config, seed);
    let perception = createPerception();
    let periodicPerception = createPerception();
    const present: number[] = [];
    for (let t = 1; t <= defaults.episodeSteps; t++) {
      assert.equal(every.truth.world.t, t);
      assert.equal(periodic.truth.world.t, t);
      assert.deepEqual(periodic.truth, every.truth);
      assert.deepEqual(periodic.rng, every.rng);
      assertBits(periodic.truth.world.frame, every.truth.world.frame);
      assertBits(periodic.truth.attention.resource, every.truth.attention.resource);
      assertBits(periodic.observation.measurement, every.observation.measurement);
      const { telemetry: actual, ...other } = periodic.observation;
      const { telemetry: expected, ...original } = every.observation;
      assert.deepEqual(other, original);
      assert.deepEqual(Object.keys(periodic.observation), Object.keys(every.observation));
      assert.equal(actual.length, 17);
      if ([1, 4, 7, 10, 13, 16, 19, 22, 25, 28].includes(t)) {
        present.push(t);
        assert.equal(actual[16], 1);
        assertBits(actual.subarray(0, 16), expected);
      } else {
        assertBits(actual, new Float64Array(17));
      }
      perception = updatePerception(defaults, perception, every.observation.measurement);
      periodicPerception = updatePerception(
        config,
        periodicPerception,
        periodic.observation.measurement,
      );
      assert.deepEqual(periodicPerception, perception);
      assertBits(poolPerception(config, periodicPerception), poolPerception(defaults, perception));
      const action = policy(every.observation);
      const periodicAction = periodicPolicy(periodic.observation);
      assert.deepEqual(periodicAction, action);
      applyAction(every, action);
      applyAction(periodic, periodicAction);
    }
    assert.deepEqual(present, [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]);
    assert.deepEqual(periodic.rng, every.rng);
    assert.deepEqual(periodic.score, every.score);
  }
});

test("telemetry width and draw count hold for every block size and period including zero noise", () => {
  const resource = Float64Array.from({ length: 256 }, (_, index) => (index % 7) / 10);
  for (const k of [2, 4, 8, 16]) {
    for (const sigmaNu of [0, defaults.sigmaNu]) {
      const baseline = configure({ k, sigmaNu });
      const blocks = (baseline.fieldSize / k) ** 2;
      assert.equal(telemetryWidth(baseline), blocks);
      for (const period of [2, 3, 4, defaults.episodeSteps]) {
        const config = configure({ k, sigmaNu, telemetryPeriod: period });
        assert.equal(telemetryWidth(config), blocks + 1);
        const rng = createRng(1, streams.telemetry);
        const everyRng = createRng(1, streams.telemetry);
        const expectedRng = createRng(1, streams.telemetry);
        for (let t = 1; t <= config.episodeSteps; t++) {
          const values = telemetry(config, resource, t, rng);
          const every = telemetry(baseline, resource, t, everyRng);
          for (let block = 0; block < blocks; block++) nextNormal(expectedRng);
          assert.deepEqual(rng, expectedRng);
          assert.deepEqual(rng, everyRng);
          assert.equal(values.length, blocks + 1);
          if ((t - 1) % period === 0) {
            assertBits(values.subarray(0, blocks), every);
            assert.equal(values[blocks], 1);
          } else assertBits(values, new Float64Array(blocks + 1));
        }
      }
    }
  }
});
