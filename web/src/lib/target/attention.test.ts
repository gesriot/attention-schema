import assert from "node:assert/strict";
import test from "node:test";
import { createRng, nextNormal } from "../rng.ts";
import { configure, defaults } from "./config.ts";
import {
  advanceAttention,
  bump,
  commandedCenter,
  distribute,
  initialAttention,
  resourceShare,
} from "./attention.ts";
import { measure, telemetry } from "./measurement.ts";

test("resource is capped without redistribution and effort sharpens the same priority", () => {
  const priority = bump(defaults, { row: 7, col: 7 });
  const relaxed = distribute(defaults, priority, 0);
  const effort = distribute(defaults, priority, 1);
  for (const attention of [relaxed, effort]) {
    assert.ok(attention.resource.every((value) => value >= 0 && value <= 1));
    assert.ok(attention.total <= defaults.R + 1e-12);
  }
  assert.ok(effort.entropy < relaxed.entropy);
  const point = new Float64Array(256);
  point[0] = 1000;
  const saturated = distribute(defaults, point, 0);
  assert.equal(saturated.total, 1);
  assert.equal(saturated.resource[0], 1);
});

test("initial target share exceeds 0.9 at every cell including edges", () => {
  for (let row = 0; row < 16; row++) {
    for (let col = 0; col < 16; col++) {
      const position = { row, col };
      const attention = initialAttention(defaults, position);
      assert.ok(resourceShare(defaults, attention, position) >= 0.9);
      const expected = bump(defaults, position);
      for (let cell = 0; cell < 256; cell++) {
        assert.equal(attention.priority[cell], expected[cell]! / (1 - defaults.rho));
      }
    }
  }
});

test("the bump is relative to the real center and clamps at the boundary", () => {
  const config = configure({ rho: 0, sigmaEpsilon: 0, betaEndo: 1.7 });
  const previous = initialAttention(config, { row: 6, col: 9 });
  const command = { row: 1, col: -1 };
  const expected = bump(config, { row: previous.center.row + 1, col: previous.center.col - 1 });
  const next = advanceAttention(config, previous, command, 0, null, createRng(11, 18));
  for (let i = 0; i < 256; i++) assert.equal(next.priority[i], config.betaEndo * expected[i]!);
  assert.deepEqual(commandedCenter(config, { row: 0.2, col: 14.6 }, { row: -1, col: 1 }), {
    row: 0,
    col: 15,
  });
});

test("the recurrence uses inertia, a single flash cell and row-major independent normals", () => {
  const previous = initialAttention(defaults, { row: 4, col: 5 });
  const command = { row: 0, col: 1 };
  const rng = createRng(41, 18);
  const expectedRng = createRng(41, 18);
  const expectedBump = bump(defaults, commandedCenter(defaults, previous.center, command));
  const next = advanceAttention(
    defaults,
    previous,
    command,
    1,
    { row: 10, col: 11, strength: 1.2 },
    rng,
  );
  for (let i = 0; i < 256; i++) {
    const expected =
      defaults.rho * previous.priority[i]! +
      defaults.betaEndo * expectedBump[i]! +
      defaults.betaExo * (i === 171 ? 1.2 : 0) +
      defaults.sigmaEpsilon * nextNormal(expectedRng);
    assert.equal(next.priority[i], expected);
  }
  assert.deepEqual(rng, expectedRng);
  assert.deepEqual(next.resource, distribute(defaults, next.priority, 1).resource);
});

test("measurement handles zero noise, fully enhanced cells and uniform C0 noise", () => {
  const frame = Float64Array.from({ length: 512 }, (_, i) => i / 512);
  const resource = new Float64Array(256);
  resource[71] = 1;
  const clean = measure(configure({ sigmaEta: 0 }), frame, resource, true, createRng(2, 19));
  assert.deepEqual(clean, frame);
  const enhanced = measure(configure({ alpha: 1 }), frame, resource, true, createRng(2, 19));
  assert.equal(enhanced[71], frame[71]);
  assert.equal(enhanced[327], frame[327]);
  assert.notEqual(enhanced[70], frame[70]);
  const full = new Float64Array(256).fill(1);
  const c0 = measure(defaults, frame, resource, false, createRng(2, 19));
  assert.deepEqual(c0, measure(defaults, frame, full, false, createRng(2, 19)));
  const rng = createRng(2, 19);
  for (let i = 0; i < 512; i++) {
    assert.equal(c0[i], frame[i]! + defaults.sigmaEta * nextNormal(rng));
  }
  assert.notEqual(c0[71]! - frame[71]!, c0[327]! - frame[327]!);
});

test("noiseless telemetry gives block sums and preserves total resource", () => {
  const config = configure({ sigmaNu: 0 });
  const resource = Float64Array.from({ length: 256 }, (_, i) => i / 256);
  const pooled = telemetry(config, resource, createRng(7, 20));
  assert.equal(pooled.length, 16);
  const expected = new Float64Array(16);
  for (let row = 0; row < 16; row++) {
    for (let col = 0; col < 16; col++) {
      const index = Math.floor(row / 4) * 4 + Math.floor(col / 4);
      expected[index] = expected[index]! + resource[row * 16 + col]!;
    }
  }
  assert.deepEqual(pooled, expected);
  assert.equal(
    pooled.reduce((sum, value) => sum + value, 0),
    resource.reduce((sum, value) => sum + value, 0),
  );
});
