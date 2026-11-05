import assert from "node:assert/strict";
import test from "node:test";
import { parseOptions } from "./train-cli.ts";
import { holdResource } from "./dqn.ts";
import { createWorld, moveWindow, stepWorld, type WorldConfig } from "./world.ts";

const field = (frame: Uint8Array) => Array.from(frame.slice(100));
const base = { variant: "piefke", noise: 0.5, windowActive: true } as const;

test("the parked field equals a free square that only ever moves straight down", () => {
  for (let seed = 1; seed <= 50; seed++) {
    const free = createWorld({ ...base, schema: "free" }, seed);
    const parked = createWorld({ ...base, schema: "parked" }, seed);
    assert.deepEqual(field(parked.frame), field(free.frame));
    for (let t = 1; t < 10; t++) {
      stepWorld(free, holdResource((seed + t) % 18));
      stepWorld(parked, Math.floor(((seed + t) % 18) / 9) * 3 + Math.floor(((seed + t) % 9) / 3));
      assert.deepEqual(field(parked.frame), field(free.frame));
    }
  }
});

test("the clock field shows only time and the start field only the starting column", () => {
  const clocks = [createWorld({ ...base, schema: "clock" }, 1), createWorld({ ...base, schema: "clock" }, 2)];
  for (let t = 1; t <= 9; t++) {
    assert.deepEqual(field(clocks[0]!.frame), field(clocks[1]!.frame));
    const row = Math.min(t, 8);
    for (let y = 0; y < 10; y++) {
      const expected = Math.abs(y - row) <= 1 ? 1 : 0;
      for (let x = 0; x < 10; x++) assert.equal(field(clocks[0]!.frame)[y * 10 + x], expected);
    }
    if (t < 9) for (const world of clocks) stepWorld(world, 0);
  }
  const start = createWorld({ ...base, schema: "start" }, 3);
  const first = field(start.frame);
  const col = Math.max(1, Math.min(8, start.startCol));
  for (let y = 0; y < 10; y++) {
    for (let x = 0; x < 10; x++) assert.equal(first[y * 10 + x], Math.abs(x - col) <= 1 ? 1 : 0);
  }
  stepWorld(start, 0);
  assert.deepEqual(field(start.frame), first);
});

test("a slip moves the window past the command without touching the noise or the ball", () => {
  const plain: WorldConfig = { variant: "wilterson", schema: "given", noise: 0.5, windowActive: true };
  let slips = 0;
  for (let seed = 1; seed <= 2000; seed++) {
    const steady = createWorld(plain, seed);
    const slipping = createWorld({ ...plain, slip: 0.2 }, seed);
    for (let t = 1; t < 10; t++) {
      const expected = moveWindow(slipping.window, "wilterson", 0);
      stepWorld(steady, 0);
      stepWorld(slipping, 0);
      if (slipping.window.row !== expected.row || slipping.window.col !== expected.col) slips++;
      assert.deepEqual(slipping.ball, steady.ball);
      assert.deepEqual(slipping.rng.noise, steady.rng.noise);
    }
  }
  const share = slips / (2000 * 9);
  assert.ok(share > 0.1 && share < 0.2, `share ${share}`);
});

test("slip is accepted only as a probability", () => {
  assert.equal(parseOptions(["--slip", "0.2"]).slip, 0.2);
  assert.equal(parseOptions([]).slip, undefined);
  for (const value of ["0", "1", "-0.1", "x"]) assert.throws(() => parseOptions(["--slip", value]));
});
