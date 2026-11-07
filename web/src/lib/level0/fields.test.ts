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

test("a locked square keeps its row and moves only sideways", () => {
  const config = { variant: "piefke", schema: "free", noise: 0.5, windowActive: true } as const;
  const world = createWorld({ ...config, lockResource: true }, 7);
  assert.equal(world.resource!.row, 4);
  const col = world.resource!.col;
  // Attention straight down, resource one column to the right.
  stepWorld(world, 1);
  assert.equal(world.resource!.row, 4);
  assert.equal(world.resource!.col, Math.min(8, col + 1));
  assert.equal(world.window.row, 2);
});

test("the frame-read schema follows the window without reading it", () => {
  const config = { variant: "wilterson", schema: "read", noise: 0.5, windowActive: true } as const;
  let exact = 0;
  let frames = 0;
  for (let seed = 1; seed <= 300; seed++) {
    const world = createWorld(config, seed);
    for (;;) {
      frames++;
      if (world.estimate!.row === world.window.row && world.estimate!.col === world.window.col) {
        exact++;
      }
      if (world.t === 10) break;
      stepWorld(world, (seed + world.t) % 16);
    }
  }
  assert.ok(exact / frames > 0.95, `share ${exact / frames}`);
});

test("the frame-read schema draws only the estimate into the field", () => {
  const config = { variant: "wilterson", schema: "read", noise: 0.5, windowActive: true } as const;
  const world = createWorld(config, 11);
  stepWorld(world, 1);
  const { row, col } = world.estimate!;
  const expected = new Uint8Array(100);
  for (let y = row - 1; y <= row + 1; y++) expected.fill(1, y * 10 + col - 1, y * 10 + col + 2);
  assert.deepEqual(Array.from(world.frame.slice(100)), Array.from(expected));
});

test("the pooled field shows the block of window positions and hides the place within it", () => {
  const config = { variant: "wilterson", schema: "given", noise: 0.5, windowActive: true } as const;
  const rows = (frame: Uint8Array) => {
    const cells = field(frame).flatMap((lit, cell) => (lit === 1 ? [Math.floor(cell / 10)] : []));
    return [Math.min(...cells), Math.max(...cells)];
  };
  const world = createWorld({ ...config, pool: 3 }, 5);
  const seen = [rows(world.frame)];
  for (let step = 0; step < 6; step++) seen.push(rows(stepWorld(world, 0).frame));
  assert.deepEqual(seen, [
    [0, 4],
    [0, 4],
    [0, 4],
    [3, 7],
    [3, 7],
    [3, 7],
    [6, 9],
  ]);
});

test("a block of one is the given field", () => {
  const config = { variant: "wilterson", schema: "given", noise: 0.5, windowActive: true } as const;
  const given = createWorld(config, 5);
  const pooled = createWorld({ ...config, pool: 1 }, 5);
  for (let step = 0; step < 9; step++) {
    assert.deepEqual(stepWorld(pooled, step % 8).frame, stepWorld(given, step % 8).frame);
  }
});

test("telemetry keeps the frame-read estimate inside the block that holds the window", () => {
  const config = {
    variant: "wilterson",
    schema: "read",
    noise: 0.5,
    windowActive: true,
    slip: 0.3,
    pool: 2,
  } as const;
  const block = (value: number) => Math.floor((value - 1) / 2);
  for (let seed = 1; seed <= 50; seed++) {
    const world = createWorld(config, seed);
    for (let step = 0; ; step++) {
      assert.equal(block(world.estimate!.row), block(world.window.row));
      assert.equal(block(world.estimate!.col), block(world.window.col));
      if (world.t === 10) break;
      stepWorld(world, (seed + step) % 8);
    }
  }
});
