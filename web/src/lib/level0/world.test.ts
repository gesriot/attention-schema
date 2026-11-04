import assert from "node:assert/strict";
import test from "node:test";
import { createRng, nextInt, nextUnit } from "../rng.ts";
import {
  createTruthPolicy,
  meanScores,
  runEpisode,
  runTruthEpisode,
  windowReadable,
} from "./evaluate.ts";
import { createRandomPolicy, createTrackerPolicy, type Policy } from "./policies.ts";
import {
  actionCount,
  createNoiseMask,
  createWorld,
  decodeAction,
  encodeAction,
  stepWorld,
  type ActionSpace,
  type World,
  type WorldConfig,
} from "./world.ts";

const wilterson: WorldConfig = {
  variant: "wilterson",
  schema: "given",
  noise: 0.5,
  windowActive: true,
};
const piefke: WorldConfig = { ...wilterson, variant: "piefke", schema: "free" };
const configurations: WorldConfig[] = [
  wilterson,
  { ...wilterson, schema: "off" },
  { ...piefke, schema: "given" },
  { ...piefke, schema: "off" },
  piefke,
];

function blankFrame(pixels: [number, number][]): Uint8Array {
  const frame = new Uint8Array(200);
  for (const [row, col] of pixels) frame[row * 10 + col] = 1;
  return frame;
}

function trackerAction(config: ActionSpace, paddle: 0 | 1, attention: number): number {
  return encodeAction(config, {
    paddle,
    attention,
    resource: config.schema === "free" ? attention : 0,
  });
}

test("the first frame has the window on the ball, the paddle and no reward", () => {
  const starts = new Set<number>();
  for (let seed = 1; seed <= 100; seed++) {
    const expectedRng = createRng(seed, 0);
    const start = nextInt(expectedRng, 10);
    const landing = nextInt(expectedRng, 10);
    for (const config of configurations) {
      const world = createWorld(config, seed);
      assert.equal(world.t, 1);
      assert.equal(world.startCol, start);
      assert.equal(world.landingCol, landing);
      assert.deepEqual(world.rng.world, expectedRng);
      assert.deepEqual(world.ball, { row: 0, col: start });
      assert.deepEqual(world.window, { row: 1, col: Math.max(1, Math.min(8, start)) });
      assert.equal(world.paddle, 4);
      assert.deepEqual([...world.frame.slice(90, 100)], [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]);
      assert.equal(world.frame[start], 1);
      assert.equal(world.reward, 0);
      assert.equal(world.trackingScore, 0);
      assert.equal(world.catchScore, 0);
      assert.deepEqual(world.resource, config.schema === "free" ? world.window : null);
      if (world.resource !== null) assert.notEqual(world.resource, world.window);
    }
    starts.add(start);
  }
  assert.equal(starts.size, 10);
});

test("the ball finishes its path in nine actions on 10000 seeds in both variants", () => {
  for (const config of [wilterson, piefke]) {
    for (let seed = 1; seed <= 10000; seed++) {
      const world = createWorld(config, seed);
      const direction = Math.sign(world.landingCol - world.startCol);
      const distance = Math.abs(world.landingCol - world.startCol);
      for (let t = 2; t <= 10; t++) {
        stepWorld(world, (seed + t) % actionCount(config));
        assert.equal(world.t, t);
        assert.equal(world.ball.row, t - 1);
        assert.equal(world.ball.col, world.startCol + direction * Math.min(t - 1, distance));
      }
      assert.equal(world.ball.col, world.landingCol);
      assert.equal(world.ball.row, 9);
    }
  }
});

test("every joint action has the specified encoding and decodes uniquely", () => {
  for (const config of configurations) {
    const count = config.variant === "wilterson" ? 8 : 3;
    const resources = config.schema === "free" ? 3 : 1;
    assert.equal(actionCount(config), 2 * count * resources);
    const seen = new Set<number>();
    for (const paddle of [0, 1] as const) {
      for (let attention = 0; attention < count; attention++) {
        for (let resource = 0; resource < resources; resource++) {
          const action = { paddle, attention, resource };
          const expected = paddle * count * resources + attention * resources + resource;
          assert.equal(encodeAction(config, action), expected);
          assert.deepEqual(decodeAction(config, expected), action);
          seen.add(expected);
        }
      }
    }
    assert.equal(seen.size, actionCount(config));
    for (const action of [-1, 0.5, actionCount(config), NaN, Infinity]) {
      assert.throws(() => decodeAction(config, action), /Недопустимый номер/);
    }
    assert.throws(
      () => encodeAction(config, { paddle: 0, attention: count, resource: 0 }),
      /Недопустимое действие/,
    );
  }
});

test("window and resource clip per axis and a paddle move past the edge is undone", () => {
  const shifts = {
    wilterson: [
      [0, 1],
      [1, 1],
      [1, 0],
      [1, -1],
      [0, -1],
      [-1, -1],
      [-1, 0],
      [-1, 1],
    ],
    piefke: [
      [0, 1],
      [1, 1],
      [-1, 1],
    ],
  } as const;
  for (const config of [wilterson, piefke]) {
    for (const row of [1, 4, 8]) {
      for (const col of [1, 4, 8]) {
        for (const paddle of [1, 8]) {
          for (let action = 0; action < actionCount(config); action++) {
            const world = createWorld(config, 1);
            world.window = { row, col };
            if (world.resource !== null) world.resource = { row, col };
            world.paddle = paddle;
            const command = decodeAction(config, action);
            const [dcol, drow] = shifts[config.variant][command.attention]!;
            stepWorld(world, action);
            assert.deepEqual(world.window, {
              row: Math.max(1, Math.min(8, row + drow)),
              col: Math.max(1, Math.min(8, col + dcol)),
            });
            if (world.resource !== null) {
              const [dx, dy] = shifts.piefke[command.resource]!;
              assert.deepEqual(world.resource, {
                row: Math.max(1, Math.min(8, row + dy)),
                col: Math.max(1, Math.min(8, col + dx)),
              });
            }
            const target = paddle + (command.paddle === 0 ? -1 : 1);
            assert.equal(world.paddle, target < 1 || target > 8 ? paddle : target);
          }
        }
      }
    }
  }
});

test("the noise mask has the exact pixel count and always spends 99 draws", () => {
  for (const probability of [0, 0.25, 0.5, 0.75, 1, 0.005, 0.505]) {
    for (let seed = 1; seed <= 20; seed++) {
      const rng = createRng(seed, 1);
      const expectedRng = createRng(seed, 1);
      const mask = createNoiseMask(rng, probability);
      const expected = new Uint8Array(100);
      expected.fill(1, 0, Math.round(probability * 100));
      for (let i = 99; i > 0; i--) {
        const j = Math.floor(nextUnit(expectedRng) * (i + 1));
        [expected[i], expected[j]] = [expected[j]!, expected[i]!];
      }
      assert.equal(mask.length, 100);
      assert.equal(mask.reduce((sum, pixel) => sum + pixel, 0), Math.round(probability * 100));
      assert.deepEqual(mask, expected);
      assert.deepEqual(rng, expectedRng);
    }
  }
});

test("a frame clears the window and bottom row and draws ball, paddle and schema", () => {
  for (const settings of configurations) {
    for (const noise of [0, 0.25, 0.5, 0.75, 1]) {
      for (const windowActive of [true, false]) {
        const config = { ...settings, noise, windowActive };
        const world = createWorld(config, 19);
        const noiseRng = createRng(19, 1);
        for (let t = 1; t <= 10; t++) {
          const mask = createNoiseMask(noiseRng, noise);
          assert.equal(world.frame.length, 200);
          for (let row = 0; row < 10; row++) {
            for (let col = 0; col < 10; col++) {
              let expected = row === 9 ? 0 : mask[row * 10 + col]!;
              if (
                windowActive &&
                Math.abs(row - world.window.row) <= 1 &&
                Math.abs(col - world.window.col) <= 1
              ) {
                expected = 0;
              }
              if (row === world.ball.row && col === world.ball.col) expected = 1;
              if (row === 9 && Math.abs(col - world.paddle) <= 1) expected = 1;
              assert.equal(world.frame[row * 10 + col], expected);
              const center = config.schema === "given" ? world.window : world.resource;
              const schema =
                center !== null &&
                Math.abs(row - center.row) <= 1 &&
                Math.abs(col - center.col) <= 1;
              assert.equal(world.frame[100 + row * 10 + col], schema ? 1 : 0);
            }
          }
          assert.deepEqual(world.rng.noise, noiseRng);
          if (t < 10) stepWorld(world, (t * 7) % actionCount(config));
        }
      }
    }
  }
});

test("the ball stays visible outside the window at zero noise", () => {
  const world = createWorld({ ...wilterson, noise: 0 }, 1);
  world.window = { row: 8, col: 1 };
  stepWorld(world, 0);
  assert.ok(Math.abs(world.ball.row - world.window.row) > 1);
  assert.equal(world.frame[world.ball.row * 10 + world.ball.col], 1);
});

test("noise does not depend on commands, schema mode or window state", () => {
  for (let seed = 1; seed <= 30; seed++) {
    const first = createWorld(wilterson, seed);
    const second = createWorld({ ...wilterson, schema: "off", windowActive: false }, seed);
    const third = createWorld(piefke, seed);
    const worlds = [first, second, third];
    for (let t = 1; t <= 10; t++) {
      for (const world of worlds) {
        assert.deepEqual(world.ball, first.ball);
        assert.deepEqual(world.rng.noise, first.rng.noise);
        assert.deepEqual(world.rng.world, first.rng.world);
      }
      for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 10; col++) {
          if (row === first.ball.row && col === first.ball.col) continue;
          if (
            worlds.some(
              (world) =>
                Math.abs(row - world.window.row) <= 1 && Math.abs(col - world.window.col) <= 1,
            )
          ) {
            continue;
          }
          assert.equal(first.frame[row * 10 + col], second.frame[row * 10 + col]);
          assert.equal(first.frame[row * 10 + col], third.frame[row * 10 + col]);
        }
      }
      if (t < 10) {
        stepWorld(first, t % 16);
        stepWorld(second, (t * 3) % 16);
        stepWorld(third, (t * 5) % 18);
      }
    }
  }
});

test("reading a frame spends no noise and the next step keeps the old frame intact", () => {
  const world = createWorld(wilterson, 31);
  const frame = world.frame;
  const copy = frame.slice();
  const state = world.rng.noise.state;
  assert.equal(world.frame, frame);
  assert.equal(world.rng.noise.state, state);
  stepWorld(world, 0);
  assert.notEqual(world.frame, frame);
  assert.deepEqual(frame, copy);
  assert.notEqual(world.rng.noise.state, state);
});

test("tracking scores only on frames 2 to 9 and catching only on frame 10", () => {
  for (const config of [wilterson, piefke]) {
    for (let seed = 1; seed <= 100; seed++) {
      const world = createWorld(config, seed);
      const random = createRandomPolicy(config, seed);
      let tracking = 0;
      let total = 0;
      let outside = 0;
      while (world.t < 10) {
        stepWorld(world, random(world.frame, world.t));
        total += world.reward;
        if (world.t < 10) {
          const inside =
            Math.abs(world.ball.row - world.window.row) <= 1 &&
            Math.abs(world.ball.col - world.window.col) <= 1;
          if (!inside) outside++;
          tracking += inside ? 0.5 : -0.5;
          assert.equal(world.reward, inside ? 0.5 : -0.5);
          assert.equal(world.catchScore, 0);
        } else {
          const caught = Math.abs(world.ball.col - world.paddle) <= 1;
          assert.equal(world.reward, caught ? 2 : -2);
          assert.equal(world.catchScore, world.reward);
        }
        assert.equal(world.trackingScore, tracking);
        assert.ok(Math.abs(world.trackingScore) <= 4);
      }
      assert.equal(world.trackingScore, 4 - outside);
      assert.equal(total, world.trackingScore + world.catchScore);
      const snapshot = structuredClone(world);
      assert.throws(() => stepWorld(world, 0), /Эпизод уже завершён/);
      assert.deepEqual(world, snapshot);
    }
  }
});

test("the truth policy scores exactly 6 on each of 2000 seeds in both variants", () => {
  for (const config of [wilterson, piefke]) {
    for (const windowActive of [true, false]) {
      for (let seed = 1; seed <= 2000; seed++) {
        assert.deepEqual(runTruthEpisode(createWorld({ ...config, windowActive }, seed)), {
          tracking: 4,
          catching: 2,
          total: 6,
        });
      }
    }
  }
});

test("the truth policy takes the first best move, alternates the paddle and moves the resource", () => {
  for (const config of [wilterson, piefke]) {
    const world = createWorld(config, 1);
    world.ball = { row: 4, col: 4 };
    world.window = { row: 4, col: 4 };
    world.landingCol = 4;
    world.paddle = 4;
    const truth = createTruthPolicy();
    assert.deepEqual(decodeAction(config, truth(world)), { paddle: 1, attention: 0, resource: 0 });
    assert.deepEqual(decodeAction(config, truth(world)), { paddle: 0, attention: 0, resource: 0 });
    world.landingCol = 8;
    const command = decodeAction(config, truth(world));
    assert.equal(command.attention, 1);
    assert.equal(command.paddle, 1);
    assert.equal(command.resource, config.schema === "free" ? command.attention : 0);
  }
});

test("the tracker takes the first initial block and falls back to column 4", () => {
  const tracker = createTrackerPolicy(wilterson);
  // Both columns 1 and 2 qualify; column 1 must win the initialization tie.
  const first = blankFrame([[0, 1]]);
  assert.equal(tracker(first, 1), trackerAction(wilterson, 0, 4));
  // The estimated center is still (1, 1); the pixel in column 3 is outside it.
  assert.equal(tracker(blankFrame([[1, 3]]), 2), trackerAction(wilterson, 0, 0));

  const fallback = createTrackerPolicy(wilterson);
  assert.equal(fallback(blankFrame([]), 1), trackerAction(wilterson, 1, 4));
  assert.equal(fallback(blankFrame([]), 2), trackerAction(wilterson, 0, 0));
  assert.equal(fallback(blankFrame([]), 3), trackerAction(wilterson, 0, 0));
});

test("the tracker breaks distance ties by row, then column", () => {
  for (const pixels of [
    [[1, 3], [1, 5]],
    [[1, 3], [2, 4]],
  ] as [number, number][][]) {
    const tracker = createTrackerPolicy(wilterson);
    tracker(blankFrame([]), 1);
    assert.equal(tracker(blankFrame(pixels), 2), trackerAction(wilterson, 0, 7));
  }
});

test("the tracker extends its forecast without observations and never re-reads the window", () => {
  const tracker = createTrackerPolicy(piefke);
  // No initial candidate: estimate (1, 4), then command (0, 1).
  tracker(blankFrame([]), 1);
  assert.equal(tracker(blankFrame([[1, 4]]), 2), trackerAction(piefke, 0, 0));
  assert.equal(tracker(blankFrame([[2, 5]]), 3), trackerAction(piefke, 1, 1));
  // Continue column 6 -> 7 with the remembered shift, ignoring a distant pixel.
  assert.equal(tracker(blankFrame([[0, 0]]), 4), trackerAction(piefke, 1, 1));
});

test("an ordinary policy gets only the frame and step, and the tracker ignores the schema", () => {
  const world = createWorld(wilterson, 1);
  let calls = 0;
  const policy: Policy = (...args) => {
    assert.equal(args.length, 2);
    assert.ok(args[0] instanceof Uint8Array);
    assert.equal(args[1], ++calls);
    return 0;
  };
  runEpisode(world, policy);
  assert.equal(calls, 9);
  const tracker: Policy = createTrackerPolicy(wilterson);
  const altered = createTrackerPolicy(wilterson);
  const episode = createWorld(wilterson, 42);
  while (episode.t < 10) {
    const frame = episode.frame.slice();
    frame.fill(1, 100);
    const action = tracker(episode.frame, episode.t);
    assert.equal(action, altered(frame, episode.t));
    stepWorld(episode, action);
  }
  // Type checking must reject passing evaluator truth through the policy interface.
  const checkSignature = (ordinary: Policy, truth: World) => {
    // @ts-expect-error A world is not a frame.
    ordinary(truth, 1);
    // @ts-expect-error No third argument can carry world truth.
    ordinary(truth.frame, 1, truth);
  };
  void checkSignature;
});

test("readability compares columns and ignores row 9 and the schema", () => {
  const world = createWorld(wilterson, 1);
  world.window = { row: 4, col: 4 };
  const frame = new Uint8Array(200).fill(1);
  assert.equal(windowReadable(frame, world), false);
  assert.equal(windowReadable(new Uint8Array(200), world), false);
  for (let row = 0; row <= 8; row++) frame.fill(0, row * 10 + 3, row * 10 + 6);
  assert.equal(windowReadable(frame, world), true);
  frame.fill(0, 90);
  assert.equal(windowReadable(frame, world), true);
  world.window.row = 8;
  assert.equal(windowReadable(frame, world), true);
  frame.fill(0, 0, 3);
  frame.fill(0, 10, 13);
  frame.fill(0, 20, 23);
  assert.equal(windowReadable(frame, world), false);
});

test("the tracker score matches an independent count of exits on frames 2 to 9", () => {
  for (const config of [wilterson, piefke]) {
    for (const windowActive of [true, false]) {
      let tracking = 0;
      let outside = 0;
      let frames = 0;
      for (let seed = 1; seed <= 2000; seed++) {
        const world = createWorld({ ...config, windowActive }, seed);
        const tracker = createTrackerPolicy(config);
        while (world.t < 10) {
          stepWorld(world, tracker(world.frame, world.t));
          if (world.t === 10) continue;
          frames++;
          if (
            Math.abs(world.ball.row - world.window.row) > 1 ||
            Math.abs(world.ball.col - world.window.col) > 1
          ) {
            outside++;
          }
        }
        tracking += world.trackingScore;
      }
      assert.equal(frames, 2000 * 8);
      assert.equal(tracking, 4 * 2000 - outside);
      assert.ok(Math.abs(tracking / 2000 - 4 * (1 - (2 * outside) / frames)) < 1e-12);
    }
  }
});

test("two runs with one seed give the same frames, commands and rewards", () => {
  for (const config of configurations) {
    for (const makePolicy of [createTrackerPolicy, createRandomPolicy]) {
      const first = createWorld(config, 37);
      const second = createWorld(config, 37);
      const firstPolicy = makePolicy(config, 37);
      const secondPolicy = makePolicy(config, 37);
      for (let t = 1; t <= 10; t++) {
        assert.deepEqual(first, second);
        if (t === 10) break;
        const action = firstPolicy(first.frame, t);
        assert.equal(action, secondPolicy(second.frame, t));
        stepWorld(first, action);
        stepWorld(second, action);
      }
    }
  }
});

test("seed means equal the sum of separate episodes", () => {
  for (const name of ["truth", "tracker", "random"] as const) {
    const totals = { tracking: 0, catching: 0, total: 0 };
    for (let seed = 11; seed <= 13; seed++) {
      const world = createWorld(piefke, seed);
      const score =
        name === "truth"
          ? runTruthEpisode(world)
          : runEpisode(
              world,
              name === "tracker" ? createTrackerPolicy(piefke) : createRandomPolicy(piefke, seed),
            );
      totals.tracking += score.tracking / 3;
      totals.catching += score.catching / 3;
      totals.total += score.total / 3;
    }
    const mean = meanScores(piefke, 11, 13, name);
    for (const key of ["tracking", "catching", "total"] as const) {
      assert.ok(Math.abs(mean[key] - totals[key]) < 1e-12);
    }
  }
  assert.throws(() => meanScores(piefke, 2, 1, "tracker"), /диапазон зёрен/);
});

test("invalid noise and action numbers are rejected without changing state", () => {
  for (const noise of [-0.1, 1.1, NaN, Infinity]) {
    assert.throws(() => createWorld({ ...wilterson, noise }, 1), /Уровень шума/);
  }
  const world = createWorld(wilterson, 1);
  const snapshot = structuredClone(world);
  assert.throws(() => stepWorld(world, 16), /Недопустимый номер/);
  assert.deepEqual(world, snapshot);
});
