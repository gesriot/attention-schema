import {
  chooseAttention,
  createRandomPolicy,
  createTrackerPolicy,
  type Policy,
} from "./policies.ts";
import { createWorld, encodeAction, stepWorld, type World, type WorldConfig } from "./world.ts";

export type Scores = { tracking: number; catching: number; total: number };
export type PolicyName = "truth" | "tracker" | "random";

// This evaluator-only policy deliberately reads truth; ordinary policies cannot.
export function createTruthPolicy(): (world: World) => number {
  let nextTie: 0 | 1 = 1;
  return (world) => {
    const target = {
      row: world.ball.row + 1,
      col: world.ball.col + Math.sign(world.landingCol - world.ball.col),
    };
    const attention = chooseAttention(world.config.variant, world.window, target);
    let paddle: 0 | 1;
    if (world.paddle === world.landingCol) {
      paddle = nextTie;
      nextTie = nextTie === 1 ? 0 : 1;
    } else {
      paddle = world.paddle < world.landingCol ? 1 : 0;
    }
    return encodeAction(world.config, {
      paddle,
      attention,
      resource: world.config.schema === "free" ? attention : 0,
    });
  };
}

export function runEpisode(world: World, policy: Policy): Scores {
  while (world.t < 10) stepWorld(world, policy(world.frame, world.t));
  return {
    tracking: world.trackingScore,
    catching: world.catchScore,
    total: world.trackingScore + world.catchScore,
  };
}

export function runTruthEpisode(world: World): Scores {
  const truth = createTruthPolicy();
  return runEpisode(world, () => truth(world));
}

export function meanScores(
  config: WorldConfig,
  firstSeed: number,
  lastSeed: number,
  policy: PolicyName,
): Scores {
  if (!Number.isInteger(firstSeed) || !Number.isInteger(lastSeed) || firstSeed > lastSeed) {
    throw new RangeError("Нужен непустой целочисленный диапазон зёрен");
  }
  const sum: Scores = { tracking: 0, catching: 0, total: 0 };
  for (let seed = firstSeed; seed <= lastSeed; seed++) {
    const world = createWorld(config, seed);
    const scores =
      policy === "truth"
        ? runTruthEpisode(world)
        : runEpisode(
            world,
            policy === "tracker" ? createTrackerPolicy(config) : createRandomPolicy(config, seed),
          );
    sum.tracking += scores.tracking;
    sum.catching += scores.catching;
    sum.total += scores.total;
  }
  const count = lastSeed - firstSeed + 1;
  return {
    tracking: sum.tracking / count,
    catching: sum.catching / count,
    total: sum.total / count,
  };
}

export function windowReadable(frame: Uint8Array, world: World): boolean {
  let found = false;
  for (let row = 1; row <= 8; row++) {
    for (let col = 1; col <= 8; col++) {
      let pixels = 0;
      for (let y = row - 1; y <= Math.min(8, row + 1); y++) {
        for (let x = col - 1; x <= col + 1; x++) pixels += frame[y * 10 + x]!;
      }
      if (pixels > 1) continue;
      if (col !== world.window.col) return false;
      found = true;
    }
  }
  return found;
}
