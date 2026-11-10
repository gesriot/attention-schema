import { nextInt, nextUnit, type Rng } from "../rng.ts";
import type { Config } from "./config.ts";

export type Position = { row: number; col: number };
export type ObjectState = Position & { velocity: Position; category: number; hue: number };
export type Occluder = Position;
export type Flash = Position & { strength: number };
export type World = {
  t: number;
  target: ObjectState;
  distractor: ObjectState;
  occluder: Occluder;
  flash: Flash | null;
  jumpStep: number;
  frame: Float64Array;
};

// Row-major order, omitting rest, fixes all direction draws.
const directions: readonly Position[] = [
  { row: -1, col: -1 },
  { row: -1, col: 0 },
  { row: -1, col: 1 },
  { row: 0, col: -1 },
  { row: 0, col: 1 },
  { row: 1, col: -1 },
  { row: 1, col: 0 },
  { row: 1, col: 1 },
];

function drawHue(config: Config, category: number, rng: Rng): number {
  return (category + 0.5) / config.hueCategories + (2 * nextUnit(rng) - 1) * config.hueJitter;
}

function createObject(config: Config, position: Position, rng: Rng): ObjectState {
  const velocity = { ...directions[nextInt(rng, directions.length)]! };
  const category = nextInt(rng, config.hueCategories);
  return { ...position, velocity, category, hue: drawHue(config, category, rng) };
}

function drawFlash(config: Config, rng: Rng): Flash | null {
  // One gate every frame, then row, column, strength only on a flash.
  if (nextUnit(rng) >= config.pFlash) return null;
  const row = nextInt(rng, config.fieldSize);
  const col = nextInt(rng, config.fieldSize);
  const strength = config.flashMin + (config.flashMax - config.flashMin) * nextUnit(rng);
  return { row, col, strength };
}

export function isOccluded(config: Config, occluder: Occluder, position: Position): boolean {
  return (
    position.row >= occluder.row &&
    position.row < occluder.row + config.occluderRows &&
    position.col >= occluder.col &&
    position.col < occluder.col + config.occluderCols
  );
}

export function renderWorld(config: Config, world: World): Float64Array {
  const size = config.fieldSize;
  const cells = size * size;
  const frame = new Float64Array(2 * cells);
  for (let row = world.occluder.row; row < world.occluder.row + config.occluderRows; row++) {
    for (let col = world.occluder.col; col < world.occluder.col + config.occluderCols; col++) {
      frame[row * size + col] = config.occluderBrightness;
    }
  }
  for (const object of [world.distractor, world.target]) {
    if (isOccluded(config, world.occluder, object)) continue;
    const cell = object.row * size + object.col;
    frame[cell] = config.objectBrightness;
    frame[cells + cell] = object.hue;
  }
  if (world.flash !== null) {
    const cell = world.flash.row * size + world.flash.col;
    frame[cell] = frame[cell]! + world.flash.strength;
  }
  return frame;
}

export function createWorld(config: Config, rng: Rng, flashRng: Rng): World {
  // Initialization: target row, col, direction, category, jitter; distractor admissible
  // cell (row-major), direction, category, jitter; occluder row, col; jump step.
  const target = createObject(
    config,
    {
      row: nextInt(rng, config.fieldSize),
      col: nextInt(rng, config.fieldSize),
    },
    rng,
  );
  const candidates: Position[] = [];
  for (let row = 0; row < config.fieldSize; row++) {
    for (let col = 0; col < config.fieldSize; col++) {
      if (Math.max(Math.abs(row - target.row), Math.abs(col - target.col)) >= config.minGap) {
        candidates.push({ row, col });
      }
    }
  }
  const distractor = createObject(config, candidates[nextInt(rng, candidates.length)]!, rng);
  const occluder = {
    row: nextInt(rng, config.fieldSize - config.occluderRows + 1),
    col: nextInt(rng, config.fieldSize - config.occluderCols + 1),
  };
  const jumpStep = config.hueJumpMin + nextInt(rng, config.hueJumpMax - config.hueJumpMin + 1);
  const world: World = {
    t: 1,
    target,
    distractor,
    occluder,
    jumpStep,
    flash: drawFlash(config, flashRng),
    frame: new Float64Array(0),
  };
  world.frame = renderWorld(config, world);
  return world;
}

export function moveObject(config: Config, object: ObjectState, rng: Rng): void {
  // Move gate, then turn gate only when moving, then a direction only on a turn.
  // A turn may select the existing direction. Reflection happens before displacement.
  if (nextUnit(rng) >= config.pMove) return;
  if (nextUnit(rng) < config.pTurn) {
    object.velocity = { ...directions[nextInt(rng, directions.length)]! };
  }
  for (const axis of ["row", "col"] as const) {
    const next = object[axis] + object.velocity[axis];
    if (next < 0 || next >= config.fieldSize) object.velocity[axis] *= -1;
    object[axis] += object.velocity[axis];
  }
}

export function advanceWorld(config: Config, world: World, rng: Rng, flashRng: Rng): void {
  if (world.t >= config.episodeSteps) throw new Error("Эпизод уже завершён");
  world.t++;
  // Each transition: target motion, distractor motion, optional category/jitter jump.
  moveObject(config, world.target, rng);
  moveObject(config, world.distractor, rng);
  if (world.t === world.jumpStep) {
    const draw = nextInt(rng, config.hueCategories - 1);
    world.target.category = draw >= world.target.category ? draw + 1 : draw;
    world.target.hue = drawHue(config, world.target.category, rng);
  }
  world.flash = drawFlash(config, flashRng);
  world.frame = renderWorld(config, world);
}
