import { createRng, nextInt, type Rng } from "../rng.ts";

export type Variant = "wilterson" | "piefke";
export type ActionSpace =
  | { variant: "wilterson"; schema: "given" | "off" | "commands" }
  | { variant: "piefke"; schema: "given" | "off" | "free" };
export type WorldConfig = ActionSpace & { noise: number; windowActive: boolean };
export type Position = { row: number; col: number };
export type Action = { paddle: 0 | 1; attention: number; resource: number };
export type World = {
  config: WorldConfig;
  rng: { world: Rng; noise: Rng };
  t: number;
  startCol: number;
  landingCol: number;
  ball: Position;
  window: Position;
  paddle: number;
  resource: Position | null;
  frame: Uint8Array;
  reward: number;
  trackingScore: number;
  catchScore: number;
};

// Each pair is (dcol, drow); order also defines the tie-break for policies.
const moves = {
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

export function attentionMoves(variant: Variant): readonly (readonly [number, number])[] {
  return moves[variant];
}

export function actionCount(config: ActionSpace): number {
  return config.schema === "free" ? 18 : 2 * moves[config.variant].length;
}

export function encodeAction(config: ActionSpace, action: Action): number {
  const count = moves[config.variant].length;
  if (
    (action.paddle !== 0 && action.paddle !== 1) ||
    !Number.isInteger(action.attention) ||
    action.attention < 0 ||
    action.attention >= count ||
    !Number.isInteger(action.resource) ||
    action.resource < 0 ||
    action.resource >= (config.schema === "free" ? 3 : 1)
  ) {
    throw new RangeError("Недопустимое действие");
  }
  return config.schema === "free"
    ? action.paddle * 9 + action.attention * 3 + action.resource
    : action.paddle * count + action.attention;
}

export function decodeAction(config: ActionSpace, action: number): Action {
  if (!Number.isInteger(action) || action < 0 || action >= actionCount(config)) {
    throw new RangeError("Недопустимый номер действия");
  }
  const count = config.schema === "free" ? 9 : moves[config.variant].length;
  return {
    paddle: Math.floor(action / count) as 0 | 1,
    attention: config.schema === "free" ? Math.floor((action % 9) / 3) : action % count,
    resource: config.schema === "free" ? action % 3 : 0,
  };
}

export function clampCenter(value: number): number {
  return Math.max(1, Math.min(8, value));
}

export function moveWindow(position: Position, variant: Variant, action: number): Position {
  const [dcol, drow] = moves[variant][action]!;
  return { row: clampCenter(position.row + drow), col: clampCenter(position.col + dcol) };
}

export function movePaddle(paddle: number, direction: 0 | 1): number {
  const next = paddle + (direction === 0 ? -1 : 1);
  return next < 1 || next > 8 ? paddle : next;
}

export function inWindow(position: Position, center: Position): boolean {
  return Math.abs(position.row - center.row) <= 1 && Math.abs(position.col - center.col) <= 1;
}

export function createNoiseMask(rng: Rng, probability: number): Uint8Array {
  const mask = new Uint8Array(100);
  mask.fill(1, 0, Math.round(probability * 100));
  // Exactly 99 draws, including at p = 0 and p = 1.
  for (let i = 99; i > 0; i--) {
    const j = nextInt(rng, i + 1);
    const value = mask[i]!;
    mask[i] = mask[j]!;
    mask[j] = value;
  }
  return mask;
}

function fillSquare(frame: Uint8Array, center: Position, offset: number, value: number): void {
  for (let row = center.row - 1; row <= center.row + 1; row++) {
    for (let col = center.col - 1; col <= center.col + 1; col++) {
      frame[offset + row * 10 + col] = value;
    }
  }
}

function createFrame(world: Omit<World, "frame">, action: number | null = null): Uint8Array {
  const frame = new Uint8Array(200);
  frame.set(createNoiseMask(world.rng.noise, world.config.noise));
  frame.fill(0, 90, 100);
  if (world.config.windowActive) fillSquare(frame, world.window, 0, 0);
  frame[world.ball.row * 10 + world.ball.col] = 1;
  frame.fill(1, 90 + world.paddle - 1, 90 + world.paddle + 2);
  if (world.config.schema === "given") fillSquare(frame, world.window, 100, 1);
  if (world.resource !== null) fillSquare(frame, world.resource, 100, 1);
  if (world.config.schema === "commands" && action !== null) frame[100 + action] = 1;
  return frame;
}

export function createWorld(config: WorldConfig, seed: number): World {
  if (!Number.isFinite(config.noise) || config.noise < 0 || config.noise > 1) {
    throw new RangeError("Уровень шума должен быть от 0 до 1");
  }
  // Stream 0 draws start, then landing, once; stream 1 draws every noise mask.
  const rng = { world: createRng(seed, 0), noise: createRng(seed, 1) };
  const startCol = nextInt(rng.world, 10);
  const landingCol = nextInt(rng.world, 10);
  const window = { row: 1, col: clampCenter(startCol) };
  const state: Omit<World, "frame"> = {
    config: { ...config },
    rng,
    t: 1,
    startCol,
    landingCol,
    ball: { row: 0, col: startCol },
    window,
    paddle: 4,
    resource: config.schema === "free" ? { ...window } : null,
    reward: 0,
    trackingScore: 0,
    catchScore: 0,
  };
  return { ...state, frame: createFrame(state) };
}

export function stepWorld(world: World, action: number): World {
  if (world.t === 10) throw new RangeError("Эпизод уже завершён");
  const command = decodeAction(world.config, action);
  world.window = moveWindow(world.window, world.config.variant, command.attention);
  if (world.resource !== null) {
    world.resource = moveWindow(world.resource, world.config.variant, command.resource);
  }
  world.ball.col += Math.sign(world.landingCol - world.ball.col);
  world.paddle = movePaddle(world.paddle, command.paddle);
  world.ball.row++;
  world.t++;
  world.frame = createFrame(world, action);
  if (world.t === 10) {
    world.reward = Math.abs(world.ball.col - world.paddle) <= 1 ? 2 : -2;
    world.catchScore += world.reward;
  } else {
    world.reward = inWindow(world.ball, world.window) ? 0.5 : -0.5;
    world.trackingScore += world.reward;
  }
  return world;
}
