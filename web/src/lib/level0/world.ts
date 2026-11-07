import { createRng, nextInt, nextUnit, type Rng } from "../rng.ts";

export type Variant = "wilterson" | "piefke";
export type ActionSpace =
  | { variant: "wilterson"; schema: "given" | "off" | "commands" | "read" }
  | { variant: "piefke"; schema: "given" | "off" | "free" | "clock" | "start" | "parked" };
export type WorldConfig = ActionSpace & {
  noise: number;
  windowActive: boolean;
  slip?: number;
  lockResource?: boolean;
  pool?: number;
};
export type Position = { row: number; col: number };
export type Action = { paddle: 0 | 1; attention: number; resource: number };
export type World = {
  config: WorldConfig;
  rng: { world: Rng; noise: Rng; slip: Rng };
  t: number;
  startCol: number;
  landingCol: number;
  ball: Position;
  window: Position;
  paddle: number;
  resource: Position | null;
  estimate: Position | null;
  frame: Uint8Array;
  reward: number;
  trackingScore: number;
  catchScore: number;
};

// A locked square holds only what its column says; its row cannot count time.
const lockedRow = 4;

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

// Centres are split into blocks of `size` along each axis; the field lights every cell that some
// window of the block would cover, so it hides the place within the block.
function fillPooled(frame: Uint8Array, center: Position, size: number): void {
  const span = (value: number) => {
    const first = 1 + Math.floor((value - 1) / size) * size;
    return [first - 1, Math.min(8, first + size - 1) + 1] as const;
  };
  const [top, bottom] = span(center.row);
  const [left, right] = span(center.col);
  for (let row = top; row <= bottom; row++) {
    frame.fill(1, 100 + row * 10 + left, 100 + row * 10 + right + 1);
  }
}

// The window worked out from the frame and the agent's own commands: of the blocks with at most
// the ball lit, the one nearest the predicted position. On the first frame nothing is predicted,
// so it looks on row 1 for a block with the ball at the middle of its top row.
export function readWindow(frame: Uint8Array, predicted: Position | null): Position {
  let best = predicted ?? { row: 1, col: 4 };
  let bestDistance = Infinity;
  for (let row = 1; row <= (predicted === null ? 1 : 8); row++) {
    for (let col = 1; col <= 8; col++) {
      let pixels = 0;
      for (let y = row - 1; y <= Math.min(8, row + 1); y++) {
        for (let x = col - 1; x <= col + 1; x++) pixels += frame[y * 10 + x]!;
      }
      if (pixels > 1) continue;
      const distance =
        predicted === null
          ? 1 - frame[col]!
          : Math.max(Math.abs(row - predicted.row), Math.abs(col - predicted.col));
      if (distance < bestDistance) {
        best = { row, col };
        bestDistance = distance;
      }
    }
  }
  return best;
}

function createFrame(world: Omit<World, "frame">, action: number | null = null): Uint8Array {
  const frame = new Uint8Array(200);
  frame.set(createNoiseMask(world.rng.noise, world.config.noise));
  frame.fill(0, 90, 100);
  if (world.config.windowActive) fillSquare(frame, world.window, 0, 0);
  frame[world.ball.row * 10 + world.ball.col] = 1;
  frame.fill(1, 90 + world.paddle - 1, 90 + world.paddle + 2);
  if (world.config.schema === "given") fillPooled(frame, world.window, world.config.pool ?? 1);
  if (world.config.schema === "read") {
    const predicted =
      world.estimate === null || action === null
        ? null
        : moveWindow(
            world.estimate, world.config.variant, decodeAction(world.config, action).attention,
          );
    world.estimate = readWindow(frame, predicted);
    fillSquare(frame, world.estimate, 100, 1);
  }
  if (world.resource !== null) fillSquare(frame, world.resource, 100, 1);
  if (world.config.schema === "commands" && action !== null) frame[100 + action] = 1;
  // These fields hold what a square left in place shows, or one part of it.
  const row = Math.min(world.t, 8);
  const col = clampCenter(world.startCol);
  if (world.config.schema === "clock") frame.fill(1, 100 + (row - 1) * 10, 100 + (row + 2) * 10);
  if (world.config.schema === "parked") fillSquare(frame, { row, col }, 100, 1);
  if (world.config.schema === "start") {
    for (let y = 0; y < 10; y++) frame.fill(1, 100 + y * 10 + col - 1, 100 + y * 10 + col + 2);
  }
  return frame;
}

export function createWorld(config: WorldConfig, seed: number): World {
  if (!Number.isFinite(config.noise) || config.noise < 0 || config.noise > 1) {
    throw new RangeError("Уровень шума должен быть от 0 до 1");
  }
  // Stream 0 draws start, then landing, once; stream 1 draws every noise mask.
  const rng = { world: createRng(seed, 0), noise: createRng(seed, 1), slip: createRng(seed, 8) };
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
    resource:
      config.schema === "free"
        ? { ...window, row: config.lockResource === true ? lockedRow : window.row }
        : null,
    estimate: null,
    reward: 0,
    trackingScore: 0,
    catchScore: 0,
  };
  // Drawing the first frame also sets the first estimate.
  const frame = createFrame(state);
  return { ...state, frame };
}

export function stepWorld(world: World, action: number): World {
  if (world.t === 10) throw new RangeError("Эпизод уже завершён");
  const command = decodeAction(world.config, action);
  world.window = moveWindow(world.window, world.config.variant, command.attention);
  // A slip moves the window past the command; nothing in the frame reports it directly.
  if (world.config.slip !== undefined && nextUnit(world.rng.slip) < world.config.slip) {
    world.window = moveWindow(world.window, "wilterson", nextInt(world.rng.slip, 8));
  }
  if (world.resource !== null) {
    world.resource = moveWindow(world.resource, world.config.variant, command.resource);
    if (world.config.lockResource === true) world.resource.row = lockedRow;
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
