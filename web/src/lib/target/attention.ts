import { nextNormal, type Rng } from "../rng.ts";
import type { Config } from "./config.ts";
import type { Flash, Position } from "./world.ts";

export type Attention = {
  priority: Float64Array;
  resource: Float64Array;
  center: Position;
  total: number;
  entropy: number;
};

export function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function commandedCenter(config: Config, center: Position, command: Position): Position {
  return {
    row: clamp(center.row + command.row, 0, config.fieldSize - 1),
    col: clamp(center.col + command.col, 0, config.fieldSize - 1),
  };
}

export function distance(first: Position, second: Position): number {
  const row = first.row - second.row;
  const col = first.col - second.col;
  return Math.sqrt(row * row + col * col);
}

export function bump(config: Config, center: Position): Float64Array {
  const values = new Float64Array(config.fieldSize * config.fieldSize);
  for (let row = 0; row < config.fieldSize; row++) {
    for (let col = 0; col < config.fieldSize; col++) {
      const dr = row - center.row;
      const dc = col - center.col;
      values[row * config.fieldSize + col] = Math.exp(
        -(dr * dr + dc * dc) / (2 * config.sigmaB * config.sigmaB),
      );
    }
  }
  return values;
}

export function distribute(config: Config, priority: Float64Array, effort: number): Attention {
  const tau = config.tau0 / (1 + effort);
  let maximum = -Infinity;
  for (const value of priority) maximum = Math.max(maximum, value);
  const resource = new Float64Array(priority.length);
  let denominator = 0;
  for (let i = 0; i < priority.length; i++) {
    resource[i] = Math.exp((priority[i]! - maximum) / tau);
    denominator += resource[i]!;
  }
  let total = 0;
  let rowSum = 0;
  let colSum = 0;
  for (let i = 0; i < resource.length; i++) {
    const value = Math.min(1, (config.R * resource[i]!) / denominator);
    resource[i] = value;
    total += value;
    rowSum += Math.floor(i / config.fieldSize) * value;
    colSum += (i % config.fieldSize) * value;
  }
  let entropy = 0;
  for (const value of resource) {
    const probability = value / total;
    if (probability > 0) entropy -= probability * Math.log(probability);
  }
  // Capping spends less than R; it is not followed by renormalization.
  return {
    priority,
    resource,
    total,
    center: { row: rowSum / total, col: colSum / total },
    entropy,
  };
}

export function initialAttention(config: Config, target: Position): Attention {
  const priority = bump(config, target);
  for (let i = 0; i < priority.length; i++) {
    priority[i] = (config.betaEndo * priority[i]!) / (1 - config.rho);
  }
  return distribute(config, priority, 0);
}

export function advanceAttention(
  config: Config,
  previous: Attention,
  command: Position,
  effort: number,
  flash: Flash | null,
  rng: Rng,
): Attention {
  const endo = bump(config, commandedCenter(config, previous.center, command));
  const priority = new Float64Array(endo.length);
  const flashCell = flash === null ? -1 : flash.row * config.fieldSize + flash.col;
  // Exactly one normal per cell in row-major order, including at zero noise.
  // Flash is the only world input after initialization; objects never enter this function.
  for (let i = 0; i < priority.length; i++) {
    const salience = i === flashCell ? flash!.strength : 0;
    priority[i] =
      config.rho * previous.priority[i]! +
      config.betaEndo * endo[i]! +
      config.betaExo * salience +
      config.sigmaEpsilon * nextNormal(rng);
  }
  return distribute(config, priority, effort);
}

export function resourceShare(config: Config, attention: Attention, position: Position): number {
  let sum = 0;
  for (
    let row = Math.max(0, position.row - config.resourceRadius);
    row <= Math.min(config.fieldSize - 1, position.row + config.resourceRadius);
    row++
  ) {
    for (
      let col = Math.max(0, position.col - config.resourceRadius);
      col <= Math.min(config.fieldSize - 1, position.col + config.resourceRadius);
      col++
    ) {
      sum += attention.resource[row * config.fieldSize + col]!;
    }
  }
  return sum / attention.total;
}
