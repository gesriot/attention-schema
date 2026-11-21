import type { Config } from "./config.ts";
import { clamp } from "./attention.ts";
import type { Action, Move } from "./episode.ts";
import { hueCategory } from "./policies.ts";
import type { Position } from "./world.ts";

export type HandC6Input = {
  perception: Float64Array;
  paddle: number;
  resource: Float64Array;
};

export function resourceCenter(
  config: Pick<Config, "fieldSize">,
  resource: Float64Array,
): Position {
  const size = config.fieldSize;
  if (resource.length !== size * size) throw new RangeError("Неверная ширина карты ресурса C6");
  let total = 0;
  let rowSum = 0;
  let colSum = 0;
  // Ascending row-major cells; add total, row moment, then column moment per cell.
  // Use the resource values as received, without renormalization or thresholding.
  for (let cell = 0; cell < resource.length; cell++) {
    const value = resource[cell]!;
    total += value;
    rowSum += Math.floor(cell / size) * value;
    colSum += (cell % size) * value;
  }
  return { row: rowSum / total, col: colSum / total };
}

export function handC6Action(config: Config, input: HandC6Input, final: boolean): Action {
  const q = input.perception;
  if (q.length !== 18) throw new RangeError("Неверная ширина сводки восприятия C6");
  const center = resourceCenter(config, input.resource);
  const row = q[1]! * (config.fieldSize - 1);
  const col = q[2]! * (config.fieldSize - 1);
  // Inactive slots already contain zeros in pool(z); no alternate action is substituted.
  // Math.round resolves half-integer ties toward +Infinity, then clamp each command.
  return {
    command: {
      row: clamp(Math.round(row + q[3]! - center.row), -1, 1) as Move,
      col: clamp(Math.round(col + q[4]! - center.col), -1, 1) as Move,
    },
    effort: 0,
    paddleMove: Math.sign(Math.round(col) - input.paddle) as Move,
    ...(final ? { answer: hueCategory(config, q[5]!) } : {}),
  };
}
