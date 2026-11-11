import type { Config } from "./config.ts";

export type PerceptionConfig = Pick<
  Config,
  | "fieldSize"
  | "sigmaFloor"
  | "gateRadius"
  | "detectThreshold"
  | "velocityGain"
  | "hueForget"
  | "maxMissed"
  | "minBrightness"
>;
export type Slot = {
  active: boolean;
  row: number;
  col: number;
  velocityRow: number;
  velocityCol: number;
  hue: number;
  evidence: number;
  predictionSupport: number;
  uncertainty: number;
  hueWeight: number;
  missed: number;
};
export type Perception = { slots: [Slot, Slot] };

function inactiveSlot(): Slot {
  return {
    active: false,
    row: 0,
    col: 0,
    velocityRow: 0,
    velocityCol: 0,
    hue: 0,
    evidence: 0,
    predictionSupport: 0,
    uncertainty: 0,
    hueWeight: 0,
    missed: 0,
  };
}

export function createPerception(): Perception {
  return { slots: [inactiveSlot(), inactiveSlot()] };
}

// Phi^-1(0.625), the lower quartile of |N(0, 1)|.
export const standardNormalAbsLowerQuartile = 0.3186393639643752;

export function localNoise(measurement: Float64Array, size: number): Float64Array {
  const noise = new Float64Array(size * size);
  const neighbors: number[] = [];
  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      neighbors.length = 0;
      for (let r = Math.max(0, row - 1); r <= Math.min(size - 1, row + 1); r++) {
        for (let c = Math.max(0, col - 1); c <= Math.min(size - 1, col + 1); c++) {
          if (r !== row || c !== col) {
            neighbors.push(Math.abs(measurement[r * size + c]!));
            neighbors.push(Math.abs(measurement[size * size + r * size + c]!));
          }
        }
      }
      neighbors.sort((a, b) => a - b);
      const position = 0.25 * (neighbors.length - 1);
      // Average the two order statistics; do not interpolate by the fractional rank.
      const quartile = (neighbors[Math.floor(position)]! + neighbors[Math.ceil(position)]!) / 2;
      noise[row * size + col] = quartile / standardNormalAbsLowerQuartile;
    }
  }
  return noise;
}

function globalCandidate(
  config: PerceptionConfig,
  measurement: Float64Array,
  snr: Float64Array,
): number {
  let best = -1;
  for (let cell = 0; cell < snr.length; cell++) {
    if (measurement[cell]! >= config.minBrightness && (best === -1 || snr[cell]! > snr[best]!)) {
      best = cell;
    }
  }
  return best;
}

function activate(
  config: PerceptionConfig,
  cell: number,
  measurement: Float64Array,
  noise: Float64Array,
  snr: Float64Array,
): Slot {
  const sigma = Math.max(noise[cell]!, config.sigmaFloor);
  return {
    active: true,
    row: Math.floor(cell / config.fieldSize),
    col: cell % config.fieldSize,
    velocityRow: 0,
    velocityCol: 0,
    hue: measurement[config.fieldSize * config.fieldSize + cell]!,
    evidence: snr[cell]!,
    // Initialization has no preceding prediction to support.
    predictionSupport: 0,
    uncertainty: noise[cell]!,
    hueWeight: 1 / (sigma * sigma),
    missed: 0,
  };
}

export function updatePerception(
  config: PerceptionConfig,
  previous: Perception,
  measurement: Float64Array,
): Perception {
  const size = config.fieldSize;
  const cells = size * size;
  if (measurement.length !== 2 * cells) throw new Error("Неверный размер измерения");
  const noise = localNoise(measurement, size);
  const snr = new Float64Array(cells);
  for (let cell = 0; cell < cells; cell++) {
    snr[cell] = measurement[cell]! / Math.max(noise[cell]!, config.sigmaFloor);
  }
  const slots: [Slot, Slot] = [{ ...previous.slots[0] }, { ...previous.slots[1] }];
  if (!slots[0].active) {
    const best = globalCandidate(config, measurement, snr);
    return {
      slots: [
        best === -1 ? inactiveSlot() : activate(config, best, measurement, noise, snr),
        inactiveSlot(),
      ],
    };
  }
  for (let index = 0; index < slots.length; index++) {
    const slot = slots[index]!;
    const occupied = index === 0 ? -1 : slots[0].row * size + slots[0].col;
    // Test the miss count at the beginning of the step. A successful new search
    // replaces all slot-zero state and skips its ordinary gated update.
    if (index === 0 && slot.missed >= config.maxMissed) {
      const best = globalCandidate(config, measurement, snr);
      if (best !== -1 && snr[best]! >= config.detectThreshold) {
        slots[0] = activate(config, best, measurement, noise, snr);
        continue;
      }
    }
    if (!slot.active) {
      let best = -1;
      for (let row = 0; row < size; row++) {
        for (let col = 0; col < size; col++) {
          if (
            Math.max(Math.abs(row - slots[0].row), Math.abs(col - slots[0].col)) <=
            config.gateRadius
          )
            continue;
          const cell = row * size + col;
          if (
            measurement[cell]! >= config.minBrightness &&
            (best === -1 || snr[cell]! > snr[best]!)
          )
            best = cell;
        }
      }
      if (best !== -1 && snr[best]! >= config.detectThreshold) {
        slots[index] = activate(config, best, measurement, noise, snr);
      }
      continue;
    }
    // Math.round resolves half-integer ties toward +Infinity, as in the world controls.
    const predictedRow = Math.max(0, Math.min(size - 1, Math.round(slot.row + slot.velocityRow)));
    const predictedCol = Math.max(0, Math.min(size - 1, Math.round(slot.col + slot.velocityCol)));
    let best = -1;
    for (
      let row = Math.max(0, predictedRow - config.gateRadius);
      row <= Math.min(size - 1, predictedRow + config.gateRadius);
      row++
    ) {
      for (
        let col = Math.max(0, predictedCol - config.gateRadius);
        col <= Math.min(size - 1, predictedCol + config.gateRadius);
        col++
      ) {
        const cell = row * size + col;
        if (
          cell !== occupied &&
          measurement[cell]! >= config.minBrightness &&
          (best === -1 || snr[cell]! > snr[best]!)
        )
          best = cell;
      }
    }
    if (best !== -1 && snr[best]! >= config.detectThreshold) {
      const row = Math.floor(best / size);
      const col = best % size;
      slot.velocityRow =
        (1 - config.velocityGain) * slot.velocityRow + config.velocityGain * (row - slot.row);
      slot.velocityCol =
        (1 - config.velocityGain) * slot.velocityCol + config.velocityGain * (col - slot.col);
      slot.row = row;
      slot.col = col;
      slot.evidence = snr[best]!;
      slot.predictionSupport =
        1 / (1 + Math.max(Math.abs(row - predictedRow), Math.abs(col - predictedCol)));
      slot.hueWeight *= config.hueForget;
      const sigma = Math.max(noise[best]!, config.sigmaFloor);
      const weight = 1 / (sigma * sigma);
      slot.hue =
        (slot.hueWeight * slot.hue + weight * measurement[cells + best]!) /
        (slot.hueWeight + weight);
      slot.hueWeight += weight;
      slot.missed = 0;
    } else {
      slot.row = predictedRow;
      slot.col = predictedCol;
      slot.evidence = 0;
      slot.predictionSupport = 0;
      slot.missed++;
    }
    slot.uncertainty = noise[slot.row * size + slot.col]!;
    if (index === 1 && slot.missed >= config.maxMissed) slots[1] = inactiveSlot();
  }
  return { slots };
}

export function poolPerception(config: Pick<Config, "fieldSize">, state: Perception): Float64Array {
  const values = new Float64Array(18);
  for (let index = 0; index < state.slots.length; index++) {
    const slot = state.slots[index]!;
    if (!slot.active) continue;
    values.set(
      [
        1,
        slot.row / (config.fieldSize - 1),
        slot.col / (config.fieldSize - 1),
        slot.velocityRow,
        slot.velocityCol,
        slot.hue,
        slot.evidence / 10,
        slot.predictionSupport,
        slot.uncertainty,
      ],
      index * 9,
    );
  }
  return values;
}
