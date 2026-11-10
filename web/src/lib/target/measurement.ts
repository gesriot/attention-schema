import { nextNormal, type Rng } from "../rng.ts";
import type { Config } from "./config.ts";

export function measurementSigma(config: Config, resource: number, enhanced: boolean): number {
  return config.sigmaEta * (1 - (enhanced ? config.alpha * resource : 0));
}

export function measure(
  config: Config,
  frame: Float64Array,
  resource: Float64Array,
  enhanced: boolean,
  rng: Rng,
): Float64Array {
  const measurement = new Float64Array(frame.length);
  // Brightness channel first, then hue; row-major cells. Always draw even at sigma = 0.
  for (let i = 0; i < frame.length; i++) {
    const sigma = measurementSigma(config, resource[i % resource.length]!, enhanced);
    measurement[i] = frame[i]! + sigma * nextNormal(rng);
  }
  return measurement;
}

export function telemetry(config: Config, resource: Float64Array, rng: Rng): Float64Array {
  const blocks = config.fieldSize / config.k;
  const values = new Float64Array(blocks * blocks);
  // Row-major blocks, row-major cells within each block, then one normal per block.
  for (let br = 0; br < blocks; br++) {
    for (let bc = 0; bc < blocks; bc++) {
      let sum = 0;
      for (let row = br * config.k; row < (br + 1) * config.k; row++) {
        for (let col = bc * config.k; col < (bc + 1) * config.k; col++) {
          sum += resource[row * config.fieldSize + col]!;
        }
      }
      values[br * blocks + bc] = sum + config.sigmaNu * nextNormal(rng);
    }
  }
  return values;
}
