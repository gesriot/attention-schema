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

export function telemetryWidth(config: Config): number {
  return (config.fieldSize / config.k) ** 2 + (config.telemetryPeriod > 1 ? 1 : 0);
}

export function telemetry(
  config: Config,
  resource: Float64Array,
  step: number,
  rng: Rng,
): Float64Array {
  const blocks = config.fieldSize / config.k;
  const values = new Float64Array(telemetryWidth(config));
  // Row-major blocks, row-major cells within each block, then one normal per block.
  // Draw on every step before masking, so the period never changes stream consumption.
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
  if (config.telemetryPeriod > 1) {
    if ((step - 1) % config.telemetryPeriod === 0) values[values.length - 1] = 1;
    else values.fill(0);
  }
  return values;
}
