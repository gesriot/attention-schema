import { createRng, nextNormal, type Rng } from "../rng.ts";
import { streams, type Config } from "./config.ts";
import { localNoise } from "./perception.ts";

export const falseAlarmFrames = 2000;
export type FalseAlarms = { frames: number; rate: number; percentile95: number };

export function noiseMaxima(
  config: Pick<Config, "fieldSize" | "sigmaFloor">,
  frames: number,
  rng: Rng,
): Float64Array {
  if (!Number.isSafeInteger(frames) || frames < 1)
    throw new Error("Нужно положительное число кадров");
  const cells = config.fieldSize * config.fieldSize;
  const measurement = new Float64Array(2 * cells);
  const maxima = new Float64Array(frames);
  for (let frame = 0; frame < frames; frame++) {
    // Frame order, brightness then hue, row-major cells: exactly 2 * cells normals.
    // Unit noise is independent of sigmaEta and no brightness filter is applied.
    for (let index = 0; index < measurement.length; index++) measurement[index] = nextNormal(rng);
    const noise = localNoise(measurement, config.fieldSize);
    let maximum = -Infinity;
    for (let cell = 0; cell < cells; cell++) {
      maximum = Math.max(maximum, measurement[cell]! / Math.max(noise[cell]!, config.sigmaFloor));
    }
    maxima[frame] = maximum;
  }
  return maxima;
}

export function summarizeFalseAlarms(maxima: Float64Array, threshold: number): FalseAlarms {
  if (maxima.length === 0) throw new Error("Нужны максимумы SNR хотя бы одного кадра");
  let alarms = 0;
  for (const maximum of maxima) if (maximum >= threshold) alarms++;
  const ordered = maxima.slice().sort();
  return {
    frames: maxima.length,
    rate: alarms / maxima.length,
    percentile95: ordered[Math.floor(0.95 * maxima.length)]!,
  };
}

export function checkFalseAlarms(config: Config): FalseAlarms {
  const maxima = noiseMaxima(config, falseAlarmFrames, createRng(1, streams.falseAlarm));
  return summarizeFalseAlarms(maxima, config.detectThreshold);
}
