import {
  publishedTraining,
  trainDqn,
  type Check,
  type CurvePoint,
  type DqnEnvironment,
  type TrainingConfig,
  type TrainingEvent,
} from "./dqn.ts";
import {
  publishedProbes,
  runProbes,
  type ProbeConfig,
  type ProbeResult,
  type ProbeSource,
} from "./probes.ts";

export const noiseGrid = [0, 0.25, 0.5, 0.75, 1];
export const emergenceConditions = [
  "free",
  "given",
  "off",
  "clock",
  "start",
  "parked",
  "locked",
] as const;
export type EmergenceCondition = (typeof emergenceConditions)[number];
export const resourceConditions = ["free", "locked"] as const;
export type EmergenceSettings = { training: TrainingConfig; probes: ProbeConfig };
export const publishedEmergence: EmergenceSettings = {
  training: { ...publishedTraining, sizes: [200, 200, 200, 200, 18] },
  probes: publishedProbes,
};
export type EmergenceJob = {
  condition: EmergenceCondition;
  noise: number;
  subject: number;
  iterations: number;
  evalEvery: number;
  config: TrainingConfig;
  environment: DqnEnvironment;
  probes: ProbeConfig;
};
export type EmergenceResult = {
  condition: EmergenceCondition;
  noise: number;
  subject: number;
  points: CurvePoint[];
  probes: ProbeResult[];
};

export function effectiveSubject(subject: number, noise: number): number {
  const index = noiseGrid.indexOf(noise);
  if (index < 0) throw new Error("Шум должен быть из сетки 0,0.25,0.5,0.75,1");
  if (!Number.isInteger(subject) || subject < 1 || subject > 100) {
    throw new Error("Для независимых уровней шума номер субъекта должен быть от 1 до 100");
  }
  return subject + 100 * index;
}

function hasResource(condition: EmergenceCondition): boolean {
  return condition === "free" || condition === "locked";
}

export function emergenceChecks(condition: EmergenceCondition): Check[] {
  if (hasResource(condition)) {
    return ["normal", "resourceRandom", "resourceStill", "fieldOff", "windowOff"];
  }
  if (condition === "off") return ["normal", "windowOff"];
  return ["normal", "fieldOff", "windowOff"];
}

export function makeEmergenceJob(
  condition: EmergenceCondition,
  noise: number,
  subject: number,
  iterations: number,
  evalEvery: number,
  settings: EmergenceSettings = publishedEmergence,
): EmergenceJob {
  effectiveSubject(subject, noise);
  return {
    condition,
    noise,
    subject,
    iterations,
    evalEvery,
    config: {
      ...settings.training,
      sizes: [...settings.training.sizes.slice(0, -1), hasResource(condition) ? 18 : 6],
    },
    environment: {
      world:
        condition === "locked"
          ? { variant: "piefke", schema: "free", lockResource: true, noise, windowActive: true }
          : { variant: "piefke", schema: condition, noise, windowActive: true },
      frames: 1,
      discount: 0.9,
    },
    probes: { ...settings.probes },
  };
}

export function trainEmergence(
  job: EmergenceJob,
  onEvent?: (event: TrainingEvent) => void,
  onProbe?: (source: ProbeSource, stage: "start" | "end") => void,
): EmergenceResult {
  const subject = effectiveSubject(job.subject, job.noise);
  const { model, points } = trainDqn(
    {
      subject,
      iterations: job.iterations,
      evalEvery: job.evalEvery,
      config: job.config,
      environment: job.environment,
      checks: emergenceChecks(job.condition),
    },
    onEvent,
  );
  const probes =
    hasResource(job.condition)
      ? runProbes(model, job.environment.world, subject, job.probes, onProbe)
      : [];
  return { condition: job.condition, noise: job.noise, subject: job.subject, points, probes };
}
