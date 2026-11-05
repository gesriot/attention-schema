import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseTrainingOptions, type TrainingOptions } from "./cli-options.ts";
import type { Check } from "./dqn.ts";
import {
  effectiveSubject,
  emergenceChecks,
  emergenceConditions,
  makeEmergenceJob,
  noiseGrid,
  publishedEmergence,
  type EmergenceCondition,
  type EmergenceJob,
  type EmergenceResult,
  type EmergenceSettings,
} from "./emergence.ts";
import type { EmergenceProgress } from "./emergence-worker.ts";
import { runStoredJobs } from "./job-runner.ts";
import { probeSources, type ProbeSource } from "./probes.ts";
import { meanAndError } from "./train-cli.ts";

export type EmergenceOptions = TrainingOptions<EmergenceCondition> & { noise: number[] };
const conditionNames: Record<EmergenceCondition, string> = {
  free: "свободный ресурс",
  given: "заданная схема",
  off: "без ресурса",
  clock: "часы",
  start: "колонка старта",
  parked: "неподвижный квадрат",
};
const sourceNames: Record<ProbeSource, string> = {
  resource: "ресурс",
  visual: "стимул",
  both: "стимул и ресурс",
  resourceRandom: "случайный ресурс",
  resourceStill: "неподвижный ресурс",
  untrained: "необученный агент",
  context: "только время",
};
const checkNames: Record<Check, string> = {
  normal: "норма",
  resourceRandom: "ресурс случаен",
  resourceStill: "ресурс неподвижен",
  fieldOff: "поле отключено",
  windowOff: "окно отключено",
};

export function parseEmergenceOptions(args: string[]): EmergenceOptions {
  const options = parseTrainingOptions(
    args, emergenceConditions, 50, noiseGrid,
  ) as EmergenceOptions;
  effectiveSubject(options.subjects, 0);
  return options;
}

export function runEmergenceJobs(
  options: EmergenceOptions,
  settings: EmergenceSettings = publishedEmergence,
  onProgress?: (job: EmergenceJob, progress: EmergenceProgress) => void,
): Promise<EmergenceResult[]> {
  const jobs: EmergenceJob[] = [];
  for (const condition of emergenceConditions) {
    if (!options.conditions.includes(condition)) continue;
    for (const noise of noiseGrid) {
      if (!options.noise.includes(noise)) continue;
      for (let subject = 1; subject <= options.subjects; subject++) {
        jobs.push(makeEmergenceJob(
          condition, noise, subject, options.iterations, options.evalEvery, settings,
        ));
      }
    }
  }
  return runStoredJobs<EmergenceJob, EmergenceResult, EmergenceProgress>({
    jobs,
    workers: options.workers,
    out: options.out,
    filename: (job) => `${job.condition}-p${job.noise.toFixed(2)}-${job.subject}.json`,
    worker: new URL("./emergence-worker.ts", import.meta.url),
    onProgress,
  });
}

export function tailMean(
  result: EmergenceResult,
  check: Check,
  key: "tracking" | "catching",
): number {
  const points = result.points.filter((point) => point.check === check).slice(-10);
  return meanAndError(points.map((point) => point[key])).mean;
}

export function resourceBenefits(
  results: EmergenceResult[],
  key: "tracking" | "catching",
  baseline: Check = "resourceRandom",
): number[] {
  return results.map(
    (result) => tailMean(result, "normal", key) - tailMean(result, baseline, key),
  );
}

// Label frequencies differ between sources, so raw accuracies are not comparable.
export function probeLift(accuracy: number, majority: number): number {
  return (accuracy - majority) / (1 - majority);
}

export function formatEmergenceResults(results: EmergenceResult[]): string {
  const ordered = [...results].sort(
    (a, b) =>
      emergenceConditions.indexOf(a.condition) - emergenceConditions.indexOf(b.condition) ||
      a.noise - b.noise || a.subject - b.subject,
  );
  const lines = ["Кривые E6", "Условие | Шум | Субъект | Итерация | Проверка | Слежение | Ловля"];
  for (const result of ordered) {
    for (const point of result.points) {
      lines.push(
        `${conditionNames[result.condition]} | ${result.noise.toFixed(2)} | ` +
          `${result.subject.toFixed(0)} | ${point.iteration.toFixed(0)} | ` +
          `${checkNames[point.check]} | ${point.tracking.toFixed(3)} | ${point.catching.toFixed(3)}`,
      );
    }
  }
  const format = (values: number[]) => {
    const { mean, error } = meanAndError(values);
    return `${mean.toFixed(3)} ± ${error === null ? "н/д" : error.toFixed(3)}`;
  };
  lines.push("", "Итог · до 10 последних проверок · среднее ± стандартная ошибка по субъектам");
  lines.push("Условие | Шум | Проверка | Слежение | Ловля");
  for (const condition of emergenceConditions) {
    for (const noise of noiseGrid) {
      const subjects = ordered.filter(
        (result) => result.condition === condition && result.noise === noise,
      );
      if (subjects.length === 0) continue;
      for (const check of emergenceChecks(condition)) {
        lines.push(
          `${conditionNames[condition]} | ${noise.toFixed(2)} | ${checkNames[check]} | ` +
            `${format(subjects.map((result) => tailMean(result, check, "tracking")))} | ` +
            `${format(subjects.map((result) => tailMean(result, check, "catching")))}`,
        );
      }
    }
  }
  lines.push(
    "", "Полезность ресурса · норма минус база · парные разности по субъектам",
  );
  lines.push("Шум | База | Слежение | Ловля");
  for (const noise of noiseGrid) {
    const subjects = ordered.filter(
      (result) => result.condition === "free" && result.noise === noise,
    );
    if (subjects.length === 0) continue;
    for (const baseline of ["resourceRandom", "resourceStill"] as const) {
      lines.push(
        `${noise.toFixed(2)} | ${checkNames[baseline]} | ` +
          `${format(resourceBenefits(subjects, "tracking", baseline))} | ` +
          `${format(resourceBenefits(subjects, "catching", baseline))}`,
      );
    }
  }
  lines.push("", "Зонды · проверочные эпизоды · доли · среднее ± стандартная ошибка по субъектам");
  lines.push("Шум | Источник | Точность | Частый класс | Прирост над частым классом");
  for (const noise of noiseGrid) {
    const subjects = ordered.filter(
      (result) => result.condition === "free" && result.noise === noise,
    );
    if (subjects.length === 0) continue;
    for (const source of probeSources) {
      const probes = subjects.map(
        (result) => result.probes.find((probe) => probe.source === source)!,
      );
      lines.push(
        `${noise.toFixed(2)} | ${sourceNames[source]} | ` +
          `${format(probes.map((probe) => probe.accuracy))} | ` +
          `${format(probes.map((probe) => probe.majority))} | ` +
          `${format(probes.map((probe) => probeLift(probe.accuracy, probe.majority)))}`,
      );
    }
  }
  lines.push("Стандартная ошибка при одном субъекте не определена (н/д).");
  return lines.join("\n") + "\n";
}

export async function runEmergenceCli(
  args: string[],
  settings: EmergenceSettings = publishedEmergence,
): Promise<void> {
  const options = parseEmergenceOptions(args);
  const results = await runEmergenceJobs(options, settings, (job, progress) => {
    const prefix =
      `${conditionNames[job.condition]} · шум ${job.noise.toFixed(2)} · ` +
      `субъект ${job.subject.toFixed(0)}`;
    if (progress.phase === "probe") {
      process.stderr.write(
        `${prefix} · зонд ${sourceNames[progress.source]} · ` +
          `${(progress.elapsedMs / 1000).toFixed(3)} с\n`,
      );
    } else {
      const p = progress.training;
      process.stderr.write(
        `${prefix} · итерация ${p.iteration.toFixed(0)}/${job.iterations.toFixed(0)} · ` +
          `сбор ${(p.collectMs / 1000).toFixed(3)} с · обновление ${p.updateMs.toFixed(3)} мс · ` +
          `проверка ${(p.evaluateMs / 1000).toFixed(3)} с · ` +
          `итерация ${(p.iterationMs / 1000).toFixed(3)} с · потери ${p.loss.toFixed(6)}\n`,
      );
    }
  });
  process.stdout.write(formatEmergenceResults(results));
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runEmergenceCli(process.argv.slice(2)).catch((error: unknown) => {
    process.stderr.write(`Ошибка: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
