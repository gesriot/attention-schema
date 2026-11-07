import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  checksFor,
  conditions,
  publishedTraining,
  type Check,
  type Condition,
  type TrainJob,
  type TrainingConfig,
  type TrainResult,
} from "./dqn.ts";
import type { Progress } from "./train-worker.ts";
import { runStoredJobs } from "./job-runner.ts";
import { parseTrainingOptions, type TrainingOptions } from "./cli-options.ts";

export type Options = TrainingOptions<Condition>;
// Single checkpoints swing widely under this training, so the summary averages the last ones.
const tailPoints = 10;
const conditionNames: Record<Condition, string> = {
  schema: "схема",
  none: "без схемы",
  commands: "команды",
  read: "схема из кадра",
  pool2: "блок 2",
  pool3: "блок 3",
  pool4: "блок 4",
  read2: "кадр и телеметрия 2",
};
const checkNames: Record<Check, string> = {
  normal: "норма",
  windowOff: "окно отключено",
  fieldOff: "поле отключено",
  resourceRandom: "ресурс случаен",
  resourceStill: "ресурс неподвижен",
};

export function parseOptions(args: string[]): Options {
  return parseTrainingOptions(args, conditions, 10, undefined, true);
}

export async function runJobs(
  options: Options,
  config: TrainingConfig = publishedTraining,
  onProgress?: (job: TrainJob, progress: Progress) => void,
): Promise<TrainResult[]> {
  const jobs: TrainJob[] = [];
  for (const condition of conditions) {
    if (!options.conditions.includes(condition)) continue;
    for (let subject = 1; subject <= options.subjects; subject++) {
      jobs.push({
        condition,
        subject,
        iterations: options.iterations,
        evalEvery: options.evalEvery,
        config,
        ...(options.slip === undefined ? {} : { slip: options.slip }),
      });
    }
  }
  return runStoredJobs<TrainJob, TrainResult, Progress>({
    jobs,
    workers: options.workers,
    out: options.out,
    filename: (job) => `${job.condition}-${job.subject}.json`,
    worker: new URL("./train-worker.ts", import.meta.url),
    onProgress,
  });
}

export function meanAndError(values: number[]): { mean: number; error: number | null } {
  let sum = 0;
  for (const value of values) sum += value;
  const mean = sum / values.length;
  if (values.length < 2) return { mean, error: null };
  let squared = 0;
  for (const value of values) squared += (value - mean) * (value - mean);
  return { mean, error: Math.sqrt(squared / (values.length * (values.length - 1))) };
}

export function formatResults(results: TrainResult[]): string {
  const ordered = [...results].sort(
    (a, b) =>
      conditions.indexOf(a.condition) - conditions.indexOf(b.condition) || a.subject - b.subject,
  );
  const lines = ["Кривые", "Условие | Субъект | Итерация | Проверка | Слежение | Ловля"];
  for (const result of ordered) {
    for (const point of result.points) {
      lines.push(
        `${conditionNames[result.condition]} | ${result.subject.toFixed(0)} | ` +
          `${point.iteration.toFixed(0)} | ${checkNames[point.check]} | ` +
          `${point.tracking.toFixed(3)} | ${point.catching.toFixed(3)}`,
      );
    }
  }
  lines.push("", `Итог · последние ${tailPoints} точек · среднее ± стандартная ошибка по субъектам`);
  lines.push("Условие | Проверка | Слежение | Ловля");
  const format = (values: number[]) => {
    const { mean, error } = meanAndError(values);
    return `${mean.toFixed(3)} ± ${error === null ? "н/д" : error.toFixed(3)}`;
  };
  for (const condition of conditions) {
    const subjects = ordered.filter((result) => result.condition === condition);
    if (subjects.length === 0) continue;
    for (const check of checksFor(condition)) {
      const tails = subjects.map((result) =>
        result.points.filter((point) => point.check === check).slice(-tailPoints),
      );
      const subjectMeans = (key: "tracking" | "catching") =>
        tails.map((points) => meanAndError(points.map((point) => point[key])).mean);
      lines.push(
        `${conditionNames[condition]} | ${checkNames[check]} | ` +
          `${format(subjectMeans("tracking"))} | ${format(subjectMeans("catching"))}`,
      );
    }
  }
  lines.push("Стандартная ошибка при одном субъекте не определена (н/д).");
  return lines.join("\n") + "\n";
}

export async function runCli(
  args: string[],
  config: TrainingConfig = publishedTraining,
): Promise<void> {
  const options = parseOptions(args);
  const results = await runJobs(options, config, (job, progress) => {
    process.stderr.write(
      `${conditionNames[job.condition]} · субъект ${job.subject.toFixed(0)} · ` +
        `итерация ${progress.iteration.toFixed(0)}/${job.iterations.toFixed(0)} · ` +
        `сбор ${(progress.collectMs / 1000).toFixed(3)} с · ` +
        `обновление ${progress.updateMs.toFixed(3)} мс · ` +
        `проверка ${(progress.evaluateMs / 1000).toFixed(3)} с · ` +
        `итерация ${(progress.iterationMs / 1000).toFixed(3)} с · ` +
        `потери ${progress.loss.toFixed(6)}\n`,
    );
  });
  process.stdout.write(formatResults(results));
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli(process.argv.slice(2)).catch((error: unknown) => {
    process.stderr.write(`Ошибка: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
