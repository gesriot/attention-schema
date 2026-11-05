import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Worker } from "node:worker_threads";
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
import type { Progress, WorkerMessage } from "./train-worker.ts";

export type Options = {
  subjects: number;
  iterations: number;
  evalEvery: number;
  workers: number;
  conditions: Condition[];
  out: string | null;
};
const conditionNames: Record<Condition, string> = {
  schema: "схема",
  none: "без схемы",
  commands: "команды",
};
const checkNames: Record<Check, string> = {
  normal: "норма",
  windowOff: "окно отключено",
  fieldOff: "поле отключено",
};

export function parseOptions(args: string[]): Options {
  const result: Options = {
    subjects: 5,
    iterations: 1500,
    evalEvery: 10,
    workers: 8,
    conditions: [...conditions],
    out: null,
  };
  const keys = {
    "--subjects": "subjects",
    "--iterations": "iterations",
    "--eval-every": "evalEvery",
    "--workers": "workers",
  } as const;
  const seen = new Set<string>();
  for (let i = 0; i < args.length; i += 2) {
    const flag = args[i]!;
    const value = args[i + 1];
    if (seen.has(flag) || value === undefined) throw new Error(`Неверный аргумент: ${flag}`);
    seen.add(flag);
    if (flag === "--conditions") {
      const requested = value.split(",");
      if (
        requested.length !== new Set(requested).size ||
        requested.some((name) => !conditions.includes(name as Condition))
      ) {
        throw new Error("Допустимые условия: schema,none,commands, без повторов");
      }
      result.conditions = conditions.filter((name) => requested.includes(name));
    } else if (flag === "--out") {
      result.out = value;
    } else if (Object.hasOwn(keys, flag)) {
      const number = Number(value);
      if (!/^\d+$/.test(value) || !Number.isSafeInteger(number) || number < 1) {
        throw new Error(`Нужно положительное целое число: ${flag}`);
      }
      result[keys[flag as keyof typeof keys]] = number;
    } else {
      throw new Error(`Неизвестный аргумент: ${flag}`);
    }
  }
  return result;
}

// A finished job survives a crash of a long run and is reused only for identical parameters.
function loadJob(file: string, job: TrainJob): TrainResult | null {
  if (!existsSync(file)) return null;
  const saved = JSON.parse(readFileSync(file, "utf8")) as { job: TrainJob; result: TrainResult };
  if (JSON.stringify(saved.job) !== JSON.stringify(job)) {
    throw new Error(`Сохранённое обучение получено с другими параметрами: ${file}`);
  }
  return saved.result;
}

function saveJob(file: string, job: TrainJob, result: TrainResult): void {
  writeFileSync(`${file}.tmp`, JSON.stringify({ job, result }));
  renameSync(`${file}.tmp`, file);
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
      });
    }
  }
  const results: TrainResult[] = new Array(jobs.length);
  const active = new Set<Worker>();
  if (options.out !== null) mkdirSync(options.out, { recursive: true });
  let next = 0;
  const consume = async () => {
    while (next < jobs.length) {
      const index = next++;
      const job = jobs[index]!;
      const file =
        options.out === null ? null : join(options.out, `${job.condition}-${job.subject}.json`);
      const saved = file === null ? null : loadJob(file, job);
      if (saved !== null) {
        results[index] = saved;
        continue;
      }
      results[index] = await new Promise<TrainResult>((resolveResult, reject) => {
        const worker = new Worker(new URL("./train-worker.ts", import.meta.url), {
          workerData: job,
        });
        active.add(worker);
        let result: TrainResult | undefined;
        worker.on("message", (message: WorkerMessage) => {
          if (message.type === "progress") onProgress?.(job, message.progress);
          else result = message.result;
        });
        worker.once("error", reject);
        worker.once("exit", (code) => {
          active.delete(worker);
          if (code === 0 && result !== undefined) resolveResult(result);
          else reject(new Error(`Обучение завершилось без результата, код ${code}`));
        });
      });
      if (file !== null) saveJob(file, job, results[index]!);
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(options.workers, jobs.length) }, consume));
    return results;
  } catch (error) {
    // Do not start queued jobs after an error; terminate only this pool's workers.
    next = jobs.length;
    await Promise.all([...active].map((worker) => worker.terminate()));
    throw error;
  }
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
  lines.push("", "Итог · последняя точка · среднее ± стандартная ошибка по субъектам");
  lines.push("Условие | Проверка | Слежение | Ловля");
  const format = (values: number[]) => {
    const { mean, error } = meanAndError(values);
    return `${mean.toFixed(3)} ± ${error === null ? "н/д" : error.toFixed(3)}`;
  };
  for (const condition of conditions) {
    const subjects = ordered.filter((result) => result.condition === condition);
    if (subjects.length === 0) continue;
    for (const check of checksFor(condition)) {
      const points = subjects.map(
        (result) => result.points.filter((point) => point.check === check).at(-1)!,
      );
      lines.push(
        `${conditionNames[condition]} | ${checkNames[check]} | ` +
          `${format(points.map((point) => point.tracking))} | ` +
          `${format(points.map((point) => point.catching))}`,
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
