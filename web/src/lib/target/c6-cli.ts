import { availableParallelism } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { formatParameters, parseOptions } from "./check-cli.ts";
import { runC6Jobs, type WorkerProgress } from "./c6-workers.ts";
import type { Config } from "./config.ts";
import type { C6Result, C6Metrics } from "./ppo.ts";

export type C6Options = { config: Config; runs: number; rates: number[]; workers: number };

export function parseC6Options(args: string[]): C6Options {
  let runs = 1;
  let rates = [0.0003];
  let workers = Math.min(4, availableParallelism());
  const remaining: string[] = [];
  const seen = new Set<string>();
  for (let i = 0; i < args.length; i += 2) {
    const flag = args[i]!;
    const value = args[i + 1];
    if (value === undefined || !value.trim() || value.startsWith("--"))
      throw new Error(`Нет значения: ${flag}`);
    if (flag === "--set") {
      remaining.push(flag, value);
      continue;
    }
    if (seen.has(flag)) throw new Error(`Повторный параметр: ${flag}`);
    seen.add(flag);
    if (flag === "--runs") runs = Number(value);
    else if (flag === "--workers") workers = Number(value);
    else if (flag === "--rates") rates = value.split(",").map(Number);
    else throw new Error(`Неизвестный аргумент: ${flag}`);
  }
  if (!Number.isSafeInteger(runs) || runs < 1 || runs > 99)
    throw new RangeError("Число прогонов должно быть от 1 до 99");
  if (!Number.isSafeInteger(workers) || workers < 1)
    throw new RangeError("Нужно положительное число потоков");
  if (
    rates.some((rate) => !Number.isFinite(rate) || rate <= 0) ||
    new Set(rates).size !== rates.length
  )
    throw new RangeError("Нужен список различных положительных конечных скоростей обучения");
  rates.sort((a, b) => a - b);
  return { config: parseOptions(remaining).config, runs, rates, workers };
}

function formatMetrics(metrics: C6Metrics): string {
  return [metrics.reward, metrics.caught, metrics.correct, metrics.effort, metrics.focus]
    .map((value) => value.toFixed(6))
    .join(" | ");
}

export function formatC6Results(config: Config, results: C6Result[]): string {
  const lines = formatParameters(config);
  for (const result of [...results].sort((a, b) => a.rate - b.rate || a.run - b.run)) {
    lines.push(
      "",
      `C6 · скорость ${result.rate.toFixed(6)} · прогон ${result.run.toFixed(0)}`,
      "Обновление | Награда | Ловля | Верные ответы | Доля усилия | Фокус у цели",
    );
    for (const point of result.checkpoints)
      lines.push(`${point.update.toFixed(0)} | ${formatMetrics(point)}`);
  }
  return lines.join("\n") + "\n";
}

export function formatC6Progress(progress: WorkerProgress): string {
  const { run, rate, point, collectionReward, elapsedMs, updateMs } = progress;
  return (
    `C6 · скорость ${rate.toFixed(6)} · прогон ${run.toFixed(0)} · обновление ${point.update.toFixed(0)} · ` +
    `награда сбора ${collectionReward === null ? "н/д" : collectionReward.toFixed(6)} · ` +
    `проверка (награда | ловля | ответы | усилие | фокус): ${formatMetrics(point)} · ` +
    `среднее обновление ${updateMs === null ? "н/д" : (updateMs / 1000).toFixed(3) + " с"} · ` +
    `${(elapsedMs / 1000).toFixed(3)} с\n`
  );
}

export async function runCli(args: string[]): Promise<void> {
  const options = parseC6Options(args);
  const jobs = options.rates.flatMap((rate) =>
    Array.from({ length: options.runs }, (_, i) => ({ rate, run: i + 1 })),
  );
  const start = performance.now();
  const results = await runC6Jobs(options.config, jobs, options.workers, (progress) =>
    process.stderr.write(formatC6Progress(progress)),
  );
  process.stdout.write(formatC6Results(options.config, results));
  process.stderr.write(`Время прогона: ${((performance.now() - start) / 1000).toFixed(3)} с\n`);
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli(process.argv.slice(2)).catch((error: unknown) => {
    process.stderr.write(`Ошибка: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
