import { availableParallelism } from "node:os";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { formatParameters, parseOptions, type Options } from "./check-cli.ts";
import {
  collectCorpus,
  f4FirstSeed,
  sources,
  splitEpisodes,
  type ProbeResult,
  type Source,
} from "./f4.ts";
import { runProbes } from "./f4-workers.ts";

const sourceNames: Record<Source, string> = {
  frames: "Чистые кадры",
  perception: "Восприятие",
  perceptionStep: "Восприятие, один шаг",
  commands: "Команды",
  telemetry: "Телеметрия",
  telemetryStep: "Телеметрия, один шаг",
  schema: "Все входы схемы",
  context: "Только время",
  measurement: "Измерение (справочно)",
};

export type F4Options = Options & { sources: Source[] };

export function parseF4Options(args: string[]): F4Options {
  const remaining: string[] = [];
  let selected: Source[] | undefined;
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index]!;
    const value = args[index + 1];
    if (flag !== "--sources") {
      remaining.push(flag);
      if (value !== undefined) remaining.push(value);
      continue;
    }
    if (selected !== undefined) throw new Error("Повторный параметр: --sources");
    if (value === undefined || !value.trim() || value.startsWith("--"))
      throw new Error("Нужен непустой список источников");
    const names = value.split(",").map((name) => name.trim());
    for (const name of names) {
      if (!sources.includes(name as Source)) throw new Error(`Неизвестный источник: ${name}`);
    }
    selected = sources.filter((source) => names.includes(source));
  }
  return { ...parseOptions(remaining), sources: selected ?? [...sources] };
}

export function formatResults(options: Options, results: ProbeResult[]): string {
  const split = splitEpisodes(options.config, options.episodes);
  const lines = formatParameters(options.config);
  lines.push(
    "",
    `F4 · эпизодов ${options.episodes.toFixed(0)} · зёрна ` +
      `${f4FirstSeed.toFixed(0)}–${(f4FirstSeed + options.episodes - 1).toFixed(0)}`,
  );
  lines.push(
    `Обучение: первые ${split.train.toFixed(0)} эпизодов; проверка: последние ${split.test.toFixed(0)}.`,
  );
  lines.push(
    "Источник | Средняя ошибка, клеток | Доля с ошибкой не больше 1 клетки | " +
      "Наименьшая ошибка по контрольным точкам",
  );
  for (const source of sources) {
    const result = results.find((item) => item.source === source);
    if (result === undefined) continue;
    lines.push(
      `${sourceNames[source]} | ${result.meanError.toFixed(6)} | ${result.withinOne.toFixed(6)} | ` +
        result.bestError.toFixed(6),
    );
  }
  return lines.join("\n") + "\n";
}

export async function runCli(args: string[]): Promise<void> {
  const options = parseF4Options(args);
  const start = performance.now();
  const corpus = collectCorpus(options.config, options.episodes, options.sources);
  const workers = Math.min(4, availableParallelism(), options.sources.length);
  process.stderr.write(
    `Сбор данных: ${((performance.now() - start) / 1000).toFixed(3)} с; ` +
      `потоков зондов: ${workers.toFixed(0)}\n`,
  );
  const results = await runProbes(
    options.config,
    corpus,
    workers,
    (progress) => {
      process.stderr.write(
        `${sourceNames[progress.source]} · ` +
          `${progress.update.toFixed(0)}/${options.config.f4Updates.toFixed(0)} · ` +
          `потери ${progress.loss.toFixed(6)} · ошибка ${progress.meanError.toFixed(6)} · ` +
          `${(progress.elapsedMs / 1000).toFixed(3)} с\n`,
      );
    },
    options.sources,
  );
  process.stdout.write(formatResults(options, results));
  process.stderr.write(`Время прогона: ${((performance.now() - start) / 1000).toFixed(3)} с\n`);
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli(process.argv.slice(2)).catch((error: unknown) => {
    process.stderr.write(`Ошибка: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
