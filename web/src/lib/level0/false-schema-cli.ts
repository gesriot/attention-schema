import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseTrainingOptions } from "./cli-options.ts";
import { publishedTraining, type TrainJob } from "./dqn.ts";
import { falseScopes, falseShifts, type FalseSchemaResult } from "./false-schema.ts";
import type { FalseSchemaRun } from "./false-schema-worker.ts";
import { runStoredJobs } from "./job-runner.ts";
import { meanAndError } from "./train-cli.ts";
import type { Progress } from "./training-progress.ts";

const scopeNames = {
  history: "вся история",
  latest: "последний кадр",
  cleaned: "вся история без шума под квадратом",
} as const;

function format(values: number[]): string {
  const { mean, error } = meanAndError(values);
  return `${mean.toFixed(3)} ± ${error === null ? "н/д" : error.toFixed(3)}`;
}

function formatProbe(name: string, probes: FalseSchemaResult[]): string {
  let text = "";
  for (const scope of falseScopes) {
    for (const shift of falseShifts) {
      const rows = probes.map(
        (probe) => probe.shifts.find((row) => row.scope === scope && row.shift === shift)!,
      );
      text +=
        `${name} | ${scopeNames[scope]} | ${shift.toFixed(0)} | ` +
        `${format(rows.map((row) => row.direction))} | ${format(rows.map((row) => row.changed))}\n`;
    }
  }
  return text;
}

function formatConsequences(name: string, probes: FalseSchemaResult[]): string {
  let text = "";
  for (const shift of falseShifts) {
    const rows = probes.map((probe) => probe.consequences.find((row) => row.shift === shift)!);
    text +=
      `${name} | ${shift.toFixed(0)} | ${format(rows.map((row) => row.focusClean))} | ` +
      `${format(rows.map((row) => row.focusFalse))} | ` +
      `${format(rows.map((row) => row.trackingClean))} | ` +
      `${format(rows.map((row) => row.trackingFalse))}\n`;
  }
  return text;
}

export function formatResults(runs: FalseSchemaRun[]): string {
  const trained = runs.map((run) => run.trained);
  const untrained = runs.map((run) => run.untrained);
  return (
    "Итог · среднее ± стандартная ошибка по субъектам\n" +
    "Сеть | Подмена | Сдвиг поля | Команда против сдвига | Доля изменённых команд\n" +
    formatProbe("обученная", trained) +
    formatProbe("необученная", untrained) +
    "\nПоследствия одной подмены на шаге 5\n" +
    "Сеть | Сдвиг поля | Фокус на цели: норма | подмена | Слежение дальше: норма | подмена\n" +
    formatConsequences("обученная", trained) +
    formatConsequences("необученная", untrained)
  );
}

export async function runCli(args: string[]): Promise<void> {
  const options = parseTrainingOptions(args, ["schema"] as const, 10, undefined, true);
  const jobs: TrainJob[] = [];
  for (let subject = 1; subject <= options.subjects; subject++) {
    jobs.push({
      condition: "schema",
      subject,
      iterations: options.iterations,
      evalEvery: options.evalEvery,
      config: publishedTraining,
      ...(options.slip === undefined ? {} : { slip: options.slip }),
    });
  }
  const runs = await runStoredJobs<TrainJob, FalseSchemaRun, Progress>({
    jobs,
    workers: options.workers,
    out: options.out,
    filename: (job) => `schema-${job.subject}.json`,
    worker: new URL("./false-schema-worker.ts", import.meta.url),
    onProgress: (job, progress) => {
      process.stderr.write(
        `субъект ${job.subject.toFixed(0)} · ` +
          `итерация ${progress.iteration.toFixed(0)}/${job.iterations.toFixed(0)} · ` +
          `потери ${progress.loss.toFixed(6)}\n`,
      );
    },
  });
  process.stdout.write(formatResults(runs));
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  runCli(process.argv.slice(2)).catch((error: unknown) => {
    process.stderr.write(`Ошибка: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  });
}
