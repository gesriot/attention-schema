import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { Worker } from "node:worker_threads";

type PoolOptions<Job, Progress> = {
  jobs: Job[];
  workers: number;
  out: string | null;
  filename: (job: Job) => string;
  worker: URL;
  onProgress?: (job: Job, progress: Progress) => void;
};

export async function runStoredJobs<Job, Result, Progress>(
  options: PoolOptions<Job, Progress>,
): Promise<Result[]> {
  const { jobs } = options;
  const files = jobs.map((job) =>
    options.out === null ? null : join(options.out, options.filename(job)),
  );
  // Validate all existing files before starting any new work or creating a directory.
  const saved = jobs.map((job, index): Result | null => {
    const file = files[index]!;
    if (file === null || !existsSync(file)) return null;
    const stored = JSON.parse(readFileSync(file, "utf8")) as { job: Job; result: Result };
    if (JSON.stringify(stored.job) !== JSON.stringify(job)) {
      throw new Error(`Сохранённое обучение получено с другими параметрами: ${file}`);
    }
    return stored.result;
  });
  if (options.out !== null && saved.some((result) => result === null)) {
    mkdirSync(options.out, { recursive: true });
  }
  const results: Result[] = new Array(jobs.length);
  const active = new Set<Worker>();
  let next = 0;
  const consume = async () => {
    while (next < jobs.length) {
      const index = next++;
      const job = jobs[index]!;
      if (saved[index] !== null) {
        results[index] = saved[index]!;
        continue;
      }
      results[index] = await new Promise<Result>((resolveResult, reject) => {
        const worker = new Worker(options.worker, { workerData: job });
        active.add(worker);
        let result: Result | undefined;
        worker.on(
          "message",
          (message: { type: "progress"; progress: Progress } | { type: "result"; result: Result }) => {
            if (message.type === "progress") options.onProgress?.(job, message.progress);
            else result = message.result;
          },
        );
        worker.once("error", reject);
        worker.once("exit", (code) => {
          active.delete(worker);
          if (code === 0 && result !== undefined) resolveResult(result);
          else reject(new Error(`Обучение завершилось без результата, код ${code}`));
        });
      });
      const file = files[index]!;
      if (file !== null) {
        writeFileSync(`${file}.tmp`, JSON.stringify({ job, result: results[index]! }));
        renameSync(`${file}.tmp`, file);
      }
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(options.workers, jobs.length) }, consume));
    return results;
  } catch (error) {
    next = jobs.length;
    await Promise.all([...active].map((worker) => worker.terminate()));
    throw error;
  }
}
