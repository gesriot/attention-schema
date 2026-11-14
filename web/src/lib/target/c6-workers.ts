import { Worker } from "node:worker_threads";
import type { Config } from "./config.ts";
import type { C6Result, Checkpoint } from "./ppo.ts";

export type C6Job = { run: number; rate: number };
export type WorkerProgress = C6Job & {
  point: Checkpoint;
  collectionReward: number | null;
  elapsedMs: number;
  updateMs: number | null;
};

export async function runC6Jobs(
  config: Config,
  jobs: C6Job[],
  workers: number,
  onProgress?: (progress: WorkerProgress) => void,
): Promise<C6Result[]> {
  if (!Number.isSafeInteger(workers) || workers < 1)
    throw new RangeError("Нужно положительное число потоков");
  const ordered = [...jobs].sort((a, b) => a.rate - b.rate || a.run - b.run);
  const results: C6Result[] = new Array(ordered.length);
  const active = new Set<Worker>();
  let next = 0;
  const consume = async () => {
    while (next < ordered.length) {
      const index = next++;
      const job = ordered[index]!;
      results[index] = await new Promise<C6Result>((resolve, reject) => {
        const worker = new Worker(new URL("./c6-worker.ts", import.meta.url), {
          workerData: { config, ...job },
        });
        active.add(worker);
        let result: C6Result | undefined;
        worker.on(
          "message",
          (
            message: ({ type: "progress" } & WorkerProgress) | { type: "result"; result: C6Result },
          ) => {
            if (message.type === "progress") onProgress?.(message);
            else result = message.result;
          },
        );
        worker.once("error", reject);
        worker.once("exit", (code) => {
          active.delete(worker);
          if (code === 0 && result !== undefined) resolve(result);
          else reject(new Error(`Прогон C6 завершился без результата, код ${code}`));
        });
      });
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(workers, ordered.length) }, consume));
    return results;
  } catch (error) {
    next = ordered.length;
    await Promise.all([...active].map((worker) => worker.terminate()));
    throw error;
  }
}
