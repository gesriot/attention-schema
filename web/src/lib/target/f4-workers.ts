import { Worker } from "node:worker_threads";
import type { Config } from "./config.ts";
import {
  probeData,
  sources,
  type Corpus,
  type ProbeProgress,
  type ProbeResult,
  type Source,
} from "./f4.ts";

export type WorkerProgress = ProbeProgress & { source: Source; elapsedMs: number };

export async function runProbes(
  config: Config,
  corpus: Corpus,
  workers: number,
  onProgress?: (progress: WorkerProgress) => void,
): Promise<ProbeResult[]> {
  if (!Number.isSafeInteger(workers) || workers < 1)
    throw new Error("Нужно положительное число потоков");
  const results: ProbeResult[] = new Array(sources.length);
  const active = new Set<Worker>();
  let next = 0;
  const consume = async () => {
    while (next < sources.length) {
      const index = next++;
      const source = sources[index]!;
      results[index] = await new Promise<ProbeResult>((resolveResult, reject) => {
        const worker = new Worker(new URL("./f4-worker.ts", import.meta.url), {
          workerData: { config, data: probeData(config, corpus, source), source },
        });
        active.add(worker);
        let result: ProbeResult | undefined;
        worker.on(
          "message",
          (
            message:
              ({ type: "progress" } & WorkerProgress) | { type: "result"; result: ProbeResult },
          ) => {
            if (message.type === "progress") onProgress?.(message);
            else result = message.result;
          },
        );
        worker.once("error", reject);
        worker.once("exit", (code) => {
          active.delete(worker);
          if (code === 0 && result !== undefined) resolveResult(result);
          else reject(new Error(`Зонд завершился без результата, код ${code}`));
        });
      });
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(workers, sources.length) }, consume));
    return results;
  } catch (error) {
    next = sources.length;
    await Promise.all([...active].map((worker) => worker.terminate()));
    throw error;
  }
}
