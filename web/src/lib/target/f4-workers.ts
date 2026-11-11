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
  selected: readonly Source[] = sources,
): Promise<ProbeResult[]> {
  if (!Number.isSafeInteger(workers) || workers < 1)
    throw new Error("Нужно положительное число потоков");
  const ordered = sources.filter((source) => selected.includes(source));
  if (ordered.length === 0 || selected.some((source) => !sources.includes(source)))
    throw new Error("Нужен непустой список известных источников");
  const results: ProbeResult[] = new Array(ordered.length);
  const active = new Set<Worker>();
  let next = 0;
  const consume = async () => {
    while (next < ordered.length) {
      const index = next++;
      const source = ordered[index]!;
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
    await Promise.all(Array.from({ length: Math.min(workers, ordered.length) }, consume));
    return results;
  } catch (error) {
    next = ordered.length;
    await Promise.all([...active].map((worker) => worker.terminate()));
    throw error;
  }
}
