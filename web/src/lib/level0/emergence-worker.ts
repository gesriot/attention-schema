import { performance } from "node:perf_hooks";
import { parentPort, workerData } from "node:worker_threads";
import { trainEmergence, type EmergenceJob } from "./emergence.ts";
import type { ProbeSource } from "./probes.ts";
import { trainingReporter, type Progress } from "./training-progress.ts";

export type EmergenceProgress =
  | { phase: "training"; training: Progress }
  | { phase: "probe"; source: ProbeSource; elapsedMs: number };

const job = workerData as EmergenceJob;
let probeStart = 0;
const result = trainEmergence(
  job,
  trainingReporter(job.config.updates, (training) => {
    parentPort!.postMessage({
      type: "progress",
      progress: { phase: "training", training } satisfies EmergenceProgress,
    });
  }),
  (source, stage) => {
    if (stage === "start") probeStart = performance.now();
    else {
      parentPort!.postMessage({
        type: "progress",
        progress: {
          phase: "probe", source, elapsedMs: performance.now() - probeStart,
        } satisfies EmergenceProgress,
      });
    }
  },
);
parentPort!.postMessage({ type: "result", result });
