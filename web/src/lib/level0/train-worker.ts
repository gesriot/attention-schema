import { performance } from "node:perf_hooks";
import { parentPort, workerData } from "node:worker_threads";
import { trainSubject, type Phase, type TrainJob, type TrainResult } from "./dqn.ts";

export type Progress = {
  iteration: number;
  iterationMs: number;
  collectMs: number;
  updateMs: number;
  evaluateMs: number;
  loss: number;
};
export type WorkerMessage =
  | { type: "progress"; progress: Progress }
  | { type: "result"; result: TrainResult };

const job = workerData as TrainJob;
const starts: Record<Phase, number> = { iteration: 0, collect: 0, update: 0, evaluate: 0 };
const elapsed: Record<Phase, number> = { ...starts };
const { result } = trainSubject(job, (event) => {
  const now = performance.now();
  if (event.stage === "start") starts[event.phase] = now;
  else {
    elapsed[event.phase] = now - starts[event.phase];
    if (event.phase === "iteration") {
      parentPort!.postMessage({
        type: "progress",
        progress: {
          iteration: event.iteration,
          iterationMs: elapsed.iteration,
          collectMs: elapsed.collect,
          updateMs: elapsed.update / job.config.updates,
          evaluateMs: elapsed.evaluate,
          loss: event.loss,
        },
      } satisfies WorkerMessage);
    }
  }
});
parentPort!.postMessage({ type: "result", result } satisfies WorkerMessage);
