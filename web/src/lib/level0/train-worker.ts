import { parentPort, workerData } from "node:worker_threads";
import { trainSubject, type TrainJob, type TrainResult } from "./dqn.ts";
import { trainingReporter, type Progress } from "./training-progress.ts";
export type { Progress } from "./training-progress.ts";

export type WorkerMessage =
  | { type: "progress"; progress: Progress }
  | { type: "result"; result: TrainResult };

const job = workerData as TrainJob;
const { result } = trainSubject(
  job,
  trainingReporter(job.config.updates, (progress) => {
    parentPort!.postMessage({ type: "progress", progress } satisfies WorkerMessage);
  }),
);
parentPort!.postMessage({ type: "result", result } satisfies WorkerMessage);
