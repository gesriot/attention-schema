import { parentPort, workerData } from "node:worker_threads";
import type { Config } from "./config.ts";
import { trainC6 } from "./ppo.ts";

const { config, run, rate } = workerData as { config: Config; run: number; rate: number };
const start = performance.now();
let updateStart = start;
let updateTotal = 0;
let updates = 0;
const result = trainC6(config, run, rate, (event) => {
  if (event.type === "updateStart") updateStart = performance.now();
  if (event.type === "updateEnd") {
    updateTotal += performance.now() - updateStart;
    updates++;
  }
  if (event.type === "checkpoint")
    parentPort!.postMessage({
      type: "progress",
      run,
      rate,
      point: event.point,
      collectionReward: event.collectionReward,
      elapsedMs: performance.now() - start,
      updateMs: updates === 0 ? null : updateTotal / updates,
    });
});
parentPort!.postMessage({ type: "result", result });
