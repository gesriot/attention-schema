import { parentPort, workerData } from "node:worker_threads";
import type { Config } from "./config.ts";
import { fitProbe, type ProbeData, type Source } from "./f4.ts";

const { config, data, source } = workerData as { config: Config; data: ProbeData; source: Source };
const start = performance.now();
const result = fitProbe(config, data, source, (progress) => {
  parentPort!.postMessage({
    type: "progress",
    source,
    ...progress,
    elapsedMs: performance.now() - start,
  });
});
parentPort!.postMessage({ type: "result", result });
