import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parentPort, workerData } from "node:worker_threads";
import { trainSubject, type TrainJob } from "./dqn.ts";
import { trainingReporter } from "./training-progress.ts";
import { encodeParameters, type ModelMetadata } from "../lab/model.ts";

const { saveModel, ...job } = workerData as TrainJob & { saveModel: string };
const { model, result } = trainSubject(
  job,
  trainingReporter(job.config.updates, (progress) => {
    parentPort!.postMessage({ type: "progress", progress });
  }),
);
const metadata: ModelMetadata = {
  sizes: model.sizes,
  condition: job.condition,
  subject: job.subject,
  iterations: job.iterations,
};
mkdirSync(saveModel, { recursive: true });
const prefix = join(saveModel, `${job.condition}-${job.subject}`);
writeFileSync(`${prefix}.bin`, encodeParameters(model.parameters));
writeFileSync(`${prefix}.json`, JSON.stringify(metadata, null, 2) + "\n");
parentPort!.postMessage({ type: "result", result });
