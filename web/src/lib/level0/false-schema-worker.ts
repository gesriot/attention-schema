import { parentPort, workerData } from "node:worker_threads";
import { createMlp } from "../nn/mlp.ts";
import { createRng } from "../rng.ts";
import { conditionConfig, trainSubject, type TrainJob, type TrainResult } from "./dqn.ts";
import { modelPolicy, probeFalseSchema, type FalseSchemaResult } from "./false-schema.ts";
import { trainingReporter } from "./training-progress.ts";

export type FalseSchemaRun = TrainResult & {
  trained: FalseSchemaResult;
  untrained: FalseSchemaResult;
};

const probeEpisodes = 2000;
const job = workerData as TrainJob;
const config = conditionConfig(job.condition, true, job.slip);
// Built from the same stream as the network that goes on to train, so it starts identical.
const untrained = createMlp(job.config.sizes, createRng(job.subject, 3));
const { model, result } = trainSubject(
  job,
  trainingReporter(job.config.updates, (progress) => {
    parentPort!.postMessage({ type: "progress", progress });
  }),
);
parentPort!.postMessage({
  type: "result",
  result: {
    ...result,
    trained: probeFalseSchema(modelPolicy(model), config, probeEpisodes),
    untrained: probeFalseSchema(modelPolicy(untrained), config, probeEpisodes),
  } satisfies FalseSchemaRun,
});
