import { performance } from "node:perf_hooks";
import type { Phase, TrainingEvent } from "./dqn.ts";

export type Progress = {
  iteration: number;
  iterationMs: number;
  collectMs: number;
  updateMs: number;
  evaluateMs: number;
  loss: number;
};

export function trainingReporter(
  updates: number,
  emit: (progress: Progress) => void,
): (event: TrainingEvent) => void {
  const starts: Record<Phase, number> = { iteration: 0, collect: 0, update: 0, evaluate: 0 };
  const elapsed: Record<Phase, number> = { ...starts };
  return (event) => {
    const now = performance.now();
    if (event.stage === "start") starts[event.phase] = now;
    else {
      elapsed[event.phase] = now - starts[event.phase];
      if (event.phase === "iteration") {
        emit({
          iteration: event.iteration,
          iterationMs: elapsed.iteration,
          collectMs: elapsed.collect,
          updateMs: elapsed.update / updates,
          evaluateMs: elapsed.evaluate,
          loss: event.loss,
        });
      }
    }
  };
}
