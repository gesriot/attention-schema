import { nextInt, type Rng } from "../rng.ts";

export type StepType = 0 | 1 | 2;
export const FIRST = 0;
export const MID = 1;
export const LAST = 2;
export type Replay = {
  capacity: number;
  width: number;
  size: number;
  head: number;
  observations: Uint8Array;
  types: Uint8Array;
  actions: Uint8Array;
  rewards: Float64Array;
  discounts: Float64Array;
};
export type ReplayBatch = {
  size: number;
  observations: Float64Array;
  nextObservations: Float64Array;
  actions: Uint8Array;
  rewards: Float64Array;
  discounts: Float64Array;
  masks: Float64Array;
};

export function createReplay(capacity = 10000, width = 2000): Replay {
  if (!Number.isInteger(capacity) || capacity < 2 || !Number.isInteger(width) || width < 1) {
    throw new RangeError("Буферу нужны хотя бы два шага и положительный размер наблюдения");
  }
  return {
    capacity,
    width,
    size: 0,
    head: 0,
    observations: new Uint8Array(capacity * width),
    types: new Uint8Array(capacity),
    actions: new Uint8Array(capacity),
    rewards: new Float64Array(capacity),
    discounts: new Float64Array(capacity),
  };
}

export function pushReplay(
  replay: Replay,
  observation: Uint8Array,
  type: StepType,
  action: number,
  reward: number,
  discount: number,
): void {
  if (observation.length !== replay.width) throw new RangeError("Неверный размер наблюдения");
  const index = replay.head;
  replay.observations.set(observation, index * replay.width);
  replay.types[index] = type;
  replay.actions[index] = action;
  replay.rewards[index] = reward;
  replay.discounts[index] = discount;
  replay.head = (index + 1) % replay.capacity;
  replay.size = Math.min(replay.capacity, replay.size + 1);
}

export function createReplayBatch(size: number, width: number): ReplayBatch {
  return {
    size,
    observations: new Float64Array(size * width),
    nextObservations: new Float64Array(size * width),
    actions: new Uint8Array(size),
    rewards: new Float64Array(size),
    discounts: new Float64Array(size),
    masks: new Float64Array(size),
  };
}

export function sampleReplay(replay: Replay, rng: Rng, batch: ReplayBatch): void {
  if (replay.size < 2) throw new RangeError("В буфере ещё нет пары последовательных шагов");
  const oldest = replay.size === replay.capacity ? replay.head : 0;
  for (let sample = 0; sample < batch.size; sample++) {
    // The newest entry has no successor; never pair it with the oldest entry.
    const index = (oldest + nextInt(rng, replay.size - 1)) % replay.capacity;
    const next = (index + 1) % replay.capacity;
    batch.observations.set(
      replay.observations.subarray(index * replay.width, (index + 1) * replay.width),
      sample * replay.width,
    );
    batch.nextObservations.set(
      replay.observations.subarray(next * replay.width, (next + 1) * replay.width),
      sample * replay.width,
    );
    batch.actions[sample] = replay.actions[index]!;
    batch.rewards[sample] = replay.rewards[index]!;
    batch.discounts[sample] = replay.discounts[index]!;
    batch.masks[sample] = replay.types[index] === LAST ? 0 : 1;
  }
}
