import assert from "node:assert/strict";
import test from "node:test";
import { createRng } from "../rng.ts";
import {
  createReplay,
  createReplayBatch,
  FIRST,
  LAST,
  MID,
  pushReplay,
  sampleReplay,
} from "./replay.ts";

test("replay stores compact copies and requires a successor before sampling", () => {
  const replay = createReplay(3, 2);
  const observation = new Uint8Array([1, 2]);
  const batch = createReplayBatch(20, 2);
  pushReplay(replay, observation, FIRST, 3, 0.5, 0.4);
  observation.fill(9);
  assert.deepEqual([...replay.observations.slice(0, 2)], [1, 2]);
  assert.throws(() => sampleReplay(replay, createRng(1, 5), batch), /пары/);
  pushReplay(replay, observation, MID, 4, -0.5, 0.4);
  sampleReplay(replay, createRng(1, 5), batch);
  for (let i = 0; i < 20; i++) {
    assert.deepEqual([...batch.observations.slice(i * 2, i * 2 + 2)], [1, 2]);
    assert.deepEqual([...batch.nextObservations.slice(i * 2, i * 2 + 2)], [9, 9]);
    assert.equal(batch.actions[i], 3);
    assert.equal(batch.rewards[i], 0.5);
    assert.equal(batch.discounts[i], 0.4);
  }
});

test("replay evicts oldest steps, includes boundaries, and never crosses the write head", () => {
  const replay = createReplay(6, 1);
  for (let i = 0; i <= 10; i++) {
    pushReplay(
      replay,
      new Uint8Array([i]),
      i === 8 ? LAST : i === 9 ? FIRST : MID,
      i,
      i + 0.5,
      i === 7 ? 0 : 0.4,
    );
  }
  assert.equal(replay.size, 6);
  assert.equal(replay.head, 5);
  const batch = createReplayBatch(1000, 1);
  sampleReplay(replay, createRng(2, 5), batch);
  const seen = new Set<number>();
  for (let i = 0; i < batch.size; i++) {
    const current = batch.observations[i]!;
    seen.add(current);
    assert.ok(current >= 5 && current <= 9);
    assert.equal(batch.nextObservations[i], current + 1);
    assert.equal(batch.actions[i], current);
    assert.equal(batch.rewards[i], current + 0.5);
    assert.equal(batch.discounts[i], current === 7 ? 0 : 0.4);
    assert.equal(batch.masks[i], current === 8 ? 0 : 1);
  }
  assert.deepEqual([...seen].sort(), [5, 6, 7, 8, 9]);
  const repeat = createReplayBatch(1000, 1);
  sampleReplay(replay, createRng(2, 5), repeat);
  assert.deepEqual(batch, repeat);
});
