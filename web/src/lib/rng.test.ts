import assert from "node:assert/strict";
import test from "node:test";
import { createRng, nextInt, nextUnit } from "./rng.ts";

test("the generator reproduces reference states and first values", () => {
  const cases = [
    {
      seed: 0,
      stream: 0,
      state: 2462723854,
      values: [2673918951, 1525398398, 1851933717, 1087923286, 744930619, 2388801345],
    },
    {
      seed: 1,
      stream: 0,
      state: 920564995,
      values: [2243319414, 3892457524, 3774914640, 1708185691, 4150724432, 802420870],
    },
    {
      seed: 1,
      stream: 1,
      state: 314344336,
      values: [468467039, 2307939659, 328510042, 2430640151, 1403280415, 1249085783],
    },
    {
      seed: 1,
      stream: 2,
      state: 697614773,
      values: [2200402435, 3618857721, 1214982740, 224665642, 1793123673, 2989738937],
    },
    {
      seed: 0xffffffff,
      stream: 0xffffffff,
      state: 2180083513,
      values: [146612443, 1647005904, 128168844, 1578492502, 3066638271, 4111147734],
    },
  ];
  for (const expected of cases) {
    const rng = createRng(expected.seed, expected.stream);
    assert.equal(rng.state, expected.state);
    for (const value of expected.values) {
      const state = rng.state;
      assert.equal(nextUnit(rng), value / 4294967296);
      assert.equal(rng.state, (state + 0x6d2b79f5) >>> 0);
    }
  }
});

test("draws from one stream leave another stream unchanged", () => {
  const first = createRng(71, 0);
  const reference = createRng(71, 0);
  const other = createRng(71, 1);
  const firstValues: number[] = [];
  const otherValues: number[] = [];
  for (let i = 0; i < 100; i++) {
    const value = nextUnit(first);
    firstValues.push(value);
    otherValues.push(nextUnit(other));
    for (let j = 0; j < i; j++) nextUnit(other);
    assert.equal(value, nextUnit(reference));
  }
  assert.notDeepEqual(firstValues, otherValues);
});

test("integer draws use one step and round down", () => {
  for (const n of [1, 3, 8, 10, 100, 4294967296]) {
    const integers = createRng(42, 2);
    const units = createRng(42, 2);
    for (let i = 0; i < 1000; i++) {
      const unit = nextUnit(units);
      const integer = nextInt(integers, n);
      assert.ok(unit >= 0 && unit < 1);
      assert.equal(integer, Math.floor(unit * n));
      assert.ok(integer >= 0 && integer < n);
      assert.equal(integers.state, units.state);
    }
  }
});
