import assert from "node:assert/strict";
import { test } from "node:test";
import { conditionConfig } from "./dqn.ts";
import { encodeAction } from "./world.ts";
import { probeFalseSchema, shiftField, type Act } from "./false-schema.ts";

const config = conditionConfig("schema");
const size = 2000;

function schemaColumn(observation: Uint8Array): number {
  const field = observation.subarray(size - 100);
  for (let row = 0; row < 10; row++) {
    for (let col = 0; col < 10; col++) {
      if (field[row * 10 + col] === 1) return col + 1;
    }
  }
  return 4;
}

test("the shift moves only the field and refuses to leave the grid", () => {
  const source = new Uint8Array(size);
  source[size - 200 + 5] = 1;
  source[size - 100 + 12] = 1;
  source[100 + 11] = 1;
  const target = new Uint8Array(size);
  assert.equal(shiftField(source, target, 2, "history"), true);
  assert.equal(target[size - 200 + 5], 1);
  assert.equal(target[size - 100 + 14], 1);
  assert.equal(target[size - 100 + 12], 0);
  assert.equal(target[100 + 13], 1);
  assert.equal(shiftField(source, target, 2, "latest"), true);
  assert.equal(target[100 + 11], 1);
  assert.equal(target[100 + 13], 0);
  assert.equal(shiftField(source, target, -3, "history"), false);
});

test("a policy that ignores the field gives no directional effect", () => {
  const act: Act = () => 0;
  for (const row of probeFalseSchema(act, config, 20).shifts) {
    assert.equal(row.direction, 0);
    assert.equal(row.changed, 0);
  }
});

test("a policy steered by the field moves against the shift on both sides", () => {
  // Heads for column 4 from wherever the field says the window is.
  const act: Act = (observation) => {
    const error = 4 - schemaColumn(observation);
    const attention = error > 0 ? 1 : error < 0 ? 7 : 0;
    return encodeAction(config, { paddle: 0, attention, resource: 0 });
  };
  const probe = probeFalseSchema(act, config, 50);
  for (const row of probe.shifts) {
    assert.ok(row.decisions > 0);
    assert.ok(row.direction > 0, `${row.scope} ${row.shift}: ${row.direction}`);
  }
  const shift = probe.consequences.find((row) => row.shift === 2)!;
  assert.ok(shift.episodes > 0);
});
