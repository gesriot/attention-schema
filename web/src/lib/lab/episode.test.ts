import assert from "node:assert/strict";
import test from "node:test";
import { createRng } from "../rng.ts";
import { createMlp } from "../nn/mlp.ts";
import {
  appendFrame,
  clearObservationField,
  conditionConfig,
  evaluatePolicy,
} from "../level0/dqn.ts";
import { modelPolicy, shiftField } from "../level0/false-schema.ts";
import { createWorld, stepWorld } from "../level0/world.ts";
import { advanceLabEpisode, createLabEpisode, defaultSettings, labConditions } from "./episode.ts";

test("laboratory traces and greedy actions reproduce all three evaluator checks", () => {
  const model = createMlp([2000, 5, 16], createRng(9, 3));
  for (const condition of labConditions)
    for (const check of ["normal", "windowOff", "fieldOff"] as const) {
      let tracking = 0,
        catching = 0;
      for (let index = 1; index <= 4; index++) {
        const seed = 4000000000 + index;
        const lab = createLabEpisode(
          model,
          { ...defaultSettings, condition, intervention: check },
          seed,
        );
        const world = createWorld(
          { ...conditionConfig(condition), windowActive: check !== "windowOff" },
          seed,
        );
        const act = modelPolicy(model);
        const observation = new Uint8Array(2000);
        appendFrame(observation, world.frame, true);
        while (world.t < 10) {
          assert.deepEqual(lab.world, world);
          if (check === "fieldOff") clearObservationField(observation);
          assert.deepEqual(lab.input, observation);
          const action = act(observation);
          assert.equal(lab.action, action);
          stepWorld(world, action);
          appendFrame(observation, world.frame, false);
          advanceLabEpisode(lab);
        }
        assert.deepEqual(lab.world, world);
        assert.equal(lab.action, null);
        assert.throws(() => advanceLabEpisode(lab), /завершён/);
        tracking += lab.world.trackingScore;
        catching += lab.world.catchScore;
      }
      assert.deepEqual(
        { tracking: tracking / 4, catching: catching / 4 },
        evaluatePolicy(model, conditionConfig(condition), check, 4, 1),
      );
    }
});

test("history substitution changes only the schema input and never mutates the world or stored history", () => {
  const model = createMlp([2000, 3, 16], createRng(1, 3));
  // Constant greedy action keeps both trajectories paired after the one-step substitution.
  model.parameters.fill(0);
  let applied = 0,
    outside = 0;
  for (let seed = 1; seed <= 12; seed++)
    for (const shift of [-3, -1, 0, 1, 3]) {
      const clean = createLabEpisode(model, defaultSettings, seed);
      const lied = createLabEpisode(
        model,
        { ...defaultSettings, intervention: "falseSchema", falseStep: 5, falseShift: shift },
        seed,
      );
      for (let step = 1; step <= 10; step++) {
        assert.deepEqual(lied.world, clean.world);
        assert.deepEqual(lied.observation, clean.observation);
        if (step === 5) {
          const expected = new Uint8Array(2000);
          if (shiftField(clean.input, expected, shift, "history")) {
            applied++;
            assert.equal(lied.falseStatus, "applied");
            assert.deepEqual(lied.input, expected);
            for (let i = 0; i < 2000; i++)
              if (i % 200 < 100) assert.equal(lied.input[i], clean.input[i]);
          } else {
            outside++;
            assert.equal(lied.falseStatus, "outside");
            assert.deepEqual(lied.input, clean.input);
          }
        } else assert.deepEqual(lied.input, clean.input);
        if (step < 10) {
          advanceLabEpisode(clean);
          advanceLabEpisode(lied);
        }
      }
    }
  assert.ok(applied > 0 && outside > 0);
});
