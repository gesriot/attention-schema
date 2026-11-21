import assert from "node:assert/strict";
import test from "node:test";
import { configure, defaults, firstSeed } from "./config.ts";
import { encodeC6 } from "./controller.ts";
import { applyAction, createEpisode } from "./episode.ts";
import { handC6Action, resourceCenter, type HandC6Input } from "./hand-c6.ts";
import { createPerception, poolPerception, updatePerception } from "./perception.ts";
import { checkHandC6 } from "./perception-controls.ts";
import { createNeutralPolicy } from "./policies.ts";

function pointResource(row: number, col: number): Float64Array {
  const resource = new Float64Array(defaults.fieldSize * defaults.fieldSize);
  resource[row * defaults.fieldSize + col] = 1;
  return resource;
}

test("hand C6 rounds the forecast before clamping and rounds the paddle target independently", () => {
  const input: HandC6Input = {
    perception: new Float64Array(18),
    paddle: 8,
    resource: pointResource(4, 6),
  };
  input.perception.set([1, 7 / 15, 1 / 15, -0.25, 0.75, 0.99]);
  assert.deepEqual(handC6Action(defaults, input, true), {
    command: { row: 1, col: -1 },
    effort: 0,
    paddleMove: -1,
    answer: 3,
  });
  assert.equal(handC6Action(defaults, input, false).answer, undefined);

  input.perception.set([1, 4 / 15, 6 / 15, -0.5, 0.5, 0.5]);
  input.paddle = 6;
  assert.deepEqual(handC6Action(defaults, input, true), {
    command: { row: -0, col: 1 },
    effort: 0,
    paddleMove: 0,
    answer: 2,
  });

  input.perception.set([1, 4 / 15, 6.5 / 15, 0, -1, 0.25]);
  assert.deepEqual(handC6Action(defaults, input, true), {
    command: { row: 0, col: -0 },
    effort: 0,
    paddleMove: 1,
    answer: 1,
  });
  input.perception[2] = 6.49 / 15;
  assert.equal(handC6Action(defaults, input, true).paddleMove, 0);
});

test("hand C6 uses pooled zeros for an inactive slot without a fallback to slot one", () => {
  const state = createPerception();
  Object.assign(state.slots[0], { row: 15, col: 15, hue: 0.8, velocityRow: 1, velocityCol: 1 });
  Object.assign(state.slots[1], { active: true, row: 10, col: 12, hue: 0.9 });
  const perception = poolPerception(defaults, state);
  assert.deepEqual(perception.slice(0, 9), new Float64Array(9));
  assert.deepEqual(
    handC6Action(
      defaults,
      {
        perception,
        paddle: 7,
        resource: pointResource(5, 7),
      },
      true,
    ),
    {
      command: { row: -1, col: -1 },
      effort: 0,
      paddleMove: -1,
      answer: 0,
    },
  );
});

test("hand C6 accepts only the pooled perception, paddle and resource shared with C6", () => {
  const first = createEpisode(defaults, 17);
  const second = createEpisode(defaults, 18);
  assert.notDeepEqual(first.truth.world, second.truth.world);
  const perception = updatePerception(defaults, createPerception(), first.observation.measurement);
  second.observation.paddle = first.observation.paddle;
  second.truth.attention.resource = first.truth.attention.resource.slice();
  second.truth.attention.center = { row: -100, col: 100 };
  const input = (episode: typeof first): HandC6Input => ({
    perception: poolPerception(defaults, perception),
    paddle: episode.observation.paddle,
    resource: episode.truth.attention.resource,
  });
  const values = input(first);
  const before = structuredClone(values);
  const encoded = encodeC6(defaults, perception, first.observation, values.resource, true);
  assert.deepEqual(Object.keys(values).sort(), ["paddle", "perception", "resource"]);
  assert.deepEqual(values.perception, encoded.slice(0, 18));
  assert.deepEqual(values.resource, encoded.slice(32));
  assert.equal(
    encoded[31],
    (values.paddle - defaults.paddleMin) / (defaults.paddleMax - defaults.paddleMin),
  );
  const action = handC6Action(defaults, values, true);
  assert.deepEqual(action, handC6Action(defaults, input(second), true));
  assert.deepEqual(values, before);

  // Once the three input fields are fixed, none of these fields may be accessed.
  for (const name of ["measurement", "telemetry"] as const) {
    Object.defineProperty(second.observation, name, {
      get() {
        throw new Error("Forbidden input");
      },
    });
  }
  Object.defineProperty(second.truth, "world", {
    get() {
      throw new Error("Forbidden truth");
    },
  });
  Object.defineProperty(second.truth.attention, "center", {
    get() {
      throw new Error("Forbidden center");
    },
  });
  assert.deepEqual(action, handC6Action(defaults, input(second), true));
  const checkSignature = () => {
    // @ts-expect-error Raw observations are not inputs of the hand-written policy.
    handC6Action(defaults, first.observation, true);
    // @ts-expect-error Evaluator truth is not accepted as an additional argument.
    handC6Action(defaults, values, true, first.truth);
  };
  void checkSignature;
});

test("resource moments reproduce the mechanism center bit for bit across episode traces", () => {
  const map = new Float64Array([1, 2, 3, 4]);
  assert.deepEqual(resourceCenter({ fieldSize: 2 }, map), { row: 7 / 10, col: 6 / 10 });
  for (let index = 0; index < 8; index++) {
    const episode = createEpisode(defaults, firstSeed + index);
    const policy = createNeutralPolicy(defaults, firstSeed + index);
    for (let step = 1; step <= defaults.episodeSteps; step++) {
      assert.deepEqual(
        resourceCenter(defaults, episode.truth.attention.resource),
        episode.truth.attention.center,
      );
      applyAction(episode, policy(episode.observation));
    }
  }
});

test("hand C6 evaluation uses control seeds and all observed frames before applying actions", () => {
  const config = configure({ episodeSteps: 5, hueJumpMin: 2, hueJumpMax: 5 });
  const episodes = 3;
  let caught = 0,
    correct = 0,
    focused = 0;
  for (let index = 0; index < episodes; index++) {
    const episode = createEpisode(config, firstSeed + index);
    let perception = createPerception();
    for (let step = 0; step < config.episodeSteps; step++) {
      perception = updatePerception(config, perception, episode.observation.measurement);
      const encoded = encodeC6(
        config,
        perception,
        episode.observation,
        episode.truth.attention.resource,
        step === 0,
      );
      const { target } = episode.truth.world;
      const { center } = episode.truth.attention;
      focused += Number(
        Math.abs(center.row - target.row) <= 1 && Math.abs(center.col - target.col) <= 1,
      );
      applyAction(
        episode,
        handC6Action(
          config,
          {
            perception: encoded.slice(0, 18),
            paddle: episode.observation.paddle,
            resource: encoded.slice(32),
          },
          step + 1 === config.episodeSteps,
        ),
      );
    }
    caught += Number(episode.score!.caught);
    correct += Number(episode.score!.correct);
    assert.equal(episode.score!.effortSteps, 0);
  }
  assert.deepEqual(checkHandC6(config, episodes), {
    correctRate: correct / episodes,
    catchRate: caught / episodes,
    focusRate: focused / (episodes * config.episodeSteps),
  });
});
