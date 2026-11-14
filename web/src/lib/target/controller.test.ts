import assert from "node:assert/strict";
import { test } from "node:test";
import { createRng, nextUnit } from "../rng.ts";
import { createAdam, adamStep } from "../nn/adam.ts";
import { configure, streams } from "./config.ts";
import { createEpisode, applyAction, type Action } from "./episode.ts";
import { createPerception, poolPerception, updatePerception } from "./perception.ts";
import {
  advanceController,
  c6InputWidth,
  controllerAction,
  createController,
  createControllerBatch,
  encodeC6,
  forwardController,
} from "./controller.ts";
import {
  clipGradient,
  collectC6,
  computeAdvantages,
  createRollout,
  optimizePpo,
  ppoGradient,
  shuffleEpisodes,
  stepReward,
  trainingSeed,
  trainC6,
} from "./ppo.ts";

const close = (actual: number, expected: number, tolerance = 1e-10) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
const smallConfig = () =>
  configure({
    controllerHidden: 4,
    episodeSteps: 3,
    hueJumpMin: 2,
    hueJumpMax: 3,
    ppoEpisodes: 2,
    ppoMinibatch: 1,
    ppoEpochs: 2,
    ppoUpdates: 2,
    ppoEvalEvery: 2,
    ppoEvalEpisodes: 3,
  });

test("Adam preserves legacy bits and scales the parameter displacement", () => {
  const initial = new Float64Array([0.1, -2, 0, 4]);
  const actual = initial.slice();
  const expected = initial.slice();
  const explicit = initial.slice();
  const a = createAdam(4),
    b = createAdam(4),
    c = createAdam(4);
  for (let step = 1; step <= 12; step++) {
    const gradient = new Float64Array([0.2 * step, -0.3, 0, 1 / step]);
    b.step++;
    const rate = (0.001 * Math.sqrt(1 - Math.pow(0.999, b.step))) / (1 - Math.pow(0.9, b.step));
    for (let i = 0; i < 4; i++) {
      const g = gradient[i]!;
      const m = 0.9 * b.m[i]! + 0.1 * g;
      const v = 0.999 * b.v[i]! + 0.001 * g * g;
      b.m[i] = m;
      b.v[i] = v;
      expected[i] = expected[i]! - (rate * m) / (Math.sqrt(v) + 1e-7);
    }
    adamStep(actual, gradient, a);
    adamStep(explicit, gradient, c, 0.001);
  }
  assert.deepEqual(actual, expected);
  assert.deepEqual(actual, explicit);
  assert.deepEqual(a, b);
  const first = new Float64Array(4),
    second = new Float64Array(4);
  const g = new Float64Array([1, -1, 0, 0.001]);
  adamStep(first, g, createAdam(4));
  adamStep(second, g, createAdam(4), 0.004);
  assert.deepEqual(
    second,
    first.map((value) => value * 4),
  );
});

test("C6 encodes only declared inputs with the resource map at offset 32", () => {
  const config = configure();
  const episode = createEpisode(config, 42);
  const perception = updatePerception(config, createPerception(), episode.observation.measurement);
  const observation = {
    ...episode.observation,
    previousCommand: { row: 1, col: -1 },
    previousEffort: 1,
    previousPaddleMove: -1,
    paddle: 10,
  };
  const resource = Float64Array.from({ length: 256 }, (_, i) => i / 256);
  const encoded = encodeC6(config, perception, observation, resource, false);
  assert.equal(encoded.length, 288);
  assert.deepEqual(encoded.slice(0, 18), poolPerception(config, perception));
  assert.deepEqual([...encoded.slice(18, 31)], [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0]);
  assert.equal(encoded[31], 9 / 13);
  assert.deepEqual(encoded.slice(32), resource);
  assert.deepEqual(
    encodeC6(config, perception, observation, resource, true).slice(18, 31),
    new Float64Array(13),
  );
  const other = createEpisode(config, 99);
  other.observation = {
    ...observation,
    measurement: new Float64Array(0),
    telemetry: new Float64Array([Infinity]),
  };
  other.truth.attention.resource = resource;
  assert.notDeepEqual(other.truth.world, episode.truth.world);
  assert.deepEqual(
    encodeC6(config, perception, other.observation, other.truth.attention.resource, false),
    encoded,
  );
  assert.equal(
    encodeC6(
      configure({ paddleMin: 7, paddleMax: 7 }),
      perception,
      observation,
      resource,
      false,
    )[31],
    0,
  );
});

test("online GRU matches whole-sequence GRU bit for bit and resets initial states", () => {
  const rng = createRng(3, 25);
  const model = createController(5, 4, 2, rng);
  const online = createControllerBatch(model, 2, 3),
    full = createControllerBatch(model, 2, 3);
  for (let i = 0; i < full.recurrent.inputs.length; i++)
    full.recurrent.inputs[i] = i % 3 ? nextUnit(rng) - 0.5 : 0;
  online.recurrent.inputs.set(full.recurrent.inputs);
  for (let step = 0; step < 3; step++) advanceController(model, online, step);
  full.recurrent.states.fill(17);
  forwardController(model, full);
  for (const name of [
    "states",
    "reset",
    "update",
    "candidate",
    "resetState",
    "counts",
    "indices",
  ] as const)
    assert.deepEqual(online.recurrent[name], full.recurrent[name]);
  assert.deepEqual(online.logits, full.logits);
  full.recurrent.inputs.fill(0);
  forwardController(model, full);
  assert.deepEqual(full.recurrent.states, new Float64Array(full.recurrent.states.length));
});

test("head initialization, sampling and shuffles consume exactly the specified draws", () => {
  const rng = createRng(1, streams.controller),
    expected = createRng(1, 25);
  const model = createController(5, 4, 2, rng);
  const weightCount = 3 * 4 * (5 + 4) + (9 + 2 + 3 + 2 + 1) * 4;
  for (let i = 0; i < weightCount; i++) nextUnit(expected);
  assert.deepEqual(rng, expected);
  let offset = model.gru.parameters.length;
  for (const head of model.heads) {
    assert.equal(head.offset, offset);
    assert.deepEqual(head.biases, new Float64Array(head.width));
    offset += head.width * 5;
  }
  assert.equal(offset, model.parameters.length);
  const actions = createRng(1, streams.controllerAction),
    actionExpected = createRng(1, 26);
  const logits = new Float64Array(model.outputs);
  const nonterminal = controllerAction(model, logits, 0, false, actions);
  for (let i = 0; i < 3; i++) nextUnit(actionExpected);
  assert.deepEqual(actions, actionExpected);
  close(nonterminal.logProb, -Math.log(9) - Math.log(2) - Math.log(3));
  close(nonterminal.entropy, Math.log(9) + Math.log(2) + Math.log(3));
  assert.equal(nonterminal.action.answer, undefined);
  const terminal = controllerAction(model, logits, 0, true, actions);
  for (let i = 0; i < 4; i++) nextUnit(actionExpected);
  assert.deepEqual(actions, actionExpected);
  close(terminal.logProb, nonterminal.logProb - Math.log(2));
  close(terminal.entropy, nonterminal.entropy + Math.log(2));
  const greedy = controllerAction(model, logits, 0, true);
  assert.deepEqual([...greedy.choices], [0, 0, 0, 0]);
  logits[model.heads[3]!.output] = 100;
  const ignored = controllerAction(model, logits, 0, false);
  assert.equal(ignored.logProb, nonterminal.logProb);
  assert.equal(ignored.entropy, nonterminal.entropy);
  const shuffled = shuffleEpisodes(7, rng);
  for (let i = 0; i < 6; i++) nextUnit(expected);
  assert.deepEqual(rng, expected);
  assert.deepEqual(
    [...shuffled].sort((a, b) => a - b),
    [0, 1, 2, 3, 4, 5, 6],
  );
});

test("GAE uses terminal zero value, raw targets and population normalization", () => {
  const data = createRollout(2, 3, 1);
  data.values.set([1, 2, 3, -1, 0, 1]);
  data.rewards.set([0, 0, 4, 1, 2, -1]);
  computeAdvantages(data, 0.5);
  const raw = [1.75, 1.5, 1, 3, 2, -2];
  assert.deepEqual([...data.targets], [2.75, 3.5, 4, 2, 2, -1]);
  const mean = raw.reduce((a, b) => a + b) / 6;
  const std = Math.sqrt(raw.reduce((sum, x) => sum + (x - mean) ** 2, 0) / 6);
  raw.forEach((value, i) => close(data.advantages[i]!, (value - mean) / (std + 1e-8)));
});

test("step rewards sum to the independent episode score including final effort", () => {
  const config = configure();
  const outcomes = new Set<string>();
  for (let seed = 1; seed <= 60; seed++) {
    const episode = createEpisode(config, seed);
    let reward = 0;
    for (let step = 0; step < config.episodeSteps; step++) {
      const action: Action = {
        command: { row: 0, col: 0 },
        paddleMove: 0,
        effort: step % 2 === 1 ? 1 : 0,
        answer: seed % 4,
      };
      applyAction(episode, action);
      reward += stepReward(config, action, episode.score);
    }
    close(reward, episode.score!.total);
    assert.equal(episode.score!.effortSteps, 15);
    outcomes.add(`${episode.score!.caught}/${episode.score!.correct}`);
  }
  assert.equal(outcomes.size, 4);
});

test("every PPO parameter matches central differences inside and outside clipping", () => {
  const config = configure();
  const rng = createRng(18, 25);
  const model = createController(5, 4, 3, rng);
  const batch = createControllerBatch(model, 2, 3);
  const data = createRollout(2, 3, 5);
  for (let i = 0; i < data.inputs.length; i++) data.inputs[i] = nextUnit(rng) * 2 - 1;
  batch.recurrent.inputs.set(data.inputs);
  forwardController(model, batch);
  const ratios = [1, 1.6, 0.5, 1.5, 0.4, 1];
  data.advantages.set([0.7, 0.4, -0.9, -0.2, 0.3, -0.6]);
  data.targets.set([0.1, -0.3, 0.7, -0.6, 1.2, 0.4]);
  for (let sample = 0; sample < 6; sample++) {
    const chosen = controllerAction(model, batch.logits, sample, sample % 3 === 2, rng);
    data.actions.set(chosen.choices, sample * 4);
    data.logProbs[sample] = chosen.logProb - Math.log(ratios[sample]!);
  }
  ppoGradient(config, model, batch, data, [0, 1]);
  const analytic = batch.gradient.slice();
  const epsilon = 1e-5;
  for (let i = 0; i < model.parameters.length; i++) {
    const original = model.parameters[i]!;
    model.parameters[i] = original + epsilon;
    const plus = ppoGradient(config, model, batch, data, [0, 1]);
    model.parameters[i] = original - epsilon;
    const minus = ppoGradient(config, model, batch, data, [0, 1]);
    model.parameters[i] = original;
    close(analytic[i]!, (plus - minus) / (2 * epsilon), 2e-7);
  }
  for (const head of model.heads)
    assert.ok(
      analytic.slice(head.offset, head.offset + head.width * 5).some((x) => Math.abs(x) > 1e-6),
    );
  data.logProbs[0] = NaN;
  assert.throws(() => ppoGradient(config, model, batch, data, [0, 1]), /Неконечные потери/);
});

test("global gradient clipping preserves direction and rejects nonfinite values", () => {
  const values = new Float64Array([3, 4, 0]);
  assert.equal(clipGradient(values, 2), 5);
  close(values[0]!, 1.2);
  close(values[1]!, 1.6);
  assert.equal(values[2], 0);
  const before = values.slice();
  clipGradient(values, 3);
  assert.deepEqual(values, before);
  assert.throws(() => clipGradient(new Float64Array([Infinity]), 1), /Неконечная/);
});

test("training seeds, rollout inputs and RNG counts follow the collection schedule", () => {
  const config = smallConfig();
  assert.equal(trainingSeed(config, 1, 1, 0), 7000000001);
  assert.equal(trainingSeed(config, 2, 2, 1), 7010000004);
  assert.equal(
    trainingSeed(configure({ ppoUpdates: 100000, ppoEpisodes: 100 }), 99, 100000, 99),
    7990000000,
  );
  assert.throws(() => trainingSeed(config, 100, 1, 0));
  assert.throws(() => configure({ ppoUpdates: 10000001 }));
  const model = createController(c6InputWidth(config), 4, 4, createRng(1, 25));
  const rng = createRng(1, 26),
    expected = createRng(1, 26);
  const data = collectC6(config, model, 1, 1, rng);
  for (let i = 0; i < config.ppoEpisodes * (3 * config.episodeSteps + 1); i++) nextUnit(expected);
  assert.deepEqual(rng, expected);
  const episode = createEpisode(config, trainingSeed(config, 1, 1, 0));
  const perception = updatePerception(config, createPerception(), episode.observation.measurement);
  assert.deepEqual(
    data.inputs.slice(0, data.input),
    encodeC6(config, perception, episode.observation, episode.truth.attention.resource, true),
  );
  const batch = createControllerBatch(model, data.size, data.steps);
  batch.recurrent.inputs.set(data.inputs);
  forwardController(model, batch);
  assert.deepEqual(
    data.values,
    Float64Array.from(
      { length: data.size * data.steps },
      (_, i) => batch.logits[i * model.outputs + model.heads[4]!.output]!,
    ),
  );
  const replay = createRng(1, 26);
  for (let step = 0; step < data.steps; step++) {
    for (let i = 0; i < data.size; i++) {
      const sample = i * data.steps + step;
      const chosen = controllerAction(model, batch.logits, sample, step + 1 === data.steps, replay);
      assert.deepEqual(data.actions.slice(sample * 4, sample * 4 + 4), chosen.choices);
      assert.equal(data.logProbs[sample], chosen.logProb);
    }
  }
  const second = collectC6(config, model, 1, 1, createRng(1, 26));
  assert.deepEqual(data, second);
  const first = trainC6(config, 1, 0.001);
  assert.deepEqual(first, trainC6(config, 1, 0.001));
  assert.deepEqual(
    first.checkpoints.map((x) => x.update),
    [0, 2],
  );
  assert.deepEqual(
    trainC6({ ...config, ppoUpdates: 3 }, 1, 0.001).checkpoints.map((x) => x.update),
    [0, 2, 3],
  );
  assert.deepEqual(
    first.checkpoints[0],
    trainC6({ ...config, ppoMinibatch: 2 }, 1, 0.001).checkpoints[0],
  );
});

test("PPO clipping objective and policy signs match a uniform categorical example", () => {
  const model = createController(1, 1, 2, createRng(1, 25));
  model.parameters.fill(0);
  const data = createRollout(1, 1, 1);
  data.actions.fill(0);
  const batch = createControllerBatch(model, 1, 1);
  const config = { ppoClip: 0.2, ppoValueCoef: 0.5, ppoEntropyCoef: 0.01 };
  const entropy = Math.log(9) + Math.log(2) + Math.log(3) + Math.log(2);
  data.targets[0] = 2;
  for (const [ratio, advantage, objective, derivative] of [
    [1, 2, -2, -2],
    [1.5, 2, -2.4, 0],
    [0.5, -2, 1.6, 0],
    [1.5, -2, 3, 3],
    [0.5, 2, -1, -1],
  ]) {
    data.advantages[0] = advantage!;
    data.logProbs[0] = -entropy - Math.log(ratio!);
    close(ppoGradient(config, model, batch, data, [0]), objective! + 2 - 0.01 * entropy);
    const command = model.heads[0]!;
    close(batch.gradient[command.offset + 9]!, (derivative! * 8) / 9);
    close(batch.gradient[model.heads[4]!.offset + 1]!, -2);
  }
});

test("PPO updates preserve parameters and optimizer bits with a short last minibatch", () => {
  const config = { ...smallConfig(), ppoEpisodes: 3, ppoMinibatch: 2 };
  const run = () => {
    const rng = createRng(2, 25);
    const model = createController(
      c6InputWidth(config),
      config.controllerHidden,
      config.hueCategories,
      rng,
    );
    const adam = createAdam(model.parameters.length);
    const actions = createRng(2, 26);
    const data = collectC6(config, model, 2, 1, actions);
    const before = model.parameters.slice();
    const expected = { ...rng };
    for (let i = 0; i < config.ppoEpochs * (config.ppoEpisodes - 1); i++) nextUnit(expected);
    optimizePpo(config, model, data, adam, 0.001, rng);
    assert.deepEqual(rng, expected);
    assert.equal(adam.step, config.ppoEpochs * 2);
    assert.notDeepEqual(model.parameters, before);
    return { parameters: model.parameters, adam, rng, actions };
  };
  assert.deepEqual(run(), run());
});

test("PPO learns a delayed binary cue using recurrent memory", () => {
  const config = configure({ ppoEpisodes: 32, ppoMinibatch: 16, ppoEpochs: 3 });
  const rng = createRng(7, 25),
    actionRng = createRng(7, 26);
  const model = createController(2, 8, 2, rng);
  const adam = createAdam(model.parameters.length);
  for (let update = 0; update < 90; update++) {
    const data = createRollout(32, 3, 2);
    const batch = createControllerBatch(model, 32, 3);
    for (let i = 0; i < 32; i++) {
      data.inputs[i * 6] = i % 2 ? 1 : -1;
      data.inputs[i * 6 + 1] = 1;
    }
    batch.recurrent.inputs.set(data.inputs);
    for (let step = 0; step < 3; step++) {
      advanceController(model, batch, step);
      for (let i = 0; i < 32; i++) {
        const sample = i * 3 + step;
        const chosen = controllerAction(model, batch.logits, sample, step === 2, actionRng);
        data.actions.set(chosen.choices, sample * 4);
        data.logProbs[sample] = chosen.logProb;
        data.values[sample] = batch.logits[sample * model.outputs + model.heads[4]!.output]!;
        if (step === 2) data.rewards[sample] = chosen.action.answer === i % 2 ? 1 : -1;
      }
    }
    computeAdvantages(data, config.ppoLambda);
    optimizePpo(config, model, data, adam, 0.01, rng);
  }
  const batch = createControllerBatch(model, 2, 3);
  batch.recurrent.inputs.set([-1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]);
  forwardController(model, batch);
  let correct = 0;
  for (let i = 0; i < 2; i++) {
    for (let trial = 0; trial < 200; trial++) {
      const chosen = controllerAction(model, batch.logits, i * 3 + 2, true, actionRng);
      correct += Number(chosen.action.answer === i);
    }
  }
  assert.ok(correct / 400 >= 0.95, `Delayed cue accuracy: ${correct / 400}`);
  // Removing the first cue makes the final inputs identical and destroys discrimination.
  batch.recurrent.inputs.fill(0);
  forwardController(model, batch);
  assert.equal(
    controllerAction(model, batch.logits, 2, true).action.answer,
    controllerAction(model, batch.logits, 5, true).action.answer,
  );
});
