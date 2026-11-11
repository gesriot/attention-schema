import assert from "node:assert/strict";
import test from "node:test";
import { createRng, nextInt, nextUnit } from "../rng.ts";
import { configure, defaults } from "./config.ts";
import { applyAction, createEpisode } from "./episode.ts";
import { createNeutralPolicy } from "./policies.ts";
import { createPerception, poolPerception, updatePerception } from "./perception.ts";
import {
  createProbe,
  createProbeBatch,
  forwardProbe,
  probeGradient,
  collectCorpus,
  encodeCommands,
  f4FirstSeed,
  fitProbe,
  probeData,
  scorePredictions,
  sourceInputs,
  sources,
  splitEpisodes,
  writeInput,
  type Corpus,
  type ProbeData,
} from "./f4.ts";
import { runProbes } from "./f4-workers.ts";
import { formatResults, parseF4Options } from "./f4-cli.ts";
import { checkTarget } from "./evaluate.ts";
import { parseOptions } from "./check-cli.ts";

function input(corpus: Corpus, source: (typeof sources)[number], sample: number): Float64Array {
  const inputs = sourceInputs(corpus, source);
  const output = new Float64Array(inputs.width).fill(999);
  writeInput(inputs, sample, output);
  return output;
}

test("each source contains one step and schema concatenates exactly its three inputs", () => {
  const corpus = collectCorpus(defaults, 2);
  const widths = [512, 18, 10, 16, 44, 30, 512];
  sources.forEach((source, index) =>
    assert.equal(sourceInputs(corpus, source).width, widths[index]),
  );
  for (const sample of [0, 1, 29, 30, 31, 59]) {
    for (const source of [
      "frames",
      "perception",
      "commands",
      "telemetry",
      "measurement",
    ] as const) {
      const width = sourceInputs(corpus, source).width;
      assert.deepEqual(
        input(corpus, source, sample),
        corpus[source].slice(sample * width, (sample + 1) * width),
      );
    }
    assert.deepEqual(
      [...input(corpus, "schema", sample)],
      [
        ...input(corpus, "perception", sample),
        ...input(corpus, "commands", sample),
        ...input(corpus, "telemetry", sample),
      ],
    );
    const context = input(corpus, "context", sample);
    assert.equal(context[sample % 30], 1);
    assert.equal(
      context.reduce((sum, value) => sum + value, 0),
      1,
    );
  }
  assert.deepEqual([...corpus.commands.slice(0, 10)], new Array(10).fill(0));
  assert.deepEqual([...corpus.commands.slice(300, 310)], new Array(10).fill(0));
  const initial = input(corpus, "schema", 0);
  corpus.centers.fill(-100);
  corpus.rowLabels.fill(255);
  corpus.colLabels.fill(255);
  assert.deepEqual(input(corpus, "schema", 0), initial);
});

test("corpus labels use rounded true centers and each source stores only its declared values", () => {
  const corpus = collectCorpus(defaults, 2);
  assert.deepEqual(corpus, collectCorpus(defaults, 2));
  const episode = createEpisode(defaults, f4FirstSeed);
  const policy = createNeutralPolicy(defaults, f4FirstSeed);
  let perception = createPerception();
  for (let step = 0; step < 30; step++) {
    perception = updatePerception(defaults, perception, episode.observation.measurement);
    assert.equal(corpus.rowLabels[step], Math.round(episode.truth.attention.center.row));
    assert.equal(corpus.colLabels[step], Math.round(episode.truth.attention.center.col));
    assert.equal(corpus.centers[step * 2], episode.truth.attention.center.row);
    assert.equal(corpus.centers[step * 2 + 1], episode.truth.attention.center.col);
    assert.deepEqual(corpus.frames.slice(step * 512, (step + 1) * 512), episode.truth.world.frame);
    assert.deepEqual(
      corpus.measurement.slice(step * 512, (step + 1) * 512),
      episode.observation.measurement,
    );
    assert.deepEqual(
      corpus.perception.slice(step * 18, (step + 1) * 18),
      poolPerception(defaults, perception),
    );
    assert.deepEqual(
      corpus.telemetry.slice(step * 16, (step + 1) * 16),
      episode.observation.telemetry,
    );
    assert.deepEqual(
      corpus.commands.slice(step * 10, (step + 1) * 10),
      encodeCommands(episode.observation, step === 0),
    );
    applyAction(episode, policy(episode.observation));
  }
  assert.deepEqual(splitEpisodes(defaults, 2000), { train: 1600, test: 400 });
  assert.deepEqual(splitEpisodes(defaults, 7), { train: 5, test: 2 });
  assert.equal(probeData(defaults, corpus, "schema").trainEpisodes, 1);
  assert.throws(() => splitEpisodes(defaults, 1), /хотя бы два/);
});

test("probe scoring uses continuous centers instead of rounded class labels", () => {
  const data = { trainSamples: 1, centers: new Float64Array([9, 9, 0.3, 0.4, 1, 1]) };
  assert.deepEqual(scorePredictions(data, new Uint8Array([0, 2]), new Uint8Array([0, 2])), {
    meanError: (0.5 + Math.sqrt(2)) / 2,
    withinOne: 0.5,
  });
});

test("joint row and column loss differentiates every GRU and readout parameter", () => {
  const rng = createRng(10, 22);
  const expected = createRng(10, 22);
  const model = createProbe(3, 4, 5, rng);
  for (let i = 0; i < 3 * 4 * (3 + 4) + 2 * 5 * 4; i++) nextUnit(expected);
  assert.deepEqual(rng, expected);
  assert.ok(model.biases.every((value) => value === 0));
  assert.equal(model.gru.parameters.buffer, model.parameters.buffer);
  const batch = createProbeBatch(model, 2, 5);
  batch.recurrent.inputs.set(Float64Array.from(batch.recurrent.inputs, () => nextUnit(rng) - 0.5));
  const rows = Uint8Array.from({ length: 10 }, () => nextInt(rng, 5));
  const cols = Uint8Array.from({ length: 10 }, () => nextInt(rng, 5));
  const loss = () => {
    forwardProbe(model, batch);
    return probeGradient(model, batch, rows, cols);
  };
  loss();
  const gradient = batch.gradient.slice();
  for (let index = 0; index < model.parameters.length; index++) {
    const value = model.parameters[index]!;
    model.parameters[index] = value + 1e-5;
    const plus = loss();
    model.parameters[index] = value - 1e-5;
    const minus = loss();
    model.parameters[index] = value;
    assert.ok(Math.abs(gradient[index]! - (plus - minus) / 2e-5) < 1e-8, `parameter ${index}`);
  }
  model.parameters.fill(0);
  assert.ok(Math.abs(loss() - 2 * Math.log(5)) < 1e-14);
});

function toyData(informative: boolean, context = false): ProbeData {
  const episodes = 256;
  const steps = 6;
  const samples = episodes * steps;
  const values = new Float64Array(samples * 3);
  const rowLabels = new Uint8Array(samples).fill(2);
  const colLabels = new Uint8Array(samples);
  const centers = new Float64Array(samples * 2);
  const rng = createRng(124, 0);
  for (let episode = 0; episode < episodes; episode++) {
    let position = 2;
    for (let step = 0; step < steps; step++) {
      const sample = episode * steps + step;
      const move = nextInt(rng, 3);
      position = Math.max(0, Math.min(4, position + move - 1));
      if (informative) values[sample * 3 + move] = 1;
      colLabels[sample] = position;
      centers[sample * 2] = 2;
      centers[sample * 2 + 1] = position;
    }
  }
  return {
    inputs: {
      blocks: context ? [] : [{ values, width: 3 }],
      steps,
      samples,
      width: context ? steps : 3,
      context,
    },
    rowLabels,
    colLabels,
    centers,
    trainEpisodes: 192,
  };
}

test("the recurrent probe accumulates clipped moves and blank input remains near time-only error", () => {
  // This toy uses five classes independently of the target world's geometry.
  const config = { ...defaults, fieldSize: 5, f4Hidden: 16, f4BatchSize: 16, f4Updates: 800 };
  const checkpoints: number[] = [];
  const sum = fitProbe(config, toyData(true), "commands", (progress) =>
    checkpoints.push(progress.update),
  );
  assert.ok(sum.meanError < 0.15, JSON.stringify(sum));
  assert.ok(sum.withinOne > 0.98);
  assert.ok(sum.bestError <= sum.meanError);
  assert.deepEqual(checkpoints, [500, 800]);
  const blank = fitProbe(config, toyData(false), "measurement");
  const context = fitProbe(config, toyData(false, true), "context");
  assert.ok(blank.meanError > sum.meanError + 0.5);
  assert.ok(Math.abs(blank.meanError - context.meanError) < 0.3);
});

test("source selection preserves data, result rows and stdout across worker counts and completion orders", async () => {
  const config = configure({ f4Hidden: 4, f4BatchSize: 2, f4Updates: 2 });
  const corpus = collectCorpus(config, 2);
  const subset = collectCorpus(config, 2, ["schema", "context"]);
  assert.equal(subset.frames.length, 0);
  assert.equal(subset.measurement.length, 0);
  for (const key of [
    "perception",
    "commands",
    "telemetry",
    "centers",
    "rowLabels",
    "colLabels",
  ] as const)
    assert.deepEqual(subset[key], corpus[key]);
  assert.throws(() => sourceInputs(subset, "frames"), /не сохранён/);
  const serial = await runProbes(config, corpus, 1);
  const parallel = await runProbes(config, corpus, 3);
  const selected = await runProbes(config, subset, 2, undefined, ["context", "schema"]);
  assert.deepEqual(serial, parallel);
  assert.deepEqual(
    selected,
    serial.filter((result) => ["schema", "context"].includes(result.source)),
  );
  assert.deepEqual(
    serial.map((result) => result.source),
    [...sources],
  );
  const options = { episodes: 2, config };
  assert.equal(formatResults(options, serial), formatResults(options, parallel.slice().reverse()));
  const full = formatResults(options, serial).split("\n");
  for (const line of formatResults(options, selected).trimEnd().split("\n"))
    assert.ok(full.includes(line));
  assert.doesNotMatch(formatResults(options, serial), /NaN|Infinity|Время|пройден/);
  for (const result of serial) assert.equal(result.bestError, result.meanError);
});

test("source parsing validates names and orders selected rows canonically", () => {
  assert.deepEqual(parseF4Options([]).sources, [...sources]);
  assert.deepEqual(parseF4Options(["--sources", "schema,telemetry,schema"]).sources, [
    "telemetry",
    "schema",
  ]);
  assert.equal(parseF4Options(["--episodes", "10", "--sources", "commands"]).episodes, 10);
  for (const value of ["", " ", "unknown", ",schema", "schema,", "schema,,commands"])
    assert.throws(() => parseF4Options(["--sources", value]));
  assert.throws(() => parseF4Options(["--sources"]));
  assert.throws(() => parseF4Options(["--sources", "schema", "--sources", "commands"]));
});

test("new settings validate and attention event summaries retain their exact values", () => {
  for (const entry of [
    "sigmaFloor=0",
    "gateRadius=1.5",
    "detectThreshold=-1",
    "velocityGain=2",
    "hueForget=2",
    "maxMissed=0",
    "f4Hidden=0",
    "f4BatchSize=0",
    "f4Updates=0",
    "f4TrainFraction=1",
    "minBrightness=-1",
  ]) {
    assert.throws(() => parseOptions(["--set", entry]));
  }
  const { neutral, ideal } = checkTarget(defaults, 3);
  // Attention events are independent of the measurement noise and perception.
  const result = { neutral, ideal };
  assert.deepEqual(result, {
    neutral: {
      flashes: 6,
      captureRate: 1 / 3,
      captureLength: 3,
      shiftsPerEpisode: 20 / 3,
      exitsPerEpisode: 7 / 3,
      occludedRate: 0,
      commandGap: 0.6106710542914043,
      focusedRate: 2 / 9,
      targetShare: 0.198549781848112,
      entropy: 1.7928593116302902,
      effortRate: 0.6111111111111112,
    },
    ideal: {
      flashes: 6,
      captureRate: 0,
      captureLength: null,
      shiftsPerEpisode: 16 / 3,
      exitsPerEpisode: 7 / 3,
      occludedRate: 0,
      commandGap: 0.5417439138408099,
      focusedRate: 0.7555555555555555,
      targetShare: 0.6389256188637366,
      entropy: 1.9878515446528884,
      effortRate: 0.32222222222222224,
    },
  });
});
