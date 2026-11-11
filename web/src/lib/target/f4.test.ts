import assert from "node:assert/strict";
import test from "node:test";
import { configure, defaults } from "./config.ts";
import { applyAction, createEpisode } from "./episode.ts";
import { createNeutralPolicy } from "./policies.ts";
import { createPerception, poolPerception, updatePerception } from "./perception.ts";
import {
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
import { formatResults } from "./f4-cli.ts";
import { checkTarget } from "./evaluate.ts";
import { parseOptions } from "./check-cli.ts";

function input(corpus: Corpus, source: (typeof sources)[number], sample: number): Float64Array {
  const inputs = sourceInputs(corpus, source);
  const output = new Float64Array(inputs.width).fill(999);
  writeInput(inputs, sample, output);
  return output;
}

test("source histories run newest first, reset by episode and concatenate exactly for schema", () => {
  const corpus = collectCorpus(defaults, 2);
  assert.equal(sourceInputs(corpus, "frames").width, 15360);
  assert.equal(sourceInputs(corpus, "schema").width, 1320);
  for (const sample of [0, 1, 29, 30, 31, 59]) {
    const step = sample % 30;
    for (const source of ["frames", "perception", "commands", "telemetry"] as const) {
      const values = input(corpus, source, sample);
      const width =
        source === "frames" ? 512 : source === "perception" ? 18 : source === "commands" ? 10 : 16;
      for (let lag = 0; lag <= step; lag++) {
        assert.deepEqual(
          values.slice(lag * width, (lag + 1) * width),
          corpus[source].slice((sample - lag) * width, (sample - lag + 1) * width),
        );
      }
      assert.ok(values.slice((step + 1) * width).every((value) => value === 0));
    }
    assert.deepEqual(
      [...input(corpus, "schema", sample)],
      [
        ...input(corpus, "perception", sample),
        ...input(corpus, "commands", sample),
        ...input(corpus, "telemetry", sample),
      ],
    );
    assert.deepEqual(
      input(corpus, "measurement", sample),
      corpus.measurement.slice(sample * 512, (sample + 1) * 512),
    );
    const context = input(corpus, "context", sample);
    assert.equal(context.length, 30);
    assert.equal(context[step], 1);
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
  assert.equal(probeData(defaults, corpus, "schema").trainSamples, 30);
  assert.throws(() => splitEpisodes(defaults, 1), /хотя бы два/);
});

test("probe scoring uses continuous centers instead of rounded class labels", () => {
  const data = { trainSamples: 1, centers: new Float64Array([9, 9, 0.3, 0.4, 1, 1]) };
  assert.deepEqual(scorePredictions(data, new Uint8Array([0, 2]), new Uint8Array([0, 2])), {
    meanError: (0.5 + Math.sqrt(2)) / 2,
    withinOne: 0.5,
  });
});

function toyData(informative: boolean, context = false): ProbeData {
  const samples = 160;
  const values = new Float64Array(samples * 4);
  const rowLabels = new Uint8Array(samples);
  const colLabels = new Uint8Array(samples);
  const centers = new Float64Array(samples * 2);
  for (let sample = 0; sample < samples; sample++) {
    const label = sample % 4;
    if (informative) values[sample * 4 + label] = 1;
    colLabels[sample] = label;
    centers[sample * 2 + 1] = label;
  }
  return {
    inputs: {
      blocks: context ? [] : [{ values, width: 4, history: false }],
      steps: 1,
      samples,
      width: context ? 1 : 4,
      context,
    },
    rowLabels,
    colLabels,
    centers,
    trainSamples: 128,
  };
}

test("the probe learns explicit class information while blank and time inputs stay at baseline", () => {
  const config = configure({ f4Hidden: 24, f4BatchSize: 16, f4Updates: 300 });
  const explicit = fitProbe(config, toyData(true), "measurement");
  assert.ok(explicit.meanError < 0.1);
  assert.ok(explicit.withinOne > 0.98);
  const blank = fitProbe(config, toyData(false), "measurement");
  const context = fitProbe(config, toyData(false, true), "context");
  assert.ok(blank.meanError >= 1);
  assert.ok(Math.abs(blank.meanError - context.meanError) <= 0.5);
});

test("worker count and completion order do not change probe results or stdout", async () => {
  const config = configure({ f4Hidden: 4, f4BatchSize: 2, f4Updates: 2 });
  const corpus = collectCorpus(config, 2);
  const serial = await runProbes(config, corpus, 1);
  const parallel = await runProbes(config, corpus, 3);
  assert.deepEqual(serial, parallel);
  assert.deepEqual(
    serial.map((result) => result.source),
    [...sources],
  );
  const options = { episodes: 2, config };
  assert.equal(formatResults(options, serial), formatResults(options, parallel.slice().reverse()));
  assert.doesNotMatch(formatResults(options, serial), /NaN|Infinity|Время|пройден/);
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
