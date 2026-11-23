import assert from "node:assert/strict";
import test from "node:test";
import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createBatch, forward } from "../nn/mlp.ts";
import { appendFrame, conditionConfig, trainSubject } from "../level0/dqn.ts";
import { formatResults, parseOptions, runJobs } from "../level0/train-cli.ts";
import { createWorld, stepWorld } from "../level0/world.ts";
import { decodeModel, encodeParameters } from "./model.ts";

test("training stdout without export matches the recorded pre-export run", async () => {
  const cli = new URL("../level0/train-cli.ts", import.meta.url);
  const { stdout } = await promisify(execFile)(process.execPath, [
    cli.pathname,
    "--subjects",
    "1",
    "--iterations",
    "2",
    "--eval-every",
    "1",
    "--workers",
    "3",
    "--conditions",
    "schema,commands",
  ]);
  assert.equal(
    createHash("sha256").update(stdout).digest("hex"),
    "842c44bb779e91d4d4db28ed1a925f8d11faf8ee9d0a674487ae3a93e85735d0",
  );
});

test("exported models retain exact network outputs and cached summaries cannot replace weights", async () => {
  const dir = await mkdtemp(join(tmpdir(), "lab-model-"));
  try {
    const config = {
      sizes: [2000, 3, 16],
      collectSteps: 12,
      updates: 2,
      batchSize: 4,
      evalEpisodes: 2,
    };
    const args = [
      "--subjects",
      "1",
      "--iterations",
      "2",
      "--eval-every",
      "1",
      "--conditions",
      "schema,none,commands",
      "--workers",
      "3",
    ];
    const plain = await runJobs(parseOptions([...args, "--out", join(dir, "curves")]), config);
    const saved = await runJobs(
      parseOptions([...args, "--out", join(dir, "curves"), "--save-model", join(dir, "models")]),
      config,
    );
    assert.equal(formatResults(saved), formatResults(plain));
    assert.deepEqual((await readdir(join(dir, "models"))).sort(), [
      "commands-1.bin",
      "commands-1.json",
      "none-1.bin",
      "none-1.json",
      "schema-1.bin",
      "schema-1.json",
    ]);
    for (const condition of ["schema", "none", "commands"] as const) {
      const metadata = JSON.parse(
        await readFile(join(dir, "models", `${condition}-1.json`), "utf8"),
      );
      assert.deepEqual(metadata, { sizes: config.sizes, condition, subject: 1, iterations: 2 });
      const bytes = new Uint8Array(await readFile(join(dir, "models", `${condition}-1.bin`)));
      const loaded = decodeModel(metadata, bytes.buffer).model;
      const original = trainSubject({
        config,
        condition,
        subject: 1,
        iterations: 2,
        evalEvery: 1,
      }).model;
      assert.deepEqual(encodeParameters(loaded.parameters), encodeParameters(original.parameters));
      const first = createBatch(original, 1),
        second = createBatch(loaded, 1);
      const observation = new Uint8Array(2000);
      for (const seed of [1, 61, 4000000001]) {
        const world = createWorld(conditionConfig(condition), seed);
        appendFrame(observation, world.frame, true);
        for (let step = 1; step < 10; step++) {
          first.activations[0]!.set(observation);
          second.activations[0]!.set(observation);
          assert.deepEqual(
            encodeParameters(forward(original, first)),
            encodeParameters(forward(loaded, second)),
          );
          stepWorld(world, step % 16);
          appendFrame(observation, world.frame, false);
        }
      }
      assert.throws(
        () => decodeModel(metadata, bytes.buffer.slice(0, -8)),
        /Неверная длина файла весов/,
      );
    }
    assert.deepEqual(await readdir(dir), ["curves", "models"]);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("model files use explicit little-endian binary64 and reject malformed descriptions", () => {
  assert.deepEqual([...encodeParameters(new Float64Array([1]))], [0, 0, 0, 0, 0, 0, 240, 63]);
  for (const value of [null, {}, { sizes: [2, 0], condition: "schema", subject: 1, iterations: 2 }])
    assert.throws(() => decodeModel(value, new ArrayBuffer(0)));
  for (const args of [
    ["--save-model"],
    ["--save-model", ""],
    ["--save-model", "a", "--save-model", "b"],
    ["--out", "a", "--save-model", "a"],
  ])
    assert.throws(() => parseOptions(args));
});
