import assert from "node:assert/strict";
import { test } from "node:test";
import { spawnSync } from "node:child_process";
import { parseC6Options } from "./c6-cli.ts";

test("C6 options reject invalid runs, rates, worker counts and PPO settings", () => {
  for (const args of [
    ["--runs", "100"],
    ["--runs", "1.5"],
    ["--rates", "NaN"],
    ["--rates", "0"],
    ["--rates", "0.001,"],
    ["--rates", "0.001,0.001"],
    ["--workers", "0"],
    ["--episodes", "2"],
    ["--set", "ppoLambda=1.1"],
    ["--set", "ppoMinibatch=65"],
    ["--set", "ppoClip=1"],
    ["--set", "ppoUpdates=10000000"],
    ["--runs"],
  ])
    assert.throws(() => parseC6Options(args));
  assert.deepEqual(parseC6Options(["--rates", "0.001,0.0003"]).rates, [0.0003, 0.001]);
});

test("C6 CLI stdout is independent of worker count and rate order", () => {
  const args = [
    new URL("./c6-cli.ts", import.meta.url).pathname,
    "--runs",
    "2",
    "--set",
    "controllerHidden=4,episodeSteps=3,hueJumpMin=2,hueJumpMax=3,ppoEpisodes=2,ppoMinibatch=1,ppoEpochs=1,ppoUpdates=1,ppoEvalEvery=1,ppoEvalEpisodes=2",
  ];
  const run = (workers: string, rates: string) =>
    spawnSync(process.execPath, [...args, "--workers", workers, "--rates", rates], {
      encoding: "utf8",
    });
  const one = run("1", "0.001,0.0003"),
    two = run("2", "0.0003,0.001");
  assert.equal(one.status, 0, one.stderr);
  assert.equal(two.status, 0, two.stderr);
  assert.equal(one.stdout, two.stdout);
  assert.equal((one.stdout.match(/Обновление \| Награда/g) ?? []).length, 4);
  assert.equal((one.stderr.match(/награда сбора н\/д/g) ?? []).length, 4);
  assert.equal((one.stderr.match(/обновление 1/g) ?? []).length, 4);
  assert.doesNotMatch(one.stdout, /Время|среднее обновление/);
});
