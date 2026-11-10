import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { defaults, parameters } from "./config.ts";
import { formatResults, parseOptions } from "./check-cli.ts";
import { checkTarget } from "./evaluate.ts";

test("all parameters can be overridden and invalid or unknown settings are rejected", () => {
  assert.deepEqual(parseOptions([]), { episodes: 2000, config: defaults });
  const all = parameters.map((name) => `${name}=${defaults[name]}`).join(",");
  assert.deepEqual(parseOptions(["--set", all]).config, defaults);
  const options = parseOptions(["--episodes", "3", "--set", "sigmaEta=0,pFlash=0.2,k=8"]);
  assert.equal(options.episodes, 3);
  assert.equal(options.config.sigmaEta, 0);
  assert.equal(options.config.pFlash, 0.2);
  assert.equal(options.config.k, 8);
  for (const args of [
    ["--set", "unknown=1"],
    ["--set", "toString=1"],
    ["--set", "pMove=-1"],
    ["--set", "rho=1"],
    ["--set", "k=3"],
    ["--set", "sigmaEta=NaN"],
    ["--set", "sigmaEta=Infinity"],
    ["--set", "sigmaEta="],
    ["--set", "pFlash=2"],
    ["--set", "pFlash=0.1,pFlash=0.2"],
    ["--set", "hueJumpMin=28"],
    ["--set", "hueJumpMax=31"],
    ["--set", "flashMin=2"],
    ["--set", "hueJitter=0.2"],
    ["--set", "occluderRows=17"],
    ["--set", "minGap=9"],
    ["--set", "tau0=0"],
    ["--set", "R=0"],
    ["--set", "sigmaB=0"],
    ["--set", "paddleStart=15"],
    ["--episodes", "0"],
    ["--episodes", "1.2"],
    ["--episodes"],
    ["--workers", "2"],
  ])
    assert.throws(() => parseOptions(args));
});

test("small checks are repeatable and undefined event fractions print explicitly", () => {
  const options = parseOptions(["--episodes", "2", "--set", "pFlash=0"]);
  const first = checkTarget(options.config, options.episodes);
  const second = checkTarget(options.config, options.episodes);
  assert.deepEqual(first, second);
  assert.equal(first.neutral.flashes, 0);
  assert.equal(first.ideal.flashes, 0);
  const output = formatResults(options, first);
  assert.match(output, /Доля захваченных вспышек \| н\/д \| н\/д/);
  assert.match(output, /5000000001–5000000002/);
  assert.doesNotMatch(output, /NaN|Infinity|Время/);
  const controlLines = output.split("\n").filter((line) => /^(F2|C0) \|/.test(line));
  assert.equal(controlLines.length, 2);
  assert.throws(() => checkTarget(defaults, 0));
});

test("the command prints deterministic results and sends only timing or errors to stderr", () => {
  const cli = new URL("./check-cli.ts", import.meta.url);
  const run = () =>
    spawnSync(process.execPath, [cli.pathname, "--episodes", "2"], { encoding: "utf8" });
  const first = run();
  const second = run();
  assert.equal(first.status, 0);
  assert.equal(second.status, 0);
  assert.equal(first.stdout, second.stdout);
  assert.match(first.stderr, /^Время прогона: \d+\.\d{3} с\n$/);
  const invalid = spawnSync(process.execPath, [cli.pathname, "--set", "wrong=1"], {
    encoding: "utf8",
  });
  assert.equal(invalid.status, 1);
  assert.equal(invalid.stdout, "");
  assert.match(invalid.stderr, /Неизвестный параметр/);
});
