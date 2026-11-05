import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import {
  formatEmergenceResults,
  parseEmergenceOptions,
  probeLift,
  resourceBenefits,
  tailMean,
} from "./emergence-cli.ts";
import type { EmergenceResult } from "./emergence.ts";
import { probeSources } from "./probes.ts";

test("emergence arguments accept only the fixed noise grid and canonical conditions", () => {
  const options = parseEmergenceOptions(["--noise", "1,0.5,0", "--conditions", "off,free"]);
  assert.deepEqual(options.noise, [0, 0.5, 1]);
  assert.deepEqual(options.conditions, ["free", "off"]);
  assert.equal(options.evalEvery, 50);
  for (const args of [
    ["--noise", "0.1"], ["--noise", "0.5,0.50"], ["--noise", "NaN"],
    ["--noise", ""], ["--noise", "0,"], ["--conditions", "schema"], ["--subjects", "101"],
  ]) assert.throws(() => parseEmergenceOptions(args));
});

test("summaries average ten checks within each subject before paired resource differences", () => {
  const makeResult = (subject: number): EmergenceResult => ({
    condition: "free", noise: 0.5, subject,
    points: Array.from({ length: 12 }, (_, i) => [
      { iteration: i + 1, check: "normal" as const, tracking: i < 2 ? 100 : subject, catching: 1 },
      { iteration: i + 1, check: "resourceRandom" as const, tracking: i < 2 ? -100 : subject - 0.5, catching: -1 },
      { iteration: i + 1, check: "resourceStill" as const, tracking: subject - 0.25, catching: 0 },
      { iteration: i + 1, check: "fieldOff" as const, tracking: 0, catching: 0 },
      { iteration: i + 1, check: "windowOff" as const, tracking: 0, catching: 0 },
    ]).flat(),
    probes: probeSources.map((source) => ({ source, accuracy: 0.5, majority: 0.25 })),
  });
  const subjects = [makeResult(1), makeResult(3)];
  assert.equal(tailMean(subjects[0]!, "normal", "tracking"), 1);
  assert.deepEqual(resourceBenefits(subjects, "tracking"), [0.5, 0.5]);
  const output = formatEmergenceResults(subjects);
  assert.match(output, /0\.50 \| ресурс случаен \| 0\.500 ± 0\.000 \| 2\.000 ± 0\.000/);
  assert.match(output, /0\.50 \| ресурс неподвижен \| 0\.250 ± 0\.000 \| 1\.000 ± 0\.000/);
  assert.equal(probeLift(0.5, 0.25), 1 / 3);
  assert.equal(output, formatEmergenceResults([...subjects].reverse()));
});

test("emergence stdout ignores worker count and saved jobs validate all parameters", async () => {
  const directory = await mkdtemp(join(tmpdir(), "emergence-jobs-"));
  try {
    const runner = join(directory, "run.mjs");
    const settings = {
      training: { sizes: [200, 4, 18], collectSteps: 12, updates: 2, batchSize: 4, evalEpisodes: 2 },
      probes: { episodes: 8, trainEpisodes: 6, hidden: 4, batchSize: 4, updates: 3 },
    };
    const moduleUrl = JSON.stringify(new URL("./emergence-cli.ts", import.meta.url).href);
    const writeRunner = async () => writeFile(runner,
      `import { runEmergenceCli } from ${moduleUrl};\n` +
      `await runEmergenceCli(process.argv.slice(2), ${JSON.stringify(settings)});\n`);
    await writeRunner();
    const args = [runner, "--conditions", "free,given,off", "--noise", "0.5", "--subjects", "1",
      "--iterations", "2", "--eval-every", "1"];
    const execute = promisify(execFile);
    const first = await execute(process.execPath, [...args, "--workers", "1"]);
    const out = join(directory, "saved");
    const second = await execute(process.execPath, [...args, "--workers", "3", "--out", out]);
    assert.equal(second.stdout, first.stdout);
    assert.match(second.stderr, /зонд/);
    assert.doesNotMatch(second.stdout, / мс/);
    assert.deepEqual(await readdir(out), ["free-p0.50-1.json", "given-p0.50-1.json", "off-p0.50-1.json"]);
    const file = join(out, "free-p0.50-1.json");
    const contents = await readFile(file, "utf8");
    const reused = await execute(process.execPath, [...args, "--workers", "1", "--out", out]);
    assert.equal(reused.stdout, first.stdout);
    assert.equal(reused.stderr, "");
    assert.equal(await readFile(file, "utf8"), contents);
    const changed = [...args];
    changed[changed.indexOf("--iterations") + 1] = "3";
    await assert.rejects(execute(process.execPath, [...changed, "--out", out]), /другими параметрами/);
    settings.probes.updates++;
    await writeRunner();
    await assert.rejects(execute(process.execPath, [...args, "--out", out]), /другими параметрами/);
    assert.equal(await readFile(file, "utf8"), contents);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
