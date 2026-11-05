import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { mkdtemp, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { meanAndError, parseOptions } from "./train-cli.ts";

test("CLI validates arguments and keeps the canonical condition order", () => {
  const options = parseOptions(["--conditions", "commands,schema", "--workers", "3"]);
  assert.deepEqual(options.conditions, ["schema", "commands"]);
  assert.equal(options.workers, 3);
  for (const args of [
    ["--workers", "0"],
    ["--subjects", "1.5"],
    ["--iterations"],
    ["--conditions", "schema,schema"],
    ["--conditions", ""],
    ["--conditions", "unknown"],
    ["--workers", "1", "--workers", "2"],
    ["--unknown", "1"],
  ]) {
    assert.throws(() => parseOptions(args));
  }
});

test("saved jobs are reused for equal parameters and rejected for different ones", async () => {
  const directory = await mkdtemp(join(tmpdir(), "level0-saved-"));
  try {
    const runner = join(directory, "run.mjs");
    const config = { sizes: [2000, 3, 16], collectSteps: 12, updates: 2, batchSize: 4, evalEpisodes: 2 };
    const moduleUrl = JSON.stringify(new URL("./train-cli.ts", import.meta.url).href);
    await writeFile(
      runner,
      `import { runCli } from ${moduleUrl};\nawait runCli(process.argv.slice(2), ${JSON.stringify(config)});\n`,
    );
    const out = join(directory, "runs");
    const args = [runner, "--subjects", "1", "--eval-every", "1", "--conditions", "none"];
    const execute = promisify(execFile);
    const first = await execute(process.execPath, [...args, "--iterations", "2", "--out", out]);
    assert.deepEqual(await readdir(out), ["none-1.json"]);
    const second = await execute(process.execPath, [...args, "--iterations", "2", "--out", out]);
    assert.equal(second.stdout, first.stdout);
    assert.equal(second.stderr, "");
    await assert.rejects(
      execute(process.execPath, [...args, "--iterations", "3", "--out", out]),
      /другими параметрами/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("standard error uses sample variance and is undefined for one subject", () => {
  assert.deepEqual(meanAndError([4]), { mean: 4, error: null });
  assert.deepEqual(meanAndError([1, 2, 3]), { mean: 2, error: Math.sqrt(1 / 3) });
});

test("stdout is byte-identical with one and three workers on a short training run", async () => {
  const directory = await mkdtemp(join(tmpdir(), "level0-workers-"));
  try {
    const runner = join(directory, "run.mjs");
    const config = {
      sizes: [2000, 3, 16],
      collectSteps: 12,
      updates: 2,
      batchSize: 4,
      evalEpisodes: 2,
    };
    const moduleUrl = JSON.stringify(new URL("./train-cli.ts", import.meta.url).href);
    const source = `import { runCli } from ${moduleUrl};\n` +
      `await runCli(process.argv.slice(2), ${JSON.stringify(config)});\n`;
    await writeFile(runner, source);
    const execute = promisify(execFile);
    const args = [
      runner,
      "--subjects", "1",
      "--iterations", "2",
      "--eval-every", "1",
      "--conditions", "commands,none,schema",
      "--workers",
    ];
    const first = await execute(process.execPath, [...args, "1"]);
    const second = await execute(process.execPath, [...args, "3"]);
    assert.equal(first.stdout, second.stdout);
    assert.match(first.stdout, /^Кривые\n/);
    assert.match(first.stdout, /Итог/);
    assert.match(first.stderr, /обновление .* мс/);
    assert.doesNotMatch(first.stdout, / мс|итерация .* с/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
