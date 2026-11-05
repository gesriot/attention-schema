export type TrainingOptions<Condition extends string> = {
  subjects: number;
  iterations: number;
  evalEvery: number;
  workers: number;
  conditions: Condition[];
  out: string | null;
  noise?: number[];
};

export function parseTrainingOptions<Condition extends string>(
  args: string[],
  conditions: readonly Condition[],
  evalEvery: number,
  noiseGrid?: readonly number[],
): TrainingOptions<Condition> {
  const result: TrainingOptions<Condition> = {
    subjects: 5,
    iterations: 1500,
    evalEvery,
    workers: 8,
    conditions: [...conditions],
    out: null,
  };
  if (noiseGrid !== undefined) result.noise = [...noiseGrid];
  const keys = {
    "--subjects": "subjects",
    "--iterations": "iterations",
    "--eval-every": "evalEvery",
    "--workers": "workers",
  } as const;
  const seen = new Set<string>();
  for (let i = 0; i < args.length; i += 2) {
    const flag = args[i]!;
    const value = args[i + 1];
    if (seen.has(flag) || value === undefined) throw new Error(`Неверный аргумент: ${flag}`);
    seen.add(flag);
    if (flag === "--conditions") {
      const requested = value.split(",");
      if (
        requested.length !== new Set(requested).size ||
        requested.some((name) => !conditions.includes(name as Condition))
      ) {
        throw new Error(`Допустимые условия: ${conditions.join(",")}, без повторов`);
      }
      result.conditions = conditions.filter((name) => requested.includes(name));
    } else if (flag === "--noise" && noiseGrid !== undefined) {
      const entries = value.split(",");
      const requested = entries.map(Number);
      if (
        entries.some((entry) => entry.trim() === "") ||
        requested.length !== new Set(requested).size ||
        requested.some((noise) => !noiseGrid.includes(noise))
      ) {
        throw new Error(`Допустимые уровни шума: ${noiseGrid.join(",")}, без повторов`);
      }
      result.noise = noiseGrid.filter((noise) => requested.includes(noise));
    } else if (flag === "--out") {
      result.out = value;
    } else if (Object.hasOwn(keys, flag)) {
      const number = Number(value);
      if (!/^\d+$/.test(value) || !Number.isSafeInteger(number) || number < 1) {
        throw new Error(`Нужно положительное целое число: ${flag}`);
      }
      result[keys[flag as keyof typeof keys]] = number;
    } else {
      throw new Error(`Неизвестный аргумент: ${flag}`);
    }
  }
  return result;
}
