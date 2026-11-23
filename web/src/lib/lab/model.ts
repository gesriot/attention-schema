import { conditions, type Condition } from "../level0/dqn.ts";
import { createMlp, type Mlp } from "../nn/mlp.ts";
import { createRng } from "../rng.ts";

export type ModelMetadata = {
  sizes: number[];
  condition: Condition;
  subject: number;
  iterations: number;
};
export type LoadedModel = { model: Mlp; metadata: ModelMetadata };

export function parameterCount(sizes: readonly number[]): number {
  if (sizes.length < 2 || sizes.some((size) => !Number.isSafeInteger(size) || size < 1))
    throw new Error("Размеры слоёв модели должны быть положительными целыми числами.");
  let count = 0;
  for (let i = 1; i < sizes.length; i++) count += (sizes[i - 1]! + 1) * sizes[i]!;
  if (!Number.isSafeInteger(count * 8)) throw new Error("Слишком большой размер модели.");
  return count;
}

export function readMetadata(value: unknown): ModelMetadata {
  if (value === null || typeof value !== "object")
    throw new Error("Не удалось прочитать описание модели.");
  const data = value as Partial<ModelMetadata>;
  if (
    !Array.isArray(data.sizes) ||
    !conditions.includes(data.condition as Condition) ||
    !Number.isSafeInteger(data.subject) ||
    data.subject! < 1 ||
    !Number.isSafeInteger(data.iterations) ||
    data.iterations! < 1
  )
    throw new Error(
      "Неверное описание модели: нужны размеры слоёв, условие, субъект и число итераций.",
    );
  parameterCount(data.sizes);
  return {
    sizes: [...data.sizes],
    condition: data.condition!,
    subject: data.subject!,
    iterations: data.iterations!,
  };
}

// File order is model.parameters; DataView fixes little-endian on every host.
export function encodeParameters(parameters: Float64Array): Uint8Array {
  const bytes = new Uint8Array(parameters.length * 8);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < parameters.length; i++) view.setFloat64(i * 8, parameters[i]!, true);
  return bytes;
}

export function decodeModel(description: unknown, buffer: ArrayBuffer): LoadedModel {
  const metadata = readMetadata(description);
  const expected = parameterCount(metadata.sizes) * 8;
  if (buffer.byteLength !== expected)
    throw new Error(
      `Неверная длина файла весов: получено ${buffer.byteLength} байт, ожидается ${expected}.`,
    );
  const values = new DataView(buffer);
  const model = createMlp(metadata.sizes, createRng(0, 3));
  for (let i = 0; i < model.parameters.length; i++) {
    const value = values.getFloat64(i * 8, true);
    if (!Number.isFinite(value)) throw new Error("Файл весов содержит неконечные значения.");
    model.parameters[i] = value;
  }
  return { model, metadata };
}

export async function fetchModel(
  base: string,
  condition: Condition,
  signal?: AbortSignal,
): Promise<LoadedModel> {
  const prefix = `${base.endsWith("/") ? base : base + "/"}models/${condition}-1`;
  const [description, weights] = await Promise.all([
    fetch(`${prefix}.json`, { signal }),
    fetch(`${prefix}.bin`, { signal }),
  ]);
  if (!description.ok || !weights.ok)
    throw new Error(
      "Не удалось загрузить модель. Проверьте наличие файлов весов и описания, затем повторите загрузку.",
    );
  const loaded = decodeModel(await description.json(), await weights.arrayBuffer());
  if (loaded.metadata.condition !== condition || loaded.metadata.subject !== 1)
    throw new Error("Описание модели не соответствует выбранному условию и субъекту.");
  if (loaded.model.sizes[0] !== 2000 || loaded.model.sizes.at(-1) !== 16)
    throw new Error("Для уровня 0 нужна сеть с 2000 входами и 16 выходами.");
  return loaded;
}
