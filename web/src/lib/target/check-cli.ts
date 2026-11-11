import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  configure,
  defaults,
  firstSeed,
  parameters,
  type Config,
  type Parameter,
} from "./config.ts";
import { checkTarget, type CheckResult } from "./evaluate.ts";

const labels: Record<Parameter, string> = {
  fieldSize: "Сторона поля",
  episodeSteps: "Шагов в эпизоде",
  objectBrightness: "Яркость объекта",
  pMove: "Вероятность движения",
  pTurn: "Вероятность поворота перед движением",
  minGap: "Начальное расстояние объектов",
  hueCategories: "Категорий оттенка",
  hueJitter: "Разброс оттенка",
  hueJumpMin: "Первый шаг скачка",
  hueJumpMax: "Последний шаг скачка",
  occluderRows: "Высота заслонки",
  occluderCols: "Ширина заслонки",
  occluderBrightness: "Яркость заслонки",
  pFlash: "Вероятность вспышки",
  flashMin: "Минимум силы вспышки",
  flashMax: "Максимум силы вспышки",
  paddleMin: "Минимум центра платформы",
  paddleMax: "Максимум центра платформы",
  paddleStart: "Старт платформы",
  catchRadius: "Радиус ловли",
  catchReward: "Модуль награды за ловлю",
  answerReward: "Модуль награды за ответ",
  effortCost: "Цена усилия",
  R: "Бюджет ресурса",
  tau0: "Температура без усилия",
  rho: "Инерция приоритета",
  betaEndo: "Сила команды",
  sigmaB: "Ширина бугра",
  betaExo: "Сила захвата",
  sigmaEpsilon: "Шум механизма",
  sigmaEta: "Шум измерения",
  alpha: "Усиление измерения",
  k: "Сторона блока телеметрии",
  sigmaNu: "Шум телеметрии",
  resourceRadius: "Радиус подсчёта доли ресурса",
  effortThreshold: "Порог включения усилия",
  captureThreshold: "Порог захвата",
  focusThreshold: "Порог фокуса",
  shiftThreshold: "Порог сдвига фокуса",
  sigmaFloor: "Нижняя граница оценки шума",
  gateRadius: "Радиус ворот слота",
  detectThreshold: "Порог обнаружения",
  velocityGain: "Коэффициент обновления скорости",
  minBrightness: "Минимальная яркость обнаружения",
  hueForget: "Сохранение веса оттенка",
  maxMissed: "Пропусков до гашения слота 1",
  f4Hidden: "Узлов состояния зонда F4",
  f4BatchSize: "Эпизодов в батче зонда F4",
  f4Updates: "Обновлений зонда F4",
  f4TrainFraction: "Доля обучающих эпизодов F4",
};

export type Options = { episodes: number; config: Config };

export function parseOptions(args: string[]): Options {
  let episodes = 2000;
  const overrides: Partial<Config> = {};
  const seen = new Set<string>();
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index]!;
    const value = args[index + 1];
    if (value === undefined || value.startsWith("--")) throw new Error(`Нет значения: ${flag}`);
    if (flag === "--episodes") {
      if (seen.has(flag)) throw new Error(`Повторный параметр: ${flag}`);
      episodes = Number(value);
      if (
        !Number.isSafeInteger(episodes) ||
        episodes < 1 ||
        firstSeed + episodes > Number.MAX_SAFE_INTEGER
      ) {
        throw new Error("Число эпизодов должно быть положительным целым");
      }
      seen.add(flag);
    } else if (flag === "--set") {
      for (const entry of value.split(",")) {
        const parts = entry.split("=");
        const name = parts[0]!;
        if (parts.length !== 2 || !parts[1]!.trim())
          throw new Error(`Нужна запись имя=значение: ${entry}`);
        if (!Object.hasOwn(defaults, name)) throw new Error(`Неизвестный параметр: ${name}`);
        if (seen.has(name)) throw new Error(`Повторный параметр: ${name}`);
        overrides[name as Parameter] = Number(parts[1]);
        seen.add(name);
      }
    } else {
      throw new Error(`Неизвестный аргумент: ${flag}`);
    }
  }
  return { episodes, config: configure(overrides) };
}

export function formatParameters(config: Config): string[] {
  const lines = ["Параметры", "Имя | Описание | Значение"];
  for (const name of parameters)
    lines.push(`${name} | ${labels[name]} | ${config[name].toFixed(6)}`);
  return lines;
}

export function formatResults(options: Options, result: CheckResult): string {
  const lines = formatParameters(options.config);
  lines.push(
    "",
    `Контроли · эпизодов ${options.episodes.toFixed(0)} · зёрна ` +
      `${firstSeed.toFixed(0)}–${(firstSeed + options.episodes - 1).toFixed(0)}`,
  );
  lines.push("Контроль | Доля верных ответов | Доля ловли | Средний итог с ценой усилия");
  for (const name of ["F2", "C0"] as const) {
    const control = result[name];
    lines.push(
      `${name} | ${control.correctRate.toFixed(6)} | ${control.catchRate.toFixed(6)} | ` +
        control.meanScore.toFixed(6),
    );
  }
  lines.push(
    "",
    "Контроли с восприятием",
    "Контроль | Доля верных ответов | Доля ловли | Доля шагов со слотом 0 рядом с целью",
  );
  for (const name of ["F2", "C0"] as const) {
    const control = result.perception[name];
    lines.push(
      `${name} с восприятием | ${control.correctRate.toFixed(6)} | ` +
        `${control.catchRate.toFixed(6)} | ${control.trackingRate.toFixed(6)}`,
    );
  }
  lines.push(
    "",
    `Ложные тревоги · чистый единичный шум · кадров ${result.falseAlarms.frames.toFixed(0)}`,
    "Доля кадров с максимумом SNR не ниже порога | 95-й перцентиль максимума SNR",
    `${result.falseAlarms.rate.toFixed(6)} | ${result.falseAlarms.percentile95.toFixed(6)}`,
  );
  lines.push("", "F6 · события", "Показатель | Нейтральная политика | Идеальная политика");
  const rows = [
    ["flashes", "Число вспышек"],
    ["captureRate", "Доля захваченных вспышек"],
    ["captureLength", "Средняя длина захвата, шагов"],
    ["shiftsPerEpisode", "Сдвиги фокуса за эпизод"],
    ["exitsPerEpisode", "Выходы цели из фокуса за эпизод"],
    ["occludedRate", "Доля шагов под заслонкой"],
    ["commandGap", "Средний разрыв команды и фокуса"],
    ["focusedRate", "Доля шагов с целью в фокусе"],
    ["targetShare", "Средняя доля ресурса на цели"],
    ["entropy", "Средняя энтропия, нат"],
    ["effortRate", "Доля шагов с усилием"],
  ] as const;
  for (const [key, label] of rows) {
    const format = (value: number | null) =>
      value === null ? "н/д" : value.toFixed(key === "flashes" ? 0 : 6);
    lines.push(`${label} | ${format(result.neutral[key])} | ${format(result.ideal[key])}`);
  }
  lines.push("Доли шагов: все кадры; разрыв команды: переходы со второго кадра.");
  lines.push(
    "Захват: с кадра вспышки до первого падения доли; конец эпизода обрывает длительность.",
  );
  lines.push("Ловля проверяет механику платформы; численный порог решения задачи не задан.");
  return lines.join("\n") + "\n";
}

export function runCli(args: string[]): void {
  const options = parseOptions(args);
  const start = performance.now();
  const result = checkTarget(options.config, options.episodes);
  process.stdout.write(formatResults(options, result));
  process.stderr.write(`Время прогона: ${((performance.now() - start) / 1000).toFixed(3)} с\n`);
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    runCli(process.argv.slice(2));
  } catch (error) {
    process.stderr.write(`Ошибка: ${error instanceof Error ? error.message : String(error)}\n`);
    process.exitCode = 1;
  }
}
