import { createTruthPolicy, meanScores, windowReadable, type PolicyName } from "./evaluate.ts";
import { createTrackerPolicy } from "./policies.ts";
import { createWorld, inWindow, stepWorld, type WorldConfig } from "./world.ts";

const variants: WorldConfig[] = [
  { variant: "wilterson", schema: "given", noise: 0.5, windowActive: true },
  { variant: "piefke", schema: "free", noise: 0.5, windowActive: true },
];
const names: Record<PolicyName, string> = {
  truth: "истина",
  tracker: "трекер",
  random: "случайная",
};

console.log("Счёт · зёрна 1–2000 · шум 0.500");
console.log("Вариант / схема | Окно | Политика | Слежение | Ловля | Сумма");
for (const config of variants) {
  const variant = config.variant === "wilterson" ? "Уилтерсон / заданная" : "Пифке / свободная";
  for (const windowActive of [true, false]) {
    for (const policy of ["truth", "tracker", "random"] as const) {
      const scores = meanScores({ ...config, windowActive }, 1, 2000, policy);
      console.log(
        `${variant} | ${windowActive ? "включено" : "отключено"} | ${names[policy]} | ` +
          `${scores.tracking.toFixed(3)} | ${scores.catching.toFixed(3)} | ` +
          `${scores.total.toFixed(3)}`,
      );
    }
  }
}

console.log("\nСобытия · Уилтерсон / заданная · окно включено · шум 0.500 · зёрна 1–10000");
const config = variants[0]!;
let straight = 0;
let diagonalSteps = 0;
let outside = 0;
let frames = 0;
let maxLandingError = 0;
let trackingSum = 0;
for (let seed = 1; seed <= 10000; seed++) {
  const world = createWorld(config, seed);
  const tracker = createTrackerPolicy(config);
  if (world.startCol === world.landingCol) straight++;
  while (world.t < 10) {
    const previousCol = world.ball.col;
    stepWorld(world, tracker(world.frame, world.t));
    if (world.ball.col !== previousCol) diagonalSteps++;
    if (world.t >= 2 && world.t <= 9) {
      frames++;
      if (!inWindow(world.ball, world.window)) outside++;
    }
  }
  maxLandingError = Math.max(maxLandingError, Math.abs(world.ball.col - world.landingCol));
  trackingSum += world.trackingScore;
}
const outsideRate = outside / frames;
const trackingMean = trackingSum / 10000;
const trackingFromOutside = 4 * (1 - 2 * outsideRate);
console.log(`Доля эпизодов без изгиба (старт = приземление): ${(straight / 10000).toFixed(6)}`);
console.log(`Средняя длина диагонального участка, шагов: ${(diagonalSteps / 10000).toFixed(6)}`);
console.log(`Доля кадров с мячом вне окна трекера (t = 2..9): ${outsideRate.toFixed(6)}`);
console.log(`Максимум ошибки колонки приземления: ${maxLandingError.toFixed(0)}`);
console.log(`Средний счёт слежения трекера: ${trackingMean.toFixed(6)}`);
console.log(`4 × (1 − 2 × доля выходов): ${trackingFromOutside.toFixed(6)}`);
console.log(`Разность двух подсчётов: ${(trackingMean - trackingFromOutside).toFixed(6)}`);
if (maxLandingError !== 0 || trackingSum !== 4 * 10000 - outside) {
  throw new Error("Нарушен инвариант траектории или подсчёта слежения");
}

console.log("\nОбход · Уилтерсон / заданная · истина · окно включено · t = 2..9 · зёрна 1–2000");
console.log("Шум | Доля читаемых кадров");
for (const noise of [0, 0.25, 0.5, 0.75, 1]) {
  let readable = 0;
  let count = 0;
  for (let seed = 1; seed <= 2000; seed++) {
    const world = createWorld({ ...config, noise }, seed);
    const truth = createTruthPolicy();
    while (world.t < 9) {
      stepWorld(world, truth(world));
      if (windowReadable(world.frame, world)) readable++;
      count++;
    }
  }
  console.log(`${noise.toFixed(2)} | ${(readable / count).toFixed(6)}`);
}
