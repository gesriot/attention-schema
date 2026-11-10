// All adjustable model and evaluator constants live here, in output order.
export const defaults = {
  fieldSize: 16,
  episodeSteps: 30,
  objectBrightness: 1,
  pMove: 0.5,
  pTurn: 0.2,
  minGap: 4,
  hueCategories: 4,
  hueJitter: 0.05,
  hueJumpMin: 16,
  hueJumpMax: 27,
  occluderRows: 4,
  occluderCols: 3,
  occluderBrightness: 0.3,
  pFlash: 0.08,
  flashMin: 0.5,
  flashMax: 1.5,
  paddleMin: 1,
  paddleMax: 14,
  paddleStart: 7,
  catchRadius: 1,
  catchReward: 1,
  answerReward: 1,
  effortCost: 0.02,
  R: 9,
  tau0: 0.104,
  rho: 0.4,
  betaEndo: 1,
  sigmaB: 2.5,
  betaExo: 2,
  sigmaEpsilon: 0.1,
  sigmaEta: 1,
  alpha: 0.9,
  k: 4,
  sigmaNu: 0.5,
  resourceRadius: 1,
  effortThreshold: 0.6,
  captureThreshold: 0.3,
  focusThreshold: 0.5,
  shiftThreshold: 1,
};

export type Config = typeof defaults;
export type Parameter = keyof Config;
export const parameters = Object.keys(defaults) as Parameter[];
export const firstSeed = 5000000001;

// Level 0 occupies streams 0..8. Stream state is private to the environment or policy.
export const streams = {
  world: 16,
  flash: 17,
  attention: 18,
  measurement: 19,
  telemetry: 20,
  policy: 21,
};

export function validateConfig(config: Config): void {
  const fail = (name: string) => {
    throw new RangeError(`Недопустимое значение параметра: ${name}`);
  };
  for (const name of Object.keys(config)) {
    if (!Object.hasOwn(defaults, name)) fail(name);
  }
  for (const name of parameters) {
    if (!Number.isFinite(config[name]) || config[name] < 0) fail(name);
  }
  for (const name of [
    "fieldSize",
    "episodeSteps",
    "minGap",
    "hueCategories",
    "hueJumpMin",
    "hueJumpMax",
    "occluderRows",
    "occluderCols",
    "paddleMin",
    "paddleMax",
    "paddleStart",
    "catchRadius",
    "k",
    "resourceRadius",
  ] as const) {
    if (!Number.isSafeInteger(config[name])) fail(name);
  }
  if (config.fieldSize < 2) fail("fieldSize");
  if (config.episodeSteps < 2) fail("episodeSteps");
  if (config.minGap > Math.floor(config.fieldSize / 2)) fail("minGap");
  if (config.hueCategories < 2) fail("hueCategories");
  if (config.hueJitter > 0.5 / config.hueCategories) fail("hueJitter");
  if (
    config.hueJumpMin < 2 ||
    config.hueJumpMax > config.episodeSteps ||
    config.hueJumpMin > config.hueJumpMax
  )
    fail("hueJumpMin, hueJumpMax");
  for (const name of ["occluderRows", "occluderCols", "k"] as const) {
    if (config[name] < 1 || config[name] > config.fieldSize) fail(name);
  }
  if (config.fieldSize % config.k !== 0) fail("k");
  if (
    config.paddleMax >= config.fieldSize ||
    config.paddleMin > config.paddleStart ||
    config.paddleStart > config.paddleMax
  )
    fail("paddleMin, paddleStart, paddleMax");
  if (config.flashMin > config.flashMax) fail("flashMin, flashMax");
  for (const name of [
    "pMove",
    "pTurn",
    "pFlash",
    "alpha",
    "effortThreshold",
    "captureThreshold",
    "focusThreshold",
  ] as const) {
    if (config[name] > 1) fail(name);
  }
  if (config.rho >= 1) fail("rho");
  for (const name of ["R", "tau0", "sigmaB"] as const) {
    if (config[name] <= 0) fail(name);
  }
  if (config.R > config.fieldSize * config.fieldSize) fail("R");
}

export function configure(overrides: Partial<Config> = {}): Config {
  const config = { ...defaults, ...overrides };
  validateConfig(config);
  return config;
}
