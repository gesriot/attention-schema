import { firstSeed, type Config } from "./config.ts";
import { commandedCenter, distance, resourceShare } from "./attention.ts";
import { applyAction, createEpisode, type Action, type Score, type Truth } from "./episode.ts";
import { createIdealPolicy, createNeutralPolicy } from "./policies.ts";
import { isOccluded, type Position } from "./world.ts";
import { checkPerception, type PerceptionControls } from "./perception-controls.ts";

export type EventCounts = {
  episodes: number;
  frames: number;
  transitions: number;
  flashes: number;
  captures: number;
  captureSteps: number;
  shifts: number;
  exits: number;
  occluded: number;
  commandGap: number;
  focused: number;
  targetShare: number;
  entropy: number;
  effort: number;
};
export type EventTracker = {
  counts: EventCounts;
  previous: { center: Position; targetShare: number; command: Position } | null;
  captures: Position[];
};

export function createEventTracker(): EventTracker {
  return {
    counts: {
      episodes: 1,
      frames: 0,
      transitions: 0,
      flashes: 0,
      captures: 0,
      captureSteps: 0,
      shifts: 0,
      exits: 0,
      occluded: 0,
      commandGap: 0,
      focused: 0,
      targetShare: 0,
      entropy: 0,
      effort: 0,
    },
    previous: null,
    captures: [],
  };
}

export function recordEvents(
  config: Config,
  tracker: EventTracker,
  truth: Truth,
  action: Action,
): void {
  const { counts, previous } = tracker;
  const { world, attention } = truth;
  const share = resourceShare(config, attention, world.target);
  counts.frames++;
  counts.targetShare += share;
  counts.entropy += attention.entropy;
  counts.effort += action.effort;
  if (share >= config.focusThreshold) counts.focused++;
  if (isOccluded(config, world.occluder, world.target)) counts.occluded++;
  if (previous !== null) {
    counts.transitions++;
    if (distance(attention.center, previous.center) >= config.shiftThreshold) counts.shifts++;
    if (previous.targetShare >= config.focusThreshold && share < config.focusThreshold)
      counts.exits++;
    counts.commandGap += distance(
      attention.center,
      commandedCenter(config, previous.center, previous.command),
    );
  }
  // A captured flash is followed at its fixed cell until its share first falls below
  // threshold. Overlapping events are counted separately; the episode end censors duration.
  tracker.captures = tracker.captures.filter(
    (position) => resourceShare(config, attention, position) >= config.captureThreshold,
  );
  if (world.flash !== null) {
    counts.flashes++;
    if (resourceShare(config, attention, world.flash) >= config.captureThreshold) {
      counts.captures++;
      tracker.captures.push({ row: world.flash.row, col: world.flash.col });
    }
  }
  counts.captureSteps += tracker.captures.length;
  tracker.previous = {
    center: { ...attention.center },
    targetShare: share,
    command: { ...action.command },
  };
}

export type EventSummary = {
  flashes: number;
  captureRate: number | null;
  captureLength: number | null;
  shiftsPerEpisode: number;
  exitsPerEpisode: number;
  occludedRate: number;
  commandGap: number;
  focusedRate: number;
  targetShare: number;
  entropy: number;
  effortRate: number;
};

export function summarizeEvents(counts: EventCounts): EventSummary {
  return {
    flashes: counts.flashes,
    captureRate: counts.flashes === 0 ? null : counts.captures / counts.flashes,
    captureLength: counts.captures === 0 ? null : counts.captureSteps / counts.captures,
    shiftsPerEpisode: counts.shifts / counts.episodes,
    exitsPerEpisode: counts.exits / counts.episodes,
    occludedRate: counts.occluded / counts.frames,
    commandGap: counts.commandGap / counts.transitions,
    focusedRate: counts.focused / counts.frames,
    targetShare: counts.targetShare / counts.frames,
    entropy: counts.entropy / counts.frames,
    effortRate: counts.effort / counts.frames,
  };
}

export type ControlSummary = { correctRate: number; catchRate: number; meanScore: number };
export type CheckResult = {
  F2: ControlSummary;
  C0: ControlSummary;
  neutral: EventSummary;
  ideal: EventSummary;
  perception: PerceptionControls;
};

function addCounts(total: EventCounts, counts: EventCounts): void {
  for (const key of Object.keys(total) as (keyof EventCounts)[]) total[key] += counts[key];
}

export function checkTarget(config: Config, episodes: number): CheckResult {
  if (
    !Number.isSafeInteger(episodes) ||
    episodes < 1 ||
    firstSeed + episodes > Number.MAX_SAFE_INTEGER
  ) {
    throw new RangeError("Число эпизодов должно быть положительным целым");
  }
  const idealCounts = createEventTracker().counts;
  const neutralCounts = createEventTracker().counts;
  idealCounts.episodes = 0;
  neutralCounts.episodes = 0;
  const F2 = { correctRate: 0, catchRate: 0, meanScore: 0 };
  const C0 = { correctRate: 0, catchRate: 0, meanScore: 0 };
  const addScore = (total: ControlSummary, score: Score) => {
    total.correctRate += Number(score.correct);
    total.catchRate += Number(score.caught);
    total.meanScore += score.total;
  };
  // Fixed order: seed, F2, C0, neutral. Each run starts its own identical exogenous streams.
  for (let index = 0; index < episodes; index++) {
    const seed = firstSeed + index;
    for (const enhanced of [true, false]) {
      const episode = createEpisode(config, seed, enhanced);
      const policy = createIdealPolicy(config, enhanced);
      const tracker = enhanced ? createEventTracker() : null;
      while (episode.score === null) {
        const action = policy(episode.observation, episode.truth);
        if (tracker !== null) recordEvents(config, tracker, episode.truth, action);
        applyAction(episode, action);
      }
      addScore(enhanced ? F2 : C0, episode.score);
      if (tracker !== null) addCounts(idealCounts, tracker.counts);
    }
    const episode = createEpisode(config, seed);
    const policy = createNeutralPolicy(config, seed);
    const tracker = createEventTracker();
    while (episode.score === null) {
      const action = policy(episode.observation);
      recordEvents(config, tracker, episode.truth, action);
      applyAction(episode, action);
    }
    addCounts(neutralCounts, tracker.counts);
  }
  for (const total of [F2, C0]) {
    total.correctRate /= episodes;
    total.catchRate /= episodes;
    total.meanScore /= episodes;
  }
  return {
    F2,
    C0,
    neutral: summarizeEvents(neutralCounts),
    ideal: summarizeEvents(idealCounts),
    perception: checkPerception(config, episodes),
  };
}
