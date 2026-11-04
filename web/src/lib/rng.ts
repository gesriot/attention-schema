export type Rng = { state: number };

export function createRng(seed: number, stream: number): Rng {
  // Mix two u32 inputs with the MurmurHash3 finalizer before the first draw.
  let state = (seed >>> 0) ^ Math.imul((stream + 1) >>> 0, 0x9e3779b9);
  state = Math.imul(state ^ (state >>> 16), 0x85ebca6b);
  state = Math.imul(state ^ (state >>> 13), 0xc2b2ae35);
  return { state: (state ^ (state >>> 16)) >>> 0 };
}

export function nextUnit(rng: Rng): number {
  // One mulberry32 state advance per draw; all products wrap to 32 bits.
  rng.state = (rng.state + 0x6d2b79f5) >>> 0;
  let value = Math.imul(rng.state ^ (rng.state >>> 15), rng.state | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

export function nextInt(rng: Rng, n: number): number {
  return Math.floor(nextUnit(rng) * n);
}
