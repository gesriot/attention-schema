export type Adam = { step: number; m: Float64Array; v: Float64Array };

export function createAdam(parameterCount: number): Adam {
  return { step: 0, m: new Float64Array(parameterCount), v: new Float64Array(parameterCount) };
}

export function adamStep(parameters: Float64Array, gradient: Float64Array, adam: Adam): void {
  adam.step++;
  const rate =
    (0.001 * Math.sqrt(1 - Math.pow(0.999, adam.step))) / (1 - Math.pow(0.9, adam.step));
  for (let i = 0; i < parameters.length; i++) {
    const g = gradient[i]!;
    const m = 0.9 * adam.m[i]! + 0.1 * g;
    const v = 0.999 * adam.v[i]! + 0.001 * g * g;
    adam.m[i] = m;
    adam.v[i] = v;
    parameters[i] = parameters[i]! - rate * m / (Math.sqrt(v) + 1e-7);
  }
}
