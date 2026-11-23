# attention-schema

A research bench for Michael Graziano's Attention Schema Theory. An agent controls a limited attention resource, builds a simplified model of that attention, and uses it for control and report. The bench tests the control function of the schema and its causal role. It does not establish phenomenal experience.

**Interactive laboratory:** https://gesriot.github.io/attention-schema/

- `docs/spec.md` – the specification and a summary of the theory, checked against primary sources.
- `docs/results.md` – experiment results; it opens with the current status and the plans.
- `docs/wilterson-code-audit.md` – an audit of the published code of Wilterson & Graziano.
- `web/` – the TypeScript implementation and the laboratory.

## Status

The work is paused. Level 0 is done: the replication of Wilterson & Graziano, the exploratory tests T1 to T15 and the first stage of schema emergence (E6). The target world is built and passes its feasibility controls except F4, which passes in part, and C6, which fails: an oracle controller does not learn to steer attention by reinforcement within this budget. A hand-written regulator shows instead how much knowledge of attention is worth for control. The planned next steps, including the Rust port, are listed at the top of `docs/results.md`.

## Running `web/`

Requires Node.js 22.18 or newer.

```sh
cd web
npm install
npm run typecheck
npm test
```

Experiments:

```sh
npm run level0:check   # level 0: manual checks of the environment
npm run level0:train   # level 0: DQN training
npm run level0:emergence   # level 0: schema emergence (E6)
npm run level0:false-schema   # level 0: false schema with attention intact (E2A)
npm run target:check   # target mode: feasibility controls C0, F2, F6 and the hand-written policy over the inputs of C6
npm run target:f4   # target mode: separation of sources (F4)
npm run target:c6   # target mode: the attention oracle (C6)
```

Command output is in Russian.

## Laboratory

The laboratory is in Russian and covers level 0: the game of Wilterson & Graziano played by a trained DQN agent with the built-in schema, without a schema or with the command history, with the interventions of the bench (window off, field hidden, false schema). Truth is shown separately from what the agent sees.

```sh
npm run dev
npm run build -- --base=/attention-schema/
npm run preview -- --base=/attention-schema/
```

The models in `web/public/models/` are subject 1 of the full level 0 run, 1500 iterations at noise 0.5; other noise levels display a warning. Each is a `.json` description and the network parameters as little-endian binary64 in a `.bin` file. They are produced by

```sh
npm run level0:train -- --subjects 1 --iterations 1500 --eval-every 10 --workers 3 --conditions schema,none,commands --save-model public/models
```

`--save-model` always trains, since stored results hold curves and no weights, and its directory must differ from `--out`. The Pages workflow builds and deploys on every push to `main`.
