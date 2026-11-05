# attention-schema

A research bench for Michael Graziano's Attention Schema Theory. An agent controls a limited attention resource, builds a simplified model of that attention, and uses it for control and report. The bench tests the control function of the schema and its causal role. It does not establish phenomenal experience.

- `docs/spec.md` – the specification and a summary of the theory, checked against primary sources.
- `docs/results.md` – experiment results.
- `web/` – the TypeScript implementation.
- `rust/` – the second implementation, with outputs compared byte for byte.

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
```

Command output is in Russian.
