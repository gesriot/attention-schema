# Bench results

*Status as of 4 November 2026. The setup is in [spec.md](spec.md). All numbers can be reproduced with commands from the `web/` directory.*

## Level 0: manual checks

There is no training here. This checks the environment of the published setup: seeds 1 to 2000, noise 0.5, a maximum of 6 per episode.

| Variant | Window | Policy | Tracking | Catch | Total |
| :-- | :-- | :-- | --: | --: | --: |
| Wilterson, given schema | on | truth | 4.000 | 2.000 | 6.000 |
| | on | tracker | 3.554 | 1.780 | 5.334 |
| | on | random | −2.213 | −0.792 | −3.006 |
| | off | tracker | −0.639 | −0.454 | −1.093 |
| Piefke, free schema | on | truth | 4.000 | 2.000 | 6.000 |
| | on | tracker | 3.505 | 1.774 | 5.279 |
| | on | random | −0.685 | −0.792 | −1.478 |
| | off | tracker | −0.380 | −0.090 | −0.470 |

The tracker receives only the frame and remembers its own commands. It does not read the schema.

**What this means:**

- The task is solvable: a policy with access to the truth scores exactly 6 on all seeds.
- The window is needed: without it the tracker falls from 5.33 to −1.09 (control C0).
- A schema is not required for this. The hand-written tracker reaches 3.55 out of 4 on tracking from the memory of its own commands alone. The trained DQN with a schema in the paper reaches about 3.5. In this setup the attention mechanism is deterministic, so the window position is fully determined by the starting position and the commands. This does not mean that a learning network without a schema will find such a solution: step 2 answers that.

The identical catch of the random policy in both variants is not a coincidence: the stream is the same, and the paddle bit is determined by the same generator value.

**Readability of the window from one frame.** A hand-written rule: a candidate is a 3 × 3 block with at most one black pixel, and a frame is readable if all candidates lie in the window's column.

| Noise | 0 | 0.25 | 0.5 | 0.75 | 1 |
| :-- | --: | --: | --: | --: | --: |
| Share of readable frames | 0 | 0 | 0.204 | 0.891 | 1 |

This is a lower bound: a learning network can use weaker statistics than this rule.

**Agreement with the published rules.** The environment was checked against a separate hand transcription of the rules of both published codes at zero noise: 324,000 steps with no mismatch in state, reward or frame. There is one known difference: when the ball starts in an edge column, the Wilterson code does not clip the window's starting column, while the bench follows the paper.

To reproduce: `npm run level0:check`.

## Level 0: DQN, check before the full run

The bench's own DQN repeats the published setup. The published notebooks store their training output, so the implementation can be compared with the authors' real curves from the first iterations.

Tracking at iteration 30, 5 subjects, mean and standard error; for the authors, the mean of iterations 28 to 32:

| Condition | Authors | Bench with the first frame as in the authors' code | Bench with the first frame as in the paper |
| :-- | --: | --: | --: |
| With schema | −0.29 | −0.20 ± 0.06 | 0.37 ± 0.05 |
| Without schema | −0.64 | −0.62 ± 0.09 | −0.46 ± 0.11 |

**What this means:**

- The implementation is correct: with the first frame reproduced, the bench lands on the authors' curves. For this check the behavior of the authors' code was reproduced in a temporary copy.
- The first-frame bug in the published code is substantial. Fixing it raises learning with the schema by about 0.57 already by iteration 30, and the gap between conditions grows from 0.43 to 0.83. A likely reason is that with the corrected frame the schema shows the ball's starting column at once. This is a supposition that was not tested separately.
- The main variant of the bench is free of this bug, so the authors' reference values serve as a qualitative guide. Matching numbers are not expected.

**Independent check of components.** The gradient matched central differences (largest difference 3·10⁻¹⁰), Adam matched the TensorFlow 2.1 formula over 400 steps, the buffer collected 111 episodes in 1000 steps, and all pairs are correct.

To reproduce: `npm run level0:train -- --subjects 5 --iterations 30 --eval-every 10 --conditions schema,none`.

## Level 0: full run

**Setup:**

- 3 conditions with 5 subjects each, 1500 iterations, a test every 10 iterations with 200 episodes.
- The first frame as in the paper. The "Commands" condition is our addition: instead of a schema the network receives the number of its last command.
- Numbers are the mean of the last 10 tests (iterations 1410 to 1500) for each subject, then the mean and standard error over subjects. Individual tests fluctuate strongly. For the authors, the mean of the last 100 iterations.

| Training | Test | Tracking out of 4 | Catch out of 2 | Authors: tracking | Authors: catch |
| :-- | :-- | --: | --: | --: | --: |
| Schema | Normal | 3.57 ± 0.01 | 1.04 ± 0.06 | 3.38 | 0.83 |
| Schema | Window off | 0.23 ± 0.13 | −0.46 ± 0.04 | 0.15 | −0.42 |
| Schema | Schema off | −0.69 ± 0.25 | −0.14 ± 0.02 | −0.46 | −0.22 |
| No schema | Normal | 1.13 ± 0.06 | −0.25 ± 0.04 | 0.34 | −0.26 |
| No schema | Window off | −0.50 ± 0.02 | −0.47 ± 0.04 | −0.49 | −0.43 |
| Commands | Normal | 1.76 ± 0.06 | −0.25 ± 0.09 | – | – |
| Commands | Window off | −0.30 ± 0.04 | −0.54 ± 0.06 | – | – |
| Commands | Commands off | 0.97 ± 0.06 | −0.37 ± 0.07 | – | – |

Paired differences in the normal test, 5 subjects:

| Comparison | Tracking | Catch | Same sign for all subjects |
| :-- | --: | --: | :-- |
| Schema minus no schema | 2.45 ± 0.05 | 1.29 ± 0.06 | Yes, in both columns |
| Schema minus commands | 1.81 ± 0.06 | 1.29 ± 0.08 | Yes, in both columns |
| Commands minus no schema | 0.64 ± 0.05 | 0.00 ± 0.10 | Only for tracking |

**Verdict:**

- **The published result is reproduced qualitatively.** With a schema the agent learns much better than without it. Turning the schema off after training destroys tracking, and turning the window off hurts the catch more, as in the paper.
- **Command history did not replace the schema.** Within this budget it gave only part of the gain in tracking and nothing in the catch.
- **The schema's role as a model of attention is not proven here.** The hand-written tracker reaches 3.55 and 1.78 on the same information as the "Commands" condition. So the information is sufficient, and the difference lies in how easily the network can use it. The attention mechanism in this setup is deterministic, and the schema equals the sum of commands. A network without memory sums poorly by itself and gets a ready answer. A model of attention can be told apart from a command integrator only where the command does not determine the focus, which means the target mode.

**Limitations:**

- Training without a schema and with commands is still improving at the end: by 0.17 and 0.22 over the last 300 iterations, against 0.09 with the schema. The gap may shrink with longer training.
- The conditions differ in form: the schema is given as a map aligned with the field, the command as an action number.
- The corrected first frame probably raised the result without a schema: 1.13 against the authors' 0.34. The published gap may therefore be partly inflated by the first-frame bug. There was no full-length run with the bug, so this is a supposition.
- 5 subjects. The best catch result is below the hand-written tracker: 1.04 against 1.78.

To reproduce: `npm run level0:train -- --subjects 5 --iterations 1500 --eval-every 10 --workers 8 --out runs/level0`, about 6.5 hours on 8 threads.
