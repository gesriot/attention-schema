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

## Level 0: E6 checks before the full run

The probes were tested on cases with a known answer, using the truth policy at noise 0.5.

| Resource | Accuracy | Lift over the most frequent class |
| :-- | --: | --: |
| Exact copy of the window | 1.000 | 1.000 |
| Never leaves its starting column | 0.411 | 0.317 |
| Random actions | 0.249 | 0.130 |
| Time only, no resource | 0.168 | 0.036 |

**A cheap bypass.** A motionless square stores the starting column, which predicts the window's column because the window follows the ball. So an agent that learned nothing about the resource would beat the random baseline. E6 therefore has a second baseline, a motionless resource, both for usefulness and for the probe. Lift is used because label frequencies differ between sources.

**Difficulty of reading the window's column from the stimulus**, truth policy:

| Noise | 0 | 0.25 | 0.5 | 0.75 | 1 |
| :-- | --: | --: | --: | --: | --: |
| Probe accuracy | 0.750 | 0.440 | 0.629 | 0.860 | 1.000 |

The difficulty peaks at p = 0.25, as in Piefke et al. At p = 0 the window itself is invisible, and the column is read from the ball, which the truth policy tracks.

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

## Level 0: E6, first stage at noise 0.5

**Setup:** the level 0 DQN instead of the PPO of Piefke et al., one frame per observation, discount 0.9, 3 conditions with 5 subjects each, 1500 iterations, a test every 10 iterations with 200 episodes. Numbers are the mean of the last 10 tests for each subject, then the mean and standard error over subjects. Piefke et al. trained one agent per noise level.

| Condition | Test | Tracking out of 4 | Catch out of 2 | Piefke et al. |
| :-- | :-- | --: | --: | --: |
| Built-in schema | Normal | 3.84 ± 0.02 | 1.67 ± 0.02 | 3.73 / 1.77 |
| No resource | Normal | −0.30 ± 0.04 | −0.81 ± 0.02 | −0.39 / −0.84 |
| Free resource | Normal | 1.09 ± 0.10 | −0.78 ± 0.01 | 3.74 / 1.73 |
| Free resource | Random resource actions | 0.64 ± 0.10 | −0.81 ± 0.02 | 0.93 / −0.03 |
| Free resource | Motionless resource | 0.75 ± 0.11 | −0.79 ± 0.02 | – |
| Free resource | Resource off | −0.55 ± 0.05 | −0.80 ± 0.02 | −0.82 / −0.89 |

Usefulness of the resource, paired differences for tracking: 0.45 ± 0.05 against random actions and 0.34 ± 0.03 against a motionless resource. For the catch it is 0.03 ± 0.03 and 0.01 ± 0.02.

Probes of the window's column, as lift over the most frequent class:

| Source | Accuracy | Lift |
| :-- | --: | --: |
| Stimulus and resource | 0.801 ± 0.008 | 0.745 ± 0.003 |
| Stimulus | 0.642 ± 0.013 | 0.542 ± 0.014 |
| Motionless resource | 0.551 ± 0.029 | 0.405 ± 0.021 |
| Untrained agent | 0.540 ± 0.064 | 0.274 ± 0.032 |
| **Resource** | 0.416 ± 0.023 | 0.252 ± 0.021 |
| Random resource | 0.359 ± 0.028 | 0.159 ± 0.009 |
| Time only | 0.257 ± 0.025 | 0.048 ± 0.011 |

**Verdict:**

- **The launch condition is met.** The built-in schema learns and the agent without a resource does not, and both land close to Piefke et al. Discount 0.9 without a separate target network caused no instability.
- **Emergence is not shown at 1500 iterations.** The agent's own resource carries less information about the window than a square left in place: lift 0.25 against 0.41. Against the random baseline alone (0.16) this would have looked like emergence, so the motionless baseline changed the conclusion.
- **The resource gives a partial gain in tracking and none in the catch.** Its active use is worth more than leaving it in place, but what it stores is not established.
- **Training has not converged.** Tracking with the free resource went from 0.62 at iteration 1000 to 0.98 at 1250 and 1.07 at 1500. Piefke et al. report 5797 training epochs at this noise level.

**Limitations:**

- The combined probe of stimulus and resource (0.745 against 0.542) has no motionless baseline, so it does not show that the learned resource adds anything a parked square would not.
- The built-in schema here beats the level 0 full run (3.84 and 1.67 against 3.57 and 1.04), although it has no frame memory. The discount and the input differ, and the cause was not tested.

To reproduce: `npm run level0:emergence -- --conditions free,given,off --noise 0.5 --subjects 5 --iterations 1500 --eval-every 10 --workers 8 --out runs/emergence`, about 3.3 hours on 8 threads.

## Level 0: E6, 6000 iterations at noise 0.5

**Setup:** the free resource only, otherwise as in the first stage, 5 subjects, 6000 iterations. The number comes from the 5797 training epochs reported by Piefke et al.; their epochs are not the same unit as our iterations.

| Test | Tracking out of 4 | Catch out of 2 |
| :-- | --: | --: |
| Normal | 2.80 ± 0.03 | −0.23 ± 0.16 |
| Motionless resource | 2.54 ± 0.05 | −0.30 ± 0.13 |
| Random resource actions | 2.39 ± 0.06 | −0.33 ± 0.12 |
| Resource off | −0.53 ± 0.07 | −0.84 ± 0.03 |
| Window off | −0.30 ± 0.03 | −0.88 ± 0.03 |

For comparison, the built-in schema reached 3.77 and 1.64 by iteration 750. Normal tracking is close to a plateau: 2.75 at iteration 5000 and 2.80 at 6000.

Usefulness of the resource, paired differences: 0.41 ± 0.08 for tracking against random actions, 0.26 ± 0.07 against a motionless resource. For the catch it is 0.10 ± 0.04 and 0.07 ± 0.04.

Probes, as lift over the most frequent class:

| Source | Lift |
| :-- | --: |
| Stimulus and resource | 0.738 ± 0.012 |
| Stimulus | 0.502 ± 0.009 |
| Motionless resource | 0.316 ± 0.010 |
| **Resource** | 0.271 ± 0.009 |
| Random resource | 0.132 ± 0.008 |
| Time only | 0.026 ± 0.002 |

**Verdict:**

- **Emergence of a schema is not shown on this DQN.** The resource carries no more information about the window's column than a square left in place: it is lower in 4 of 5 subjects and equal in one. This does not contradict Piefke et al., who used another algorithm.
- **The agent depends on the resource, but barely on where it moves it.** Removing the field drops tracking from 2.80 to −0.53. Replacing its column moves leaves 2.39 to 2.54.

**A likely explanation that was not tested.** All actions in this variant move down, so the square's row equals the time step, and a parked square also keeps the starting column. An agent without memory gets a clock from the field for free. The random-action control keeps that clock too, so it cannot detect this. The same may apply to the large gap in Piefke et al. between removing the resource and randomizing its actions, but that is a supposition.

**Limitations:** one noise level, 5 subjects, no agent without a resource at this length, and the combined probe has no motionless baseline.

To reproduce: `npm run level0:emergence -- --conditions free --noise 0.5 --subjects 5 --iterations 6000 --eval-every 10 --workers 5 --out runs/emergence-6000`, about 6 hours.

## Planned exploratory tests

Written on 5 November 2026 before any of these runs started, and not changed afterward. All are exploratory: 5 subjects each, while the specification requires 15 to 30 pairs for confirmatory comparisons.

An effect means a paired difference in tracking of at least 0.3 whose 95% interval (mean ± 2.78 standard errors) excludes zero.

**T1. Window slips.** With probability 0.3 per step the window moves one extra cell past the command, at level 0 with 1500 iterations and conditions schema, commands, none. The rate was chosen on hand policies only (tracking at slip 0 and 0.3): truth 4.00 and 3.20, command tracker 3.55 and 1.42, tracker reading the schema field 4.00 and 2.46.

- Supported if schema minus commands is an effect and commands minus none is below 0.3. It was 0.64 without slips.
- Not supported if commands minus none stays at 0.3 or more: command memory still helps when commands do not determine the focus.

**T2. Clock.** At noise 0.5 with 6000 iterations and the seeds of the free resource run, the field shows only the time step (`clock`), only the starting column (`start`), or a parked square with both (`parked`). The action space has 6 actions instead of 18, which favors these conditions.

- The clock explanation is supported if `clock` comes within 0.3 of the free resource (2.80).
- It is refuted if the free resource beats `clock` by an effect.

**T3. Active use of the resource.** Same runs.

- Active writing helps if the free resource beats `parked` by an effect.
- It does not if `parked` matches or beats it.

**T4. First-frame bug at full length.** Conditions schema and none at level 0 with the bug reproduced in a temporary copy, 1500 iterations.

- The published gap is inflated by the bug if the no-schema result with the bug is at least 0.3 below the bench's 1.13, with an interval excluding zero by unpaired standard errors.
- It is not if the two no-schema results differ by less than 0.3.
