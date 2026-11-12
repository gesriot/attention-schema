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
- The corrected first frame probably raised the result without a schema: 1.13 against the authors' 0.34. The published gap may therefore be partly inflated by the first-frame bug. Test T4 below confirms this. There was no full-length run with the bug, so this is a supposition.
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

## T1: window slips

Exploratory, 5 subjects, 1500 iterations, slip probability 0.3. Numbers are the mean of the last 10 tests.

| Training | Test | Tracking out of 4 | Catch out of 2 |
| :-- | :-- | --: | --: |
| Schema | Normal | 0.94 ± 0.06 | −0.23 ± 0.07 |
| Schema | Schema off | −0.99 ± 0.04 | −0.45 ± 0.04 |
| Commands | Normal | −0.59 ± 0.12 | −0.36 ± 0.07 |
| Commands | Commands off | −0.62 ± 0.10 | −0.40 ± 0.08 |
| No schema | Normal | −0.63 ± 0.07 | −0.39 ± 0.03 |

| Paired difference in tracking | Value | 95% interval | Without slips |
| :-- | --: | --: | --: |
| Schema minus commands | 1.53 ± 0.13 | 1.16 to 1.90 | 1.81 |
| Commands minus no schema | 0.04 ± 0.13 | −0.31 to 0.40 | 0.64 |

**Verdict: the prediction is supported.** The schema keeps a large advantage, while command memory loses all its value once commands stop determining the focus. Turning the commands off changes nothing, which shows the network no longer relies on them.

**Limitations:**

- The schema here is a given, error-free map of the actual window, not a learned model. This shows that information about the actual state of attention matters, not that a model built by the agent provides it.
- Training with the schema is still improving: 0.74 at iteration 1200 and 0.94 at 1500. The hand-written tracker that reads the schema field reaches 2.46.
- The catch does not separate the conditions.

## T2: clock

Exploratory, 5 subjects, noise 0.5, 6000 iterations, the seeds of the free resource run. The field shows only the time step and carries nothing about the window or the target.

| Test | Tracking out of 4 | Catch out of 2 |
| :-- | --: | --: |
| Normal | 2.75 ± 0.05 | 0.44 ± 0.06 |
| Field off | −0.48 ± 0.03 | −0.79 ± 0.04 |
| Window off | −0.57 ± 0.05 | −0.86 ± 0.01 |

The free resource minus `clock` in tracking is 0.05 ± 0.02, with a 95% interval of −0.02 to 0.11.

**Verdict: the clock explanation is supported.** A field holding nothing but the time step does the job of the free resource: the difference is well below the 0.3 threshold. The fall of about 3.3 when the field is switched off in testing therefore does not show that the agent built a model of its attention. An agent that sees one frame cannot tell the time without help, and the square's row tells it.

**Limitations:**

- Equal scores do not prove that the free agent uses its resource as a clock, only that a clock is enough to explain them. The `start` and `parked` conditions and the probes bear on this.
- The result holds for this level 0 setting, with one frame of input and a window that only moves down.

## T4: first-frame bug at full length

Exploratory, 5 subjects, 1500 iterations. The bug was reproduced in a temporary copy of the bench, which is not part of the repository. Numbers are the mean of the last 10 tests.

| Measure | Authors | Bench with the bug | Bench without the bug |
| :-- | --: | --: | --: |
| Schema, tracking | 3.38 | 3.37 ± 0.04 | 3.57 ± 0.01 |
| Schema, catch | 0.83 | 0.61 ± 0.06 | 1.04 ± 0.06 |
| No schema, tracking | 0.34 | 0.29 ± 0.06 | 1.13 ± 0.06 |
| No schema, catch | −0.26 | −0.36 ± 0.03 | −0.25 ± 0.04 |
| Gap in tracking | 3.04 | 3.09 | 2.44 |

With the bug, tracking without a schema falls by 0.84 ± 0.09, with a 95% interval of 0.60 to 1.07 by unpaired standard errors. Tracking with the schema falls by 0.20 ± 0.04.

**Verdict: the published gap is inflated by the bug.** Reproducing it brings the bench to the authors' tracking numbers in both conditions. Fixing it removes 0.65 of the gap, about a fifth. The schema still wins by a wide margin, so the main finding of the paper stands at a smaller size.

**Limitation:** the catch agrees less well with the authors than tracking does, with a difference of 0.22 with the schema.

## Planned replication

Registered before the run. T1 was exploratory, so it is repeated on subjects 6 to 10, whose seeds have not been used, with the same settings: slip probability 0.3, 1500 iterations, conditions schema, commands and none. Subjects 1 to 5 do not enter the verdict.

- The result replicates if schema minus commands is an effect and commands minus no schema is below 0.3 in tracking.
- It fails if either condition is not met.

## Planned tests of causal role and of emergence

Registered before the runs. Both are exploratory, with 5 subjects.

**T5. False schema.** Level 0 without slips, with the schema trained on subjects 1 to 5. Their seeds are reused and the curves must match the stored run. After training, on decisions 1 to 8 of 2000 new episodes, the schema in the input is moved sideways by 2 columns through the whole frame history, and nothing else changes. The measure, on the same decision, is the change of the column command against the shift, in columns. The real window, the frame and the network are untouched, so a change can only come from the schema. The same measure on each subject's untrained starting network is the control.

- The schema steers attention if, for the shift to each side separately, the measure is at least 0.2 and its 95% interval excludes zero, and the paired difference from the untrained network meets the same two conditions.
- It does not if the measure is below 0.2 on either side. Degradation without a direction does not count.
- Shifts of 1 and 3 columns, a shift of the latest frame only, and the consequences of one false step at step 5 are reported without a verdict.

**T6. Free resource without a free clock.** Noise 0.5, 6000 iterations, the seeds of the free resource run. The square stays on row 4 and moves only sideways, so it cannot count time by falling (`locked`). The comparison is `start` (2.04), where the same column information comes without any action.

- The resource is put to use if `locked` beats `start` by an effect, and also beats a motionless resource by an effect in the same network.
- It is not if `locked` minus `start` is below 0.3. This would say nothing about longer training.
- Probes of the window column are reported without a verdict.

## T3: active use of the resource

Exploratory, 5 subjects, noise 0.5, 6000 iterations, the seeds of the free resource run. The field in `parked` is what the free square shows when it is never moved sideways.

| Field shows | Tracking out of 4 | Catch out of 2 |
| :-- | --: | --: |
| Starting column (`start`) | 2.04 ± 0.05 | −0.74 ± 0.05 |
| Time step (`clock`) | 2.75 ± 0.05 | 0.44 ± 0.06 |
| Both (`parked`) | 3.38 ± 0.04 | 0.41 ± 0.06 |
| Free resource | 2.80 ± 0.03 | −0.23 ± 0.16 |

| Paired difference in tracking | Value | 95% interval |
| :-- | --: | --: |
| Free resource minus `parked` | −0.59 ± 0.06 | −0.75 to −0.42 |
| `parked` minus `clock` | 0.63 ± 0.08 | 0.42 to 0.84 |
| Free resource minus `clock` | 0.05 ± 0.02 | −0.02 to 0.11 |

**Verdict: active writing does not help.** A square that stays in place beats the one the agent controls by an effect. The agent scores less than it would by never moving the square sideways, and it does not collect the 0.63 that the starting column adds to the time step.

**Limitations:**

- The free agent chooses among 18 actions and `parked` among 6, and exploration moves the free square at random. Both make learning harder for the free agent, so this does not show that it could not get there with longer training.
- `parked` is still rising: 3.29 at iteration 5000 and 3.38 at 6000.
- No agent has yet been trained without any field at this length. The table has no floor, and the score of the network with the field switched off in testing is not one.

The noise grid was conditional on a positive result here, so it is not run.

**Planned test T7. Floor.** Registered before the run. Noise 0.5, 6000 iterations, the same seeds, trained with an empty field (`off`).

- The gains of `clock` and `parked` stand if each beats `off` by an effect.
- The starting column adds something on its own if `start` beats `off` by an effect.

## T1 replication

Subjects 6 to 10, new seeds, settings and criteria registered in advance. Numbers are the mean of the last 10 tests.

| Training | Tracking out of 4 | Catch out of 2 |
| :-- | --: | --: |
| Schema | 0.75 ± 0.07 | −0.28 ± 0.08 |
| Commands | −0.66 ± 0.09 | −0.51 ± 0.03 |
| No schema | −0.76 ± 0.08 | −0.43 ± 0.07 |

| Paired difference in tracking | First run | Replication | 95% interval | All 10 subjects |
| :-- | --: | --: | --: | --: |
| Schema minus commands | 1.53 | 1.41 ± 0.14 | 1.02 to 1.79 | 1.47 ± 0.09 |
| Commands minus no schema | 0.04 | 0.11 ± 0.11 | −0.20 to 0.42 | 0.07 ± 0.08 |

**Verdict: the result replicates.** Both registered conditions are met on subjects that took no part in forming the hypothesis. Once the window slips, memory of commands is worth nothing measurable, and information about the actual window keeps its full value.

**Limitations:** the upper end of the interval for commands minus no schema is 0.42 on the new subjects alone, so a small value of commands is not excluded there. On all 10 subjects the interval is −0.11 to 0.26. The schema is still given, not learned.

## T5: false schema

Exploratory, 5 subjects, 2000 episodes per network. Training repeated the stored runs exactly on all 5 subjects. The measure is the change of the column command against the shift, in columns, so a positive value means compensation and a negative value means the command moves with the shift.

| Shift of the field | Whole history | Latest frame only | Untrained, whole history |
| --: | --: | --: | --: |
| −3 | −0.10 ± 0.07 | −0.28 ± 0.04 | 0.01 ± 0.03 |
| −2 | −0.15 ± 0.06 | −0.29 ± 0.03 | 0.01 ± 0.03 |
| −1 | −0.10 ± 0.04 | −0.11 ± 0.02 | 0.01 ± 0.02 |
| 1 | −0.18 ± 0.04 | −0.18 ± 0.02 | 0.01 ± 0.01 |
| 2 | −0.28 ± 0.04 | −0.40 ± 0.04 | 0.02 ± 0.03 |
| 3 | −0.24 ± 0.05 | −0.41 ± 0.05 | 0.02 ± 0.06 |

One false step at step 5, with the shift through the whole history:

| Shift | On target next step: clean | False | Tracking after it: clean | False |
| --: | --: | --: | --: | --: |
| −2 | 0.95 | 0.69 | 1.61 | 0.99 |
| 2 | 0.96 | 0.74 | 1.63 | 1.13 |

The untrained networks do not change on either measure.

**Verdict: the registered prediction fails.** The measure is below 0.2 on both sides, so the command does not compensate for the false position. Its sign is the opposite of the prediction: at a shift of 2 the command follows the false square, significantly so for the latest frame on both sides and for the whole history at +2, with the interval at −2 including zero.

The schema does act on control. One false step takes the window off the target in about a quarter of the cases where it was on it, and the loss continues on later steps. What fails is the idea of how it acts: the network does not behave as if it read a position from the field and steered by the difference.

**An explanation found after the fact, not tested here.** The square also shows the network where the noise-free cells are. Under a false square lie noise cells that may read as a stimulus on the shifted side, and that would pull the command toward the shift. T8 tests this. The stronger effect for the latest frame may have a second source, a jump in the window's history read as motion.

**Limitations:** the measure counts only column changes, one shift of 2 columns is outside what the window can do in one step when applied to the latest frame only, and the 95% interval at −2 for the whole history reaches zero.

**Planned test T8. Why the command follows.** Registered before the run. The same subjects and seeds. The false square is shifted through the whole history as in T5, and the visual cells under it are blanked, except row 9 and the cells of the real window. Measured for shifts of 2 on each side, paired with T5's whole-history measure.

- The explanation holds if blanking raises the measure by at least 0.2 on each side, with the 95% interval excluding zero.
- It fails if the rise is below 0.2 on either side.
- Compensation appears after all if the blanked measure is at least 0.2 on each side, with the 95% interval excluding zero.

## T7: floor

Exploratory, 5 subjects, noise 0.5, 6000 iterations, the seeds of the free resource run, trained with an empty field.

Tracking is 0.91 ± 0.07 and the catch −0.65 ± 0.05. Paired differences from this floor:

| Field shows | Tracking | 95% interval | Catch |
| :-- | --: | --: | --: |
| Starting column (`start`) | 1.13 ± 0.11 | 0.83 to 1.43 | −0.09 ± 0.10 |
| Time step (`clock`) | 1.84 ± 0.11 | 1.52 to 2.15 | 1.09 ± 0.09 |
| Free resource | 1.88 ± 0.10 | 1.61 to 2.16 | 0.42 ± 0.12 |
| Both (`parked`) | 2.47 ± 0.04 | 2.35 to 2.58 | 1.07 ± 0.09 |

**Verdict: the gains stand.** `clock`, `parked` and `start` each beat the floor by an effect. The real gain from the time step is about 1.8, not the 3.3 lost when the field is switched off in testing.

**Limitation:** the floor is still rising fast, from 0.46 at iteration 5000 to 0.91 at 6000, so these gains are measured at one length of training and may shrink with a longer one.

## T6: free resource without a free clock

Exploratory, 5 subjects, noise 0.5, 6000 iterations, the seeds of the free resource run.

| Test of `locked` | Tracking out of 4 | Catch out of 2 |
| :-- | --: | --: |
| Normal | 1.31 ± 0.11 | −0.75 ± 0.02 |
| Motionless resource | 1.29 ± 0.18 | −0.75 ± 0.01 |
| Random resource actions | 1.02 ± 0.14 | −0.76 ± 0.02 |
| Field off | −0.13 ± 0.09 | −0.80 ± 0.02 |

| Paired difference in tracking | Value | 95% interval |
| :-- | --: | --: |
| `locked` minus `start` | −0.74 ± 0.07 | −0.94 to −0.54 |
| `locked` minus motionless | 0.02 ± 0.07 | −0.18 to 0.22 |
| `locked` minus the floor (T7) | 0.39 ± 0.15 | −0.02 to 0.81 |

The probe of the window column gives a lift of 0.21 ± 0.05 from the agent's own square and 0.14 ± 0.04 from a motionless one, with a paired difference of 0.07 ± 0.03 and a 95% interval of −0.03 to 0.16.

**Verdict: the resource is not put to use.** `locked` loses to `start`, where the same column comes without any action, and forcing the square to stay in place costs it nothing.

**Limitations:** the agent has 18 actions against 6, and tracking is still rising, from 1.03 at iteration 5000 to 1.31 at 6000. This says nothing about longer training.

Taken together, T2, T3 and T6 give no sign of a schema emerging in this agent within 6000 iterations. The field helps when something useful is written there for the agent, and the agent does not write it itself.

## Planned test: a schema the agent works out

Registered before the run. Every schema so far was given by the experimenter, so the results do not show that an agent can supply the information itself.

**T9. Schema from the frame.** Level 0, 5 subjects, 1500 iterations, without slips and at slip probability 0.3, the seeds of the earlier runs. The field shows an estimate instead of the true window (`read`). Of the blocks with at most the ball lit, the estimate is the one nearest the position predicted from the last estimate and the last command. The true window is not read. Under hand-written and random policies the estimate matches the window on 0.99 of frames without slips and on 0.79 to 0.81 with them.

Each setting is judged on its own, by paired differences in tracking with the same subjects.

- The advantage does not need outside information if the given schema minus `read` is below 0.3 and `read` beats no schema by an effect.
- It does if the given schema beats `read` by an effect.

## T8: why the command follows

Exploratory, 5 subjects, 2000 episodes per network. Every whole-history and latest-frame record repeats T5 exactly. The measure is the one from T5, so a positive value means compensation.

| Shift of the field | Whole history | Noise blanked | Rise | 95% interval of the rise |
| --: | --: | --: | --: | --: |
| −2 | −0.15 ± 0.06 | 0.13 ± 0.05 | 0.27 ± 0.05 | 0.13..0.41 |
| 2 | −0.28 ± 0.04 | 0.07 ± 0.05 | 0.34 ± 0.04 | 0.23..0.46 |

**Verdict: the explanation holds at the registered shifts, and compensation does not appear.** Blanking raises the measure by at least 0.2 on each side, with the interval excluding zero. The blanked measure stays below 0.2 and its intervals, −0.02..0.27 and −0.07..0.20, include zero. At a shift of 2 the pull toward the false square came mostly from the noise under it. Without that noise the command does not follow the square, and it does not reliably steer by it either.

**Outside the registered shifts the explanation does not extend.** These numbers were not registered.

| Shift of the field | Whole history | Noise blanked | Rise | 95% interval of the rise |
| --: | --: | --: | --: | --: |
| −3 | −0.10 ± 0.07 | −0.17 ± 0.06 | −0.07 ± 0.06 | −0.24..0.09 |
| −1 | −0.10 ± 0.04 | 0.13 ± 0.02 | 0.23 ± 0.02 | 0.18..0.28 |
| 1 | −0.18 ± 0.04 | 0.09 ± 0.02 | 0.27 ± 0.02 | 0.21..0.33 |
| 3 | −0.24 ± 0.05 | −0.15 ± 0.08 | 0.09 ± 0.07 | −0.10..0.27 |

A false square at 1, 2 and 3 columns adds 3, 6 and 9 cells to blank, and the rise does not grow with that number: at 3 it is gone. A shift of 3 is the only one where the false square does not overlap the real window. Why that matters is not known, so noise under the square is at best part of the explanation.

**Limitations:** blanking also removes a ball that left the real window for that area, which is rare in a trained network, and the consequences of one false step were not measured with blanking.

## Planned test: a false schema when the field is the only reliable source

Registered before the run. T8 leaves the command neither following nor compensating. One reason may be that without slips the network also has the history of its own commands, so the field is one of two sources of position. Under slips that history misleads, and the field carries an advantage of 1.47 (T1).

**T10.** Level 0, 5 subjects, 1500 iterations, slip probability 0.3, the seeds of the slip run, 2000 episodes per network. The shift is the one from T8, with the noise under the false square blanked, at 2 columns on each side.

- Compensation appears if the measure is at least 0.2 on each side, with the 95% interval excluding zero.
- It does not if the measure is below 0.2 on either side.

## Planned test: how coarse the schema can be

Registered before the run. The spec treats the schema as a caricature, and its telemetry pools the resource map by a block size k that is still open. No result so far says how much detail the control function needs.

**T11.** Level 0, 5 subjects, 1500 iterations, slip probability 0.3, the seeds of the slip run. Window centres are split into blocks of k along each axis, and the field lights every cell that some window of the block would cover, for k of 2, 3 and 4. The field then says where attention is within 4, 3 or 2 blocks per axis, not the exact window.

Each k is judged on its own, by paired differences in tracking with the stored schema and no-schema runs of the same subjects.

- The advantage survives if the given schema minus the block is below 0.3 and the block beats no schema by an effect.
- Part of it is lost if the given schema beats the block by an effect and the block beats no schema by an effect.
- All of it is lost if the block does not beat no schema by an effect.

## T9 without slips

Exploratory, 5 subjects, the last 10 checkpoints. Paired with the stored runs of the same subjects.

| Condition | Tracking | Catch |
| :-- | --: | --: |
| Given schema | 3.57 ± 0.01 | 1.04 ± 0.06 |
| Schema from the frame | 3.49 ± 0.03 | 0.89 ± 0.07 |
| Commands | 1.76 ± 0.06 | −0.25 ± 0.09 |
| No schema | 1.13 ± 0.06 | −0.25 ± 0.04 |

| Difference in tracking | Mean | 95% interval |
| :-- | --: | --: |
| Given minus from the frame | 0.08 ± 0.02 | 0.04..0.13 |
| From the frame minus no schema | 2.36 ± 0.06 | 2.20..2.52 |
| From the frame minus commands | 1.73 ± 0.05 | 1.58..1.87 |

**Verdict without slips: the advantage does not need outside information.** The given schema leads by 0.08, below 0.3, although the interval excludes zero, and the schema from the frame beats no schema by 2.36.

**What this adds.** Everything the estimate uses is in what the commands network already sees: ten frames and its own commands. That network gains 0.64 over no schema, and the same information worked into a position gains 2.36. The advantage comes from the work of turning the history into a position, not from information the agent lacks, and within 1500 iterations the network does not do that work itself.

**Limitations:** the estimate is written by hand and is almost exact without slips, so the test shows the information is within the agent's reach, not that an agent can learn to extract it.

## T9 with slips

Exploratory, 5 subjects, slip probability 0.3, the last 10 checkpoints. Paired with the stored slip runs of the same subjects.

| Condition | Tracking | Catch |
| :-- | --: | --: |
| Given schema | 0.94 ± 0.06 | −0.23 ± 0.07 |
| Schema from the frame | 0.48 ± 0.06 | −0.48 ± 0.06 |
| Commands | −0.59 ± 0.12 | −0.36 ± 0.07 |
| No schema | −0.63 ± 0.07 | −0.39 ± 0.03 |

| Difference in tracking | Mean | 95% interval |
| :-- | --: | --: |
| Given minus from the frame | 0.46 ± 0.09 | 0.21..0.71 |
| From the frame minus no schema | 1.11 ± 0.03 | 1.03..1.20 |
| From the frame minus commands | 1.07 ± 0.12 | 0.73..1.42 |

**Verdict with slips: the advantage needs outside information in part.** The given schema beats the schema from the frame by 0.46, an effect. The schema from the frame still beats no schema by 1.11, about 70% of the given schema's 1.57.

Taken together, without slips the agent's own frames and commands are enough, and when attention also moves unbidden they fall short. The estimate then matches the window on about 0.8 of frames. The spec's telemetry, a direct coarse signal about attention, is the kind of source that would close this gap.

**Limitations:** catch does not follow tracking under slips. The schema from the frame catches no better than no schema, −0.09 (−0.29..0.12), while without slips it gains 1.14 (1.02..1.25). The reason is not known.

## Planned test: longer training of the locked resource

Registered before the run. T6 found the locked resource unused after 6000 iterations, with tracking still rising, and said nothing about longer training.

**T12.** Level 0, version B, 5 subjects, noise 0.5, 20000 iterations, the seeds of T6, so the first 6000 iterations should repeat it. The measure is the paired difference in tracking between `locked` and its own test with the resource held still, over the last 10 checkpoints. It needs no other condition trained as long.

- The resource comes into use if the difference is at least 0.3, with the 95% interval excluding zero.
- It stays unused if the difference is below 0.3.

The 6000-iteration results of `start` and `clock` are reference points, not criteria, because they would also rise with longer training. The probe lift from the agent's own square against a motionless one is reported alongside.

## T10: false schema under slips

Exploratory, 5 subjects, slip probability 0.3, 2000 episodes per network. Training repeated the stored slip runs exactly on all 5 subjects.

| Shift of the field | Whole history | Noise blanked | 95% interval, blanked |
| --: | --: | --: | --: |
| −2 | −0.20 ± 0.06 | 0.08 ± 0.04 | −0.03..0.19 |
| 2 | −0.15 ± 0.04 | −0.03 ± 0.04 | −0.15..0.09 |

One false step at step 5, with the shift through the whole history:

| Shift | On target next step: clean | False | Tracking after it: clean | False |
| --: | --: | --: | --: | --: |
| −2 | 0.52 | 0.44 | −0.23 | −0.38 |
| 2 | 0.54 | 0.45 | −0.16 | −0.35 |

**Verdict: compensation does not appear.** The blanked measure is below 0.2 on both sides and both intervals include zero. The command does not steer by the false square even when the field is the only reliable source of position.

The field still acts on control: one false step costs about a sixth of the steps on target. So, as in T5 and T8, the network uses the field, but not as a position from which it steers by the difference with the target. What it reads there instead is not known.

**Limitations:** without slips the window is on target on 0.95 of steps and under slips on about 0.53, so a measure built on single column commands is noisier here.

## Planned test: frame and telemetry together

Registered before the run. Under slips T9 left a gap of 0.46 between the given schema and one worked out from the frame and the agent's commands. The spec gives the schema one more input, telemetry, a coarse direct signal about attention, and requires that the position of attention follow from the inputs together but from none of them alone (F4).

**T13.** Level 0, 5 subjects, 1500 iterations, slip probability 0.3, the seeds of the slip run. The estimate of T9 is searched only within the block of 2 by 2 window centres that holds the real window (`read2`), so the telemetry is the block of T11 with k = 2. Under hand-written and random policies the estimate matches the window on 0.89 to 0.90 of frames, against 0.79 to 0.81 without telemetry. Without slips both reach 0.99.

Paired by subject with the given schema of the slip run, with the schema from the frame of T9, and with the block of 2 of T11. Each statement is judged on its own.

- The sources add up if `read2` beats both the schema from the frame and the block of 2 by an effect.
- The gap closes if the given schema minus `read2` is below 0.3.

## T11: coarse schemas

Exploratory, 5 subjects, slip probability 0.3, the last 10 checkpoints. Paired with the stored slip runs of the same subjects.

| Condition | Tracking | Catch |
| :-- | --: | --: |
| Given schema | 0.94 ± 0.06 | −0.23 ± 0.07 |
| Block of 2 | 0.15 ± 0.08 | −0.55 ± 0.04 |
| Block of 3 | 0.14 ± 0.10 | −0.52 ± 0.05 |
| Block of 4 | −0.30 ± 0.04 | −0.53 ± 0.04 |
| No schema | −0.63 ± 0.07 | −0.39 ± 0.03 |

| Difference in tracking | Given minus block | 95% interval | Block minus no schema | 95% interval |
| :-- | --: | --: | --: | --: |
| Block of 2 | 0.79 ± 0.14 | 0.40..1.18 | 0.78 ± 0.08 | 0.56..1.00 |
| Block of 3 | 0.80 ± 0.06 | 0.63..0.98 | 0.77 ± 0.14 | 0.37..1.16 |
| Block of 4 | 1.24 ± 0.08 | 1.02..1.45 | 0.34 ± 0.07 | 0.15..0.52 |

**Verdict: every block loses part of the advantage.** For each k the given schema beats the block by an effect, and the block beats no schema by an effect. Blocks of 2 and 3 keep about half of the given schema's 1.57 and the block of 4 about a fifth.

Not registered: blocks of 2 and 3 do not differ, 0.01 (−0.46..0.48), and both beat the block of 4 by about 0.44, with intervals excluding zero. Even the block of 2 falls behind the schema from the frame of T9, by 0.33 (0.08..0.59).

So under slips the control function needs the window to the cell. A schema that says only roughly where attention is helps, but much less. For the spec's telemetry this means a coarse block is a supplement to other sources and not a substitute for them, which T13 tests.

**Limitations:** blocks of 3 split the 8 centres unevenly, into 3, 3 and 2, and the test was run only under slips.

## Planned test: what the network reads from the field

Registered before the run. T5, T8 and T10 show that the field acts on control without being read as a position. One reading is that the square tells the network where in the frame to look for the ball. Under a false square without noise it then finds nothing and acts as if it had lost the ball. If so, putting the real window's view under the false square should remove most of the loss. If any unfamiliar field simply disrupts the network, it should not.

**T14.** Level 0, 5 subjects, 1500 iterations, no slips, the seeds of T8, 2000 episodes per network. One false step at step 5, with the field shifted by 2 columns on each side through the whole history. The visual cells under the false square are either blanked as in T8 or filled with what the real window shows, moved by the same shift (`carried`). The loss is the drop in the share of steps on target right after the false step. The recovery is the blanked loss minus the carried loss, paired by subject.

- The square marks where to look if on each side the recovery is at least half the blanked loss, with the 95% interval of the recovery excluding zero.
- It does not if on either side the recovery is below half the blanked loss.

The whole-history, latest-frame and blanked records of the command should repeat T8 exactly.

## T13: frame and telemetry together

Exploratory, 5 subjects, slip probability 0.3, the last 10 checkpoints. Paired with the earlier runs of the same subjects.

| Condition | Tracking | Catch |
| :-- | --: | --: |
| Given schema | 0.94 ± 0.06 | −0.23 ± 0.07 |
| Frame and telemetry (`read2`) | 0.77 ± 0.05 | −0.33 ± 0.07 |
| Schema from the frame (T9) | 0.48 ± 0.06 | −0.48 ± 0.06 |
| Block of 2 (T11) | 0.15 ± 0.08 | −0.55 ± 0.04 |
| No schema | −0.63 ± 0.07 | −0.39 ± 0.03 |

| Difference in tracking | Mean | 95% interval |
| :-- | --: | --: |
| `read2` minus the schema from the frame | 0.29 ± 0.09 | 0.04..0.53 |
| `read2` minus the block of 2 | 0.62 ± 0.12 | 0.28..0.95 |
| Given minus `read2` | 0.17 ± 0.05 | 0.02..0.32 |

**Verdict: the gap closes, and the claim that the sources add up fails narrowly.** The given schema leads `read2` by 0.17, below 0.3. `read2` beats the block of 2 by an effect, but its gain over the schema from the frame, 0.29, is just below 0.3, so by the registered rule it is not an effect.

Read with T9 and T11, the order is consistent: telemetry alone keeps about half of the advantage, the frame with commands about 70%, and both together about 90%. The estimate's accuracy rose from 0.8 to 0.9 with telemetry, and tracking followed it.

**Limitations:** the telemetry here is exact within its block, without the noise the spec gives it, and the threshold decided the first statement by 0.01.

## T14: what the network reads from the field

Exploratory, 5 subjects, 2000 episodes per network. All command records of T8 and its one-step consequences were repeated exactly.

| Shift | Loss on target: blanked | Carried | Recovery | 95% interval of the recovery |
| --: | --: | --: | --: | --: |
| −2 | 0.26 ± 0.02 | 0.03 ± 0.00 | 0.24 ± 0.02 | 0.17..0.30 |
| 2 | 0.24 ± 0.02 | 0.03 ± 0.01 | 0.21 ± 0.02 | 0.16..0.26 |

**Verdict: the square marks where to look.** On each side the recovery is far above half the blanked loss and its interval excludes zero. With the real window's view under the false square, the share of steps on target drops by 0.03 instead of 0.24 to 0.26, and the loss of tracking over the rest of the episode falls from about 0.65 to about 0.08.

Not registered: the pattern holds at every shift. The carried loss is 0.02 to 0.07 and the recovery is 0.08 to 0.24, all with intervals excluding zero. Under carrying the command still moves slightly with the shift, by 0.09 to 0.21 columns, also with intervals excluding zero.

Together with T5, T8 and T10, the network uses the field to know where in the frame its view is clean, and reads the ball there. A false square without the view makes it act as if the ball were lost. It does not take a position from the field and steer by the error, so a false schema causes a loss and not a directional compensation. The spec's E2A expects a compensating shift, and this bench does not show one at level 0.

**Limitations:** the carried view leaves the real ball visible outside the false square as well, so the network sees two balls on some frames.

## Planned test: coarse schemas without slips

Registered before the run. T11 was run only under slips, where the agent's own commands do not say where the window went. Without slips a coarse block and the command history together could fix the window exactly.

**T15.** As T11, but without slips, with the seeds of the full run. Each k of 2, 3 and 4 is judged on its own against the stored given and no-schema runs of the same subjects, with the three statements of T11.

## T12: longer training of the locked resource

Exploratory, 5 subjects, noise 0.5, 20000 iterations. The first 6000 iterations repeated T6 exactly on all 5 subjects. Each value is the mean of the 10 checkpoints up to the iteration shown.

| Iteration | Normal | Resource held still | Normal minus still | 95% interval |
| --: | --: | --: | --: | --: |
| 6000 | 1.31 ± 0.11 | 1.29 ± 0.18 | 0.02 ± 0.07 | −0.17..0.22 |
| 10000 | 1.86 ± 0.10 | 1.69 ± 0.12 | 0.18 ± 0.08 | −0.04..0.39 |
| 15000 | 2.21 ± 0.09 | 2.01 ± 0.11 | 0.19 ± 0.08 | −0.04..0.43 |
| 20000 | 2.32 ± 0.09 | 2.09 ± 0.11 | 0.22 ± 0.11 | −0.08..0.53 |

At 20000 iterations random resource actions cost 0.50 (0.33..0.66), and the probe lift from the agent's own square is 0.25 ± 0.07 against 0.14 ± 0.02 from a motionless one, a paired difference of 0.11 (−0.11..0.33).

**Verdict: the resource stays unused.** The difference at 20000 iterations is 0.22, below 0.3, and its interval includes zero.

The trend points toward use, though. Tracking rose from 1.31 to 2.32, and the advantage over the held square grew from 0.02 to about 0.2 by iteration 10000 and then changed little. Random resource actions hurt, but that shows only that the agent's actions on the square are not random, not that it reads the square back.

**Limitations:** one length was run, tracking still rises slowly, from 2.21 to 2.32 over the last 5000 iterations, and the comparison conditions were not trained as long.

## T15: coarse schemas without slips

Exploratory, 5 subjects, no slips, the last 10 checkpoints. Paired with the stored full runs of the same subjects.

| Condition | Tracking | Catch |
| :-- | --: | --: |
| Given schema | 3.57 ± 0.01 | 1.04 ± 0.06 |
| Block of 2 | 3.15 ± 0.03 | 0.73 ± 0.03 |
| Block of 3 | 2.85 ± 0.05 | 0.48 ± 0.07 |
| Block of 4 | 2.50 ± 0.05 | 0.30 ± 0.14 |
| No schema | 1.13 ± 0.06 | −0.25 ± 0.04 |

| Difference in tracking | Given minus block | 95% interval | Block minus no schema | 95% interval |
| :-- | --: | --: | --: | --: |
| Block of 2 | 0.42 ± 0.03 | 0.32..0.51 | 2.03 ± 0.06 | 1.87..2.18 |
| Block of 3 | 0.72 ± 0.05 | 0.58..0.85 | 1.73 ± 0.08 | 1.50..1.96 |
| Block of 4 | 1.07 ± 0.06 | 0.90..1.25 | 1.37 ± 0.10 | 1.09..1.66 |

**Verdict: every block loses part of the advantage, as under slips.** For each k the given schema beats the block by an effect, and the block beats no schema by an effect.

Without slips the blocks keep more of the given schema's 2.45: about 83%, 71% and 56% for k of 2, 3 and 4, against about 50%, 49% and 21% under slips. Not registered: here each finer block beats the next coarser one, by 0.30 (0.12..0.48) and 0.36 (0.15..0.56). The block of 2 still falls behind the schema from the frame of T9, by 0.34 (0.20..0.47).

So the agent's own commands make up for part of the lost detail when they are reliable, and much less when the window slips. In neither case does a coarse block replace the exact position.

## Target mode: feasibility controls at the starting values

`npm run target:check -- --episodes 2000`, seeds 5000000001 to 5000002000. The starting values are those of the specification step, not tuned.

| Control | Correct answers | Catch |
| :-- | --: | --: |
| F2: ideal policy | 0.789 | 1.000 |
| C0: no enhancement | 0.460 | 1.000 |

Chance is 0.25. Catch in both controls follows the truth and only checks the paddle. F6 at the ideal policy: 69% of flashes captured, 2.5 exits of the target from the focus per episode, the target in focus on 60% of steps, a gap of 0.88 between the commanded and the actual center.

C0 is well above chance, so there is too little noise, and F2 misses one answer in five.

## Planned tuning of the target world

Registered before the sweep. The spec allows the world's values to be refined only on the feasibility controls. The readiness conditions are fixed here:

- F2 answers correctly in at least 0.85 of episodes.
- C0 answers correctly in at most 0.35 of episodes.
- F6 under the ideal policy: between 20% and 80% of flashes captured, at least 1 exit of the target from the focus per episode, and a mean gap of at least 0.3 between the commanded and the actual center.
- F6 under the neutral policy: the target in focus on at most 30% of steps.

Only the measurement noise, the enhancement, and the rate and strength of flashes are varied, in that order of preference, on the control seeds. The mechanism's own constants stay as they are unless no setting of these meets the conditions. Every setting tried is recorded.

## Tuning of the target world

All settings tried, 2000 episodes each on the control seeds, in the order tried. Unlisted values are the starting ones.

| Setting | F2 | C0 | Flashes captured | Exits per episode | Command gap | Neutral in focus |
| :-- | --: | --: | --: | --: | --: | --: |
| Starting values | 0.789 | 0.460 | 0.69 | 2.53 | 0.88 | 0.15 |
| σ_η 1.5 | 0.707 | 0.382 | 0.69 | 2.53 | 0.88 | 0.15 |
| σ_η 2 | 0.644 | 0.350 | 0.69 | 2.53 | 0.88 | 0.15 |
| σ_η 2, α 0.95 | 0.756 | 0.350 | 0.69 | 2.53 | 0.88 | 0.15 |
| σ_η 1.5, flashes 0.05 | 0.746 | 0.382 | 0.67 | 2.49 | 0.73 | 0.15 |
| σ_η 2, flashes 0.05 | 0.682 | 0.350 | 0.67 | 2.49 | 0.73 | 0.15 |
| σ_η 2, flashes 0.05, α 0.95 | 0.799 | 0.350 | 0.67 | 2.49 | 0.73 | 0.15 |
| σ_η 2, no flashes, α 0.95 | 0.892 | 0.350 | – | 2.34 | 0.47 | 0.17 |
| σ_η 2, α 0.95, flashes 0.05, strength 0.5..1 | 0.868 | 0.350 | 0.37 | 2.54 | 0.56 | 0.16 |
| σ_η 2, α 1, flashes 0.05 | 0.847 | 0.350 | 0.67 | 2.49 | 0.73 | 0.15 |
| σ_η 2, α 0.95, flashes 0.05, strength 0.3..1 | 0.875 | 0.350 | 0.28 | 2.50 | 0.54 | 0.16 |
| σ_η 2, α 1, flashes 0.05, strength 0.3..1 | 0.924 | 0.350 | 0.28 | 2.50 | 0.54 | 0.16 |
| σ_η 2.5, α 0.95, flashes 0.05, strength 0.5..1 | 0.833 | 0.327 | 0.37 | 2.54 | 0.56 | 0.16 |
| σ_η 2.5, α 0.95, flashes 0.05, strength 0.3..1 | 0.837 | 0.327 | 0.28 | 2.50 | 0.54 | 0.16 |

**Chosen:** σ_η 2, α 0.95, flash probability 0.05, flash strength 0.5 to 1. It meets every condition with the fewest changes: F2 0.868, C0 0.350, 37% of flashes captured, 2.5 exits per episode, a gap of 0.56, the target in focus on 16% of steps under the neutral policy. The settings with α of 1 were set aside, because they make the focus noise-free. These are now the defaults. The mechanism's own constants did not change.

**Limitations:** C0 sits exactly on its bound, and it is an ideal observer that takes the target's place and the time of the jump from the truth. An agent without attention would also have to find the target in the noise, so it would do worse.

## Planned: control F4

Registered before the run. Probes recover the center of the resource map, $\bar c_t$, from the history of one source under the neutral policy, as row and column classifiers. The measure is the mean distance in cells between the predicted cell and $\bar c_t$ on held-out episodes.

- The schema's inputs together (perception summary, command history and telemetry) give the position if the error is at most 1 cell.
- No single source gives it if the error of each of the clean frames, the perception summary, the command history and the telemetry is at least 1.5 cells.
- F4 passes if both hold.

Only the telemetry block size and noise are varied if it fails, as the spec says, and every setting tried is recorded. The current measurement $x_t$ is reported for reference and is not part of the conditions: its local noise shows the focus, which the spec leaves to control C7.

In the spec, "frame history" is read here as the clean frames $I_t$. Read as $x_t$, the condition could not be met by any measurement in which attention lowers noise.

## Control F4: first run

`npm run target:f4 -- --episodes 2000`, seeds 6000000001 to 6000002000, probes trained on the first 1600 episodes and tested on the last 400.

| Source | Mean error, cells | Within 1 cell |
| :-- | --: | --: |
| Clean frames | 6.61 | 0.04 |
| Perception | 6.36 | 0.04 |
| Commands | 7.77 | 0.03 |
| Telemetry | 1.24 | 0.50 |
| All schema inputs | 1.19 | 0.55 |
| Time only | 8.50 | 0.02 |
| Current measurement (reference) | 7.81 | 0.01 |

**F4 fails on both conditions.** The schema's inputs together miss by 1.19 cells, more than 1, and telemetry alone already gives the position to 1.24 cells, less than 1.5. Commands and perception add almost nothing to telemetry.

**Perception does not see the target.** `target:check` with perception: under the ideal policy's attention, slot 0 is within 1 cell of the target on 6% of steps and answers correctly in 0.231 of episodes, at chance; without enhancement 0.239. Two causes:

- **The local noise estimate does not fit the focus.** The focus is about one cell with four weaker neighbours, smaller than the 3 × 3 neighbourhood, so the median sees mostly unattended noise. Under the ideal policy the target's signal-to-noise averages 1.85 without effort and 0.81 with effort, while the best background cell reaches 6.6, because a median of 8 values has heavy tails.
- **The world leaves the target nearly invisible to anyone who must find it.** Even with the true noise map, slot 0 stays on the target on 48% of steps and answers 0.29. At σ_η 2 a detection threshold of 4 needs nearly the full resource on the cell. A slot that loses the target drifts away on its velocity and never searches again.

The world was tuned on an observer that takes the target's position from the truth. C0 had to reach σ_η 2 because that observer averages the hue at a known cell. The agent sees the world only through perception, so for it the task is much harder. The perception and schema rows of this run say nothing yet. The telemetry row stands.

## Planned: second tuning, through perception

Registered before the sweep. The controls are now also run through perception: the attention commands and effort of the ideal policy, the paddle to slot 0's column, the answer from slot 0's hue on the last step. The readiness conditions, on the control seeds with 2000 episodes:

- F2 with perception answers correctly in at least 0.75 of episodes.
- C0 with perception answers correctly in at most 0.35 of episodes.
- F2 with the ideal observer answers correctly in at least 0.85 of episodes.
- The F6 conditions of the first tuning still hold.

C0 with the ideal observer is reported for reference and is no longer a condition: it takes the target's position from the truth, which the agent never has.

Varied: the measurement noise and enhancement, and the perception – its noise neighbourhood, detection threshold, a minimum brightness for detection, a new search by slot 0 after it loses its object, and its other parameters. The mechanism's own constants stay as they are unless no setting meets the conditions. Every setting tried is recorded. F4 is then run once at the chosen setting; if it fails, only the telemetry block size and noise are varied, as registered for F4.

**Amendment, before any run that changes the mechanism.** The sweep so far found settings that meet the conditions above only at low measurement noise, σ_η 0.2 to 0.3, with a high detection threshold. There the target is plainly visible without enhancement: an observer at a known cell reads its hue almost always. C0 then holds only because the threshold demands a quiet, attended neighbourhood, so the need for attention would come from the perception, not from the world. The condition of the first tuning is therefore restored: C0 with the ideal observer answers correctly in at most 0.35 of episodes. It holds only at σ_η 2 and above, where no perception setting tried reaches 0.75 with perception. The mechanism's constants may now vary: first the temperature τ0, because at 0.104 the focus spends 6.6 of the budget of 9 without effort and 3.5 with it, while the spec sets the budget to correspond to a 3 × 3 window, then the bump width σ_B. The other constants stay.

## Second tuning, through perception

The sweep ran on a scratch copy of the perception with the variations as options. At the repository's design it reproduces `target:check` exactly: at σ_η 0.25, 0.443 correct with 0.572 of steps tracked, and 0.246 without enhancement. Unless noted, each row is on the first episodes of the control seeds with the ideal policy's attention. "Correct" is F2 with perception, "C0" is C0 with perception, "tracked" is the share of steps with slot 0 within 1 cell of the target. Not admissible as perception, the true noise map was used as a diagnostic.

What was learned, in order:

1. **Slot 0 must search again.** With the true noise map, the repository's design answers 0.29 and tracks 48% of steps: a slot that loses the target drifts on its velocity. A new search of the whole field after 3 misses in a row raises this to 0.79 and 72%. A minimum brightness of 0.65 for detection, halfway between the occluder (0.3) and an object (1), keeps slots off the occluder.
2. **At σ_η 2 no estimate from the frame works.** Medians and lower quantiles over 4 or 8 neighbours, with one or both channels, reach at most 0.48, also with a temperature τ0 of 0.15 to 0.25, a narrower bump, or a budget of 16 or 25. The focus is about one cell with four weaker neighbours. A median over the neighbourhood sees unattended noise, a low quantile makes the focus's surroundings look quiet, and an estimate from 8 to 16 values has heavy tails. The diagnostic with the true noise map reaches 0.79 to 0.82 at σ_η 2, so the information is in $x_t$, but a local estimate cannot recover it.
3. **At low noise it works.** The lower quartile of the 16 absolute values of both channels in the 8 neighbours, with a high detection threshold, a velocity gain of 0.25 and a gate of 3, separates the two controls. At σ_η 0.25 and threshold 10: 0.790 correct, C0 0.263.

**Second amendment, after the sweep and before the confirming runs.** The ideal-observer C0 condition restored above can be met only at σ_η 2 and above, where no perception reads the frame. It is replaced by a condition on the perception itself: its detection threshold is set by the false-alarm rate on pure noise, not by C0. On frames of pure noise of any level, the frame's largest signal-to-noise exceeds the threshold in at most 5% of frames. For the lower-quartile estimate this is 9.49, so the threshold is 10. The sweep had already tried thresholds up to 14 before this rule was written; the rule picks 10 from noise statistics alone. C0 with the ideal observer is reported: at σ_η 0.25 it answers 0.844, because it is told where the target is. In this world, attention is what lets a perception that must search 256 cells find the target. An unattended object has a signal-to-noise of about 4, below what a search at a 5% false-alarm rate can accept.

**Chosen** (2000 episodes, control seeds): σ_η 0.25; perception with the lower-quartile estimate scaled by 1/0.3186, detection threshold 10, minimum brightness 0.65, a new search by slot 0 after 3 misses, velocity gain 0.25, gate 3, the other parameters unchanged. F2 with perception 0.790, C0 with perception 0.263, F2 with the ideal observer 0.973, C0 with the ideal observer 0.844 (reference). F6 does not depend on σ_η or the perception and is unchanged: 37% of flashes captured, 2.5 exits per episode, a gap of 0.56, the target in focus on 16% of steps under the neutral policy. α 0.95 and the flashes stay. The mechanism's constants did not change.

**Limitations.** The margins are 0.04 and 0.09. Attention here is needed to find the target, not to read it: told the position, an observer without enhancement reads the hue. A perception that integrates evidence over several frames before detecting could find an unattended target, and the bench does not use one.

<details>
<summary>Every setting tried</summary>

Notation: estimator over the neighbours – `med8` median of 8 brightness values (the repository's design), `orth` median of 4 brightness values, `orth2ch` median of both channels of 4, `med82ch` median of both channels of 8, `q` a lower quantile instead of the median, `oracle` the true noise map; `rs` new search by slot 0; `mb` minimum brightness 0.65; θ detection threshold; `vg` velocity gain; `g` gate radius; `hf` hue forgetting; `mm` misses before a new search; `hr` hue reset after a jump of more than so many σ̂. Values: correct with perception / C0 with perception (tracked). Unlisted values are the defaults.

At σ_η 2, without a new search:

| Setting | Episodes | Result |
| :-- | --: | :-- |
| med8 | 300 | 0.267 / 0.283 (0.068) |
| orth | 300 | 0.263 / 0.267 (0.119) |
| orth2ch | 300 | 0.250 / 0.313 (0.185) |
| med82ch | 300 | 0.243 / 0.283 (0.111) |
| orth2ch θ3 | 100 | 0.210 / 0.260 (0.142) |
| orth2ch θ6 | 100 | 0.240 / 0.230 (0.188) |
| orth | 100 | 0.240 / 0.280 (0.126) |
| oracle | 200 | 0.290 / 0.225 (0.484) |
| oracle θ2 | 200 | 0.260 / 0.240 (0.337) |
| oracle θ3 | 200 | 0.275 / 0.240 (0.485) |
| oracle σ_η 1 | 200 | 0.305 / 0.255 (0.522) |
| oracle σ_η 1 θ3 | 200 | 0.300 / 0.245 (0.510) |
| oracle σ_η 0.5 | 200 | 0.385 / 0.210 (0.576) |
| med8 σ_η 1 | 200 | 0.250 / 0.295 (0.131) |
| med8 σ_η 0.5 | 200 | 0.245 / 0.260 (0.292) |
| orth σ_η 0.5 | 200 | 0.270 / 0.265 (0.272) |

Five more runs of the first batch were lost to an argument-splitting error: they ran `orth` or `med8` under other labels, and their duplicates are not listed.

With a new search and minimum brightness, 200 episodes:

| Setting | Result |
| :-- | :-- |
| oracle rs | 0.785 / 0.230 (0.722) |
| oracle rs mb | 0.795 / 0.230 (0.716) |
| oracle rs mb σ_η 1 | 0.840 / 0.250 (0.732) |
| oracle rs mb σ_η 1 θ3 | 0.755 / 0.355 (0.693) |
| med8 rs mb | 0.330 / 0.305 (0.104) |
| orth2ch rs mb | 0.335 / 0.255 (0.250) |
| orth2ch rs mb σ_η 1 | 0.450 / 0.270 (0.378) |
| med82ch rs mb σ_η 1 | 0.395 / 0.310 (0.319) |
| orth2ch q 0.25, σ_η 1.5 θ4 / 1.5 θ3 / 1 θ3 / 1 θ4 | 0.280, 0.300, 0.370, 0.365 / 0.265, 0.235, 0.230, 0.280 |
| orth2ch q 0, σ_η 1.5 θ4 / 1 θ3 / 1.5 θ3 / 1 θ4 | 0.285, 0.280, 0.265, 0.260 / 0.310, 0.260, 0.260, 0.265 |
| med82ch q 0.25, σ_η 1 θ4 / 1.5 θ4 / 1 θ3 / 1.5 θ3 | 0.340, 0.305, 0.360, 0.280 / 0.265, 0.305, 0.310, 0.280 |
| med82ch q 0.1, σ_η 1 θ4 / 1.5 θ4 / 1 θ3 / 1.5 θ3 | 0.305, 0.320, 0.330, 0.310 / 0.245, 0.290, 0.240, 0.275 |
| orth q 0, σ_η 1 θ4 / 1.5 θ4 / 1 θ3 / 1.5 θ3 | 0.270, 0.305, 0.290, 0.320 / 0.270, 0.280, 0.270, 0.295 |
| med8, σ_η 0.5 θ4 / 0.75 θ3 / 0.75 θ4 / 0.5 θ3 | 0.480, 0.355, 0.445, 0.390 / 0.270, 0.270, 0.325, 0.280 |
| orth2ch, σ_η 0.5 θ3 / 0.75 θ4 / 0.5 θ4 / 0.75 θ3 | 0.550, 0.475, 0.590, 0.400 / 0.285, 0.265, 0.300, 0.255 |
| med82ch q 0.25, σ_η 0.5 θ3 / 0.75 θ3 / 0.5 θ4 / 0.75 θ4 | 0.590, 0.520, 0.640, 0.460 / 0.250, 0.295, 0.255, 0.260 |
| med8, σ_η 0.35 θ6 / 0.25 θ6 / 0.25 θ4 / 0.35 θ4 | 0.620, 0.750, 0.755, 0.605 / 0.300, 0.375, 0.425, 0.325 |
| med82ch, σ_η 0.35 θ4 / 0.35 θ6 / 0.25 θ6 / 0.25 θ4 | 0.710, 0.730, 0.775, 0.785 / 0.430, 0.305, 0.410, 0.395 |
| med82ch q 0.25, σ_η 0.35 θ6 / 0.25 θ4 / 0.25 θ6 / 0.35 θ4 | 0.745, 0.775, 0.805, 0.735 / 0.305, 0.385, 0.415, 0.320 |

Slot parameters, med82ch q 0.25 rs mb θ6, 300 episodes:

| Change | σ_η 0.3 | σ_η 0.35 |
| :-- | :-- | :-- |
| hf 0.75 | 0.733 / 0.347 | 0.687 / 0.327 |
| hf 0.25 | 0.760 / 0.333 | 0.710 / 0.307 |
| g 3 | 0.800 / 0.330 | 0.727 / 0.320 |
| g 1 | 0.773 / 0.350 | 0.733 / 0.317 |
| mm 2 | 0.787 / 0.373 | 0.720 / 0.310 |
| mm 1 | 0.740 / 0.343 | 0.697 / 0.287 |
| vg 0.25 | 0.807 / 0.333 | 0.747 / 0.320 |

med82ch q 0.25 rs mb, 1000 episodes:

| Setting | Result |
| :-- | :-- |
| σ_η 0.25 θ8 vg 0.25 g3 | 0.795 / 0.349 |
| σ_η 0.25 θ7 vg 0.25 g3 | 0.798 / 0.347 |
| σ_η 0.3 θ6 g3 | 0.745 / 0.317 |
| σ_η 0.3 θ6 vg 0.25 g3 | 0.746 / 0.315 |
| σ_η 0.3 θ7 vg 0.25 g3 | 0.760 / 0.325 |
| σ_η 0.3 θ5 vg 0.25 g3 | 0.739 / 0.319 |
| σ_η 0.3 θ6 vg 0.25 | 0.757 / 0.325 |
| σ_η 0.35 θ6 vg 0.25 g3 | 0.684 / 0.302 |
| σ_η 0.3 θ7 vg 0.25 g3 hr3 | 0.727 / 0.326 |
| σ_η 0.3 θ7 vg 0.25 g3 hr2 | 0.713 / 0.326 |
| σ_η 0.3 θ7 vg 0.25 g3 hr3 hf 0.75 | 0.732 / 0.329 |
| σ_η 0.35 θ6 vg 0.25 g3 hr3 | 0.693 / 0.295 |
| σ_η 0.35 θ7 vg 0.25 g3 hr3 | 0.703 / 0.290 |
| σ_η 0.4 θ6 vg 0.25 g3 hr3 | 0.642 / 0.289 |
| σ_η 0.4 θ5 vg 0.25 g3 hr3 | 0.635 / 0.286 |
| σ_η 0.5 θ5 vg 0.25 g3 hr3 | 0.609 / 0.274 |

med82ch q 0.25 rs mb vg 0.25, 2000 episodes:

| Setting | Result |
| :-- | :-- |
| σ_η 0.3 θ6 | 0.752 / 0.302 |
| σ_η 0.3 θ7 | 0.759 / 0.307 |
| σ_η 0.3 θ7 g3 | 0.758 / 0.301 |
| σ_η 0.3 θ10 g3 | 0.754 / 0.250 |
| σ_η 0.25 θ7 g3 | 0.798 / 0.343 |
| σ_η 0.25 θ8 g3 | 0.794 / 0.328 |
| σ_η 0.25 θ9 g3 | 0.792 / 0.308 |
| σ_η 0.25 θ10 g3 | 0.793 / 0.272 |
| σ_η 0.25 θ12 g3 | 0.789 / 0.245 |
| σ_η 0.2 θ10 g3 | 0.827 / 0.350 |
| σ_η 0.2 θ12 g3 | 0.823 / 0.291 |
| σ_η 0.2 θ14 g3 | 0.825 / 0.256 |

After the first amendment, σ_η 2, rs mb, 300 episodes:

| Setting | med8 | med82ch | med82ch q 0.25 | orth | orth2ch | oracle |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| τ0 0.15 | 0.327 | 0.343 | 0.307 | | | |
| τ0 0.2 | 0.340 | 0.327 | 0.330 | 0.327 | 0.347 | |
| τ0 0.25 | 0.363 | 0.307 | 0.300 | | | |
| τ0 0.2, σ_B 1.5 | 0.323 | | | | | 0.787 |
| R 9 | | | | 0.267 | 0.347 | |
| R 16 | 0.360 | 0.380 | | 0.347 | 0.410 | 0.800 |
| R 25 | 0.417 | 0.483 | | 0.340 | 0.457 | 0.823 |

C0 with perception stayed between 0.25 and 0.30 in all these rows. The budget R was outside the amendment's list and was run only as a diagnostic.

Confirming runs with the quartile scaled by 1/0.3186, rs mb vg 0.25 g3 θ10, 2000 episodes:

| σ_η | Result |
| :-- | :-- |
| 0.2 | 0.827 / 0.347 (0.849) |
| 0.25 | 0.790 / 0.263 (0.824) |
| 0.3 | 0.750 / 0.254 (0.791) |

</details>

## F4: telemetry grid with the first perception

Run ahead of the registered order, on a copy of the code with the first perception and the first world, to see what the telemetry alone can do. The telemetry depends only on the resource map and its own noise, not on the measurement or the perception, so its rows stand. The schema rows include the first perception, which sees almost nothing, and are a preview. The frames, perception and commands rows do not depend on the telemetry and are as in the first run.

| Block k | Noise σ_ν | Telemetry, cells | All schema inputs, cells |
| --: | --: | --: | --: |
| 4 | 0.5 | 1.24 | 1.19 |
| 4 | 1 | 1.61 | 1.55 |
| 4 | 2 | 2.40 | 2.31 |
| 4 | 4 | 4.35 | 3.99 |
| 2 | 0.5 | 1.27 | 1.32 |
| 8 | 0.5 | 2.96 | 2.46 |
| 16 | 0.5 | 7.87 | 6.25 |

The first row is the first F4 run. No setting has the telemetry at 1.5 or more and the schema's inputs at 1 or less. The probe of all schema inputs barely improves on the telemetry: by 0.05 to 0.09 cells while the telemetry is informative, and by 0.5 to 1.6 cells only when the telemetry is weak, which is still far from 1. A probe that sees the command history does not integrate it into a position. With the first perception there is nothing to anchor the integration to; the second perception, which finds the target at the start, may change this.

## Second tuning: confirming runs

`npm run target:check -- --episodes 2000` in the repository at the chosen setting, which is now the default.

| Condition | Bound | Result |
| :-- | --: | --: |
| F2 with perception, correct | at least 0.75 | 0.7905 |
| C0 with perception, correct | at most 0.35 | 0.2630 |
| F2 with the ideal observer, correct | at least 0.85 | 0.9730 |
| Pure-noise frames with the largest signal-to-noise at the threshold or above | at most 0.05 | 0.0445 |

All hold, and the F6 table is unchanged byte for byte. C0 with the ideal observer is 0.8435, for reference. Slot 0 is within 1 cell of the target on 0.824 of steps under F2 and 0.180 under C0. The false-alarm check uses 2000 frames of unit noise from its own generator stream; the 95th percentile of the frame maximum is 9.74, against 9.49 on the scratch copy.

One difference from the scratch copy: when no cell reaches the minimum brightness on the first step, the scratch copy put slot 0 at cell (0, 0), and the repository leaves it inactive until a candidate appears. Three C0 episodes differ, by 17 tracked steps of 60000. Every other number matches to six decimals.

## F4 with the second perception

`npm run target:f4 -- --episodes 2000` at the new defaults.

| Source | Mean error, cells | Within 1 cell |
| :-- | --: | --: |
| Clean frames | 6.61 | 0.04 |
| Perception | 3.28 | 0.13 |
| Commands | 7.77 | 0.03 |
| Telemetry | 1.24 | 0.50 |
| All schema inputs | 1.14 | 0.59 |
| Time only | 8.50 | 0.02 |
| Current measurement (reference) | 7.64 | 0.01 |

**F4 fails on the same two conditions.** Telemetry alone gives the position to 1.24 cells, less than 1.5, and the schema's inputs together miss by 1.14 cells, more than 1. The perception now carries something, 3.28 against 6.36, and the clean frames, the perception and the commands meet their condition.

The telemetry block size and noise were then varied, as registered. Only the telemetry and schema rows depend on them. Each row is 2000 episodes with the same probes.

| Block k | Noise σ_ν | Telemetry, cells | All schema inputs, cells |
| --: | --: | --: | --: |
| 4 | 0.5 | 1.24 | 1.14 |
| 4 | 0.75 | 1.42 | 1.31 |
| 4 | 1 | 1.61 | 1.48 |
| 4 | 2 | 2.40 | 2.09 |
| 4 | 4 | 4.35 | 3.02 |
| 2 | 0.5 | 1.27 | 1.26 |
| 8 | 0.1 | 2.58 | 1.73 |
| 8 | 0.5 | 2.96 | 1.92 |
| 16 | 0.5 | 7.87 | 2.77 |

**No setting passes.** Wherever the telemetry alone is at 1.5 or more, the schema's inputs together are at 1.48 or more. The telemetry column repeats the grid run earlier with the first perception, as it should.

## F4: what the probe misses

Diagnostics, not verdicts. They ask whether the failure lies in the world or in the probe.

**More training does not help the probe, more data helps a little.** The same probe with ten times the updates or four times the episodes, k 4:

| Episodes | Updates | σ_ν | Telemetry, cells | All schema inputs, cells |
| --: | --: | --: | --: | --: |
| 2000 | 2000 | 0.5 | 1.24 | 1.14 |
| 2000 | 20000 | 0.5 | 1.49 | 1.30 |
| 8000 | 2000 | 0.5 | 1.16 | 1.01 |
| 8000 | 20000 | 0.5 | 1.21 | 1.02 |
| 8000 | 20000 | 1 | 1.56 | 1.32 |

With 2000 episodes the longer training overfits. With 8000 episodes both rows improve together and the gap between them stays near 0.15 cells.

**A filter over the same inputs does much better.** A scratch filter, per axis, over the 16 cells: the likelihood of the current telemetry from a probe that sees one step of telemetry (16–200–16, 5000 updates), and transition tables counted on the training episodes. It runs in four forms: without commands, with the previous command, with the command and a start taken from slot 0 on the first step, and with the start and commands but no telemetry. 2000 episodes, the same split.

| Block k | Noise σ_ν | One step of telemetry | Telemetry, filtered | With commands | With commands and start |
| --: | --: | --: | --: | --: | --: |
| 4 | 0.5 | 1.49 | 1.09 | 0.88 | 0.83 |
| 4 | 0.75 | 2.02 | 1.26 | 1.00 | 0.92 |
| 4 | 1 | 2.61 | 1.44 | 1.13 | 1.02 |
| 4 | 1.5 | 3.84 | 1.80 | 1.42 | 1.22 |
| 4 | 2 | 4.87 | 2.27 | 1.83 | 1.43 |
| 2 | 0.5 | 1.69 | 0.82 | 0.74 | 0.73 |
| 2 | 1 | 4.48 | 1.58 | 1.34 | 1.15 |
| 8 | 0.1 | 2.95 | 2.39 | 2.06 | 1.52 |
| 8 | 0.25 | 3.12 | 2.53 | 2.18 | 1.59 |
| 8 | 0.5 | 3.33 | 2.70 | 2.30 | 1.65 |

The start and the commands without telemetry give 1.97 cells: 0.42 on the first step and 2.93 on the last, as captures and the mechanism's noise pull the focus off the commanded path. A perfect estimate rounded to a cell would score 0.38.

What this shows:

- **The information is in the schema's inputs.** At the defaults a crude filter reaches 0.83 cells from 1600 training episodes, where the probe reaches 1.14, and 1.01 from 6400. The probe sees the whole history at once and does not learn to carry a position forward by the commands.
- **The sources do complement each other.** Commands take a quarter off the filtered telemetry's error and the start a little more. Neither the telemetry alone nor the start with commands alone reaches what they give together.
- **A filter also reads the telemetry alone better than the probe does**, 1.09 against 1.24. With a probe of this kind, the telemetry noise would have to rise to about 1.1 before telemetry alone misses by 1.5 cells.
- **The two bounds are tight for this world.** The ratio of the error from all inputs to the error from telemetry alone is 0.76 at σ_ν 0.5, 0.71 at 1 and 0.63 at 2, and the bounds ask for 0.67 at an absolute level of 1 cell. The filter misses narrowly at σ_ν 1: 1.44 and 1.02. It does not use the second slot or the effort, and it works in whole cells, so a learned recurrent probe may do better on all inputs; it may also do better on telemetry alone.

## Planned: F4 with a recurrent probe

Registered before any run of the new probe. The conditions of F4 do not change: the schema's inputs together give the position to at most 1 cell, and each of the clean frames, the perception summary, the command history and the telemetry misses by at least 1.5 cells. The measure, the seeds and the neutral policy stay. Only the probe changes.

**Why.** The first probe took the whole history as one flat input. The diagnostics above show that it does not carry a position forward by the commands, while a filter over the same inputs does, so its failure on the schema's inputs says little about the inputs. The schema itself is recurrent, and the question F4 asks is what a module of that kind can recover.

**The probe.** A GRU, the kind of module the spec gives the schema, fed one step of its source per step, with 64 hidden units and a linear readout of the row and the column at every step. It is trained by cross-entropy on both, averaged over steps, with Adam at 0.001 and batches of 32 episodes for 6000 updates. The run has 8000 episodes, the first 6400 for training. Every source gets the same probe. The reference row becomes the history of $x_t$.

**Reading.** The held-out error is taken every 500 updates. For a single source the lowest of these counts, so that an overfitted probe cannot pass for an uninformative source. For the schema's inputs the last one counts.

**The probe's own settings.** If the held-out error of the telemetry or schema probe is still falling at the end, or has turned up, the number of updates, hidden units or episodes is changed until it is flat. This is judged on the error curves at the default telemetry, not on whether F4 passes, and every setting is recorded.

**If it fails.** Only the telemetry block size and noise are varied, as before. If no setting passes, that is the result. The bounds of 1 and 1.5 cells are not moved without a separate decision registered before further runs.

## Planned: longer probes where the error still falls

Registered after the first runs of the recurrent probe, at 6000 updates, and before any longer run. The conditions of F4 and the bounds of 1 and 1.5 cells do not change.

**What was seen.** At the default telemetry the telemetry and schema probes are flat by update 6000, and F4 fails there on the telemetry condition: 1.04 cells from the telemetry alone, 0.83 from the schema's inputs. Two probes are not flat. At k 8 the schema probe is still falling at the end: 1.30, 1.28 and 1.25 cells at the last three checkpoints with σ_ν 0.1, and 1.45, 1.43 and 1.38 with σ_ν 0.5, while the telemetry alone stays above 2.2. At the defaults the perception probe is still falling too, from 3.54 to 2.76 cells. The full tables follow with the longer runs.

**Why this needs an amendment.** The registration let the probe's settings change only by the curves at the default telemetry. A probe that is still learning does not measure what its inputs give, at any setting. For the schema's inputs it overstates the error. For a single source it can pass an informative source for an uninformative one.

**The rule.** At any telemetry setting and for any source, a probe is flat when its held-out error at the last checkpoint is at most 0.01 cells below its error 2000 updates earlier. A probe that is not flat is run with twice the updates: 12000, then 24000, then 48000. The generator stream is the same, so a shorter run is the beginning of a longer one and its checkpoints are read from it. Hidden units, batch, episodes and the split stay. The reading stays: for a single source the lowest checkpoint of everything run, for the schema's inputs the last checkpoint at the first of 6000, 12000, 24000 or 48000 updates where the probe is flat.

**The runs.** Block k 8 with σ_ν 0.02, 0.1, 0.25 and 0.5, the telemetry and the schema's inputs, 24000 updates. The perception probe at the defaults, 24000 updates. The settings with k 4 are flat by this rule. Those with k 2 have turned up, and their schema probe misses by 0.3 cells where the telemetry alone reaches 1.5, which overfitting of 0.01 cells does not explain. Neither is rerun. The threshold of 0.01 was set knowing that the default curves fall by 0.009 over the last 2000 updates; no reading at k 4 depends on it, since the schema's inputs there miss by 0.16 cells where the telemetry alone reaches 1.5.

**A limit outside F4.** The spec asks that the telemetry resolve the probe offset of E1. A block of 8 leaves four values. If k 8 passes F4, that requirement is checked before the setup is frozen.
