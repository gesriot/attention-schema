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
