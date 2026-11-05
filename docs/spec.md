# Specification: a minimal bench for the functional prerequisites of consciousness under Graziano's attention schema theory

*Checked against primary sources on 4 November 2026. The bench is implemented first in TypeScript, then in Rust with byte-for-byte comparison. Caveats about incomplete checks are listed under Sources.*

## Task

We are building a minimal agent with real attention, a simplified model of that attention, and a causally effective link from that model to control and report. This is not a specification for a program with qualia. Michael Graziano's Attention Schema Theory (AST) does not promise that the right architecture will produce experience. Graziano writes that AST "does not give a prescription for how to make a magical essence of experience emerge". Under AST, asking whether a machine has "ethereal" internal feelings is a nonsensical question: it does not have them, and neither do we. The meaningful question is different: does the system have deep, automatic models of its own processes, and does it assert on their basis that it has experience?[^3] This is the theory's position, not an established fact. Current approaches to assessing machine consciousness offer indicators, not a test that certifies experience.[^13]

So we are building **not a chatbot with self-reports but a small laboratory of attention control**. The agent tracks an object through noise and distraction, real attention changes what it can perceive, and a learned model of attention estimates and predicts that state. The main object of the test is a causal chain: an error in the model changes control and report while attention and perception stay the same.

**The main caveat comes from Graziano himself.** In his 2026 article he writes that artificial agents with attention schemas show only "modest benefits", use very small networks and "have no clear relationship to human consciousness". His current guess is that all these initial approaches "have gone in the wrong direction", and that "something much simpler and much more fundamental is needed".[^4] This bench belongs to the same class. It tests the control function of the schema, not AST as a theory of consciousness. The section "What exactly we test" explains what this means for the architecture.

## Attention schema theory: what exactly it claims

This summary draws on Graziano's reviews from 2017 to 2026[^1][^3][^5][^6], on Webb & Graziano 2015[^2] and on computational work from 2021 to 2026[^7][^8][^9][^10][^11].

| Component | What the source says | Consequence for the bench |
| :-- | :-- | :-- |
| Principle 1 | "Information that comes out of a brain must have been in that brain." Everything a person thinks and claims about their consciousness derives from information in the brain. Having consciousness and believing you have it are different things[^1] | The agent's report about its own attention derives causally from information available to the agent, not from diagnostic code |
| Principle 2 | "The brain's models are never accurate." They are not empty illusions but caricatures: they reflect a real process while simplifying and distorting it[^1] | The schema is not a copy of the attention state. Its errors are part of what is studied |
| Process A | Attention is a real process: signals compete for limited processing resources (biased competition)[^2]. Some representations are enhanced and others suppressed, bottom-up or top-down, and the enhanced ones have greater impact on downstream systems[^4]. The model is built for this process[^1] | The real attention mechanism and its model are physically separated. The mechanism is competition for a limited resource |
| What counts as attention | In 2026 Graziano sees the main obstacle in choosing what exactly to model. He proposes attention in James's sense, unanchored to a specific mechanism: representations temporarily rise in activity and can influence the output[^4] | The attention state is described as the share of resource on a representation, not as the position of a window |
| Schema content | A description of the state of attention and predictions: how it will change and how it will affect decisions, memory and behavior[^1]. The schema is more about predictive relationships between attention, other processes and the environment than about attention itself. It contains no mechanistic details and does not represent the attended object[^4] | The core of the schema is predicting consequences, not the coordinates of the focus |
| Automaticity | The schema is built automatically and continuously. It cannot be turned on, turned off or modified by choice. It works even when there is no access to it for report[^4] | Updating the schema is not one of the controller's choices. Control reads it directly, report reads it through the relation $r_t$ |
| Discrepancy | Self-models are predictive in the first place. A discrepancy with reality triggers a reaction: adjust the object or update the model[^4] | E2B tests both reactions |
| Control | A good regulator needs a model of the thing it controls[^7][^19]. Without a schema, endogenous control of attention should suffer[^1] | The controller uses the schema. The theory speaks of a model, not of a separate module: an implicit model counts too |
| Control-theory predictions | Without an internal model, attention is less stably maintained in the desired state, less able to leave an undesired state, and more easily perturbed from outside[^14]. Overall strength of attention need not fall: what changes is its time course[^14][^15] | Direct measures for ablation. A drop in the mean result alone is not enough |
| The S + A + V triad | To conclude that it is aware of an apple, the brain needs a model of the object (V), a model of the self (S), and a model of the relation between them, which is attention (A)[^2]. In 2026 this place is described as a workspace where sensory models and self-models are interrelated and can affect action. It does not itself generate consciousness[^4] | A relational representation is needed: "this agent, its attention, this object" |
| Three functions | A model of one's own attention serves endogenous control, a model of others' attention serves theory of mind, and the contents of these models lead to claims of consciousness[^15][^1] | The MVP covers control and nonverbal report. The social function is an extension |
| Link between functions | One of the most specific predictions of AST: if the schema fails to model a state of attention, a person can neither control that focus nor report an experience of it. The formula is not "consciousness is attention" but "consciousness is necessary for the control of attention"[^1] | One and the same schema error should change control and report together |
| Dissociation | Without awareness of a stimulus, attention to it is possible, but it is not stabilized, not adjusted by relevance, not shifted in anticipation[^3] and not suppressed on a distractor[^1][^17]. The exogenous effect remains while the endogenous one almost disappears[^15]. Graziano himself notes that these data come from brief visual stimuli and lack generality[^3] | Expected ablation profile: capture preserved, control lost |
| Subtle illusionism | The model distorts rather than invents: a real process exists but is depicted schematically[^3]. In 2026 Graziano avoids or heavily qualifies the word "illusion": the models are useful and depict real things[^4]. He distinguishes i-consciousness (information) from m-consciousness (mysterious)[^6] | Attention in the bench is a real process. The bench studies the model and does not imitate an "illusion" |
| Conditions for a machine | Selective attention, an attention schema, the same simplifications and distortions as in humans, and access of higher cognition to the schema. The agent would then conclude and assert that it is conscious[^3]. In 2026 the condition is widened to a similar suite of self-schemas used for similar functions[^4] | The MVP implements the first two conditions and part of the fourth. It has neither human distortions nor assertions. **The bench does not implement consciousness in the AST sense even if it succeeds completely** |
| Emergence | The schema need not be built in: given a free controllable resource it emerges, and it is more useful the harder attention is to infer from the stimulus[^8] | Version B of the bench |
| Society of schemas | Generalization to "a more general schema theory": attention, decision, emotion and body schemas and personas[^4] | Extension beyond the MVP |
| Indicator | AST-1: "A predictive model representing and enabling control over the current state of attention"[^13] | Statement of the bench's goal |

Graziano himself cautioned that AST "is not yet specific enough to hand a blueprint to an engineer".[^5] Everything called architecture below is therefore an engineering interpretation, not a published algorithm.

## What exactly we test

Three levels must be kept apart:

| Level | Meaning |
| :-- | :-- |
| **Claims of AST** | Theoretical motivation: the schema as a simplified model of attention, its role in control, its link to report.[^1][^3] |
| **Proposed architecture** | A specific engineering interpretation, not an algorithm published by Graziano |
| **Testable result** | A demonstration of functions and their causal role, not a proof of phenomenal experience |

Within AST the bench tests several hypotheses that must not be mixed:

1. **Regulator.** Endogenous control of attention needs information about the state of attention, in any form. This follows directly from the theory.
2. **Explicit schema.** A learned schema gives an advantage over a control with the same resources. This is an engineering hypothesis. Its failure does not refute item 1: an implicit model inside the control is still a model.
3. **Prediction.** A predictive schema is stronger than a purely descriptive one. Prediction is what separates AST-1 from the 2021 system.[^13]
4. **Link between functions.** A schema error, with attention and perception unchanged, changes control and report together. This is the most specific prediction of the theory.[^1]
5. **Emergence.** A schema forms where attention matters and is hard to track. This is a result of Piefke et al., not a claim of AST.[^8]

A bench where the schema is given ready-made tests only the usefulness of a description. A bench without report does not test hypothesis 4.

**What follows from Graziano's 2026 caveat.** His criticism concerns which attention should be modeled and what a machine that "thinks" it is conscious would need. The alternative he proposes relies on large language models.[^4] Three consequences follow:

- **Limit of inference.** The bench tests the control function. A positive result does not confirm AST as a theory of consciousness, and a negative one does not refute it: it applies to small systems of this class.
- **What goes into the architecture.** Attention is competition among representations for a resource, not a window on an image. The schema is above all a prediction of consequences. The schema updates automatically.
- **Expected effect size.** The sharp gap from 2021 is not the norm: the author himself now speaks of modest benefits.[^4] That gap came from a ready-made error-free schema and from a task where a small network uses other sources of information about attention poorly.[^7]

Differences from the [`coherence-lab`](https://github.com/gesriot/coherence-lab) bench, built on Joscha Bach's model:

| Question | Bach's model | AST |
| :-- | :-- | :-- |
| Objective function | Coherence, actually minimized by the process | None: an external task and prediction of one's own attention |
| Second order | Perception of one's own perception in the same cycle, distinct from higher-order thought | A separate simplified model; AST can be seen as a higher-order theory[^13] |
| What is controlled | The position of a sensor window | The distribution of a limited processing resource with a fixed sensor |
| Strong form | Genesis: consciousness is a learning algorithm | Emergence of the schema is a separate hypothesis that AST does not require |
| What the test shows | Whether the loop speeds up the formation of world models | Whether control and report depend on a model of attention |

**The word "required" below means required for the proposed experiment, not an established necessity for any possible consciousness.**

## Where the "perceiver" is

The short answer: **nowhere as a separate component**. A separate observer must not be built: it would give a regress instead of an explanation.

| Question | Answer in the implementation | What is not enough |
| :-- | :-- | :-- |
| Where is "red" represented? | In the perceptual model $z_t$ | A feature vector says nothing about the agent's access to it |
| Where is the object actually selected? | In the attention mechanism $a_t$ | Attention weights alone are filtering, not a schema |
| Where is "my attention is on this object" represented? | In the schema $s_t$ and the relational state $r_t$ | A log that does not affect control |
| Who reads these representations? | The attention controller and the choice of response | There is no additional viewer above them |
| Why is attention described as direct possession? | The schema contains no mechanism: it describes the result, not indices and weights | Feeding full uncompressed telemetry gives a memory dump, not a model |

The schema is not a receptacle that makes information conscious when it enters. It does not contain information about the object. It adds to that information the fact that attention is on the object.[^15]

The proposed loop:

```text
World → sensor → attention mechanism → perceptual content z → action
                     ↑      ↓               ↓
        attention command   telemetry p     │
                     ↑      ↓               ↓
                controller ← attention schema s ← self-model e
                     ↑               ↓
                     └── relation r: "agent, attention, object"
```

**Only the working loop as a whole can be called the "perceiver" here.** Under AST the point is not that it "sees" the content. The system builds and uses information about its own selective relation to the content. If that information underlies control and report, this is all the theory proposes to explain.[^1][^3]

A fundamental question stays open: is this explanation sufficient? Graziano says yes and considers the question of additional experience ill-posed.[^3] Indicator approaches do not accept this answer as settled.[^13] The bench does not resolve this dispute, so the code must not contain a `Consciousness` class.

## Minimal experimental environment

The environment has two modes. Level 0 reproduces the published setup to anchor the bench to known results. The target mode removes its simplifications.

### Level 0: the published setup

| Parameter | Value[^7][^20] |
| :-- | :-- |
| Visual field | 10 × 10, black or gray pixels. The bottom row has no noise and holds a 3-pixel paddle |
| Noise | Exactly 50 random cells out of 100 are black on each frame. The paper describes this as a 50% probability per pixel |
| Attention window | 3 × 3, noise fully removed inside. At the start of an episode the window is automatically on the ball |
| Schema | A second 10 × 10 array with a black 3 × 3 square at the window position |
| Input | 20 × 10 per frame, a buffer of 10 frames, 200 × 10. A new frame pushes out the oldest |
| Ball | Start and finish are random and independent. It moves diagonally to the finish column, then vertically, one row per step |
| Actions | 8 window shifts by one cell, 2 paddle shifts. There is no "stay" action |
| Reward | ±0.5 for the ball in the window on frames 2 to 9, ±2 for the catch on frame 10, no reward on the first frame. The maximum is 6: 4 for attention and 2 for the catch. The paper states 5 and 7, but in the code the 10 frames give only 8 attention rewards |
| Network | DQN, 16 outputs, three hidden layers of 200 |
| Training | Adam 0.001, ε = 0.2 during collection, greedy policy during tests. Buffer of 10,000 transitions |
| Iteration | 1000 collected transitions (about 111 games, not 100), 100 updates on batches of 100 pairs, then tests of 50 games per condition. 1500 iterations in total |
| Discount | 0.4 on nonterminal transitions, set in the environment. The agent's γ is not set |
| Library defaults | Target copied on every update, Huber loss, no gradient clipping, ReLU, n-step = 1. Hidden weights: VarianceScaling (2, fan_in, truncated normal). Output layer: uniform ±0.03, bias −0.2 |
| Subjects | 5 independent initializations |

Reference values are means over 5 subjects for the last 100 iterations, taken from the saved outputs of the published notebooks:[^20]

| Training | Test | Tracking out of 4 | Catch out of 2 |
| :-- | :-- | --: | --: |
| With schema | Normal | 3.38 | 0.83 |
| With schema | Window off | 0.15 | −0.42 |
| With schema | Schema off | −0.46 | −0.22 |
| Without schema | Normal | 0.34 | −0.26 |
| Without schema | Window off | −0.49 | −0.43 |

The paper gives rounded values: about 3.5, 0.8 and 0.4.[^7]

Reproduction is qualitative. Parameters come from the authors' published code.[^20] The TF-Agents version in it is unknown, TensorFlow is 2.1.0, and the defaults were checked against the TF-Agents 0.3.0 source.

The bench deliberately differs from this code in one place. On episode reset the code does not recompute the window cells, so the first frame shows the window and the schema where the previous episode ended, and the starting column is not clipped at the edge. The paper describes the window on the ball from the first frame, and Piefke et al. do the same. The bench follows the paper: the first frame earns no reward, and episodes stay independent. This difference noticeably speeds up early learning with the schema; the numbers are in `docs/results.md`. The reference values are therefore a qualitative guide, not a target for exact agreement. The implementation was checked against the authors' curves separately, with the behavior of their code reproduced.

This setup has two known limitations:

- The authors themselves note that "complete information about the location of the attention window is present, though implicit, in the visual array".[^7] Any conclusion about the benefit of a schema must rule out that it merely duplicates the frame.
- Butlin et al. consider that such a system has only part of indicator AST-1. It is not predictive, and the spotlight is "a simple controllable feature of the input rather than an internal process with multiple degrees of freedom governing information flow".[^13]

The Piefke et al. variant: PPO, an actor of 4 layers and a critic of 3 layers with 1000 units each, no memory buffer, 3 downward actions each for the window and the resource, and a separate array with a square that the agent moves itself.[^8] The reward layout is the same, so the maximum is also 6. Numbers for comparison: at p = 0.5, tracking 3.74 out of 4 and catch 1.73 out of 2; with a built-in schema 3.73 and 1.77; with random resource actions 0.93 and −0.03; the window position is read from the resource with 61% accuracy against 1.6% chance.[^8] Another baseline matters: with random resource actions the accuracy is still 40%, so the comparison must be with that, not with chance.[^8] The authors trained one agent per noise level.

At level 0, $x_t = I_t \lor (n_t \land \lnot W(a_t))$, where $n_t$ is a random mask of exactly $100p$ cells of the whole field and $W(a_t)$ is the window mask.

### Target mode: what is in the world

**Design proposal.** Starting values are refined only on feasibility controls:

- A 16 × 16 field, brightness and hue channels, 30-step episodes, a fixed sensor covering the whole field.
- A target object that changes direction unpredictably. The target is defined by attention being automatically on it at the start of the episode.
- One distractor object similar to the target.
- A partial occluder.
- Flashes at random places and times, with random strength.
- The target's hue jumps once at a random moment in the second half of the episode. The task: report the final hue at the end and catch the target with the paddle.
- A separate episode type: cue and probe stimulus (E1).

The hue change makes attention necessary at specific moments, not only at the start.

Required properties of the world:

1. **Attention is necessary.** The task cannot be solved without it (control C0).
2. **The position of attention cannot be inferred from a single source.** Not from the frame, the perceptual summary, the command history or the telemetry alone, but it can be inferred from the schema's inputs together (control F4). This is the central requirement of the setup. Piefke et al. showed that the usefulness of a schema depends on it.[^8]
3. **The command does not determine the outcome.** Delay, inertia and capture separate the command from the actual focus.
4. **Paths and events are random.** Otherwise the agent learns a habit instead of a model.

Attention and gaze are not mixed. The MVP has no gaze. If it appears later, `gaze_state` and `attention_state` are stored and logged separately.

### The real attention mechanism

In the target mode attention is not a window on an image but competition for a limited processing resource. This takes into account the remark by Butlin et al.[^13] and the description of attention in the 2026 article.[^4] The sensor stays the same, while the measurement available to processing depends on how the resource is distributed.

**Why a map rather than a focus center.** A state of a few numbers has nothing to simplify, so the schema would not be a caricature. A shared resource also makes capture a leak of a share and not only a shift, which is how AST reads the result of Tsushima et al.[^15]

**Why commands are relative.** An absolute command makes knowledge of $a_t$ unnecessary: the agent could simply command directly at its estimate of the target.

The hidden state of the mechanism is a priority map $h_t$ over the whole field. The actual state of attention is the resource map:

$$
a_t(r) = \min\Bigl(1,\; R\,\mathrm{softmax}_r\!\bigl(h_t / \tau_t\bigr)\Bigr),
\qquad
\tau_t = \frac{\tau_0}{1 + f_t}.
$$

Measurement:

$$
x_t = I_t + \sigma_\eta \bigl(1 - \alpha\, a_t\bigr) \odot \eta_t,
$$

where $\eta_t \sim \mathcal N(0, I)$ and $0 < \alpha \le 1$. Outside the focus the signal remains but is masked by noise.

Dynamics:

$$
h_{t+1} = \rho\, h_t + \beta_{\mathrm{endo}}\, B(\bar c_t + u_t) + \beta_{\mathrm{exo}}\, S_{t+1} + \epsilon_t.
$$

Here:

- $R$ is the total resource; its starting value corresponds to a 3 × 3 window.
- $\bar c_t$ is the center of mass of the map $a_t$. It is not available to the agent.
- $u_t$ is a relative shift command, and $B$ is a Gaussian bump around the given point.
- $f_t \ge 0$ is effort: it sharpens competition and suppresses weak competitors. **Effort has a cost.** Otherwise the controller would keep it at maximum, and there would be no reason to know about distractors.
- $S_{t+1}$ is the salience map: sudden changes in the world in proportion to their strength. The attention mechanism computes it; the controller and the schema do not receive it.
- $\rho < 1$ is decay: without support, priority dissolves, so holding attention requires control.
- $\epsilon_t$ is the noise of the mechanism.

At the start of an episode priority is concentrated on the target. The dynamics are set by the developer and do not change during an experiment.

What this formulation gives without separate rules:

- **Inertia and delay.** A new bump has to outgrow the old one, so a command does not take effect at once.
- **Switching as a state.** During a shift the resource is split between two places.
- **Capture and leak.** A salient event takes a share of the common resource even if the center barely moves.

The evaluator works with summaries of the map: center, share of resource on an object, entropy. The share of resource on an object approximates "taking possession" of the object in James's sense.[^4]

Forbidden simplifications: a frame on the screen that does not affect processing; a focus that always equals the command; absolute commands.

### What the agent receives

- The measurement $x_t$ (perceptual module only; the controller does not read the frame).
- Its own commands $u_{t-1}$, $f_{t-1}$, $m_{t-1}$ (efference copy).
- Telemetry $p_t$: a coarse noisy map of the resource, $p_t = \mathrm{pool}_k(a_t) + \nu_t$, where $k$ is the block size. The values of $k$ and $\nu_t$ are tuned on control F4. The resolution must be enough to distinguish the probe offset in E1.
- Its own recurrent state from the previous step.
- Reward for the external task during training.

### What the agent does not receive

- The clean frame $I_t$.
- True coordinates, velocities and identifiers of objects.
- The true $a_t$, $h_t$, $\bar c_t$, $S_t$ and any capture flag. Exceptions: level 0 with a built-in schema, and the oracle C6.
- Labels such as "attention has just been captured" and similar hints.
- Reward for the focus matching the target in the target mode.
- Service data about the current phase of a scenario.

The full state of the world and of attention is available **only to the independent evaluator**.

## Required MVP core

The following are **design requirements**, not a list of proven conditions for consciousness.

| ID | Requirement | Unacceptable simplification |
| :-- | :-- | :-- |
| A1 | Perceptual model $z_t$: position, motion and a continuous feature of objects, built from $x_t$ | Ready-made coordinates from the world engine |
| A2 | Real attention mechanism: competition for a limited resource that changes the availability of information | A decorative frame; a state of a few numbers |
| A3 | Attention dynamics: endogenous commands, exogenous capture, inertia, noise | The focus always equals the command |
| A4 | Attention schema $s_t$: compact, updated automatically, estimates and predicts the state of attention | A copy of $a_t$ at the controller's input; a record of past coordinates without predictions; updating at the controller's choice |
| A5 | Closed-loop control: the controller uses the schema's predictions when choosing a command | A schema that influences nothing |
| A6 | Prediction of functional consequences: what becomes distinguishable after a command. This is the core of the schema[^4] | Predicting only focus coordinates |
| A7 | Minimal self-model $e_t$: commands, the controllable paddle, the prediction error of one's own actions | A constant `agent_id` |
| A8 | Relational representation $r_t$: a link between self, attention and a specific object | Feeding $z_t$ and $s_t$ independently into separate layers |
| A9 | Nonverbal report: choosing a response by estimated availability of information | A threshold on first-order confidence that bypasses the schema |
| A10 | Interventions: independent substitution, freezing and perturbation of $a_t$, $s_t$, $r_t$, $p_t$ and of their links | Assessment by external behavior only |
| A11 | Independent evaluator with the world's truth and with probes | Probe weights in the agent's working loop |

The most important combination is **A2 + A4 + A5**: real attention, its model, and causal use of that model.

**Beyond the first MVP:** language, biography, social theory of mind, decision and emotion schemas, transformers, self-preservation, pain and emotional motivation.

## How to build perception and the schema

### First order: content

**Design decision.** In version A perception is hand-written: a filter over a limited number of object slots that works from $x_t$ only.

Why: the bench's question concerns the model of attention, not representation learning. With fixed perception it is also certain that $z_t$ did not change when the schema is substituted. The price is that the quality space is not tested in version A (see below).

Slot record:

| Field | Purpose |
| :-- | :-- |
| `position`, `velocity` | Estimate of position and motion |
| `hue` | Continuous feature |
| `evidence_support` | Support from the current input |
| `prediction_support` | Support from prediction |
| `uncertainty` | First-order uncertainty |

Perception estimates local noise from the frame itself. **It does not read $s_t$, $a_t$ or $p_t$.** The controller does not read $x_t$ or the local noise map. Otherwise the position of attention becomes visible to the first order, which must be tested separately (C7).

Slots are for objects only and the background is not tracked: otherwise the `uncertainty` field would give away the position of the resource directly. The uncertainty of object slots still depends on the resource. This is a legitimate path, since the schema is also built from the consequences of attention, but its contribution is measured separately (F4). The most important case is when the resource has been pulled to an empty place far from the objects.

The key boundary: "red dominates the estimate of the object" and "the system represents itself as perceiving a red object" are different computational states.[^2]

### Attention schema

The schema is a compact recurrent module:

$$
s_t = \mathrm{GRU}_\phi\bigl(s_{t-1},\, [u_{t-1}, p_t, \mathrm{pool}(z_t)]\bigr),
\qquad s_t \in \mathbb R^{d_s}.
$$

The schema updates on every step automatically. The controller does not choose whether to update it and cannot turn it off.[^4]

Simplification comes from the small dimension $d_s$ relative to the map over the whole field: the starting grid is {4, 8, 16}. Variational and discrete bottlenecks are an extension.

Two predictive heads:

$$
\hat p_{t+1} = f^{\mathrm{state}}(s_t, u_t),
\qquad
\hat{\mathcal I}_{t+1} = f^{\mathrm{cons}}(s_t, u_t, z_t).
$$

The first predicts the next own signal about attention. The second predicts how much the uncertainty of the selected slot will fall after a command. The second head matters more: according to Graziano the schema encodes relationships such as "If my resources are focused on X, then I can understand and react to X; but if I let my focus move to Y, then X will be pushed out of my focus".[^4]

The evaluator measures what the schema keeps of the map $a_t$ and what it loses. Systematic simplifications, such as one focus in place of a split resource, are not counted as defects: they are what is being studied.

The list of what is represented is an engineering decomposition, not a prescription of AST. These properties are checked by probes, not set as fields:

| Aspect | What it represents |
| :-- | :-- |
| Current distribution | Where resources are mainly allocated |
| Switching | Whether a shift is still in progress |
| Controllability | How closely the focus follows commands |
| Perturbation | Whether there are signs of unintended capture |
| Consequences | What information will be available |
| Uncertainty | How reliable the estimate itself is |

Schema variants must not be mixed when describing a result:

| Variant | What is provided | Where it is used |
| :-- | :-- | :-- |
| Built-in | An error-free map of the focus, as in the 2021 work | Level 0 |
| Learned descriptive | An estimate of the current state without predictions | Control C8 |
| Learned predictive | $s_t$ and both heads | Version A |
| Emergent | A free resource with no assigned role | Version B |

### Minimal self-model and the relation

The minimal "self" is a model of the system's causal boundaries, not a name or a biography: which commands come from the agent, what it can change, which consequences of its actions it predicts. The vector $e_t$ accumulates the efference copy, the paddle position and the prediction error of one's own actions. This is the simplest body schema[^4], matching indicator AE-2.[^13]

The relational representation:

$$
r_t = Q_\psi(e_t, s_t, z_t),
$$

where $Q_\psi$ in the MVP is single-head attention over slots: a query from $[e_t \parallel s_t]$ over keys and values from $z_t$. This is a minimal analogue of the workspace where self-models and sensory models are interrelated and can affect action.[^4] By itself it does not generate consciousness.

Its intended meaning: "this agent has a certain access to this object through its current attention". **This describes the meaning for the developer; it is not a string in the program.**

### Feedback

The schema must control at least:

1. **The choice of the next command:** the controller receives predictions for a set of candidate commands.
2. **Holding and suppression:** the costly effort $f_t$ when loss of the target or a resource leak is expected.
3. **The choice of response:** answer, keep looking, or abstain.
4. **Correction:** a mismatch between predictions and observations triggers either a correction of attention or an update of the model.[^4]

Control reads the schema directly and continuously. Report receives it only through the relation $r_t$. This matches the distinction between the schema's continuous work and the only partial access to it.[^4]

The key requirement:

> The model of attention is needed because without it the system controls its own processing worse, not because the developer wants a report about it.

This is an engineering hypothesis that must be tested with interventions.

### Timing

The order of computation in a step rules out looking into the future:

1. The world moves to $I_t$.
2. The attention mechanism computes $a_t$ from the command at $t-1$, capture and noise.
3. $x_t$ and $p_t$ are formed.
4. Perception updates $z_t$.
5. $e_t$ and $s_t$ are updated.
6. $r_t$ is computed.
7. $\hat p_{t+1}$ and $\hat{\mathcal I}_{t+1}$ are computed for candidate commands.
8. The controller chooses $u_t$, $f_t$, $m_t$ and the report action.
9. On the next step predictions are compared with observations.

An endless "model of a model" is not needed: a model of the world and a model of attention that takes part in control are enough.

## Training

The following are **proposed training functions, not Graziano's formulas**.

$$
L = L_{\mathrm{task}} + \lambda_a L_{\mathrm{attention}} + \lambda_c L_{\mathrm{consequence}},
$$

$$
L_{\mathrm{attention}} = \lVert \hat p_{t+1} - \mathrm{sg}(p_{t+1}) \rVert^2,
\qquad
L_{\mathrm{consequence}} = \lVert \hat{\mathcal I}_{t+1} - \mathrm{sg}(\Delta H_{t+1}) \rVert^2.
$$

Here:

- $L_{\mathrm{task}}$ is the reinforcement learning loss: PPO with a recurrent controller and truncated backpropagation. The reward includes the cost of effort.
- $\Delta H_{t+1}$ is the change in the agent's own uncertainty about the selected slot.
- $\mathrm{sg}$ is the stop-gradient operator.

**The training target is the agent's own signal, not the truth.** The true $a_{t+1}$ is used as a target only in the "instrumented training" variant, which serves as a positive control and is described separately.

The difference between these variants should not be overstated. Zero-mean telemetry noise averages out in training, and the first head converges to $\mathrm{pool}_k(a_{t+1})$. The own signal differs from the truth in resolution and in the number of examples needed. The schema's sensitivity limit is set by something else: the noise at its input during operation.

Required constraints:

- The schema does not receive the target's position disguised as an internal signal.
- A prediction does not copy already computed future attention.
- Attention does not become static to make prediction easier. Probes watch the share of steps with focus movement.
- The controller does not bypass attention through the clean frame.
- Predictability is not bought by failing the main task.
- There is no reward for internal states or for "awareness".
- Hand-set constants do not contain knowledge that should be learned.

**A low prediction error is not a success in itself.**

The price of banning step rewards is known: without them the task takes much longer to learn.[^8] If the oracle C6 does not learn within the budget, that is a feasibility failure. Reward for attention must not be returned to the compared conditions; the setup is changed instead.

## Indicators of the organization of experience (not a test of qualia)

The question "is there a picture inside?" is replaced by a more precise one: does the system represent its own relation to the content, and does the organization of its processing depend on this? **Even a positive answer is not a test of qualia.**[^13]

### Five separate measurements

The evaluator measures independently:

| Level | Notation | How it is measured | What a mismatch means |
| :-- | :-- | :-- | :-- |
| Objective processing | $z_t \to y_t$ | A probe of the true feature from $z_t$ | The information exists but may not be registered by the schema |
| Actual attention | $a_t$ | The mechanism's truth | Attention is drawn but not reflected in the schema |
| Represented attention | $\hat a_t(s_t)$ | A probe of map summaries from $s_t$ | $\hat a_t \ne a_t$: a dissociation of attention and awareness |
| Estimated availability | $\hat{\mathcal I}$ | Output of the consequence head | The difference between "looking elsewhere" and "looking there, but the contrast is low" |
| Self-attribution | $r_t$ | A probe of the agent-object link and of its effect on report | Maps of the world and of attention exist, but no link between them |

A classifier's confidence scalar does not count as an attention schema.

Each probe is compared with an untrained system, with the raw input, and with a "context only" baseline: step number, episode phase. At level 0 the window always moves down with the ball, so the row follows from time. The attention probe there is computed by column only, as in Piefke et al.[^8]

### Nonverbal report

At the end of an episode, or earlier at the agent's choice, four actions are available:

| Action | Meaning | Cost |
| :-- | :-- | :-- |
| `COMMIT_HIGH` | Answer with a high stake | High reward and penalty |
| `COMMIT_LOW` | Answer with a low stake | Low reward and penalty |
| `SENSE_MORE` | One more step of observation with an attention command | A small cost per step |
| `ABSTAIN` | Decline | Zero outcome |

The choice reads both the summary of $z_t$ and $r_t$. Whether it relies on the schema is a measured result, not wiring. The stakes are tuned on controls so that no action dominates, and then frozen.

Choosing to "look again" shows by itself only a strategy for handling uncertainty. An interpretation in the spirit of AST requires a link to the model of attention specifically.

### What counts as a quality space

Indicator HOT-4 is **not tested** in version A: the hue space is set by hand. The test becomes meaningful only with learned perception: internal similarity of hues must relate to discrimination, memory and behavior. A color vector does not become an experience of red merely because it is named `quale`.

### What does not count as evidence

- The text "I see red".
- A beautiful visualization of the focus and of its estimate.
- A low prediction error on the agent's own signal.
- The fact that a probe can read $a_t$ from a state. The definition in Piefke et al. is weak here: any store with information about attention counts as a schema.[^8] With a deterministic mechanism a plain command history satisfies this definition.
- A drop in results when the schema is zeroed: zero may be a state the network never reaches.
- A drop in the mean result without a change in the control profile: this is not what the theory predicts.[^14][^15]
- A habit of looking at one side of the field instead of a link to the cue.[^15]
- A shift in the wager if the report reads only the schema by wiring.
- A single won comparison without controls.

## Three levels of the experiment

**Level 0: replication.** Reproduction of Wilterson & Graziano and of Piefke et al. The goal is to make sure that the environment and training are correct and to obtain reference numbers.

**Version A: functional bench.** The attention mechanism, perception and the division into modules are set explicitly; the schema and control are learned. It tests hypotheses 1 to 4.

**Version B: emergence.** No schema is assigned: the agent gets a free resource, and the presence of a model of attention is determined by probes and interventions. It tests hypothesis 5. The first stage repeats the Piefke et al. setup with a controllable square on a different algorithm. The next stage is an unstructured recurrent resource, which the authors themselves point to.[^8] It is closer to AST: the agent moves the square itself by choosing actions, whereas Graziano's schema is built automatically, outside choice.[^4]

Version A must not be presented as a test of emergence. Self-organization does not replace any functional experiment.

## Controls and protocol

### Feasibility controls

**Feasibility first, comparison second.** The task is tuned only on these controls, on separate seeds. The compared conditions are not run during this.

| Control | Design | Readiness condition |
| :-- | :-- | :-- |
| C0: no attention | Uniform noise, no enhancement | The task is not solved. If it is, there is too little noise |
| F2: ideal policy | A hand-written policy using the truth of the world and of attention | It solves the task: the task is solvable at this noise and budget |
| C6: attention oracle | The same controller with the true $a_t$ as input | It learns at all learning rates in the grid, including those of the participants |
| F4: separation of sources | Probes of summaries of $a_t$ under a neutral policy from the frame history, from $\mathrm{pool}(z_t)$, from the command history, from telemetry, and from all schema inputs together | No single source gives summaries of $a_t$, and the schema inputs together do |
| F5: fixed point | The agent's own training signal at the true state and with neutral attention | On average it does not lead away from the truth |
| F6: event count | Number and length of captures, actual focus shifts, target exits from the focus | Events of the required kind actually occur |

Control F4 is cheap: it is trained with supervision on a random policy and runs before any reinforcement learning. If it fails, nothing else makes sense.

### Compared conditions

| Condition | Design | What it rules out |
| :-- | :-- | :-- |
| Full loop | Schema with both heads and a link to the controller | – |
| C1: command memory | Command history without a schema or auxiliary goals | It is enough to remember what was ordered |
| C2: equal resources | A recurrent block of the same size, $L_{\mathrm{task}}$ only | Parameter count and memory |
| C3: world task | An auxiliary prediction of the target's position instead of a model of attention | Any auxiliary goal helps |
| C4: no recurrence | The same schema without $s_{t-1}$ (analogue of H3[^9]) | Whether a dynamic model is needed |
| C5: report only | The schema learns and is read by the evaluator but does not reach the controller | An epiphenomenal monitor |
| C7: estimate from the stimulus | Controller input extended with a hand-written estimate of the focus from the frame | A strong first order: the schema merely duplicates the frame |
| C8: descriptive schema | An estimate of the current state without predictions | Whether prediction is needed |
| C9: confidence | Report from the uncertainty of $z_t$ only | Whether report reduces to the first order |

What is matched is the computational budget, signals and history, not only the parameter count. The other hypotheses of Liu et al. are defined for key-value attention; their transfer to a resource map is not defined, so they are not used as controls.[^9]

### How to read the result

C2 may build an implicit model of attention. For AST this is not a refutation but a confirmation: a good regulator models its object. Equal results are therefore read only together with a probe:

| Full loop against C2 | $a_t$ readable from the state of C2 | Verdict |
| :-- | :-- | :-- |
| Better | No | The explicit schema is useful, the regulator hypothesis is supported |
| Better | Yes | The explicit schema is useful, C2 partly built an implicit model |
| Equal | Yes | The regulator is supported, a separate module brings no benefit. Compatible with AST |
| Equal | No | Control without a model of attention. An argument against the hypothesis in this setup |

A probe counts as readable only if it beats the baselines from the section on measurements.

The substitution check C7 applies only if there is an advantage over C2. Otherwise its result is "not applicable", not "ruled out".

If schema features are computed by hand from the first-order state, any policy that reads them has a first-order equivalent. The schema is therefore learned from version A onward.

### Comparison protocol

These rules come from the `coherence-lab` bench, where breaking them produced false conclusions several times:

1. Design only on controls, on separate seeds.
2. Freeze the setup.
3. One tuning of learning rates and weights $\lambda$ on control C2, not on the full loop.
4. One comparison on new seeds. Spent seeds are not reused.

The unit of observation is an independent training run. Steps and episodes of one run do not count as independent trials. The comparison is paired: 30 pairs, or at least 15 if the budget falls short. The estimate is a 95% bootstrap interval of the difference. The effect threshold and the number of pairs are fixed at the freeze.

For each metric a cheap bypass is sought in advance and tested under conditions where it does not work. A metric must measure model formation directly: the final task can saturate earlier.

## Main causal experiments

Each mechanism requires comparison with a suitable control. Each condition differs by a real mechanism, not by a label in the interface. "Expectation" here is a prediction of the theory, not a requirement on the result.

### E1. Schema ablation

Compare:

- the full loop;
- after training: random substitution of the schema state with a real state from another episode, substitution of the mean, zeroing;
- training without a schema from the start;
- an alternative controller with equal resources (C2).

Random substitution is the main variant: it stays within reachable states.[^8] Zeroing is recorded separately and can exaggerate the effect: in Piefke et al. removing the resource gave −0.82 on tracking, and random resource actions gave 0.93.[^8] The first tests acquired dependence, training from scratch tests the benefit during learning, and C2 tests the specificity of the mechanism.

The measures follow the control-theory predictions[^14] and the anticipation effect[^15]:

| Measure | How it is computed | Expectation without a schema |
| :-- | :-- | :-- |
| Exogenous effect | Share of resource drawn to a flash on the first step | Preserved |
| Mean strength of attention | Mean share of resource on the target | May not fall |
| Stability | Fluctuation of the share of resource on the target over time | Fluctuations grow |
| Leaving an undesired state | Time for the resource to leave an irrelevant flash, and the dip below baseline (analogue of inhibition of return) | Slower departure, no dip |
| Sensitivity to perturbation | Slope of capture against event strength | Steeper slope |
| Anticipation | The paradigm below | Almost disappears |

The flash in the departure measure is fully irrelevant: neither report nor reward concerns it. In humans the dip below baseline appeared only when the stimulus was not needed for the response. If it is needed, attention stays on it and there is no dip.[^14]

**The anticipation paradigm of Wilterson et al.**[^15] A cue appears at one of many random places and draws attention. A few steps later a brief probe stimulus appears next to it, and its hue must be reported. In 70% of cases it is offset from the cue to one constant side; in the others it is on the opposite side or at the cue's location. The exogenous effect is the advantage of the cue's location. The endogenous effect is the advantage of the predicted offset over the opposite one.

This test fits the bench especially well. The controller has to shift the resource relative to the place it was pulled to, so it must know that place. By the authors' logic, without awareness of the cue it is the descriptive part of the model that fails, not the predictive part.[^15]

The bypass is a habit of looking at one side of the field. Therefore there are many cue locations, and the measure is computed relative to the cue, not to the field. Besides the final value, the learning curve is recorded: in humans without awareness of the cue there is a hint of very slow learning, and the authors cautiously suggest two mechanisms, a fast model-based one and a slow model-free one.[^15] So C2 is compared with the full loop over the whole curve, not only at the end: it may catch up in the long run.

**Expectation under AST:** capture and mean strength are preserved, control gets worse.[^3][^14][^15] The human data come from an unnoticed stimulus, not from removal of the whole schema, so a matching profile is an analogy. The authors themselves stress that such matches do not prove the theory.[^14][^15]

### E2. Double dissociation

**2A. False schema with attention intact.** Attention is on the target. At step $t_0$ the schema state is replaced by a recorded state from a real episode in which the focus was on the other side. The frame, $z_t$ and $a_t$ do not change.

Measure immediately, on the same step: the direction of the next command and the choice of report.

**Expectation:** the command shifts in the direction that compensates for the false error and pulls the real focus off the target. The wager falls although $z_t$ is the same. This tests hypothesis 4: one source of error, both effects.[^1]

The effect must be directional: the sign of the shift depends on the side of the substitution. General degradation does not count. After a new command the observations naturally diverge, so immediate and delayed effects are analyzed separately.

**2B. Hidden perturbation of attention.** The resource is shifted by an impulse, telemetry is blocked for one step, and the schema does not change.

Measure: whether the mismatch is detected, what happens next (control is corrected or the model is updated[^4]), and how many steps it takes. This separates monitoring from remembering commands. **Expectation:** C1 keeps assuming the focus is where it was sent, while the schema detects the mismatch through telemetry and consequences.

### E3. Broken feedback

A comparison of the full loop with C5. The prediction of attention is good, but it cannot be used for choice.

**Testable hypothesis:** describing attention and controlling it with a model are different functions. If breaking the link changes nothing, the schema is a diagnostic by-product.

### E4. Virtual neglect

An analogue of damage to the right TPJ: with processing intact, awareness and endogenous control are lost for one side of space, while some exogenous attention remains.[^1] The right TPJ is related specifically to control of attention by an aware cue.[^16]

Intervention: when a probe shows that the schema represents the focus in the left half, its state is replaced by the nearest recorded state with the focus on the right. Perception and the attention mechanism are not touched. The probe here is the experimenter's tool: it is not part of the agent's working loop and does not train it.

**Expectation:** a flash on the left still shifts the focus, the target's feature on the left is readable from $z_t$, but there is no endogenous search to the left, and reports on left stimuli shift toward abstaining.

### E5. Distractor below the registration threshold

What Tsushima et al. showed[^17]: with a central task, task-irrelevant background motion interfered at 5% coherence, below the discrimination threshold, while at 20% and above performance did not differ from zero coherence. When contrast was lowered, the threshold and the dip shifted together. The authors suggest that the prefrontal cortex has a higher detection threshold than the visual cortex, so a weak signal does not trigger inhibition. Graziano reads this as a leak: the model failed to depict the resource going to the distractor, so the person neither reports it nor regulates it. In his words, "The controller cannot regulate the leak if it does not know about the leak."[^15]

The setup in the bench:

- The distractor is unrelated to the task, and its strength varies over a grid. It takes a share of the common resource. This is a leak, not a shift of gaze: Tsushima et al. found no systematic differences in eye movements.[^17]
- Nothing is masked from the schema. Its sensitivity limit is set by telemetry noise.
- Suppression is the costly effort $f_t$.
- The threshold is determined from data: the strength at which a probe registers the stimulus from $s_t$ in half of the cases.
- The stimulus is not part of the agent's report: otherwise it becomes relevant and the effect disappears (check 3 below). Registration is measured by a probe. A behavioral threshold, if needed, is measured in a separate block after the main one, as the authors did.[^17]

**Launch condition.** Two hand-written policies must show that the dip is not created by the mechanism itself: with constant effort harm grows monotonically with strength, and with effort switched on by the truth there is no dip.

**Expectation from Tsushima et al. as interpreted by AST:**

1. Non-monotonicity: harm peaks just below the threshold, while a stronger stimulus is suppressed and does less harm.
2. Shift: as telemetry noise grows, the threshold and the dip move together. The noise changes equally in training and testing. This is an analogue of the second experiment of Tsushima et al. and the strongest of these checks: the threshold is measured by a probe and the dip by the task.
3. Relevance: if the stimulus matters for the task, there is no dip.[^17]
4. Link between functions: below the threshold the stimulus cannot be read from $r_t$, and effort does not switch on.[^15]

A monotonic dependence is a negative result, not a fault of the setup. This is a risky prediction, and it must not be set by wiring.

### E6. Schema emergence (version B)

The Piefke et al. setup: a free resource with its own actions and a noise grid p = 0, 0.25, 0.5, 0.75, 1.

Differences from the paper:[^8]

- The algorithm is the level 0 DQN instead of PPO, with a network of three layers of 200. The observation is one frame with no memory buffer, as in the paper.
- The discount is 0.9, as in the Piefke et al. code.
- Several independent training runs per level instead of one.

This tests the same conclusion with another algorithm; it is not an exact replication.

**Launch condition.** First, at p = 0.5, a positive control with a built-in schema and a negative control without a resource are trained. If the built-in schema does not learn, the result with the free resource is not interpreted.

Measure:

- usefulness: the difference in results between learned resource actions and two baselines, random actions and a resource that never leaves its starting column;
- content: the window position by column from the resource, from the stimulus and from both, as lift over the most frequent class;
- the difficulty of inference from the stimulus.

The motionless baseline is needed because a square left in place already stores the starting column: with a tracking policy it predicts the window's column with about 0.41 accuracy against 0.25 for random actions. An agent that never learned to use the resource would otherwise look as if a schema had emerged.

**Expectation:** the usefulness and content of the resource peak at intermediate noise and relate to the difficulty of inference. In the paper usefulness peaks at 0.5 and difficulty at 0.25, and the relation is assessed qualitatively.[^8]

The efference copy is checked as well. If, with an undisturbed mechanism, command history replaces the emerged schema, the resource turns out to be a memory of commands and not a model of attention. A conclusion about emergence therefore counts only after repetition in the target mode with capture and mechanism noise.

### E7. Transfer, adaptation and generalization

The ability to adapt to new conditions is one of the main properties of an internal model.[^15]

- **Adaptation.** A constant command offset is introduced, like the prism in the motor experiments the authors cite[^15]: $B(\bar c_t + u_t + \delta)$. Measure the adaptation speed and the after-effect once the offset is removed. Compare the full loop with C2.
- **Generalization.** The link between cue and probe stimulus is trained at some locations and tested at neighboring untrained ones. In humans, adaptation from 16 locations transferred to 6 untrained ones, where the probe was equally likely on both sides, and faded after the first block.[^15] The bench can measure this more cleanly: first with frozen weights, then with updates, to see the fading separately. The recurrent state changes even with frozen weights, and this is taken into account. Purely local adaptation points to a low-level process, not to a general rule of a model.
- **Transfer.** New delays, capture strength, object combinations and flash rates.

## Extensions beyond the MVP

These run only after the MVP is accepted, each with its own freeze.

| Extension | What it tests | Conditions |
| :-- | :-- | :-- |
| E8: self-regularization | Self-modeling simplifies a network[^10] | The target is the controller's hidden activations without `stop_grad`, as in the source, where both prediction and target depend on the weights.[^10] The main loop has no such mechanism: its target is detached and does not come from the network. Measured: width of the last layer's weights, effective rank of the covariance, accuracy. RLCT is estimated by stochastic Langevin dynamics and is expensive, so it runs only in Rust. The weight grid is chosen for the architecture: small networks break down.[^10] The effect was not shown on attention, and its transfer to attention is not established |
| E9: external predictability | The schema makes an agent's attention more predictable to an observer[^11] | A separate observer network reading traces of attention. Comparison with C2 |
| E10: cooperation | The social function of the schema[^11][^9] | Pairs "schema + schema", mixed, "control + control", with no communication channel. This is not a replication: Farrell et al. use transformers and a coloring task |
| Learned perception | HOT-4, feature binding | Replacing the hand-written filter, repeating E1 to E3 |
| Discrete bottleneck | Schema robustness, an ASAC variant[^12] | Comparison with the simple limit $d_s$ |
| Decision schema | The society of schemas[^4] | A model of the process of choice, not of the contribution of commands. Testable prediction: an account of a past decision influences the next one in a series[^4] |
| Attention in James's sense | A schema of representation enhancement in large networks[^4] | A separate project on a language model. It cannot be done on small networks in TS and Rust, and this branch matters most for the question of consciousness itself |
| Local substrate | A distributed schema | Based on the experience of version B in `coherence-lab` |

## What to build on

| Material | What to take |
| :-- | :-- |
| Wilterson & Graziano 2021[^7] | The setup, reference numbers and parameters. Code and data[^20] |
| Piefke et al. 2024[^8] | The free resource, probes, random substitution. Code: https://github.com/KietzmannLab/Attention-Schema-Analysis |
| Liu et al. 2023[^9] | The breakdown of relations between attention and control, recurrent control. The authors call the results preliminary and the advantages marginal |
| Farrell et al. 2024[^11] | The social extension. Code: https://github.com/kathrynfarrell/attention_schemas_in_anns |
| `coherence-lab` | Deterministic arithmetic, TS and Rust number compatibility (`jscompat`), trainable recurrent elements and Adam in `web/src/lib/substrate`, the comparison protocol, the report format |

GitHub shows no license for either repository, and none is stated in the 2021 archive, so ideas and parameters may be taken from them but not code.

Everything should not be merged into one system at once: a failure would then be impossible to localize.

## Practical development specification

**The recommended first result is a reproducible software bench, not an "artificial personality".** What matters is determinism at a fixed seed, the ability to learn, and access to all internal states for interventions.

### Implementation

- **TypeScript is the reference.** Node, no external ML libraries, own networks on `Float64Array`.
- **Rust is the second implementation, compared byte for byte.** Episode traces, weight checksums and command output are compared, as in `coherence-lab`.
- The bench logic does not depend on the interface.
- The world's truth is not passed to the agent through any interface.

### Deliverables

| Component | Requirement |
| :-- | :-- |
| Environment | Level 0 and the target mode; episode geometry depends on the seed |
| Attention mechanism | Priority and resource maps, capture, effort, telemetry; a window for level 0 |
| Perception | A hand-written filter on $x_t$ for the target mode |
| Attention schema | Recurrent module, predictive heads, the variants from the table |
| Self-model and relation | $e_t$, $r_t$ |
| Controller | Attention commands, effort, paddle, report |
| Training | Networks, Adam, PPO, DQN for level 0, truncated backpropagation |
| Interventions | Disabling, substitution, freezing and perturbation of links and states |
| Evaluator | Truth, probes, context baselines |
| Experiment log | Configurations, seeds, curves, verdicts, invalid runs |
| Laboratory | An interface in Russian: the real resource map and the schema's estimate over the noisy frame, the mismatch, intervention buttons, noise and capture sliders. It runs on pretrained weights |

Module names are not evidence that the corresponding functions exist.

### Computational budget

A preliminary prototype measurement on the development machine (Apple M4, one core). One update of a 2000–200–200–200–16 network, 483,816 parameters, batch 100, Adam:

| Implementation | Ms per update |
| :-- | --: |
| TypeScript, Node 26.3.1 | 107 |
| Rust, scalar binary64 without FMA or reordering | 85 |

Under the constraints of byte-for-byte comparison, Rust is only about 1.3 times faster. The working implementation skips zero inputs and takes 56 ms with three parallel training runs and 82 to 104 ms with ten. The main resource is independent runs in parallel across cores, and it is available in Node too through worker threads.

Estimated cost of the replications:

- **Wilterson & Graziano.** 100 updates per iteration, 150,000 in total, 2.5 to 4 hours per training run in Node depending on the number of parallel runs.
- **Piefke et al.** About 5 million parameters, roughly 11 times heavier, so on the order of 1.2 s per update. Five noise levels are 22,974 iterations and at least 2.3 million updates: about a month on one core with one agent per level. At full size this is not feasible in TypeScript or in Rust.

Hence the rules:

- Level 0 parameters come from the authors' published code. Library defaults and differences from the code are listed in the report.
- E6 runs on the level 0 DQN, not on the authors' PPO. This difference is stated in the report.
- Training runs in parallel by seed.
- Rust is needed as a second independent implementation and for comparison, not for speed.
- Target mode networks must be small, which is one reason for hand-written perception.

### Work order

1. **The level 0 world and attention mechanism.** Check: event count, hand-written ideal policy, C0.
2. **Replication of Wilterson & Graziano.** Built-in schema, ablation, training without it. In addition, replacing the schema with a command history. Check: qualitative agreement with the reference numbers.
3. **Schema emergence (E6, first stage)** on the level 0 DQN. Check: feasibility controls, then a qualitative comparison with Piefke et al. at p = 0.5.
4. **The target world and control F4.** Freeze the setup if the sources are separated.
5. **Control C6 and baseline controls C1 to C3.** Training, fixing the baseline.
6. **The learned schema and the full loop.** C4, C5, C7, C8, experiments E1 and E3.
7. **The relation and report.** C9, E2 and E5.
8. **E4 and E7.**
9. **Version B in the target mode.**
10. **Port to Rust** with byte-for-byte comparison, if it was not needed earlier.
11. **The laboratory** and publication.

Steps 1 to 3 anchor the bench to the literature. Steps 4 to 8 test hypotheses 1 to 4. Step 9 tests hypothesis 5. A positive outcome of all steps is an argument for the control function of the schema in small systems, not for AST as a theory of consciousness.

### Acceptance criteria

**Acceptance of the bench does not depend on whether the hypotheses are confirmed.** The bench is accepted if:

- Attention really changes processing, as shown by C0.
- The command and the actual focus diverge, and control F4 is passed.
- Feasibility controls are passed before comparison, and all setups are frozen.
- Level 0 qualitatively reproduces the published results, or the discrepancy is explained.
- Each intervention changes exactly the stated state and no other. This is checked by tests.
- TS and Rust give byte-identical outputs on control runs.
- Results and invalid runs are recorded, and comparisons are run once on new seeds.
- All conclusions are separated from any claim about phenomenal experience.

**The wording "an agent with a learned, causally effective attention schema" is allowed only if all of the following hold:**

- the full loop beats C2 and C5, and the advantage is not explained by C1, C3 and C7;
- the schema estimates the actual state and does not just repeat commands: it recovers in E2B;
- a false schema causes a directional control error and a shift in report;
- the E1 measures show worse control, not only a lower mean result;
- the advantages reproduce on new trajectories and perturbations (E7).

The wording **"AST-like self-attribution"** additionally requires a meaningful "agent, attention, object" link that affects decisions and not only report, and superiority over C9.

**A negative result with honest controls is an argument against the hypothesis in this setup, not a failure of the work.**

The most meaningful practical demonstration would look like this: the agent misrepresents where its attention is, chooses the wrong command because of this, detects the mismatch from the consequences, and corrects its own model.

Protocol fields for each hypothesis:

| Field | Allowed values |
| :-- | :-- |
| Indicator | AST-1, AE-2; HOT-4 is not tested in version A |
| Mechanism | Implemented / not implemented |
| Causal role | Confirmed / not confirmed, with difference and interval |
| Alternative explanation | Ruled out / not ruled out / not applicable, separately for C1, C2, C3, C7 |
| Setup | Valid / invalid, with the reason |
| Phenomenal consciousness | Not established |

**The protocol must not contain a line "qualia: PASS".**

At this stage there is no need to introduce pain, fear of shutdown or suffering: neutral tasks of perception and control are enough. Working in stages and minimizing possible harm follows published principles for responsible research on machine consciousness.[^18]

## Sources

[^1]: Graziano M. S. A. *A conceptual framework for consciousness.* PNAS 119(18), 2022. https://doi.org/10.1073/pnas.2116933119 · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC9170028/

[^2]: Webb T. W., Graziano M. S. A. *The attention schema theory: a mechanistic account of subjective awareness.* Frontiers in Psychology 6:500, 2015. https://doi.org/10.3389/fpsyg.2015.00500

[^3]: Graziano M. S. A. *The Attention Schema Theory.* Chapter 2.4 of the section Theoretical Frameworks and their Testable Predictions, chapter manuscript posted in 2026. https://static1.squarespace.com/static/6605831012d79b3ae78b24e0/t/6a0f6f245f4e53042aa75135/1779396388801/Ch._2.4_Graziano.pdf

[^4]: Graziano M. S. A. *Understanding consciousness (biological and artificial) as a society of self-schemas.* PNAS 123(42), 2026. https://doi.org/10.1073/pnas.2628598123

[^5]: Graziano M. S. A. *The Attention Schema Theory: A Foundation for Engineering Artificial Consciousness.* Frontiers in Robotics and AI 4:60, 2017. https://doi.org/10.3389/frobt.2017.00060

[^6]: Graziano M. S. A., Guterstam A., Bio B. J., Wilterson A. I. *Toward a standard model of consciousness: Reconciling the attention schema, global workspace, higher-order thought, and illusionist theories.* Cognitive Neuropsychology 37(3–4), 2020. https://doi.org/10.1080/02643294.2019.1670630

[^7]: Wilterson A. I., Graziano M. S. A. *The attention schema theory in a neural network agent: Controlling visuospatial attention using a descriptive model of attention.* PNAS 118(33), 2021. https://doi.org/10.1073/pnas.2102421118 · PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC8379943/

[^8]: Piefke L., Doerig A., Kietzmann T., Thorat S. *Computational characterization of the role of an attention schema in controlling visuospatial attention.* arXiv:2402.01056, 2024. https://arxiv.org/abs/2402.01056

[^9]: Liu D., Bolotta S., Zhu H., Bengio Y., Dumas G. *Attention Schema in Neural Agents.* arXiv:2305.17375, 2023. https://arxiv.org/abs/2305.17375

[^10]: Premakumar V. N., Vaiana M., Pop F., Rosenblatt J., Schwerz de Lucena D., Ziman K., Graziano M. S. A. *Unexpected benefits of self-modelling in neural systems.* Philosophical Transactions of the Royal Society A 384, 20240531, 2026. https://doi.org/10.1098/rsta.2024.0531 · details were checked against the preprint: https://arxiv.org/abs/2407.10188

[^11]: Farrell K. T., Ziman K., Graziano M. S. A. *Testing Components of the Attention Schema Theory in Artificial Neural Networks.* arXiv:2411.00983, 2024, updated in August 2025. First version: *Improving How Agents Cooperate: Attention Schemas in Artificial Neural Networks.* https://arxiv.org/abs/2411.00983

[^12]: Saxena K., Jurado Ruiz F., Manzi G., Liu D., Lamb A. *Attention Schema-based Attention Control (ASAC): A Cognitive-Inspired Approach for Attention Management in Transformers.* arXiv:2509.16058, 2025. https://arxiv.org/abs/2509.16058

[^13]: Butlin P., Long R. et al. *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness.* arXiv:2308.08708, 2023. https://arxiv.org/abs/2308.08708 · the final review in Trends in Cognitive Sciences (https://doi.org/10.1016/j.tics.2025.10.011) was not checked.

[^14]: Webb T. W., Kean H. H., Graziano M. S. A. *Effects of Awareness on the Control of Attention.* Journal of Cognitive Neuroscience 28(6), 2016. https://doi.org/10.1162/jocn_a_00931

[^15]: Wilterson A. I., Kemper C. M., Kim N., Webb T. W., Reblando A. M. W., Graziano M. S. A. *Attention control and the attention schema theory of consciousness.* Progress in Neurobiology 195, 2020. https://doi.org/10.1016/j.pneurobio.2020.101844

[^16]: Wilterson A. I., Nastase S. A., Bio B. J., Guterstam A., Graziano M. S. A. *Attention, awareness, and the right temporoparietal junction.* PNAS 118(25), 2021. https://doi.org/10.1073/pnas.2026099118

[^17]: Tsushima Y., Sasaki Y., Watanabe T. *Greater disruption due to failure of inhibitory control on an ambiguous distractor.* Science 314(5806), 2006. https://doi.org/10.1126/science.1133197

[^18]: Butlin P., Lappas T. *Principles for Responsible AI Consciousness Research.* arXiv:2501.07290, 2025. https://arxiv.org/abs/2501.07290

[^19]: Conant R. C., Ashby W. R. *Every good regulator of a system must be a model of that system.* International Journal of Systems Science 1(2), 1970.

[^20]: Wilterson A., Graziano M. *The Attention Schema Theory in an Artificial Neural Network Agent: Controlling Visuospatial Attention Using a Descriptive Model of Attention* (code and data). Princeton Data Commons, 2021. https://doi.org/10.34770/8bd1-1x14
