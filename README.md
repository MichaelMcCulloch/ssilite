# ssilite

A PyTorch reconstruction of one narrow question:

> Can a learner jointly decide what failures define its objective, which
> examples efficiently estimate that objective's gradient, and where numerical
> precision is worth its cost?

The first prototype separated acquisition, robust weighting, sampling, and
precision. Fable's follow-up then exposed the missing variable: a scalar loss
ranking cannot recover a coherent partition. This repository now tests whether
out-of-fold students trained under deliberately different, label-free
environments can supply that structure.

This is a mechanism experiment, not evidence about SSI's private research.

## The variables

For loss `ell_i(theta)`, the original controller keeps four choices separate:

```math
a:\text{ acquire support},\qquad
q:\text{ objective weights},\qquad
p:\text{ gradient proposal},\qquad
b:\text{ numerical precision}.
```

Given a base measure `mu`, the capped entropic adversary is

```math
q_i=\min\{\mu_i/\alpha,
c\mu_i\exp((\ell_i-\ell_i^{\mathrm{ref}})/\tau)\}.
```

The proposal and unbiased quantized estimator are

```math
p_i^{\mathrm{raw}}\propto
q_i\sqrt{\lVert g_i\rVert^2+\sigma_i^2(b_i)},
```

```math
\widehat G=
\frac1B\sum_{j=1}^B
\frac{q_{I_j}}{p_{I_j}}Q_{b_{I_j}}(g_{I_j}).
```

Stochastic rounding makes `Q` conditionally unbiased. The precision allocator
buys the largest reduction in

```math
\sum_i \frac{q_i^2}{p_i}\sigma_i^2(b_i)
```

per unit expected cost.

The follow-up adds an environment assignment and a population:

```math
e_i=C(x_i),\qquad
u_i=\frac1R\sum_{r=1}^R
P_{\theta_{r,e_i,-i}}(y_i^{\mathrm{obs}}\mid x_i).
```

`C` is label-free clustering. The subscript `-i` means every score is out of
fold: the matched specialist never trained on the label it grades. The final
environment adversary operates on partitions rather than individual loss
ranks:

```math
L_k(\theta)=
\frac{\sum_{i:e_i=k}u_i\ell_i(\theta)}
{\sum_{i:e_i=k}u_i},
\qquad
\max_{r\in\mathcal U(\mathrm{Unif}(K))}
\sum_k r_k L_k(\theta).
```

## Run it

The command-line entry points default to CUDA when it is available:

```console
uv sync
uv run ssilite --device cuda
uv run ssilite-followup --seeds 0 1 2 3 4 --device cuda
uv run ssilite-reconstruction --seeds 0 1 2 3 --device cuda
uv run ssilite-sample-efficiency \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --budgets 0 32 64 128 256 \
  --device cuda
```

`ssilite` runs the original acquisition plus `q/p/b` prototype.
`ssilite-followup` runs the corrected version of Fable's trust bootstrap.
`ssilite-reconstruction` runs ordinary students, environment specialists,
permuted-environment controls, a partition-level adversary, and final routed
student populations.
`ssilite-sample-efficiency` runs the lean nested-support curve without the
out-of-fold trust filter.

The reported experiments below used:

```text
NVIDIA GeForce RTX 4090
PyTorch 2.13.0+cu130
CUDA runtime 13.0
```

The tiny MLPs do not saturate a 4090. CUDA here establishes the executed
device, not a speedup claim.

## Verify

```console
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
uv run --with mypy mypy src/ssilite
uv build
```

Current local verification: 62 tests, Ruff lint and format, Mypy, and source
plus wheel builds.

## Layout

- `acquisition.py`: label-free support expansion.
- `risk.py`: empirical CVaR and capped entropic robust weights.
- `allocation.py`: variance-aware sampling and budgeted precision.
- `quantization.py`: deterministic score and unbiased gradient quantization.
- `estimator.py`: vectorized per-example gradients.
- `controller.py`: damped `q/p/b` state.
- `bootstrap.py`: leakage-safe reconstruction of Fable's dynamics bootstrap.
- `environment_ensemble.py`: cross-fitted ordinary and environment-specialist
  students, iterative trust, and permanence control.
- `adversarial.py`: bounded dual optimization over discovered environments.
- `environment_mixture.py`: equal-compute final ordinary and routed specialist
  populations.
- `sample_efficiency.py`: nested-support, matched-backward-compute population
  benchmark with grid-censored target crossings.
- `scripts/plot_sample_efficiency.py`: paired uncertainty plots and compact
  CSV/JSON artifact generation for every final-population control.
- `experiment.py`: original support-acquisition experiment.
- `followup.py`: corrected Fable controls.
- `reconstruction.py`: paired SSI-hypothesis experiment and counterexample.

## What Fable established

Fable's exact NumPy experiments correctly killed naive sample-level DRO:

- clean rare failures and corrupted labels are both high-loss;
- a bounded uncertainty set limits damage but does not distinguish them;
- same-objective ensemble disagreement inherits the same bias;
- a self-referential grade/re-trust loop exhibits backaction;
- reweighting fixed support cannot manufacture information.

The original bootstrap also had avoidable confounds: students graded their own
training labels, tied ranks were arbitrarily ordered, and there was no damping,
permanence, common initialization table, or multi-seed control.

The PyTorch reconstruction fixes those confounds. Every label-dependent score
is repeated K-fold out of fold, ranks use average midranks, replicas and batch
streams are paired, trust moves under an explicit bound, and reversible versus
permanent distrust is selectable.

It still does not solve the problem.

Across five CUDA seeds, the corrected same-environment bootstrap produced:

| Minority accuracy | Mean | SD |
| --- | ---: | ---: |
| Raw-loss, fixed support | 0.535 | 0.044 |
| Corrected bootstrap, fixed support | 0.549 | 0.036 |
| Raw-loss, acquired support | 0.657 | 0.084 |
| Corrected bootstrap, acquired support | 0.701 | 0.051 |
| Oracle control, acquired support | 0.813 | 0.059 |

On fixed support it was a real corruption detector—AUROC
`0.891 +/- 0.026`—but assigned mean trust `0.833` to clean majority examples,
`0.573` to clean minority examples, and `0.399` to corrupt examples. It
therefore inherited the blind spot it was meant to repair. None of the ten
fixed/acquired loops converged. A seed-0 permanence run also failed the
convergence criterion after 24 rounds; permanence made the sequence monotone,
not correct or fast.

## The decisive treatment: change the students' environments

The next experiment discovers three environments from raw features, then uses
four equal-budget students:

- ordinary arm: every student sees the empirical objective;
- specialist arm: three students put 80% of their sampling mass on one
  environment each, and one uses environment-balanced sampling;
- permutation control: the same environment counts are randomly reassigned;
- all arms reuse the same folds, independent initialization seed table,
  architecture, optimizer steps, and batch budget.

The primary statistic is the observed-label confidence of the student's
matched environment. Hidden group and corruption indicators are used only
afterward for diagnostics.

Sixteen paired CUDA seeds gave:

| OOF grading arm | Clean rare minus corrupt support | Rare-vs-corrupt AUROC | Rare retained at 0.7 | Corrupt accepted at 0.7 |
| --- | ---: | ---: | ---: | ---: |
| Ordinary students | 0.316 +/- 0.052 | 0.770 +/- 0.037 | 0.457 | 0.081 |
| Environment specialists | **0.618 +/- 0.049** | **0.962 +/- 0.018** | **0.723** | **0.032** |
| Permuted environments | 0.314 +/- 0.059 | 0.759 +/- 0.041 | — | — |

The paired environment-minus-ordinary AUROC difference was `+0.192`, with a
95% interval `[+0.178, +0.207]` and exact sign-flip `p = 0.0000305`.
Environment versus permutation was `+0.203` with the same sign-flip p-value.
Each real arm used 72 fits and 184,320 backward examples per seed.

Generic ensemble rescue is not the statistic. The gap between the best student
and the population mean rose by about the same amount on clean rare and corrupt
examples. The discriminating object is *matched specialist confidence*: a
student trained in the relevant coherent environment, grading out of fold.

## Why weights and group DRO were not the result

A fresh single learner received the OOF signal under full-precision,
full-support gradients. Adaptive `p`, quantized `b`, and per-example raw-loss
DRO were disabled so they could not explain the outcome.

| Fresh single learner | Minority accuracy | Majority accuracy |
| --- | ---: | ---: |
| Ordinary-ensemble trust | 0.820 +/- 0.028 | 0.881 +/- 0.016 |
| Specialist trust | 0.828 +/- 0.026 | 0.881 +/- 0.014 |
| Static equal-environment objective | 0.826 +/- 0.024 | 0.875 +/- 0.015 |
| Environment adversary | 0.826 +/- 0.026 | 0.885 +/- 0.016 |
| Oracle environment/noise control | 0.878 +/- 0.021 | — |

Specialist trust improved minority accuracy by only `0.79` points over ordinary
trust, with a 95% interval `[+0.07, +1.50]`. Static balancing and the dynamic
environment adversary were not significantly better than ordinary trust.

The signal was real, but collapsing the population back into one set of
weights discarded most of it.

## Keep the population alive

The final test trains equal-compute populations and evaluates four causal
controls:

- ordinary students, probability averaged;
- ordinary students, routed through the same gate;
- specialists, naively averaged;
- specialists, routed to the nearest discovered environment;
- specialists trained and routed on size-preserving permuted environments.

Both real final populations receive the same OOF trust filter. Their
initialization seeds, batch streams, model count, and optimizer budget are
identical: four fits and 20,480 backward examples per arm.

Across 16 paired CUDA seeds:

| Final population | Minority | Majority | Overall | Balanced log loss |
| --- | ---: | ---: | ---: | ---: |
| Ordinary mean | 0.648 +/- 0.039 | 0.916 +/- 0.017 | 0.782 | 0.493 |
| Ordinary routed | 0.637 +/- 0.037 | 0.907 +/- 0.017 | 0.772 | 0.525 |
| Specialist mean | 0.622 +/- 0.029 | 0.913 +/- 0.015 | 0.768 | 0.445 |
| Permuted specialist routed | 0.631 +/- 0.045 | 0.890 +/- 0.018 | 0.761 | 0.563 |
| **Real specialist routed** | **0.893 +/- 0.021** | **0.940 +/- 0.010** | **0.916** | **0.200** |

Real routed specialists beat ordinary averaging by `+24.46` minority points,
95% interval `[+22.29, +26.63]`, and `+2.40` majority points. They beat the
permuted routed specialists by `+26.12` minority points. Exact paired sign-flip
p-values were `0.0000305`.

Ordinary student predictions had mean correlation `0.968`; specialist
predictions had correlation `0.607`. Averaging those specialists was worse,
because it recombined incompatible mechanisms. The useful unit is the pair
`(specialist, environment gate)`, not diversity in isolation.

These small final students are an equal-compute causal comparison with one
another. Their absolute accuracies should not be compared directly with the
larger two-hidden-layer model in `experiment.py`.

## Quantifying label efficiency

The clean curve removes the expensive out-of-fold trust filter. For each of 16
paired CUDA data seeds, it creates one 256-point acquisition ordering and
reuses nested prefixes. At every prefix the ordinary population and routed
specialists see identical labeled support, model seeds, batch streams, four
model fits, and 20,480 backward examples. Environment discovery sees raw
features but no labels.

Rare-group counts below are post-hoc diagnostics; group membership is never
given to acquisition or training.

![All population variants against the ordinary baseline](artifacts/sample_efficiency_variants.png)

[Vector figure](artifacts/sample_efficiency_variants.svg),
[summary data](artifacts/sample_efficiency_variants.csv), and
[paired seed-level results](artifacts/sample_efficiency_variants.json) are
included. The bottom panels subtract the ordinary mean within the same seed and
budget before computing uncertainty.

The controls identify the active conjunction. At 64 new labels, minority
accuracy was `0.566` for ordinary averaging, `0.569` for routing ordinary
students, `0.608` for averaging specialists, `0.575` for specialists routed
through permuted environments, and `0.867` for specialists routed through the
real feature environments. Routing, specialization, and decorrelation each
fail alone; coherent specialization plus the matching gate survives.

| New label queries | Total labels | Rare among queries | Total rare support | Ordinary minority | Routed minority | Routed 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 512 | 0.0 | 24.88 | 0.513 +/- 0.016 | 0.696 +/- 0.062 | [0.663, 0.729] |
| 32 | 544 | 32.0 | 56.88 | 0.531 +/- 0.019 | 0.821 +/- 0.038 | [0.801, 0.841] |
| 64 | 576 | 64.0 | 88.88 | 0.566 +/- 0.025 | 0.867 +/- 0.020 | [0.857, 0.878] |
| 128 | 640 | 125.88 | 150.75 | 0.632 +/- 0.029 | 0.900 +/- 0.027 | [0.886, 0.915] |
| 256 | 768 | 167.12 | 192.00 | 0.652 +/- 0.040 | 0.907 +/- 0.019 | [0.897, 0.916] |

The `+/-` values are sample standard deviations across data seeds. The final
column is a two-sided Student `t(15)` interval for mean minority accuracy.
Mean majority accuracy stayed between `0.928` and `0.954`.

For a minority target `tau` and a `0.90` majority floor, define the
mean-over-generator grid complexity

```math
B_m(\tau)=
\inf\{B:
\mathbb E[A_{\mathrm{minority},m}(B)]\ge\tau,\quad
\mathbb E[A_{\mathrm{majority},m}(B)]\ge0.90\}.
```

The ordinary curve never reaches even `0.80`, so the result is right-censored:
there is no honest finite point estimate. The observed grid gives lower bounds.

| Target minority mean | Routed grid crossing | Ordinary grid censoring | New-label efficiency | Total-label efficiency | Rare-support efficiency |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | <=32 | >256 | **>8x** | **>1.41x** | **>3.38x** |
| 0.85 | <=64 | >256 | **>4x** | **>1.33x** | **>2.16x** |

At the `0.85` target, the routed mean was `0.867` with interval
`[0.857, 0.878]`; the ordinary mean was only `0.652` at four times the new-label
budget. Fourteen of sixteen routed seeds met the joint `0.85/0.90` target at
64 queries, versus zero ordinary seeds at every evaluated budget. This is
sample efficiency conditional on the generator and target, not a
distribution-free sample-complexity theorem.

The plotted intervals are pointwise and should not by themselves be used to
select a crossing after inspecting the curve. A conservative two-sided
Bonferroni correction across all five budgets still puts the routed
64-query minority lower bound at `0.853` and its majority lower bound at
`0.933`; the ordinary 256-query minority upper bound is `0.681`. Thus the
`>4x` right-censored result survives that simple simultaneous correction. The
`>8x` row remains descriptive.

There are two different gains:

- **Acquisition:** the first 64 queried points were all rare, a `19.78x` mean
  enrichment over their prevalence in the realized reservoir.
- **Use of support:** even after counting the initial support, routed
  specialists needed at most 88.88 mean rare examples for the `0.85` target;
  the ordinary population failed with 192.

And there are two costs the label ratio omits:

- every run inspects all 4,096 candidate feature vectors, or 64 observations
  per acquired label at the 64-query point;
- equal backward examples are not equal total FLOPs: clustering, routing, and
  diagnostic forwards remain additional work.

Thus `>4x` is the defensible population-learning number for the `0.85` target,
while `19.78x` is a separate pool-based acquisition number. If observing a
candidate costs an environment interaction, the latter is not an RL
sample-efficiency gain. The population uses support efficiently; it does not
create support.

Reproduce the figure and its raw data with:

```console
uv run ssilite-sample-efficiency \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --budgets 0 32 64 128 256 \
  --device cuda |
uv run --with 'matplotlib>=3.10' python scripts/plot_sample_efficiency.py \
  --output-prefix artifacts/sample_efficiency_variants
```

## The identifiability wall

The positive result assumes independent label flips. In a deliberately
unidentifiable control, every label in the rare environment is flipped
coherently. The observed distribution is then equally consistent with a real
alternative mechanism.

Across eight CUDA seeds:

- specialists assigned the systematically wrong rare labels `0.805 +/- 0.026`
  confidence;
- routed clean-minority accuracy collapsed to `0.093 +/- 0.018`;
- majority accuracy remained `0.953 +/- 0.007`.

Out-of-fold scoring blocks a point from teaching its own label. It cannot
distinguish a coherent causal mechanism from coherent, feature-dependent
corruption. No algorithm using only the same observed distribution can.

## Prototype boundaries

- Raw-feature environments are transductively discoverable and almost pure in
  this synthetic problem. Language-scale environment discovery is the hard
  unsolved object.
- Internal students are not statistical replicates; the data seed is the unit
  of inference.
- Mixed precision is statistically emulated. There is no custom low-precision
  kernel or wall-clock speedup claim.
- The default `ssilite` reducible-loss arm uses generator-clean labels as an
  explicit oracle control.
- The results demonstrate a mechanism and its failure boundary, not a changed
  scaling exponent.

## What did Ilya see at SSI?

This section is reconstruction, not reporting.

The public facts constrain the answer. NVIDIA says it received rare access to
SSI's guarded work, calls it a
[new research direction](https://investor.nvidia.com/news/press-release-details/2026/Ilya-Sutskevers-Safe-Superintelligence-Inc--and-NVIDIA-Announce-Long-Term-Strategic-Partnership/default.aspx),
will expand SSI's compute tenfold with Vera Rubin, and will collaborate on
future compute platforms. Reuters reports the investment as
[$5 billion](https://www.investing.com/news/stock-market-news/nvidia-to-invest-5-billion-in-ilya-sutskevers-ai-startup-source-says-4814862).
The official announcement identifies Ilya Sutskever and Daniel Levy as current
leadership.

The reported hiring pattern is narrow but not dispositive. Daniel Levy's work
includes
[large-scale CVaR and chi-squared DRO](https://arxiv.org/abs/2010.05893).
Globes reported Yair Carmon, Shahar Papini, Nitzan Tor, and Yaron Brodsky among
the early Tel Aviv hires; Carmon's public work centers on lower bounds,
minimax optimization, reliability, and the
[price of adaptivity](https://arxiv.org/abs/2402.10898). Papini's finite-field
and hardware-arithmetic background made a precision/co-design thesis plausible,
but he is now a
[former SSI engineer](https://en.globes.co.il/en/article-Former-SSI-engineer-founds-AI-startup-in-Tel-Aviv-1001534082).
Tor's visible distinction is unusually strong pure mathematics; Brodsky's is
generative image editing and diffusion plus mathematics. [Hugh
Zhang](https://hughbzhang.com/) is also publicly a former SSI technical staff
member whose work spans test-time search, planning, evaluation, and game
dynamics. Current status for several reported 2025 hires is not publicly
confirmed.

The code rules out my original strong version:

- It is probably not vanilla per-example DRO. Loss ranking cannot recover a
  partition, and the environment adversary was not the winning arm.
- It is not ordinary ensemble disagreement. Same-objective students were
  correlated and their best-of-population rescue also rescued corrupt labels.
- Precision may be an important enabler, but precision alone does not explain
  why the learning rule would change with scale.

The surviving hypothesis is more specific:

> SSI has learned how to manufacture coherent, automatically discovered
> training environments and force a population of students into genuinely
> different mechanisms. Students cross-grade out of distribution; a learned
> gate keeps the relevant specialist matched to the relevant experience and
> possibly to inference. Robust optimization, sampling, and precision are the
> machinery that make this population affordable, not the source of the
> learning signal.

That mechanism explains the otherwise odd conjunction:

- the optimization hires know how to solve and stabilize the resulting
  minimax/allocation problem;
- a population needs much more scoring and training compute than one model,
  making a tenfold scale-up immediately useful;
- its workload creates a real systems co-design question—many cheap,
  decorrelated student passes plus selective high-precision updates;
- capability and safety can share an object: explicit environments, gates,
  and cross-student failure evidence are more inspectable than an opaque
  corpus mixture.

The code also names the secret that this reconstruction does **not** possess.
On the toy problem, raw context gives away the environment. At frontier scale,
SSI would need a stable way to discover or generate environments that are:

1. different enough to break shared model bias;
2. coherent enough that a specialist can learn them;
3. causally anchored enough not to certify systematic corruption;
4. routable without hidden group labels;
5. increasingly useful, rather than increasingly redundant, as compute grows.

That could look like adversarially generated worlds, self-play populations,
learned curricula, latent routers, cross-play teachers, or a mechanism we have
not named. The roster supports structured adaptive optimization; it does not
uniquely identify an ensemble architecture.

My best answer is therefore: **Ilya likely saw a scaling curve for structured
population diversity—more compute buys new coherent learning environments,
not merely a larger ERM model.** The result worth $5 billion would be the rule
that discovers and anchors those environments. The optimizer, sampler, and
precision system are what let NVIDIA scale it.

Most likely place this is wrong: the raw-environment assumption did all the
work here, while SSI's actual result is a numerics or optimizer-stability
breakthrough. The discriminating public artifact would be a scaling result in
environment count, cross-student transfer, routing, or continual acquisition.
A pure low-precision stability paper would instead favor the competing
precision-first story.
