# ssilite

A PyTorch prototype for one narrow question:

> Can a learner jointly decide what failures define its objective, which
> examples efficiently estimate that objective's gradient, and where numerical
> precision is worth its cost?

The repository also encodes the main correction from the preceding experiment:
reweighting a fixed support cannot repair missing coverage. Acquisition is a
separate operation.

## The four variables

For losses \(\ell_i(\theta)\), the prototype keeps four decisions visibly
separate:

\[
a:\text{ acquire new support},\qquad
q:\text{ robust objective},\qquad
p:\text{ gradient proposal},\qquad
b:\text{ precision}.
\]

Within a fixed support, reducible scores are tilted with a capped,
entropy-regularized CVaR adversary:

\[
q_i=\min\left\{\frac{1}{\alpha n},
c\exp\left(\frac{\ell_i-\ell_i^{\rm ref}}{\tau}\right)\right\}.
\]

The variance-aware proposal is

\[
p_i^{\rm raw}\propto
q_i\sqrt{\lVert g_i\rVert^2+\sigma_i^2(b_i)},
\]

followed by defensive mixtures with \(q\) and the uniform distribution. Given
IID sampling with replacement, the estimator is

\[
\widehat G=
\frac1B\sum_{j=1}^B
\frac{q_{I_j}}{p_{I_j}}Q_{b_{I_j}}(g_{I_j}).
\]

`Q` uses stochastic rounding, so the estimator remains conditionally unbiased.
The discrete precision allocator greedily buys the largest reduction in

\[
\sum_i \frac{q_i^2}{p_i}\sigma_i^2(b_i)
\]

per unit of expected precision cost.

Acquisition does something none of those variables can do: it adds examples.
The included policy clusters the union of current support and an unlabeled
reservoir in raw input space, then acquires representatives from under-covered
clusters. It never receives group or corruption labels.

## Run it

```console
uv sync
uv run ssilite
```

The default experiment creates:

- a majority mechanism;
- a rare, orthogonal mechanism identifiable from context coordinates;
- corrupted training labels;
- an unlabeled reservoir;
- a clean test distribution.

It compares ERM and joint \(q/p/b\) control on both the original and expanded
support. Output is JSON with overall, majority, and minority accuracy plus
robust mass and precision diagnostics.

A fresh seed-0 run of the checked-in defaults acquired 160 minority examples
out of 256 without labels. Minority accuracy for joint control moved from
`0.607` on fixed support to `0.864` after acquisition. Treat that as a
mechanism check, not evidence about scale.

Useful shorter run:

```console
uv run ssilite --steps 30 --batch-size 16
```

## Verify

```console
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest -q
```

The tests characterize:

- exact empirical CVaR and capped entropic weights;
- the variance-optimal importance proposal;
- importance-estimator unbiasedness by enumeration;
- defensive importance-ratio and exploration bounds;
- discrete precision-budget feasibility;
- stochastic-quantizer unbiasedness;
- exact 32-bit gradient agreement;
- label-free acquisition of the rare input cluster;
- reducible loss as an oracle-controlled noise filter.

## Prototype boundaries

This does **not** claim a hardware speedup. Per-example precision is emulated
statistically because ordinary PyTorch kernels do not execute different
examples in one batch at different formats.

The synthetic experiment uses clean generator labels to construct
\(\ell^{\rm ref}\). That is an oracle control used to isolate the coverage
question. It does not solve the demonstrated problem of learning an
irreducibility estimator from the same biased distribution.

The acquisition policy assumes useful geometry is present in raw observations.
If the rare mechanism is not separated before learning, clustering cannot
invent that structure either. Finally, the synthetic setup was chosen to make
the mechanism observable; it is not a scaling result.

## Layout

- `acquisition.py`: label-free support expansion.
- `risk.py`: hard CVaR and smooth capped robust weights.
- `allocation.py`: variance-aware \(p\) and budgeted \(b\).
- `quantization.py`: deterministic score and unbiased gradient quantization.
- `estimator.py`: reference and vectorized per-example gradient estimators.
- `controller.py`: damped joint \(q/p/b\) state.
- `experiment.py`: reproducible comparison on the support-limited problem.

## Answer to the experiment

**Qualified yes at the mechanism level; not yet at the causal-efficiency
level.** One learner can maintain a robust objective \(q\), sample from a
different variance-aware proposal \(p\), correct the resulting importance
weights, and allocate a precision \(b\) under a fixed expected-cost budget
without destabilizing training.

A fresh seed-0 run of the checked-in defaults produced:

| Arm | Overall | Majority | Minority | Noise mass | Minority mass | Precision cost | Quantization MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ERM, fixed support | 0.921 | 0.941 | 0.508 | — | — | — | — |
| ERM, acquired support | 0.927 | 0.944 | 0.581 | — | — | — | — |
| Joint \(q/p/b\), fixed support | 0.946 | 0.963 | 0.607 | 0.028 | 0.176 | 2.000 | 0.0061 |
| Joint \(q/p/b\), acquired support | 0.953 | 0.957 | 0.864 | 0.021 | 0.375 | 2.000 | 0.0032 |

On the original support, joint control improved overall accuracy by 2.5 points
and minority accuracy by 9.9 points over ERM. It assigned 17.6% of objective
mass to a group comprising 6.25% of the support, while assigning 2.8% to
corrupted labels comprising 3.9% of the support. It did this without receiving
group or corruption indicators.

After acquisition, minority examples comprised 25% of the expanded support.
The controller assigned them 37.5% of objective mass and reduced the mass on
corrupted labels from their 4.3% empirical share to 2.1%. Minority accuracy
reached `0.864`.

The four decisions did not contribute equally to what was demonstrated:

- **Objective \(q\):** the run shows useful capped robust reweighting when
  supplied with reducible-loss scores. The oracle clean-label reference used
  to construct those scores means the experiment does not show that the
  learner can discover the right objective from the biased distribution
  alone.
- **Sampling \(p\):** the estimator remains unbiased after sampling away from
  \(q\), and the proposal minimizes the modeled second moment before defensive
  mixing. The end-to-end run has no uniform-\(p\) ablation, so it does not yet
  measure the sampling-efficiency gain.
- **Precision \(b\):** the allocator respected a mean cost budget of `2.0`
  across 4-, 8-, and 16-bit choices while stochastic rounding preserved
  unbiasedness. There is no fixed-precision control or custom kernel, so this
  is not evidence of a numerical or wall-clock speedup.
- **Acquisition \(a\):** fixed-support control could only move minority
  accuracy to `0.607`. Selecting 256 examples from a 4,096-example unlabeled
  reservoir found 160 minority examples and moved it to `0.864`. A weighting
  policy cannot replace missing support.

The acquisition accounting was:

| Cost on seed 0 | Prototype |
| --- | ---: |
| Unlabeled candidates inspected | 4,096 |
| New labels acquired | 256 |
| Rare examples acquired | 160 |
| Rare examples expected from random acquisition | 12.1 |
| Rare-example enrichment | 13.3× |
| Gradient-bearing samples | 3,840 |
| Additional scoring forwards | 92,160 |

The 13.3× enrichment is a real label-efficiency result under a
label-expensive, observation-cheap accounting. It is not yet an interaction-
or compute-efficiency result: the run used 24 scoring forwards per
backward-trained example, selected from an existing reservoir, had no delayed
credit assignment, and received supervised labels after acquisition. Its
closest comparator is pool-based active learning or a contextual bandit, not
full RL.

The experiment therefore answers the opening question narrowly: the three
within-support controls can coexist coherently and the resulting learner
performs well, but the present comparison does not isolate whether adaptive
\(p\) or \(b\) outperforms simpler alternatives. The strongest observed result
is that joint within-support control helps, but support acquisition is
necessary when coverage is the binding constraint.
