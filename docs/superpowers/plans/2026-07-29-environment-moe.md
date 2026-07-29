# Environment MoE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the final environment-specialist population benchmark into a
formal hard-routed PyTorch MoE and regenerate its CUDA evidence.

**Architecture:** A single `EnvironmentMoE` owns a learned router and contiguous
expert tensors. Environment pseudo-labels supervise the router, while
teacher-forced expert-major batches preserve the discovered specialization
curriculum and prevent cold-start collapse.

**Tech Stack:** Python 3.13, PyTorch 2.13, CUDA 13, pytest, Ruff, Mypy.

## Global Constraints

- Work in `/home/michael/Development/ssilite` and preserve unrelated changes.
- Use the NVIDIA GeForce RTX 4090 for reported experiments.
- One selected expert is evaluated per inference example.
- Clean labels and hidden group/corruption metadata are evaluation-only.
- Do not claim to implement Kimi K3, Stable LatentMoE, or MoonEP.
- Do not commit generated or source changes until the complete verification
  state is known.

---

### Task 1: Tensorized MoE and sparse dispatch

**Files:**
- Create: `src/ssilite/environment_moe.py`
- Test: `tests/test_environment_moe.py`

**Interfaces:**
- Produces: `TensorizedExpertBank`, `EnvironmentMoE`,
  `EnvironmentMoEConfig`, and `train_environment_moe(...)`.
- `EnvironmentMoE.all_expert_logits(x)` returns `(expert, example)`.
- `EnvironmentMoE.selected_expert_logits(x, ids)` returns `(example,)`.
- `EnvironmentMoE.forward(x)` returns hard-routed logits and route IDs.

- [ ] **Step 1: Write failing architecture and dispatch tests**

```python
bank = TensorizedExpertBank(4, 3, 2)
selected = bank.selected_logits(x, torch.tensor([0, 1, 0]))
dense = bank.all_logits(x)
torch.testing.assert_close(selected, dense[ids, torch.arange(3)])
```

Assert that backpropagating `selected.sum()` leaves the unselected expert slice
with an exactly zero gradient.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_environment_moe.py -q`

Expected: collection fails because `ssilite.environment_moe` does not exist.

- [ ] **Step 3: Implement the minimal tensorized expert bank and router**

Use parameters shaped `(E,H,D)`, `(E,H)`, `(E,H)`, and `(E,)`. Sparse dispatch
gathers only selected parameter slices; the dense method exists for diagnostics
and equivalence testing.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run: `uv run pytest tests/test_environment_moe.py -q`

Expected: dispatch, gradient, shape, and validation tests pass.

### Task 2: Paired ordinary and specialized MoE training

**Files:**
- Modify: `src/ssilite/environment_moe.py`
- Modify: `tests/test_environment_moe.py`

**Interfaces:**
- Produces: `EnvironmentMoEResult` with paired `ordinary` and `specialist`
  `MoEPredictions`.
- Each prediction object exposes dense mean, learned top-1 route, router
  probabilities, route counts, diagnostics, model seed, and exact compute.

- [ ] **Step 1: Write failing paired-training tests**

Check identical initialization/compute/router outputs, deterministic CPU
results, strict hidden-metadata exclusion, a separable-router fixture, and rare
coherent-rule rescue.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/test_environment_moe.py -q`

Expected: training API tests fail because paired training is absent.

- [ ] **Step 3: Implement one-optimizer expert-major training**

For every step, sample `(E,B)` task indices and one balanced router batch. Sum
per-expert mean BCE losses so expert gradient magnitudes match separate fits.
Use identical router batches and initialization for paired arms.

- [ ] **Step 4: Verify GREEN on CPU and CUDA**

Run: `uv run pytest tests/test_environment_moe.py -q`

Expected: all focused tests pass, including the CUDA-marked device test.

### Task 3: Convert the nested sample-efficiency benchmark

**Files:**
- Modify: `src/ssilite/sample_efficiency.py`
- Modify: `tests/test_sample_efficiency.py`
- Modify: `scripts/plot_sample_efficiency.py`

**Interfaces:**
- `run_sample_efficiency(..., moe_config=...)` returns the existing five arm
  names with `EnvironmentMoECompute`.
- Points additionally expose router and best-individual-expert diagnostics.

- [ ] **Step 1: Change tests to require formal MoE fields and budgets**
- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/test_sample_efficiency.py -q`

Expected: import or schema assertions fail against the legacy mixture.

- [ ] **Step 3: Replace mixture calls and cached indexing with MoE outputs**
- [ ] **Step 4: Update plot labels from students/populations to experts/MoE**
- [ ] **Step 5: Run both focused suites and verify GREEN**

Run: `uv run pytest tests/test_environment_moe.py tests/test_sample_efficiency.py -q`

### Task 4: CUDA evidence, documentation, and full verification

**Files:**
- Modify: `README.md`
- Regenerate: `artifacts/sample_efficiency_variants.{json,csv,png,svg}`

- [ ] **Step 1: Run a seed-0 CUDA falsification pass**

Run the five arms at budgets `0 64 256`; inspect routing, minority, and majority
results before the full run.

- [ ] **Step 2: Run all 16 paired seeds on CUDA**

```console
uv run ssilite-sample-efficiency \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \
  --budgets 0 32 64 128 256 --device cuda
```

- [ ] **Step 3: Regenerate the plot and summary artifacts**
- [ ] **Step 4: Update the README using only measured MoE results**
- [ ] **Step 5: Run final verification**

```console
uv run pytest -q
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy src/ssilite
uv build
```

Expected: all tests and static/build checks pass, with strict JSON artifacts and
reported execution device `cuda`.

