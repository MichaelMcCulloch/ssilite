# Environment MoE Design

## Goal

Replace the final independent-student population experiment with one formal
top-1 mixture-of-experts model while preserving the discovered training rule:
label-free feature environments define coherent expert curricula, and the
matching learned gate selects an expert at inference.

## Architecture

`EnvironmentMoE` is one `torch.nn.Module` containing:

- a learned linear router over standardized input features;
- a tensorized expert bank whose parameter tensors have a leading expert axis;
- hard top-1 sparse dispatch that evaluates only the selected expert;
- an all-expert diagnostic path used only by dense-mean causal controls.

There is exactly one routed expert per discovered environment. The former
balanced fourth student is omitted because it was never selected by the gate.
This is a classical sparse MoE, not a miniature implementation of Kimi K3 or
Stable LatentMoE.

## Training

Feature-only clustering supplies pseudo-labels `e_i`. A router minibatch,
balanced over those discovered environments, minimizes cross entropy

```math
\mathcal L_{\mathrm{router}}(\phi)
=
\operatorname{CE}(r_\phi(x_i),e_i).
```

Task training uses teacher-forced dispatch so an untrained router cannot starve
an expert. Each step gives every expert one fixed-size minibatch. In the
ordinary control, all experts sample the empirical distribution. In the
specialist treatment, expert `e` places `focus_mass` on environment `e`.

```math
\mathcal L_{\mathrm{task}}(\theta)
=
\sum_{e=1}^{E}
\frac{1}{B}\sum_{j=1}^{B}
\ell(f_{\theta_e}(x_{e,j}),y_{e,j}).
```

One optimizer and one backward pass update the complete module. Ordinary and
specialist models start from identical parameters, use paired random streams,
and train their routers on identical examples. Therefore their learned routes
are identical; only expert curricula differ.

## Causal controls

The nested label-budget experiment retains five arms, now all derived from
formal MoE modules:

1. ordinary experts with a dense probability mean;
2. ordinary experts with the learned top-1 router;
3. specialized experts with a dense probability mean;
4. specialized experts and router trained on size-preserving permuted
   environments;
5. specialized experts with the matching learned environment router.

The primary comparison is arm 5 against arm 1. Hidden minority and clean-label
metadata remain evaluation-only.

## Compute accounting

Every arm reports physical optimizer steps, logical expert updates, task
backward examples, router-training examples, diagnostic all-expert forwards,
and sparse selected-expert forwards. At default settings with three
environments, the task budget is `3 * 80 * 64 = 15,360` expert examples and the
router budget is `80 * 64 = 5,120` examples.

The implementation records route counts and tests that sparse dispatch drops no
examples and produces the same selected values and gradients as indexing the
dense reference. MoonEP is not used on the single RTX 4090; its relevance is
only that contiguous expert weights and explicit route plans have an existing
multi-GPU systems path.

## Verification and success criteria

- API cannot accept clean labels, hidden group membership, or corruption flags.
- Unselected experts receive zero task gradient.
- Paired ordinary/specialist routers are identical.
- All five arms use equal task and router budgets.
- The structured routed MoE retains the minority-accuracy improvement and
  majority guardrail across 16 CUDA seeds.
- Tests, Ruff, Mypy, source build, and wheel build pass.

