"""Pure-Torch helpers for experimental HAB pruning priorities.

This module deliberately has no CUDA-extension imports so the ranking policy
can be regression-tested on CPU.  A high multi-view score means "prune", while
a high empirical-Fisher proxy means "protect".
"""

import math

import torch


def _stable_order(indices, values, descending=False):
    """Order ``indices`` by ``values`` with deterministic index tie-breaking."""
    if indices.numel() <= 1:
        return indices
    indices = indices[torch.argsort(indices, stable=True)]
    order = torch.argsort(values[indices], descending=descending, stable=True)
    return indices[order]


def select_guarded_prune_indices(
        prune_budget,
        eligible,
        opacities,
        pruning_score=None,
        fisher_proxy=None,
        mode="opacity_mv_fisher_guard",
        candidate_multiplier=2.0,
        fisher_protect_quantile=0.90):
    """Select exact prune indices for the guarded experimental policies.

    Opacity remains the primary safety boundary: multi-view consistency only
    re-ranks a small low-opacity band.  The optional Fisher proxy removes the
    most sensitive primitives from that pool.  If protection would make an
    exact budget impossible, the least-sensitive protected entries are released
    deterministically.
    """
    supported = {
        "opacity_mv_band",
        "opacity_fisher_guard",
        "opacity_mv_fisher_guard",
    }
    if mode not in supported:
        raise ValueError("unsupported guarded HAB priority mode: {}".format(mode))

    eligible = eligible.detach().bool().reshape(-1)
    opacities = torch.nan_to_num(
        opacities.detach().float().reshape(-1), nan=1.0, posinf=1.0, neginf=0.0)
    if eligible.shape[0] != opacities.shape[0]:
        raise ValueError("eligible/opacity length mismatch")

    available = int(torch.sum(eligible).item())
    prune_budget = min(max(int(prune_budget), 0), available)
    diagnostics = {
        "hab_candidate_band_count": 0,
        "hab_fisher_protected": 0,
        "hab_fisher_guard_relaxed": 0,
    }
    if prune_budget == 0:
        return torch.empty((0,), dtype=torch.long, device=eligible.device), diagnostics

    working = eligible.clone()
    uses_fisher = mode in ("opacity_fisher_guard", "opacity_mv_fisher_guard")
    if uses_fisher and fisher_proxy is not None:
        fisher = torch.nan_to_num(
            fisher_proxy.detach().float().reshape(-1).to(eligible.device),
            nan=0.0, posinf=torch.finfo(torch.float32).max, neginf=0.0)
        if fisher.shape[0] == eligible.shape[0]:
            positive = torch.logical_and(eligible, fisher > 0)
            if torch.any(positive):
                quantile = min(max(float(fisher_protect_quantile), 0.0), 1.0)
                threshold = torch.quantile(fisher[positive], quantile)
                protected = torch.logical_and(positive, fisher >= threshold)
                working = torch.logical_and(eligible, ~protected)
                diagnostics["hab_fisher_protected"] = int(torch.sum(protected).item())

                if int(torch.sum(working).item()) < prune_budget:
                    # Exact count takes precedence.  Release the least-sensitive
                    # entries, leaving as many high-Fisher primitives protected
                    # as the requested budget permits.
                    candidates = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
                    candidates = _stable_order(candidates, fisher, descending=False)
                    working[:] = False
                    working[candidates[:prune_budget]] = True
                    diagnostics["hab_fisher_protected"] = available - prune_budget
                    diagnostics["hab_fisher_guard_relaxed"] = 1

    uses_multiview = mode in ("opacity_mv_band", "opacity_mv_fisher_guard")
    working_indices = torch.nonzero(working, as_tuple=False).squeeze(-1)
    opacity_order = _stable_order(working_indices, opacities, descending=False)

    if not uses_multiview:
        diagnostics["hab_candidate_band_count"] = int(working_indices.numel())
        return opacity_order[:prune_budget], diagnostics

    multiplier = max(float(candidate_multiplier), 1.0)
    band_count = min(
        int(opacity_order.numel()),
        max(prune_budget, int(math.ceil(multiplier * prune_budget))))
    band = opacity_order[:band_count]
    diagnostics["hab_candidate_band_count"] = int(band_count)

    if pruning_score is None or pruning_score.numel() == 0:
        return band[:prune_budget], diagnostics

    flat_score = pruning_score.detach().float().reshape(-1).to(eligible.device)
    valid_count = min(int(flat_score.numel()), int(eligible.numel()))
    if valid_count <= 0:
        return band[:prune_budget], diagnostics
    score = torch.full_like(opacities, -float("inf"))
    score[:valid_count] = torch.nan_to_num(
        flat_score[:valid_count], nan=-float("inf"), posinf=1.0, neginf=-float("inf"))
    finite_band = torch.isfinite(score[band])
    if not torch.any(finite_band):
        return band[:prune_budget], diagnostics
    finite_values = score[band][finite_band]
    if (torch.max(finite_values) - torch.min(finite_values)).item() <= 1e-8:
        return band[:prune_budget], diagnostics

    # ``band`` is already ordered by opacity.  Stable score sorting therefore
    # uses opacity and then primitive index as deterministic tie-breakers.
    score_order = torch.argsort(score[band], descending=True, stable=True)
    return band[score_order[:prune_budget]], diagnostics
