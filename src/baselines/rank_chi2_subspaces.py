# src/baselines/rank_chi2_subspaces.py
# -------------------------------------------------------------------
# Thin adapter around rank_two_sample_subspaces.RankGTestSubspaces so
# the runner can do:
#   from baselines.rank_chi2_subspaces import (
#       RankChi2SubspacesTest, rank_chi2_subspaces_permutation_test
#   )
#   test = RankChi2SubspacesTest(K_ref=64, L=64, k_dim=2, B=1000, alpha=0.05, ...)
#   out  = test.run(X, Y)  # class API
#   out2 = rank_chi2_subspaces_permutation_test(
#              X, Y, K_ref=64, L=64, k_dim=2, B=1000, alpha=0.05, ...
#          )
#
# Exposes the same result shape as the plain Rank χ² / G-test adapter.
# Adds:
#   • dedup: avoid duplicate random slices
#   • pca:   choose slices by PCA loadings (ignored if dims_list provided)
#   • pca_center: center data before PCA selection
#
# New:
#   • Forwards reference_mode / reference_sampler to the core RankGTestConfig.
#   • This file is only an adapter; the actual logic change must live in
#     ranktwosample.rank_two_sample_subspaces / its core subspace runner.
# -------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List, Callable, Any

import numpy as np

from ranktwosample.rank_two_sample import RankGTestConfig
from ranktwosample.rank_two_sample_subspaces import RankGTestSubspaces, SubspaceOpts

Array = np.ndarray

__all__ = [
    "RankChi2Result",
    "RankChi2SubspacesTest",
    "rank_chi2_subspaces_permutation_test",
]


@dataclass
class RankChi2Result:
    stat_obs: float
    p_perm: float
    p_mid: float
    reject: bool
    perms_used: int
    elapsed_ms: float


class RankChi2SubspacesTest:
    def __init__(
        self,
        *,
        # ---- core RankGTestConfig (mirrors your baseline adapter) ----
        alpha: float = 0.05,
        K_ref: Optional[int] = None,
        engine: str = "auto",              # 'auto' | 'searchsorted' | 'prefixsum'
        tie: str = "jitter",               # 'jitter' | 'right' | 'left'
        alpha0: float = 0.1,               # set <0 for auto
        decision: str = "midp",            # 'pvalue' | 'midp' | 'randomized'
        B: int = 1000,
        kref_switch: int = 64,
        chunk: int = 512,
        antithetic: bool = True,
        stop_ci: str = "wilson",           # 'wilson' | 'hoeffding'
        delta_ci: float = 1e-2,
        min_b_check: int = 100,
        use_fullY_for_HY: bool = True,
        stat: str = "gtest",               # 'gtest' | 'chi2'
        jitter_scale: float = 1e-12,
        gpu: str = "off",                  # 'off' | 'auto' | 'on'
        device: int = 0,
        # ---- new: theory/practice switch ----
        reference_mode: str = "shared_pool",   # 'shared_pool' | 'fresh_iid'
        reference_sampler: Optional[Callable[[int, Any, Any], Array]] = None,
        # ---- subspace slicing options ----
        L: int = 32,                       # number of slices/subspaces
        k_dim: int = 1,                    # dimensionality per slice
        dims_list: Optional[List[Sequence[int]]
                            ] = None,  # explicit dims per slice
        dedup: bool = True,                # avoid duplicate random slices
        # PCA-guided slice selection (ignored if dims_list is provided)
        pca: bool = False,
        pca_center: bool = True,           # center data before PCA selection
    ):
        # Build core config
        self.cfg = RankGTestConfig(
            alpha=alpha,
            K_ref=K_ref,
            engine=engine,
            tie=tie,
            alpha0=alpha0,
            decision=decision,
            B=B,
            kref_switch=kref_switch,
            chunk=chunk,
            antithetic=antithetic,
            stop_ci=stop_ci,
            delta_ci=delta_ci,
            min_b_check=min_b_check,
            use_fullY_for_HY=use_fullY_for_HY,
            stat=stat,
            jitter_scale=jitter_scale,
            gpu=gpu,
            device=device,
            reference_mode=reference_mode,
            reference_sampler=reference_sampler,
        )

        # Subspace options
        self.subs = SubspaceOpts(
            L=L,
            k_dim=k_dim,
            dims_list=dims_list,
            dedup=dedup,
            pca=(pca and dims_list is None),
            pca_center=pca_center,
        )

        # Runner
        self._runner = RankGTestSubspaces(self.cfg, self.subs)

    def run(self, X: Array, Y: Array, *, seed: Optional[int] = None) -> RankChi2Result:
        # Accept lists/tuples transparently
        X = np.asarray(X)
        Y = np.asarray(Y)
        r = self._runner.test(X, Y, seed=seed)
        return RankChi2Result(
            stat_obs=float(r.T_obs),
            p_perm=float(r.p_perm),
            p_mid=float(r.p_mid),
            reject=bool(r.reject),
            perms_used=int(r.perms_used),
            elapsed_ms=float(r.elapsed_ms),
        )


def rank_chi2_subspaces_permutation_test(
    X: Array,
    Y: Array,
    *,
    # ---- core RankGTestConfig params (same defaults as class) ----
    alpha: float = 0.05,
    K_ref: Optional[int] = None,
    engine: str = "auto",
    tie: str = "jitter",
    alpha0: float = 0.1,
    decision: str = "midp",
    B: int = 1000,
    kref_switch: int = 64,
    chunk: int = 512,
    antithetic: bool = True,
    stop_ci: str = "wilson",
    delta_ci: float = 1e-2,
    min_b_check: int = 100,
    use_fullY_for_HY: bool = True,
    stat: str = "gtest",
    jitter_scale: float = 1e-12,
    gpu: str = "off",
    device: int = 0,
    # ---- new: theory/practice switch ----
    reference_mode: str = "shared_pool",   # 'shared_pool' | 'fresh_iid'
    reference_sampler: Optional[Callable[[int, Any, Any], Array]] = None,
    # ---- subspace slicing options ----
    L: int = 32,
    k_dim: int = 1,
    dims_list: Optional[List[Sequence[int]]] = None,
    dedup: bool = True,
    pca: bool = False,
    pca_center: bool = True,
    seed: Optional[int] = None,
) -> RankChi2Result:
    """
    Functional wrapper so callers can do:
        from baselines.rank_chi2_subspaces import rank_chi2_subspaces_permutation_test
        out = rank_chi2_subspaces_permutation_test(
            X, Y, K_ref=64, L=64, k_dim=2, B=1000, ...
        )

    Notes
    -----
    • If dims_list is provided, it takes precedence; pca is ignored.
    • If pca=True and dims_list=None, slices are chosen by PCA loadings.
    • reference_mode:
        - 'shared_pool' : existing practical pooled-reference implementation
        - 'fresh_iid'   : theorem-aligned mode; requires reference_sampler in the core
                          subspace implementation.
    • reference_sampler:
        Callable used only when reference_mode='fresh_iid'. Expected signature:
            reference_sampler(n: int, rng, xp) -> array of shape (n, d)
    """
    test = RankChi2SubspacesTest(
        alpha=alpha,
        K_ref=K_ref,
        engine=engine,
        tie=tie,
        alpha0=alpha0,
        decision=decision,
        B=B,
        kref_switch=kref_switch,
        chunk=chunk,
        antithetic=antithetic,
        stop_ci=stop_ci,
        delta_ci=delta_ci,
        min_b_check=min_b_check,
        use_fullY_for_HY=use_fullY_for_HY,
        stat=stat,
        jitter_scale=jitter_scale,
        gpu=gpu,
        device=device,
        reference_mode=reference_mode,
        reference_sampler=reference_sampler,
        L=L,
        k_dim=k_dim,
        dims_list=dims_list,
        dedup=dedup,
        pca=pca,
        pca_center=pca_center,
    )
    return test.run(np.asarray(X), np.asarray(Y), seed=seed)
