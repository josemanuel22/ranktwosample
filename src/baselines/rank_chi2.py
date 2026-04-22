# src/baselines/rank_chi2.py
# -------------------------------------------------------------------
# Thin adapter around ranktwosample.rank_two_sample.RankGTest so callers can do:
#   from baselines.rank_chi2 import RankChi2Test, rank_chi2_permutation_test
#   test = RankChi2Test(K_ref=64, B=1000, alpha=0.05, ...)
#   out  = test.run(X, Y)              # class API
#   out2 = rank_chi2_permutation_test(X, Y, K_ref=64, B=1000, alpha=0.05, ...)  # func API
#
# New:
#   • Forwards reference_mode / reference_sampler to the core RankGTestConfig.
#   • This file is only an adapter; the actual logic change must live in
#     ranktwosample.rank_two_sample.RankGTest.
# -------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable, Any

import numpy as np
from ranktwosample.rank_two_sample import RankGTest, RankGTestConfig

Array = np.ndarray

__all__ = ["RankChi2Result", "RankChi2Test", "rank_chi2_permutation_test"]


@dataclass
class RankChi2Result:
    stat_obs: float
    p_perm: float
    p_mid: float
    reject: bool
    perms_used: int
    elapsed_ms: float


class RankChi2Test:
    def __init__(
        self,
        *,
        # mirror RankGTestConfig (defaults chosen to match your runner)
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
        # new: theory/practice switch
        reference_mode: str = "shared_pool",   # 'shared_pool' | 'fresh_iid'
        reference_sampler: Optional[Callable[[int, Any, Any], Array]] = None,
    ):
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
        self._runner = RankGTest(self.cfg)

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


def rank_chi2_permutation_test(
    X: Array,
    Y: Array,
    *,
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
    seed: Optional[int] = None,
    # new: theory/practice switch
    reference_mode: str = "shared_pool",   # 'shared_pool' | 'fresh_iid'
    reference_sampler: Optional[Callable[[int, Any, Any], Array]] = None,
) -> RankChi2Result:
    """
    Functional wrapper so callers can do:
        from baselines.rank_chi2 import rank_chi2_permutation_test
        out = rank_chi2_permutation_test(X, Y, K_ref=64, B=1000, ...)

    Notes
    -----
    reference_mode:
        - 'shared_pool' : existing practical pooled-reference implementation
        - 'fresh_iid'   : theorem-aligned mode; requires reference_sampler in the core
                          RankGTest implementation.

    reference_sampler:
        Callable used only when reference_mode='fresh_iid'. Expected signature:
            reference_sampler(n: int, rng, xp) -> array of shape (n, d)
        where:
            - n   : number of iid reference-law samples requested
            - rng : backend RNG from RankGTest
            - xp  : numpy/cupy backend module
    """
    # Reuse the class to keep behavior identical
    test = RankChi2Test(
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
    return test.run(X, Y, seed=seed)
