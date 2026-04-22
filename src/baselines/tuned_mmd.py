# src/baselines/tuned_mmd.py
# -------------------------------------------------------------------
# Tuned MMD (two-sample) — split-sample, permutation-calibrated
#   • Tune kernel hyperparams on a held-out "tuning" split via perms
#   • Test on a disjoint "test" split via fresh permutations
#   • Optional multi-split with Bonferroni-combined p-value
#   • Compatible with run_power_bench: class TunedMMDTest has .run()
# -------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np

# Import Kernel MMD primitives from the sibling module
# (absolute import to match your "from baselines.tuned_mmd import TunedMMDTest")
from baselines.kernel_mmd import (
    kernel_mmd_permutation_test,
    kernel_mmd2,
    KernelMMDResult,
)

Array = np.ndarray

__all__ = [
    "TunedSelectionLog",
    "TunedMMDOneSplit",
    "TunedMMDResult",
    "tuned_mmd_permutation_test",
    "TunedMMDTest",
]

# ----------------------- helpers -----------------------


def _split_two_sample(
    X: Array, Y: Array, prop: float, rng: np.random.Generator
) -> Tuple[Array, Array, Array, Array]:
    """Split X/Y into (tune, test) with proportions prop and 1-prop."""
    N, K = X.shape[0], Y.shape[0]
    n_tune = max(2, int(round(prop * N)))
    k_tune = max(2, int(round(prop * K)))
    idxX = rng.permutation(N)
    idxY = rng.permutation(K)
    X_tune, X_test = X[idxX[:n_tune]], X[idxX[n_tune:]]
    Y_tune, Y_test = Y[idxY[:k_tune]], Y[idxY[k_tune:]]
    # ensure at least 2 points in each test split
    if X_test.shape[0] < 2 or Y_test.shape[0] < 2:
        n_tune = min(N - 2, max(2, n_tune))
        k_tune = min(K - 2, max(2, k_tune))
        X_tune, X_test = X[idxX[:n_tune]], X[idxX[n_tune:]]
        Y_tune, Y_test = Y[idxY[:k_tune]], Y[idxY[k_tune:]]
    return X_tune, Y_tune, X_test, Y_test


def _median_bandwidths(
    Z: Array, *, subsample: int = 1000, scales: List[float]
) -> List[float]:
    """Median heuristic (pooled sample) times user scales."""
    m = min(Z.shape[0], subsample)
    idx = np.random.default_rng(0).choice(Z.shape[0], size=m, replace=False)
    Zs = Z[idx]
    X = Zs
    x2 = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.maximum(x2 + x2.T - 2.0 * (X @ X.T), 0.0)
    tri = D2[np.triu_indices(m, k=1)]
    med = float(np.sqrt(np.median(tri))) if tri.size else 1.0
    return [float(med * s) for s in scales]


def _parse_grid_spec(grid: str) -> List[float]:
    """
    Parse strings like 'median_x{0.5,1,2,4}' or 'median_x{1/2,1,2,4}'.
    Returns list of scale multipliers.
    """
    grid = (grid or "").strip()
    if grid.startswith("median_x"):
        # extract inside braces
        lb = grid.find("{")
        rb = grid.rfind("}")
        if lb >= 0 and rb > lb:
            items = grid[lb + 1: rb].split(",")
            scales = []
            for it in items:
                it = it.strip()
                if "/" in it:
                    num, den = it.split("/", 1)
                    try:
                        scales.append(float(num) / float(den))
                    except Exception:
                        pass
                else:
                    try:
                        scales.append(float(it))
                    except Exception:
                        pass
            # fallback sane default if parse fails
            return scales or [0.5, 1.0, 2.0, 4.0]
        return [0.5, 1.0, 2.0, 4.0]
    # default grid
    return [0.5, 1.0, 2.0, 4.0]


def make_rbf_candidates_from_grid(
    X: Array, Y: Array, *, grid: str, estimator: str = "unbiased"
) -> List[Dict]:
    """Build candidate kwargs dicts for kernel_mmd_permutation_test from a grid spec."""
    Z = np.vstack([X, Y])
    scales = _parse_grid_spec(grid)
    sigmas = _median_bandwidths(Z, subsample=1000, scales=scales)
    cands: List[Dict] = []
    for s in sigmas:
        cands.append(
            dict(
                kernel="rbf",
                estimator=estimator,
                rbf_sigmas=[float(s)],
                rbf_weights=None,
                rbf_use_median=False,
            )
        )
    # small safety: always include a multi-band option too
    if len(sigmas) >= 3:
        cands.append(
            dict(
                kernel="rbf",
                estimator=estimator,
                rbf_sigmas=list(map(float, sigmas)),
                rbf_weights=None,
                rbf_use_median=False,
            )
        )
    return cands


# ----------------------- dataclasses -----------------------


@dataclass
class TunedSelectionLog:
    p_perm: float
    stat: float
    kwargs: Dict


@dataclass
class TunedMMDOneSplit:
    best_index: int
    tune_logs: List[TunedSelectionLog]
    test_result: KernelMMDResult


@dataclass
class TunedMMDResult:
    splits: List[TunedMMDOneSplit]
    p_values: List[float]
    p_combined: float
    reject: bool
    alpha: float
    note: str
    # Convenience mirrors for single-split usage in runners:
    stat_obs: float
    p_perm: float


# ----------------------- main tuned API -----------------------


def tuned_mmd_permutation_test(
    X: Array,
    Y: Array,
    *,
    # candidate kernels
    candidates: Optional[List[Dict]] = None,
    # currently only 'gaussian' (RBF) supported
    family: str = "gaussian",
    grid: str = "median_x{0.5,1,2,4}",     # parsed if candidates is None
    estimator: str = "unbiased",
    # splitting
    split_prop: float = 0.5,
    n_splits: int = 1,
    # permutations
    B_tune: int = 300,
    B_test: int = 2000,
    alpha: float = 0.05,
    decision: str = "pvalue",              # used in the final test
    # early stopping
    tune_early_stop: str = "none",
    test_early_stop: str = "wilson",
    tune_chunk: int = 256,
    test_chunk: int = 256,
    antithetic: bool = True,
    delta_ci: float = 1e-2,
    min_b_check_tune: int = 100,
    min_b_check_test: int = 200,
    seed: int = 2028,
) -> TunedMMDResult:
    """
    Split-sample tuned MMD: select a kernel on a tuning split by smallest perm p,
    then test on an independent test split using fresh permutations.
    If n_splits>1, Bonferroni-combine the per-split p-values (valid).
    """
    assert 0.0 < split_prop < 1.0, "split_prop must be in (0,1)."
    rng_master = np.random.default_rng(seed)

    splits_out: List[TunedMMDOneSplit] = []
    pvals: List[float] = []

    for _ in range(n_splits):
        rng = np.random.default_rng(rng_master.integers(1, 2**31 - 1))
        Xt, Yt, Xh, Yh = _split_two_sample(X, Y, split_prop, rng)

        # build candidates if needed
        if candidates is None:
            if family.lower() in ("gaussian", "rbf"):
                cand_list = make_rbf_candidates_from_grid(
                    Xt, Yt, grid=grid, estimator=estimator
                )
            else:
                raise NotImplementedError(
                    "Only RBF ('gaussian') family supported.")
        else:
            cand_list = list(candidates)

        # 1) tuning: pick kernel with smallest permutation p-value
        tune_logs: List[TunedSelectionLog] = []
        best_idx = 0
        best_key = (1.1, -np.inf)  # (p_perm asc, stat desc)
        for j, kw in enumerate(cand_list):
            out = kernel_mmd_permutation_test(
                Xt,
                Yt,
                B=B_tune,
                alpha=0.5,                 # alpha irrelevant; we read the p-value
                decision="pvalue",
                early_stop=tune_early_stop,
                delta_ci=delta_ci,
                min_b_check=min_b_check_tune,
                chunk=tune_chunk,
                antithetic=antithetic,
                **kw,
            )
            tune_logs.append(TunedSelectionLog(
                p_perm=out.p_perm, stat=out.stat_obs, kwargs=kw))
            key = (out.p_perm, out.stat_obs)
            if (key[0] < best_key[0]) or (key[0] == best_key[0] and key[1] > best_key[1]):
                best_idx = j
                best_key = key

        # 2) testing: run full permutation test on held-out split
        best_kwargs = cand_list[best_idx]
        test_res = kernel_mmd_permutation_test(
            Xh,
            Yh,
            B=B_test,
            alpha=alpha,
            decision=decision,
            early_stop=test_early_stop,
            delta_ci=delta_ci,
            min_b_check=min_b_check_test,
            chunk=test_chunk,
            antithetic=antithetic,
            **best_kwargs,
        )

        splits_out.append(
            TunedMMDOneSplit(best_index=best_idx,
                             tune_logs=tune_logs, test_result=test_res)
        )
        pvals.append(test_res.p_perm)

    # Bonferroni (valid, conservative)
    pvals_arr = np.asarray(pvals, float)
    p_comb = float(np.minimum(1.0, n_splits * np.min(pvals_arr)))
    reject = p_comb <= alpha
    note = f"Bonferroni across {n_splits} split(s)."

    # convenience: expose stat_obs/p_perm when single-split (runner expects these)
    if n_splits == 1:
        stat_obs = float(splits_out[0].test_result.stat_obs)
        p_perm = float(splits_out[0].test_result.p_perm)
    else:
        stat_obs = float("nan")
        p_perm = float(p_comb)

    return TunedMMDResult(
        splits=splits_out,
        p_values=list(map(float, pvals_arr)),
        p_combined=p_comb,
        reject=reject,
        alpha=alpha,
        note=note,
        stat_obs=stat_obs,
        p_perm=p_perm,
    )


# ----------------------- OO wrapper -----------------------


class TunedMMDTest:
    """
    Uniform class API — your runner can do:

        from baselines.tuned_mmd import TunedMMDTest
        test = TunedMMDTest(grid="median_x{0.5,1,2,4}", B_test=1000, decision="pvalue")
        out = test.run(X, Y)
    """

    def __init__(
        self,
        *,
        # candidate spec
        family: str = "gaussian",
        grid: str = "median_x{0.5,1,2,4}",
        estimator: str = "unbiased",
        # if provided, overrides family/grid
        candidates: Optional[List[Dict]] = None,
        # splitting
        split_prop: float = 0.5,
        n_splits: int = 1,
        # permutations
        B_tune: int = 300,
        B_test: int = 2000,
        alpha: float = 0.05,
        decision: str = "pvalue",           # <- accepted to match your runner
        # early stopping
        tune_early_stop: str = "none",
        test_early_stop: str = "wilson",
        tune_chunk: int = 256,
        test_chunk: int = 256,
        antithetic: bool = True,
        delta_ci: float = 1e-2,
        min_b_check_tune: int = 100,
        min_b_check_test: int = 200,
        seed: int = 2028,
    ) -> None:
        self.family = family
        self.grid = grid
        self.estimator = estimator
        self.candidates = candidates

        self.split_prop = float(split_prop)
        self.n_splits = int(n_splits)

        self.B_tune = int(B_tune)
        self.B_test = int(B_test)
        self.alpha = float(alpha)
        self.decision = str(decision)

        self.tune_early_stop = str(tune_early_stop)
        self.test_early_stop = str(test_early_stop)
        self.tune_chunk = int(tune_chunk)
        self.test_chunk = int(test_chunk)
        self.antithetic = bool(antithetic)
        self.delta_ci = float(delta_ci)
        self.min_b_check_tune = int(min_b_check_tune)
        self.min_b_check_test = int(min_b_check_test)

        self.seed = int(seed)

    def run(self, X: Array, Y: Array) -> TunedMMDResult:
        return tuned_mmd_permutation_test(
            X,
            Y,
            candidates=self.candidates,
            family=self.family,
            grid=self.grid,
            estimator=self.estimator,
            split_prop=self.split_prop,
            n_splits=self.n_splits,
            B_tune=self.B_tune,
            B_test=self.B_test,
            alpha=self.alpha,
            decision=self.decision,
            tune_early_stop=self.tune_early_stop,
            test_early_stop=self.test_early_stop,
            tune_chunk=self.tune_chunk,
            test_chunk=self.test_chunk,
            antithetic=self.antithetic,
            delta_ci=self.delta_ci,
            min_b_check_tune=self.min_b_check_tune,
            min_b_check_test=self.min_b_check_test,
            seed=self.seed,
        )
