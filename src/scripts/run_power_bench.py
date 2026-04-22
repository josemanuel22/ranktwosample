#!/usr/bin/env python3
# -------------------------------------------------------------------
# run_power_bench.py  (multiprocessing-safe)
#
# Benchmark rejection behavior across dataset families
# (ranktwosample/power_datasets.py) for:
#   • Rank χ² / G-test        -> baselines.rank_chi2.RankChi2Test / rank_chi2_permutation_test
#   • Rank χ² (Subspaces)     -> baselines.rank_chi2_subspaces.RankChi2SubspacesTest / rank_chi2_subspaces_permutation_test
#   • Sliced OT (SW)          -> baselines.sliced_ot.SlicedOTTest / sliced_ot_permutation_test
#   • Hotelling’s T²          -> baselines.hotelling_t2.HotellingT2 / hotelling_t2_permutation_test
#   • Kernel MMD              -> baselines.kernel_mmd.KernelMMDTest / kernel_mmd_permutation_test
#   • Tuned MMD               -> baselines.tuned_mmd.TunedMMDTest / tuned_mmd_permutation_test
#   • C2ST                    -> baselines.c2st.C2ST / c2st_permutation_test
#
# Notes
#   • No module objects are passed to workers (avoids pickle errors).
#   • Each baseline is optional; if import fails, it is SKIPped.
#   • Parallelized over repetitions via ProcessPoolExecutor.
#   • Outputs a CSV with rejection rates and SE for each
#     (family, method, d, N, M, L_subspaces, ...).
#
# Features
#   • Sweep a list of sample sizes via --Ns (space-separated).
#   • Sweep a list of reference-pool sizes via --Ms (space-separated).
#   • Pairwise sweep: (N1,M1), (N2,M2), ...
#   • Run only a subset of methods via --methods.
#   • Control K_ref for RankChi2 / RankChi2_Subspaces via --K_ref.
#   • Control RankChi2 reference construction via --rank_reference_mode:
#       - shared_pool : old pooled-reference implementation
#       - fresh_iid   : theorem-aligned fresh-reference batches sampled from
#                       the simulation reference law through generate_pair(...)
#   • Per-method wall-clock timeout via --method_timeout_s
#   • Sweep RankChi2_Subspaces over multiple L values via --Ls_subspaces
#   • Choose experiment type via --experiment:
#       - power       : estimate rejection rate under H1
#       - calibration : estimate empirical size under H0
# -------------------------------------------------------------------
from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# -------- datasets (required)
try:
    from ranktwosample.power_datasets import generate_pair
except Exception as e:
    print(
        "[FATAL] Could not import ranktwosample.power_datasets.generate_pair. "
        "Ensure PYTHONPATH includes 'src/'."
    )
    raise


# -------- optional baselines (classes preferred; functions as fallback)

# Kernel MMD
try:
    from baselines.kernel_mmd import KernelMMDTest, kernel_mmd_permutation_test
except Exception:
    KernelMMDTest = None
    kernel_mmd_permutation_test = None

# Tuned MMD
try:
    from baselines.tuned_mmd import TunedMMDTest, tuned_mmd_permutation_test
except Exception:
    TunedMMDTest = None
    tuned_mmd_permutation_test = None

# C2ST
try:
    from baselines.c2st import C2ST, c2st_permutation_test
except Exception:
    C2ST = None
    c2st_permutation_test = None

# Sliced OT
try:
    from baselines.sliced_ot import SlicedOTTest, sliced_ot_permutation_test
except Exception:
    SlicedOTTest = None
    sliced_ot_permutation_test = None

# Rank χ² / G-test
try:
    from baselines.rank_chi2 import RankChi2Test, rank_chi2_permutation_test
except Exception:
    RankChi2Test = None
    rank_chi2_permutation_test = None

# Try alternative module names for the rank function fallback
for _cand in ("rank_chi2_api", "ranktwosample.rank_chi2_api"):
    if callable(rank_chi2_permutation_test):
        break
    try:
        _m = __import__(_cand, fromlist=["rank_chi2_permutation_test"])
        _f = getattr(_m, "rank_chi2_permutation_test", None)
        if callable(_f):
            rank_chi2_permutation_test = _f
            break
    except Exception:
        pass

# Rank χ² (Subspaces)
try:
    from baselines.rank_chi2_subspaces import (
        RankChi2SubspacesTest,
        rank_chi2_subspaces_permutation_test,
    )
except Exception as e:
    print("[WARN] RankChi2_Subspaces import failed:", repr(e))
    RankChi2SubspacesTest = None
    rank_chi2_subspaces_permutation_test = None

# Hotelling T²
try:
    from baselines.hotelling_t2 import HotellingT2, hotelling_t2_permutation_test
except Exception:
    HotellingT2 = None
    hotelling_t2_permutation_test = None


# -------- helpers
def se_binom(p: float, R: int) -> float:
    return float(math.sqrt(max(p * (1 - p), 1e-12) / max(R, 1)))


def _clip_kref(kref: Optional[int], y_len: int, default: int = 4) -> int:
    """Ensure K_ref is valid: 1 <= K_ref <= len(Y)."""
    if y_len <= 0:
        return 1
    if kref is None:
        return min(y_len, default)
    k = int(kref)
    k = max(1, k)
    k = min(k, y_len)
    return k


def _pairwise_nm_lists(Ns: List[int], Ms: List[int]) -> List[Tuple[int, int]]:
    """
    Pairwise sweep: (N1,M1), (N2,M2), ...
    Requires equal lengths.
    """
    Ns = [int(n) for n in Ns]
    Ms = [int(m) for m in Ms]
    if len(Ns) != len(Ms):
        raise SystemExit(
            f"[ERROR] --Ns and --Ms must have the same length for pairwise sweeping. "
            f"Got len(Ns)={len(Ns)} and len(Ms)={len(Ms)}.\n"
            f"Example: --Ns 1000 5000 10000 --Ms 1000 5000 10000"
        )
    return list(zip(Ns, Ms))


class MethodTimeoutError(TimeoutError):
    """Raised when a single method exceeds its wall-clock limit."""
    pass


@contextmanager
def time_limit(seconds: Optional[float]):
    """
    Unix-only wall-clock timeout for the current process.
    Disabled when seconds is None or <= 0.
    """
    if seconds is None or float(seconds) <= 0:
        yield
        return

    if os.name != "posix":
        yield
        return

    def _handler(signum, frame):
        raise MethodTimeoutError(f"Method exceeded time limit ({seconds}s)")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def run_with_timeout(
    runner: Callable,
    X,
    Y,
    alpha: float,
    B: int,
    method_cfg: Dict[str, Any],
    timeout_s: Optional[float],
):
    """
    Wrap a method runner with a wall-clock timeout.
    Returns:
      (reject, stat, pval, ms, status)
    """
    t0 = time.perf_counter()
    try:
        with time_limit(timeout_s):
            return runner(X, Y, alpha, B, method_cfg)
    except MethodTimeoutError:
        ms = 1e3 * (time.perf_counter() - t0)
        return False, float("nan"), float("nan"), ms, "TIMEOUT"
    except Exception as e:
        ms = 1e3 * (time.perf_counter() - t0)
        return False, float("nan"), float("nan"), ms, f"ERROR:{type(e).__name__}:{e}"


@dataclass(frozen=True)
class BenchSpec:
    family: str
    d: int
    N: int
    M: int
    params: Dict[str, Any]
    L_subspaces: int = 64


@dataclass(frozen=True)
class SimulationReferenceSampler:
    """
    Picklable callable that samples iid points from the simulation reference law
    by reusing generate_pair(..., N=0, K=n) and returning only Y.
    """
    family: str
    d: int
    params: Dict[str, Any]

    def __call__(self, n: int, rng, xp):
        _, Yref = generate_pair(
            self.family,
            self.d,
            0,
            int(n),
            params=self.params,
            rng=rng,
        )
        return xp.asarray(Yref, dtype=xp.float64)


def generate_null_pair(spec: BenchSpec, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate X,Y under H0 by sampling both from the benchmark reference law.
    """
    ref_sampler = SimulationReferenceSampler(
        family=spec.family,
        d=spec.d,
        params=spec.params,
    )
    Z = ref_sampler(spec.N + spec.M, rng, np)
    X = np.asarray(Z[:spec.N], dtype=np.float64).copy()
    Y = np.asarray(Z[spec.N:spec.N + spec.M], dtype=np.float64).copy()
    return X, Y


# -------- per-method runners (no module objects in arguments)
def run_rank(X, Y, alpha: float, B: int, method_cfg: Dict[str, Any]):
    t0 = time.perf_counter()

    K_ref = _clip_kref(method_cfg.get("K_ref", None), len(Y), default=4)

    if RankChi2Test is not None:
        test = RankChi2Test(
            K_ref=K_ref,
            alpha=alpha,
            B=B,
            engine=method_cfg.get("engine", "auto"),
            tie=method_cfg.get("tie", "jitter"),
            alpha0=method_cfg.get("alpha0", 0.1),
            decision=method_cfg.get("decision", "midp"),
            kref_switch=method_cfg.get("kref_switch", 64),
            chunk=method_cfg.get("chunk", 256),
            antithetic=method_cfg.get("antithetic", True),
            stop_ci=method_cfg.get("stop_ci", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 100),
            use_fullY_for_HY=method_cfg.get("fullY", True),
            stat=method_cfg.get("stat", "gtest"),
            gpu="off",
            reference_mode=method_cfg.get("reference_mode", "shared_pool"),
            reference_sampler=method_cfg.get("reference_sampler", None),
        )
        out = test.run(X, Y, seed=method_cfg.get("seed", None))
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    if callable(rank_chi2_permutation_test):
        out = rank_chi2_permutation_test(
            X, Y,
            alpha=float(alpha), B=int(B),
            K_ref=K_ref,
            stat=method_cfg.get("stat", "gtest"),
            tie=method_cfg.get("tie", "jitter"),
            decision=method_cfg.get("decision", "randomized"),
            engine=method_cfg.get("engine", "auto"),
            kref_switch=method_cfg.get("kref_switch", 64),
            use_fullY_for_HY=method_cfg.get("fullY", True),
            stop_ci=method_cfg.get("stop_ci", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 100),
            chunk=method_cfg.get("chunk", 256),
            antithetic=method_cfg.get("antithetic", True),
            seed=method_cfg.get("seed", None),
            reference_mode=method_cfg.get("reference_mode", "shared_pool"),
            reference_sampler=method_cfg.get("reference_sampler", None),
        )
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    ms = 1e3 * (time.perf_counter() - t0)
    return False, float("nan"), float("nan"), ms, "SKIP"


def run_rank_subspaces(X, Y, alpha: float, B: int, method_cfg: Dict[str, Any]):
    """Rank χ² / G-test with subspace slicing."""
    t0 = time.perf_counter()

    K_ref = _clip_kref(method_cfg.get("K_ref", None), len(Y), default=4)

    if RankChi2SubspacesTest is not None:
        test = RankChi2SubspacesTest(
            K_ref=K_ref,
            alpha=alpha,
            B=B,
            engine=method_cfg.get("engine", "auto"),
            tie=method_cfg.get("tie", "jitter"),
            alpha0=method_cfg.get("alpha0", 0.1),
            decision=method_cfg.get("decision", "midp"),
            kref_switch=method_cfg.get("kref_switch", 64),
            chunk=method_cfg.get("chunk", 256),
            antithetic=method_cfg.get("antithetic", True),
            stop_ci=method_cfg.get("stop_ci", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 100),
            use_fullY_for_HY=method_cfg.get("fullY", True),
            stat=method_cfg.get("stat", "gtest"),
            gpu="off",
            device=0,
            L=method_cfg.get("L", 64),
            k_dim=method_cfg.get("k_dim", 1),
            dims_list=method_cfg.get("dims_list", None),
            dedup=method_cfg.get("dedup", True),
            pca=method_cfg.get("pca", False),
            pca_center=method_cfg.get("pca_center", True),
        )
        out = test.run(X, Y)
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    if callable(rank_chi2_subspaces_permutation_test):
        out = rank_chi2_subspaces_permutation_test(
            X, Y,
            alpha=float(alpha), B=int(B),
            K_ref=K_ref,
            engine=method_cfg.get("engine", "auto"),
            tie=method_cfg.get("tie", "jitter"),
            alpha0=method_cfg.get("alpha0", 0.1),
            decision=method_cfg.get("decision", "midp"),
            kref_switch=method_cfg.get("kref_switch", 64),
            chunk=method_cfg.get("chunk", 256),
            antithetic=method_cfg.get("antithetic", True),
            stop_ci=method_cfg.get("stop_ci", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 100),
            use_fullY_for_HY=method_cfg.get("fullY", True),
            stat=method_cfg.get("stat", "gtest"),
            jitter_scale=method_cfg.get("jitter_scale", 1e-12),
            gpu="off",
            device=0,
            L=method_cfg.get("L", 64),
            k_dim=method_cfg.get("k_dim", 1),
            dims_list=method_cfg.get("dims_list", None),
            dedup=method_cfg.get("dedup", True),
            pca=method_cfg.get("pca", False),
            pca_center=method_cfg.get("pca_center", True),
            seed=method_cfg.get("seed", None),
        )
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    ms = 1e3 * (time.perf_counter() - t0)
    return False, float("nan"), float("nan"), ms, "SKIP"


def run_sw(X, Y, alpha: float, B: int, method_cfg: Dict[str, Any]):
    t0 = time.perf_counter()

    if SlicedOTTest is not None:
        test = SlicedOTTest(
            L=method_cfg.get("num_projections", 256),
            B=B, alpha=alpha,
            decision=method_cfg.get("decision", "pvalue"),
            early_stop=method_cfg.get("early_stop", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 50),
            chunk=method_cfg.get("chunk", 256),
            antithetic=method_cfg.get("antithetic", True),
            dirs=None,
            seed=method_cfg.get("seed", 2026),
        )
        out = test.run(X, Y)
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    if callable(sliced_ot_permutation_test):
        out = sliced_ot_permutation_test(
            X, Y,
            L=method_cfg.get("num_projections", 256),
            B=B, alpha=alpha,
            decision=method_cfg.get("decision", "pvalue"),
            early_stop=method_cfg.get("early_stop", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 50),
            chunk=method_cfg.get("chunk", 256),
            antithetic=method_cfg.get("antithetic", True),
            dirs=None, rng=None,
        )
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    ms = 1e3 * (time.perf_counter() - t0)
    return False, float("nan"), float("nan"), ms, "SKIP"


def run_t2(X, Y, alpha: float, B: int, method_cfg: Dict[str, Any]):
    t0 = time.perf_counter()

    if HotellingT2 is not None:
        try:
            test = HotellingT2(
                B=B, alpha=alpha,
                decision=method_cfg.get("decision", "pvalue"),
                early_stop=method_cfg.get("early_stop", "wilson"),
                delta_ci=method_cfg.get("delta_ci", 1e-2),
                min_b_check=method_cfg.get("min_b_check", 100),
                chunk=method_cfg.get("chunk", 256),
                antithetic=method_cfg.get("antithetic", True),
                shrinkage=method_cfg.get("shrinkage", "ridge"),
                ridge_lambda=method_cfg.get("lambda", 1e-6),
                seed=method_cfg.get("seed", 2026),
            )
        except TypeError:
            test = HotellingT2()
        out = test.run(X, Y)
        ms = 1e3 * (time.perf_counter() - t0)
        p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
        stat = getattr(out, "stat_obs", getattr(out, "t2_obs", np.nan))
        return bool(out.reject), float(stat), float(p), ms, "OK"

    if callable(hotelling_t2_permutation_test):
        out = hotelling_t2_permutation_test(
            X, Y,
            B=B, alpha=alpha,
            decision=method_cfg.get("decision", "pvalue"),
            early_stop=method_cfg.get("early_stop", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 100),
            chunk=method_cfg.get("chunk", 256),
            antithetic=method_cfg.get("antithetic", True),
            shrinkage=method_cfg.get("shrinkage", "ridge"),
            ridge_lambda=method_cfg.get("lambda", 1e-6),
            rng=None,
        )
        ms = 1e3 * (time.perf_counter() - t0)
        p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
        stat = getattr(out, "stat_obs", getattr(out, "t2_obs", np.nan))
        return bool(out.reject), float(stat), float(p), ms, "OK"

    ms = 1e3 * (time.perf_counter() - t0)
    return False, float("nan"), float("nan"), ms, "SKIP"


def run_mmd(X, Y, alpha: float, B: int, method_cfg: Dict[str, Any]):
    t0 = time.perf_counter()

    kernel = method_cfg.get("kernel", "rbf")
    if kernel == "gaussian":
        kernel = "rbf"
    estimator = "unbiased" if method_cfg.get("unbiased", True) else "biased"

    rbf_sigmas: Optional[List[float]] = None
    rbf_use_median = True
    bw = method_cfg.get("bandwidth", "median")
    if isinstance(bw, (int, float)):
        rbf_sigmas = [float(bw)]
        rbf_use_median = False

    if KernelMMDTest is not None:
        try:
            test = KernelMMDTest(
                B=B, alpha=alpha,
                decision=method_cfg.get("decision", "pvalue"),
                early_stop=method_cfg.get("early_stop", "wilson"),
                delta_ci=method_cfg.get("delta_ci", 1e-2),
                min_b_check=method_cfg.get("min_b_check", 50),
                chunk=method_cfg.get("chunk", 256),
                antithetic=method_cfg.get("antithetic", True),
                kernel=kernel, estimator=estimator,
                rbf_sigmas=rbf_sigmas, rbf_use_median=rbf_use_median,
            )
        except TypeError:
            test = KernelMMDTest()
        out = test.run(X, Y)
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    if callable(kernel_mmd_permutation_test):
        out = kernel_mmd_permutation_test(
            X, Y,
            B=B, alpha=alpha,
            decision=method_cfg.get("decision", "pvalue"),
            early_stop=method_cfg.get("early_stop", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 50),
            chunk=method_cfg.get("chunk", 256),
            antithetic=method_cfg.get("antithetic", True),
            kernel=kernel, estimator=estimator,
            rbf_sigmas=rbf_sigmas, rbf_use_median=rbf_use_median,
        )
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    ms = 1e3 * (time.perf_counter() - t0)
    return False, float("nan"), float("nan"), ms, "SKIP"


def run_mmd_tuned(X, Y, alpha: float, B: int, method_cfg: Dict[str, Any]):
    t0 = time.perf_counter()

    if TunedMMDTest is not None:
        try:
            test = TunedMMDTest(
                alpha=alpha, B_test=B,
                decision=method_cfg.get("decision", "pvalue"),
                test_early_stop=method_cfg.get("early_stop", "wilson"),
                split_prop=method_cfg.get("split_prop", 0.5),
                n_splits=method_cfg.get("n_splits", 1),
                seed=method_cfg.get("seed", 2028),
            )
        except TypeError:
            test = TunedMMDTest()
        out = test.run(X, Y)
        ms = 1e3 * (time.perf_counter() - t0)
        p = float(getattr(out, "p_combined", np.nan))
        stat = float("nan")
        if getattr(out, "splits", None):
            stat = float(getattr(out.splits[0], "stat_obs", np.nan))
        return bool(out.reject), stat, p, ms, "OK"

    if callable(tuned_mmd_permutation_test):
        out = tuned_mmd_permutation_test(X, Y, alpha=alpha, B_test=B)
        ms = 1e3 * (time.perf_counter() - t0)
        p = float(getattr(out, "p_combined", np.nan))
        stat = float("nan")
        if getattr(out, "splits", None):
            stat = float(getattr(out.splits[0], "stat_obs", np.nan))
        return bool(out.reject), stat, p, ms, "OK"

    ms = 1e3 * (time.perf_counter() - t0)
    return False, float("nan"), float("nan"), ms, "SKIP"


def run_c2st(X, Y, alpha: float, B: int, method_cfg: Dict[str, Any]):
    t0 = time.perf_counter()

    if C2ST is not None:
        try:
            test = C2ST(
                B=B, alpha=alpha,
                decision=method_cfg.get("decision", "pvalue"),
                early_stop=method_cfg.get("early_stop", "wilson"),
                delta_ci=method_cfg.get("delta_ci", 1e-2),
                min_b_check=method_cfg.get("min_b_check", 50),
                cv_mode=method_cfg.get("cv_mode", "kfold"),
                k_folds=method_cfg.get("k_folds", 5),
                split_prop=method_cfg.get("split_prop", 0.5),
                metric=method_cfg.get("metric", "acc"),
                retrain=method_cfg.get("retrain", True),
                seed=method_cfg.get("seed", 2027),
            )
        except TypeError:
            test = C2ST()
        out = test.run(X, Y)
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    if callable(c2st_permutation_test):
        out = c2st_permutation_test(
            X, Y,
            B=B, alpha=alpha,
            decision=method_cfg.get("decision", "pvalue"),
            early_stop=method_cfg.get("early_stop", "wilson"),
            delta_ci=method_cfg.get("delta_ci", 1e-2),
            min_b_check=method_cfg.get("min_b_check", 50),
            cv_mode=method_cfg.get("cv_mode", "kfold"),
            k_folds=method_cfg.get("k_folds", 5),
            split_prop=method_cfg.get("split_prop", 0.5),
            metric=method_cfg.get("metric", "acc"),
            retrain=method_cfg.get("retrain", True),
            seed=method_cfg.get("seed", 2027),
        )
        ms = 1e3 * (time.perf_counter() - t0)
        return (
            bool(out.reject),
            float(getattr(out, "stat_obs", np.nan)),
            float(getattr(out, "p_perm", np.nan)),
            ms,
            "OK",
        )

    ms = 1e3 * (time.perf_counter() - t0)
    return False, float("nan"), float("nan"), ms, "SKIP"


METHODS: Dict[str, Callable[..., Tuple[bool, float, float, float, str]]] = {
    "RankChi2": run_rank,
    "RankChi2_Subspaces": run_rank_subspaces,
    "SlicedOT": run_sw,
    "HotellingT2": run_t2,
    "MMD": run_mmd,
    "MMD_Tuned": run_mmd_tuned,
    "C2ST": run_c2st,
}


def one_rep(
    rep_seed: int,
    spec: BenchSpec,
    alpha: float,
    B: int,
    per_method_cfg: Dict[str, Dict[str, Any]],
    active_methods: Dict[str, Callable],
    method_timeout_s: Optional[float] = None,
    experiment: str = "power",
):
    rng = np.random.default_rng(rep_seed)

    if experiment == "calibration":
        X, Y = generate_null_pair(spec, rng)
    else:
        X, Y = generate_pair(
            spec.family,
            spec.d,
            spec.N,
            spec.M,
            params=spec.params,
            rng=rng,
        )

    out: Dict[str, Dict[str, Any]] = {}
    for name, runner in active_methods.items():
        cfg = dict(per_method_cfg.get(name, {}))

        if name == "RankChi2" and cfg.get("reference_mode", "shared_pool") == "fresh_iid":
            cfg["reference_sampler"] = SimulationReferenceSampler(
                family=spec.family,
                d=spec.d,
                params=spec.params,
            )

        if name == "RankChi2_Subspaces":
            cfg["L"] = int(spec.L_subspaces)

        rej, stat, pval, ms, status = run_with_timeout(
            runner, X, Y, alpha, B, cfg, method_timeout_s
        )
        out[name] = {
            "reject": rej,
            "stat": stat,
            "pval": pval,
            "ms": ms,
            "status": status,
        }
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Power / calibration benchmark across dataset families and methods."
    )

    ap.add_argument(
        "--experiment",
        choices=["power", "calibration"],
        default="power",
        help=(
            "power: generate data under H1 and estimate rejection probability; "
            "calibration: generate data under H0 and estimate empirical size."
        ),
    )

    ap.add_argument(
        "--families",
        nargs="+",
        default=[
            "locshift",
            "scale-gauss",
            "shape-laplace",
            "shape-studentt",
            "mixture",
            "dep-tcopula",
            "dep-clayton",
            "dep-gumbel",
        ],
    )

    ap.add_argument("--d", type=int, default=4)

    # Pairwise sweep of (N, M)
    ap.add_argument(
        "--Ns",
        type=int,
        nargs="+",
        default=[1000],
        help="List of target sample sizes N (space-separated).",
    )
    ap.add_argument(
        "--Ms",
        type=int,
        nargs="+",
        default=[1000],
        help="List of reference-pool sizes M (space-separated). "
             "Paired with --Ns: runs (N1,M1), (N2,M2), ...",
    )

    # Optional backward-compat singletons
    ap.add_argument("--N", type=int, default=None,
                    help="(Deprecated) Single N. Prefer --Ns.")
    ap.add_argument("--M", type=int, default=None,
                    help="(Deprecated) Single M. Prefer --Ms.")
    ap.add_argument("--K", type=int, default=None,
                    help="(Deprecated alias) Same as --M (legacy scripts).")

    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=500,
                    help="Permutations per test call")
    ap.add_argument("--R", type=int, default=200,
                    help="Repetitions for estimation")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2028)
    ap.add_argument("--out_csv", type=str, default="power_results.csv")
    ap.add_argument(
        "--method_timeout_s",
        type=float,
        default=0.0,
        help="Per-method wall-clock timeout in seconds for each repetition. Set <= 0 to disable.",
    )

    ap.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS.keys()),
        help=f"Subset of methods to run. Options: {', '.join(METHODS.keys())}",
    )

    ap.add_argument(
        "--K_ref",
        type=int,
        default=None,
        help="K_ref for RankChi2 / RankChi2_Subspaces. If omitted: min(len(Y),4). "
             "Always clipped to <=len(Y) at runtime.",
    )

    ap.add_argument(
        "--Ls_subspaces",
        type=int,
        nargs="+",
        default=[64],
        help="List of L values to sweep for RankChi2_Subspaces. Example: --Ls_subspaces 8 16 32 64 128",
    )

    ap.add_argument(
        "--rank_reference_mode",
        choices=["shared_pool", "fresh_iid"],
        default="shared_pool",
        help=(
            "Reference mode for RankChi2 only. "
            "'shared_pool' keeps the current pooled-reference implementation. "
            "'fresh_iid' uses independent fresh K_ref-batches sampled from the known "
            "simulation reference law via generate_pair(..., N=0, K=n)."
        ),
    )

    # Family-specific overrides
    ap.add_argument("--rho", type=float, default=0.6,
                    help="AR1/equi correlation for some families")
    ap.add_argument("--delta", type=float, default=0.4,
                    help="location/mix effect")
    ap.add_argument("--sigma", type=float, default=1.2,
                    help="scale-gauss sigma")
    ap.add_argument("--nu", type=int, default=6, help="student-t df")

    args = ap.parse_args()
    jobs = args.jobs or (os.cpu_count() or 1)

    # Backward compatibility
    if args.N is not None:
        args.Ns = [int(args.N)]
        if len(args.Ms) != 1:
            raise SystemExit(
                "[ERROR] When using deprecated --N, please provide exactly one --Ms value (or use --M)."
            )

    if args.M is not None:
        args.Ms = [int(args.M)]
        if len(args.Ns) != 1:
            raise SystemExit(
                "[ERROR] When using deprecated --M, please provide exactly one --Ns value (or use --N)."
            )

    if args.K is not None:
        if args.M is not None:
            raise SystemExit(
                "[ERROR] Use only one of --K (legacy) or --M (preferred).")
        args.Ms = [int(args.K)]
        if len(args.Ns) != 1:
            raise SystemExit(
                "[ERROR] When using deprecated --K, please provide exactly one --Ns value (or use --N)."
            )

    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise SystemExit(
            f"[ERROR] Unknown method(s): {unknown}. Valid: {list(METHODS.keys())}"
        )

    active_methods = {m: METHODS[m] for m in args.methods}

    per_method_cfg: Dict[str, Dict[str, Any]] = {
        "RankChi2": {
            "K_ref": args.K_ref,
            "stat": "chi2",
            "tie": "jitter",
            "decision": "randomized",
            "engine": "auto",
            "kref_switch": 64,
            "fullY": True,
            "stop_ci": "wilson",
            "delta_ci": 1e-2,
            "min_b_check": 100,
            "chunk": 256,
            "antithetic": True,
            "reference_mode": args.rank_reference_mode,
            "reference_sampler": None,
        },
        "RankChi2_Subspaces": {
            "K_ref": args.K_ref,
            "stat": "chi2",
            "tie": "jitter",
            "decision": "midp",
            "engine": "auto",
            "kref_switch": 64,
            "fullY": True,
            "stop_ci": "wilson",
            "delta_ci": 1e-2,
            "min_b_check": 100,
            "chunk": 256,
            "antithetic": True,
            "L": int(args.Ls_subspaces[0]),
            "k_dim": 2,
            "dims_list": None,
            "dedup": True,
            "pca": False,
            "pca_center": True,
        },
        "SlicedOT": {
            "num_projections": 64,
            "decision": "pvalue",
            "early_stop": "wilson",
            "delta_ci": 1e-2,
            "min_b_check": 50,
        },
        "HotellingT2": {
            "lambda": 1e-6,
            "shrinkage": "ridge",
            "decision": "pvalue",
            "early_stop": "wilson",
            "delta_ci": 1e-2,
            "min_b_check": 100,
        },
        "MMD": {
            "kernel": "rbf",
            "bandwidth": "median",
            "unbiased": True,
            "decision": "pvalue",
            "early_stop": "wilson",
            "delta_ci": 1e-2,
            "min_b_check": 50,
        },
        "MMD_Tuned": {
            "decision": "pvalue",
            "early_stop": "wilson",
            "split_prop": 0.7,
            "n_splits": 1,
        },
        "C2ST": {
            "metric": "acc",
            "cv_mode": "holdout",
            "split_prop": 0.7,
            "retrain": True,
            "decision": "pvalue",
            "early_stop": "wilson",
            "delta_ci": 1e-2,
            "min_b_check": 50,
        },
    }

    nm_pairs = _pairwise_nm_lists(args.Ns, args.Ms)

    specs: List[BenchSpec] = []
    for fam in args.families:
        params: Dict[str, Any] = {}
        if fam == "locshift":
            params = {"rho": args.rho, "delta": args.delta}
        elif fam == "scale-gauss":
            params = {"sigma": args.sigma}
        elif fam == "shape-studentt":
            params = {"nu": args.nu}
        elif fam in ("dep-tcopula", "dep-clayton", "dep-gumbel"):
            params = {"rho": args.rho, "nu": args.nu, "equicorr": True}
        elif fam == "mixture":
            params = {"delta": max(args.delta, 0.8)}
        elif fam == "shape-laplace":
            params = {}
        else:
            print(f"[WARN] Unknown family {fam}, skipping.")
            continue

        for (N, M) in nm_pairs:
            for L_sub in args.Ls_subspaces:
                specs.append(
                    BenchSpec(
                        family=fam,
                        d=args.d,
                        N=int(N),
                        M=int(M),
                        params=params,
                        L_subspaces=int(L_sub),
                    )
                )

    metric_symbol = "π̂" if args.experiment == "power" else "α̂"
    metric_name = "power" if args.experiment == "power" else "size"

    ss = np.random.SeedSequence(args.seed)
    spawned = ss.spawn(args.R)
    rep_seeds = [int(s.generate_state(1)[0]) for s in spawned]

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        print(
            f"\n=== Experiment: {args.experiment} | Family: {spec.family} | d={spec.d} "
            f"N={spec.N} M={spec.M} L_subspaces={spec.L_subspaces} params={spec.params} ==="
        )
        print(
            f"=== Methods: {list(active_methods.keys())} | "
            f"K_ref(arg)={args.K_ref} | timeout={args.method_timeout_s}s ==="
        )

        counts = {name: 0 for name in active_methods.keys()}
        completed = {name: 0 for name in active_methods.keys()}
        timeout_counts = {name: 0 for name in active_methods.keys()}
        error_counts = {name: 0 for name in active_methods.keys()}
        times = {name: [] for name in active_methods.keys()}
        statuses = {name: "OK" for name in active_methods.keys()}

        if jobs == 1:
            for r, rs in enumerate(rep_seeds, 1):
                out = one_rep(
                    rs, spec, args.alpha, args.B,
                    per_method_cfg, active_methods,
                    args.method_timeout_s,
                    args.experiment,
                )
                for name in active_methods.keys():
                    info = out[name]
                    if info["status"] == "SKIP":
                        statuses[name] = "SKIP"
                        continue

                    times[name].append(info["ms"])

                    if info["status"] == "TIMEOUT":
                        timeout_counts[name] += 1
                        if statuses[name] == "OK":
                            statuses[name] = "PARTIAL_TIMEOUT"
                        continue

                    if str(info["status"]).startswith("ERROR:"):
                        error_counts[name] += 1
                        if statuses[name] == "OK":
                            statuses[name] = "PARTIAL_ERROR"
                        continue

                    completed[name] += 1
                    counts[name] += int(info["reject"])

                if r % max(1, args.R // 10) == 0:
                    print(f"  progress {r}/{args.R} ...", end="\r")
        else:
            with ProcessPoolExecutor(max_workers=jobs) as ex:
                futs = [
                    ex.submit(
                        one_rep,
                        rs, spec, args.alpha, args.B,
                        per_method_cfg, active_methods,
                        args.method_timeout_s,
                        args.experiment,
                    )
                    for rs in rep_seeds
                ]
                done = 0
                for f in as_completed(futs):
                    out = f.result()
                    for name in active_methods.keys():
                        info = out[name]
                        if info["status"] == "SKIP":
                            statuses[name] = "SKIP"
                            continue

                        times[name].append(info["ms"])

                        if info["status"] == "TIMEOUT":
                            timeout_counts[name] += 1
                            if statuses[name] == "OK":
                                statuses[name] = "PARTIAL_TIMEOUT"
                            continue

                        if str(info["status"]).startswith("ERROR:"):
                            error_counts[name] += 1
                            if statuses[name] == "OK":
                                statuses[name] = "PARTIAL_ERROR"
                            continue

                        completed[name] += 1
                        counts[name] += int(info["reject"])

                    done += 1
                    if done % max(1, args.R // 10) == 0:
                        print(f"  progress {done}/{args.R} ...", end="\r")

        for name in active_methods.keys():
            kref_value = args.K_ref if name in (
                "RankChi2", "RankChi2_Subspaces") else ""

            if statuses[name] == "SKIP":
                print(f"  {name:16s}: SKIP (module not found)")
                rows.append({
                    "experiment": args.experiment,
                    "metric_name": metric_name,
                    "family": spec.family,
                    "method": name,
                    "d": spec.d,
                    "N": spec.N,
                    "M": spec.M,
                    "L_subspaces": spec.L_subspaces,
                    "alpha": args.alpha,
                    "B": args.B,
                    "R": args.R,
                    "R_completed": 0,
                    "R_timeout": 0,
                    "R_error": 0,
                    "K_ref": kref_value,
                    "reject_rate": "NA",
                    "power": "NA",
                    "size": "NA",
                    "se": "NA",
                    "avg_ms": "NA",
                    "status": "SKIP",
                })
                continue

            R_eff = completed[name]
            avg_ms = float(np.mean(times[name])
                           ) if times[name] else float("nan")

            if R_eff == 0:
                p_hat = float("nan")
                se_hat = float("nan")
                print(
                    f"  {name:16s}: no completed reps "
                    f"(timeouts={timeout_counts[name]}, errors={error_counts[name]})"
                )
            else:
                p_hat = counts[name] / float(R_eff)
                se_hat = se_binom(p_hat, R_eff)
                print(
                    f"  {name:16s}: {metric_symbol}={p_hat:0.3f}  SE≈{se_hat:0.3f}   "
                    f"avg_ms/attempt≈{avg_ms:0.1f}   "
                    f"completed={R_eff}/{args.R}   "
                    f"timeouts={timeout_counts[name]}   errors={error_counts[name]}"
                )

            p_str = f"{p_hat:.6f}" if not np.isnan(p_hat) else "NA"
            se_str = f"{se_hat:.6f}" if not np.isnan(se_hat) else "NA"
            ms_str = f"{avg_ms:.2f}" if not np.isnan(avg_ms) else "NA"

            rows.append({
                "experiment": args.experiment,
                "metric_name": metric_name,
                "family": spec.family,
                "method": name,
                "d": spec.d,
                "N": spec.N,
                "M": spec.M,
                "L_subspaces": spec.L_subspaces,
                "alpha": args.alpha,
                "B": args.B,
                "R": args.R,
                "R_completed": completed[name],
                "R_timeout": timeout_counts[name],
                "R_error": error_counts[name],
                "K_ref": kref_value,
                "reject_rate": p_str,
                "power": p_str,  # backward compatibility with existing plotting code
                "size": p_str if args.experiment == "calibration" else "NA",
                "se": se_str,
                "avg_ms": ms_str,
                "status": statuses[name],
            })

    outp = args.out_csv
    if not rows:
        raise RuntimeError(
            "No rows produced (maybe all families were skipped?).")

    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[Done] Wrote {len(rows)} rows to {outp}")
    print("Hint: filter by the 'experiment' column when plotting.")


if __name__ == "__main__":
    # Examples:
    #
    # Power:
    #   PYTHONPATH=./src python src/scripts/run_power_bench.py \
    #       --experiment power \
    #       --Ns 1000 5000 10000 --Ms 1000 5000 10000
    #
    # Power, RankChi2_Subspaces L sweep:
    #   PYTHONPATH=./src python src/scripts/run_power_bench.py \
    #       --experiment power \
    #       --methods RankChi2_Subspaces \
    #       --Ns 1000 2000 4000 --Ms 1000 2000 4000 \
    #       --Ls_subspaces 8 16 32 64 128 \
    #       --K_ref 4 \
    #       --out_csv power_subspaces_L_sweep.csv
    #
    # Calibration:
    #   PYTHONPATH=./src python src/scripts/run_power_bench.py \
    #       --experiment calibration \
    #       --methods RankChi2 RankChi2_Subspaces SlicedOT HotellingT2 MMD \
    #       --Ns 100 200 500 1000 2000 4000 \
    #       --Ms 100 200 500 1000 2000 4000 \
    #       --Ls_subspaces 8 16 64 \
    #       --K_ref 4 \
    #       --alpha 0.05 \
    #       --R 100 \
    #       --B 500 \
    #       --out_csv calibration_results.csv
    #
    # RankChi2 with fresh iid references:
    #   PYTHONPATH=./src python src/scripts/run_power_bench.py \
    #       --experiment power \
    #       --methods RankChi2 \
    #       --K_ref 4 \
    #       --rank_reference_mode fresh_iid
    #
    # Timeout example:
    #   PYTHONPATH=./src python src/scripts/run_power_bench.py \
    #       --experiment power \
    #       --methods RankChi2 MMD \
    #       --method_timeout_s 10
    main()
