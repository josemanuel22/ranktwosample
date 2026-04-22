#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIFAR-10 vs CIFAR-10-C power benchmark (paper-style, no PyTorch / no torchvision)

Methods:
  - joint_subspaces : random-subspace joint-ranks χ² test (paper method)
  - swd             : Sliced Wasserstein, permutation-calibrated
  - hotelling       : Hotelling's T² with ridge shrinkage, permutation-calibrated
  - mmd             : Gaussian-kernel MMD, permutation-calibrated
  - c2st            : classifier two-sample test, permutation-calibrated

This version avoids importing torch/torchvision entirely to sidestep macOS
OpenMP runtime conflicts from those packages.

Expected paths:
  --cifar-root ./data/cifar-10-batches-py
      containing test_batch
  --c10c-root  ./data
      containing gaussian_noise.npy, jpeg_compression.npy, etc.

Examples:

Pairwise sweep:
  PYTHONPATH=./src python cifar10c_power_rank_subspaces.py \
      --cifar-root ./data/cifar-10-batches-py \
      --c10c-root ./data \
      --corruptions gaussian_noise,jpeg_compression \
      --severities 1,3,5 \
      --methods joint_subspaces,swd,mmd \
      --trials 20 --alpha 0.01 \
      --n-refs 100,200,500,1000,2000 \
      --n-tars 100,200,500,1000,2000 \
      --nm-mode pairwise \
      --K-ref 4 --d-sub 2 --L-subspaces 20

Grid sweep:
  PYTHONPATH=./src python cifar10c_power_rank_subspaces.py \
      --cifar-root ./data/cifar-10-batches-py \
      --c10c-root ./data \
      --corruptions gaussian_noise \
      --severities 1,3 \
      --methods joint_subspaces,swd \
      --trials 10 --alpha 0.05 \
      --n-refs 100,500 \
      --n-tars 100,500,1000 \
      --nm-mode grid \
      --K-ref 4 --d-sub 2 --L-subspaces 20
"""

from __future__ import annotations

import os
import re
import csv
import json
import math
import time
import pickle
import argparse
from typing import Dict, Tuple, List, Optional

import numpy as np
from tqdm import tqdm


# ============================================================
# Utilities
# ============================================================

def set_deterministic(seed: int = 0) -> None:
    np.random.seed(seed)


def _wald_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    se = math.sqrt(max(p_hat * (1.0 - p_hat) / max(n, 1), 0.0))
    return max(p_hat - z * se, 0.0), min(p_hat + z * se, 1.0)


def standardize_by_ref(X_ref: np.ndarray, X_tar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X_ref.mean(axis=0, keepdims=True)
    sd = X_ref.std(axis=0, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (X_ref - mu) / sd, (X_tar - mu) / sd


def _clip_kref(kref: Optional[int], y_len: int, default: int = 4) -> int:
    if y_len <= 0:
        return 1
    if kref is None:
        return min(y_len, default)
    k = int(kref)
    k = max(1, k)
    k = min(k, y_len)
    return k


def _parse_int_list(s: str) -> List[int]:
    """
    Parse comma-separated and/or whitespace-separated integers.
    Examples:
      "100,200,500"
      "100 200 500"
      "100, 200, 500"
    """
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"[,\s]+", s)
    vals = [int(x) for x in parts if x]
    if not vals:
        raise ValueError(f"Could not parse integer list from: {s!r}")
    return vals


# ============================================================
# Lazy imports for optional baselines
# ============================================================

def _import_rank_subspaces_backend():
    from baselines.rank_chi2_subspaces import (
        RankChi2SubspacesTest,
        rank_chi2_subspaces_permutation_test,
    )
    return RankChi2SubspacesTest, rank_chi2_subspaces_permutation_test


def _import_swd_backend():
    from baselines.sliced_ot import SlicedOTTest, sliced_ot_permutation_test
    return SlicedOTTest, sliced_ot_permutation_test


def _import_hotelling_backend():
    from baselines.hotelling_t2 import HotellingT2, hotelling_t2_permutation_test
    return HotellingT2, hotelling_t2_permutation_test


def _import_mmd_backend():
    from baselines.kernel_mmd import KernelMMDTest, kernel_mmd_permutation_test
    return KernelMMDTest, kernel_mmd_permutation_test


def _import_c2st_backend():
    from baselines.c2st import C2ST, c2st_permutation_test
    return C2ST, c2st_permutation_test


# ============================================================
# CIFAR-10 / CIFAR-10-C loading (no torchvision)
# ============================================================

def cache_path_c10(cache_dir: str) -> str:
    return os.path.join(cache_dir, "cifar10_test_pixels.npy")


def cache_path_c10c(cache_dir: str, corruption: str, severity: int) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", corruption)
    return os.path.join(cache_dir, f"cifar10c_{safe_name}_sev{severity}_pixels.npy")


def _ensure_dir(path: str, arg_name: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{arg_name} must be a directory, got: {path}\n"
            f"If you passed a .tar file, extract it first."
        )


def ensure_cifar10c(npy_root: str) -> None:
    _ensure_dir(npy_root, "--c10c-root")
    sentinel = os.path.join(npy_root, "gaussian_noise.npy")
    if not os.path.exists(sentinel):
        raise FileNotFoundError(
            f"CIFAR-10-C .npy files not found under {npy_root}. "
            f"Expected files like gaussian_noise.npy, jpeg_compression.npy, etc."
        )


def _load_cifar10_test_from_batches_py(cifar_root: str) -> np.ndarray:
    """
    Load CIFAR-10 test set from extracted cifar-10-batches-py folder.
    Expects:
      cifar_root/test_batch
    """
    _ensure_dir(cifar_root, "--cifar-root")
    test_batch_path = os.path.join(cifar_root, "test_batch")
    if not os.path.exists(test_batch_path):
        raise FileNotFoundError(
            f"Could not find {test_batch_path}.\n"
            f"Set --cifar-root to the extracted 'cifar-10-batches-py' directory."
        )

    with open(test_batch_path, "rb") as f:
        obj = pickle.load(f, encoding="bytes")

    # shape: (10000, 3072) uint8
    X = obj[b"data"]
    X = np.asarray(X, dtype=np.float32) / 255.0
    return X


def load_or_cache_cifar10_test_pixels(cifar_root: str, cache_dir: str) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    p = cache_path_c10(cache_dir)
    if os.path.exists(p):
        return np.load(p)

    X = _load_cifar10_test_from_batches_py(cifar_root)
    np.save(p, X)
    return X


def load_or_cache_cifar10c_pixels(
    c10c_root: str,
    cache_dir: str,
    corruption: str,
    severity: int,
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    p = cache_path_c10c(cache_dir, corruption, severity)
    if os.path.exists(p):
        return np.load(p)

    ensure_cifar10c(c10c_root)
    # (50000, 32, 32, 3)
    arr = np.load(os.path.join(c10c_root, f"{corruption}.npy"))
    start = (severity - 1) * 10000
    end = severity * 10000
    X = arr[start:end].reshape(10000, -1).astype(np.float32) / 255.0
    np.save(p, X)
    return X


# ============================================================
# Method runners
# ============================================================

def run_joint_subspaces(
    X_tar: np.ndarray,
    X_ref: np.ndarray,
    alpha: float,
    B: int,
    *,
    K_ref: int,
    L_subspaces: int,
    d_sub: int,
    decision: str = "midp",
    stat: str = "chi2",
    tie: str = "jitter",
    seed: int = 2026,
) -> Dict[str, float]:
    RankChi2SubspacesTest, rank_chi2_subspaces_permutation_test = _import_rank_subspaces_backend()

    t0 = time.perf_counter()
    K_ref = _clip_kref(K_ref, len(X_ref), default=4)

    if RankChi2SubspacesTest is not None:
        try:
            test = RankChi2SubspacesTest(
                K_ref=K_ref,
                alpha=alpha,
                B=B,
                engine="auto",
                tie=tie,
                alpha0=0.1,
                decision=decision,
                kref_switch=64,
                chunk=256,
                antithetic=True,
                stop_ci="wilson",
                delta_ci=1e-2,
                min_b_check=100,
                use_fullY_for_HY=True,
                stat=stat,
                gpu="off",
                device=0,
                L=L_subspaces,
                k_dim=d_sub,
                dims_list=None,
                dedup=True,
                pca=False,
                pca_center=True,
            )
            out = test.run(X_tar, X_ref)
            ms = 1e3 * (time.perf_counter() - t0)
            p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
            stat_obs = getattr(out, "stat_obs", np.nan)
            reject = getattr(out, "reject", p <= alpha)
            return {
                "stat": float(stat_obs),
                "p": float(p),
                "reject": bool(reject),
                "time_sec": ms / 1e3,
            }
        except TypeError:
            pass

    out = rank_chi2_subspaces_permutation_test(
        X_tar,
        X_ref,
        alpha=float(alpha),
        B=int(B),
        K_ref=K_ref,
        engine="auto",
        tie=tie,
        alpha0=0.1,
        decision=decision,
        kref_switch=64,
        chunk=256,
        antithetic=True,
        stop_ci="wilson",
        delta_ci=1e-2,
        min_b_check=100,
        use_fullY_for_HY=True,
        stat=stat,
        jitter_scale=1e-12,
        gpu="off",
        device=0,
        L=L_subspaces,
        k_dim=d_sub,
        dims_list=None,
        dedup=True,
        pca=False,
        pca_center=True,
        seed=seed,
    )
    ms = 1e3 * (time.perf_counter() - t0)
    p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
    stat_obs = getattr(out, "stat_obs", np.nan)
    reject = getattr(out, "reject", p <= alpha)
    return {
        "stat": float(stat_obs),
        "p": float(p),
        "reject": bool(reject),
        "time_sec": ms / 1e3,
    }


def run_swd(
    X_tar: np.ndarray,
    X_ref: np.ndarray,
    alpha: float,
    B: int,
    *,
    L: int = 256,
) -> Dict[str, float]:
    SlicedOTTest, sliced_ot_permutation_test = _import_swd_backend()

    t0 = time.perf_counter()
    if SlicedOTTest is not None:
        try:
            test = SlicedOTTest(
                L=L,
                B=B,
                alpha=alpha,
                decision="pvalue",
                early_stop="wilson",
                delta_ci=1e-2,
                min_b_check=50,
                chunk=256,
                antithetic=True,
                dirs=None,
                seed=2026,
            )
            out = test.run(X_tar, X_ref)
            ms = 1e3 * (time.perf_counter() - t0)
            return {
                "stat": float(getattr(out, "stat_obs", np.nan)),
                "p": float(getattr(out, "p_perm", np.nan)),
                "reject": bool(getattr(out, "reject", False)),
                "time_sec": ms / 1e3,
            }
        except TypeError:
            pass

    out = sliced_ot_permutation_test(
        X_tar,
        X_ref,
        L=L,
        B=B,
        alpha=alpha,
        decision="pvalue",
        early_stop="wilson",
        delta_ci=1e-2,
        min_b_check=50,
        chunk=256,
        antithetic=True,
        dirs=None,
        rng=None,
    )
    ms = 1e3 * (time.perf_counter() - t0)
    p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
    reject = getattr(out, "reject", p <= alpha)
    return {
        "stat": float(getattr(out, "stat_obs", np.nan)),
        "p": float(p),
        "reject": bool(reject),
        "time_sec": ms / 1e3,
    }


def run_hotelling(
    X_tar: np.ndarray,
    X_ref: np.ndarray,
    alpha: float,
    B: int,
    *,
    ridge_lambda: float = 1e-2,
) -> Dict[str, float]:
    HotellingT2, hotelling_t2_permutation_test = _import_hotelling_backend()

    t0 = time.perf_counter()
    if HotellingT2 is not None:
        try:
            test = HotellingT2(
                B=B,
                alpha=alpha,
                decision="pvalue",
                early_stop="wilson",
                delta_ci=1e-2,
                min_b_check=100,
                chunk=256,
                antithetic=True,
                shrinkage="ridge",
                ridge_lambda=ridge_lambda,
                seed=2026,
            )
            out = test.run(X_tar, X_ref)
            ms = 1e3 * (time.perf_counter() - t0)
            p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
            stat_obs = getattr(out, "stat_obs", getattr(out, "t2_obs", np.nan))
            reject = getattr(out, "reject", p <= alpha)
            return {
                "stat": float(stat_obs),
                "p": float(p),
                "reject": bool(reject),
                "time_sec": ms / 1e3,
            }
        except TypeError:
            pass

    out = hotelling_t2_permutation_test(
        X_tar,
        X_ref,
        B=B,
        alpha=alpha,
        decision="pvalue",
        early_stop="wilson",
        delta_ci=1e-2,
        min_b_check=100,
        chunk=256,
        antithetic=True,
        shrinkage="ridge",
        ridge_lambda=ridge_lambda,
        rng=None,
    )
    ms = 1e3 * (time.perf_counter() - t0)
    p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
    stat_obs = getattr(out, "stat_obs", getattr(out, "t2_obs", np.nan))
    reject = getattr(out, "reject", p <= alpha)
    return {
        "stat": float(stat_obs),
        "p": float(p),
        "reject": bool(reject),
        "time_sec": ms / 1e3,
    }


def run_mmd(
    X_tar: np.ndarray,
    X_ref: np.ndarray,
    alpha: float,
    B: int,
) -> Dict[str, float]:
    KernelMMDTest, kernel_mmd_permutation_test = _import_mmd_backend()

    t0 = time.perf_counter()
    if KernelMMDTest is not None:
        try:
            test = KernelMMDTest(
                B=B,
                alpha=alpha,
                decision="pvalue",
                early_stop="wilson",
                delta_ci=1e-2,
                min_b_check=50,
                chunk=256,
                antithetic=True,
                kernel="rbf",
                estimator="unbiased",
                rbf_sigmas=None,
                rbf_use_median=True,
            )
            out = test.run(X_tar, X_ref)
            ms = 1e3 * (time.perf_counter() - t0)
            p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
            reject = getattr(out, "reject", p <= alpha)
            return {
                "stat": float(getattr(out, "stat_obs", np.nan)),
                "p": float(p),
                "reject": bool(reject),
                "time_sec": ms / 1e3,
            }
        except TypeError:
            pass

    out = kernel_mmd_permutation_test(
        X_tar,
        X_ref,
        B=B,
        alpha=alpha,
        decision="pvalue",
        early_stop="wilson",
        delta_ci=1e-2,
        min_b_check=50,
        chunk=256,
        antithetic=True,
        kernel="rbf",
        estimator="unbiased",
        rbf_sigmas=None,
        rbf_use_median=True,
    )
    ms = 1e3 * (time.perf_counter() - t0)
    p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
    reject = getattr(out, "reject", p <= alpha)
    return {
        "stat": float(getattr(out, "stat_obs", np.nan)),
        "p": float(p),
        "reject": bool(reject),
        "time_sec": ms / 1e3,
    }


def run_c2st(
    X_tar: np.ndarray,
    X_ref: np.ndarray,
    alpha: float,
    B: int,
    *,
    split_prop: float = 0.7,
) -> Dict[str, float]:
    C2ST, c2st_permutation_test = _import_c2st_backend()

    t0 = time.perf_counter()
    if C2ST is not None:
        try:
            test = C2ST(
                B=B,
                alpha=alpha,
                decision="pvalue",
                early_stop="wilson",
                delta_ci=1e-2,
                min_b_check=50,
                cv_mode="holdout",
                k_folds=5,
                split_prop=split_prop,
                metric="acc",
                retrain=True,
                seed=2027,
            )
            out = test.run(X_tar, X_ref)
            ms = 1e3 * (time.perf_counter() - t0)
            p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
            reject = getattr(out, "reject", p <= alpha)
            return {
                "stat": float(getattr(out, "stat_obs", np.nan)),
                "p": float(p),
                "reject": bool(reject),
                "time_sec": ms / 1e3,
            }
        except TypeError:
            pass

    out = c2st_permutation_test(
        X_tar,
        X_ref,
        B=B,
        alpha=alpha,
        decision="pvalue",
        early_stop="wilson",
        delta_ci=1e-2,
        min_b_check=50,
        cv_mode="holdout",
        k_folds=5,
        split_prop=split_prop,
        metric="acc",
        retrain=True,
        seed=2027,
    )
    ms = 1e3 * (time.perf_counter() - t0)
    p = getattr(out, "p_perm", getattr(out, "p_value", np.nan))
    reject = getattr(out, "reject", p <= alpha)
    return {
        "stat": float(getattr(out, "stat_obs", np.nan)),
        "p": float(p),
        "reject": bool(reject),
        "time_sec": ms / 1e3,
    }


# ============================================================
# Trial runner
# ============================================================

def run_trial(
    X_ref_all: np.ndarray,
    X_tar_all: np.ndarray,
    *,
    n_ref: int,
    n_tar: int,
    alpha: float,
    standardize: bool,
    methods: List[str],
    K_ref: int,
    d_sub: int,
    L_subspaces: int,
    B_joint: int,
    B_swd: int,
    B_hotelling: int,
    B_mmd: int,
    B_c2st: int,
    swd_L: int,
    hotelling_ridge: float,
    c2st_split: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)

    idx_ref = rng.choice(X_ref_all.shape[0], size=n_ref, replace=False)
    idx_tar = rng.choice(X_tar_all.shape[0], size=n_tar, replace=False)
    X_ref = X_ref_all[idx_ref]
    X_tar = X_tar_all[idx_tar]

    if standardize:
        X_ref, X_tar = standardize_by_ref(X_ref, X_tar)

    results: Dict[str, Dict[str, float]] = {}

    for method in methods:
        if method == "joint_subspaces":
            results[method] = run_joint_subspaces(
                X_tar,
                X_ref,
                alpha,
                B_joint,
                K_ref=K_ref,
                L_subspaces=L_subspaces,
                d_sub=d_sub,
                decision="midp",
                stat="chi2",
                tie="jitter",
                seed=int(rng.integers(1 << 31) - 1),
            )
        elif method == "swd":
            results[method] = run_swd(X_tar, X_ref, alpha, B_swd, L=swd_L)
        elif method == "hotelling":
            results[method] = run_hotelling(
                X_tar, X_ref, alpha, B_hotelling, ridge_lambda=hotelling_ridge)
        elif method == "mmd":
            results[method] = run_mmd(X_tar, X_ref, alpha, B_mmd)
        elif method == "c2st":
            results[method] = run_c2st(
                X_tar, X_ref, alpha, B_c2st, split_prop=c2st_split)
        else:
            raise ValueError(f"Unknown method: {method}")

    return results


def estimate_power(
    X_ref_all: np.ndarray,
    X_tar_all: np.ndarray,
    *,
    trials: int,
    alpha: float,
    n_ref: int,
    n_tar: int,
    standardize: bool,
    methods: List[str],
    K_ref: int,
    d_sub: int,
    L_subspaces: int,
    B_joint: int,
    B_swd: int,
    B_hotelling: int,
    B_mmd: int,
    B_c2st: int,
    swd_L: int,
    hotelling_ridge: float,
    c2st_split: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)

    counts = {m: 0 for m in methods}
    pvals = {m: [] for m in methods}
    sum_time = {m: 0.0 for m in methods}

    for _ in tqdm(range(trials), desc=f"Trials α={alpha:.2f}", leave=False):
        out = run_trial(
            X_ref_all,
            X_tar_all,
            n_ref=n_ref,
            n_tar=n_tar,
            alpha=alpha,
            standardize=standardize,
            methods=methods,
            K_ref=K_ref,
            d_sub=d_sub,
            L_subspaces=L_subspaces,
            B_joint=B_joint,
            B_swd=B_swd,
            B_hotelling=B_hotelling,
            B_mmd=B_mmd,
            B_c2st=B_c2st,
            swd_L=swd_L,
            hotelling_ridge=hotelling_ridge,
            c2st_split=c2st_split,
            seed=int(rng.integers(1 << 31) - 1),
        )
        for m in methods:
            counts[m] += int(bool(out[m]["reject"]))
            pvals[m].append(float(out[m]["p"]))
            sum_time[m] += float(out[m]["time_sec"])

    stats = {}
    for m in methods:
        power = counts[m] / max(trials, 1)
        lo, hi = _wald_ci(power, trials)
        med_p = float(np.median(np.asarray(pvals[m], dtype=float)))
        avg_time = sum_time[m] / max(trials, 1)
        stats[m] = {
            "power": power,
            "ci_lo": lo,
            "ci_hi": hi,
            "median_p": med_p,
            "avg_time_sec": avg_time,
        }
    return stats


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 vs CIFAR-10-C power benchmark using the paper's random-subspace joint-ranks test (no PyTorch)."
    )
    parser.add_argument("--cifar-root", type=str, required=True,
                        help="Path to extracted cifar-10-batches-py directory")
    parser.add_argument("--c10c-root", type=str, required=True,
                        help="Path to extracted CIFAR-10-C directory")
    parser.add_argument("--cache-dir", type=str,
                        default="./pixel_cache_cifar10c")
    parser.add_argument("--save-csv", type=str,
                        default="./cifar10c_power_rank_subspaces.csv")
    parser.add_argument("--save-json", type=str, default="")

    parser.add_argument("--corruptions", type=str, default="gaussian_noise")
    parser.add_argument("--severities", type=str, default="1,2,3,4,5")
    parser.add_argument("--methods", type=str, default="joint_subspaces",
                        help="comma-separated subset of: joint_subspaces,swd,hotelling,mmd,c2st")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)

    # Single-size mode
    parser.add_argument("--n-ref", type=int, default=None,
                        help="Single reference sample size (legacy / single-run mode).")
    parser.add_argument("--n-tar", type=int, default=None,
                        help="Single target sample size (legacy / single-run mode).")

    # Sweep mode
    parser.add_argument("--n-refs", type=str, default="",
                        help="Comma-separated list of reference sample sizes, e.g. 100,200,500,1000,2000")
    parser.add_argument("--n-tars", type=str, default="",
                        help="Comma-separated list of target sample sizes, e.g. 100,200,500,1000,2000")
    parser.add_argument("--nm-mode", type=str, default="pairwise",
                        choices=["pairwise", "grid"],
                        help="How to combine n-refs and n-tars.")

    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--K-ref", type=int, default=4)
    parser.add_argument("--d-sub", type=int, default=2)
    parser.add_argument("--L-subspaces", type=int, default=20)
    parser.add_argument("--B-joint", type=int, default=300)

    parser.add_argument("--swd-L", type=int, default=256)
    parser.add_argument("--B-swd", type=int, default=300)

    parser.add_argument("--hotelling-ridge", type=float, default=1e-2)
    parser.add_argument("--B-hotelling", type=int, default=300)

    parser.add_argument("--B-mmd", type=int, default=300)

    parser.add_argument("--B-c2st", type=int, default=300)
    parser.add_argument("--c2st-split", type=float, default=0.7)

    args = parser.parse_args()
    set_deterministic(args.seed)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    valid_methods = {"joint_subspaces", "swd", "hotelling", "mmd", "c2st"}
    bad = [m for m in methods if m not in valid_methods]
    if bad:
        raise ValueError(
            f"Unknown methods: {bad}. Valid options: {sorted(valid_methods)}")

    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    severities = [int(s.strip())
                  for s in args.severities.split(",") if s.strip()]

    # Resolve sample-size sweeps
    n_refs = _parse_int_list(args.n_refs) if args.n_refs else []
    n_tars = _parse_int_list(args.n_tars) if args.n_tars else []

    if not n_refs:
        n_refs = [args.n_ref if args.n_ref is not None else 2000]
    if not n_tars:
        if args.n_refs and not args.n_tars:
            n_tars = list(n_refs)
        else:
            n_tars = [args.n_tar if args.n_tar is not None else 2000]

    if args.nm_mode == "pairwise":
        if len(n_refs) == len(n_tars):
            nm_pairs = list(zip(n_refs, n_tars))
        elif len(n_refs) == 1:
            nm_pairs = [(n_refs[0], nt) for nt in n_tars]
        elif len(n_tars) == 1:
            nm_pairs = [(nr, n_tars[0]) for nr in n_refs]
        else:
            raise ValueError(
                "In pairwise mode, --n-refs and --n-tars must have the same length, "
                "or one of them must have length 1.\n"
                f"Got n_refs={n_refs}, n_tars={n_tars}"
            )
    else:
        nm_pairs = [(nr, nt) for nr in n_refs for nt in n_tars]

    os.makedirs(os.path.dirname(os.path.abspath(args.save_csv)), exist_ok=True)

    with open(args.save_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "method", "corruption", "severity",
            "n_ref", "n_tar", "alpha", "standardize",
            "K_ref", "d_sub", "L_subspaces", "B_joint",
            "swd_L", "B_swd",
            "hotelling_ridge", "B_hotelling",
            "B_mmd",
            "c2st_split", "B_c2st",
            "trials", "power", "ci_lo", "ci_hi", "median_p", "avg_time_sec"
        ])

    per_setting = []

    X_ref_all = load_or_cache_cifar10_test_pixels(
        args.cifar_root, args.cache_dir)
    print(f"Loaded clean CIFAR-10 reference pool: {X_ref_all.shape}")

    for corr in corruptions:
        for sev in severities:
            print(f"\n=== corruption='{corr}', severity={sev} ===")
            X_tar_all = load_or_cache_cifar10c_pixels(
                args.c10c_root, args.cache_dir, corr, sev)
            print(f"Loaded corrupted target pool: {X_tar_all.shape}")

            for n_ref, n_tar in nm_pairs:
                print(f"   n_ref={n_ref}, n_tar={n_tar}")

                t0 = time.time()
                stats = estimate_power(
                    X_ref_all,
                    X_tar_all,
                    trials=args.trials,
                    alpha=args.alpha,
                    n_ref=n_ref,
                    n_tar=n_tar,
                    standardize=args.standardize,
                    methods=methods,
                    K_ref=args.K_ref,
                    d_sub=args.d_sub,
                    L_subspaces=args.L_subspaces,
                    B_joint=args.B_joint,
                    B_swd=args.B_swd,
                    B_hotelling=args.B_hotelling,
                    B_mmd=args.B_mmd,
                    B_c2st=args.B_c2st,
                    swd_L=args.swd_L,
                    hotelling_ridge=args.hotelling_ridge,
                    c2st_split=args.c2st_split,
                    seed=args.seed,
                )
                elapsed = time.time() - t0

                with open(args.save_csv, "a", newline="") as f:
                    w = csv.writer(f)
                    for m, s in stats.items():
                        w.writerow([
                            m, corr, sev,
                            n_ref, n_tar, args.alpha, int(args.standardize),
                            args.K_ref, args.d_sub, args.L_subspaces, args.B_joint,
                            args.swd_L, args.B_swd,
                            args.hotelling_ridge, args.B_hotelling,
                            args.B_mmd,
                            args.c2st_split, args.B_c2st,
                            args.trials,
                            f"{s['power']:.6f}",
                            f"{s['ci_lo']:.6f}",
                            f"{s['ci_hi']:.6f}",
                            f"{s['median_p']:.6f}",
                            f"{s['avg_time_sec']:.6f}",
                        ])

                for m, s in stats.items():
                    per_setting.append({
                        "method": m,
                        "corruption": corr,
                        "severity": sev,
                        "n_ref": n_ref,
                        "n_tar": n_tar,
                        "alpha": args.alpha,
                        "standardize": bool(args.standardize),
                        "K_ref": args.K_ref,
                        "d_sub": args.d_sub,
                        "L_subspaces": args.L_subspaces,
                        "B_joint": args.B_joint,
                        "swd_L": args.swd_L,
                        "B_swd": args.B_swd,
                        "hotelling_ridge": args.hotelling_ridge,
                        "B_hotelling": args.B_hotelling,
                        "B_mmd": args.B_mmd,
                        "c2st_split": args.c2st_split,
                        "B_c2st": args.B_c2st,
                        "power": s["power"],
                        "ci_lo": s["ci_lo"],
                        "ci_hi": s["ci_hi"],
                        "median_p": s["median_p"],
                        "avg_time_sec": s["avg_time_sec"],
                        "elapsed_sec_total_setting": elapsed,
                    })

                print("-> Results:")
                for m, s in stats.items():
                    print(
                        f"   {m:16s} power={s['power']:.3f} "
                        f"(95% CI {s['ci_lo']:.3f}-{s['ci_hi']:.3f}), "
                        f"median p={s['median_p']:.3g}, avg time={s['avg_time_sec']:.3f}s"
                    )

    if args.save_json:
        with open(args.save_json, "w") as jf:
            json.dump(per_setting, jf, indent=2)
        print(f"\nSaved JSON to: {args.save_json}")

    print(f"Saved CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()
