#!/usr/bin/env python3
# -------------------------------------------------------------------
# power_rank_subspaces.py
# Estimate power for the Subspace-Sliced non-symmetric rank χ² / G-test
#
# Usage (example):
#   PYTHONPATH=src python src/scripts/power_rank_subspaces.py \
#       --scenario shift --d 8 --N 600 --K 600 --delta 0.5 \
#       --L 64 --k-dim 2 --K-ref 64 --B 500 --R 200 --engine auto
#
# Fixed:
#   * --dims is optional (no error when omitted)
#   * add --pca to pick slices via PCA loadings (top-|loading| coords)
#   * no-duplicate random slices by default (can be disabled)
# -------------------------------------------------------------------
from __future__ import annotations
import argparse
import math
import os
import numpy as np

from rank_two_sample import RankGTestConfig  # re-exported config
from rank_two_sample_subspaces import RankGTestSubspaces, SubspaceOpts


# ---------- simple scenario generators (NumPy) ----------
def gen_shift(rng, d, N, K, delta=0.3, direction="e1"):
    if direction == "random":
        u = rng.normal(size=d)
        u = u / (np.linalg.norm(u) + 1e-12)
    else:
        u = np.zeros(d)
        u[0] = 1.0
    X = rng.normal(loc=delta * u, scale=1.0, size=(N, d))
    Y = rng.normal(loc=0.0, scale=1.0, size=(K, d))
    return X, Y


def gen_scale(rng, d, N, K, scale=1.5):
    X = rng.normal(loc=0.0, scale=scale, size=(N, d))
    Y = rng.normal(loc=0.0, scale=1.0, size=(K, d))
    return X, Y


def gen_corr(rng, d, N, K, rho_x=0.7, rho_y=0.0):
    def corr_mat(d, rho):
        S = np.full((d, d), rho, dtype=float)
        np.fill_diagonal(S, 1.0)
        w, V = np.linalg.eigh(S)
        w = np.clip(w, 1e-12, None)
        return (V * w) @ V.T
    Sx = corr_mat(d, rho_x)
    Sy = corr_mat(d, rho_y)
    X = rng.multivariate_normal(mean=np.zeros(d), cov=Sx, size=N)
    Y = rng.multivariate_normal(mean=np.zeros(d), cov=Sy, size=K)
    return X, Y


def gen_tdf(rng, d, N, K, nu_x=5, nu_y=10):
    Zx = rng.normal(size=(N, d))
    Ux = rng.chisquare(nu_x, size=N)
    Zy = rng.normal(size=(K, d))
    Uy = rng.chisquare(nu_y, size=K)
    X = Zx / np.sqrt(Ux[:, None] / float(nu_x))
    Y = Zy / np.sqrt(Uy[:, None] / float(nu_y))
    return X, Y


def gen_mixture(rng, d, N, K, eps=0.1, delta=0.5):
    n1 = int(rng.binomial(N, eps))
    n0 = N - n1
    X0 = rng.normal(size=(n0, d))
    mu1 = np.zeros(d)
    mu1[0] = delta
    X1 = rng.normal(loc=mu1, size=(n1, d))
    X = np.vstack([X0, X1])
    X = X[rng.permutation(N)]
    Y = rng.normal(size=(K, d))
    return X, Y


SCENARIOS = {
    "shift": gen_shift,
    "scale": gen_scale,
    "corr": gen_corr,
    "tdf": gen_tdf,
    "mixture": gen_mixture,
}


def se_binom(p: float, R: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(1, R))


def _parse_dims_string(dims_str: str | None):
    """
    Accepts:
      - None        -> returns None
      - "3"         -> [(3,)]
      - "0,2"       -> [(0,2)]
      - "0,1;2,3"   -> [(0,1), (2,3)]
      - "4;5;6"     -> [(4,), (5,), (6,)]
    """
    if dims_str is None or dims_str == "":
        return None
    dims_str = dims_str.strip()
    out = []
    for group in dims_str.split(";"):
        group = group.strip()
        if not group:
            continue
        parts = [p.strip() for p in group.split(",") if p.strip() != ""]
        tup = tuple(int(x) for x in parts)
        out.append(tup)
    return out if out else None


def main():
    ap = argparse.ArgumentParser(
        description="Power for subspace-sliced rank χ² / G-test")
    # data / scenario
    ap.add_argument(
        "--scenario", choices=list(SCENARIOS.keys()), default="shift")
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--K", type=int, default=1000)
    # for shift/mixture
    ap.add_argument("--delta", type=float, default=0.3)
    ap.add_argument("--scale", type=float, default=1.5)       # for scale
    ap.add_argument("--rho-x", type=float, default=0.7)       # for corr
    ap.add_argument("--rho-y", type=float, default=0.0)       # for corr
    ap.add_argument("--nu-x", type=int, default=5)            # for tdf
    ap.add_argument("--nu-y", type=int, default=10)           # for tdf
    ap.add_argument("--eps", type=float, default=0.1)         # for mixture
    ap.add_argument("--direction", choices=["e1", "random"], default="e1")

    # subspace options
    ap.add_argument("--L", type=int, default=32,
                    help="number of subspaces/slices")
    ap.add_argument("--k-dim", type=int, default=1,
                    dest="k_dim", help="dimension per slice")
    ap.add_argument("--dims", type=str, default=None,
                    help='Fixed slices string, e.g. "0,1;2,3" or "0;1;2" or "3"')
    ap.add_argument("--pca", action="store_true", default=False,
                    help="Select slices using PCA loadings (ignored if --dims provided)")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false",
                    help="Allow duplicated subspace slices when sampling randomly")
    ap.set_defaults(dedup=True)

    # core test params
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--K-ref", type=int, default=4, dest="K_ref")
    ap.add_argument(
        "--engine", choices=["auto", "searchsorted", "prefixsum"], default="auto")
    ap.add_argument(
        "--tie", choices=["jitter", "right", "left"], default="jitter")
    ap.add_argument("--alpha0", type=float, default=0.1)
    ap.add_argument(
        "--decision", choices=["pvalue", "midp", "randomized"], default="midp")
    ap.add_argument("--kref-switch", type=int, default=64, dest="kref_switch")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--antithetic", action="store_true", default=True)
    ap.add_argument("--no-antithetic", action="store_false", dest="antithetic")
    ap.add_argument(
        "--stop-ci", choices=["wilson", "hoeffding"], default="wilson")
    ap.add_argument("--delta-ci", type=float, default=1e-2)
    ap.add_argument("--min-b-check", type=int, default=100)
    ap.add_argument("--stat", choices=["gtest", "chi2"], default="gtest")
    ap.add_argument("--jitter-scale", type=float, default=1e-12)
    ap.add_argument("--gpu", choices=["off", "auto", "on"], default="off")
    ap.add_argument("--device", type=int, default=0)

    # Monte-Carlo
    ap.add_argument("--R", type=int, default=200, help="repetitions for power")
    ap.add_argument("--seed", type=int, default=2026)

    args = ap.parse_args()

    # Build config + subspace options
    cfg = RankGTestConfig(
        alpha=args.alpha,
        K_ref=args.K_ref,
        engine=args.engine,
        tie=args.tie,
        alpha0=args.alpha0,
        decision=args.decision,
        B=args.B,
        kref_switch=args.kref_switch,
        chunk=args.chunk,
        antithetic=args.antithetic,
        stop_ci=args.stop_ci,
        delta_ci=args.delta_ci,
        min_b_check=args.min_b_check,
        use_fullY_for_HY=True,
        stat=args.stat,
        jitter_scale=args.jitter_scale,
        gpu=args.gpu,
        device=args.device,
    )

    dims_list = _parse_dims_string(args.dims)
    subs = SubspaceOpts(
        L=args.L,
        k_dim=args.k_dim,
        dims_list=dims_list,
        dedup=args.dedup,
        pca=(args.pca and dims_list is None),
        pca_center=True,
    )
    runner = RankGTestSubspaces(cfg, subs)

    # Scenario
    gen = SCENARIOS[args.scenario]
    ss = np.random.SeedSequence(args.seed)
    seeds = [int(s.entropy) for s in ss.spawn(args.R)]

    rejs, times, used = [], [], []
    for i, s in enumerate(seeds, 1):
        rng = np.random.default_rng(s)
        if args.scenario == "shift":
            X, Y = gen(rng, args.d, args.N, args.K,
                       delta=args.delta, direction=args.direction)
        elif args.scenario == "scale":
            X, Y = gen(rng, args.d, args.N, args.K, scale=args.scale)
        elif args.scenario == "corr":
            X, Y = gen(rng, args.d, args.N, args.K,
                       rho_x=args.rho_x, rho_y=args.rho_y)
        elif args.scenario == "tdf":
            X, Y = gen(rng, args.d, args.N, args.K,
                       nu_x=args.nu_x, nu_y=args.nu_y)
        elif args.scenario == "mixture":
            X, Y = gen(rng, args.d, args.N, args.K,
                       eps=args.eps, delta=args.delta)
        else:
            raise ValueError("unknown scenario")

        res = runner.test(X, Y, seed=s)
        rejs.append(int(res.reject))
        times.append(float(res.elapsed_ms))
        used.append(int(res.perms_used))

        if i % max(1, args.R // 10) == 0:
            print(f"[progress] {i}/{args.R}", end="\r")

    p_hat = float(np.mean(rejs))
    se = se_binom(p_hat, args.R)
    avg_ms = float(np.mean(times))
    avg_perms = float(np.mean(used))

    print("\n=== Subspace-sliced Rank χ² / G-test (power) ===")
    print(f"scenario={args.scenario} | d={args.d} N={args.N} K={args.K} | "
          f"L={args.L} k_dim={args.k_dim} K_ref={args.K_ref}")
    if dims_list is not None:
        print(f"fixed dims={dims_list}")
    else:
        print(f"pca={args.pca} dedup={args.dedup}")
    print(
        f"alpha={args.alpha} B={args.B} engine={args.engine} decision={args.decision}")
    print(
        f"power_hat={p_hat:.3f}  SE≈{se:.3f}  avg_ms/rep≈{avg_ms:.1f}  avg_perms≈{avg_perms:.1f}")


if __name__ == "__main__":
    # PYTHONPATH=src python src/scripts/power_rank_subspaces.py ...
    main()
