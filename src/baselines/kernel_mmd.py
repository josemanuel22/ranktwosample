# src/baselines/kernel_mmd.py
# -------------------------------------------------------------------
# Kernel MMD (two-sample) — importable API + class wrapper
#   • Kernels: RBF (multi-bandwidth with median heuristic), linear, polynomial
#   • Estimators: unbiased (U-stat) | biased (includes diagonal)
#   • Permutation-calibrated test with antithetic pairs & early stopping
#   • Simple NumPy-only implementation (parallelism can be external)
# -------------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import numpy as np

Array = np.ndarray

__all__ = [
    "kernel_mmd2",
    "kernel_mmd_permutation_test",
    "KernelMMDResult",
    "KernelMMDTest",
    "estimate_type1",
    "estimate_power",
]

# ========================= Small utilities =========================


def _ndtri(p: float) -> float:
    """Fast approximation to Φ^{-1}(p) (Hastings)."""
    if p <= 0.0:
        return -1e300
    if p >= 1.0:
        return 1e300
    a = [
        -39.6968302866538,
        220.946098424521,
        -275.928510446969,
        138.357751867269,
        -30.6647980661472,
        2.50662827745924,
    ]
    b = [
        -54.4760987982241,
        161.585836858041,
        -155.698979859887,
        66.8013118877197,
        -13.2806815528857,
    ]
    c = [
        -0.00778489400243029,
        -0.322396458041136,
        -2.40075827716184,
        -2.54973253934373,
        4.37466414146497,
        2.93816398269878,
    ]
    d = [0.00778469570904146, 0.32246712907004,
         2.445134137143, 3.75440866190742]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _wilson_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = _ndtri(1 - delta / 2.0)
    phat = k / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt(
        max(0.0, phat * (1 - phat) / n + (z * z) / (4 * n * n))
    )
    return max(0.0, center - half), min(1.0, center + half)


def _hoeffding_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    r = math.sqrt(max(0.0, math.log(2.0 / max(1e-16, delta))) / (2.0 * n))
    return max(0.0, phat - r), min(1.0, phat + r)


def _se_binom(p: float, R: int) -> float:
    return float(np.sqrt(max(p * (1 - p), 1e-12) / max(1, R)))


# ========================= Kernels & helpers =========================


def _pairwise_sq_dists(X: Array, Y: Array) -> Array:
    """Return matrix of ||x - y||^2."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (X @ Y.T), 0.0)


def _gram_rbf(
    X: Array, Y: Array, sigmas: Iterable[float], weights: Optional[Iterable[float]] = None
) -> Array:
    D2 = _pairwise_sq_dists(X, Y)
    sigmas = np.asarray(list(sigmas), dtype=float)
    if weights is None:
        weights = np.ones_like(sigmas) / max(1, len(sigmas))
    else:
        weights = np.asarray(list(weights), dtype=float)
        s = np.sum(weights)
        weights = weights / (s if s != 0 else 1.0)
    K = np.zeros_like(D2)
    for w, s in zip(weights, sigmas):
        if s > 0:
            K += w * np.exp(-D2 / (2.0 * s * s))
    return K


def _gram_linear(X: Array, Y: Array) -> Array:
    return X @ Y.T


def _gram_poly(X: Array, Y: Array, degree: int = 3, coef0: float = 1.0) -> Array:
    return (X @ Y.T + float(coef0)) ** int(degree)


def _median_bandwidths(
    Z: Array, *, subsample: int = 1000, num_scales: int = 5, base: float = 2.0
) -> np.ndarray:
    """Median heuristic bandwidth grid around median pairwise distance."""
    n = Z.shape[0]
    m = min(n, subsample)
    idx = np.random.default_rng(0).choice(n, size=m, replace=False)
    Zs = Z[idx]
    D2 = _pairwise_sq_dists(Zs, Zs)
    tri = D2[np.triu_indices(m, k=1)]
    med = float(np.sqrt(np.median(tri))) if tri.size else 1.0
    half = num_scales // 2
    exps = np.arange(-half, half + 1)
    sigmas = med * (base ** exps)
    return sigmas


def _mmd2_from_grams(
    Kxx: Array, Kyy: Array, Kxy: Array, *, estimator: str = "unbiased"
) -> float:
    N = Kxx.shape[0]
    K = Kyy.shape[0]
    if estimator == "unbiased":
        sx = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / max(1, N * (N - 1))
        sy = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / max(1, K * (K - 1))
        sxy = np.sum(Kxy) / max(1, N * K)
        return float(sx + sy - 2.0 * sxy)
    elif estimator == "biased":
        sx = np.sum(Kxx) / max(1, N * N)
        sy = np.sum(Kyy) / max(1, K * K)
        sxy = np.sum(Kxy) / max(1, N * K)
        return float(sx + sy - 2.0 * sxy)
    raise ValueError("estimator must be 'unbiased' or 'biased'.")


# ========================= Public statistic =========================


def kernel_mmd2(
    X: Array,
    Y: Array,
    *,
    kernel: str = "rbf",  # 'rbf' | 'linear' | 'poly'
    estimator: str = "unbiased",  # 'unbiased' | 'biased'
    # RBF params
    rbf_sigmas: Optional[Union[float, Iterable[float]]] = None,
    rbf_weights: Optional[Iterable[float]] = None,
    rbf_use_median: bool = True,
    rbf_num_scales: int = 5,
    rbf_base: float = 2.0,
    rbf_median_subsample: int = 1000,
    # Poly params
    poly_degree: int = 3,
    poly_coef0: float = 1.0,
) -> Tuple[float, Dict]:
    """
    Return (MMD^2, extras) for the selected kernel and estimator.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError(
            "X and Y must be 2D arrays with the same number of columns.")

    if kernel == "rbf":
        if rbf_sigmas is None:
            if rbf_use_median:
                Z = np.vstack([X, Y])
                rbf_sigmas = _median_bandwidths(
                    Z,
                    subsample=rbf_median_subsample,
                    num_scales=rbf_num_scales,
                    base=rbf_base,
                )
            else:
                raise ValueError(
                    "Provide rbf_sigmas or enable rbf_use_median=True.")
        elif isinstance(rbf_sigmas, (int, float, np.floating)):
            rbf_sigmas = [float(rbf_sigmas)]
        Kxx = _gram_rbf(X, X, rbf_sigmas, rbf_weights)
        Kyy = _gram_rbf(Y, Y, rbf_sigmas, rbf_weights)
        Kxy = _gram_rbf(X, Y, rbf_sigmas, rbf_weights)
        stat = _mmd2_from_grams(Kxx, Kyy, Kxy, estimator=estimator)
        extras = {
            "kernel": "rbf",
            "sigmas": np.asarray(list(rbf_sigmas), float),
            "weights": None if rbf_weights is None else np.asarray(list(rbf_weights), float),
            "estimator": estimator,
        }

    elif kernel == "linear":
        Kxx = _gram_linear(X, X)
        Kyy = _gram_linear(Y, Y)
        Kxy = _gram_linear(X, Y)
        stat = _mmd2_from_grams(Kxx, Kyy, Kxy, estimator=estimator)
        extras = {"kernel": "linear", "estimator": estimator}

    elif kernel == "poly":
        Kxx = _gram_poly(X, X, degree=poly_degree, coef0=poly_coef0)
        Kyy = _gram_poly(Y, Y, degree=poly_degree, coef0=poly_coef0)
        Kxy = _gram_poly(X, Y, degree=poly_degree, coef0=poly_coef0)
        stat = _mmd2_from_grams(Kxx, Kyy, Kxy, estimator=estimator)
        extras = {
            "kernel": "poly",
            "degree": int(poly_degree),
            "coef0": float(poly_coef0),
            "estimator": estimator,
        }

    else:
        raise ValueError("kernel must be 'rbf', 'linear', or 'poly'.")

    return stat, extras


# ========================= Permutation test =========================


@dataclass
class KernelMMDResult:
    stat_obs: float
    p_perm: float
    p_mid: float
    reject: bool
    ge: int
    gt: int
    eq: int
    perms_used: int
    extras: Dict


def kernel_mmd_permutation_test(
    X: Array,
    Y: Array,
    *,
    B: int = 1000,
    alpha: float = 0.05,
    decision: str = "pvalue",  # "pvalue" | "midp" | "randomized"
    early_stop: str = "wilson",  # "wilson" | "hoeffding" | "none"
    delta_ci: float = 1e-2,
    min_b_check: int = 100,
    chunk: int = 256,
    antithetic: bool = True,
    # kernel options forwarded to kernel_mmd2
    kernel: str = "rbf",
    estimator: str = "unbiased",
    rbf_sigmas: Optional[Union[float, Iterable[float]]] = None,
    rbf_weights: Optional[Iterable[float]] = None,
    rbf_use_median: bool = True,
    rbf_num_scales: int = 5,
    rbf_base: float = 2.0,
    rbf_median_subsample: int = 1000,
    poly_degree: int = 3,
    poly_coef0: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> KernelMMDResult:
    """Conditional permutation test for Kernel MMD."""
    if rng is None:
        rng = np.random.default_rng(123)

    T_obs, extras = kernel_mmd2(
        X,
        Y,
        kernel=kernel,
        estimator=estimator,
        rbf_sigmas=rbf_sigmas,
        rbf_weights=rbf_weights,
        rbf_use_median=rbf_use_median,
        rbf_num_scales=rbf_num_scales,
        rbf_base=rbf_base,
        rbf_median_subsample=rbf_median_subsample,
        poly_degree=poly_degree,
        poly_coef0=poly_coef0,
    )

    Z = np.vstack([np.asarray(X, float), np.asarray(Y, float)])
    N = X.shape[0]
    K = Y.shape[0]
    n_tot = N + K

    gt = 0
    eq = 0
    ge = 0
    perms_used = 0
    b = 0
    C = max(1, int(chunk))
    allow_early = early_stop in ("wilson", "hoeffding")

    def _should_stop(ge_: int, b_: int) -> bool:
        if not allow_early or b_ < min_b_check:
            return False
        if early_stop == "wilson":
            lo, hi = _wilson_ci(ge_, b_, delta_ci)
        else:
            lo, hi = _hoeffding_ci(ge_, b_, delta_ci)
        return (hi <= alpha) or (lo > alpha)

    while b < B:
        n_this = min(C, B - b)

        if antithetic:
            pairs = n_this // 2
            rem = n_this % 2

            for _ in range(pairs):
                idx = rng.permutation(n_tot)
                idxR = idx[::-1]

                # π
                Xb, Yb = Z[idx[:N]], Z[idx[N:]]
                Tb, _ = kernel_mmd2(
                    Xb,
                    Yb,
                    kernel=kernel,
                    estimator=estimator,
                    rbf_sigmas=rbf_sigmas,
                    rbf_weights=rbf_weights,
                    rbf_use_median=rbf_use_median,
                    rbf_num_scales=rbf_num_scales,
                    rbf_base=rbf_base,
                    rbf_median_subsample=rbf_median_subsample,
                    poly_degree=poly_degree,
                    poly_coef0=poly_coef0,
                )
                if Tb > T_obs:
                    gt += 1
                elif Tb == T_obs:
                    eq += 1
                ge = gt + eq
                b += 1
                perms_used = b
                if _should_stop(ge, b) or b >= B:
                    break

                # π^R
                Xb, Yb = Z[idxR[:N]], Z[idxR[N:]]
                Tb, _ = kernel_mmd2(
                    Xb,
                    Yb,
                    kernel=kernel,
                    estimator=estimator,
                    rbf_sigmas=rbf_sigmas,
                    rbf_weights=rbf_weights,
                    rbf_use_median=rbf_use_median,
                    rbf_num_scales=rbf_num_scales,
                    rbf_base=rbf_base,
                    rbf_median_subsample=rbf_median_subsample,
                    poly_degree=poly_degree,
                    poly_coef0=poly_coef0,
                )
                if Tb > T_obs:
                    gt += 1
                elif Tb == T_obs:
                    eq += 1
                ge = gt + eq
                b += 1
                perms_used = b
                if _should_stop(ge, b):
                    break

            if _should_stop(ge, b):
                break

            if rem and b < B:
                idx = rng.permutation(n_tot)
                Xb, Yb = Z[idx[:N]], Z[idx[N:]]
                Tb, _ = kernel_mmd2(
                    Xb,
                    Yb,
                    kernel=kernel,
                    estimator=estimator,
                    rbf_sigmas=rbf_sigmas,
                    rbf_weights=rbf_weights,
                    rbf_use_median=rbf_use_median,
                    rbf_num_scales=rbf_num_scales,
                    rbf_base=rbf_base,
                    rbf_median_subsample=rbf_median_subsample,
                    poly_degree=poly_degree,
                    poly_coef0=poly_coef0,
                )
                if Tb > T_obs:
                    gt += 1
                elif Tb == T_obs:
                    eq += 1
                ge = gt + eq
                b += 1
                perms_used = b
                if _should_stop(ge, b):
                    break

        else:
            for _ in range(n_this):
                idx = rng.permutation(n_tot)
                Xb, Yb = Z[idx[:N]], Z[idx[N:]]
                Tb, _ = kernel_mmd2(
                    Xb,
                    Yb,
                    kernel=kernel,
                    estimator=estimator,
                    rbf_sigmas=rbf_sigmas,
                    rbf_weights=rbf_weights,
                    rbf_use_median=rbf_use_median,
                    rbf_num_scales=rbf_num_scales,
                    rbf_base=rbf_base,
                    rbf_median_subsample=rbf_median_subsample,
                    poly_degree=poly_degree,
                    poly_coef0=poly_coef0,
                )
                if Tb > T_obs:
                    gt += 1
                elif Tb == T_obs:
                    eq += 1
                ge = gt + eq
                b += 1
                perms_used = b
                if _should_stop(ge, b):
                    break

        if _should_stop(ge, b):
            break

    b = max(1, perms_used)
    p_perm = (1 + ge) / (b + 1)
    p_mid = (gt + 0.5 * eq) / b

    if decision == "pvalue":
        reject = p_perm <= alpha
    elif decision == "midp":
        reject = p_mid <= alpha
    elif decision == "randomized":
        p_lower = (1 + gt) / (b + 1)
        if p_lower > alpha:
            reject = False
        elif eq == 0:
            reject = p_perm <= alpha
        else:
            omega = (alpha - p_lower) / (eq / (b + 1))
            omega = float(np.clip(omega, 0.0, 1.0))
            reject = np.random.default_rng().uniform() <= omega
    else:
        reject = p_perm <= alpha

    return KernelMMDResult(
        stat_obs=float(T_obs),
        p_perm=float(p_perm),
        p_mid=float(p_mid),
        reject=bool(reject),
        ge=int(ge),
        gt=int(gt),
        eq=int(eq),
        perms_used=int(b),
        extras=extras,
    )


# ========================= Monte Carlo wrappers =========================


def estimate_type1(
    gen_null: Callable[[np.random.Generator], Tuple[Array, Array]],
    *,
    R: int = 200,
    test_kwargs: Optional[dict] = None,
    seed: int = 2026,
) -> Dict[str, float]:
    """Estimate Type-I error under H0 via Monte Carlo."""
    test_kwargs = dict(test_kwargs or {})
    ss = np.random.SeedSequence(seed)
    rej = 0
    perms = []
    for s in ss.spawn(R):
        rng = np.random.default_rng(s.entropy)
        X, Y = gen_null(rng)
        out = kernel_mmd_permutation_test(X, Y, rng=rng, **test_kwargs)
        rej += int(out.reject)
        perms.append(out.perms_used)
    p = rej / R
    se = _se_binom(p, R)
    return {"type1_hat": float(p), "se": float(se), "avg_perms": float(np.mean(perms))}


def estimate_power(
    gen_alt: Callable[[np.random.Generator], Tuple[Array, Array]],
    *,
    R: int = 200,
    test_kwargs: Optional[dict] = None,
    seed: int = 2027,
) -> Dict[str, float]:
    """Estimate power under H1 via Monte Carlo."""
    test_kwargs = dict(test_kwargs or {})
    ss = np.random.SeedSequence(seed)
    rej = 0
    perms = []
    for s in ss.spawn(R):
        rng = np.random.default_rng(s.entropy)
        X, Y = gen_alt(rng)
        out = kernel_mmd_permutation_test(X, Y, rng=rng, **test_kwargs)
        rej += int(out.reject)
        perms.append(out.perms_used)
    p = rej / R
    se = _se_binom(p, R)
    return {"power_hat": float(p), "se": float(se), "avg_perms": float(np.mean(perms))}


# ========================= OO Wrapper =========================


class KernelMMDTest:
    """
    Uniform class API so your runner can do:

        from baselines.kernel_mmd import KernelMMDTest
        test = KernelMMDTest(B=1000, alpha=0.05, kernel="rbf")
        out = test.run(X, Y)
    """

    def __init__(
        self,
        *,
        B: int = 1000,
        alpha: float = 0.05,
        decision: str = "pvalue",
        early_stop: str = "wilson",
        delta_ci: float = 1e-2,
        min_b_check: int = 100,
        chunk: int = 256,
        antithetic: bool = True,
        # kernel options
        kernel: str = "rbf",
        estimator: str = "unbiased",
        rbf_sigmas: Optional[Union[float, Iterable[float]]] = None,
        rbf_weights: Optional[Iterable[float]] = None,
        rbf_use_median: bool = True,
        rbf_num_scales: int = 5,
        rbf_base: float = 2.0,
        rbf_median_subsample: int = 1000,
        poly_degree: int = 3,
        poly_coef0: float = 1.0,
        seed: int = 2027,
    ) -> None:
        self.B = int(B)
        self.alpha = float(alpha)
        self.decision = str(decision)
        self.early_stop = str(early_stop)
        self.delta_ci = float(delta_ci)
        self.min_b_check = int(min_b_check)
        self.chunk = int(chunk)
        self.antithetic = bool(antithetic)

        self.kernel = str(kernel)
        self.estimator = str(estimator)
        self.rbf_sigmas = rbf_sigmas
        self.rbf_weights = rbf_weights
        self.rbf_use_median = bool(rbf_use_median)
        self.rbf_num_scales = int(rbf_num_scales)
        self.rbf_base = float(rbf_base)
        self.rbf_median_subsample = int(rbf_median_subsample)
        self.poly_degree = int(poly_degree)
        self.poly_coef0 = float(poly_coef0)

        self.seed = int(seed)

    def run(self, X: Array, Y: Array) -> KernelMMDResult:
        rng = np.random.default_rng(self.seed)
        return kernel_mmd_permutation_test(
            X,
            Y,
            B=self.B,
            alpha=self.alpha,
            decision=self.decision,
            early_stop=self.early_stop,
            delta_ci=self.delta_ci,
            min_b_check=self.min_b_check,
            chunk=self.chunk,
            antithetic=self.antithetic,
            kernel=self.kernel,
            estimator=self.estimator,
            rbf_sigmas=self.rbf_sigmas,
            rbf_weights=self.rbf_weights,
            rbf_use_median=self.rbf_use_median,
            rbf_num_scales=self.rbf_num_scales,
            rbf_base=self.rbf_base,
            rbf_median_subsample=self.rbf_median_subsample,
            poly_degree=self.poly_degree,
            poly_coef0=self.poly_coef0,
            rng=rng,
        )

    # Convenience wrappers (optional)
    def estimate_type1(
        self,
        gen_null: Callable[[np.random.Generator], Tuple[Array, Array]],
        *,
        R: int = 200,
        seed: int = 4041,
    ) -> dict:
        return estimate_type1(
            gen_null,
            R=R,
            test_kwargs=dict(
                B=self.B,
                alpha=self.alpha,
                decision=self.decision,
                early_stop=self.early_stop,
                delta_ci=self.delta_ci,
                min_b_check=self.min_b_check,
                chunk=self.chunk,
                antithetic=self.antithetic,
                kernel=self.kernel,
                estimator=self.estimator,
                rbf_sigmas=self.rbf_sigmas,
                rbf_weights=self.rbf_weights,
                rbf_use_median=self.rbf_use_median,
                rbf_num_scales=self.rbf_num_scales,
                rbf_base=self.rbf_base,
                rbf_median_subsample=self.rbf_median_subsample,
                poly_degree=self.poly_degree,
                poly_coef0=self.poly_coef0,
            ),
            seed=seed,
        )

    def estimate_power(
        self,
        gen_alt: Callable[[np.random.Generator], Tuple[Array, Array]],
        *,
        R: int = 200,
        seed: int = 4042,
    ) -> dict:
        return estimate_power(
            gen_alt,
            R=R,
            test_kwargs=dict(
                B=self.B,
                alpha=self.alpha,
                decision=self.decision,
                early_stop=self.early_stop,
                delta_ci=self.delta_ci,
                min_b_check=self.min_b_check,
                chunk=self.chunk,
                antithetic=self.antithetic,
                kernel=self.kernel,
                estimator=self.estimator,
                rbf_sigmas=self.rbf_sigmas,
                rbf_weights=self.rbf_weights,
                rbf_use_median=self.rbf_use_median,
                rbf_num_scales=self.rbf_num_scales,
                rbf_base=self.rbf_base,
                rbf_median_subsample=self.rbf_median_subsample,
                poly_degree=self.poly_degree,
                poly_coef0=self.poly_coef0,
            ),
            seed=seed,
        )
