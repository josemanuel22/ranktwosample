# hotelling_t2.py
# -------------------------------------------------------------------
# Two-sample Hotelling's T^2 test with permutation calibration
#   • Statistic: T^2 = (N*K/(N+K)) (μX-μY)^T Σ_p^{-1} (μX-μY)
#     where Σ_p is the pooled covariance (or a shrinkage estimate).
#   • Calibration: conditional permutation test on labels.
#   • Early stopping: Wilson/Hoeffding CI on the permutation p-value.
#   • Robust inversion via pseudoinverse; optional shrinkage.
#
# Public API:
#   - hotelling_t2_stat(X, Y, shrinkage=..., ridge_lambda=..., ...)
#   - hotelling_t2_permutation_test(X, Y, B=..., early_stop=..., ...)
#   - estimate_type1(...), estimate_power(...)
#   - class HotellingT2(...).run(X, Y) -> HotellingT2Result
# -------------------------------------------------------------------
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Callable, List

import numpy as np

# optional: Ledoit–Wolf from scikit-learn if available
try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf
except Exception:
    _LedoitWolf = None

Array = np.ndarray

__all__ = [
    "HotellingT2Result",
    "hotelling_t2_stat",
    "hotelling_t2_permutation_test",
    "estimate_type1",
    "estimate_power",
    "HotellingT2",
]

# ---------------- utilities (shared with other baselines style) ----------------


def _ndtri(p: float) -> float:
    # Hastings approximation to Φ^{-1}(p)
    if p <= 0.0:
        return -1e300
    if p >= 1.0:
        return 1e300
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184,
         -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004,
         2.445134137143, 3.75440866190742]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
            ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)


def _wilson_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = _ndtri(1 - delta / 2.0)
    phat = k / n
    denom = 1 + (z*z) / n
    center = (phat + (z*z) / (2*n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, phat*(1-phat)/n + (z*z) / (4*n*n)))
    return max(0.0, center - half), min(1.0, center + half)


def _hoeffding_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    r = math.sqrt(max(0.0, math.log(2.0 / max(1e-16, delta))) / (2.0 * n))
    return max(0.0, phat - r), min(1.0, phat + r)


def _stops(ge: int, b: int, alpha: float, method: str, delta_ci: float) -> bool:
    if b <= 0 or method == "none":
        return False
    if method == "wilson":
        lo, hi = _wilson_ci(ge, b, delta_ci)
    elif method == "hoeffding":
        lo, hi = _hoeffding_ci(ge, b, delta_ci)
    else:
        return False
    return (hi <= alpha) or (lo > alpha)

# ---------------- Hotelling T^2 core ----------------


def _pooled_cov(X: Array, Y: Array) -> Array:
    """Unbiased pooled covariance (ddof=1)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, d = X.shape
    K, d2 = Y.shape
    if d2 != d:
        raise ValueError("X and Y must have same dimension.")
    if N < 2 or K < 2:
        raise ValueError(
            "Need at least 2 samples per group for pooled covariance.")
    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    Xc = X - mx
    Yc = Y - my
    Sx = (Xc.T @ Xc) / (N - 1)
    Sy = (Yc.T @ Yc) / (K - 1)
    Sp = ((N - 1) * Sx + (K - 1) * Sy) / (N + K - 2)
    return Sp


def _ledoit_wolf_cov(Z: Array) -> Array:
    """Ledoit–Wolf covariance on concatenated data Z (mean-centered)."""
    if _LedoitWolf is None:
        # Fallback: manual ridge to sample covariance
        Zc = Z - Z.mean(axis=0, keepdims=True)
        S = (Zc.T @ Zc) / max(1, Zc.shape[0] - 1)
        lam = 1e-3
        return S + lam * np.eye(S.shape[0])
    lw = _LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(Z)  # sklearn handles centering internally
    return lw.covariance_


def hotelling_t2_stat(
    X: Array, Y: Array, *,
    shrinkage: str = "none",          # "none" | "ledoitwolf" | "ridge"
    ridge_lambda: float = 1e-6,
) -> Tuple[float, Dict]:
    """
    Compute the two-sample Hotelling's T^2 statistic and extras.

    Returns:
        T2 (float), extras (dict with: d, N, K, delta, Sigma, invSigma, F_obs, df1, df2)
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    N, d = X.shape
    K, d2 = Y.shape
    if d2 != d:
        raise ValueError("X and Y must have same dimension.")

    mx = X.mean(axis=0)
    my = Y.mean(axis=0)
    delta = (mx - my)

    if shrinkage == "none":
        Sigma = _pooled_cov(X, Y)
    elif shrinkage == "ledoitwolf":
        Z = np.vstack([X, Y])
        Sigma = _ledoit_wolf_cov(Z)
    elif shrinkage == "ridge":
        Sigma = _pooled_cov(X, Y)
        Sigma = Sigma + float(ridge_lambda) * np.eye(d)
    else:
        raise ValueError("shrinkage must be 'none', 'ledoitwolf', or 'ridge'.")

    # Robust inverse (handles rank deficiency)
    invSigma = np.linalg.pinv(Sigma)

    coef = (N * K) / (N + K)
    T2 = float(coef * (delta @ invSigma @ delta))

    # Classical F-transform (for reference; permutation p-value is used)
    # Valid under normality and equal covariances:
    df1 = d
    df2 = max(1.0, N + K - d - 1.0)
    F_obs = float(((N + K - d - 1) / (d * (N + K - 2))) *
                  T2) if (N + K - 2) > 0 else float("nan")

    extras = {
        "d": d, "N": N, "K": K,
        "delta": delta,
        "Sigma": Sigma,
        "invSigma": invSigma,
        "F_obs": F_obs,
        "df1": float(df1),
        "df2": float(df2),
        "shrinkage": shrinkage,
        "ridge_lambda": float(ridge_lambda),
    }
    return T2, extras

# ---------------- permutation test ----------------


@dataclass
class HotellingT2Result:
    stat_obs: float
    p_perm: float
    p_mid: float
    reject: bool
    ge: int
    gt: int
    eq: int
    perms_used: int
    d: int
    N: int
    K: int
    F_obs: float
    df1: float
    df2: float
    shrinkage: str
    ridge_lambda: float


def hotelling_t2_permutation_test(
    X: Array, Y: Array, *,
    B: int = 1000,
    alpha: float = 0.05,
    decision: str = "pvalue",        # "pvalue" | "midp" | "randomized"
    early_stop: str = "wilson",      # "wilson" | "hoeffding" | "none"
    delta_ci: float = 1e-2,
    min_b_check: int = 100,
    chunk: int = 256,
    antithetic: bool = True,
    shrinkage: str = "none",
    ridge_lambda: float = 1e-6,
    rng: Optional[np.random.Generator] = None,
) -> HotellingT2Result:
    """
    Conditional permutation test for two-sample Hotelling's T^2.
    """
    if rng is None:
        rng = np.random.default_rng(123)

    # Observed statistic
    T_obs, extras = hotelling_t2_stat(
        X, Y, shrinkage=shrinkage, ridge_lambda=ridge_lambda)
    N = int(extras["N"])
    K = int(extras["K"])
    d = int(extras["d"])

    Z = np.vstack([np.asarray(X, float), np.asarray(Y, float)])  # (N+K, d)
    n_tot = N + K

    gt = eq = ge = 0
    perms_used = 0
    b = 0
    C = max(1, int(chunk))
    allow_early = early_stop in ("wilson", "hoeffding")

    def _t2_from_index(ixX: np.ndarray) -> float:
        Xb = Z[ixX]
        mask = np.ones(n_tot, dtype=bool)
        mask[ixX] = False
        Yb = Z[mask]
        Tb, _ = hotelling_t2_stat(
            Xb, Yb, shrinkage=shrinkage, ridge_lambda=ridge_lambda)
        return Tb

    while b < B:
        n_this = min(C, B - b)

        if antithetic:
            pairs = n_this // 2
            rem = n_this % 2

            for _ in range(pairs):
                perm = rng.permutation(n_tot)
                ixX = perm[:N]
                ixXR = perm[::-1][:N]

                # π
                Tb = _t2_from_index(ixX)
                if Tb > T_obs:
                    gt += 1
                elif Tb == T_obs:
                    eq += 1
                ge = gt + eq
                b += 1
                perms_used = b
                if allow_early and b >= min_b_check and _stops(ge, b, alpha, early_stop, delta_ci):
                    break
                if b >= B:
                    break

                # π^R
                Tb = _t2_from_index(ixXR)
                if Tb > T_obs:
                    gt += 1
                elif Tb == T_obs:
                    eq += 1
                ge = gt + eq
                b += 1
                perms_used = b
                if allow_early and b >= min_b_check and _stops(ge, b, alpha, early_stop, delta_ci):
                    break

            if allow_early and b >= min_b_check and _stops(ge, b, alpha, early_stop, delta_ci):
                break

            if rem and b < B:
                perm = rng.permutation(n_tot)
                ixX = perm[:N]
                Tb = _t2_from_index(ixX)
                if Tb > T_obs:
                    gt += 1
                elif Tb == T_obs:
                    eq += 1
                ge = gt + eq
                b += 1
                perms_used = b
                if allow_early and b >= min_b_check and _stops(ge, b, alpha, early_stop, delta_ci):
                    break

        else:
            for _ in range(n_this):
                perm = rng.permutation(n_tot)
                ixX = perm[:N]
                Tb = _t2_from_index(ixX)
                if Tb > T_obs:
                    gt += 1
                elif Tb == T_obs:
                    eq += 1
                ge = gt + eq
                b += 1
                perms_used = b
                if allow_early and b >= min_b_check and _stops(ge, b, alpha, early_stop, delta_ci):
                    break

        if allow_early and b >= min_b_check and _stops(ge, b, alpha, early_stop, delta_ci):
            break

    b = max(1, perms_used)
    p_perm = (1 + ge) / (b + 1)
    p_mid = (gt + 0.5 * eq) / b

    if decision == "pvalue":
        reject = (p_perm <= alpha)
    elif decision == "midp":
        reject = (p_mid <= alpha)
    elif decision == "randomized":
        p_lower = (1 + gt) / (b + 1)
        if p_lower > alpha:
            reject = False
        elif eq == 0:
            reject = (p_perm <= alpha)
        else:
            omega = (alpha - p_lower) / (eq / (b + 1))
            omega = float(np.clip(omega, 0.0, 1.0))
            reject = (np.random.default_rng().uniform() <= omega)
    else:
        reject = (p_perm <= alpha)

    return HotellingT2Result(
        stat_obs=float(T_obs),
        p_perm=float(p_perm),
        p_mid=float(p_mid),
        reject=bool(reject),
        ge=int(ge), gt=int(gt), eq=int(eq), perms_used=int(b),
        d=int(d), N=int(N), K=int(K),
        F_obs=float(extras["F_obs"]),
        df1=float(extras["df1"]), df2=float(extras["df2"]),
        shrinkage=str(shrinkage), ridge_lambda=float(ridge_lambda),
    )

# ---------------- Monte Carlo helpers ----------------


def estimate_type1(
    gen_null: Callable[[np.random.Generator], Tuple[Array, Array]],
    *,
    R: int = 200,
    test_kwargs: Optional[dict] = None,
    seed: int = 2026,
) -> Dict[str, float]:
    test_kwargs = dict(test_kwargs or {})
    ss = np.random.SeedSequence(seed)
    rej = 0
    perms: List[int] = []
    for s in ss.spawn(R):
        rng = np.random.default_rng(s.entropy)
        X, Y = gen_null(rng)
        out = hotelling_t2_permutation_test(X, Y, rng=rng, **test_kwargs)
        rej += int(out.reject)
        perms.append(out.perms_used)
    p = rej / R
    se = float(np.sqrt(max(p * (1 - p), 1e-12) / R))
    return {"type1_hat": p, "se": se, "avg_perms": float(np.mean(perms))}


def estimate_power(
    gen_alt: Callable[[np.random.Generator], Tuple[Array, Array]],
    *,
    R: int = 200,
    test_kwargs: Optional[dict] = None,
    seed: int = 2027,
) -> Dict[str, float]:
    test_kwargs = dict(test_kwargs or {})
    ss = np.random.SeedSequence(seed)
    rej = 0
    perms: List[int] = []
    for s in ss.spawn(R):
        rng = np.random.default_rng(s.entropy)
        X, Y = gen_alt(rng)
        out = hotelling_t2_permutation_test(X, Y, rng=rng, **test_kwargs)
        rej += int(out.reject)
        perms.append(out.perms_used)
    p = rej / R
    se = float(np.sqrt(max(p * (1 - p), 1e-12) / R))
    return {"power_hat": p, "se": se, "avg_perms": float(np.mean(perms))}

# ---------------- OO wrapper ----------------


class HotellingT2:
    """
    Uniform class API for your runner.

    Example:
        from baselines.hotelling_t2 import HotellingT2
        test = HotellingT2(B=2000, shrinkage="ledoitwolf")
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
        shrinkage: str = "none",        # "none" | "ledoitwolf" | "ridge"
        ridge_lambda: float = 1e-6,
        seed: int = 2026,
    ):
        self.B = B
        self.alpha = alpha
        self.decision = decision
        self.early_stop = early_stop
        self.delta_ci = delta_ci
        self.min_b_check = min_b_check
        self.chunk = chunk
        self.antithetic = antithetic
        self.shrinkage = shrinkage
        self.ridge_lambda = ridge_lambda
        self.seed = seed

    def run(self, X: Array, Y: Array) -> HotellingT2Result:
        rng = np.random.default_rng(self.seed)
        return hotelling_t2_permutation_test(
            X, Y,
            B=self.B,
            alpha=self.alpha,
            decision=self.decision,
            early_stop=self.early_stop,
            delta_ci=self.delta_ci,
            min_b_check=self.min_b_check,
            chunk=self.chunk,
            antithetic=self.antithetic,
            shrinkage=self.shrinkage,
            ridge_lambda=self.ridge_lambda,
            rng=rng,
        )

    def estimate_type1(self, gen_null: Callable[[np.random.Generator], Tuple[Array, Array]], *, R: int = 200, seed: int = 4045) -> dict:
        return estimate_type1(gen_null, R=R, test_kwargs=dict(
            B=self.B, alpha=self.alpha, decision=self.decision,
            early_stop=self.early_stop, delta_ci=self.delta_ci,
            min_b_check=self.min_b_check, chunk=self.chunk,
            antithetic=self.antithetic, shrinkage=self.shrinkage,
            ridge_lambda=self.ridge_lambda
        ), seed=seed)

    def estimate_power(self, gen_alt: Callable[[np.random.Generator], Tuple[Array, Array]], *, R: int = 200, seed: int = 4046) -> dict:
        return estimate_power(gen_alt, R=R, test_kwargs=dict(
            B=self.B, alpha=self.alpha, decision=self.decision,
            early_stop=self.early_stop, delta_ci=self.delta_ci,
            min_b_check=self.min_b_check, chunk=self.chunk,
            antithetic=self.antithetic, shrinkage=self.shrinkage,
            ridge_lambda=self.ridge_lambda
        ), seed=seed)
