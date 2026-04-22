# sliced_ot.py
# -------------------------------------------------------------------
# Sliced OT (W1) two-sample permutation test — API + OO wrapper
#   - Observed statistic: average 1D W1 over L random projections
#   - Reuse global sorted orders per direction across permutations
#   - Permutation calibration with optional early stopping
#   - Uniform class API: SlicedOTTest.run(X, Y)
# -------------------------------------------------------------------
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Dict, List

import numpy as np

Array = np.ndarray

__all__ = [
    "SlicedOTResult",
    "sliced_w1_stat",
    "sliced_ot_permutation_test",
    "estimate_type1",
    "estimate_power",
    "SlicedOTTest",
]

# ---------- helpers ----------


def _unit_random_dirs(L: int, d: int, rng: np.random.Generator) -> Array:
    V = rng.normal(size=(L, d))
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V  # (L, d)


def _ndtri(p: float) -> float:
    # Hastings approximation to Phi^{-1}(p)
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
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                (((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _wilson_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = _ndtri(1 - delta / 2.0)
    phat = k / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, phat *
                                       (1 - phat) / n + (z * z) / (4 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)


def _hoeffding_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    phat = k / n
    r = math.sqrt(max(0.0, math.log(2.0 / max(1e-16, delta))) / (2.0 * n))
    return max(0.0, phat - r), min(1.0, phat + r)


def _stops(ge: int, b: int, alpha: float, method: str, delta_ci: float) -> bool:
    if b <= 0:
        return False
    if method == "wilson":
        lo, hi = _wilson_ci(ge, b, delta_ci)
    elif method == "hoeffding":
        lo, hi = _hoeffding_ci(ge, b, delta_ci)
    else:
        return False
    return (hi <= alpha) or (lo > alpha)


def _quantile_pair_aligned(x_sorted: Array, y_sorted: Array, m: int) -> Tuple[Array, Array]:
    # Return quantile-aligned samples of length m drawn from the two sorted arrays.
    nx = x_sorted.shape[0]
    ny = y_sorted.shape[0]
    ix = np.minimum((np.linspace(0, nx - 1, m)).astype(int), nx - 1)
    iy = np.minimum((np.linspace(0, ny - 1, m)).astype(int), ny - 1)
    return x_sorted[ix], y_sorted[iy]


def _w1_from_sorted(x_sorted: Array, y_sorted: Array) -> float:
    # 1D Wasserstein-1 via quantile alignment on a common grid.
    nx, ny = x_sorted.size, y_sorted.size
    m = max(nx, ny)
    xs, ys = _quantile_pair_aligned(x_sorted, y_sorted, m)
    return float(np.mean(np.abs(xs - ys)))

# ---------- core statistic ----------


def sliced_w1_stat(
    X: Array, Y: Array, *,
    L: int = 128,
    dirs: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, Dict[str, Array]]:
    """
    Compute the sliced W1 statistic between X and Y:
        T = (1/L) * sum_{ell=1..L} W1(<v_ell, X>, <v_ell, Y>)
    Returns (T, cache) where cache contains projections and sort orders usable by permutations.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N, d = X.shape
    K, d2 = Y.shape
    if d2 != d:
        raise ValueError("X and Y must have the same dimension")

    V = dirs if dirs is not None else _unit_random_dirs(L, d, rng)  # (L, d)
    Z = np.vstack([X, Y])                  # (N+K, d)
    Zproj = Z @ V.T                        # (N+K, L)

    order = np.argsort(Zproj, axis=0)      # (N+K, L)
    Zproj_sorted = np.take_along_axis(Zproj, order, axis=0)

    maskX = np.zeros(N + K, dtype=bool)
    maskX[:N] = True

    w1s = []
    for l in range(L):
        mX = maskX[order[:, l]]
        x_sorted = Zproj_sorted[mX, l]
        y_sorted = Zproj_sorted[~mX, l]
        w1s.append(_w1_from_sorted(x_sorted, y_sorted))
    T = float(np.mean(w1s))

    cache = dict(V=V, Zproj_sorted=Zproj_sorted, order=order, N=N, K=K)
    return T, cache

# ---------- permutation test ----------


@dataclass
class SlicedOTResult:
    stat_obs: float
    p_perm: float
    p_mid: float
    reject: bool
    ge: int
    gt: int
    eq: int
    perms_used: int


def _perm_stat_fast(maskX: Array, Zproj_sorted: Array, order: Array, L: int) -> float:
    # Given a selection mask for X over {0..n_tot-1}, compute sliced W1
    # reusing precomputed global sorted orders per direction.
    w1s = []
    for l in range(L):
        mX_ord = maskX[order[:, l]]
        x_sorted = Zproj_sorted[mX_ord, l]
        y_sorted = Zproj_sorted[~mX_ord, l]
        w1s.append(_w1_from_sorted(x_sorted, y_sorted))
    return float(np.mean(w1s))


def sliced_ot_permutation_test(
    X: Array, Y: Array, *,
    L: int = 128,
    B: int = 1000,
    alpha: float = 0.05,
    decision: str = "pvalue",        # "pvalue" | "midp" | "randomized"
    early_stop: str = "wilson",      # "wilson" | "hoeffding" | "none"
    delta_ci: float = 1e-2,
    min_b_check: int = 100,
    chunk: int = 256,
    antithetic: bool = True,
    dirs: Optional[Array] = None,
    rng: Optional[np.random.Generator] = None,
) -> SlicedOTResult:
    """
    Conditional permutation test for sliced W1.
    Fast path: reuse global sorted orders per direction (no re-sorting in perms).
    """
    if rng is None:
        rng = np.random.default_rng(123)

    T_obs, cache = sliced_w1_stat(X, Y, L=L, dirs=dirs, rng=rng)
    Zproj_sorted = cache["Zproj_sorted"]
    order = cache["order"]
    N = int(cache["N"])
    K = int(cache["K"])
    n_tot = N + K

    gt = eq = ge = 0
    perms_used = 0
    b = 0
    C = max(1, int(chunk))
    allow_early = early_stop in ("wilson", "hoeffding")

    while b < B:
        n_this = min(C, B - b)

        if antithetic:
            pairs = n_this // 2
            rem = n_this % 2
            for _ in range(pairs):
                perm = rng.permutation(n_tot)
                permR = perm[::-1]

                maskX = np.zeros(n_tot, dtype=bool)
                maskX[perm[:N]] = True
                Tb = _perm_stat_fast(maskX, Zproj_sorted, order, L)
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

                maskX[:] = False
                maskX[permR[:N]] = True
                Tb = _perm_stat_fast(maskX, Zproj_sorted, order, L)
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
                maskX = np.zeros(n_tot, dtype=bool)
                maskX[perm[:N]] = True
                Tb = _perm_stat_fast(maskX, Zproj_sorted, order, L)
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
                maskX = np.zeros(n_tot, dtype=bool)
                maskX[perm[:N]] = True
                Tb = _perm_stat_fast(maskX, Zproj_sorted, order, L)
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

    return SlicedOTResult(
        stat_obs=T_obs, p_perm=p_perm, p_mid=p_mid, reject=reject,
        ge=ge, gt=gt, eq=eq, perms_used=b
    )

# ---------- simple Monte Carlo wrappers ----------


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
        out = sliced_ot_permutation_test(X, Y, rng=rng, **test_kwargs)
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
        out = sliced_ot_permutation_test(X, Y, rng=rng, **test_kwargs)
        rej += int(out.reject)
        perms.append(out.perms_used)
    p = rej / R
    se = float(np.sqrt(max(p * (1 - p), 1e-12) / R))
    return {"power_hat": p, "se": se, "avg_perms": float(np.mean(perms))}

# ---------- OO wrapper ----------


class SlicedOTTest:
    """
    Example:
        from baselines.sliced_ot import SlicedOTTest
        test = SlicedOTTest(L=128, B=1000, alpha=0.05)
        out = test.run(X, Y)
    """

    def __init__(
        self,
        *,
        L: int = 128,
        B: int = 1000,
        alpha: float = 0.05,
        decision: str = "pvalue",
        early_stop: str = "wilson",
        delta_ci: float = 1e-2,
        min_b_check: int = 100,
        chunk: int = 256,
        antithetic: bool = True,
        dirs: Optional[Array] = None,
        seed: int = 2026,
    ):
        self.L = L
        self.B = B
        self.alpha = alpha
        self.decision = decision
        self.early_stop = early_stop
        self.delta_ci = delta_ci
        self.min_b_check = min_b_check
        self.chunk = chunk
        self.antithetic = antithetic
        self.dirs = dirs
        self.seed = seed

    def run(self, X: Array, Y: Array) -> SlicedOTResult:
        rng = np.random.default_rng(self.seed)
        return sliced_ot_permutation_test(
            X, Y,
            L=self.L,
            B=self.B,
            alpha=self.alpha,
            decision=self.decision,
            early_stop=self.early_stop,
            delta_ci=self.delta_ci,
            min_b_check=self.min_b_check,
            chunk=self.chunk,
            antithetic=self.antithetic,
            dirs=self.dirs,
            rng=rng,
        )

    def estimate_type1(self, gen_null: Callable[[np.random.Generator], Tuple[Array, Array]], *, R: int = 200, seed: int = 4043) -> dict:
        return estimate_type1(gen_null, R=R, test_kwargs=dict(
            L=self.L, B=self.B, alpha=self.alpha, decision=self.decision,
            early_stop=self.early_stop, delta_ci=self.delta_ci,
            min_b_check=self.min_b_check, chunk=self.chunk,
            antithetic=self.antithetic, dirs=self.dirs
        ), seed=seed)

    def estimate_power(self, gen_alt: Callable[[np.random.Generator], Tuple[Array, Array]], *, R: int = 200, seed: int = 4044) -> dict:
        return estimate_power(gen_alt, R=R, test_kwargs=dict(
            L=self.L, B=self.B, alpha=self.alpha, decision=self.decision,
            early_stop=self.early_stop, delta_ci=self.delta_ci,
            min_b_check=self.min_b_check, chunk=self.chunk,
            antithetic=self.antithetic, dirs=self.dirs
        ), seed=seed)
