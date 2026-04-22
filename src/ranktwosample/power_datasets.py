# power_datasets.py
# ---------------------------------------------------------------
# Dataset generators for H1 power experiments to pair with the
# "Parallel fast role-swapping (non-symmetric) χ² two-sample test".
#
# Each function returns (X, Y) with:
#   • X ~ p  (targets)
#   • Y ~ \tilde p (reference/pool)
#
# Families included:
#   (a) Location shift on AR(1) covariance
#   (b) Scale/shape: Gaussian scale; Laplace / Student-t (unit variance)
#   (c) Multimodal mixture vs single Gaussian
#   (d) Pure dependence with Beta(2,5) marginals:
#         Gaussian copula vs t-copula (same ρ), or vs Clayton/Gumbel matched in Kendall’s τ
# ---------------------------------------------------------------

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Optional, Literal, Dict, Any
from dataclasses import dataclass

from scipy.stats import norm, t as student_t, beta as beta_dist

# ------------------------ helpers ------------------------

def rng_like(seed_or_rng=None) -> np.random.Generator:
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)

def ar1_corr(d: int, rho: float) -> np.ndarray:
    i = np.arange(d)
    return rho ** np.abs(i[:, None] - i[None, :])

def equicorr(d: int, rho: float) -> np.ndarray:
    # correlation with rho off-diagonal; PSD if rho ∈ [-1/(d-1), 1)
    I = np.eye(d)
    J = np.ones((d, d))
    return (1 - rho) * I + rho * J

def mvn(n: int, mean: ArrayLike, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(np.asarray(mean, float), np.asarray(cov, float), size=n)

def safe_beta_ppf(u: np.ndarray, a=2.0, b=5.0) -> np.ndarray:
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return beta_dist.ppf(u, a, b)

def kendall_tau_of_gaussian(rho: float) -> float:
    # Valid also for t-copula
    return (2.0 / np.pi) * np.arcsin(rho)

def clayton_theta_from_tau(tau: float) -> float:
    # tau = theta/(theta+2)
    return 2.0 * tau / max(1e-12, (1.0 - tau))

def gumbel_theta_from_tau(tau: float) -> float:
    # tau = 1 - 1/theta  => theta = 1/(1 - tau)
    return 1.0 / max(1e-12, (1.0 - tau))

# Positive-stable sampler (Weron 1996) with Laplace E[e^{-t S}] = exp(-t^alpha), alpha∈(0,1]
def sample_positive_stable(alpha: float, size: int, rng: np.random.Generator) -> np.ndarray:
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0,1].")
    if alpha == 1.0:
        # degenerates to Dirac at 1 (Laplace = e^{-t})
        return np.ones(size)
    U = rng.uniform(0.0, np.pi, size=size)  # (0, π)
    W = rng.exponential(scale=1.0, size=size)
    # Weron’s formula for positive stable with Laplace exp(-t^alpha)
    sinU = np.sin(U)
    sin_aU = np.sin(alpha * U)
    sin_1aU = np.sin((1 - alpha) * U)
    # avoid numerical issues
    sinU = np.clip(sinU, 1e-16, None)
    part1 = (sin_aU / (sinU ** alpha))
    part2 = (sin_1aU / W) ** ((1 - alpha) / alpha)
    return part1 * part2

# Archimedean copulas via Marshall–Olkin
def sample_clayton_u(theta: float, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    if theta <= 0:
        raise ValueError("Clayton theta must be > 0.")
    V = rng.gamma(shape=1.0/theta, scale=1.0, size=n)  # Gamma(k, θ=1)
    E = rng.exponential(scale=1.0, size=(n, d))
    U = (1.0 + E / V[:, None]) ** (-1.0 / theta)
    return U

def sample_gumbel_u(theta: float, n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    if theta < 1.0:
        raise ValueError("Gumbel theta must be ≥ 1.")
    alpha = 1.0 / theta
    V = sample_positive_stable(alpha, size=n, rng=rng)  # positive stable
    E = rng.exponential(scale=1.0, size=(n, d))
    U = np.exp(- (E / V[:, None]) ** (1.0 / theta))
    return U

def gaussian_copula_u(n: int, d: int, rho: float, rng: np.random.Generator, equi: bool = True) -> np.ndarray:
    S = equicorr(d, rho) if equi else ar1_corr(d, rho)
    Z = mvn(n, np.zeros(d), S, rng)
    return norm.cdf(Z)

def t_copula_u(n: int, d: int, rho: float, nu: int, rng: np.random.Generator, equi: bool = True) -> np.ndarray:
    S = equicorr(d, rho) if equi else ar1_corr(d, rho)
    Z = mvn(n, np.zeros(d), S, rng)
    W = rng.chisquare(df=nu, size=n) / nu
    T = Z / np.sqrt(W)[:, None]
    return student_t.cdf(T, df=nu)

# ------------------------ (a) Location shift ------------------------

def sample_locshift_pair(
    d: int, N: int, K: int, rho: float = 0.6, delta: float = 0.4, rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y ~ N(0, Σ_AR1(rho));
    X ~ N(μ, Σ_AR1(rho)), μ = δ * 1/√d * 1
    """
    rng = rng_like(rng)
    S = ar1_corr(d, rho)
    mu = (delta / np.sqrt(d)) * np.ones(d)
    Y = mvn(K, np.zeros(d), S, rng)
    X = mvn(N, mu, S, rng)
    return X, Y

# ------------------------ (b) Scale / shape ------------------------

def sample_scale_gauss_pair(
    d: int, N: int, K: int, sigma: float = 1.2, normalize_var: bool = False, rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y ~ N(0, I_d);
    X ~ N(0, sigma^2 I_d).
    If normalize_var=True, X is scaled to unit variance (pure shape is trivial here; default False).
    """
    rng = rng_like(rng)
    Y = rng.normal(size=(K, d))
    X = rng.normal(scale=sigma, size=(N, d))
    if normalize_var and sigma != 0:
        X = X / sigma  # back to unit variance
    return X, Y

def sample_laplace_unitvar_pair(
    d: int, N: int, K: int, rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y ~ N(0, I_d);
    X has i.i.d. Laplace(0, b) with unit variance (var=1 => b=1/√2).
    """
    rng = rng_like(rng)
    b = 1.0 / np.sqrt(2.0)
    X = rng.laplace(loc=0.0, scale=b, size=(N, d))
    Y = rng.normal(size=(K, d))
    return X, Y

def sample_studentt_unitvar_pair(
    d: int, N: int, K: int, nu: int = 6, rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y ~ N(0, I_d);
    X has i.i.d. Student-t_ν scaled to unit variance: scale = sqrt((ν-2)/ν) (ν>2).
    """
    if nu <= 2:
        raise ValueError("For unit variance Student-t, require nu > 2.")
    rng = rng_like(rng)
    T = rng.standard_t(df=nu, size=(N, d))
    scale = np.sqrt((nu - 2.0) / nu)
    X = T * scale
    Y = rng.normal(size=(K, d))
    return X, Y

# ------------------------ (c) Multimodality ------------------------

def sample_mixture_pair(
    d: int, N: int, K: int, delta: float = 1.0, rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y ~ N(0, I_d);
    X ~ 0.5 N(+μ, I_d) + 0.5 N(-μ, I_d), μ = δ * 1/√d * 1.
    """
    rng = rng_like(rng)
    mu = (delta / np.sqrt(d)) * np.ones(d)
    signs = rng.choice([-1.0, 1.0], size=N)
    means = signs[:, None] * mu[None, :]
    X = means + rng.normal(size=(N, d))
    Y = rng.normal(size=(K, d))
    return X, Y

# ------------------------ (d) Pure dependence / Copulas ------------------------

@dataclass
class CopulaSpec:
    # base (tilde p) is always Gaussian copula with correlation rho_base
    rho_base: float = 0.5
    marginals: Literal["beta"] = "beta"
    marg_a: float = 2.0
    marg_b: float = 5.0
    # alternatives:
    alt: Literal["t", "clayton_matchtau", "gumbel_matchtau"] = "t"
    t_nu: int = 6  # for alt="t"
    equicorr: bool = True  # True: equicorrelation; False: AR(1)

def _apply_marginals(U: np.ndarray, spec: CopulaSpec) -> np.ndarray:
    if spec.marginals == "beta":
        return safe_beta_ppf(U, spec.marg_a, spec.marg_b)
    else:
        raise NotImplementedError("Only Beta(2,5) marginals are implemented per spec.")

def sample_copula_pair(
    d: int, N: int, K: int, spec: CopulaSpec, rng=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y uses Gaussian copula with rho_base and Beta(2,5) marginals.
    X uses one of:
      • t-copula with same linear corr and df=spec.t_nu, same marginals
      • Clayton/Gumbel copula matched in Kendall's tau to Gaussian with rho_base
    """
    rng = rng_like(rng)

    # Reference/pool Y ~ Gaussian copula (rho_base)
    Uy = gaussian_copula_u(K, d, spec.rho_base, rng, equi=spec.equicorr)
    Y = _apply_marginals(Uy, spec)

    # Alternative X
    if spec.alt == "t":
        Ux = t_copula_u(N, d, spec.rho_base, nu=spec.t_nu, rng=rng, equi=spec.equicorr)
    elif spec.alt == "clayton_matchtau":
        tau = kendall_tau_of_gaussian(spec.rho_base)
        theta = clayton_theta_from_tau(tau)
        Ux = sample_clayton_u(theta=theta, n=N, d=d, rng=rng)
    elif spec.alt == "gumbel_matchtau":
        tau = kendall_tau_of_gaussian(spec.rho_base)
        theta = gumbel_theta_from_tau(tau)  # ≥ 1
        Ux = sample_gumbel_u(theta=theta, n=N, d=d, rng=rng)
    else:
        raise ValueError(f"Unknown spec.alt = {spec.alt}")

    X = _apply_marginals(Ux, spec)
    return X, Y

# ------------------------ Dispatcher ------------------------

def generate_pair(
    family: Literal[
        "locshift",
        "scale-gauss",
        "shape-laplace",
        "shape-studentt",
        "mixture",
        "dep-tcopula",
        "dep-clayton",
        "dep-gumbel",
    ],
    d: int,
    N: int,
    K: int,
    params: Optional[Dict[str, Any]] = None,
    rng=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    family:
      - "locshift": params = {"rho": 0.6, "delta": 0.4}
      - "scale-gauss": params = {"sigma": 1.2, "normalize_var": False}
      - "shape-laplace": (no params)
      - "shape-studentt": params = {"nu": 6}
      - "mixture": params = {"delta": 1.0}
      - "dep-tcopula": params = {"rho": 0.5, "nu": 6, "equicorr": True}
      - "dep-clayton": params = {"rho": 0.5, "equicorr": True}
      - "dep-gumbel":  params = {"rho": 0.5, "equicorr": True}
    """
    params = params or {}
    rng = rng_like(rng)

    if family == "locshift":
        return sample_locshift_pair(d, N, K,
                                    rho=float(params.get("rho", 0.6)),
                                    delta=float(params.get("delta", 0.4)),
                                    rng=rng)
    elif family == "scale-gauss":
        return sample_scale_gauss_pair(d, N, K,
                                       sigma=float(params.get("sigma", 1.2)),
                                       normalize_var=bool(params.get("normalize_var", False)),
                                       rng=rng)
    elif family == "shape-laplace":
        return sample_laplace_unitvar_pair(d, N, K, rng=rng)
    elif family == "shape-studentt":
        return sample_studentt_unitvar_pair(d, N, K, nu=int(params.get("nu", 6)), rng=rng)
    elif family == "mixture":
        return sample_mixture_pair(d, N, K, delta=float(params.get("delta", 1.0)), rng=rng)
    elif family in ("dep-tcopula", "dep-clayton", "dep-gumbel"):
        alt = {"dep-tcopula": "t", "dep-clayton": "clayton_matchtau", "dep-gumbel": "gumbel_matchtau"}[family]
        spec = CopulaSpec(
            rho_base=float(params.get("rho", 0.5)),
            marginals="beta", marg_a=2.0, marg_b=5.0,
            alt=alt,
            t_nu=int(params.get("nu", 6)),
            equicorr=bool(params.get("equicorr", True)),
        )
        return sample_copula_pair(d, N, K, spec=spec, rng=rng)
    else:
        raise ValueError(f"Unknown family: {family}")
