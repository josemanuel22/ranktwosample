# rank_two_sample.py
# API for non-symmetric rank χ² / G-test with permutations (CPU or optional CuPy GPU)
#
# Modes:
#   - reference_mode="shared_pool":
#         old practical version; ranks are computed against an active subset of Y
#   - reference_mode="fresh_iid":
#         practical fresh-iid version; for each observation z, draw a fresh batch
#         of size K_ref WITH REPLACEMENT from the empirical law of the CURRENT Y-pool
#
# Notes:
#   • This file keeps the old helper API used by rank_two_sample_subspaces.py
#   • reference_sampler is retained only for backward compatibility and is ignored here

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor

import numpy as _np


# ---------------- Optional GPU backend ----------------

def _try_enable_cupy(mode: str = "off", device: int = 0):
    """
    mode: 'off' | 'auto' | 'on'
    Returns (xp, using_cupy: bool, cp or None)
    """
    if mode == "off":
        return _np, False, None
    try:
        import cupy as _cp
        ndev = _cp.cuda.runtime.getDeviceCount()
        if ndev <= 0:
            if mode == "on":
                print("[GPU] No CUDA device found; staying on CPU.")
            return _np, False, None
        _cp.cuda.Device(device).use()
        return _cp, True, _cp
    except Exception as e:
        if mode == "on":
            print(f"[GPU] Failed to enable CuPy: {e}. Using CPU.")
        return _np, False, None


# ---------------- Dataclasses ----------------

@dataclass
class RankGTestConfig:
    alpha: float = 0.05
    K_ref: Optional[int] = None
    engine: str = "auto"             # 'auto' | 'searchsorted' | 'prefixsum'
    tie: str = "jitter"              # 'jitter' | 'right' | 'left'
    alpha0: float = 0.1              # Jeffreys smoothing; or set <0 for auto
    decision: str = "midp"           # 'pvalue' | 'midp' | 'randomized'
    B: int = 1000                    # max permutations per repetition
    kref_switch: int = 64            # threshold where prefixsum wins over searchsorted
    chunk: int = 512                 # perm chunk size
    antithetic: bool = True
    stop_ci: str = "wilson"          # 'wilson' | 'hoeffding'
    delta_ci: float = 1e-2
    min_b_check: int = 100
    use_fullY_for_HY: bool = True
    stat: str = "gtest"              # 'gtest' | 'chi2'
    jitter_scale: float = 1e-12
    gpu: str = "off"                 # 'off' | 'auto' | 'on'
    device: int = 0                  # CUDA device id (if gpu != off)

    # New mode switch
    reference_mode: str = "shared_pool"  # 'shared_pool' | 'fresh_iid'

    # Retained only for backward compatibility with older wrappers.
    # In this version, fresh_iid samples from the empirical law of the CURRENT Y-pool,
    # so this field is ignored.
    reference_sampler: Optional[Callable[..., Any]] = None


@dataclass
class TestResult:
    reject: bool
    p_perm: float
    p_mid: float
    T_obs: float
    perms_used: int
    elapsed_ms: float


# ---------------- Core (NumPy/CuPy-agnostic) ----------------

class RankGTest:
    def __init__(self, cfg: RankGTestConfig):
        self.cfg = cfg
        self.xp, self.using_cupy, self.cp = _try_enable_cupy(
            cfg.gpu, cfg.device)

    # --- RNG helpers ---
    def _rng(self, seed: Optional[int] = None):
        if self.using_cupy:
            return self.cp.random.RandomState(None if seed is None else int(seed))
        return _np.random.default_rng(None if seed is None else int(seed))

    # --- Linear algebra / draws ---
    def _corr_matrix(self, d: int, rho: float):
        xp = self.xp
        if d == 1:
            return xp.array([[1.0]])
        S = xp.full((d, d), rho, dtype=xp.float64)
        xp.fill_diagonal(S, 1.0)
        w, V = xp.linalg.eigh(S)
        w = xp.clip(w, 1e-12, None)
        return (V * w) @ V.T

    def _draw_gauss(self, n: int, mean, cov, rng):
        xp = self.xp
        mean = xp.asarray(mean, dtype=xp.float64)
        d = int(mean.size)
        L = xp.linalg.cholesky(cov)
        Z = rng.normal(size=(n, d))
        Z = xp.asarray(Z, dtype=xp.float64)
        return Z @ L.T + mean

    def _draw_studentt(self, n: int, d: int, nu: int, rng):
        xp = self.xp
        Z = rng.normal(size=(n, d))
        U = rng.chisquare(nu, size=n)
        Z = xp.asarray(Z, dtype=xp.float64)
        U = xp.asarray(U, dtype=xp.float64)
        return Z / xp.sqrt(U[:, None] / float(nu))

    # --- Prefixsum precompute ---
    def _precompute_orders(self, Z, tie="right", jitter_scale=1e-12, rng=None):
        xp = self.xp
        n_tot, d = Z.shape

        if tie == "jitter":
            if rng is None:
                rng = self._rng(0)
            Zw = Z + xp.asarray(rng.normal(scale=jitter_scale,
                                size=Z.shape), dtype=xp.float64)
            side = "right"
        else:
            Zw = Z
            side = tie

        orders, pos_all, last_right_all, first_left_all = [], [], [], []
        for j in range(d):
            order = xp.argsort(Zw[:, j], kind="mergesort")
            vals = Zw[order, j]
            pos = xp.empty(n_tot, dtype=xp.int64)
            pos[order] = xp.arange(n_tot, dtype=xp.int64)

            _, inv, counts = xp.unique(
                vals, return_inverse=True, return_counts=True)
            first_of_group = xp.cumsum(
                xp.concatenate([xp.array([0], dtype=xp.int64),
                               counts[:-1].astype(xp.int64)])
            )
            last_of_group = first_of_group + counts.astype(xp.int64) - 1
            first_left = first_of_group[inv]
            last_right = last_of_group[inv]

            orders.append(order)
            pos_all.append(pos)
            last_right_all.append(last_right.astype(xp.int64))
            first_left_all.append(first_left.astype(xp.int64))

        return {
            "orders": orders,
            "pos": pos_all,
            "last_right": last_right_all,
            "first_left": first_left_all,
            "side": side,
        }

    # --- Fine-grid indexing ---
    def _fine_linear_index(self, m, K_ref: int):
        xp = self.xp
        d = m.shape[1]
        base = int(K_ref) + 1
        w_fine = (base ** xp.arange(d - 1, -1, -1,
                  dtype=xp.int64)).astype(xp.int64)
        idx = (m.astype(xp.int64) * w_fine).sum(axis=1)
        M_eff = int(base ** d)
        return idx, w_fine, M_eff

    # --- Rank engines ---
    def _ranks_prefixsum(self, indices, sel_mask, ords, side_pick):
        xp = self.xp
        d = len(ords["orders"])
        m_cols = []
        for j in range(d):
            order_j = ords["orders"][j]
            pos_j = ords["pos"][j]
            pick = ords[side_pick][j]
            s_ord = sel_mask[order_j].astype(xp.int8)
            S = xp.cumsum(s_ord, dtype=xp.int64)
            pj = pos_j[indices]
            pr = pick[pj]
            m_cols.append(S[pr])
        return xp.stack(m_cols, axis=1)

    def _ranks_searchsorted(self, indices, Iref, Z_cols, side="right", jitter=None):
        xp = self.xp
        d = len(Z_cols)
        m_cols = []
        for j in range(d):
            Zj = Z_cols[j]
            if jitter is not None:
                Zj = Zj + jitter[j]
            Rj = Zj[Iref]
            Rj_sorted = xp.sort(Rj)
            zj = Zj[indices]
            m_cols.append(xp.searchsorted(Rj_sorted, zj, side=side))
        return xp.stack(m_cols, axis=1)

    def _ranks_fresh_batches(self, Z, ref_batches, tie="jitter", jitter_scale=1e-12, rng=None):
        """
        Z:          shape (n_tot, d)
        ref_batches shape (n_tot, K_ref, d)

        Returns:
            m shape (n_tot, d), where m[i,j] is the rank count of Z[i,j]
            against ref_batches[i,:,j].
        """
        xp = self.xp
        Z = xp.asarray(Z, dtype=xp.float64)
        ref_batches = xp.asarray(ref_batches, dtype=xp.float64)

        if Z.ndim != 2:
            raise ValueError("Z must be 2D.")
        if ref_batches.ndim != 3:
            raise ValueError("ref_batches must be 3D.")

        n_tot, d = Z.shape
        if ref_batches.shape[0] != n_tot or ref_batches.shape[2] != d:
            raise ValueError("ref_batches must have shape (n_tot, K_ref, d).")

        if tie == "jitter":
            if rng is None:
                rng = self._rng(0)
            Zj = Z + xp.asarray(rng.normal(scale=jitter_scale,
                                size=Z.shape), dtype=xp.float64)
            Rj = ref_batches + xp.asarray(
                rng.normal(scale=jitter_scale, size=ref_batches.shape), dtype=xp.float64
            )
            # After jitter, ties occur with probability ~0; use <=
            m = xp.sum(Rj <= Zj[:, None, :], axis=1)
        elif tie == "right":
            m = xp.sum(ref_batches <= Z[:, None, :], axis=1)
        elif tie == "left":
            m = xp.sum(ref_batches < Z[:, None, :], axis=1)
        else:
            raise ValueError("tie must be 'jitter' | 'right' | 'left'.")

        return m.astype(xp.int64)

    # --- Sampling fresh batches from empirical Y ---
    def _sample_fresh_reference_batches_from_Y(self, Y, n_items: int, K_ref: int, rng):
        """
        Sample fresh reference batches from the empirical law of the CURRENT Y-pool.

        Returns an array of shape (n_items, K_ref, d), where each row is sampled
        i.i.d. WITH REPLACEMENT from the rows of Y.
        """
        xp = self.xp
        Y = xp.asarray(Y, dtype=xp.float64)

        M, d = Y.shape
        if M <= 0:
            raise ValueError("Y must contain at least one reference sample.")
        if K_ref < 1:
            raise ValueError("K_ref must be >= 1.")

        if self.using_cupy:
            idx = rng.randint(0, M, size=(n_items, K_ref))
        else:
            idx = rng.integers(0, M, size=(n_items, K_ref))

        idx = xp.asarray(idx, dtype=xp.int64)
        return Y[idx]   # shape (n_items, K_ref, d)

    # --- Stat from counts ---
    def _stat_from_counts(self, HX, HY, N_local: int, Ky_local: int, alpha0: float, kind="gtest"):
        xp = self.xp
        HX = HX.astype(xp.float64, copy=False)
        HY = HY.astype(xp.float64, copy=False)
        M = float(HX.size)
        denom = float(Ky_local) + float(alpha0) * M
        qhat = (HY + float(alpha0)) / denom
        EX = float(N_local) * qhat

        if kind == "chi2":
            T = xp.where(EX > 0.0, (HX - EX) ** 2 / EX, 0.0).sum()
            return float(T)
        if kind == "gtest":
            ratio = xp.where(HX > 0.0, HX / EX, 1.0)
            contrib = xp.where(HX > 0.0, HX * xp.log(ratio), 0.0)
            T = 2.0 * contrib.sum()
            return float(T)
        raise ValueError("kind must be 'chi2' or 'gtest'")

    # --- CIs for early stop ---
    @staticmethod
    def _ndtri(p: float) -> float:
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
            return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                   ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        if p > phigh:
            q = math.sqrt(-2 * math.log(1 - p))
            return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                    ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        q = p - 0.5
        r = q * q
        return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    def _wilson_ci(self, k: int, n: int, delta: float) -> Tuple[float, float]:
        if n <= 0:
            return (0.0, 1.0)
        z = self._ndtri(1 - delta / 2.0)
        phat = k / n
        denom = 1 + (z*z) / n
        center = (phat + (z*z) / (2*n)) / denom
        half = (z / denom) * \
            math.sqrt(max(0.0, phat*(1-phat)/n + (z*z)/(4*n*n)))
        return max(0.0, center - half), min(1.0, center + half)

    def _hoeffding_ci(self, k: int, n: int, delta: float) -> Tuple[float, float]:
        if n <= 0:
            return (0.0, 1.0)
        phat = k / n
        r = math.sqrt(max(0.0, math.log(2.0 / max(1e-16, delta))) / (2.0 * n))
        return max(0.0, phat - r), min(1.0, phat + r)

    # --- Shared-pool observed statistic ---
    def _compute_T_obs_shared_pool(self, X, Y, rng) -> float:
        xp = self.xp
        cfg = self.cfg

        N, d = X.shape
        K = Y.shape[0]
        K_ref = cfg.K_ref if cfg.K_ref is not None else int(K)
        if not (1 <= K_ref <= K):
            raise ValueError("K_ref must be in [1, K]")

        Z = xp.vstack([X, Y])
        Z_cols = [Z[:, j] for j in range(d)]
        sel = xp.zeros(N + K, dtype=xp.uint8)

        jitter = None
        ss_side = cfg.tie
        if cfg.tie == "jitter":
            jitter = xp.asarray(rng.normal(
                scale=cfg.jitter_scale, size=(d, N + K)), dtype=xp.float64)
            ss_side = "right"

        ords = None
        side_pick = None
        need_prefix = (cfg.engine in ("auto", "prefixsum")) or (
            cfg.engine == "searchsorted" and K_ref > cfg.kref_switch
        )
        if need_prefix:
            ords = self._precompute_orders(
                Z,
                tie=("right" if cfg.tie == "jitter" else cfg.tie),
                rng=rng,
                jitter_scale=cfg.jitter_scale,
            )
            side_pick = "last_right" if ords["side"] == "right" else "first_left"

        if cfg.engine == "auto":
            use_ss = (K_ref <= cfg.kref_switch)
        elif cfg.engine == "searchsorted":
            use_ss = True
        elif cfg.engine == "prefixsum":
            use_ss = False
        else:
            raise ValueError("engine must be auto|searchsorted|prefixsum")

        Iy_obs_inY = rng.permutation(K)[:K_ref]
        Iy_obs = xp.int64(N) + Iy_obs_inY
        idx_X_obs = xp.arange(N, dtype=xp.int64)

        if use_ss:
            mX_obs = self._ranks_searchsorted(
                idx_X_obs, Iy_obs, Z_cols, side=ss_side, jitter=jitter)
            if cfg.use_fullY_for_HY:
                Iy_pool_obs = xp.arange(N, N + K, dtype=xp.int64)
                mY_obs = self._ranks_searchsorted(
                    Iy_pool_obs, Iy_obs, Z_cols, side=ss_side, jitter=jitter)
            else:
                mY_obs = self._ranks_searchsorted(
                    Iy_obs, Iy_obs, Z_cols, side=ss_side, jitter=jitter)
        else:
            sel[:] = 0
            sel[Iy_obs] = 1
            mX_obs = self._ranks_prefixsum(idx_X_obs, sel, ords, side_pick)
            if cfg.use_fullY_for_HY:
                Iy_pool_obs = xp.arange(N, N + K, dtype=xp.int64)
                mY_obs = self._ranks_prefixsum(
                    Iy_pool_obs, sel, ords, side_pick)
            else:
                mY_obs = self._ranks_prefixsum(Iy_obs, sel, ords, side_pick)

        idxX, _, M_eff = self._fine_linear_index(mX_obs, K_ref)
        idxY, _, _ = self._fine_linear_index(mY_obs, K_ref)

        HX = xp.bincount(idxX, minlength=M_eff)
        HY = xp.bincount(idxY, minlength=M_eff)
        Ky = int(float(HY.sum()))

        alpha0 = cfg.alpha0
        if alpha0 < 0:
            alpha0 = max(1e-6, 0.02 * (Ky / max(1, M_eff)))

        return float(self._stat_from_counts(HX, HY, N_local=N, Ky_local=Ky, alpha0=alpha0, kind=cfg.stat))

    # --- Fresh-iid empirical observed statistic ---
    def _compute_T_obs_fresh_iid_empirical(self, X, Y, rng) -> float:
        """
        Fresh-iid version based on the EMPIRICAL law of the current Y-pool.
        Each pooled observation gets its own K_ref-sized reference batch, sampled
        with replacement from Y.
        """
        xp = self.xp
        cfg = self.cfg

        X = xp.asarray(X, dtype=xp.float64)
        Y = xp.asarray(Y, dtype=xp.float64)

        N, d = X.shape
        K = Y.shape[0]
        K_ref = cfg.K_ref if cfg.K_ref is not None else int(K)
        if K_ref < 1:
            raise ValueError("K_ref must be >= 1.")

        Z = xp.vstack([X, Y])                # shape (N+K, d)
        n_tot = N + K

        ref_batches = self._sample_fresh_reference_batches_from_Y(
            Y=Y,
            n_items=n_tot,
            K_ref=K_ref,
            rng=rng,
        )                                    # shape (N+K, K_ref, d)

        m = self._ranks_fresh_batches(
            Z,
            ref_batches,
            tie=cfg.tie,
            jitter_scale=cfg.jitter_scale,
            rng=rng,
        )

        idx, _, M_eff = self._fine_linear_index(m, K_ref)

        HX = xp.bincount(idx[:N], minlength=M_eff)
        HY = xp.bincount(idx[N:], minlength=M_eff)

        alpha0 = cfg.alpha0
        if alpha0 < 0:
            alpha0 = max(1e-6, 0.02 * (K / max(1, M_eff)))

        return float(self._stat_from_counts(
            HX, HY,
            N_local=N,
            Ky_local=K,
            alpha0=alpha0,
            kind=cfg.stat,
        ))

    # --- Shared-pool permutation test ---
    def _test_shared_pool(self, X_in, Y_in, seed: Optional[int] = None) -> TestResult:
        xp = self.xp
        cfg = self.cfg

        X = xp.asarray(X_in)
        Y = xp.asarray(Y_in)
        rng = self._rng(seed)
        t0 = time.perf_counter()

        T_obs = self._compute_T_obs_shared_pool(X, Y, rng)

        N, d = X.shape
        K = Y.shape[0]
        n_tot = N + K
        Z = xp.vstack([X, Y])
        Z_cols = [Z[:, j] for j in range(d)]

        K_ref = cfg.K_ref if cfg.K_ref is not None else int(K)
        sel = xp.zeros(n_tot, dtype=xp.uint8)

        jitter = None
        ss_side = cfg.tie
        if cfg.tie == "jitter":
            jitter = xp.asarray(rng.normal(
                scale=cfg.jitter_scale, size=(d, n_tot)), dtype=xp.float64)
            ss_side = "right"

        ords = None
        side_pick = None
        need_prefix = (cfg.engine in ("auto", "prefixsum")) or (
            cfg.engine == "searchsorted" and K_ref > cfg.kref_switch
        )
        if need_prefix:
            ords = self._precompute_orders(
                Z,
                tie=("right" if cfg.tie == "jitter" else cfg.tie),
                rng=rng,
                jitter_scale=cfg.jitter_scale,
            )
            side_pick = "last_right" if ords["side"] == "right" else "first_left"

        if cfg.engine == "auto":
            use_ss = (K_ref <= cfg.kref_switch)
        elif cfg.engine == "searchsorted":
            use_ss = True
        elif cfg.engine == "prefixsum":
            use_ss = False
        else:
            raise ValueError("engine must be auto|searchsorted|prefixsum")

        gt = eq = ge = 0
        perms_used = 0
        C = max(1, int(cfg.chunk))
        b = 0
        allow_early = (cfg.decision in ("pvalue", "midp", "randomized"))

        while b < cfg.B:
            n_this = min(C, cfg.B - b)

            if cfg.antithetic:
                pairs = n_this // 2
                remainder = n_this % 2

                for _ in range(pairs):
                    perm = rng.permutation(n_tot)
                    permR = perm[::-1]

                    for cur_perm in (perm, permR):
                        Iy_pool = cur_perm[:K]
                        Ix_pool = cur_perm[K:]
                        ref_pos = rng.permutation(K)[:K_ref]
                        Iref = Iy_pool[ref_pos]

                        if use_ss:
                            mX = self._ranks_searchsorted(
                                Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                            if cfg.use_fullY_for_HY:
                                mY = self._ranks_searchsorted(
                                    Iy_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                            else:
                                mY = self._ranks_searchsorted(
                                    Iref, Iref, Z_cols, side=ss_side, jitter=jitter)
                        else:
                            sel[:] = 0
                            sel[Iref] = 1
                            mX = self._ranks_prefixsum(
                                Ix_pool, sel, ords, side_pick)
                            if cfg.use_fullY_for_HY:
                                mY = self._ranks_prefixsum(
                                    Iy_pool, sel, ords, side_pick)
                            else:
                                mY = self._ranks_prefixsum(
                                    Iref, sel, ords, side_pick)

                        idxX, _, M_eff = self._fine_linear_index(mX, K_ref)
                        idxY, _, _ = self._fine_linear_index(mY, K_ref)
                        HX = xp.bincount(idxX, minlength=M_eff)
                        HY = xp.bincount(idxY, minlength=M_eff)
                        Ky = int(float(HY.sum()))

                        alpha0 = cfg.alpha0
                        if alpha0 < 0:
                            alpha0 = max(1e-6, 0.02 * (Ky / max(1, M_eff)))

                        Tb = self._stat_from_counts(
                            HX, HY, N, Ky, alpha0, cfg.stat)

                        if Tb > T_obs:
                            gt += 1
                        elif Tb == T_obs:
                            eq += 1
                        ge = gt + eq
                        b += 1
                        perms_used = b

                        if allow_early and b >= cfg.min_b_check:
                            if cfg.stop_ci == "wilson":
                                lo, hi = self._wilson_ci(ge, b, cfg.delta_ci)
                            else:
                                lo, hi = self._hoeffding_ci(
                                    ge, b, cfg.delta_ci)
                            if hi <= cfg.alpha or lo > cfg.alpha:
                                break

                        if b >= cfg.B:
                            break

                    if b >= cfg.B:
                        break

                    if allow_early and b >= cfg.min_b_check:
                        if cfg.stop_ci == "wilson":
                            lo, hi = self._wilson_ci(ge, b, cfg.delta_ci)
                        else:
                            lo, hi = self._hoeffding_ci(ge, b, cfg.delta_ci)
                        if hi <= cfg.alpha or lo > cfg.alpha:
                            break

                if remainder and b < cfg.B:
                    perm = rng.permutation(n_tot)
                    Iy_pool = perm[:K]
                    Ix_pool = perm[K:]
                    ref_pos = rng.permutation(K)[:K_ref]
                    Iref = Iy_pool[ref_pos]

                    if use_ss:
                        mX = self._ranks_searchsorted(
                            Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                        if cfg.use_fullY_for_HY:
                            mY = self._ranks_searchsorted(
                                Iy_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                        else:
                            mY = self._ranks_searchsorted(
                                Iref, Iref, Z_cols, side=ss_side, jitter=jitter)
                    else:
                        sel[:] = 0
                        sel[Iref] = 1
                        mX = self._ranks_prefixsum(
                            Ix_pool, sel, ords, side_pick)
                        if cfg.use_fullY_for_HY:
                            mY = self._ranks_prefixsum(
                                Iy_pool, sel, ords, side_pick)
                        else:
                            mY = self._ranks_prefixsum(
                                Iref, sel, ords, side_pick)

                    idxX, _, M_eff = self._fine_linear_index(mX, K_ref)
                    idxY, _, _ = self._fine_linear_index(mY, K_ref)
                    HX = xp.bincount(idxX, minlength=M_eff)
                    HY = xp.bincount(idxY, minlength=M_eff)
                    Ky = int(float(HY.sum()))

                    alpha0 = cfg.alpha0
                    if alpha0 < 0:
                        alpha0 = max(1e-6, 0.02 * (Ky / max(1, M_eff)))

                    Tb = self._stat_from_counts(
                        HX, HY, N, Ky, alpha0, cfg.stat)

                    if Tb > T_obs:
                        gt += 1
                    elif Tb == T_obs:
                        eq += 1
                    ge = gt + eq
                    b += 1
                    perms_used = b

            else:
                for _ in range(n_this):
                    perm = rng.permutation(n_tot)
                    Iy_pool = perm[:K]
                    Ix_pool = perm[K:]
                    ref_pos = rng.permutation(K)[:K_ref]
                    Iref = Iy_pool[ref_pos]

                    if use_ss:
                        mX = self._ranks_searchsorted(
                            Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                        if cfg.use_fullY_for_HY:
                            mY = self._ranks_searchsorted(
                                Iy_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                        else:
                            mY = self._ranks_searchsorted(
                                Iref, Iref, Z_cols, side=ss_side, jitter=jitter)
                    else:
                        sel[:] = 0
                        sel[Iref] = 1
                        mX = self._ranks_prefixsum(
                            Ix_pool, sel, ords, side_pick)
                        if cfg.use_fullY_for_HY:
                            mY = self._ranks_prefixsum(
                                Iy_pool, sel, ords, side_pick)
                        else:
                            mY = self._ranks_prefixsum(
                                Iref, sel, ords, side_pick)

                    idxX, _, M_eff = self._fine_linear_index(mX, K_ref)
                    idxY, _, _ = self._fine_linear_index(mY, K_ref)
                    HX = xp.bincount(idxX, minlength=M_eff)
                    HY = xp.bincount(idxY, minlength=M_eff)
                    Ky = int(float(HY.sum()))

                    alpha0 = cfg.alpha0
                    if alpha0 < 0:
                        alpha0 = max(1e-6, 0.02 * (Ky / max(1, M_eff)))

                    Tb = self._stat_from_counts(
                        HX, HY, N, Ky, alpha0, cfg.stat)

                    if Tb > T_obs:
                        gt += 1
                    elif Tb == T_obs:
                        eq += 1
                    ge = gt + eq
                    b += 1
                    perms_used = b

                    if allow_early and b >= cfg.min_b_check:
                        if cfg.stop_ci == "wilson":
                            lo, hi = self._wilson_ci(ge, b, cfg.delta_ci)
                        else:
                            lo, hi = self._hoeffding_ci(ge, b, cfg.delta_ci)
                        if hi <= cfg.alpha or lo > cfg.alpha:
                            break

            if allow_early and b >= cfg.min_b_check:
                if cfg.stop_ci == "wilson":
                    lo, hi = self._wilson_ci(ge, b, cfg.delta_ci)
                else:
                    lo, hi = self._hoeffding_ci(ge, b, cfg.delta_ci)
                if hi <= cfg.alpha or lo > cfg.alpha:
                    break

        b = max(1, perms_used)
        p_perm = (1 + ge) / (b + 1)
        p_mid = (gt + 0.5 * eq) / b

        if cfg.decision == "pvalue":
            reject = (p_perm <= cfg.alpha)
        elif cfg.decision == "midp":
            reject = (p_mid <= cfg.alpha)
        elif cfg.decision == "randomized":
            p_lower = (1 + gt) / (b + 1)
            if p_lower > cfg.alpha:
                reject = False
            elif eq == 0:
                reject = (p_perm <= cfg.alpha)
            else:
                omega = (cfg.alpha - p_lower) / (eq / (b + 1))
                omega = float(min(max(omega, 0.0), 1.0))
                reject = (_np.random.default_rng().uniform() <= omega)
        else:
            reject = (p_perm <= cfg.alpha)

        elapsed_ms = 1e3 * (time.perf_counter() - t0)
        return TestResult(bool(reject), float(p_perm), float(p_mid), float(T_obs), int(perms_used), float(elapsed_ms))

    # --- Fresh-iid empirical permutation test ---
    def _test_fresh_iid(self, X_in, Y_in, seed: Optional[int] = None) -> TestResult:
        """
        Fresh-iid branch where fresh batches are sampled WITH REPLACEMENT
        from the empirical law of the CURRENT reference pool Y.
        """
        xp = self.xp
        cfg = self.cfg

        X = xp.asarray(X_in, dtype=xp.float64)
        Y = xp.asarray(Y_in, dtype=xp.float64)

        rng = self._rng(seed)
        t0 = time.perf_counter()

        N = X.shape[0]
        K = Y.shape[0]
        n_tot = N + K

        T_obs = self._compute_T_obs_fresh_iid_empirical(X, Y, rng)
        Z = xp.vstack([X, Y])

        gt = eq = ge = 0
        perms_used = 0
        C = max(1, int(cfg.chunk))
        b = 0
        allow_early = (cfg.decision in ("pvalue", "midp", "randomized"))

        while b < cfg.B:
            n_this = min(C, cfg.B - b)

            if cfg.antithetic:
                pairs = n_this // 2
                remainder = n_this % 2

                for _ in range(pairs):
                    perm = rng.permutation(n_tot)

                    for p in (perm, perm[::-1]):
                        Zp = Z[p]
                        Yp = Zp[:K]
                        Xp = Zp[K:]

                        Tb = self._compute_T_obs_fresh_iid_empirical(
                            Xp, Yp, rng)

                        if Tb > T_obs:
                            gt += 1
                        elif Tb == T_obs:
                            eq += 1
                        ge = gt + eq

                        b += 1
                        perms_used = b

                        if allow_early and b >= cfg.min_b_check:
                            if cfg.stop_ci == "wilson":
                                lo, hi = self._wilson_ci(ge, b, cfg.delta_ci)
                            else:
                                lo, hi = self._hoeffding_ci(
                                    ge, b, cfg.delta_ci)
                            if hi <= cfg.alpha or lo > cfg.alpha:
                                break

                        if b >= cfg.B:
                            break

                    if b >= cfg.B:
                        break

                    if allow_early and b >= cfg.min_b_check:
                        if cfg.stop_ci == "wilson":
                            lo, hi = self._wilson_ci(ge, b, cfg.delta_ci)
                        else:
                            lo, hi = self._hoeffding_ci(ge, b, cfg.delta_ci)
                        if hi <= cfg.alpha or lo > cfg.alpha:
                            break

                if remainder and b < cfg.B:
                    perm = rng.permutation(n_tot)
                    Zp = Z[perm]
                    Yp = Zp[:K]
                    Xp = Zp[K:]

                    Tb = self._compute_T_obs_fresh_iid_empirical(Xp, Yp, rng)

                    if Tb > T_obs:
                        gt += 1
                    elif Tb == T_obs:
                        eq += 1
                    ge = gt + eq

                    b += 1
                    perms_used = b

            else:
                for _ in range(n_this):
                    perm = rng.permutation(n_tot)
                    Zp = Z[perm]
                    Yp = Zp[:K]
                    Xp = Zp[K:]

                    Tb = self._compute_T_obs_fresh_iid_empirical(Xp, Yp, rng)

                    if Tb > T_obs:
                        gt += 1
                    elif Tb == T_obs:
                        eq += 1
                    ge = gt + eq

                    b += 1
                    perms_used = b

                    if allow_early and b >= cfg.min_b_check:
                        if cfg.stop_ci == "wilson":
                            lo, hi = self._wilson_ci(ge, b, cfg.delta_ci)
                        else:
                            lo, hi = self._hoeffding_ci(ge, b, cfg.delta_ci)
                        if hi <= cfg.alpha or lo > cfg.alpha:
                            break

            if allow_early and b >= cfg.min_b_check:
                if cfg.stop_ci == "wilson":
                    lo, hi = self._wilson_ci(ge, b, cfg.delta_ci)
                else:
                    lo, hi = self._hoeffding_ci(ge, b, cfg.delta_ci)
                if hi <= cfg.alpha or lo > cfg.alpha:
                    break

        b = max(1, perms_used)
        p_perm = (1 + ge) / (b + 1)
        p_mid = (gt + 0.5 * eq) / b

        if cfg.decision == "pvalue":
            reject = (p_perm <= cfg.alpha)
        elif cfg.decision == "midp":
            reject = (p_mid <= cfg.alpha)
        elif cfg.decision == "randomized":
            p_lower = (1 + gt) / (b + 1)
            if p_lower > cfg.alpha:
                reject = False
            elif eq == 0:
                reject = (p_perm <= cfg.alpha)
            else:
                omega = (cfg.alpha - p_lower) / (eq / (b + 1))
                omega = float(min(max(omega, 0.0), 1.0))
                reject = (_np.random.default_rng().uniform() <= omega)
        else:
            reject = (p_perm <= cfg.alpha)

        elapsed_ms = 1e3 * (time.perf_counter() - t0)
        return TestResult(bool(reject), float(p_perm), float(p_mid), float(T_obs), int(perms_used), float(elapsed_ms))

    # --- Public API ---
    def test(self, X_in, Y_in, seed: Optional[int] = None) -> TestResult:
        """
        Run one conditional permutation test on given samples X, Y.
        """
        if self.cfg.reference_mode == "fresh_iid":
            return self._test_fresh_iid(X_in, Y_in, seed=seed)
        if self.cfg.reference_mode == "shared_pool":
            return self._test_shared_pool(X_in, Y_in, seed=seed)
        raise ValueError("reference_mode must be 'shared_pool' or 'fresh_iid'")


# ---------------- Scenario builders (for H0/H1) ----------------

def make_gen_H0(null_mode: str, d: int, N: int, K: int, rho: float = 0.5, nu: int = 5):
    """
    Returns a callable rng -> (X, Y) under H0.
    """
    def _gen(rng, runner: RankGTest):
        xp = runner.xp
        if null_mode == "null-gauss":
            S = xp.eye(d, dtype=xp.float64)
            X = runner._draw_gauss(N, xp.zeros(d), S, rng)
            Y = runner._draw_gauss(K, xp.zeros(d), S, rng)
            return X, Y
        if null_mode == "null-copula":
            S = runner._corr_matrix(d, rho)
            X = runner._draw_gauss(N, xp.zeros(d), S, rng)
            Y = runner._draw_gauss(K, xp.zeros(d), S, rng)
            return X, Y
        if null_mode == "null-studentt":
            X = runner._draw_studentt(N, d, nu, rng)
            Y = runner._draw_studentt(K, d, nu, rng)
            return X, Y
        raise ValueError("unknown null_mode")
    return _gen


def make_gen_H1(scenario: str, d: int, N: int, K: int,
                delta: float = 0.3, direction: str = "e1",
                scale: float = 1.5, rho_x: float = 0.7, rho_y: float = 0.0,
                nu_x: int = 5, nu_y: int = 10, eps: float = 0.1):
    """
    Returns a callable rng -> (X, Y) under H1.
    """
    def _gen(rng, runner: RankGTest):
        xp = runner.xp
        if scenario == "shift":
            if direction == "random":
                u = xp.asarray(rng.normal(size=d), dtype=xp.float64)
                u = u / (xp.linalg.norm(u) + 1e-12)
            else:
                u = xp.zeros(d, dtype=xp.float64)
                u[0] = 1.0
            S = xp.eye(d, dtype=xp.float64)
            X = runner._draw_gauss(N, delta * u, S, rng)
            Y = runner._draw_gauss(K, xp.zeros(d), S, rng)
            return X, Y
        if scenario == "scale":
            Sx = (scale**2) * xp.eye(d, dtype=xp.float64)
            Sy = xp.eye(d, dtype=xp.float64)
            X = runner._draw_gauss(N, xp.zeros(d), Sx, rng)
            Y = runner._draw_gauss(K, xp.zeros(d), Sy, rng)
            return X, Y
        if scenario == "corr":
            Sx = runner._corr_matrix(d, rho_x)
            Sy = runner._corr_matrix(d, rho_y)
            X = runner._draw_gauss(N, xp.zeros(d), Sx, rng)
            Y = runner._draw_gauss(K, xp.zeros(d), Sy, rng)
            return X, Y
        if scenario == "tdf":
            X = runner._draw_studentt(N, d, nu_x, rng)
            Y = runner._draw_studentt(K, d, nu_y, rng)
            return X, Y
        if scenario == "mixture":
            n1 = int(rng.binomial(N, eps))
            n0 = N - n1
            X0 = runner._draw_gauss(
                n0, runner.xp.zeros(d), runner.xp.eye(d), rng)
            mu1 = runner.xp.zeros(d, dtype=runner.xp.float64)
            mu1[0] = delta
            X1 = runner._draw_gauss(n1, mu1, runner.xp.eye(d), rng)
            X = runner.xp.vstack([X0, X1])
            idx = rng.permutation(N)
            X = X[idx]
            Y = runner._draw_gauss(K, runner.xp.zeros(d),
                                   runner.xp.eye(d), rng)
            return X, Y
        raise ValueError("unknown scenario")
    return _gen


# ---------------- Monte-Carlo drivers ----------------

def _run_one_rep(seed: int, cfg: RankGTestConfig, gen_callable, mode: str) -> Tuple[int, float, int]:
    runner = RankGTest(cfg)
    rng = runner._rng(seed)
    X, Y = gen_callable(rng, runner)
    res = runner.test(X, Y, seed=seed)
    return int(res.reject), float(res.elapsed_ms), int(res.perms_used)


def _se_binom(p: float, R: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(1, R))


def simulate_type1(cfg: RankGTestConfig, gen_H0, R: int = 500, seed: int = 2026, jobs: Optional[int] = None):
    spawned = _np.random.SeedSequence(seed).spawn(R)
    seeds = [int(s.generate_state(1)[0]) for s in spawned]
    jobs = jobs or (os.cpu_count() or 1)

    if jobs == 1:
        outs = [_run_one_rep(s, cfg, gen_H0, "type1") for s in seeds]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            outs = list(ex.map(lambda ss: _run_one_rep(
                ss, cfg, gen_H0, "type1"), seeds))

    rej, times, used = zip(*outs)
    phat = sum(rej) / R
    se = _se_binom(phat, R)
    return {
        "alpha_hat": phat,
        "se": se,
        "avg_ms": sum(times) / R,
        "avg_perms_used": sum(used) / R,
    }


def simulate_power(cfg: RankGTestConfig, gen_H1, R: int = 500, seed: int = 2026, jobs: Optional[int] = None):
    spawned = _np.random.SeedSequence(seed).spawn(R)
    seeds = [int(s.generate_state(1)[0]) for s in spawned]
    jobs = jobs or (os.cpu_count() or 1)

    if jobs == 1:
        outs = [_run_one_rep(s, cfg, gen_H1, "power") for s in seeds]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            outs = list(ex.map(lambda ss: _run_one_rep(
                ss, cfg, gen_H1, "power"), seeds))

    rej, times, used = zip(*outs)
    phat = sum(rej) / R
    se = _se_binom(phat, R)
    return {
        "power_hat": phat,
        "se": se,
        "avg_ms": sum(times) / R,
        "avg_perms_used": sum(used) / R,
    }


# ---------------- Generic harness to compare methods ----------------

def make_rank_gtest_method(cfg: RankGTestConfig):
    runner = RankGTest(cfg)

    def _method(X, Y, rng, alpha):
        seed = int(_np.random.SeedSequence(
            rng.integers(0, 2**32 - 1)).generate_state(1)[0])
        return runner.test(X, Y, seed=seed)
    return _method


def evaluate_power_methods(methods: Dict[str, Callable], gen_H1, R: int = 500, seed: int = 2026,
                           jobs: Optional[int] = None, alpha: float = 0.05):
    """
    Runs the SAME draws for all methods to get paired estimates.
    Returns dict: name -> {'power_hat', 'se', 'avg_ms'}.
    """
    rng_master = _np.random.default_rng(seed)
    jobs = jobs or (os.cpu_count() or 1)

    def _one_rep(srep: int):
        cfg_dummy = RankGTestConfig(gpu="off")
        dummy = RankGTest(cfg_dummy)
        rng = _np.random.default_rng(srep)
        X, Y = gen_H1(rng, dummy)
        out = {}
        for name, meth in methods.items():
            t0 = time.perf_counter()
            r = meth(X, Y, rng, alpha)
            ms = 1e3 * (time.perf_counter() - t0)
            out[name] = (int(getattr(r, "reject", bool(r))), float(ms))
        return out

    seeds_rep = rng_master.integers(0, 2**32 - 1, size=R)

    if jobs == 1:
        all_out = [_one_rep(int(s)) for s in seeds_rep]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            all_out = list(ex.map(_one_rep, [int(s) for s in seeds_rep]))

    acc: Dict[str, Dict[str, float]] = {}
    for rep_out in all_out:
        for name, (rej, ms) in rep_out.items():
            if name not in acc:
                acc[name] = {"rej": 0.0, "ms": 0.0, "n": 0}
            acc[name]["rej"] += rej
            acc[name]["ms"] += ms
            acc[name]["n"] += 1

    summary = {}
    for name, v in acc.items():
        n = int(v["n"])
        ph = v["rej"] / n
        se = _se_binom(ph, n)
        avg_ms = v["ms"] / n
        summary[name] = {"power_hat": ph, "se": se, "avg_ms": avg_ms}
    return summary


# ---------------- Functional helpers re-exported for subspace code ----------------

def _get_backend(gpu: str = "off", device: int = 0):
    xp, _, _ = _try_enable_cupy(gpu, device)
    return xp


def precompute_orders(
    Z, *, tie: str = "right", jitter_scale: float = 1e-12,
    gpu: str = "off", device: int = 0, seed: Optional[int] = None
) -> Dict[str, Any]:
    xp = _get_backend(gpu, device)
    Z = xp.asarray(Z)
    n_tot, d = Z.shape

    if tie == "jitter":
        if seed is not None:
            rng = _np.random.default_rng(int(seed))
            J = rng.normal(scale=jitter_scale, size=Z.shape)
            J = xp.asarray(J, dtype=xp.float64)
        else:
            J = xp.asarray(_np.random.default_rng().normal(
                scale=jitter_scale, size=Z.shape), dtype=xp.float64)
        Zw = Z + J
        side = "right"
    else:
        Zw = Z
        side = tie

    orders, pos_all, last_right_all, first_left_all = [], [], [], []
    for j in range(d):
        order = xp.argsort(Zw[:, j], kind="mergesort")
        vals = Zw[order, j]
        pos = xp.empty(n_tot, dtype=xp.int64)
        pos[order] = xp.arange(n_tot, dtype=xp.int64)

        _, inv, counts = xp.unique(
            vals, return_inverse=True, return_counts=True)
        first_of_group = xp.cumsum(
            xp.concatenate([xp.array([0], dtype=xp.int64),
                           counts[:-1].astype(xp.int64)])
        )
        last_of_group = first_of_group + counts.astype(xp.int64) - 1
        first_left = first_of_group[inv]
        last_right = last_of_group[inv]

        orders.append(order)
        pos_all.append(pos)
        last_right_all.append(last_right.astype(xp.int64))
        first_left_all.append(first_left.astype(xp.int64))

    return {
        "orders": orders,
        "pos": pos_all,
        "last_right": last_right_all,
        "first_left": first_left_all,
        "side": side,
    }


def ranks_searchsorted(
    indices, Iref, Z_cols, *, side: str = "right", jitter=None,
    gpu: str = "off", device: int = 0
):
    xp = _get_backend(gpu, device)
    indices = xp.asarray(indices, dtype=xp.int64)
    Iref = xp.asarray(Iref, dtype=xp.int64)
    d = len(Z_cols)
    m_cols = []
    for j in range(d):
        Zj = xp.asarray(Z_cols[j])
        if jitter is not None:
            Jj = jitter[j] if getattr(jitter, "ndim", 1) > 1 else jitter
            Zj = Zj + xp.asarray(Jj)
        Rj = Zj[Iref]
        Rj_sorted = xp.sort(Rj)
        zj = Zj[indices]
        m_cols.append(xp.searchsorted(Rj_sorted, zj, side=side))
    return xp.stack(m_cols, axis=1)


def ranks_prefixsum(
    indices, sel_mask, ords: Dict[str, Any], *, side_pick: str,
    gpu: str = "off", device: int = 0
):
    xp = _get_backend(gpu, device)
    indices = xp.asarray(indices, dtype=xp.int64)
    sel_mask = xp.asarray(sel_mask).astype(xp.uint8)
    d = len(ords["orders"])
    m_cols = []
    for j in range(d):
        order_j = ords["orders"][j]
        pos_j = ords["pos"][j]
        pick = ords[side_pick][j]
        s_ord = sel_mask[order_j].astype(xp.int8)
        S = xp.cumsum(s_ord, dtype=xp.int64)
        pj = pos_j[indices]
        pr = pick[pj]
        m_cols.append(S[pr])
    return xp.stack(m_cols, axis=1)


def fine_linear_index(m, K_ref: int, *, gpu: str = "off", device: int = 0):
    xp = _get_backend(gpu, device)
    m = xp.asarray(m, dtype=xp.int64)
    d = int(m.shape[1])
    base = int(K_ref) + 1
    w_fine = (base ** xp.arange(d - 1, -1, -1,
              dtype=xp.int64)).astype(xp.int64)
    idx = (m * w_fine).sum(axis=1)
    M_eff = int(base ** d)
    return idx, w_fine, M_eff


def stat_from_counts(
    HX, HY, *, N_local: int, Ky_local: int, alpha0: float, kind: str = "gtest",
    gpu: str = "off", device: int = 0
) -> float:
    xp = _get_backend(gpu, device)
    HX = xp.asarray(HX, dtype=xp.float64)
    HY = xp.asarray(HY, dtype=xp.float64)
    M = float(HX.size)
    denom = float(Ky_local) + float(alpha0) * M
    qhat = (HY + float(alpha0)) / denom
    EX = float(N_local) * qhat

    if kind == "chi2":
        T = xp.where(EX > 0.0, (HX - EX) ** 2 / EX, 0.0).sum()
        return float(T)
    if kind == "gtest":
        ratio = xp.where(HX > 0.0, HX / EX, 1.0)
        contrib = xp.where(HX > 0.0, HX * xp.log(ratio), 0.0)
        T = 2.0 * contrib.sum()
        return float(T)
    raise ValueError("kind must be 'chi2' or 'gtest'")
