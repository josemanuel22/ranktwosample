# -------------------------------------------------------------------
# Subspace-sliced non-symmetric rank χ² / G-test (permutation-based)
#
# Modes:
#   • reference_mode="shared_pool":
#       old practical version; ranks use an active subset of Y
#   • reference_mode="fresh_iid":
#       practical fresh-iid version; for each observation z and each slice,
#       draw a fresh batch of size K_ref WITH REPLACEMENT from the empirical
#       law of the CURRENT Y-pool
#
# Reuses functional helpers from rank_two_sample.py
# -------------------------------------------------------------------
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence, Optional, Tuple, Any

import numpy as np

from ranktwosample.rank_two_sample import (
    RankGTestConfig, TestResult, RankGTest,
    precompute_orders, ranks_searchsorted, ranks_prefixsum,
    fine_linear_index, stat_from_counts,
)

Array = np.ndarray


@dataclass
class SubspaceOpts:
    L: int = 32                 # number of subspaces/slices
    k_dim: int = 1              # subspace dimension
    dims_list: Optional[List[Sequence[int]]] = None
    dedup: bool = True
    pca: bool = False
    pca_center: bool = True


class RankGTestSubspaces:
    """
    Same decision logic as RankGTest, but uses a statistic formed by aggregating
    per-subspace χ² / G-test contributions over L slices.
    """

    def __init__(self, cfg: RankGTestConfig, subs: SubspaceOpts):
        self.cfg = cfg
        self.subs = subs
        self._runner = RankGTest(cfg)
        self.xp = self._runner.xp

    # ---------------- internal helpers ----------------

    def _pca_dims_list(self, Z_np: np.ndarray, d: int) -> List[Tuple[int, ...]]:
        L = int(self.subs.L)
        k = int(self.subs.k_dim)
        if not (1 <= k <= d):
            raise ValueError("k_dim must be in [1, d].")

        Xc = Z_np - \
            Z_np.mean(
                axis=0, keepdims=True) if self.subs.pca_center else Z_np.copy()
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        components = Vt

        dims_list: List[Tuple[int, ...]] = []
        seen = set()
        for i in range(d):
            load = np.abs(components[i])
            idx = np.argsort(load)[::-1][:k]
            dims = tuple(sorted(int(j) for j in idx))
            if (not self.subs.dedup) or (dims not in seen):
                seen.add(dims)
                dims_list.append(dims)
                if len(dims_list) >= L:
                    break
        return dims_list

    def _random_dims_list(self, d: int, rng) -> List[Tuple[int, ...]]:
        L = int(self.subs.L)
        k = int(self.subs.k_dim)
        if not (1 <= k <= d):
            raise ValueError("k_dim must be in [1, d].")

        dims_list: List[Tuple[int, ...]] = []
        if not self.subs.dedup:
            for _ in range(L):
                sel = list(map(int, rng.choice(d, size=k, replace=False)))
                dims_list.append(tuple(sorted(sel)))
            return dims_list

        seen = set()
        max_tries = max(10 * L, 1000)
        tries = 0
        while len(dims_list) < L and tries < max_tries:
            sel = tuple(sorted(int(i)
                        for i in rng.choice(d, size=k, replace=False)))
            if sel not in seen:
                seen.add(sel)
                dims_list.append(sel)
            tries += 1
        return dims_list

    def _sample_dims_list(self, X: Array, Y: Array, rng) -> List[Tuple[int, ...]]:
        d = int(X.shape[1])

        if self.subs.dims_list is not None:
            dims_list: List[Tuple[int, ...]] = []
            for dims in self.subs.dims_list:
                dd = tuple(sorted(int(i) for i in dims))
                if len(dd) == 0 or len(dd) > d or any((j < 0 or j >= d) for j in dd):
                    raise ValueError(f"Invalid dims in dims_list: {dims}")
                dims_list.append(dd)
            return dims_list

        if self.subs.pca:
            Z_np = np.asarray(np.vstack([np.asarray(X), np.asarray(Y)]))
            dims_list = self._pca_dims_list(Z_np, d)

            if len(dims_list) < int(self.subs.L):
                need = int(self.subs.L) - len(dims_list)
                existing = set(dims_list)
                while need > 0:
                    sel = tuple(sorted(int(i) for i in rng.choice(
                        d, size=int(self.subs.k_dim), replace=False)))
                    if (not self.subs.dedup) or (sel not in existing):
                        dims_list.append(sel)
                        existing.add(sel)
                        need -= 1
            return dims_list

        return self._random_dims_list(d, rng)

    @staticmethod
    def _slice_ords(ords_full: dict, dims: Sequence[int]) -> dict:
        return {
            "orders":      [ords_full["orders"][j] for j in dims],
            "pos":         [ords_full["pos"][j] for j in dims],
            "last_right":  [ords_full["last_right"][j] for j in dims],
            "first_left":  [ords_full["first_left"][j] for j in dims],
            "side":         ords_full["side"],
        }

    def _resolve_alpha0(self, Ky: int, M_eff: int) -> float:
        alpha0 = self.cfg.alpha0
        if alpha0 < 0:
            alpha0 = max(1e-6, 0.02 * (Ky / max(1, M_eff)))
        return float(alpha0)

    # ---------------- shared_pool mode ----------------

    def _compute_T_obs_sum_shared_pool(self, X, Y, rng) -> Tuple[float, Tuple[Any, ...]]:
        xp = self.xp
        cfg = self.cfg

        N, d = X.shape
        K = Y.shape[0]
        K_ref = cfg.K_ref if cfg.K_ref is not None else int(K)
        if not (1 <= K_ref <= K):
            raise ValueError("K_ref must be in [1, K]")

        Z = xp.vstack([X, Y])
        Z_cols = [Z[:, j] for j in range(d)]

        jitter = None
        ss_side = cfg.tie
        if cfg.tie == "jitter":
            jitter = xp.asarray(rng.normal(
                scale=cfg.jitter_scale, size=(d, N + K)))
            ss_side = "right"

        need_prefix = (cfg.engine in ("auto", "prefixsum")) or (
            cfg.engine == "searchsorted" and K_ref > cfg.kref_switch
        )
        ords_full = None
        side_pick = None
        if need_prefix:
            ords_full = precompute_orders(
                Z,
                tie=("right" if cfg.tie == "jitter" else cfg.tie),
                jitter_scale=cfg.jitter_scale,
                gpu=cfg.gpu,
                device=cfg.device,
                seed=0,
            )
            side_pick = "last_right" if ords_full["side"] == "right" else "first_left"

        if cfg.engine == "auto":
            use_ss = (K_ref <= cfg.kref_switch)
        elif cfg.engine == "searchsorted":
            use_ss = True
        elif cfg.engine == "prefixsum":
            use_ss = False
        else:
            raise ValueError("engine must be auto|searchsorted|prefixsum")

        dims_list = self._sample_dims_list(X, Y, rng)

        # fixed reference positions reused across permutations
        ref_pos_list = [rng.permutation(K)[:K_ref]
                        for _ in range(len(dims_list))]
        Iref_list = [xp.int64(N) + rp for rp in ref_pos_list]

        idx_X_obs = xp.arange(N, dtype=xp.int64)
        T_sum = 0.0

        for dims, Iref in zip(dims_list, Iref_list):
            Zc = [Z_cols[j] for j in dims]
            jit = None if jitter is None else jitter[list(dims), :]

            if use_ss:
                mX = ranks_searchsorted(
                    idx_X_obs, Iref, Zc, side=ss_side, jitter=jit,
                    gpu=cfg.gpu, device=cfg.device
                )
                if cfg.use_fullY_for_HY:
                    Iy_pool_obs = xp.arange(N, N + K, dtype=xp.int64)
                    mY = ranks_searchsorted(
                        Iy_pool_obs, Iref, Zc, side=ss_side, jitter=jit,
                        gpu=cfg.gpu, device=cfg.device
                    )
                else:
                    mY = ranks_searchsorted(
                        Iref, Iref, Zc, side=ss_side, jitter=jit,
                        gpu=cfg.gpu, device=cfg.device
                    )
            else:
                ords_slice = self._slice_ords(ords_full, dims)
                sel = xp.zeros(N + K, dtype=xp.uint8)
                sel[Iref] = 1
                mX = ranks_prefixsum(
                    idx_X_obs, sel, ords_slice, side_pick=side_pick,
                    gpu=cfg.gpu, device=cfg.device
                )
                if cfg.use_fullY_for_HY:
                    Iy_pool_obs = xp.arange(N, N + K, dtype=xp.int64)
                    mY = ranks_prefixsum(
                        Iy_pool_obs, sel, ords_slice, side_pick=side_pick,
                        gpu=cfg.gpu, device=cfg.device
                    )
                else:
                    mY = ranks_prefixsum(
                        Iref, sel, ords_slice, side_pick=side_pick,
                        gpu=cfg.gpu, device=cfg.device
                    )

            idxX, _, _ = fine_linear_index(
                mX, K_ref, gpu=cfg.gpu, device=cfg.device)
            idxY, _, _ = fine_linear_index(
                mY, K_ref, gpu=cfg.gpu, device=cfg.device)
            M_eff = int((K_ref + 1) ** len(dims))
            HX = xp.bincount(idxX, minlength=M_eff)
            HY = xp.bincount(idxY, minlength=M_eff)
            Ky = int(float(HY.sum()))
            alpha0 = self._resolve_alpha0(Ky, M_eff)

            T_sum += stat_from_counts(
                HX, HY,
                N_local=N, Ky_local=Ky, alpha0=alpha0,
                kind=cfg.stat, gpu=cfg.gpu, device=cfg.device
            )

        env = (
            K_ref, d, use_ss, ords_full, side_pick, jitter, ss_side, Z_cols,
            dims_list, N, K, ref_pos_list
        )
        return float(T_sum), env

    def _test_shared_pool(self, X_in: Array, Y_in: Array, seed: Optional[int] = None) -> TestResult:
        xp = self.xp
        cfg = self.cfg

        X = xp.asarray(X_in)
        Y = xp.asarray(Y_in)
        rng = self._runner._rng(seed)
        tstart = time.perf_counter()

        T_obs, env = self._compute_T_obs_sum_shared_pool(X, Y, rng)
        (K_ref, d, use_ss, ords_full, side_pick, jitter, ss_side, Z_cols,
         dims_list, N, K, ref_pos_list) = env

        n_tot = N + K
        ge = gt = eq = 0
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
                        Tb = 0.0

                        for dims, ref_pos in zip(dims_list, ref_pos_list):
                            Iref = Iy_pool[ref_pos]
                            Zc = [Z_cols[j] for j in dims]
                            jit = None if jitter is None else jitter[list(
                                dims), :]

                            if use_ss:
                                mX = ranks_searchsorted(
                                    Ix_pool, Iref, Zc, side=ss_side, jitter=jit,
                                    gpu=cfg.gpu, device=cfg.device
                                )
                                if cfg.use_fullY_for_HY:
                                    mY = ranks_searchsorted(
                                        Iy_pool, Iref, Zc, side=ss_side, jitter=jit,
                                        gpu=cfg.gpu, device=cfg.device
                                    )
                                else:
                                    mY = ranks_searchsorted(
                                        Iref, Iref, Zc, side=ss_side, jitter=jit,
                                        gpu=cfg.gpu, device=cfg.device
                                    )
                            else:
                                sel = xp.zeros(n_tot, dtype=xp.uint8)
                                sel[Iref] = 1
                                ords_slice = self._slice_ords(ords_full, dims)
                                mX = ranks_prefixsum(
                                    Ix_pool, sel, ords_slice, side_pick=side_pick,
                                    gpu=cfg.gpu, device=cfg.device
                                )
                                if cfg.use_fullY_for_HY:
                                    mY = ranks_prefixsum(
                                        Iy_pool, sel, ords_slice, side_pick=side_pick,
                                        gpu=cfg.gpu, device=cfg.device
                                    )
                                else:
                                    mY = ranks_prefixsum(
                                        Iref, sel, ords_slice, side_pick=side_pick,
                                        gpu=cfg.gpu, device=cfg.device
                                    )

                            idxX, _, _ = fine_linear_index(
                                mX, K_ref, gpu=cfg.gpu, device=cfg.device)
                            idxY, _, _ = fine_linear_index(
                                mY, K_ref, gpu=cfg.gpu, device=cfg.device)
                            M_eff = int((K_ref + 1) ** len(dims))
                            HX = xp.bincount(idxX, minlength=M_eff)
                            HY = xp.bincount(idxY, minlength=M_eff)
                            Ky = int(float(HY.sum()))
                            alpha0 = self._resolve_alpha0(Ky, M_eff)

                            Tb += stat_from_counts(
                                HX, HY,
                                N_local=N, Ky_local=Ky, alpha0=alpha0,
                                kind=cfg.stat, gpu=cfg.gpu, device=cfg.device
                            )

                        if Tb > T_obs:
                            gt += 1
                        elif Tb == T_obs:
                            eq += 1
                        ge = gt + eq
                        b += 1
                        perms_used = b

                        if allow_early and b >= cfg.min_b_check:
                            if cfg.stop_ci == "wilson":
                                lo, hi = self._runner._wilson_ci(
                                    ge, b, cfg.delta_ci)
                            else:
                                lo, hi = self._runner._hoeffding_ci(
                                    ge, b, cfg.delta_ci)
                            if hi <= cfg.alpha or lo > cfg.alpha:
                                break

                        if b >= cfg.B:
                            break

                    if b >= cfg.B:
                        break

                    if allow_early and b >= cfg.min_b_check:
                        if cfg.stop_ci == "wilson":
                            lo, hi = self._runner._wilson_ci(
                                ge, b, cfg.delta_ci)
                        else:
                            lo, hi = self._runner._hoeffding_ci(
                                ge, b, cfg.delta_ci)
                        if hi <= cfg.alpha or lo > cfg.alpha:
                            break

                if remainder and b < cfg.B:
                    perm = rng.permutation(n_tot)
                    Iy_pool = perm[:K]
                    Ix_pool = perm[K:]
                    Tb = 0.0

                    for dims, ref_pos in zip(dims_list, ref_pos_list):
                        Iref = Iy_pool[ref_pos]
                        Zc = [Z_cols[j] for j in dims]
                        jit = None if jitter is None else jitter[list(dims), :]

                        if use_ss:
                            mX = ranks_searchsorted(
                                Ix_pool, Iref, Zc, side=ss_side, jitter=jit,
                                gpu=cfg.gpu, device=cfg.device
                            )
                            if cfg.use_fullY_for_HY:
                                mY = ranks_searchsorted(
                                    Iy_pool, Iref, Zc, side=ss_side, jitter=jit,
                                    gpu=cfg.gpu, device=cfg.device
                                )
                            else:
                                mY = ranks_searchsorted(
                                    Iref, Iref, Zc, side=ss_side, jitter=jit,
                                    gpu=cfg.gpu, device=cfg.device
                                )
                        else:
                            sel = xp.zeros(n_tot, dtype=xp.uint8)
                            sel[Iref] = 1
                            ords_slice = self._slice_ords(ords_full, dims)
                            mX = ranks_prefixsum(
                                Ix_pool, sel, ords_slice, side_pick=side_pick,
                                gpu=cfg.gpu, device=cfg.device
                            )
                            if cfg.use_fullY_for_HY:
                                mY = ranks_prefixsum(
                                    Iy_pool, sel, ords_slice, side_pick=side_pick,
                                    gpu=cfg.gpu, device=cfg.device
                                )
                            else:
                                mY = ranks_prefixsum(
                                    Iref, sel, ords_slice, side_pick=side_pick,
                                    gpu=cfg.gpu, device=cfg.device
                                )

                        idxX, _, _ = fine_linear_index(
                            mX, K_ref, gpu=cfg.gpu, device=cfg.device)
                        idxY, _, _ = fine_linear_index(
                            mY, K_ref, gpu=cfg.gpu, device=cfg.device)
                        M_eff = int((K_ref + 1) ** len(dims))
                        HX = xp.bincount(idxX, minlength=M_eff)
                        HY = xp.bincount(idxY, minlength=M_eff)
                        Ky = int(float(HY.sum()))
                        alpha0 = self._resolve_alpha0(Ky, M_eff)

                        Tb += stat_from_counts(
                            HX, HY,
                            N_local=N, Ky_local=Ky, alpha0=alpha0,
                            kind=cfg.stat, gpu=cfg.gpu, device=cfg.device
                        )

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
                    Tb = 0.0

                    for dims, ref_pos in zip(dims_list, ref_pos_list):
                        Iref = Iy_pool[ref_pos]
                        Zc = [Z_cols[j] for j in dims]
                        jit = None if jitter is None else jitter[list(dims), :]

                        if use_ss:
                            mX = ranks_searchsorted(
                                Ix_pool, Iref, Zc, side=ss_side, jitter=jit,
                                gpu=cfg.gpu, device=cfg.device
                            )
                            if cfg.use_fullY_for_HY:
                                mY = ranks_searchsorted(
                                    Iy_pool, Iref, Zc, side=ss_side, jitter=jit,
                                    gpu=cfg.gpu, device=cfg.device
                                )
                            else:
                                mY = ranks_searchsorted(
                                    Iref, Iref, Zc, side=ss_side, jitter=jit,
                                    gpu=cfg.gpu, device=cfg.device
                                )
                        else:
                            sel = xp.zeros(n_tot, dtype=xp.uint8)
                            sel[Iref] = 1
                            ords_slice = self._slice_ords(ords_full, dims)
                            mX = ranks_prefixsum(
                                Ix_pool, sel, ords_slice, side_pick=side_pick,
                                gpu=cfg.gpu, device=cfg.device
                            )
                            if cfg.use_fullY_for_HY:
                                mY = ranks_prefixsum(
                                    Iy_pool, sel, ords_slice, side_pick=side_pick,
                                    gpu=cfg.gpu, device=cfg.device
                                )
                            else:
                                mY = ranks_prefixsum(
                                    Iref, sel, ords_slice, side_pick=side_pick,
                                    gpu=cfg.gpu, device=cfg.device
                                )

                        idxX, _, _ = fine_linear_index(
                            mX, K_ref, gpu=cfg.gpu, device=cfg.device)
                        idxY, _, _ = fine_linear_index(
                            mY, K_ref, gpu=cfg.gpu, device=cfg.device)
                        M_eff = int((K_ref + 1) ** len(dims))
                        HX = xp.bincount(idxX, minlength=M_eff)
                        HY = xp.bincount(idxY, minlength=M_eff)
                        Ky = int(float(HY.sum()))
                        alpha0 = self._resolve_alpha0(Ky, M_eff)

                        Tb += stat_from_counts(
                            HX, HY,
                            N_local=N, Ky_local=Ky, alpha0=alpha0,
                            kind=cfg.stat, gpu=cfg.gpu, device=cfg.device
                        )

                    if Tb > T_obs:
                        gt += 1
                    elif Tb == T_obs:
                        eq += 1
                    ge = gt + eq
                    b += 1
                    perms_used = b

                    if allow_early and b >= cfg.min_b_check:
                        if cfg.stop_ci == "wilson":
                            lo, hi = self._runner._wilson_ci(
                                ge, b, cfg.delta_ci)
                        else:
                            lo, hi = self._runner._hoeffding_ci(
                                ge, b, cfg.delta_ci)
                        if hi <= cfg.alpha or lo > cfg.alpha:
                            break

            if allow_early and b >= cfg.min_b_check:
                if cfg.stop_ci == "wilson":
                    lo, hi = self._runner._wilson_ci(ge, b, cfg.delta_ci)
                else:
                    lo, hi = self._runner._hoeffding_ci(ge, b, cfg.delta_ci)
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
                reject = (np.random.default_rng().uniform() <= omega)
        else:
            reject = (p_perm <= cfg.alpha)

        elapsed_ms = 1e3 * (time.perf_counter() - tstart)
        return TestResult(bool(reject), float(p_perm), float(p_mid), float(T_obs), int(perms_used), float(elapsed_ms))

    # ---------------- fresh_iid empirical mode ----------------

    def _slice_stat_fresh_iid_empirical(self, X, Y, dims, rng) -> float:
        """
        One slice contribution using fresh batches sampled WITH REPLACEMENT
        from the empirical law of the CURRENT Y-pool.
        """
        xp = self.xp
        cfg = self.cfg

        X = xp.asarray(X, dtype=xp.float64)
        Y = xp.asarray(Y, dtype=xp.float64)

        N = X.shape[0]
        K = Y.shape[0]
        K_ref = cfg.K_ref if cfg.K_ref is not None else int(K)
        if K_ref < 1:
            raise ValueError("K_ref must be >= 1.")

        Z_slice = xp.vstack([X[:, list(dims)], Y[:, list(dims)]])
        n_tot = N + K

        ref_batches_full = self._runner._sample_fresh_reference_batches_from_Y(
            Y=Y,
            n_items=n_tot,
            K_ref=K_ref,
            rng=rng,
        )
        ref_batches_slice = ref_batches_full[:, :, list(dims)]

        m = self._runner._ranks_fresh_batches(
            Z_slice,
            ref_batches_slice,
            tie=cfg.tie,
            jitter_scale=cfg.jitter_scale,
            rng=rng,
        )

        idx, _, M_eff = self._runner._fine_linear_index(m, K_ref)
        HX = xp.bincount(idx[:N], minlength=M_eff)
        HY = xp.bincount(idx[N:], minlength=M_eff)

        alpha0 = self._resolve_alpha0(K, M_eff)

        return float(self._runner._stat_from_counts(
            HX, HY,
            N_local=N,
            Ky_local=K,
            alpha0=alpha0,
            kind=cfg.stat,
        ))

    def _compute_T_obs_sum_fresh_iid(self, X, Y, rng):
        dims_list = self._sample_dims_list(X, Y, rng)
        T_sum = 0.0
        for dims in dims_list:
            T_sum += self._slice_stat_fresh_iid_empirical(X, Y, dims, rng)
        return float(T_sum), dims_list

    def _test_fresh_iid(self, X_in: Array, Y_in: Array, seed: Optional[int] = None) -> TestResult:
        xp = self.xp
        cfg = self.cfg

        X = xp.asarray(X_in, dtype=xp.float64)
        Y = xp.asarray(Y_in, dtype=xp.float64)

        rng = self._runner._rng(seed)
        tstart = time.perf_counter()

        N = X.shape[0]
        K = Y.shape[0]
        n_tot = N + K
        Z = xp.vstack([X, Y])

        T_obs, dims_list = self._compute_T_obs_sum_fresh_iid(X, Y, rng)

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

                        Tb = 0.0
                        for dims in dims_list:
                            Tb += self._slice_stat_fresh_iid_empirical(
                                Xp, Yp, dims, rng)

                        if Tb > T_obs:
                            gt += 1
                        elif Tb == T_obs:
                            eq += 1
                        ge = gt + eq
                        b += 1
                        perms_used = b

                        if allow_early and b >= cfg.min_b_check:
                            if cfg.stop_ci == "wilson":
                                lo, hi = self._runner._wilson_ci(
                                    ge, b, cfg.delta_ci)
                            else:
                                lo, hi = self._runner._hoeffding_ci(
                                    ge, b, cfg.delta_ci)
                            if hi <= cfg.alpha or lo > cfg.alpha:
                                break

                        if b >= cfg.B:
                            break

                    if b >= cfg.B:
                        break

                    if allow_early and b >= cfg.min_b_check:
                        if cfg.stop_ci == "wilson":
                            lo, hi = self._runner._wilson_ci(
                                ge, b, cfg.delta_ci)
                        else:
                            lo, hi = self._runner._hoeffding_ci(
                                ge, b, cfg.delta_ci)
                        if hi <= cfg.alpha or lo > cfg.alpha:
                            break

                if remainder and b < cfg.B:
                    perm = rng.permutation(n_tot)
                    Zp = Z[perm]
                    Yp = Zp[:K]
                    Xp = Zp[K:]

                    Tb = 0.0
                    for dims in dims_list:
                        Tb += self._slice_stat_fresh_iid_empirical(
                            Xp, Yp, dims, rng)

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

                    Tb = 0.0
                    for dims in dims_list:
                        Tb += self._slice_stat_fresh_iid_empirical(
                            Xp, Yp, dims, rng)

                    if Tb > T_obs:
                        gt += 1
                    elif Tb == T_obs:
                        eq += 1
                    ge = gt + eq
                    b += 1
                    perms_used = b

                    if allow_early and b >= cfg.min_b_check:
                        if cfg.stop_ci == "wilson":
                            lo, hi = self._runner._wilson_ci(
                                ge, b, cfg.delta_ci)
                        else:
                            lo, hi = self._runner._hoeffding_ci(
                                ge, b, cfg.delta_ci)
                        if hi <= cfg.alpha or lo > cfg.alpha:
                            break

            if allow_early and b >= cfg.min_b_check:
                if cfg.stop_ci == "wilson":
                    lo, hi = self._runner._wilson_ci(ge, b, cfg.delta_ci)
                else:
                    lo, hi = self._runner._hoeffding_ci(ge, b, cfg.delta_ci)
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
                reject = (np.random.default_rng().uniform() <= omega)
        else:
            reject = (p_perm <= cfg.alpha)

        elapsed_ms = 1e3 * (time.perf_counter() - tstart)
        return TestResult(bool(reject), float(p_perm), float(p_mid), float(T_obs), int(perms_used), float(elapsed_ms))

    # ---------------- public API ----------------

    def test(self, X_in: Array, Y_in: Array, seed: Optional[int] = None) -> TestResult:
        if self.cfg.reference_mode == "fresh_iid":
            return self._test_fresh_iid(X_in, Y_in, seed=seed)
        if self.cfg.reference_mode == "shared_pool":
            return self._test_shared_pool(X_in, Y_in, seed=seed)
        raise ValueError("reference_mode must be 'shared_pool' or 'fresh_iid'")
