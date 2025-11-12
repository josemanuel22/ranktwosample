#!/usr/bin/env python3
# -------------------------------------------------------------------
# Non-symmetric rank two-sample permutation test (χ² / G-test)
#  • Optional COARSENING via --G (bins per dimension); fine grid when omitted
#  • Permutation test only (no calibration files)
#  • Early stopping (Wilson/Hoeffding), antithetic pairs, chunking
#  • Rank engines: auto | searchsorted | prefixsum
#  • HY from full Y-pool (recommended, --fullY) or from Iref only
#  • Sparse G-test path (--sparse_gtest) avoids length-M arrays
#
# Modes:
#   --mode type1 : empirical Type-I under H0
#   --mode power : empirical power under H1 scenarios
# -------------------------------------------------------------------
import argparse, os, time, math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm.auto import tqdm

# ---------------- RNG ----------------
def make_rng(seed_or_ss):
    return np.random.default_rng(seed_or_ss)

# ---------------- Linear algebra helpers ----------------
def corr_matrix(d, rho):
    if d == 1:
        return np.array([[1.0]])
    S = np.full((d, d), rho, dtype=float)
    np.fill_diagonal(S, 1.0)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 1e-12, None)
    return (V * w) @ V.T

# ---------------- Draws ----------------
def draw_gauss(n, mean, cov, rng):
    mean = np.asarray(mean, dtype=float)
    return rng.multivariate_normal(mean, cov, size=n)

def draw_studentt(n, d, nu, rng):
    Z = rng.normal(size=(n, d))
    U = rng.chisquare(nu, size=n)
    return Z / np.sqrt(U[:, None] / nu)

# ---------------- Prefix-sum precomputation ----------------
def precompute_orders(Z, tie="right", jitter_scale=1e-12, rng=None):
    """
    Precompute stable argsort and tie run boundaries for prefix-sum rank queries.
    """
    n_tot, d = Z.shape
    if tie == "jitter":
        if rng is None:
            rng = np.random.default_rng(0)
        Zw = Z + rng.normal(scale=jitter_scale, size=Z.shape)
        side = "right"
    else:
        Zw = Z
        side = tie  # 'right' or 'left'

    orders, pos_all, last_right_all, first_left_all = [], [], [], []
    for j in range(d):
        order = np.argsort(Zw[:, j], kind="mergesort")
        vals = Zw[order, j]
        pos = np.empty(n_tot, dtype=np.int64)
        pos[order] = np.arange(n_tot, dtype=np.int64)

        last_right = np.empty(n_tot, dtype=np.int64)
        first_left = np.empty(n_tot, dtype=np.int64)
        start = 0
        for t in range(1, n_tot + 1):
            if t == n_tot or vals[t] != vals[t - 1]:
                last_right[start:t] = t - 1
                first_left[start:t] = start
                start = t

        orders.append(order)
        pos_all.append(pos)
        last_right_all.append(last_right)
        first_left_all.append(first_left)

    return {
        "orders": orders,
        "pos": pos_all,
        "last_right": last_right_all,
        "first_left": first_left_all,
        "side": side,
    }

# ---------------- Fine-grid indexing ----------------
def fine_linear_index(m, K_ref):
    """
    Map rank vectors m ∈ {0,...,K_ref}^d to linear indices in [0, (K_ref+1)^d - 1].
    """
    d = m.shape[1]
    base = K_ref + 1
    w_fine = (base ** np.arange(d - 1, -1, -1)).astype(np.int64)  # weights base^(d-1-i)
    return (m * w_fine).sum(axis=1), w_fine, (base ** d)

# ---------------- Generic mixed-radix indexing (for coarse) ----------------
def linear_index_from_m(m, base, xp=np):
    """
    Generic mixed-radix index from m ∈ {0,...,base-1}^d to [0, base^d-1].
    """
    d = m.shape[-1]
    w = (base ** xp.arange(d - 1, -1, -1)).astype(np.int64)
    idx = (m * w).sum(axis=-1)
    M = int(base ** d)
    return idx, w, M

def map_to_coarse(m, K_ref, G, xp=np):
    """
    Map fine ranks m ∈ {0,...,K_ref}^d to coarse bins c ∈ {0,...,G-1}^d
    via equal-width partition of the (K_ref+1) rank positions.
    """
    base_fine = K_ref + 1
    if G is None or G >= base_fine:
        return m, base_fine
    c = (m * G) // base_fine
    c = xp.clip(c, 0, G - 1)
    return c, G

def effective_alpha0(alpha0, M_from, M_to, mode):
    """
    If mode='fixed_total', keep total prior mass α0 * M constant when changing resolution:
        α0_to = α0_from * (M_from / M_to)
    If mode='per_cell', leave α0 unchanged per cell.
    """
    if mode == "fixed_total":
        return alpha0 * (M_from / max(1, M_to))
    return alpha0

# ---------------- Rank engines ----------------
def ranks_prefixsum(indices, sel_mask, ords, side_pick):
    """
    Prefix-sum queries: O(d*n_tot) prepass + O(d*|indices|) queries.
    """
    d = len(ords["orders"])
    m_cols = []
    for j in range(d):
        order_j = ords["orders"][j]
        pos_j = ords["pos"][j]
        pick = ords[side_pick][j]  # last_right or first_left
        s_ord = sel_mask[order_j].astype(np.int8)
        S = np.cumsum(s_ord, dtype=np.int64)
        pj = pos_j[indices]
        pr = pick[pj]
        m_cols.append(S[pr])
    return np.stack(m_cols, axis=1)

def ranks_searchsorted(indices, Iref, Z_cols, side="right", jitter=None):
    """
    Binary search engine: for each dim, sort Rj=Z[Iref,j], then query zj via searchsorted.
    """
    d = len(Z_cols)
    m_cols = []
    for j in range(d):
        Zj = Z_cols[j]
        if jitter is not None:
            Zj = Zj + jitter[j]
        Rj = Zj[Iref]
        Rj_sorted = np.sort(Rj)
        zj = Zj[indices]
        m_cols.append(np.searchsorted(Rj_sorted, zj, side=side))
    return np.stack(m_cols, axis=1)

# ---------------- Test statistics ----------------
def stat_from_counts(HX, HY, N_local, Ky_local, alpha0, kind="gtest"):
    """
    Build qhat from HY (Ky_local counts), EX = N*qhat, then:
      chi2 : sum (HX-EX)^2/EX
      gtest: 2*sum HX*log(HX/EX)
    """
    HX = HX.astype(float)
    HY = HY.astype(float)
    M = float(HX.size)

    denom = float(Ky_local) + alpha0 * M
    with np.errstate(divide="ignore", invalid="ignore"):
        qhat = (HY + alpha0) / denom
        EX = N_local * qhat

    if kind == "chi2":
        with np.errstate(divide="ignore", invalid="ignore"):
            T = np.where(EX > 0.0, (HX - EX) ** 2 / EX, 0.0).sum()
        return float(T)

    if kind == "gtest":
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(HX > 0.0, HX / EX, 1.0)
            contrib = np.where(HX > 0.0, HX * np.log(ratio), 0.0)
            T = 2.0 * contrib.sum()
        return float(T)

    raise ValueError("kind must be 'chi2' or 'gtest'")

def gtest_sparse_from_indices(idxX, idxY, N, Ky, alpha0, M):
    """
    Sparse G-test:
      T = 2 * sum_{cells with HX>0} HX * log(HX / EX),
      EX = N * qhat,  qhat = (HY + alpha0) / (Ky + alpha0*M).
    Only X-occurring cells are materialized; HY aligned safely via intersect1d.
    """
    # counts for X (contributors)
    uX, cX = np.unique(idxX, return_counts=True)

    # counts for Y aligned to uX
    if idxY is None or idxY.size == 0:
        HYu = np.zeros_like(uX, dtype=np.int64)
    else:
        uY, cY = np.unique(idxY, return_counts=True)
        HYu = np.zeros(uX.shape, dtype=np.int64)
        # robust alignment (no out-of-bounds; both arrays sorted/unique)
        _, ixX, ixY = np.intersect1d(uX, uY, assume_unique=True, return_indices=True)
        HYu[ixX] = cY[ixY]

    denom = float(Ky) + float(alpha0) * float(M)
    qhat = (HYu + alpha0) / denom
    EXu = N * qhat

    with np.errstate(divide="ignore", invalid="ignore"):
        T = 2.0 * np.sum(cX * np.log(cX / EXu))
    return float(T)

# ---------------- CI for early stop ----------------
def _ndtri(p):
    # Rapid normal quantile (Hastings approximation)
    if p <= 0.0: return -1e300
    if p >= 1.0: return 1e300
    a = [-39.6968302866538, 220.946098424521, -275.928510446969, 138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887, 66.8013118877197, -13.2806815528857]
    c = [-0.00778489400243029, -0.322396458041136, -2.40075827716184, -2.54973253934373, 4.37466414146497, 2.93816398269878]
    d = [0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742]
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

def wilson_ci(k, n, delta):
    if n <= 0: return 0.0, 1.0
    z = _ndtri(1 - delta / 2.0)
    phat = k / n
    denom = 1 + (z*z) / n
    center = (phat + (z*z) / (2*n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, phat * (1 - phat) / n + (z*z) / (4 * n * n)))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

def hoeffding_ci(k, n, delta):
    if n <= 0: return 0.0, 1.0
    phat = k / n
    r = math.sqrt(max(0.0, math.log(2.0 / max(1e-16, delta))) / (2.0 * n))
    return max(0.0, phat - r), min(1.0, phat + r)

# ---------------- Build T_obs (with optional coarsen; no perms) ----------------
def compute_T_obs(
    X, Y, K_ref, tie, alpha0,
    rng, jitter_scale, engine, kref_switch, stat_kind, use_fullY_for_HY=True,
    print_grid_diag=False,
    G=None, alpha0_mode="fixed_total", use_sparse_gtest=False
):
    """
    Build observed Iref ⊂ Y, compute T_obs with HX from X and HY from:
      - full Y-pool (default), or only Iref (if use_fullY_for_HY=False).
    Uses fine ranks; optionally maps to coarse grid of size G^d before counting.
    """
    N, d = X.shape
    K = Y.shape[0]
    if K_ref is None:
        K_ref = K
    if not (1 <= K_ref <= K):
        raise ValueError("K_ref must be in [1, K]")

    Z = np.vstack([X, Y])
    Z_cols = [Z[:, j] for j in range(d)]
    sel = np.zeros(N + K, dtype=np.uint8)

    # tie handling
    jitter = None
    ss_side = tie
    if tie == "jitter":
        jitter = rng.normal(scale=jitter_scale, size=(d, N + K))
        ss_side = "right"

    # prefixsum env if needed
    ords = None; side_pick = None
    if engine in ("auto", "prefixsum") or (engine == "searchsorted" and K_ref > kref_switch):
        ords = precompute_orders(Z, tie=("right" if tie == "jitter" else tie), rng=rng, jitter_scale=jitter_scale)
        side_pick = "last_right" if ords["side"] == "right" else "first_left"

    # engine choice
    if engine == "auto":
        use_searchsorted = (K_ref <= kref_switch)
    elif engine == "searchsorted":
        use_searchsorted = True
    elif engine == "prefixsum":
        use_searchsorted = False
    else:
        raise ValueError("--engine must be auto|searchsorted|prefixsum")

    # Observed reference ⊂ Y
    Iy_obs_inY = rng.choice(K, size=K_ref, replace=False)
    Iy_obs = np.int64(N) + Iy_obs_inY
    idx_X_obs = np.arange(N, dtype=np.int64)

    if use_searchsorted:
        mX_obs = ranks_searchsorted(idx_X_obs, Iy_obs, Z_cols, side=ss_side, jitter=jitter)
        if use_fullY_for_HY:
            Iy_pool_obs = np.arange(N, N + K, dtype=np.int64)
            mY_obs = ranks_searchsorted(Iy_pool_obs, Iy_obs, Z_cols, side=ss_side, jitter=jitter)
        else:
            mY_obs = ranks_searchsorted(Iy_obs, Iy_obs, Z_cols, side=ss_side, jitter=jitter)
    else:
        sel[:] = 0; sel[Iy_obs] = 1
        mX_obs = ranks_prefixsum(idx_X_obs, sel, ords, side_pick)
        if use_fullY_for_HY:
            Iy_pool_obs = np.arange(N, N + K, dtype=np.int64)
            mY_obs = ranks_prefixsum(Iy_pool_obs, sel, ords, side_pick)
        else:
            mY_obs = ranks_prefixsum(Iy_obs, sel, ords, side_pick)

    # Fine sizes
    _, _, M_fine = fine_linear_index(mX_obs, K_ref)

    # Optional coarsening → linear indices at active base
    base_fine = K_ref + 1
    if (G is None) or (G >= base_fine):
        base_used = base_fine
        idx_lin_X, _, M_eff = linear_index_from_m(mX_obs, base_used, np)
        idx_lin_Y, _, _ = linear_index_from_m(mY_obs, base_used, np)
    else:
        mXc, base_used = map_to_coarse(mX_obs, K_ref, G, np)
        mYc, _         = map_to_coarse(mY_obs, K_ref, G, np)
        idx_lin_X, _, M_eff = linear_index_from_m(mXc, base_used, np)
        idx_lin_Y, _, _     = linear_index_from_m(mYc, base_used, np)

    # α0 effective
    alpha0_eff = effective_alpha0(alpha0, M_fine, M_eff, alpha0_mode)

    # Build T_obs (sparse or dense)
    if stat_kind == "gtest" and use_sparse_gtest:
        Ky_obs = len(idx_lin_Y)
        T_obs = gtest_sparse_from_indices(idx_lin_X, idx_lin_Y, N, Ky_obs, alpha0_eff, M_eff)
    else:
        HX_obs = np.bincount(idx_lin_X, minlength=M_eff)
        HY_obs = np.bincount(idx_lin_Y, minlength=M_eff)
        Ky_obs = int(HY_obs.sum())
        if print_grid_diag:
            mean_X = N / M_eff
            mean_Y = Ky_obs / M_eff
            print(f"[grid] M_eff={M_eff}, mean_X={mean_X:.2f}, mean_Y={mean_Y:.2f}, "
                  f"zeros_X={np.mean(HX_obs==0):.2f}, zeros_Y={np.mean(HY_obs==0):.2f}")
        T_obs = stat_from_counts(HX_obs, HY_obs, N_local=N, Ky_local=Ky_obs,
                                 alpha0=alpha0_eff, kind=stat_kind)

    # Return env with coarsening info
    return T_obs, (K_ref, M_eff, use_searchsorted, ords, side_pick, jitter, ss_side, Z, Z_cols,
                   base_used, alpha0_eff)

# ---------------- One permutation run (conditional) ----------------
def perm_test_once(
    X, Y, B, alpha, K_ref, tie, alpha0,
    rng, jitter_scale, engine, kref_switch, decision,
    stop_ci, delta, min_b_check, chunk, antithetic,
    stat_kind="gtest", use_fullY_for_HY=True, print_grid_diag=False,
    G=None, alpha0_mode="fixed_total", use_sparse_gtest=False, profile=False
):
    """
    Conditional permutation test with early stopping.
    Optional coarsening (same as used in T_obs).
    Returns (reject:int, elapsed_ms:float, perms_used:int).
    """
    t0_overall = time.perf_counter()
    # Observed statistic and cached env (timed)
    t0_obs = time.perf_counter()
    T_obs, env = compute_T_obs(
        X, Y, K_ref, tie, alpha0, rng,
        jitter_scale, engine, kref_switch, stat_kind, use_fullY_for_HY, print_grid_diag,
        G=G, alpha0_mode=alpha0_mode, use_sparse_gtest=use_sparse_gtest
    )
    t_obs = time.perf_counter() - t0_obs

    (K_ref, M_eff, use_searchsorted, ords, side_pick, jitter, ss_side, Z, Z_cols,
     base_used, alpha0_eff) = env

    N = X.shape[0]
    K = Y.shape[0]
    n_tot = N + K
    sel = np.zeros(n_tot, dtype=np.uint8)

    ge = gt = eq = 0
    perms_used = 0
    allow_early = (decision in ("pvalue", "midp", "randomized"))
    C = max(1, int(chunk))
    b = 0

    # profiling accumulators (per permutations)
    t_rank = 0.0
    t_hist = 0.0

    # Helper: indexer that mirrors T_obs resolution
    def indexer(m):
        if base_used == (K_ref + 1):
            idx, _, _ = linear_index_from_m(m, base_used, np)
        else:
            m_c = (m * base_used) // (K_ref + 1)
            m_c = np.clip(m_c, 0, base_used - 1)
            idx, _, _ = linear_index_from_m(m_c, base_used, np)
        return idx

    while b < B:
        n_this = min(C, B - b)

        if antithetic:
            pairs = n_this // 2
            remainder = n_this % 2
            for _ in range(pairs):
                perm = rng.permutation(n_tot)
                permR = perm[::-1]
                ref_pos = rng.choice(K, size=K_ref, replace=False)
                ref_posR = K - 1 - ref_pos

                # π
                t1 = time.perf_counter()
                Iy_pool = perm[:K]; Ix_pool = perm[K:]; Iref = Iy_pool[ref_pos]
                if use_searchsorted:
                    mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                    mY = ranks_searchsorted(Iy_pool if use_fullY_for_HY else Iref, Iref, Z_cols, side=ss_side, jitter=jitter)
                else:
                    sel[:] = 0; sel[Iref] = 1
                    mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
                    mY = ranks_prefixsum(Iy_pool if use_fullY_for_HY else Iref, sel, ords, side_pick)
                t_rank += time.perf_counter() - t1

                t2 = time.perf_counter()
                idxX = indexer(mX)
                idxY = indexer(mY)

                if stat_kind == "gtest" and use_sparse_gtest:
                    Ky = len(idxY)
                    Tb = gtest_sparse_from_indices(idxX, idxY, N, Ky, alpha0_eff, M_eff)
                else:
                    HX = np.bincount(idxX, minlength=M_eff)
                    HY = np.bincount(idxY, minlength=M_eff)
                    Ky = int(HY.sum())
                    Tb = stat_from_counts(HX, HY, N, Ky, alpha0_eff, stat_kind)
                t_hist += time.perf_counter() - t2

                if Tb > T_obs: gt += 1
                elif Tb == T_obs: eq += 1
                ge = gt + eq; b += 1; perms_used = b

                if allow_early and b >= min_b_check:
                    lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson" else hoeffding_ci(ge, b, delta))
                    if hi <= alpha or lo > alpha:
                        break

                if b >= B:
                    break

                # π^R
                t1 = time.perf_counter()
                Iy_pool = permR[:K]; Ix_pool = permR[K:]; Iref = Iy_pool[ref_posR]
                if use_searchsorted:
                    mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                    mY = ranks_searchsorted(Iy_pool if use_fullY_for_HY else Iref, Iref, Z_cols, side=ss_side, jitter=jitter)
                else:
                    sel[:] = 0; sel[Iref] = 1
                    mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
                    mY = ranks_prefixsum(Iy_pool if use_fullY_for_HY else Iref, sel, ords, side_pick)
                t_rank += time.perf_counter() - t1

                t2 = time.perf_counter()
                idxX = indexer(mX)
                idxY = indexer(mY)
                if stat_kind == "gtest" and use_sparse_gtest:
                    Ky = len(idxY)
                    Tb = gtest_sparse_from_indices(idxX, idxY, N, Ky, alpha0_eff, M_eff)
                else:
                    HX = np.bincount(idxX, minlength=M_eff)
                    HY = np.bincount(idxY, minlength=M_eff)
                    Ky = int(HY.sum())
                    Tb = stat_from_counts(HX, HY, N, Ky, alpha0_eff, stat_kind)
                t_hist += time.perf_counter() - t2

                if Tb > T_obs: gt += 1
                elif Tb == T_obs: eq += 1
                ge = gt + eq; b += 1; perms_used = b

                if allow_early and b >= min_b_check:
                    lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson" else hoeffding_ci(ge, b, delta))
                    if hi <= alpha or lo > alpha:
                        break

            if allow_early and b >= min_b_check:
                lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson" else hoeffding_ci(ge, b, delta))
                if hi <= alpha or lo > alpha:
                    break

            if remainder and b < B:
                t1 = time.perf_counter()
                perm = rng.permutation(n_tot)
                Iy_pool = perm[:K]; Ix_pool = perm[K:]
                Iref = rng.choice(Iy_pool, size=K_ref, replace=False)
                if use_searchsorted:
                    mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                    mY = ranks_searchsorted(Iy_pool if use_fullY_for_HY else Iref, Iref, Z_cols, side=ss_side, jitter=jitter)
                else:
                    sel[:] = 0; sel[Iref] = 1
                    mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
                    mY = ranks_prefixsum(Iy_pool if use_fullY_for_HY else Iref, sel, ords, side_pick)
                t_rank += time.perf_counter() - t1

                t2 = time.perf_counter()
                idxX = indexer(mX)
                idxY = indexer(mY)
                if stat_kind == "gtest" and use_sparse_gtest:
                    Ky = len(idxY)
                    Tb = gtest_sparse_from_indices(idxX, idxY, N, Ky, alpha0_eff, M_eff)
                else:
                    HX = np.bincount(idxX, minlength=M_eff)
                    HY = np.bincount(idxY, minlength=M_eff)
                    Ky = int(HY.sum())
                    Tb = stat_from_counts(HX, HY, N, Ky, alpha0_eff, stat_kind)
                t_hist += time.perf_counter() - t2

                if Tb > T_obs: gt += 1
                elif Tb == T_obs: eq += 1
                ge = gt + eq; b += 1; perms_used = b

                if allow_early and b >= min_b_check:
                    lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson" else hoeffding_ci(ge, b, delta))
                    if hi <= alpha or lo > alpha:
                        break

        else:
            for _ in range(n_this):
                t1 = time.perf_counter()
                perm = rng.permutation(n_tot)
                Iy_pool = perm[:K]; Ix_pool = perm[K:]
                Iref = rng.choice(Iy_pool, size=K_ref, replace=False)

                if use_searchsorted:
                    mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                    mY = ranks_searchsorted(Iy_pool if use_fullY_for_HY else Iref, Iref, Z_cols, side=ss_side, jitter=jitter)
                else:
                    sel[:] = 0; sel[Iref] = 1
                    mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
                    mY = ranks_prefixsum(Iy_pool if use_fullY_for_HY else Iref, sel, ords, side_pick)
                t_rank += time.perf_counter() - t1

                t2 = time.perf_counter()
                idxX = indexer(mX)
                idxY = indexer(mY)
                if stat_kind == "gtest" and use_sparse_gtest:
                    Ky = len(idxY)
                    Tb = gtest_sparse_from_indices(idxX, idxY, N, Ky, alpha0_eff, M_eff)
                else:
                    HX = np.bincount(idxX, minlength=M_eff)
                    HY = np.bincount(idxY, minlength=M_eff)
                    Ky = int(HY.sum())
                    Tb = stat_from_counts(HX, HY, N, Ky, alpha0_eff, stat_kind)
                t_hist += time.perf_counter() - t2

                if Tb > T_obs: gt += 1
                elif Tb == T_obs: eq += 1
                ge = gt + eq; b += 1; perms_used = b

                if allow_early and b >= min_b_check:
                    lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson" else hoeffding_ci(ge, b, delta))
                    if hi <= alpha or lo > alpha:
                        break

        if allow_early and b >= min_b_check:
            lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson" else hoeffding_ci(ge, b, delta))
            if hi <= alpha or lo > alpha:
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
            reject = (rng.random() <= omega)
    else:
        reject = (p_perm <= alpha)

    elapsed_ms = 1e3 * (time.perf_counter() - t0_overall)

    if profile:
        print(f"[profile] T_obs={t_obs:.3f}s  ranks={t_rank:.3f}s  hist+stat={t_hist:.3f}s  "
              f"perms_used={perms_used}")

    return int(reject), float(elapsed_ms), int(perms_used)

# ---------------- H0/H1 generators ----------------
def gen_pair_H0(mode, d, N, K, rng, rho=0.5, nu=5):
    if mode == "null-gauss":
        X = draw_gauss(N, np.zeros(d), np.eye(d), rng)
        Y = draw_gauss(K, np.zeros(d), np.eye(d), rng)
    elif mode == "null-copula":
        S = corr_matrix(d, rho)
        X = draw_gauss(N, np.zeros(d), S, rng)
        Y = draw_gauss(K, np.zeros(d), S, rng)
    elif mode == "null-studentt":
        X = draw_studentt(N, d, nu, rng)
        Y = draw_studentt(K, d, nu, rng)
    else:
        raise ValueError("--null_mode must be null-gauss|null-copula|null-studentt")
    return X, Y

def gen_pair_H1(scenario, d, N, K, rng, args):
    if scenario == "shift":
        delta = args.delta
        if args.dir == "random":
            u = rng.normal(size=d); u /= (np.linalg.norm(u) + 1e-12)
        else:
            u = np.zeros(d); u[0] = 1.0
        X = draw_gauss(N, delta * u, np.eye(d), rng)
        Y = draw_gauss(K, np.zeros(d), np.eye(d), rng)
        return X, Y

    if scenario == "scale":
        s = args.scale
        X = draw_gauss(N, np.zeros(d), (s**2) * np.eye(d), rng)
        Y = draw_gauss(K, np.zeros(d), np.eye(d), rng)
        return X, Y

    if scenario == "corr":
        Sx = corr_matrix(d, args.rho_x)
        Sy = corr_matrix(d, args.rho_y)
        X = draw_gauss(N, np.zeros(d), Sx, rng)
        Y = draw_gauss(K, np.zeros(d), Sy, rng)
        return X, Y

    if scenario == "tdf":
        X = draw_studentt(N, d, args.nu_x, rng)
        Y = draw_studentt(K, d, args.nu_y, rng)
        return X, Y

    if scenario == "mixture":
        eps = args.eps; delta = args.delta
        n1 = rng.binomial(N, eps); n0 = N - n1
        X0 = draw_gauss(n0, np.zeros(d), np.eye(d), rng)
        mu1 = np.zeros(d); mu1[0] = delta
        X1 = draw_gauss(n1, mu1, np.eye(d), rng)
        X = np.vstack([X0, X1]); rng.shuffle(X, axis=0)
        Y = draw_gauss(K, np.zeros(d), np.eye(d), rng)
        return X, Y

    raise ValueError("Unknown --scenario")

# ---------------- Utilities ----------------
def se_binom(p, R):
    return float(np.sqrt(max(p * (1 - p), 1e-12) / max(1, R)))

def run_one_rep_type1(seed_seq, setup):
    (null_mode, d, N, K, B, alpha, K_ref, tie, alpha0,
     engine, kref_switch, decision,
     stop_ci, delta_ci, min_b_check, chunk, antithetic,
     stat_kind, use_fullY_for_HY, rho, nu, print_grid_diag,
     G, alpha0_mode, use_sparse_gtest, profile) = setup

    rng = make_rng(seed_seq)
    X, Y = gen_pair_H0(null_mode, d, N, K, rng, rho=rho, nu=nu)

    r, ms, used = perm_test_once(
        X, Y, B=B, alpha=alpha, K_ref=K_ref, tie=tie, alpha0=alpha0,
        rng=rng, jitter_scale=1e-12, engine=engine, kref_switch=kref_switch,
        decision=decision, stop_ci=stop_ci, delta=delta_ci, min_b_check=min_b_check,
        chunk=chunk, antithetic=antithetic, stat_kind=stat_kind,
        use_fullY_for_HY=use_fullY_for_HY, print_grid_diag=print_grid_diag,
        G=G, alpha0_mode=alpha0_mode, use_sparse_gtest=use_sparse_gtest, profile=profile
    )
    return r, ms, used

def run_one_rep_power(seed_seq, setup):
    (scenario, d, N, K, B, alpha, K_ref, tie, alpha0,
     engine, kref_switch, decision,
     stop_ci, delta_ci, min_b_check, chunk, antithetic,
     stat_kind, use_fullY_for_HY,
     # scenario params
     delta, direction, scale, rho_x, rho_y, nu_x, nu_y, eps,
     print_grid_diag, G, alpha0_mode, use_sparse_gtest, profile) = setup

    rng = make_rng(seed_seq)
    class A: pass
    args = A()
    args.delta = delta; args.dir = direction; args.scale = scale
    args.rho_x = rho_x; args.rho_y = rho_y; args.nu_x = nu_x; args.nu_y = nu_y; args.eps = eps
    X, Y = gen_pair_H1(scenario, d, N, K, rng, args)

    r, ms, used = perm_test_once(
        X, Y, B=B, alpha=alpha, K_ref=K_ref, tie=tie, alpha0=alpha0,
        rng=rng, jitter_scale=1e-12, engine=engine, kref_switch=kref_switch,
        decision=decision, stop_ci=stop_ci, delta=delta_ci, min_b_check=min_b_check,
        chunk=chunk, antithetic=antithetic, stat_kind=stat_kind,
        use_fullY_for_HY=use_fullY_for_HY, print_grid_diag=print_grid_diag,
        G=G, alpha0_mode=alpha0_mode, use_sparse_gtest=use_sparse_gtest, profile=profile
    )
    return r, ms, used

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Permutation non-symmetric rank χ² / G-test (optional coarsening + sparse path)")
    # Mode
    ap.add_argument("--mode", choices=["type1", "power"], default="type1")

    # Core sizes
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--K_ref", type=int, default=None)

    # Test config
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alpha0", type=float, default=0.1)
    ap.add_argument("--alpha0_mode", choices=["per_cell","fixed_total"], default="fixed_total",
                    help="Pseudocount handling under coarsening.")
    ap.add_argument("--tie", choices=["right", "left", "jitter"], default="jitter")
    ap.add_argument("--engine", choices=["auto", "searchsorted", "prefixsum"], default="auto")
    ap.add_argument("--kref_switch", type=int, default=64)
    ap.add_argument("--stat", choices=["gtest", "chi2"], default="gtest")

    # Coarsening
    ap.add_argument("--G", type=int, default=None,
                    help="Coarse bins per dimension (<= K_ref+1). Omit for fine grid.")

    # Sparse G-test
    ap.add_argument("--sparse_gtest", action="store_true",
                    help="Use sparse path for G-test (ignores zero-HX cells).")

    # Permutations
    ap.add_argument("--B", type=int, default=1000, help="Max permutations per repetition")
    ap.add_argument("--decision", choices=["pvalue", "midp", "randomized"], default="randomized")
    ap.add_argument("--stop_ci", choices=["wilson", "hoeffding"], default="wilson")
    ap.add_argument("--delta_ci", type=float, default=1e-2)
    ap.add_argument("--min_b_check", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--antithetic", action="store_true")
    ap.add_argument("--fullY", action="store_true",
                    help="Use HY from full Y-pool vs Iref (recommended). If omitted, uses Iref only.")
    ap.add_argument("--grid_diag", action="store_true", help="Print grid sparsity diagnostics (observed split)")

    # Profiling
    ap.add_argument("--profile", action="store_true", help="Print coarse timing breakdown per rep")

    # H0 (type1) null modes
    ap.add_argument("--null_mode", choices=["null-gauss", "null-copula", "null-studentt"], default="null-gauss")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--nu", type=int, default=5)

    # H1 scenarios
    ap.add_argument("--scenario", choices=["shift", "scale", "corr", "tdf", "mixture"], default="scale")
    ap.add_argument("--delta", type=float, default=0.30, help="Effect for shift/mixture")
    ap.add_argument("--dir", choices=["e1", "random"], default="e1")
    ap.add_argument("--scale", type=float, default=1.50, help="σ_x in 'scale'")
    ap.add_argument("--rho_x", type=float, default=0.7, help="ρ_x in 'corr'")
    ap.add_argument("--rho_y", type=float, default=0.0, help="ρ_y in 'corr'")
    ap.add_argument("--nu_x", type=int, default=5, help="ν_x in 'tdf'")
    ap.add_argument("--nu_y", type=int, default=10, help="ν_y in 'tdf'")
    ap.add_argument("--eps", type=float, default=0.10, help="ε in 'mixture'")

    # Reps / parallel
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--single_thread_blas", action="store_true")

    args = ap.parse_args()

    if args.single_thread_blas:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    jobs = args.jobs or os.cpu_count() or 1
    mode_str = "TYPE-I (H0)" if args.mode == "type1" else "POWER (H1)"
    base_str = "(fine grid)" if (args.G is None or (args.K_ref is not None and args.G >= args.K_ref+1)) else f"(coarse G={args.G})"
    sparse_str = "sparse" if (args.stat == "gtest" and args.sparse_gtest) else "dense"
    print(f"[{mode_str} | permutations only] {base_str} | {sparse_str} {args.stat}")
    print(f" d={args.d}, N={args.N}, K={args.K}, K_ref={args.K_ref or args.K}, "
          f"B={args.B}, R={args.R}, alpha={args.alpha}, tie={args.tie}, "
          f"alpha0={args.alpha0} ({args.alpha0_mode})")
    print(f" engine={args.engine}, kref_switch={args.kref_switch}, jobs={jobs}, seed={args.seed}")
    print(f" decision={args.decision}, early-stop={args.stop_ci}, delta_ci={args.delta_ci}, "
          f"min_b_check={args.min_b_check}, chunk={args.chunk}, antithetic={args.antithetic}, fullY={args.fullY}, profile={args.profile}")

    if args.mode == "type1":
        print(f" null_mode={args.null_mode} (rho={args.rho}, nu={args.nu})")
        setup = (args.null_mode, args.d, args.N, args.K, args.B, args.alpha, args.K_ref, args.tie, args.alpha0,
                 args.engine, args.kref_switch, args.decision,
                 args.stop_ci, args.delta_ci, args.min_b_check, args.chunk, args.antithetic,
                 args.stat, bool(args.fullY), args.rho, args.nu, bool(args.grid_diag),
                 args.G, args.alpha0_mode, bool(args.sparse_gtest), bool(args.profile))
        fn = run_one_rep_type1
    else:
        print(f" scenario={args.scenario} | delta={args.delta}, dir={args.dir}, "
              f"scale={args.scale}, rho_x={args.rho_x}, rho_y={args.rho_y}, nu_x={args.nu_x}, nu_y={args.nu_y}, eps={args.eps}")
        setup = (args.scenario, args.d, args.N, args.K, args.B, args.alpha, args.K_ref, args.tie, args.alpha0,
                 args.engine, args.kref_switch, args.decision,
                 args.stop_ci, args.delta_ci, args.min_b_check, args.chunk, args.antithetic,
                 args.stat, bool(args.fullY),
                 args.delta, args.dir, args.scale, args.rho_x, args.rho_y, args.nu_x, args.nu_y, args.eps,
                 bool(args.grid_diag), args.G, args.alpha0_mode, bool(args.sparse_gtest), bool(args.profile))
        fn = run_one_rep_power

    ss = np.random.SeedSequence(args.seed)
    rep_seeds = ss.spawn(args.R)

    rejects, times, used_list = [], [], []
    if jobs == 1:
        with tqdm(total=args.R, desc=("H0 reps" if args.mode == "type1" else "H1 reps"), leave=True) as pbar:
            for s in rep_seeds:
                r, ms, used = fn(s, setup)
                rejects.append(r); times.append(ms); used_list.append(used)
                pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            it = ex.map(fn, rep_seeds, (setup,) * args.R, chunksize=max(1, args.R // (jobs * 4)))
            with tqdm(total=args.R, desc=("H0 reps" if args.mode == "type1" else "H1 reps"), leave=True) as pbar:
                for r, ms, used in it:
                    rejects.append(r); times.append(ms); used_list.append(used)
                    pbar.update(1)

    R = args.R
    p_hat = float(np.mean(rejects)) if R else 0.0
    se_hat = se_binom(p_hat, R)
    avg_ms = float(np.mean(times)) if times else 0.0
    avg_used = float(np.mean(used_list)) if used_list else 0.0

    print("\n[Results]")
    print("---------")
    label = "α̂ (Type-I)" if args.mode == "type1" else "π̂ (Power)"
    print(f"{label}: {p_hat:.4f}   SE ≈ {se_hat:.4f}")
    print(f"Avg time per repetition: {avg_ms:8.2f} ms")
    print(f"Avg permutations used:   {avg_used:8.1f} / B={args.B}")
    print("Notes:")
    print(" • Choose G so expected counts per cell ≳ 5–10 for stability.")
    print(" • '--alpha0_mode fixed_total' avoids over/under-smoothing when changing resolution.")
    print(" • '--engine prefixsum --kref_switch 0' can speed rank computations.")
    print(" • Try '--sparse_gtest' for large grids; combine with '--fullY' for power.")

if __name__ == "__main__":
    main()
