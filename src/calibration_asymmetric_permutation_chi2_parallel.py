#!/usr/bin/env python3
# -------------------------------------------------------------------
# Parallel fast role-swapping (non-symmetric) χ² two-sample test
#  • R repetitions parallelized with ProcessPoolExecutor
#  • Clean RNG splitting via numpy.random.SeedSequence.spawn(R)
#  • Observed: ref = subset of Y (size K_ref); targets = X (size N)
#  • Permutation pools:
#       (default) subset sampling Iy_pool ~ Unif(K-subsets of {0..n_tot-1})
#       (--antithetic) paired pools from a permutation π and its reverse π^R:
#           Iy_pool1 = π[:K], Ix_pool1 = π[K:]
#           Iy_pool2 = π^R[:K], Ix_pool2 = π^R[K:]
#  • Rank engines:
#       - searchsorted : O(d · (K_ref log K_ref + m log K_ref)) per perm
#       - prefixsum    : one cumsum per dim per perm (good when K_ref large)
#       - auto         : choose by K_ref threshold (kref_switch)
#  • Coarsening L1×…×Ld (auto or explicit), Jeffreys smoothing α0
#  • Early stopping on permutation tail prob with CI:
#       --stop_ci {hoeffding,wilson} (Wilson usually tighter), --delta, --min_b_check
#  • NEW:
#       --chunk <int>     : batch perms to reduce Python overhead
#       --antithetic      : paired (π, π^R) perms for variance reduction
#  • Reports type-I under H0, timing, avg perms used
# -------------------------------------------------------------------
import argparse, os, time, math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from tqdm.auto import tqdm


# ---------------- RNG ----------------
def make_rng(seed_or_ss):
    return np.random.default_rng(seed_or_ss)


# ---------------- Null generators (H0) ----------------
def corr_matrix(d, rho):
    if d == 1: return np.array([[1.0]])
    S = np.full((d, d), rho, dtype=float); np.fill_diagonal(S, 1.0)
    w, V = np.linalg.eigh(S); w = np.clip(w, 1e-12, None)
    return (V * w) @ V.T

def draw_null(mode, d, n, rng, rho=0.5, nu=5):
    if mode == "null-gauss":
        return rng.multivariate_normal(np.zeros(d), np.eye(d), size=n)
    elif mode == "null-copula":
        S = corr_matrix(d, rho)
        return rng.multivariate_normal(np.zeros(d), S, size=n)
    elif mode == "null-studentt":
        Z = rng.normal(size=(n, d)); U = rng.chisquare(nu, size=n)
        return Z / np.sqrt(U[:, None] / nu)
    else:
        raise ValueError(mode)


# --------- Precompute orders for prefix-sum ranks ---------
def precompute_orders(Z, tie="right", jitter_scale=1e-12, rng=None):
    n_tot, d = Z.shape
    if tie == "jitter":
        if rng is None: rng = np.random.default_rng(0)
        Zw = Z + rng.normal(scale=jitter_scale, size=Z.shape)
        side = "right"
    else:
        Zw = Z
        side = tie

    orders, pos_all, last_right_all, first_left_all = [], [], [], []
    for j in range(d):
        order = np.argsort(Zw[:, j], kind="mergesort")
        vals  = Zw[order, j]
        pos   = np.empty(n_tot, dtype=np.int64)
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


# ---------------- Coarsening helpers ----------------
def coarse_linear_index(m, K_ref, L_vec, w_coarse):
    d = m.shape[1]
    if L_vec is None:
        base = K_ref + 1
        w_fine = (base ** np.arange(d - 1, -1, -1)).astype(np.int64)
        return (m * w_fine).sum(axis=1)
    L = np.asarray(L_vec, dtype=np.int64)
    mc = np.floor(m * (L / float(K_ref + 1))).astype(np.int64)
    mc = np.minimum(mc, L - 1)
    return (mc * w_coarse).sum(axis=1)

def choose_L_auto(d, K_ref, N_targets, target_exp=5):
    base = K_ref + 1
    M_goal = max(1, int(N_targets // max(1, target_exp)))
    if M_goal >= base ** d:
        return None  # no coarsening
    L = np.ones(d, dtype=np.int64)
    while np.prod(L) < M_goal:
        j = int(np.argmin(L))
        if L[j] < base: L[j] += 1
        else:
            cand = np.where(L < base)[0]
            if cand.size == 0: break
            L[int(cand[0])] += 1
    return L


# ---------------- Pearson (asymmetric) ----------------
def pearson_asym_from_counts(HX, HY, N_local, Kref_local, alpha0_local=0.0):
    HXf = HX.astype(float); HYf = HY.astype(float)
    M = float(HX.size)
    denom = float(Kref_local) + alpha0_local * M
    with np.errstate(divide="ignore", invalid="ignore"):
        qhat = (HYf + alpha0_local) / denom
        EX   = N_local * qhat
        T    = np.where(EX > 0, (HXf - EX) ** 2 / EX, 0.0).sum()
    return float(T)


# ---------------- Rank engines ----------------
def ranks_prefixsum(indices, sel_mask, ords, side_pick):
    d = len(ords["orders"])
    m_cols = []
    for j in range(d):
        order_j = ords["orders"][j]
        pos_j   = ords["pos"][j]
        pick    = ords[side_pick][j]
        s_ord   = sel_mask[order_j].astype(np.int8)
        S       = np.cumsum(s_ord, dtype=np.int64)
        pj      = pos_j[indices]
        pr      = pick[pj]
        m_cols.append(S[pr])
    return np.stack(m_cols, axis=1)

def ranks_searchsorted(indices, Iref, Z_cols, side="right", jitter=None):
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


# ---------------- Early-stopping CI helpers ----------------
def hoeffding_ci(k, n, delta):
    if n <= 0: return 0.0, 1.0
    phat = k / n
    r = math.sqrt(max(0.0, math.log(2.0 / max(1e-16, delta))) / (2.0 * n))
    return max(0.0, phat - r), min(1.0, phat + r)

def _ndtri(p):
    if p <= 0.0: return -1e300
    if p >= 1.0: return 1e300
    a = [ -39.6968302866538, 220.946098424521, -275.928510446969,
          138.357751867269, -30.6647980661472, 2.50662827745924 ]
    b = [ -54.4760987982241, 161.585836858041, -155.698979859887,
           66.8013118877197, -13.2806815528857 ]
    c = [ -0.00778489400243029, -0.322396458041136, -2.40075827716184,
          -2.54973253934373, 4.37466414146497, 2.93816398269878 ]
    d = [ 0.00778469570904146, 0.32246712907004, 2.445134137143,
          3.75440866190742 ]
    plow  = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2*math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2*math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                 ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def wilson_ci(k, n, delta):
    if n <= 0: return 0.0, 1.0
    z = _ndtri(1 - delta/2.0)
    phat = k / n
    denom = 1 + (z*z)/n
    center = (phat + (z*z)/(2*n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, phat*(1-phat)/n + (z*z)/(4*n*n)))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


# ----------- One repetition (worker) -----------
def roleswap_fast_once(
    X, Y, B, alpha, K_ref, tie, alpha0, L_coarse, target_exp,
    rng, jitter_scale, engine, kref_switch, decision,
    stop_ci, delta, min_b_check, chunk, antithetic
):
    """
    One repetition with:
      • Optional antithetic paired permutations (π, π^R)
      • Early stopping via Wilson/Hoeffding CI
      • Chunked inner loop
    Returns (reject:int, elapsed_ms:float, perms_used:int).
    """
    t0 = time.perf_counter()
    N, d = X.shape
    K    = Y.shape[0]
    n_tot = N + K
    if K_ref is None: K_ref = K
    if not (1 <= K_ref <= K):
        raise ValueError("K_ref must be in [1, K]")

    # Pooled data
    Z = np.vstack([X, Y])
    Z_cols = [Z[:, j] for j in range(d)]
    all_idx = np.arange(n_tot, dtype=np.int64)
    mask = np.zeros(n_tot, dtype=bool)
    sel  = np.zeros(n_tot, dtype=np.uint8)

    # Jitter arrays for searchsorted if tie='jitter'
    jitter = None
    ss_side = tie
    if tie == "jitter":
        jitter = rng.normal(scale=jitter_scale, size=(d, n_tot))
        ss_side = "right"

    # Precompute orders for prefix-sum if needed
    ords = None; side_pick = None
    if engine in ("auto", "prefixsum") or (engine == "searchsorted" and K_ref > kref_switch):
        ords = precompute_orders(Z, tie=("right" if tie=="jitter" else tie), rng=rng, jitter_scale=jitter_scale)
        side_pick = "last_right" if ords["side"] == "right" else "first_left"

    # Coarsening bases/weights
    if L_coarse is None:
        L_coarse = choose_L_auto(d, K_ref, N_targets=N, target_exp=target_exp)
    if L_coarse is not None:
        L = np.asarray(L_coarse, dtype=np.int64)
        w_coarse = np.ones(d, dtype=np.int64)
        for j in range(d - 1):
            w_coarse[j] = int(np.prod(L[j + 1:], dtype=np.int64))
        M_eff = int(np.prod(L))
    else:
        L = None
        w_coarse = None
        M_eff = (K_ref + 1) ** d

    # Decide engine
    if engine == "auto":
        use_searchsorted = (K_ref <= kref_switch)
    elif engine == "searchsorted":
        use_searchsorted = True
    elif engine == "prefixsum":
        use_searchsorted = False
    else:
        raise ValueError("--engine must be auto|searchsorted|prefixsum")

    # --------------- Observed ---------------
    Iy_obs_inY = rng.choice(K, size=K_ref, replace=False)
    Iy_obs     = np.int64(N) + Iy_obs_inY
    idx_X_obs  = np.arange(N, dtype=np.int64)

    if use_searchsorted:
        mX_obs = ranks_searchsorted(idx_X_obs, Iy_obs, Z_cols, side=ss_side, jitter=jitter)
        mY_obs = ranks_searchsorted(Iy_obs,     Iy_obs, Z_cols, side=ss_side, jitter=jitter)
    else:
        sel[:] = 0; sel[Iy_obs] = 1
        mX_obs = ranks_prefixsum(idx_X_obs, sel, ords, side_pick)
        mY_obs = ranks_prefixsum(Iy_obs,     sel, ords, side_pick)

    idx_lin_X = coarse_linear_index(mX_obs, K_ref, L, w_coarse)
    idx_lin_Y = coarse_linear_index(mY_obs, K_ref, L, w_coarse)
    HX_obs = np.bincount(idx_lin_X, minlength=M_eff)
    HY_obs = np.bincount(idx_lin_Y, minlength=M_eff)
    T_obs  = pearson_asym_from_counts(HX_obs, HY_obs, N_local=N, Kref_local=K_ref, alpha0_local=alpha0)

    # --------------- Permutations (chunked, optional antithetic) ---------------
    ge = gt = eq = 0
    perms_used = 0
    allow_early = (decision in ("pvalue", "midp", "randomized"))
    C = max(1, int(chunk))
    b = 0

    while b < B:
        n_this = min(C, B - b)

        # If antithetic: we will process in PAIRS; adjust local budget.
        if antithetic:
            # pairs to do this chunk
            pairs = n_this // 2
            # if odd remainder and total budget allows, we’ll do one extra single
            remainder = n_this % 2
            for _ in range(pairs):
                # One base permutation and its reverse
                perm = rng.permutation(n_tot)
                permR = perm[::-1]

                # Choose same positions inside the pool for Iref, mirrored for anti-correlation
                ref_pos = rng.choice(K, size=K_ref, replace=False)
                ref_posR = K - 1 - ref_pos  # mirror positions

                # --- First (π) ---
                Iy_pool = perm[:K]
                Ix_pool = perm[K:]
                Iref = Iy_pool[ref_pos]

                if use_searchsorted:
                    mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                    mY = ranks_searchsorted(Iref,    Iref, Z_cols, side=ss_side, jitter=jitter)
                else:
                    sel[:] = 0; sel[Iref] = 1
                    mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
                    mY = ranks_prefixsum(Iref,    sel, ords, side_pick)

                idxX = coarse_linear_index(mX, K_ref, L, w_coarse)
                idxY = coarse_linear_index(mY, K_ref, L, w_coarse)
                HX = np.bincount(idxX, minlength=M_eff)
                HY = np.bincount(idxY, minlength=M_eff)
                T_b = pearson_asym_from_counts(HX, HY, N_local=N, Kref_local=K_ref, alpha0_local=alpha0)
                if T_b > T_obs: gt += 1
                elif T_b == T_obs: eq += 1
                ge = gt + eq
                b += 1; perms_used = b

                # --- Second (π^R) ---
                if b >= B: break
                Iy_pool = permR[:K]
                Ix_pool = permR[K:]
                Iref = Iy_pool[ref_posR]

                if use_searchsorted:
                    mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                    mY = ranks_searchsorted(Iref,    Iref, Z_cols, side=ss_side, jitter=jitter)
                else:
                    sel[:] = 0; sel[Iref] = 1
                    mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
                    mY = ranks_prefixsum(Iref,    sel, ords, side_pick)

                idxX = coarse_linear_index(mX, K_ref, L, w_coarse)
                idxY = coarse_linear_index(mY, K_ref, L, w_coarse)
                HX = np.bincount(idxX, minlength=M_eff)
                HY = np.bincount(idxY, minlength=M_eff)
                T_b = pearson_asym_from_counts(HX, HY, N_local=N, Kref_local=K_ref, alpha0_local=alpha0)
                if T_b > T_obs: gt += 1
                elif T_b == T_obs: eq += 1
                ge = gt + eq
                b += 1; perms_used = b

                # early-stop after the pair
                if allow_early and b >= min_b_check:
                    lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson"
                              else hoeffding_ci(ge, b, delta))
                    if hi <= alpha or lo > alpha:
                        break

            if allow_early and b >= min_b_check:
                if (stop_ci == "wilson" and (wilson_ci(ge, b, delta)[1] <= alpha or wilson_ci(ge, b, delta)[0] > alpha)) \
                   or (stop_ci == "hoeffding" and (hoeffding_ci(ge, b, delta)[1] <= alpha or hoeffding_ci(ge, b, delta)[0] > alpha)):
                    break

            # If an odd one remains in this chunk and budget allows, do a single subset sample
            if remainder and b < B:
                Iy_pool = rng.choice(n_tot, size=K, replace=False)
                mask[:] = False; mask[Iy_pool] = True
                Ix_pool = all_idx[~mask]
                Iref = rng.choice(Iy_pool, size=K_ref, replace=False)

                if use_searchsorted:
                    mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                    mY = ranks_searchsorted(Iref,    Iref, Z_cols, side=ss_side, jitter=jitter)
                else:
                    sel[:] = 0; sel[Iref] = 1
                    mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
                    mY = ranks_prefixsum(Iref,    sel, ords, side_pick)

                idxX = coarse_linear_index(mX, K_ref, L, w_coarse)
                idxY = coarse_linear_index(mY, K_ref, L, w_coarse)
                HX = np.bincount(idxX, minlength=M_eff)
                HY = np.bincount(idxY, minlength=M_eff)
                T_b = pearson_asym_from_counts(HX, HY, N_local=N, Kref_local=K_ref, alpha0_local=alpha0)
                if T_b > T_obs: gt += 1
                elif T_b == T_obs: eq += 1
                ge = gt + eq
                b += 1; perms_used = b

                if allow_early and b >= min_b_check:
                    lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson"
                              else hoeffding_ci(ge, b, delta))
                    if hi <= alpha or lo > alpha:
                        break

        else:
            # Standard subset-sample path (no antithetic)
            for _ in range(n_this):
                Iy_pool = rng.choice(n_tot, size=K, replace=False)
                mask[:] = False; mask[Iy_pool] = True
                Ix_pool = all_idx[~mask]
                Iref    = rng.choice(Iy_pool, size=K_ref, replace=False)

                if use_searchsorted:
                    mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
                    mY = ranks_searchsorted(Iref,    Iref, Z_cols, side=ss_side, jitter=jitter)
                else:
                    sel[:] = 0; sel[Iref] = 1
                    mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
                    mY = ranks_prefixsum(Iref,    sel, ords, side_pick)

                idxX = coarse_linear_index(mX, K_ref, L, w_coarse)
                idxY = coarse_linear_index(mY, K_ref, L, w_coarse)
                HX = np.bincount(idxX, minlength=M_eff)
                HY = np.bincount(idxY, minlength=M_eff)
                T_b = pearson_asym_from_counts(HX, HY, N_local=N, Kref_local=K_ref, alpha0_local=alpha0)

                if T_b > T_obs: gt += 1
                elif T_b == T_obs: eq += 1
                ge = gt + eq
                b += 1; perms_used = b

                if allow_early and b >= min_b_check:
                    lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson"
                              else hoeffding_ci(ge, b, delta))
                    if hi <= alpha or lo > alpha:
                        break

        if allow_early and b >= min_b_check:
            lo, hi = (wilson_ci(ge, b, delta) if stop_ci == "wilson"
                      else hoeffding_ci(ge, b, delta))
            if hi <= alpha or lo > alpha:
                break

    # final p-values using actually-used permutations
    b = max(1, perms_used)
    p_perm = (1 + ge) / (b + 1)
    p_mid  = (gt + 0.5 * eq) / b

    # decision
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

    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    return int(reject), float(elapsed_ms), int(perms_used)


# ---------------- Utilities ----------------
def se_binom(a_hat, R):
    return float(np.sqrt(max(a_hat * (1 - a_hat), 1e-12) / R))


# ---------------- Worker wrapper for executor ----------------
def run_one_rep(seed_seq, setup):
    (mode, d, N, K, B, alpha, K_ref, tie, alpha0,
     L_coarse, target_exp, engine, kref_switch,
     decision, rho, nu, stop_ci, delta, min_b_check, chunk, antithetic) = setup

    rng = make_rng(seed_seq)
    X = draw_null(mode, d, N, rng, rho=rho, nu=nu)
    Y = draw_null(mode, d, K, rng, rho=rho, nu=nu)

    r, ms, used = roleswap_fast_once(
        X, Y, B=B, alpha=alpha, K_ref=K_ref,
        tie=tie, alpha0=alpha0, L_coarse=L_coarse, target_exp=target_exp,
        rng=rng, jitter_scale=1e-12, engine=engine, kref_switch=kref_switch,
        decision=decision, stop_ci=stop_ci, delta=delta, min_b_check=min_b_check,
        chunk=chunk, antithetic=antithetic
    )
    return r, ms, used


# ---------------- CLI driver ----------------
def main():
    ap = argparse.ArgumentParser(description="Parallel fast role-swapping χ² (non-symmetric) | early-stop + chunking + antithetic")
    ap.add_argument("--mode", choices=["null-gauss", "null-copula", "null-studentt"], default="null-gauss")
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--K_ref", type=int, default=None, help="Reference size (subset of Y), default K")
    ap.add_argument("--B", type=int, default=1000, help="Max permutations per repetition")
    ap.add_argument("--R", type=int, default=200, help="Outer repetitions under H0")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alpha0", type=float, default=0.5, help="Jeffreys smoothing per cell")
    ap.add_argument("--tie", choices=["right","left","jitter"], default="jitter")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--nu", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)

    # Rank engine + coarsening
    ap.add_argument("--engine", choices=["auto","searchsorted","prefixsum"], default="auto")
    ap.add_argument("--kref_switch", type=int, default=64)
    ap.add_argument("--L", type=str, default=None, help="Coarsen to L1,...,Ld (e.g. '3,3,3,3'); omit for auto")
    ap.add_argument("--target_exp", type=int, default=5)

    # Decision rule
    ap.add_argument("--decision", choices=["pvalue","midp","randomized","threshold"], default="randomized")

    # Early stopping CI
    ap.add_argument("--stop_ci", choices=["hoeffding","wilson"], default="wilson")
    ap.add_argument("--delta", type=float, default=1e-2)
    ap.add_argument("--min_b_check", type=int, default=100)

    # Chunking and antithetics
    ap.add_argument("--chunk", type=int, default=128, help="Permutations per batch")
    ap.add_argument("--antithetic", action="store_true", help="Use antithetic permutation pairs (π, π^R)")

    # Parallelism
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--single_thread_blas", action="store_true",
                    help="Set OMP/MKL threads=1 inside workers to avoid oversubscription")

    args = ap.parse_args()

    if args.single_thread_blas:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    rng = make_rng(args.seed)

    L_coarse = None
    if args.L:
        L_coarse = [int(x) for x in args.L.split(",") if x.strip() != ""]

    jobs = args.jobs or os.cpu_count() or 1

    print(f"[Parallel fast role-swapping χ² | engine={args.engine}]")
    print(f" d={args.d}, N={args.N}, K={args.K}, K_ref={args.K_ref or args.K}, "
          f"B={args.B}, R={args.R}, alpha={args.alpha}, tie={args.tie}, alpha0={args.alpha0}")
    print(f" decision={args.decision}, kref_switch={args.kref_switch}, target_exp={args.target_exp}, "
          f"L={(L_coarse if L_coarse is not None else 'auto')}, jobs={jobs}, seed={args.seed}")
    print(f" early-stop: stop_ci={args.stop_ci}, delta={args.delta}, min_b_check={args.min_b_check}, "
          f"chunk={args.chunk}, antithetic={args.antithetic}")

    setup = (args.mode, args.d, args.N, args.K, args.B, args.alpha, args.K_ref, args.tie,
             args.alpha0, L_coarse, args.target_exp, args.engine, args.kref_switch,
             args.decision, args.rho, args.nu, args.stop_ci, args.delta, args.min_b_check,
             args.chunk, args.antithetic)

    ss = np.random.SeedSequence(args.seed)
    rep_seeds = ss.spawn(args.R)

    rejects, times, used_list = [], [], []
    if jobs == 1:
        with tqdm(total=args.R, desc="Total H0 repetitions", leave=True) as pbar:
            for s in rep_seeds:
                r, ms, used = run_one_rep(s, setup)
                rejects.append(r); times.append(ms); used_list.append(used)
                pbar.update(1)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            it = ex.map(run_one_rep, rep_seeds, (setup,)*args.R, chunksize=max(1, args.R // (jobs*4)))
            with tqdm(total=args.R, desc="Total H0 repetitions", leave=True) as pbar:
                for r, ms, used in it:
                    rejects.append(r); times.append(ms); used_list.append(used)
                    pbar.update(1)

    R = args.R
    a_hat = float(np.mean(rejects)) if R else 0.0
    se_hat = se_binom(a_hat, R) if R else 0.0
    avg_ms = float(np.mean(times)) if times else 0.0
    avg_used = float(np.mean(used_list)) if used_list else 0.0

    print("\n[Empirical Type-I under H0]")
    print("---------------------------")
    print(f"α̂ (reject/total): {a_hat:.4f}   SE ≈ {se_hat:.4f}")
    print(f"Avg time per repetition: {avg_ms:8.2f} ms")
    print(f"Avg permutations used (early-stop): {avg_used:8.1f} / B={args.B}")
    print("Notes:")
    print(" • Antithetic pairs (π, π^R) reduce MC variance of the tail estimator.")
    print(" • Chunking reduces Python overhead; Wilson CI usually stops earlier than Hoeffding.")
    print(" • For exact-α at ties use --decision randomized; --tie jitter is recommended.")
    print(" • Coarsen (auto or --L) to avoid ultra-sparse HY when d or K_ref grows.")

if __name__ == "__main__":
    main()
