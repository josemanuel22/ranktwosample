#!/usr/bin/env python3
# -------------------------------------------------------------------
# Parallel fast role-swapping (non-symmetric) χ² two-sample test
#  • Outer repetitions (R) parallelized with ProcessPoolExecutor
#  • Clean RNG splitting via numpy.random.SeedSequence.spawn(R)
#  • Observed: ref = subset of Y (size K_ref); targets = X (size N)
#  • Permutation: pick Y-pool of size K from pooled Z; targets=complement;
#                 ref = subset of that permuted Y (size K_ref)
#  • Rank engines:
#       - searchsorted : O(d · (K_ref log K_ref + m log K_ref)) per perm
#       - prefixsum    : one cumsum per dim per perm (good when K_ref large)
#       - auto         : choose by K_ref threshold (kref_switch)
#  • Optional coarsening L1×…×Ld, Jeffreys smoothing α0, mid-p/randomized
#  • Reports empirical type-I under H0 (α-hat + SE) and timing
#  • Requires: numpy>=1.20, tqdm
# -------------------------------------------------------------------
import argparse, os, time
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


# --------- Precompute global orders and tie-run maps (once per rep) ---------
def precompute_orders(Z, tie="right", jitter_scale=1e-12, rng=None):
    """
    For each dim j:
      order_j: argsort of Z[:, j]
      pos_j[i]: position of i in order_j
      last_right_j[pos] / first_left_j[pos]: end/start of equal-run at pos
    Used by prefix-sum engine. If tie='jitter', sort jittered Z.
    """
    n_tot, d = Z.shape
    if tie == "jitter":
        if rng is None: rng = np.random.default_rng(0)
        Zw = Z + rng.normal(scale=jitter_scale, size=Z.shape)
        side = "right"
    else:
        Zw = Z
        side = tie  # 'right' or 'left'

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
    """
    m: (#points, d) ranks in {0..K_ref}
    If L_vec is None -> fine-grid linear index.
    Else mc = floor(m * L / (K_ref+1)) -> linear with w_coarse.
    """
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
    """
    Choose L_j so M = ∏ L_j ≈ N_targets / target_exp, each L_j ≤ K_ref+1.
    """
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
    """
    Prefix-sum engine: O(d * n_tot) for S per perm, then O(d * |indices|) queries.
    """
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
    """
    Searchsorted engine: O(d * (K_ref log K_ref + (|indices|) log K_ref)).
    If jitter provided (shape (d, n_tot)), add to Z_cols to break ties.
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


# ----------- One repetition (worker) -----------
def roleswap_fast_once(
    X, Y, B, alpha, K_ref, tie, alpha0, L_coarse, target_exp,
    rng, jitter_scale, engine, kref_switch, decision
):
    """
    One repetition of role-swapping fast non-symmetric test.
    Returns (reject:int, elapsed_ms:float).
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
        sel_obs = np.zeros(n_tot, dtype=np.uint8); sel_obs[Iy_obs] = 1
        mX_obs = ranks_prefixsum(idx_X_obs, sel_obs, ords, side_pick)
        mY_obs = ranks_prefixsum(Iy_obs,     sel_obs, ords, side_pick)

    idx_lin_X = coarse_linear_index(mX_obs, K_ref, L, w_coarse)
    idx_lin_Y = coarse_linear_index(mY_obs, K_ref, L, w_coarse)
    HX_obs = np.bincount(idx_lin_X, minlength=M_eff)
    HY_obs = np.bincount(idx_lin_Y, minlength=M_eff)
    T_obs  = pearson_asym_from_counts(HX_obs, HY_obs, N_local=N, Kref_local=K_ref, alpha0_local=alpha0)

    # --------------- Permutations ---------------
    T_perm = np.empty(B, dtype=float)
    for b in range(B):
        perm = rng.permutation(n_tot)
        Iy_pool = perm[:K]                    # permuted Y pool
        Ix_pool = perm[K:]                    # permuted targets
        Iref    = rng.choice(Iy_pool, size=K_ref, replace=False)

        if use_searchsorted:
            mX = ranks_searchsorted(Ix_pool, Iref, Z_cols, side=ss_side, jitter=jitter)
            mY = ranks_searchsorted(Iref,    Iref, Z_cols, side=ss_side, jitter=jitter)
        else:
            sel = np.zeros(n_tot, dtype=np.uint8); sel[Iref] = 1
            mX = ranks_prefixsum(Ix_pool, sel, ords, side_pick)
            mY = ranks_prefixsum(Iref,    sel, ords, side_pick)

        idxX = coarse_linear_index(mX, K_ref, L, w_coarse)
        idxY = coarse_linear_index(mY, K_ref, L, w_coarse)
        HX = np.bincount(idxX, minlength=M_eff)
        HY = np.bincount(idxY, minlength=M_eff)
        T_perm[b] = pearson_asym_from_counts(HX, HY, N_local=N, Kref_local=K_ref, alpha0_local=alpha0)

    ge = int((T_perm >= T_obs).sum())
    gt = int((T_perm >  T_obs).sum())
    eq = ge - gt

    # decision
    p_perm = (1 + ge) / (B + 1)
    p_mid  = (gt + 0.5 * eq) / B if B > 0 else 1.0

    if decision == "pvalue":
        reject = (p_perm <= alpha)
    elif decision == "midp":
        reject = (p_mid <= alpha)
    elif decision == "randomized":
        p_lower = (1 + gt) / (B + 1)
        if p_lower > alpha:
            reject = False
        elif eq == 0:
            reject = True
        else:
            omega = (alpha - p_lower) / (eq / (B + 1))
            omega = float(np.clip(omega, 0.0, 1.0))
            reject = (rng.random() <= omega)
    else:
        # fallback: threshold on T_perm
        k = max(0, min(B - 1, int(np.ceil((B + 1) * (1 - alpha))) - 1))
        tau_alpha = float(np.partition(T_perm, k)[k])
        reject = bool(T_obs >= tau_alpha)

    elapsed_ms = 1e3 * (time.perf_counter() - t0)
    return int(reject), float(elapsed_ms)


# ---------------- Utilities ----------------
def se_binom(a_hat, R):
    return float(np.sqrt(max(a_hat * (1 - a_hat), 1e-12) / R))


# ---------------- Worker wrapper for executor ----------------
def run_one_rep(seed_seq, setup):
    """
    Build rng from SeedSequence; draw X,Y; run one repetition; return (reject, ms)
    """
    (mode, d, N, K, B, alpha, K_ref, tie, alpha0,
     L_coarse, target_exp, engine, kref_switch,
     decision, rho, nu) = setup

    rng = make_rng(seed_seq)
    X = draw_null(mode, d, N, rng, rho=rho, nu=nu)
    Y = draw_null(mode, d, K, rng, rho=rho, nu=nu)

    r, ms = roleswap_fast_once(
        X, Y, B=B, alpha=alpha, K_ref=K_ref,
        tie=tie, alpha0=alpha0, L_coarse=L_coarse, target_exp=target_exp,
        rng=rng, jitter_scale=1e-12, engine=engine, kref_switch=kref_switch,
        decision=decision
    )
    return r, ms


# ---------------- CLI driver ----------------
def main():
    ap = argparse.ArgumentParser(description="Parallel fast role-swapping χ² (non-symmetric)")
    ap.add_argument("--mode", choices=["null-gauss", "null-copula", "null-studentt"], default="null-gauss")
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--K_ref", type=int, default=None, help="Reference size (subset of Y), default K")
    ap.add_argument("--B", type=int, default=1000, help="Permutations per repetition")
    ap.add_argument("--R", type=int, default=200, help="Outer repetitions under H0")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--alpha0", type=float, default=0.5, help="Jeffreys smoothing per cell")
    ap.add_argument("--tie", choices=["right","left","jitter"], default="jitter")
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--nu", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2025)

    # Rank engine + coarsening
    ap.add_argument("--engine", choices=["auto","searchsorted","prefixsum"], default="auto")
    ap.add_argument("--kref_switch", type=int, default=64, help="When engine=auto, use searchsorted if K_ref ≤ this")
    ap.add_argument("--L", type=str, default=None, help="Coarsen to L1,...,Ld (e.g. '3,3,3,3'); omit for auto")
    ap.add_argument("--target_exp", type=int, default=5, help="Auto-coarsen target expected count per cell")

    # Decision rule
    ap.add_argument("--decision", choices=["pvalue","midp","randomized","threshold"], default="randomized")

    # Parallelism
    ap.add_argument("--jobs", type=int, default=None, help="Parallel workers over repetitions (default: os.cpu_count())")
    ap.add_argument("--single_thread_blas", action="store_true",
                    help="Set OMP/MKL threads=1 inside workers to avoid oversubscription")

    args = ap.parse_args()

    # Optional: restrict BLAS threads to avoid oversubscription
    if args.single_thread_blas:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

    rng = make_rng(args.seed)

    # Parse L
    L_coarse = None
    if args.L:
        L_coarse = [int(x) for x in args.L.split(",") if x.strip() != ""]

    jobs = args.jobs or os.cpu_count() or 1

    print(f"[Parallel fast role-swapping χ² | engine={args.engine}]")
    print(f" d={args.d}, N={args.N}, K={args.K}, K_ref={args.K_ref or args.K}, "
          f"B={args.B}, R={args.R}, alpha={args.alpha}, tie={args.tie}, alpha0={args.alpha0}")
    print(f" decision={args.decision}, kref_switch={args.kref_switch}, target_exp={args.target_exp}, "
          f"L={(L_coarse if L_coarse is not None else 'auto')}, jobs={jobs}, seed={args.seed}")

    # Immutable setup for workers
    setup = (args.mode, args.d, args.N, args.K, args.B, args.alpha, args.K_ref, args.tie,
             args.alpha0, L_coarse, args.target_exp, args.engine, args.kref_switch,
             args.decision, args.rho, args.nu)

    # Spawn independent seeds for each repetition
    ss = np.random.SeedSequence(args.seed)
    rep_seeds = ss.spawn(args.R)  # list of SeedSequence objects

    rejects, times = [], []
    if jobs == 1:
        # Serial (still nice to test correctness)
        with tqdm(total=args.R, desc="Total H0 repetitions", leave=True) as pbar:
            for s in rep_seeds:
                r, ms = run_one_rep(s, setup)
                rejects.append(r); times.append(ms)
                pbar.update(1)
    else:
        # Parallel over repetitions
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            # map preserves order; wrap with tqdm for one progress bar
            it = ex.map(run_one_rep, rep_seeds, (setup,)*args.R, chunksize=max(1, args.R // (jobs*4)))
            with tqdm(total=args.R, desc="Total H0 repetitions", leave=True) as pbar:
                for r, ms in it:
                    rejects.append(r); times.append(ms)
                    pbar.update(1)

    # Summaries
    R = args.R
    a_hat = float(np.mean(rejects)) if R else 0.0
    se_hat = se_binom(a_hat, R) if R else 0.0
    avg_ms = float(np.mean(times)) if times else 0.0

    print("\n[Empirical Type-I under H0]")
    print("---------------------------")
    print(f"α̂ (reject/total): {a_hat:.4f}   SE ≈ {se_hat:.4f}")
    print(f"Avg time per repetition: {avg_ms:8.2f} ms")
    print("Notes:")
    print(" • Parallelizes only the outer repetitions (R) to keep IPC light.")
    print(" • Use --decision randomized for exact-α at tie boundaries; mid-p reduces conservativeness.")
    print(" • Coarsen (auto via --target_exp or explicit --L) for stability/speed when d or K_ref grows.")
    print(" • For best calibration under H0, prefer the symmetric counts-only MVHG variant when feasible.")

if __name__ == "__main__":
    main()
