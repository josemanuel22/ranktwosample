# c2st.py
# -------------------------------------------------------------------
# Classifier Two-Sample Test (C2ST) with permutation calibration.
#   - Train a classifier to distinguish X (label=1) vs Y (label=0)
#   - Score out-of-sample via k-fold CV or a hold-out split
#   - Permutation test: shuffle labels and repeat training to get p-val
#   - Early stopping (Wilson/Hoeffding) to cut permutations
#   - Optional "retrain=False" fast mode (train once; permute test labels)
#
# Requires: scikit-learn, numpy
# -------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Tuple
import time
import math
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, train_test_split

Array = np.ndarray

__all__ = ["C2STResult", "c2st_permutation_test", "C2ST"]

# ---------- result containers ----------


@dataclass
class C2STResult:
    stat_obs: float
    p_perm: float
    p_mid: float
    reject: bool
    alpha: float
    metric: str
    cv_mode: str
    folds: int
    perms_used: int
    train_seconds: float
    note: str


# ---------- utilities ----------
def _ndtri(p: float) -> float:
    # Fast normal quantile (Hastings)
    if p <= 0.0:
        return -1e300
    if p >= 1.0:
        return 1e300
    a = [-39.6968302866538, 220.946098424521, -275.928510446969,
         138.357751867269, -30.6647980661472, 2.50662827745924]
    b = [-54.4760987982241, 161.585836858041, -155.698979859887,
         66.8013118877197, -13.2806815528857]
    c = [-0.007784894002430293, -0.322396458041136, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.93816398269878]
    d = [0.007784695709041462, 0.3224671290700398,
         2.445134137142996, 3.754408661907416]
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
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def _wilson_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    z = _ndtri(1 - delta / 2.0)
    phat = k / n
    denom = 1 + (z*z) / n
    center = (phat + (z*z) / (2*n)) / denom
    half = (z / denom) * math.sqrt(max(0.0, phat *
                                       (1 - phat) / n + (z*z) / (4 * n * n)))
    return max(0.0, center - half), min(1.0, center + half)


def _hoeffding_ci(k: int, n: int, delta: float) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    r = math.sqrt(max(0.0, math.log(2.0 / max(1e-16, delta))) / (2.0 * n))
    return max(0.0, phat - r), min(1.0, phat + r)


def _make_default_estimator(random_state: int):
    # Logistic regression with scaling; balanced handles class imbalance (N != K)
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=random_state,
        )
    )


def _stat_from_scores(
    y_true: Array, y_prob: Optional[Array], y_pred: Optional[Array], metric: str
) -> float:
    """
    Larger-is-better statistic:
      - acc: accuracy in [0,1]
      - auc: ROC AUC in [0,1] (falls back to 0.5 if undefined)
      - logloss: we use T = -log_loss (so larger is better)
    """
    if metric == "acc":
        if y_pred is None and y_prob is not None:
            y_pred = (y_prob >= 0.5).astype(int)
        return float(accuracy_score(y_true, y_pred))

    if metric == "auc":
        if y_prob is None:
            y_prob = y_pred.astype(float)
        try:
            return float(roc_auc_score(y_true, y_prob))
        except Exception:
            return 0.5  # undefined AUC

    if metric == "logloss":
        if y_prob is None:
            y_prob = y_pred.astype(float)
        try:
            return -float(log_loss(y_true, y_prob, labels=[0, 1]))
        except Exception:
            return -np.log(2.0)  # random guess baseline

    raise ValueError("metric must be 'acc', 'auc', or 'logloss'")


def _cv_splits(
    y: Array,
    mode: Literal["kfold", "holdout"],
    k: int,
    split_prop: float,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return a list of (train_idx, test_idx) splits reused across permutations.
    """
    n = y.shape[0]
    if mode == "kfold":
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in skf.split(np.zeros(n), y)]
    elif mode == "holdout":
        tr, te = train_test_split(
            np.arange(n), test_size=(1 - split_prop), stratify=y, random_state=seed
        )
        return [(tr, te)]
    else:
        raise ValueError("cv_mode must be 'kfold' or 'holdout'")


def _fit_score_once(
    X: Array, y: Array, splits: List[Tuple[np.ndarray, np.ndarray]],
    estimator_factory: Callable[[], object],
    metric: Literal["acc", "auc", "logloss"],
) -> float:
    stats = []
    for (tr, te) in splits:
        clf = estimator_factory()
        clf.fit(X[tr], y[tr])
        y_prob = None
        y_pred = None
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X[te])[:, 1]
        elif hasattr(clf, "decision_function"):
            s = clf.decision_function(X[te])
            y_prob = 1 / (1 + np.exp(-s))
        else:
            y_pred = clf.predict(X[te])
        stat = _stat_from_scores(y[te], y_prob, y_pred, metric)
        stats.append(stat)
    return float(np.mean(stats)) if stats else 0.5


# ---------- main function API ----------
def c2st_permutation_test(
    X: Array, Y: Array, *,
    estimator_factory: Optional[Callable[[], object]] = None,
    metric: Literal["acc", "auc", "logloss"] = "acc",
    cv_mode: Literal["kfold", "holdout"] = "kfold",
    k_folds: int = 5,
    split_prop: float = 0.5,                 # for holdout
    B: int = 1000,
    alpha: float = 0.05,
    decision: Literal["pvalue", "midp"] = "pvalue",
    early_stop: Literal["none", "wilson", "hoeffding"] = "wilson",
    delta_ci: float = 1e-2,
    min_b_check: int = 50,
    retrain: bool = True,
    seed: int = 2027,
) -> C2STResult:
    """
    C2ST with permutation calibration.
    If retrain=True: exact conditional permutation test (retrain each perm).
    If retrain=False: train once; permute labels on test fold(s) only (fast, slightly liberal).
    """
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n1, n0 = X.shape[0], Y.shape[0]
    Xall = np.vstack([X, Y])
    y = np.concatenate([np.ones(n1, dtype=int), np.zeros(n0, dtype=int)])

    if estimator_factory is None:
        def estimator_factory():
            return _make_default_estimator(rng.integers(1, 2**31 - 1))

    # Fixed CV splits across permutations
    splits = _cv_splits(y, cv_mode, k_folds, split_prop, seed)

    # Observed statistic
    stat_obs = _fit_score_once(Xall, y, splits, estimator_factory, metric)

    # Permutation loop
    gt = eq = ge = 0
    perms_used = 0

    # Train-once fast mode
    trained_models = None
    if not retrain:
        trained_models = []
        for (tr, te) in splits:
            clf = estimator_factory()
            clf.fit(Xall[tr], y[tr])
            trained_models.append((clf, te))

    for b in range(1, B + 1):
        if retrain:
            yb = np.random.default_rng(seed + b).permutation(y)
            stat_b = _fit_score_once(
                Xall, yb, splits, estimator_factory, metric)
        else:
            stats = []
            for clf, te in trained_models:
                y_perm = np.random.default_rng(seed + b).permutation(y[te])
                y_prob = None
                y_pred = None
                if hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(Xall[te])[:, 1]
                elif hasattr(clf, "decision_function"):
                    s = clf.decision_function(Xall[te])
                    y_prob = 1 / (1 + np.exp(-s))
                else:
                    y_pred = clf.predict(Xall[te])
                stats.append(_stat_from_scores(y_perm, y_prob, y_pred, metric))
            stat_b = float(np.mean(stats))

        if stat_b > stat_obs:
            gt += 1
        elif stat_b == stat_obs:
            eq += 1
        ge = gt + eq
        perms_used = b

        # Early stop
        if early_stop != "none" and b >= min_b_check:
            if early_stop == "wilson":
                lo, hi = _wilson_ci(ge, b, delta_ci)
            else:
                lo, hi = _hoeffding_ci(ge, b, delta_ci)
            if hi <= alpha or lo > alpha:
                break

    b = max(1, perms_used)
    p_perm = (1 + ge) / (b + 1)
    p_mid = (gt + 0.5 * eq) / b
    p_use = p_perm if decision == "pvalue" else p_mid
    reject = (p_use <= alpha)

    t1 = time.perf_counter()
    note = ("C2ST with k-fold CV" if cv_mode == "kfold" else "C2ST with hold-out") + \
           (", retrain-per-permutation (exact)" if retrain else ", train-once permuted labels (fast, approximate)")

    return C2STResult(
        stat_obs=float(stat_obs),
        p_perm=float(p_perm),
        p_mid=float(p_mid),
        reject=bool(reject),
        alpha=float(alpha),
        metric=metric,
        cv_mode=cv_mode,
        folds=(k_folds if cv_mode == "kfold" else 1),
        perms_used=int(perms_used),
        train_seconds=float(t1 - t0),
        note=note,
    )


# ---------- OO wrapper for a uniform baseline API ----------
class C2ST:
    """
    Uniform class API so you can do:

        from baselines.c2st import C2ST
        test = C2ST(alpha=0.05, cv_mode="kfold", k_folds=5)
        out = test.run(X, Y)
        print(out.p_perm, out.reject)
    """

    def __init__(
        self,
        *,
        estimator_factory: Optional[Callable[[], object]] = None,
        metric: Literal["acc", "auc", "logloss"] = "acc",
        cv_mode: Literal["kfold", "holdout"] = "kfold",
        k_folds: int = 5,
        split_prop: float = 0.5,
        B: int = 1000,
        alpha: float = 0.05,
        decision: Literal["pvalue", "midp"] = "pvalue",
        early_stop: Literal["none", "wilson", "hoeffding"] = "wilson",
        delta_ci: float = 1e-2,
        min_b_check: int = 50,
        retrain: bool = True,
        seed: int = 2027,
    ):
        self.estimator_factory = estimator_factory
        self.metric = metric
        self.cv_mode = cv_mode
        self.k_folds = k_folds
        self.split_prop = split_prop
        self.B = B
        self.alpha = alpha
        self.decision = decision
        self.early_stop = early_stop
        self.delta_ci = delta_ci
        self.min_b_check = min_b_check
        self.retrain = retrain
        self.seed = seed

    def run(self, X: Array, Y: Array) -> C2STResult:
        return c2st_permutation_test(
            X, Y,
            estimator_factory=self.estimator_factory,
            metric=self.metric,
            cv_mode=self.cv_mode,
            k_folds=self.k_folds,
            split_prop=self.split_prop,
            B=self.B,
            alpha=self.alpha,
            decision=self.decision,
            early_stop=self.early_stop,
            delta_ci=self.delta_ci,
            min_b_check=self.min_b_check,
            retrain=self.retrain,
            seed=self.seed,
        )

    # Monte Carlo helpers
    def estimate_type1(
        self,
        gen_null: Callable[[np.random.Generator], Tuple[Array, Array]],
        *,
        R: int = 200,
        seed: int = 4041,
    ) -> dict:
        ss = np.random.SeedSequence(seed)
        rej = 0
        for s in ss.spawn(R):
            rng = np.random.default_rng(s.entropy)
            X, Y = gen_null(rng)
            out = self.run(X, Y)
            rej += int(out.reject)
        p = rej / R
        se = float(np.sqrt(max(p * (1 - p), 1e-12) / max(1, R)))
        return {"type1_hat": p, "se": se}

    def estimate_power(
        self,
        gen_alt: Callable[[np.random.Generator], Tuple[Array, Array]],
        *,
        R: int = 200,
        seed: int = 4042,
    ) -> dict:
        ss = np.random.SeedSequence(seed)
        rej = 0
        for s in ss.spawn(R):
            rng = np.random.default_rng(s.entropy)
            X, Y = gen_alt(rng)
            out = self.run(X, Y)
            rej += int(out.reject)
        p = rej / R
        se = float(np.sqrt(max(p * (1 - p), 1e-12) / max(1, R)))
        return {"power_hat": p, "se": se}
