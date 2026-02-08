#!/usr/bin/env python3
"""
Reproducible experiment for:
- Grassmann projection-kernel SVM with balanced-accuracy threshold tuning (subject-level)
- Subject-level Euclidean baselines using mean+std aggregation:
  * class-weighted logistic regression
  * SMOTE + logistic regression
  * BalancedRandomForestClassifier

Inputs:
  - pd_speech_features.csv (from Kaggle/UCI mirror)

Outputs (in current working directory):
  - fold_metrics_main_v2.csv
  - summary_metrics_main_v2.csv
  - ablation_fold_metrics_v2.csv
  - ablation_summary_v2.csv
  - fig1_pipeline_v2.png
  - fig2_balanced_accuracy_v3.png
  - fig3_f1_v3.png
  - fig4_confusion_matrix_v2.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier

import matplotlib.pyplot as plt
import matplotlib.patches as patches


RANDOM_STATE = 0


# --------------------------
# Grassmann utilities
# --------------------------
def orthonormal_basis_from_matrix(M: np.ndarray, r: int) -> np.ndarray:
    """Return an orthonormal basis of the column space of M via thin SVD."""
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :r]


def projection_kernel(Qa: np.ndarray, Qb: np.ndarray) -> float:
    """k(Qa,Qb) = ||Qa^T Qb||_F^2"""
    M = Qa.T @ Qb
    return float(np.sum(M * M))


def build_kernel_matrix(QA: list[np.ndarray], QB: list[np.ndarray] | None = None) -> np.ndarray:
    if QB is None:
        QB = QA
    n, m = len(QA), len(QB)
    K = np.zeros((n, m), dtype=float)
    for i in range(n):
        Qi = QA[i]
        for j in range(m):
            K[i, j] = projection_kernel(Qi, QB[j])
    return K


def tune_C_precomputed(K: np.ndarray, y: np.ndarray, C_grid=(0.1, 1.0, 10.0), seed: int = 0) -> float:
    """Inner 3-fold CV on subject-level kernel matrix."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_C = None
    best_ba = -1.0
    best_f1 = -1.0
    for C in C_grid:
        bas, f1s = [], []
        for tr_idx, va_idx in skf.split(np.zeros(len(y)), y):
            K_tr = K[np.ix_(tr_idx, tr_idx)]
            K_va = K[np.ix_(va_idx, tr_idx)]
            clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
            clf.fit(K_tr, y[tr_idx])
            pred = (clf.decision_function(K_va) > 0).astype(int)
            bas.append(balanced_accuracy_score(y[va_idx], pred))
            f1s.append(f1_score(y[va_idx], pred, pos_label=1))
        mean_ba, mean_f1 = float(np.mean(bas)), float(np.mean(f1s))
        if (mean_ba > best_ba + 1e-9) or (abs(mean_ba - best_ba) < 1e-9 and mean_f1 > best_f1):
            best_ba, best_f1, best_C = mean_ba, mean_f1, C
    assert best_C is not None
    return float(best_C)


def choose_threshold_max_ba(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick threshold maximizing balanced accuracy over score quantiles."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    best_tau = float(candidates[0])
    best_ba = -1.0
    for tau in candidates:
        pred = (scores > tau).astype(int)
        ba = balanced_accuracy_score(y_true, pred)
        if ba > best_ba:
            best_ba = ba
            best_tau = float(tau)
    return best_tau


# --------------------------
# Figure helpers
# --------------------------
def make_pipeline_fig(out_path: str) -> None:
    plt.close("all")
    fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=220)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#1A1A1A"):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.0,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color="#111111")
        return rect

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))

    ax.text(
        0.50,
        0.94,
        "Proposed Grassmannian pipeline vs. imbalance-aware baselines",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="semibold",
        color="#111111",
    )

    box(0.02, 0.58, 0.20, 0.32, "PD speech features\n756 recordings\n252 subjects × 3 reps", fc="#F7F7F7")
    box(0.02, 0.10, 0.20, 0.32, "Leakage-safe split\n5-fold GroupKFold\n(group = subject id)", fc="#F7F7F7")
    arrow(0.22, 0.74, 0.28, 0.74)
    arrow(0.22, 0.26, 0.28, 0.26)

    ax.text(0.50, 0.52, "Proposed method (kernel learning on $\\mathrm{Gr}(3,d)$)", ha="center", va="center", fontsize=11, color="#1D4E89")
    ax.text(0.50, 0.04, "Baselines (Euclidean aggregation + SMOTE / BalancedRF)", ha="center", va="center", fontsize=11, color="#8A5A00")

    box(0.28, 0.60, 0.22, 0.28, "Standardize\n(train-fold only)\n$\\tilde x=(x-\\mu)/\\sigma$", fc="#EAF2FB", ec="#1D4E89")
    box(0.52, 0.60, 0.22, 0.28, "Subject-to-subspace\n$M_i=[\\tilde x_{i1},\\tilde x_{i2},\\tilde x_{i3}]$\n$M_i=U\\Sigma V^\\top$\n$Q_i=U_{:,1:3}$", fc="#EAF2FB", ec="#1D4E89")
    box(0.76, 0.60, 0.22, 0.28, "Grassmann kernel SVM\n$k(Q_i,Q_j)=\\|Q_i^\\top Q_j\\|_F^2$\n+ threshold tuning\n(max BA in CV)", fc="#EAF2FB", ec="#1D4E89")
    arrow(0.50, 0.74, 0.52, 0.74)
    arrow(0.74, 0.74, 0.76, 0.74)
    arrow(0.22, 0.74, 0.28, 0.74)

    box(0.28, 0.12, 0.22, 0.28, "Aggregate per subject\n(mean + std across 3 reps)", fc="#FFF5E6", ec="#8A5A00")
    box(0.52, 0.12, 0.22, 0.28, "Imbalance handling\nSMOTE or\nBalanced Random Forest", fc="#FFF5E6", ec="#8A5A00")
    box(0.76, 0.12, 0.22, 0.28, "Standard ML classifiers\n(LogReg, BRF)\nEvaluate on subjects\n(BalAcc, F1)", fc="#FFF5E6", ec="#8A5A00")
    arrow(0.22, 0.26, 0.28, 0.26)
    arrow(0.50, 0.26, 0.52, 0.26)
    arrow(0.74, 0.26, 0.76, 0.26)

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_metric_bar(summary_df: pd.DataFrame, metric: str, out_path: str, y_start: float, y_end: float) -> None:
    plt.close("all")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=220, constrained_layout=True)

    models_order = [
        "Grassmann ProjKernel SVM (threshold-tuned)",
        "LogReg (mean+std, balanced)",
        "SMOTE + LogReg (mean+std)",
        "BalancedRF (mean+std)",
    ]
    means = [summary_df.loc[m, (metric, "mean")] for m in models_order]
    stds = [summary_df.loc[m, (metric, "std")] for m in models_order]
    x = np.arange(len(models_order))
    colors = ["#1D4E89", "#6BBBAE", "#F2C14E", "#B9B9B9"]

    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="#222222", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Grassmann\n(ours)", "LogReg\n(balanced)", "SMOTE\n+LogReg", "Balanced\nRF"], fontsize=10)
    ax.set_ylim(y_start, y_end)
    ax.set_ylabel("Balanced Accuracy" if metric == "balanced_acc" else "F1 Score (PD positive)", fontsize=11)
    ax.set_title(f"Subject-level {'Balanced Accuracy' if metric=='balanced_acc' else 'F1 Score'} (5-fold GroupKFold)", fontsize=12, pad=8)

    for b, m, s in zip(bars, means, stds):
        ytxt = min(m + s + 0.003, y_end - 0.004)
        ax.text(b.get_x() + b.get_width() / 2, ytxt, f"{m:.3f}", ha="center", va="bottom", fontsize=9, color="#111111")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_confusion_matrix(cm: np.ndarray, out_path: str) -> None:
    plt.close("all")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=220, constrained_layout=True)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["Control (0)", "PD (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion matrix (pooled over folds)\nGrassmann ProjKernel SVM (threshold-tuned)",
    )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "#111111",
                fontsize=12,
                fontweight="semibold",
            )
    ax.grid(False)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------
# Main experiment
# --------------------------
def main():
    df = pd.read_csv("pd_speech_features.csv")
    assert "id" in df.columns and "class" in df.columns

    feature_cols = [c for c in df.columns if c not in ["id", "class"]]
    X_rows = df[feature_cols].to_numpy()
    y_rows = df["class"].to_numpy().astype(int)
    groups = df["id"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    outer_splits = []
    for tr, te in gkf.split(X_rows, y_rows, groups=groups):
        outer_splits.append((np.unique(groups[tr]), np.unique(groups[te])))

    # Subject-level baseline features: mean+std across repetitions
    subj_stats = df.groupby("id")[feature_cols].agg(["mean", "std"])
    subj_stats.columns = [f"{c}_{stat}" for c, stat in subj_stats.columns]
    subj_y = df.groupby("id")["class"].first()
    subj_ids = subj_y.index.to_numpy()
    X_sub = subj_stats.loc[subj_ids].to_numpy()
    y_sub = subj_y.loc[subj_ids].to_numpy().astype(int)
    id_to_idx = {sid: i for i, sid in enumerate(subj_ids)}

    def eval_baseline(model, name):
        rows = []
        for fold, (train_ids, test_ids) in enumerate(outer_splits, start=1):
            tr_idx = [id_to_idx[s] for s in train_ids]
            te_idx = [id_to_idx[s] for s in test_ids]
            Xtr, Xte = X_sub[tr_idx], X_sub[te_idx]
            ytr, yte = y_sub[tr_idx], y_sub[te_idx]
            mdl = clone(model)
            mdl.fit(Xtr, ytr)
            s = mdl.predict_proba(Xte)[:, 1]
            pred = (s >= 0.5).astype(int)
            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "balanced_acc": balanced_accuracy_score(yte, pred),
                    "f1": f1_score(yte, pred, pos_label=1),
                }
            )
        return pd.DataFrame(rows)

    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear")),
    ])

    smote_logreg = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
        ("clf", LogisticRegression(max_iter=5000, solver="liblinear")),
    ])

    brf = BalancedRandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    baseline_results = pd.concat(
        [
            eval_baseline(logreg, "LogReg (mean+std, balanced)"),
            eval_baseline(smote_logreg, "SMOTE + LogReg (mean+std)"),
            eval_baseline(brf, "BalancedRF (mean+std)"),
        ],
        ignore_index=True,
    )

    # Proposed Grassmann method (on raw rows, but evaluated on subjects)
    grass_rows = []
    all_true = []
    all_pred = []

    for fold, (tr, te) in enumerate(gkf.split(X_rows, y_rows, groups=groups), start=1):
        Xtr_raw, Xte_raw = X_rows[tr], X_rows[te]
        ytr_raw, yte_raw = y_rows[tr], y_rows[te]
        id_tr, id_te = groups[tr], groups[te]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        subj_train = pd.DataFrame({"id": id_tr, "y": ytr_raw}).groupby("id")["y"].first()
        subj_test = pd.DataFrame({"id": id_te, "y": yte_raw}).groupby("id")["y"].first()
        train_ids = subj_train.index.to_numpy()
        test_ids = subj_test.index.to_numpy()

        # Build Grassmann points
        Q_train = []
        for sid in train_ids:
            idx = np.where(id_tr == sid)[0]
            M = Xtr[idx, :].T  # d x 3
            Q_train.append(orthonormal_basis_from_matrix(M, r=3))
        y_train = subj_train.loc[train_ids].to_numpy().astype(int)

        Q_test = []
        for sid in test_ids:
            idx = np.where(id_te == sid)[0]
            M = Xte[idx, :].T
            Q_test.append(orthonormal_basis_from_matrix(M, r=3))
        y_test = subj_test.loc[test_ids].to_numpy().astype(int)

        K_train = build_kernel_matrix(Q_train)
        C_best = tune_C_precomputed(K_train, y_train, seed=RANDOM_STATE + fold)

        # Threshold selection via CV scores
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold)
        scores_cv = np.zeros(len(y_train))
        for tri, vai in skf.split(np.zeros(len(y_train)), y_train):
            K_tri = K_train[np.ix_(tri, tri)]
            K_vai = K_train[np.ix_(vai, tri)]
            clf = SVC(kernel="precomputed", C=C_best, class_weight="balanced")
            clf.fit(K_tri, y_train[tri])
            scores_cv[vai] = clf.decision_function(K_vai)
        tau = choose_threshold_max_ba(y_train, scores_cv)

        clf_full = SVC(kernel="precomputed", C=C_best, class_weight="balanced")
        clf_full.fit(K_train, y_train)

        K_test = build_kernel_matrix(Q_test, Q_train)
        scores_test = clf_full.decision_function(K_test)
        pred = (scores_test > tau).astype(int)

        grass_rows.append(
            {
                "model": "Grassmann ProjKernel SVM (threshold-tuned)",
                "fold": fold,
                "balanced_acc": balanced_accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred, pos_label=1),
            }
        )

        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())

    grass_results = pd.DataFrame(grass_rows)

    # Main results CSVs
    fold_metrics = pd.concat([baseline_results, grass_results], ignore_index=True)
    fold_metrics.to_csv("fold_metrics_main_v2.csv", index=False)
    summary = fold_metrics.groupby("model")[["balanced_acc", "f1"]].agg(["mean", "std"])
    summary.to_csv("summary_metrics_main_v2.csv")

    # Ablation: Grassmann-SMOTE (optional oversampling)
    # (Implemented minimally; retained for completeness but not used for main figure set.)
    def grassmann_smote(Q_list, y_list, minority_label=0, k=5, seed=0):
        rng = np.random.default_rng(seed)
        Q_list = list(Q_list)
        y_list = list(y_list)
        idx_min = [i for i, yy in enumerate(y_list) if yy == minority_label]
        idx_maj = [i for i, yy in enumerate(y_list) if yy != minority_label]
        n_min, n_maj = len(idx_min), len(idx_maj)
        if n_min == 0 or n_min == n_maj:
            return Q_list, np.asarray(y_list)

        # principal angles distance for neighbor selection
        def grassmann_distance(Qa, Qb):
            M = Qa.T @ Qb
            _, s, _ = np.linalg.svd(M, full_matrices=False)
            s = np.clip(s, -1.0, 1.0)
            theta = np.arccos(s)
            return float(np.linalg.norm(theta))

        Q_min = [Q_list[i] for i in idx_min]
        n = len(Q_min)
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = grassmann_distance(Q_min[i], Q_min[j])
                D[i, j] = D[j, i] = d
        neigh = []
        for i in range(n):
            order = np.argsort(D[i])
            order = [j for j in order if j != i]
            neigh.append(order[: min(k, len(order))])

        # geodesic interpolation (with QR re-orthonormalization)
        def geodesic(Qa, Qb, t, eps=1e-8):
            M = Qa.T @ Qb
            U, s, Vt = np.linalg.svd(M, full_matrices=False)
            s = np.clip(s, -1.0, 1.0)
            theta = np.arccos(s)
            V = Vt.T
            B = Qb - Qa @ M
            BV = B @ V
            sin_theta = np.sin(theta)
            Uperp = np.zeros_like(BV)
            for k_ in range(BV.shape[1]):
                if sin_theta[k_] > eps:
                    Uperp[:, k_] = BV[:, k_] / sin_theta[k_]
            cos_t = np.cos(t * theta)
            sin_t = np.sin(t * theta)
            Qt = (Qa @ U) @ np.diag(cos_t) @ V.T + Uperp @ np.diag(sin_t) @ V.T
            Qt, _ = np.linalg.qr(Qt)
            return Qt[:, : Qa.shape[1]]

        n_to_add = n_maj - n_min
        synth = []
        for _ in range(n_to_add):
            a = rng.integers(0, n)
            b = rng.choice(neigh[a]) if neigh[a] else rng.integers(0, n)
            t = rng.random()
            synth.append(geodesic(Q_min[a], Q_min[b], t))

        Q_aug = Q_list + synth
        y_aug = np.asarray(y_list + [minority_label] * len(synth), dtype=int)
        return Q_aug, y_aug

    ab_rows = []
    for fold, (tr, te) in enumerate(gkf.split(X_rows, y_rows, groups=groups), start=1):
        Xtr_raw, Xte_raw = X_rows[tr], X_rows[te]
        ytr_raw, yte_raw = y_rows[tr], y_rows[te]
        id_tr, id_te = groups[tr], groups[te]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        subj_train = pd.DataFrame({"id": id_tr, "y": ytr_raw}).groupby("id")["y"].first()
        subj_test = pd.DataFrame({"id": id_te, "y": yte_raw}).groupby("id")["y"].first()
        train_ids = subj_train.index.to_numpy()
        test_ids = subj_test.index.to_numpy()

        Q_train = []
        for sid in train_ids:
            idx = np.where(id_tr == sid)[0]
            Q_train.append(orthonormal_basis_from_matrix(Xtr[idx, :].T, r=3))
        y_train = subj_train.loc[train_ids].to_numpy().astype(int)

        Q_test = []
        for sid in test_ids:
            idx = np.where(id_te == sid)[0]
            Q_test.append(orthonormal_basis_from_matrix(Xte[idx, :].T, r=3))
        y_test = subj_test.loc[test_ids].to_numpy().astype(int)

        K_train = build_kernel_matrix(Q_train)
        C_best = tune_C_precomputed(K_train, y_train, seed=RANDOM_STATE + fold)

        # threshold from non-oversampled training CV scores
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + fold)
        scores_cv = np.zeros(len(y_train))
        for tri, vai in skf.split(np.zeros(len(y_train)), y_train):
            K_tri = K_train[np.ix_(tri, tri)]
            K_vai = K_train[np.ix_(vai, tri)]
            clf = SVC(kernel="precomputed", C=C_best, class_weight="balanced")
            clf.fit(K_tri, y_train[tri])
            scores_cv[vai] = clf.decision_function(K_vai)
        tau = choose_threshold_max_ba(y_train, scores_cv)

        Q_aug, y_aug = grassmann_smote(Q_train, y_train.tolist(), minority_label=0, k=5, seed=RANDOM_STATE + fold)
        K_aug = build_kernel_matrix(Q_aug)

        clf = SVC(kernel="precomputed", C=C_best, class_weight="balanced")
        clf.fit(K_aug, y_aug)

        K_test = build_kernel_matrix(Q_test, Q_aug)
        pred = (clf.decision_function(K_test) > tau).astype(int)

        ab_rows.append({"variant": "Grassmann-SMOTE oversampling", "fold": fold, "balanced_acc": balanced_accuracy_score(y_test, pred), "f1": f1_score(y_test, pred, pos_label=1)})

    ablation = pd.DataFrame(ab_rows)
    ablation.to_csv("ablation_fold_metrics_v2.csv", index=False)
    ablation.groupby("variant")[["balanced_acc", "f1"]].agg(["mean", "std"]).to_csv("ablation_summary_v2.csv")

    # Figures
    make_pipeline_fig("fig1_pipeline_v2.png")

    summary_tbl = fold_metrics.groupby("model")[["balanced_acc", "f1"]].agg(["mean", "std"])
    make_metric_bar(summary_tbl, "balanced_acc", "fig2_balanced_accuracy_v3.png", y_start=0.70, y_end=0.83)
    make_metric_bar(summary_tbl, "f1", "fig3_f1_v3.png", y_start=0.78, y_end=0.94)

    cm = confusion_matrix(np.array(all_true), np.array(all_pred), labels=[0, 1])
    make_confusion_matrix(cm, "fig4_confusion_matrix_v2.png")

    print("Wrote: fold_metrics_main_v2.csv, summary_metrics_main_v2.csv, ablation_*.csv, fig*.png")


if __name__ == "__main__":
    main()
