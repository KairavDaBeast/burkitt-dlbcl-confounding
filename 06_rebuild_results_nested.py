"""Stage 06 nested rerun: quantile normalization fitted inside each CV fold (draft Limitation 8 fix).

Differences from 06_rebuild_results.py (joint normalization over all 764 samples):
- The reference distribution is the mean sorted profile of the TRAINING fold only.
  Held-out samples are mapped onto that reference, the same rule Stage 05 uses for external cohorts.
- Unchanged: gene filter (drop genes missing in any sample; label-free), log2(TPM+1), top-500 variance
  (label-free), StandardScaler, L2 logistic regression (max_iter=1000), 5-fold stratified shuffle
  random_state=42 (same folds), 200 label shuffles with folds fixed.

Outputs carry the suffix _nested, so the joint-normalization results are not overwritten.
Run: python 06_rebuild_results_nested.py   (from this folder; reads data/full/)
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA = "data/full/"
OUT = ""
N_TOP, N_PERM = 500, 200

# %% Step 1. Load and split (same as 06_rebuild_results.py)
# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(DATA + "gdc_tpm_all.pkl")
full = full[full.index.isin(pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0]).index)]  # Stage 09 switch (2026-11-20): protein-coding genes only
proj = pd.read_csv(DATA + "samples_all.csv", index_col="sample")["project"].reindex(full.columns)
bl_cols = full.columns[proj == "CGCI-BLGSP"]
nc_cols = full.columns[proj == "NCICCR-DLBCL"]
train_raw = full[list(bl_cols) + list(nc_cols)].dropna(axis=0, how="any")
genes = train_raw.index
cols = train_raw.columns
M = np.log2(train_raw.values.astype(float) + 1)          # genes x samples, log2, not yet normalized
bl_set = set(bl_cols)
y = np.array(["BL" if c in bl_set else "DLBCL" for c in cols], dtype=object)
print(f"Genes {len(genes)} | BL {(y == 'BL').sum()} | DLBCL {(y == 'DLBCL').sum()}")


# %% Step 2. Quantile-normalization helpers
def reference_from(M_train):
    """Mean of the sorted per-sample profiles, computed from training samples only."""
    return np.sort(M_train, axis=0).mean(axis=1)


def map_to_reference(M_in, ref):
    """Give each sample the reference values at its own ranks (same rule as Stage 02 and Stage 05)."""
    ranks = pd.DataFrame(M_in).rank(method="average", axis=0).values
    pos = np.arange(1, len(ref) + 1)
    out = np.empty(M_in.shape, dtype=float)
    for c in range(M_in.shape[1]):
        out[:, c] = np.interp(ranks[:, c], pos, ref)
    return out


# %% Step 3. Cross-validation with normalization refit inside each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(np.zeros(len(y)), y))             # same folds as 06_rebuild_results.py
fold_data = []                                          # label-free pieces, reused by the permutation test
oof_pred = np.empty(len(y), dtype=object)
oof_dec = np.zeros(len(y))                              # decision value, > 0 = DLBCL
for tr, te in folds:
    ref = reference_from(M[:, tr])                      # reference from training fold only
    Xtr = map_to_reference(M[:, tr], ref).T             # samples x genes
    Xte = map_to_reference(M[:, te], ref).T             # held-out samples mapped onto training reference
    sel = np.argsort(-Xtr.var(axis=0))[:N_TOP]          # top-500 variance, label-free
    scaler = StandardScaler().fit(Xtr[:, sel])
    Xtr_s, Xte_s = scaler.transform(Xtr[:, sel]), scaler.transform(Xte[:, sel])
    fold_data.append((tr, te, Xtr_s, Xte_s))
    clf = LogisticRegression(max_iter=1000, penalty="l2").fit(Xtr_s, y[tr])
    oof_pred[te] = clf.predict(Xte_s)
    oof_dec[te] = clf.decision_function(Xte_s)

acc = accuracy_score(y, oof_pred)
bal = balanced_accuracy_score(y, oof_pred)
rep = classification_report(y, oof_pred, digits=4, output_dict=True)
cm = confusion_matrix(y, oof_pred, labels=["BL", "DLBCL"])
print(f"\nNested CV accuracy {acc:.4%} | balanced {bal:.4%}")
print(pd.DataFrame(rep).T.round(4).to_string())
print("Confusion (rows true BL, DLBCL; cols predicted BL, DLBCL):")
print(cm)


# %% Step 4. Permutation test (folds fixed; per-fold data already built above)
def fold_mean_acc(labels):
    accs = []
    for tr, te, Xtr_s, Xte_s in fold_data:
        clf = LogisticRegression(max_iter=1000, penalty="l2").fit(Xtr_s, labels[tr])
        accs.append(accuracy_score(labels[te], clf.predict(Xte_s)))
    return float(np.mean(accs))


observed = fold_mean_acc(y)
rng = np.random.default_rng(42)
null = np.array([fold_mean_acc(rng.permutation(y)) for _ in range(N_PERM)])
p_value = (np.sum(null >= observed) + 1) / (N_PERM + 1)
print(f"\nObserved fold-mean accuracy {observed:.4%}")
print(f"Permutation null ({N_PERM} shuffles): mean {null.mean():.4%}, sd {null.std(ddof=1):.4%}, max {null.max():.4%}")
print(f"p = {p_value:.4f}  (floor with {N_PERM} shuffles = {1 / (N_PERM + 1):.4f})")

# %% Step 5. Save (suffix _nested) and compare with the joint-normalization run
pd.DataFrame(rep).T.to_csv(OUT + "rebuild_performance_nested.csv")
pd.DataFrame(cm, index=["true_BL", "true_DLBCL"], columns=["pred_BL", "pred_DLBCL"]).to_csv(OUT + "rebuild_confusion_nested.csv")
pd.DataFrame({"null_fold_mean_accuracy": null}).to_csv(OUT + "rebuild_permutation_null_nested.csv", index_label="shuffle")
pd.Series({"cv_accuracy": acc, "cv_balanced": bal, "observed_fold_mean": observed, "null_mean": null.mean(),
           "null_sd": null.std(ddof=1), "null_max": null.max(), "p_value": p_value, "n_perm": N_PERM,
           "n_bl": int((y == "BL").sum()), "n_dlbcl": int((y == "DLBCL").sum()), "n_genes": len(genes)}) \
    .to_csv(OUT + "rebuild_summary_nested.csv", header=["value"])
pd.DataFrame({"sample": cols, "label": y, "oof_decision": oof_dec}).to_csv(OUT + "oof_scores_nested.csv", index=False)

joint = pd.read_csv(OUT + "rebuild_summary.csv", index_col=0)["value"]
print("\nComparison (joint normalization vs nested):")
for k in ["cv_accuracy", "cv_balanced", "observed_fold_mean", "null_mean", "null_sd", "null_max", "p_value"]:
    print(f"  {k:20s} joint {joint[k]:.6f}   nested {float(pd.Series({'cv_accuracy': acc, 'cv_balanced': bal, 'observed_fold_mean': observed, 'null_mean': null.mean(), 'null_sd': null.std(ddof=1), 'null_max': null.max(), 'p_value': p_value})[k]):.6f}")
print("saved *_nested.csv and oof_scores_nested.csv")
