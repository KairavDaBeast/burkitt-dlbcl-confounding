# %% [markdown]
# # Results on the local rebuild (one pipeline for every number in the paper)
#
# Same preprocessing and classifier as 05_batch_confound_check_local.py:
#   Stage 01/02: all genes, log2(TPM+1), quantile normalization, drop incomplete genes
#   Stage 03:    top-500 highest-variance genes chosen inside each fold, StandardScaler, L2 logistic regression
#                5-fold stratified CV, shuffled, random_state=42
#
# Outputs
#   - cross-validated accuracy, balanced accuracy, per-class precision/recall, confusion matrix
#   - permutation test: 200 label shuffles, p = (count >= observed + 1) / (200 + 1)
#   - MYC starter-gene comparison in the full normalized matrix, Welch t-test, BH-FDR across genes
#
# Permutation shortcut (documented in the paper's Methods): the variance filter ignores labels, so the
# selected genes for each fold are computed once and reused in every shuffle. The folds are fixed
# to the real labels. Each shuffle refits the scaler and logistic regression on the same folds.

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data/"
OUT = ""
N_TOP, N_PERM = 500, 200

# %% Step 1. Load and split by project (same as 05_batch_confound_check_local.py)
# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(DATA + "full/gdc_tpm_all.pkl")
full = full[full.index.isin(pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0]).index)]  # Stage 09 switch (2026-11-20): protein-coding genes only
proj = pd.read_csv(DATA + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
bl_cols = full.columns[proj == "CGCI-BLGSP"]
nc_cols = full.columns[proj == "NCICCR-DLBCL"]
print(f"BL {len(bl_cols)} | NCICCR-DLBCL {len(nc_cols)} (TCGA and CTSP are not part of the classifier)")

# %% Step 2. Stage 02 preprocessing
train_raw = full[list(bl_cols) + list(nc_cols)].dropna(axis=0, how="any")
genes = train_raw.index
print(f"Genes after dropping incomplete rows: {len(genes)}")
log_tr = np.log2(train_raw + 1)


def quantile_normalize(df):
    reference = np.sort(df.values, axis=0).mean(axis=1)
    ranks = df.rank(method="average", axis=0).values
    out = np.empty_like(df.values, dtype=float)
    pos = np.arange(1, len(reference) + 1)
    for c in range(df.shape[1]):
        out[:, c] = np.interp(ranks[:, c], pos, reference)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


norm = quantile_normalize(log_tr)
bl_set = set(bl_cols)
y = np.array(["BL" if c in bl_set else "DLBCL" for c in norm.columns], dtype=object)
X = norm.T.values                       # samples x genes
print(f"Samples: {pd.Series(y).value_counts().to_dict()}")


def gene_variance_score(X_fold, y_fold=None):
    return np.asarray(X_fold).var(axis=0)   # unsupervised: labels are not used


def make_pipe():
    return Pipeline([
        ("select_top_variance", SelectKBest(score_func=gene_variance_score, k=N_TOP)),
        ("scale", StandardScaler()),
        ("classify", LogisticRegression(max_iter=1000, penalty="l2")),
    ])


# %% Step 3. Cross-validated performance (Table 2 facts)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_pred = cross_val_predict(make_pipe(), X, y, cv=cv)
acc = accuracy_score(y, cv_pred)
bal = balanced_accuracy_score(y, cv_pred)
rep = classification_report(y, cv_pred, digits=4, output_dict=True)
cm = confusion_matrix(y, cv_pred, labels=["BL", "DLBCL"])
print(f"\nCV accuracy {acc:.4%} | balanced {bal:.4%}")
print(pd.DataFrame(rep).T.round(4).to_string())
print("Confusion matrix (rows = true BL, DLBCL; cols = predicted BL, DLBCL):")
print(cm)

# %% Step 4. Permutation test (Figure 4 facts)
folds = list(cv.split(X, y))
sel_per_fold = [SelectKBest(score_func=gene_variance_score, k=N_TOP).fit(X[tr], y[tr]).get_support(indices=True)
                for tr, _ in folds]


def fold_mean_accuracy(labels):
    scores = []
    for (tr, te), sel in zip(folds, sel_per_fold):
        scaler = StandardScaler().fit(X[tr][:, sel])
        clf = LogisticRegression(max_iter=1000, penalty="l2").fit(scaler.transform(X[tr][:, sel]), labels[tr])
        scores.append(accuracy_score(labels[te], clf.predict(scaler.transform(X[te][:, sel]))))
    return float(np.mean(scores))


real = fold_mean_accuracy(y)
check = cross_val_score(make_pipe(), X, y, cv=cv).mean()
print(f"\nObserved fold-mean accuracy {real:.4%} (sklearn cross_val_score check: {check:.4%})")

rng = np.random.default_rng(42)
null = np.array([fold_mean_accuracy(rng.permutation(y)) for _ in range(N_PERM)])
p_value = (np.sum(null >= real) + 1) / (N_PERM + 1)
print(f"Permutation null ({N_PERM} shuffles): mean {null.mean():.4%}, sd {null.std(ddof=1):.4%}, "
      f"max {null.max():.4%}")
print(f"p = {p_value:.4f}  (floor with {N_PERM} shuffles = {1 / (N_PERM + 1):.4f})")

# %% Step 5. MYC starter genes in the full normalized matrix (Section 4.5 facts)
names = pd.read_csv(DATA + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"]
name_map = names.reindex(genes)
var_rank = norm.var(axis=1).rank(ascending=False)   # 1 = most variable of all genes
starters = ["MYC", "ODC1", "CAD", "LDHA", "NCL", "NPM1", "CCND2", "CDK4", "EIF4E", "TFRC"]
bl_in = norm.columns[norm.columns.isin(bl_cols)]
dl_in = norm.columns[~norm.columns.isin(bl_cols)]
rows = []
for g in starters:
    ids = name_map.index[name_map == g]
    if len(ids) == 0:
        rows.append({"gene": g, "note": "not resolved by name lookup"})
        continue
    gid = ids[0]
    b, d = norm.loc[gid, bl_in], norm.loc[gid, dl_in]
    t, p = stats.ttest_ind(d, b, equal_var=False)
    rows.append({"gene": g, "gene_id": gid, "bl_mean": b.mean(), "dlbcl_mean": d.mean(),
                 "diff_dlbcl_minus_bl": d.mean() - b.mean(), "welch_p": p,
                 "variance_rank": int(var_rank[gid]),
                 "in_top500_variance": bool(var_rank[gid] <= N_TOP)})
myc = pd.DataFrame(rows)
ok = myc["welch_p"].notna()
myc.loc[ok, "bh_fdr"] = stats.false_discovery_control(myc.loc[ok, "welch_p"].values)
print("\nMYC starter genes (normalized log2, full matrix):")
print(myc.round(4).to_string(index=False))

# %% Step 6. Save
pd.DataFrame(rep).T.to_csv(OUT + "rebuild_performance.csv")
pd.DataFrame(cm, index=["true_BL", "true_DLBCL"], columns=["pred_BL", "pred_DLBCL"]).to_csv(OUT + "rebuild_confusion.csv")
pd.DataFrame({"null_fold_mean_accuracy": null}).to_csv(OUT + "rebuild_permutation_null.csv", index_label="shuffle")
myc.to_csv(OUT + "rebuild_myc_starter_genes.csv", index=False)
pd.Series({"cv_accuracy": acc, "cv_balanced": bal, "observed_fold_mean": real, "null_mean": null.mean(),
           "null_sd": null.std(ddof=1), "null_max": null.max(), "p_value": p_value, "n_perm": N_PERM,
           "n_bl": int((y == "BL").sum()), "n_dlbcl": int((y == "DLBCL").sum()), "n_genes": len(genes)}) \
    .to_csv(OUT + "rebuild_summary.csv", header=["value"])
print("\nsaved rebuild_performance.csv, rebuild_confusion.csv, rebuild_permutation_null.csv, "
      "rebuild_myc_starter_genes.csv, rebuild_summary.csv")
