# %% [markdown]
# # HBB and MALAT1 check
#
# Two of the top-20 model genes need checking before interpretation:
#   - HBB (hemoglobin beta): blood-derived transcripts can reflect how much blood a sample contains,
#     not tumour biology.
#   - MALAT1: a nuclear-retained lncRNA, sensitive to RNA-preparation differences.
#
# Checks:
#   A. Do hemoglobin genes or MALAT1 differ between BL and DLBCL?
#   B. Is a hemoglobin "blood score" associated with the model's score, and does it differ by group?
#   C. Does the classifier still work when hemoglobin genes and/or MALAT1 are removed from the pool?
#      (same 5-fold CV, same top-500 variance rule)

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data/"
OUT = ""
N_TOP = 500

# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(DATA + "full/gdc_tpm_all.pkl")
full = full[full.index.isin(pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0]).index)]  # Stage 09 switch (2026-11-20): protein-coding genes only
proj = pd.read_csv(DATA + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
bl_cols = full.columns[proj == "CGCI-BLGSP"]
nc_cols = full.columns[proj == "NCICCR-DLBCL"]

train_raw = full[list(bl_cols) + list(nc_cols)].dropna(axis=0, how="any")
genes = train_raw.index
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
X = norm.T.values
is_bl = y == "BL"

names = pd.read_csv(DATA + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"].reindex(genes)
HB_NAMES = ["HBB", "HBA1", "HBA2", "HBD", "HBM", "HBQ1", "HBG1", "HBG2", "HBE1", "HBZ"]
hb_ids = list(genes[names.isin(HB_NAMES).values])
malat_id = "ENSG00000251562.8"
has_malat = malat_id in genes   # MALAT1 is non-coding, so it is absent after the protein-coding switch
hbb_id = list(genes[(names == "HBB").values])[0]
print("Hemoglobin genes in the matrix:", [names[g] for g in hb_ids])


def gene_variance_score(X_fold, y_fold=None):
    return np.asarray(X_fold).var(axis=0)


def make_pipe():
    return Pipeline([
        ("select_top_variance", SelectKBest(score_func=gene_variance_score, k=N_TOP)),
        ("scale", StandardScaler()),
        ("classify", LogisticRegression(max_iter=1000, penalty="l2")),
    ])


# %% A. Group differences
rows = []
for gid in [hbb_id] + ([malat_id] if has_malat else []) + [g for g in hb_ids if g != hbb_id]:
    b, d = norm.loc[gid, norm.columns[is_bl]], norm.loc[gid, norm.columns[~is_bl]]
    t, p = stats.ttest_ind(d, b, equal_var=False)
    rows.append({"gene": names.get(gid, "MALAT1") if gid != malat_id else "MALAT1", "gene_id": gid,
                 "bl_mean": b.mean(), "dlbcl_mean": d.mean(), "dlbcl_minus_bl": d.mean() - b.mean(),
                 "welch_p": p, "variance_rank": int(norm.var(axis=1).rank(ascending=False)[gid])})
diffs = pd.DataFrame(rows)
print("\nA. Group differences (normalized log2):")
print(diffs.round(4).to_string(index=False))

# %% B. Blood score vs model score
hb_z = norm.loc[hb_ids].T
hb_z = (hb_z - hb_z.mean()) / hb_z.std()
blood_score = hb_z.mean(axis=1)            # one value per sample
b, d = blood_score[is_bl], blood_score[~is_bl]
t, p = stats.ttest_ind(d, b, equal_var=False)
print(f"\nB. Blood score (mean z of {len(hb_ids)} hemoglobin genes): BL {b.mean():.3f} vs DLBCL {d.mean():.3f}, "
      f"Welch p = {p:.2g}")

full_model = make_pipe().fit(X, y)
score = pd.Series(full_model.decision_function(X), index=norm.columns)   # > 0 = DLBCL
rho_all = stats.spearmanr(blood_score, score).statistic
rho_bl = stats.spearmanr(blood_score[is_bl], score[is_bl]).statistic
rho_dl = stats.spearmanr(blood_score[~is_bl], score[~is_bl]).statistic
print(f"   Spearman(blood score, model score): all {rho_all:.3f} | within BL {rho_bl:.3f} | within DLBCL {rho_dl:.3f}")

# %% C. Ablation: rebuild the classifier without these genes in the candidate pool
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gene_index = pd.Index(genes)
ablations = {
    "none (rebuild as published)": [],
    "HBB only": [hbb_id],
    "all hemoglobin genes": hb_ids,
    "MALAT1 only": [malat_id],
    "hemoglobin + MALAT1": hb_ids + [malat_id],
}
if not has_malat:
    ablations = {k: v for k, v in ablations.items() if "MALAT1" not in k}
abl_rows = []
for label, drop in ablations.items():
    keep = ~gene_index.isin(drop)
    Xs = X[:, keep]
    pred = cross_val_predict(make_pipe(), Xs, y, cv=cv)
    sel_genes = gene_index[keep]
    full_fit = make_pipe().fit(Xs, y)
    sel = sel_genes[full_fit.named_steps["select_top_variance"].get_support()]
    coef = pd.Series(full_fit.named_steps["classify"].coef_[0], index=sel).abs().sort_values(ascending=False)
    top20 = coef.index[:20]
    hb_in_top20 = [names.get(g, "MALAT1") if g != malat_id else "MALAT1" for g in top20 if g in set(hb_ids + [malat_id])]
    abl_rows.append({"ablation": label, "genes_in_pool": int(keep.sum()),
                     "cv_accuracy": round(accuracy_score(y, pred), 4),
                     "cv_balanced": round(balanced_accuracy_score(y, pred), 4),
                     "hb_or_malat1_in_top20": ", ".join(hb_in_top20) if hb_in_top20 else "none"})
abl = pd.DataFrame(abl_rows)
print("\nC. Ablation (same CV, same top-500 rule):")
print(abl.to_string(index=False))

# %% Save
diffs.to_csv(OUT + "check_hbb_malat1_group_diffs.csv", index=False)
abl.to_csv(OUT + "check_hbb_malat1_ablation.csv", index=False)
pd.Series({"blood_score_bl": b.mean(), "blood_score_dlbcl": d.mean(), "blood_score_p": p,
           "rho_all": rho_all, "rho_within_bl": rho_bl, "rho_within_dlbcl": rho_dl}) \
    .to_csv(OUT + "check_hbb_blood_score.csv", header=["value"])
print("\nsaved check_hbb_malat1_group_diffs.csv, check_hbb_malat1_ablation.csv, check_hbb_blood_score.csv")
