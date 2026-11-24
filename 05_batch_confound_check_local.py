# %% [markdown]
# # Stage 05 (local rerun): batch / confounding check
#
# The Kaggle run's cleaned matrix and saved model are not on this PC, so this rebuilds them
# from the same GDC files with the same steps as the notebooks:
#   Stage 01  tpm_unstranded, all genes, one profile per patient
#   Stage 02  drop genes missing in any training sample, log2(TPM+1), quantile normalization
#   Stage 03  top-500 highest-variance genes (chosen inside each CV fold), StandardScaler,
#             L2 logistic regression
#   Stage 05  new cohorts: align to training genes (missing -> 0), log2, quantile-normalize onto
#             the training reference distribution, then predict with the model that never saw them
#
# Test 1: external validation. TCGA-DLBC and CTSP-DLBCL1 are DLBCL and never seen by the model.
# Test 2: lab-signature probe. NCICCR-DLBCL vs TCGA-DLBC (same disease, different labs).
#         Chance is 50% (balanced accuracy); the majority-class baseline is 75% (144 vs 48).
#
# Run: python 05_download_full_tpm.py (once), then python 05_batch_confound_check_local.py

# %% Step 1. Load the full-gene TPM table and split it by project.
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA = "data/full/"
N_TOP = 500
rng = np.random.default_rng(0)

# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(DATA + "gdc_tpm_all.pkl")
full = full[full.index.isin(pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0]).index)]  # Stage 09 switch (2026-11-20): protein-coding genes only
proj = pd.read_csv(DATA + "samples_all.csv", index_col="sample")["project"].reindex(full.columns)
bl_cols = full.columns[proj == "CGCI-BLGSP"]
nc_cols = full.columns[proj == "NCICCR-DLBCL"]
tc_cols = full.columns[proj == "TCGA-DLBC"]
ct_cols = full.columns[proj == "CTSP-DLBCL1"]
print(f"BL (CGCI-BLGSP) {len(bl_cols)} | NCICCR-DLBCL {len(nc_cols)} | TCGA-DLBC {len(tc_cols)} | CTSP-DLBCL1 {len(ct_cols)}")


def gene_variance_score(X, y=None):
    """Unsupervised: variance of each gene in the training fold (labels not used)."""
    return np.asarray(X).var(axis=0)


def make_pipe(k=N_TOP):
    """Same structure as the Stage 03 pipeline."""
    return Pipeline([
        ("select_top_variance", SelectKBest(score_func=gene_variance_score, k=k)),
        ("scale", StandardScaler()),
        ("classify", LogisticRegression(max_iter=1000, penalty="l2")),
    ])


# %% Step 2. Stage 02: drop incomplete genes, log2, quantile-normalize the training cohort.
train_raw = full[list(bl_cols) + list(nc_cols)].dropna(axis=0, how="any")
genes = train_raw.index
print(f"Training genes after dropping incomplete rows: {len(genes)}")
log_tr = np.log2(train_raw + 1)


def quantile_normalize(df):
    """Force every sample onto the mean sorted distribution (same as Stage 02)."""
    reference = np.sort(df.values, axis=0).mean(axis=1)
    ranks = df.rank(method="average", axis=0).values
    out = np.empty_like(df.values, dtype=float)
    pos = np.arange(1, len(reference) + 1)
    for c in range(df.shape[1]):
        out[:, c] = np.interp(ranks[:, c], pos, reference)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


norm = quantile_normalize(log_tr)
ref_sorted = np.sort(norm.iloc[:, 0].values)      # the reference new cohorts are matched to (Stage 05)
ref_pos = np.arange(1, len(ref_sorted) + 1)
y = pd.Series(np.where(norm.columns.isin(bl_cols), "BL", "DLBCL"), index=norm.columns)
y_arr = np.array(y.tolist(), dtype=object)   # plain object array: pandas 3 string dtype breaks sklearn indexing
X = norm.T.values                                  # samples x genes
print(y.value_counts().to_dict())


def preprocess_like_training(raw):
    """Stage 05 step: align to training genes (missing -> 0), log2, quantile onto training reference."""
    aligned = raw.reindex(genes).fillna(0.0)
    logged = np.log2(aligned + 1)
    ranks = logged.rank(method="average", axis=0).values
    out = np.empty_like(ranks, dtype=float)
    for c in range(ranks.shape[1]):
        out[:, c] = np.interp(ranks[:, c], ref_pos, ref_sorted)
    return pd.DataFrame(out, index=genes, columns=logged.columns)


# %% Step 3. Stage 03: cross-validation sanity check against the documented Kaggle result, then fit on all.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_pred = cross_val_predict(make_pipe(), X, y_arr, cv=cv)
print(f"\nCV accuracy {accuracy_score(y_arr, cv_pred):.2%}, balanced {balanced_accuracy_score(y_arr, cv_pred):.2%}"
      "   (documented Stage 03 Kaggle run: 99.87% / 99.82%)")

model = make_pipe().fit(X, y_arr)
sel = model.named_steps["select_top_variance"].get_support()
sel_genes = genes[sel]
coef = pd.Series(model.named_steps["classify"].coef_[0], index=sel_genes)   # positive -> DLBCL
top20 = coef.reindex(coef.abs().sort_values(ascending=False).index[:20])
# Gene names come from the protein-coding table made by 07_download_gdc_tpm.py. Non-coding genes have no name here.
names = pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"]
top20_named = pd.DataFrame({"coef": top20, "name": names.reindex(top20.index).fillna("(non-coding, name not saved)")})
print("Top-20 coefficients (positive -> DLBCL):")
print(top20_named.round(4).to_string())

# %% Step 4. Test 1: external validation on TCGA-DLBC and CTSP-DLBCL1.
ext_rows = []
ext_samples = []   # per-sample P(DLBCL) for Figure 7
for name, cols in [("TCGA-DLBC", tc_cols), ("CTSP-DLBCL1", ct_cols)]:
    raw = full[cols]
    ready = preprocess_like_training(raw)
    n_absent_train = int((~genes.isin(raw.dropna(how="any").index)).sum())
    n_absent_top = int(raw.reindex(sel_genes).isna().any(axis=1).sum())
    Xn = ready.T.values
    pred = model.predict(Xn)
    p_dlbcl = model.predict_proba(Xn)[:, list(model.classes_).index("DLBCL")]
    frac = float((pred == "DLBCL").mean())
    print(f"\n{name}: {int((pred == 'DLBCL').sum())}/{len(pred)} called DLBCL ({frac:.1%}); "
          f"mean P(DLBCL) {p_dlbcl.mean():.1%}, median {np.median(p_dlbcl):.1%}")
    print(f"   training genes absent or incomplete in this cohort: {n_absent_train} (filled with 0)")
    print(f"   of the {len(sel_genes)} model genes, absent or incomplete here: {n_absent_top}")
    ext_samples.append(pd.DataFrame({"cohort": name, "sample": ready.columns, "p_dlbcl": p_dlbcl, "called": pred}))
    ext_rows.append({"cohort": name, "n": len(pred), "called_DLBCL": int((pred == "DLBCL").sum()),
                     "fraction": round(frac, 4), "mean_P_DLBCL": round(float(p_dlbcl.mean()), 4),
                     "median_P_DLBCL": round(float(np.median(p_dlbcl)), 4),
                     "model_genes_missing": n_absent_top})

# %% Step 5. Test 2: lab-signature probe, NCICCR-DLBCL vs TCGA-DLBC.
# Stage 05 used the first 144 NCICCR columns. Here 5 random draws of 144 are used, so the result
# does not depend on column order. Balanced accuracy is the number to read (chance = 50%).
tcga_ready = preprocess_like_training(full[tc_cols])
nc_norm = norm[nc_cols]
n_sub = 3 * tcga_ready.shape[1]
probe = []
for seed in range(5):
    sub = nc_norm.iloc[:, np.random.default_rng(seed).choice(nc_norm.shape[1], n_sub, replace=False)]
    Xp = pd.concat([sub, tcga_ready], axis=1).T.values
    yp = np.array(["NCICCR"] * sub.shape[1] + ["TCGA"] * tcga_ready.shape[1])
    pp = cross_val_predict(make_pipe(), Xp, yp, cv=StratifiedKFold(5, shuffle=True, random_state=42))
    probe.append((accuracy_score(yp, pp), balanced_accuracy_score(yp, pp)))
probe = np.array(probe)
print(f"\nLab-signature probe (NCICCR {n_sub} vs TCGA {tcga_ready.shape[1]}, 5 draws):")
print(f"   accuracy {probe[:, 0].mean():.2%} +/- {probe[:, 0].std():.2%}  "
      f"(majority baseline {n_sub / (n_sub + tcga_ready.shape[1]):.0%})")
print(f"   balanced accuracy {probe[:, 1].mean():.2%} +/- {probe[:, 1].std():.2%}  (chance 50%)")

# %% Step 6. Save results for the paper.
pd.DataFrame(ext_rows).to_csv("stage05_external_validation.csv", index=False)
pd.concat(ext_samples).to_csv("stage05_per_sample.csv", index=False)
pd.DataFrame(probe, columns=["accuracy", "balanced"]).to_csv("stage05_probe_draws.csv", index_label="draw")
top20.round(4).rename("coef").to_csv("stage05_top20_coefficients.csv", index_label="gene_id")
top20_named.assign(gene_name=top20_named["name"], gene_type="protein_coding")[["coef", "gene_name", "gene_type"]].round(4).to_csv("stage05_top20_named.csv", index_label="gene_id")
pd.Series({"cv_accuracy": accuracy_score(y_arr, cv_pred), "cv_balanced": balanced_accuracy_score(y_arr, cv_pred),
           "probe_accuracy_mean": probe[:, 0].mean(), "probe_balanced_mean": probe[:, 1].mean(),
           "probe_balanced_sd": probe[:, 1].std(), "n_train_genes": len(genes)}) \
    .to_csv("stage05_summary.csv", header=["value"])
print("\nsaved stage05_external_validation.csv, stage05_top20_coefficients.csv, stage05_summary.csv")
