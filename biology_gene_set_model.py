"""Biology-restricted model and cross-platform transfer test.

Gene set: MSigDB Hallmark MYC Targets V1 + V2 (Enrichr library MSigDB_Hallmark_2020; a local copy is saved
next to this script on first run so the gene list is fixed).

Part A. Nested 5-fold CV (same folds and per-fold normalization as 06_rebuild_results_nested.py) with the
        feature pool restricted to the MYC target genes (all of them; no variance filter).
        Reference: the usual top-500-variance model on all genes.
        Control: 100 random protein-coding gene sets of the same size, matched to the MYC genes on mean expression.
Part B. Lab probe (NCICCR-DLBCL vs TCGA-DLBC, same disease, two labs) with the MYC genes only,
        the top-500 rule, and 20 matched random sets. High balanced accuracy = the gene set carries lab signal.
Part C. Transfer test. Train on all 764 RNA-seq patients (BL vs DLBCL), predict GSE4475 (one lab, one array
        platform, so no lab confound), and score with AUC. Each dataset is z-scored per gene on its own, so
        platform scale cancels; AUC ignores the constant shift this leaves. Scored overall and in adults only
        (age >= 18), which also removes most of the age difference between BL and DLBCL.
        Models: MYC genes, top-500 variance genes, 100 matched random sets, and a lab-only model
        (trained NCICCR vs TCGA) as a negative control.

Run: python biology_gene_set_model.py   (from the project folder)
"""
import gzip
import os
import urllib.request
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = "data/"
DATA = BASE + "full/"
OUT = ""
N_TOP = 500
N_RAND_CV, N_RAND_PROBE, N_RAND_TRANSFER, N_BOOT = 100, 20, 100, 1000
rng = np.random.default_rng(11)

# %% Step 0. Gene sets (fixed local copy)
GS_FILE = OUT + "hallmark_myc_targets_MSigDB_Hallmark_2020.txt"
if not os.path.exists(GS_FILE):
    url = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=MSigDB_Hallmark_2020"
    txt = urllib.request.urlopen(url, timeout=120).read().decode()
    keep = [l for l in txt.splitlines() if l.startswith("Myc Targets V1") or l.startswith("Myc Targets V2")]
    with open(GS_FILE, "w") as f:
        f.write("\n".join(keep) + "\n")
sets = {}
with open(GS_FILE) as f:
    for line in f:
        parts = [p.split(",")[0] for p in line.rstrip("\n").split("\t") if p]
        if parts:
            sets[parts[0]] = set(parts[1:])
MYC = sets["Myc Targets V1"] | sets["Myc Targets V2"]
print({k: len(v) for k, v in sets.items()}, "| union", len(MYC))


def reference_from(M_train):
    return np.sort(M_train, axis=0).mean(axis=1)


def map_to_reference(M_in, ref):
    ranks = pd.DataFrame(M_in).rank(method="average", axis=0).values
    pos = np.arange(1, len(ref) + 1)
    out = np.empty(M_in.shape, dtype=float)
    for c in range(M_in.shape[1]):
        out[:, c] = np.interp(ranks[:, c], pos, ref)
    return out


def matched_random(pool_idx, target_idx, means, rng, n_bins=10):
    """Random genes from pool_idx with the same mean-expression decile counts as target_idx."""
    edges = np.quantile(means[np.r_[pool_idx, target_idx]], np.linspace(0, 1, n_bins + 1))
    bin_of = lambda v: np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_bins - 1)
    pool_bin, tgt_bin = bin_of(means[pool_idx]), bin_of(means[target_idx])
    out = []
    for b, n in zip(*np.unique(tgt_bin, return_counts=True)):
        out.extend(rng.choice(pool_idx[pool_bin == b], size=n, replace=False))
    return np.array(out)


def fit_dec(Xtr, ytr, Xte, idx, var_top=False):
    A, B = Xtr[:, idx], Xte[:, idx]
    if var_top:
        sel = np.argsort(-A.var(axis=0))[:N_TOP]
        A, B = A[:, sel], B[:, sel]
    sc = StandardScaler().fit(A)
    clf = LogisticRegression(max_iter=1000, penalty="l2").fit(sc.transform(A), ytr)
    return clf.decision_function(sc.transform(B))       # > 0 = second class (DLBCL or TCGA)


# %% Part A. Nested CV restricted to MYC targets
# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(DATA + "gdc_tpm_all.pkl")
full = full[full.index.isin(pd.read_csv("data/gdc_tpm.csv.gz", index_col=0, usecols=[0]).index)]  # Stage 09 switch (2026-11-20): protein-coding genes only
proj = pd.read_csv(DATA + "samples_all.csv", index_col="sample")["project"].reindex(full.columns)
bl_cols = full.columns[proj == "CGCI-BLGSP"]
nc_cols = full.columns[proj == "NCICCR-DLBCL"]
tc_cols = full.columns[proj == "TCGA-DLBC"]
train_raw = full[list(bl_cols) + list(nc_cols)].dropna(axis=0, how="any")
genes = train_raw.index
M = np.log2(train_raw.values.astype(float) + 1)
bl_set = set(bl_cols)
y = np.array(["BL" if c in bl_set else "DLBCL" for c in train_raw.columns], dtype=object)

names = pd.read_csv(BASE + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"].reindex(genes)
pc_idx = np.where(names.notna().values)[0]                       # protein-coding genes
myc_idx = np.where(names.isin(MYC).values)[0]
gmean = M.mean(axis=1)
cand_idx = np.setdiff1d(pc_idx, myc_idx)
rand_sets = [matched_random(cand_idx, myc_idx, gmean, rng) for _ in range(N_RAND_CV)]
print(f"MYC target genes found in RNA-seq: {len(myc_idx)} of {len(MYC)}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(np.zeros(len(y)), y))
all_idx = np.arange(len(genes))
dec_top, dec_myc = np.zeros(len(y)), np.zeros(len(y))
dec_rand = np.zeros((N_RAND_CV, len(y)))
for tr, te in folds:
    ref = reference_from(M[:, tr])
    Xtr = map_to_reference(M[:, tr], ref).T
    Xte = map_to_reference(M[:, te], ref).T
    dec_top[te] = fit_dec(Xtr, y[tr], Xte, all_idx, var_top=True)
    dec_myc[te] = fit_dec(Xtr, y[tr], Xte, myc_idx)
    for d, idx in enumerate(rand_sets):
        dec_rand[d, te] = fit_dec(Xtr, y[tr], Xte, idx)
    del Xtr, Xte


def cv_metrics(d):
    pred = np.where(d > 0, "DLBCL", "BL")
    return accuracy_score(y, pred), balanced_accuracy_score(y, pred), roc_auc_score(y == "DLBCL", d)


a_top, a_myc = cv_metrics(dec_top), cv_metrics(dec_myc)
a_rand = np.array([cv_metrics(d) for d in dec_rand])
print("\nPart A. Nested CV, BL vs DLBCL (accuracy, balanced accuracy, AUC)")
print(f"  top-500 variance, all genes      {a_top[0]:.4f} {a_top[1]:.4f} {a_top[2]:.4f}")
print(f"  MYC targets ({len(myc_idx)} genes)        {a_myc[0]:.4f} {a_myc[1]:.4f} {a_myc[2]:.4f}")
print(f"  matched random sets ({N_RAND_CV}): balanced mean {a_rand[:, 1].mean():.4f}, "
      f"5th-95th pct {np.percentile(a_rand[:, 1], 5):.4f}-{np.percentile(a_rand[:, 1], 95):.4f}, "
      f"share >= MYC {np.mean(a_rand[:, 1] >= a_myc[1]):.2f}")

# %% Part B. Lab probe with each gene set (joint normalization, as in Stage 05)
ref_all = reference_from(M)
nc_pos = np.where(y == "DLBCL")[0]
NCn = map_to_reference(M[:, nc_pos], ref_all).T
tc_raw = full[tc_cols].reindex(genes).fillna(0.0)
TCn = map_to_reference(np.log2(tc_raw.values.astype(float) + 1), ref_all).T
del full


def probe(idx, var_top=False):
    bals = []
    for seed in range(5):
        pick = np.random.default_rng(seed).choice(len(nc_pos), 3 * TCn.shape[0], replace=False)
        Xp = np.vstack([NCn[pick], TCn])
        yp = np.array(["NCICCR"] * len(pick) + ["TCGA"] * TCn.shape[0], dtype=object)
        dp = np.zeros(len(yp))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xp, yp):
            dp[te] = fit_dec(Xp[tr], yp[tr], Xp[te], idx, var_top=var_top)
        bals.append(balanced_accuracy_score(yp, np.where(dp > 0, "TCGA", "NCICCR")))
    return float(np.mean(bals))


b_top = probe(all_idx, var_top=True)
b_myc = probe(myc_idx)
b_rand = np.array([probe(idx) for idx in rand_sets[:N_RAND_PROBE]])
print("\nPart B. Lab probe, NCICCR vs TCGA (balanced accuracy; chance 0.50)")
print(f"  top-500 variance {b_top:.4f} | MYC targets {b_myc:.4f} | matched random mean {b_rand.mean():.4f} "
      f"(min {b_rand.min():.4f})")

# %% Part C. Transfer to GSE4475 (same lab, array)
tpm = pd.read_csv(BASE + "gdc_tpm.csv.gz", index_col=0)
samp = pd.read_csv(BASE + "gdc_samples.csv", index_col="sample")["project"]
rna = np.log2(tpm.drop(columns="gene_name").groupby(tpm["gene_name"]).mean() + 1)
pr = samp.reindex(rna.columns)
bl_r = list(rna.columns[pr == "CGCI-BLGSP"])
dl_r = list(rna.columns[pr == "NCICCR-DLBCL"])
tc_r = list(rna.columns[pr == "TCGA-DLBC"])

with gzip.open(BASE + "geo/GSE4475_series_matrix.txt.gz", "rt") as f:
    lines = f.read().splitlines()


def field(key):
    l = next(l for l in lines if l.startswith("!Sample_characteristics") and key in l)
    return [d.strip('"').split(" : ", 1)[1].strip() for d in l.split("\t")[1:]]


morph = np.array(field("Morphologic.diagnosis"))
molec = np.array(field("Molecular.Diagnosis"))
age_a = pd.to_numeric(pd.Series(field('"Age : ')), errors="coerce").values
start = lines.index("!series_matrix_table_begin") + 1
end = lines.index("!series_matrix_table_end")
arr = pd.DataFrame([l.split("\t") for l in lines[start + 1:end]]).set_index(0)
arr.columns = [c.strip('"') for c in lines[start].split("\t")[1:]]
arr.index = arr.index.str.strip('"')
arr = arr.apply(pd.to_numeric, errors="coerce")
annot = pd.read_csv(BASE + "geo/GPL96.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"]).dropna()
annot = annot[~annot["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]
arr = arr.loc[arr.index.intersection(annot.index)]
arr["gene"] = annot.reindex(arr.index)
arr["mean"] = arr.drop(columns="gene").mean(axis=1)
arr = arr.sort_values("mean", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="mean")
print("\nGSE4475 molecular labels:", pd.Series(molec).value_counts().to_dict())

genes_c = rna.index.intersection(arr.index)
genes_c = genes_c[rna.loc[genes_c, bl_r + dl_r].mean(axis=1) >= 1]
print(f"Genes on both platforms (expressed): {len(genes_c)}")


def zscore(df):
    sd = df.std(axis=1).replace(0, 1)
    return df.sub(df.mean(axis=1), axis=0).div(sd, axis=0).fillna(0)


Rz = zscore(rna.loc[genes_c, bl_r + dl_r])
Az = zscore(arr.loc[genes_c])
yR = np.array(["BL"] * len(bl_r) + ["DLBCL"] * len(dl_r), dtype=object)
Rz_lab = zscore(rna.loc[genes_c, dl_r + tc_r])
yL = np.array(["NCICCR"] * len(dl_r) + ["TCGA"] * len(tc_r), dtype=object)

is_bl_m = np.isin(morph, ["BL", "atypical BL", "BL-leukemia"])
is_dl_m = morph == "DLBCL"
is_bl_mol = molec == "mBL"
is_dl_mol = molec == "non-mBL"
adult = age_a >= 18
groups = {
    "morphology, all": (is_bl_m, is_dl_m),
    "morphology, adults (age >= 18)": (is_bl_m & adult, is_dl_m & adult),
    "molecular, all": (is_bl_mol, is_dl_mol),
    "molecular, adults (age >= 18)": (is_bl_mol & adult, is_dl_mol & adult),
}
for g, (b, d) in groups.items():
    print(f"  {g}: BL {int(b.sum())} vs DLBCL {int(d.sum())}")


def train_apply(gene_list, Z=Rz, labels=yR):
    clf = LogisticRegression(max_iter=1000, penalty="l2").fit(Z.loc[gene_list].T.values, labels)
    return clf.decision_function(Az.loc[gene_list].T.values)   # > 0 = DLBCL (or TCGA for the lab model)


def auc_group(dec, b, d):
    s = np.r_[dec[d], dec[b]]
    t = np.r_[np.ones(d.sum()), np.zeros(b.sum())]               # positive class = DLBCL
    return roc_auc_score(t, s)


def boot_ci(dec, b, d):
    """95% interval: resample BL and DLBCL samples separately, N_BOOT times."""
    bi, di = np.where(b)[0], np.where(d)[0]
    out = []
    for _ in range(N_BOOT):
        rb, rd = rng.choice(bi, len(bi)), rng.choice(di, len(di))
        s = np.r_[dec[rd], dec[rb]]
        t = np.r_[np.ones(len(rd)), np.zeros(len(rb))]
        out.append(roc_auc_score(t, s))
    return np.percentile(out, [2.5, 97.5])


myc_c = [g for g in genes_c if g in MYC]
var_c = rna.loc[genes_c, bl_r + dl_r].var(axis=1)
top_c = list(var_c.sort_values(ascending=False).index[:N_TOP])
means_c = rna.loc[genes_c, bl_r + dl_r].mean(axis=1).values
gc_idx = np.arange(len(genes_c))
myc_pos = np.where(np.isin(genes_c, myc_c))[0]
rand_c = [list(genes_c[matched_random(np.setdiff1d(gc_idx, myc_pos), myc_pos, means_c, rng)])
          for _ in range(N_RAND_TRANSFER)]
print(f"MYC target genes on both platforms: {len(myc_c)}")

dec_c = {"MYC targets": train_apply(myc_c), "top-500 variance": train_apply(top_c)}
dec_lab = {"lab-only model, top-500 variance": train_apply(
               list(rna.loc[genes_c, dl_r + tc_r].var(axis=1).sort_values(ascending=False).index[:N_TOP]),
               Rz_lab, yL),
           "lab-only model, MYC targets": train_apply(myc_c, Rz_lab, yL)}
dec_rand_c = [train_apply(gl) for gl in rand_c]

rows = []
for g, (b, d) in groups.items():
    for name, dec in dec_c.items():
        lo, hi = boot_ci(dec, b, d)
        rows.append({"labels": g, "model": name, "auc": auc_group(dec, b, d), "ci_low": lo, "ci_high": hi})
    r = np.array([auc_group(dec, b, d) for dec in dec_rand_c])
    rows.append({"labels": g, "model": f"matched random sets (mean of {N_RAND_TRANSFER})", "auc": r.mean(),
                 "ci_low": np.percentile(r, 5), "ci_high": np.percentile(r, 95)})
    for name, dec in dec_lab.items():
        a = auc_group(dec, b, d)
        rows.append({"labels": g, "model": name + " (orientation-free)", "auc": max(a, 1 - a),
                     "ci_low": np.nan, "ci_high": np.nan})
transfer = pd.DataFrame(rows)
print("\nPart C. Transfer to GSE4475, AUC for BL vs DLBCL (0.5 = no information)")
print("  (random-set row shows 5th-95th percentile in the CI columns)")
print(transfer.round(3).to_string(index=False))

# %% Save
pd.DataFrame([
    {"part": "A_cv", "model": "top-500 variance, all genes", "accuracy": a_top[0], "balanced": a_top[1], "auc": a_top[2]},
    {"part": "A_cv", "model": f"MYC targets ({len(myc_idx)} genes)", "accuracy": a_myc[0], "balanced": a_myc[1], "auc": a_myc[2]},
    {"part": "A_cv", "model": f"matched random mean ({N_RAND_CV})", "accuracy": a_rand[:, 0].mean(),
     "balanced": a_rand[:, 1].mean(), "auc": a_rand[:, 2].mean()},
    {"part": "B_lab_probe", "model": "top-500 variance", "balanced": b_top},
    {"part": "B_lab_probe", "model": "MYC targets", "balanced": b_myc},
    {"part": "B_lab_probe", "model": f"matched random mean ({N_RAND_PROBE})", "balanced": b_rand.mean()},
]).to_csv(OUT + "biology_set_results.csv", index=False)
pd.DataFrame(a_rand, columns=["accuracy", "balanced", "auc"]).to_csv(OUT + "biology_set_random_cv.csv", index=False)
transfer.to_csv(OUT + "transfer_gse4475_results.csv", index=False)
print("\nsaved biology_set_results.csv, biology_set_random_cv.csv, transfer_gse4475_results.csv")
