"""Stage 09: main model rebuilt on protein-coding genes only.

Why: Stage 08b showed the lab probe (NCICCR vs TCGA) is 0.997 on non-coding genes but 0.82-0.85 on
protein-coding genes, and 399 of the original model's top-500 genes were non-coding.

Same pipeline as 06_rebuild_results_nested.py except the gene filter: the 19,962 protein-coding genes
(gene table from 07_download_gdc_tpm.py). Per-fold quantile normalization, top-500 variance (label-free),
StandardScaler, L2 logistic regression, same 5 folds (random_state=42), 200 label shuffles with folds fixed.
Also: external DLBCL cohorts (TCGA-DLBC, CTSP-DLBCL1) and the final model applied directly to GSE4475
(each dataset z-scored per gene; model genes missing from the array set to 0, the mean).

Run: python 09_protein_coding_model.py   (from the project folder; a few minutes)
"""
import gzip
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = "data/"
OUT = ""
N_TOP, N_PERM = 500, 200

# %% Step 1. Load, keep protein-coding genes complete in all training samples.
# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(BASE + "full/gdc_tpm_all.pkl")
proj = pd.read_csv(BASE + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
names = pd.read_csv(BASE + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"]
BL, NC, TC, CT = (list(full.columns[(proj == p).values])
                  for p in ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC", "CTSP-DLBCL1"])
train_raw = full.loc[full.index.isin(names.index), BL + NC].dropna(how="any")
genes = train_raw.index
M = np.log2(train_raw.values.astype(float) + 1)
y = np.array(["BL"] * len(BL) + ["DLBCL"] * len(NC), dtype=object)
print(f"Protein-coding genes {len(genes)} | BL {len(BL)} | DLBCL {len(NC)}")


def reference_from(M_train):
    return np.sort(M_train, axis=0).mean(axis=1)


def map_to_reference(M_in, ref):
    ranks = pd.DataFrame(M_in).rank(method="average", axis=0).values
    pos = np.arange(1, len(ref) + 1)
    return np.column_stack([np.interp(ranks[:, c], pos, ref) for c in range(M_in.shape[1])])


# %% Step 2. Nested CV (normalization refit inside each fold).
folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
fold_data, oof_pred, oof_dec = [], np.empty(len(y), dtype=object), np.zeros(len(y))
for tr, te in folds:
    ref = reference_from(M[:, tr])
    Xtr, Xte = map_to_reference(M[:, tr], ref).T, map_to_reference(M[:, te], ref).T
    sel = np.argsort(-Xtr.var(axis=0))[:N_TOP]
    sc = StandardScaler().fit(Xtr[:, sel])
    Xtr_s, Xte_s = sc.transform(Xtr[:, sel]), sc.transform(Xte[:, sel])
    fold_data.append((tr, te, Xtr_s, Xte_s))
    clf = LogisticRegression(max_iter=1000).fit(Xtr_s, y[tr])
    oof_pred[te], oof_dec[te] = clf.predict(Xte_s), clf.decision_function(Xte_s)
acc, bal = accuracy_score(y, oof_pred), balanced_accuracy_score(y, oof_pred)
cm = confusion_matrix(y, oof_pred, labels=["BL", "DLBCL"])
print(f"\nNested CV accuracy {acc:.4%} | balanced {bal:.4%}")
print("Confusion (rows true BL, DLBCL; cols predicted BL, DLBCL):\n", cm)

# %% Step 3. Permutation test (folds fixed).


def fold_mean_acc(labels):
    return float(np.mean([accuracy_score(labels[te], LogisticRegression(max_iter=1000).fit(a, labels[tr]).predict(b))
                          for tr, te, a, b in fold_data]))


observed = fold_mean_acc(y)
rng = np.random.default_rng(42)
null = np.array([fold_mean_acc(rng.permutation(y)) for _ in range(N_PERM)])
p_value = (np.sum(null >= observed) + 1) / (N_PERM + 1)
print(f"Permutation: observed {observed:.4%}, null mean {null.mean():.4%}, max {null.max():.4%}, p = {p_value:.4f}")

# %% Step 4. Final model on all 764, then external DLBCL cohorts.
ref_all = reference_from(M)
X_all = map_to_reference(M, ref_all).T
sel = np.argsort(-X_all.var(axis=0))[:N_TOP]
sel_genes = genes[sel]
sc = StandardScaler().fit(X_all[:, sel])
final = LogisticRegression(max_iter=1000).fit(sc.transform(X_all[:, sel]), y)
coef = pd.Series(final.coef_[0], index=sel_genes)                       # > 0 pushes toward DLBCL
ext = []
for name, cols in [("TCGA-DLBC", TC), ("CTSP-DLBCL1", CT)]:
    raw = full.loc[genes, cols].fillna(0.0)
    Xe = map_to_reference(np.log2(raw.values.astype(float) + 1), ref_all).T
    pred = final.predict(sc.transform(Xe[:, sel]))
    ext.append({"cohort": name, "n": len(cols), "called_DLBCL": int((pred == "DLBCL").sum())})
    print(f"{name}: {ext[-1]['called_DLBCL']}/{len(cols)} called DLBCL")
del full

# %% Step 5. Final model applied directly to GSE4475 (no retraining).
with gzip.open(BASE + "geo/GSE4475_series_matrix.txt.gz", "rt") as f:
    lines = f.read().splitlines()


def field(key):
    l = next(l for l in lines if l.startswith("!Sample_characteristics") and key in l)
    return [d.strip('"').split(" : ", 1)[1].strip() for d in l.split("\t")[1:]]


morph, molec = np.array(field("Morphologic.diagnosis")), np.array(field("Molecular.Diagnosis"))
start, end = lines.index("!series_matrix_table_begin") + 1, lines.index("!series_matrix_table_end")
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

sym = names.reindex(sel_genes).values
on_array = np.isin(sym, arr.index)
Z = np.zeros((arr.shape[1], N_TOP))
sub = arr.loc[sym[on_array]]
sd = sub.std(axis=1, ddof=1).replace(0, 1)
Z[:, on_array] = np.nan_to_num(sub.sub(sub.mean(axis=1), axis=0).div(sd, axis=0).values.T)
dec = final.decision_function(Z)                                         # > 0 = DLBCL


def auc(b, d):
    return roc_auc_score(np.r_[np.ones(d.sum()), np.zeros(b.sum())], np.r_[dec[d], dec[b]])


m_bl, m_dl = np.isin(morph, ["BL", "atypical BL", "BL-leukemia"]), morph == "DLBCL"
auc_m, auc_mol = auc(m_bl, m_dl), auc(molec == "mBL", molec == "non-mBL")
print(f"\nDirect transfer to GSE4475: {int(on_array.sum())}/{N_TOP} model genes on the array; "
      f"AUC morphology {auc_m:.3f}, molecular {auc_mol:.3f}")
adult = pd.to_numeric(pd.Series(field('"Age : ')), errors="coerce").values >= 18
boot_rng = np.random.default_rng(11)
t_rows = []
for g, (b, d) in {"morphology, all": (m_bl, m_dl), "morphology, adults (age >= 18)": (m_bl & adult, m_dl & adult),
                  "molecular, all": (molec == "mBL", molec == "non-mBL"),
                  "molecular, adults (age >= 18)": ((molec == "mBL") & adult, (molec == "non-mBL") & adult)}.items():
    bi, di = np.where(b)[0], np.where(d)[0]
    boots = []
    for _ in range(1000):                                                # resample BL and DLBCL separately
        rb, rd = boot_rng.choice(bi, len(bi)), boot_rng.choice(di, len(di))
        boots.append(roc_auc_score(np.r_[np.ones(len(rd)), np.zeros(len(rb))], np.r_[dec[rd], dec[rb]]))
    t_rows.append({"labels": g, "n_bl": len(bi), "n_dlbcl": len(di), "auc": auc(b, d),
                   "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5)})
transfer = pd.DataFrame(t_rows)
print(transfer.round(3).to_string(index=False))
transfer.to_csv(OUT + "stage09_direct_transfer.csv", index=False)

# %% Step 6. Top genes, MYC, and save.
top20 = coef.reindex(coef.abs().sort_values(ascending=False).index[:20])
top20 = pd.DataFrame({"gene": names.reindex(top20.index).values, "coef": top20.values.round(4)})
print("\nTop-20 coefficients (positive -> DLBCL):\n", top20.to_string(index=False))
myc_id = names.index[names == "MYC"][0]
print(f"MYC in top-500: {myc_id in sel_genes}")

old = pd.read_csv(OUT + "archive_all_genes_2026-11-20/rebuild_summary_nested.csv", index_col=0)["value"]
summary = pd.Series({"cv_accuracy": acc, "cv_balanced": bal, "p_value": p_value, "null_mean": null.mean(),
                     "null_max": null.max(), "n_genes": len(genes), "model_genes_on_array": int(on_array.sum()),
                     "direct_transfer_auc_morph": auc_m, "direct_transfer_auc_molec": auc_mol,
                     "tcga_called_dlbcl": ext[0]["called_DLBCL"], "ctsp_called_dlbcl": ext[1]["called_DLBCL"],
                     "lab_probe_balanced_stage08b": 0.854})
print(f"\nOld model (all genes): CV {old['cv_accuracy']:.4%}, balanced {old['cv_balanced']:.4%}, p {old['p_value']:.4f}")
summary.to_csv(OUT + "stage09_summary.csv", header=["value"])
pd.DataFrame(cm, index=["true_BL", "true_DLBCL"], columns=["pred_BL", "pred_DLBCL"]).to_csv(OUT + "stage09_confusion.csv")
top20.to_csv(OUT + "stage09_top20.csv", index=False)
pd.DataFrame({"null_fold_mean_accuracy": null}).to_csv(OUT + "stage09_permutation_null.csv", index_label="shuffle")
print("saved stage09_summary.csv, stage09_confusion.csv, stage09_top20.csv, stage09_permutation_null.csv")
