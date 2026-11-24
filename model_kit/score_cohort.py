"""Score an RNA-seq cohort with the frozen Burkitt lymphoma vs DLBCL model. The data never leave your computer.

Input:  a TPM table (CSV or TSV) with genes in rows (Ensembl IDs, with or without version, or gene symbols)
        and samples in columns. Optional: a labels file with columns sample,label (label = BL or DLBCL).
Usage:  python score_cohort.py tpm.csv [labels.csv]
Output: scores.csv (per-sample P(DLBCL)) and, with labels, a printed summary. Only the summary needs to be
        shared back; it contains no patient-level data.
Needs:  Python 3, numpy, pandas, scikit-learn.
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

K = os.path.dirname(os.path.abspath(__file__)) + os.sep
ref = pd.read_csv(K + "reference_genes.csv")
mg = pd.read_csv(K + "model_genes.csv")
info = json.load(open(K + "model_info.json"))

tpm = pd.read_csv(sys.argv[1], index_col=0, sep=None, engine="python")
idx = tpm.index.astype(str)
key = "gene_id" if idx.str.startswith("ENSG").mean() > 0.5 else "symbol"
unver = lambda s: s.str.split(".").str[0] + np.where(s.str.endswith("_PAR_Y"), "_PAR_Y", "")   # keep PAR_Y copies distinct
tpm.index = unver(idx) if key == "gene_id" else idx
tpm = tpm.apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()
aligned = tpm.reindex(ref[key]).fillna(0.0)                    # missing gene -> 0, as in the paper
print(f"Genes matched by {key}: {int(ref[key].isin(tpm.index).sum())} of {len(ref)}; samples: {tpm.shape[1]}")

logged = np.log2(aligned.values.astype(float) + 1)
ranks = pd.DataFrame(logged).rank(method="average", axis=0).values
pos = np.arange(1, len(ref) + 1)
Q = np.column_stack([np.interp(ranks[:, c], pos, ref["reference_sorted_log2"].values) for c in range(logged.shape[1])])
row = {g: i for i, g in enumerate(ref["gene_id"])}
F = Q[[row[g] for g in mg["gene_id"]], :].T                    # samples x 500 model genes

dec = ((F - mg["scaler_mean"].values) / mg["scaler_scale"].values) @ mg["coef"].values + info["intercept"]
p_dlbcl = 1 / (1 + np.exp(-dec))
Fz = (F - F.mean(axis=0)) / np.where(F.std(axis=0) > 0, F.std(axis=0), 1)
dec_within = Fz @ mg["coef"].values                             # cohort-centred score (removes a shared lab offset)
out = pd.DataFrame({"sample": tpm.columns, "p_dlbcl": p_dlbcl.round(4), "called": np.where(p_dlbcl >= 0.5, "DLBCL", "BL"),
                    "score_within_cohort": dec_within.round(4)})
out.to_csv("scores.csv", index=False)
print(f"Called DLBCL: {int((p_dlbcl >= 0.5).sum())} of {len(p_dlbcl)}. Per-sample scores saved to scores.csv")

if len(sys.argv) > 2:
    lab = pd.read_csv(sys.argv[2]).set_index("sample")["label"].reindex(tpm.columns)
    ok = lab.isin(["BL", "DLBCL"]).values
    t = (lab[ok] == "DLBCL").astype(int).values
    print("\nSUMMARY (safe to share):")
    for name in ["BL", "DLBCL"]:
        m = (lab == name).values
        print(f"  true {name}: n={m.sum()}, called DLBCL {int((p_dlbcl[m] >= 0.5).sum())}")
    if 0 < t.sum() < len(t):
        print(f"  AUC (model as is): {roc_auc_score(t, dec[ok]):.3f}")
        print(f"  AUC (cohort-centred score): {roc_auc_score(t, dec_within[ok]):.3f}")
