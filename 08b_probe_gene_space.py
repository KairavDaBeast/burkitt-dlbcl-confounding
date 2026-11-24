"""Stage 08b: why is the lab probe 0.995 in Stage 05 but 0.755 in Stage 08?

Stage 05 probe: all 60,660 genes (incl. non-coding), log2 + quantile normalization.
Stage 08 probe: 9,032 protein-coding genes also on the GSE4475 array and expressed, log2 only.
This reruns the same probe (NCICCR vs TCGA, 5 draws of 144 vs 48, 5-fold CV, top-500 variance,
L2 logistic regression; balanced accuracy, chance 0.50) on each gene set with and without quantile
normalization, so the two changes are separated.

Run: python 08b_probe_gene_space.py   (from the project folder; a few minutes)
"""
import gzip
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = "data/"
OUT = ""
N_TOP = 500

# read_pickle is safe here: the file is written by 05_download_full_tpm.py on this PC from GDC data.
full = pd.read_pickle(BASE + "full/gdc_tpm_all.pkl")
proj = pd.read_csv(BASE + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
BL, NC, TC = (list(full.columns[(proj == p).values]) for p in ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC"])
names = pd.read_csv(BASE + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"]
full = full[BL + NC + TC]
train_genes = full[BL + NC].dropna(how="any").index          # 60,660, as in Stage 05
full = full.loc[train_genes].fillna(0.0).astype(np.float32)

# Shared genes from Stage 08: protein-coding symbols on the GPL96 array, mean log2(TPM+1) >= 1 in BL+NCICCR.
with gzip.open(BASE + "geo/GSE4475_series_matrix.txt.gz", "rt") as f:
    lines = f.read().splitlines()
start = lines.index("!series_matrix_table_begin") + 1
end = lines.index("!series_matrix_table_end")
probes = pd.Index([l.split("\t", 1)[0].strip('"') for l in lines[start + 1:end]])
annot = pd.read_csv(BASE + "geo/GPL96.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"]).dropna()
annot = annot[~annot["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]
array_symbols = set(annot.reindex(probes.intersection(annot.index)))

pc_ids = train_genes.intersection(names.index)
log_pc_train = np.log2(full.loc[pc_ids, BL + NC] + 1)
expressed = pc_ids[(log_pc_train.mean(axis=1) >= 1).values]
sym = np.log2(full.loc[pc_ids].groupby(names.reindex(pc_ids).values).mean() + 1)
shared_sym = sym.index[sym.index.isin(array_symbols) & (sym[BL + NC].mean(axis=1) >= 1).values]
print(f"genes: all {len(train_genes)}, protein-coding {len(pc_ids)}, non-coding {len(train_genes) - len(pc_ids)}, "
      f"protein-coding expressed {len(expressed)}, shared with array (symbols) {len(shared_sym)}")


def quantile_map(M, ref):
    """Map every column of M onto the reference distribution by rank (Stage 02/05 method)."""
    ranks = pd.DataFrame(M).rank(method="average", axis=0).values
    pos = np.arange(1, len(ref) + 1)
    return np.column_stack([np.interp(ranks[:, c], pos, ref) for c in range(M.shape[1])])


def probe(Xn, Xt):
    out = []
    for seed in range(5):
        pick = np.random.default_rng(seed).choice(len(Xn), 3 * len(Xt), replace=False)
        Xp = np.vstack([Xn[pick], Xt])
        yp = np.r_[np.zeros(len(pick)), np.ones(len(Xt))].astype(int)
        dp = np.zeros(len(yp))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(Xp, yp):
            sel = np.argsort(-Xp[tr].var(axis=0))[:N_TOP]
            sc = StandardScaler().fit(Xp[tr][:, sel])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xp[tr][:, sel]), yp[tr])
            dp[te] = clf.decision_function(sc.transform(Xp[te][:, sel]))
        out.append(balanced_accuracy_score(yp, dp > 0))
    return float(np.mean(out))


def run(L):
    """L: genes x samples log2 table with columns BL + NC + TC. Returns (log2-only probe, quantile probe)."""
    L = np.asarray(L, dtype=np.float32)
    nBL, nNC = len(BL), len(NC)
    tr, nc, tc = L[:, :nBL + nNC], L[:, nBL:nBL + nNC], L[:, nBL + nNC:]
    p_log = probe(nc.T, tc.T)
    ref = np.sort(tr, axis=0).mean(axis=1)
    p_q = probe(quantile_map(nc, ref).T, quantile_map(tc, ref).T)
    return p_log, p_q


sets = {
    "all genes (Stage 05 set)": train_genes,
    "non-coding only": train_genes.difference(pc_ids),
    "protein-coding, all": pc_ids,
    "protein-coding, expressed": expressed,
}
rows = []
for name, ids in sets.items():
    p_log, p_q = run(np.log2(full.loc[ids] + 1))
    rows.append({"gene_set": name, "n_genes": len(ids), "probe_log2_only": p_log, "probe_quantile": p_q})
    print(f"{name:34s} n={len(ids):6d}  log2 only {p_log:.3f} | quantile {p_q:.3f}")
p_log, p_q = run(sym.loc[shared_sym])
rows.append({"gene_set": "shared with array (Stage 08 set)", "n_genes": len(shared_sym),
             "probe_log2_only": p_log, "probe_quantile": p_q})
print(f"{'shared with array (Stage 08 set)':34s} n={len(shared_sym):6d}  log2 only {p_log:.3f} | quantile {p_q:.3f}")

res = pd.DataFrame(rows)
print("matches pre-clean run (expected to drift after the BL cleanup):", abs(res.iloc[0]["probe_quantile"] - 0.995) < 0.01)
print("matches pre-clean run (expected to drift after the BL cleanup):", abs(res.iloc[-1]["probe_log2_only"] - 0.755) < 0.01)
res.round(4).to_csv(OUT + "stage08b_probe_gene_space.csv", index=False)
print("saved stage08b_probe_gene_space.csv")
