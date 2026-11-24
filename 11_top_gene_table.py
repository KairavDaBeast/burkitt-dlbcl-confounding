"""Stage 11: table for the author's top-gene replication check, plus a portable copy of the frozen model.

Part 1. Top-50 genes of the frozen protein-coding model, with the BL-minus-DLBCL difference and Welch p in
        three same-laboratory judges: GSE4475 (morphology), GSE10172 (children, consensus morphology) and
        Leeds (GSE69049 + GSE69051, each platform z-scored, then pooled; difference in SD units).
        The author applies the replication rule by hand (see Burkitt_TODO_for_you.md).
Part 2. Model kit (no pickles): genes, reference distribution, scaler and coefficients as CSV, plus a
        scoring script, so another lab can score its own data without sharing it.
Run: python 11_top_gene_table.py
"""
import gzip, json, os, shutil, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = "data/"
NEW = BASE + "new/"
OUT = ""
KIT = "model_kit/"

# %% Frozen model (same as 09_protein_coding_model.py Step 4, cleaned cohort)
# read_pickle is safe here: the file was written on this PC from GDC downloads (05_ / 10b_ scripts).
full = pd.read_pickle(BASE + "full/gdc_tpm_all.pkl")
proj = pd.read_csv(BASE + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
names = pd.read_csv(BASE + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"]
BL, NC = (list(full.columns[(proj == p).values]) for p in ["CGCI-BLGSP", "NCICCR-DLBCL"])
raw = full.loc[full.index.isin(names.index), BL + NC].dropna(how="any")
genes = raw.index
M = np.log2(raw.values.astype(float) + 1)
y = np.array(["BL"] * len(BL) + ["DLBCL"] * len(NC), dtype=object)
ref = np.sort(M, axis=0).mean(axis=1)
ranks = pd.DataFrame(M).rank(method="average", axis=0).values
X = np.column_stack([np.interp(ranks[:, c], np.arange(1, len(ref) + 1), ref) for c in range(M.shape[1])]).T
sel = np.argsort(-X.var(axis=0))[:500]
sc = StandardScaler().fit(X[:, sel])
clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[:, sel]), y)
coef = pd.Series(clf.coef_[0], index=genes[sel])
d_rna = pd.Series(X[:len(BL)].mean(axis=0) - X[len(BL):].mean(axis=0), index=genes)


# %% Judges
def read_series(path):
    with gzip.open(path, "rt", errors="replace") as f:
        lines = f.read().splitlines()
    s, e = lines.index("!series_matrix_table_begin"), lines.index("!series_matrix_table_end")
    cols = [c.strip('"') for c in lines[s + 1].split("\t")[1:]]
    data = pd.DataFrame([l.split("\t") for l in lines[s + 2:e]]).set_index(0)
    data.index = data.index.str.strip('"')
    data.columns = cols
    meta = [{} for _ in cols]
    for l in lines:
        if l.startswith("!Sample_characteristics_ch1"):
            for i, v in enumerate(l.split("\t")[1:]):
                v = v.strip('"')
                if ": " in v:
                    k, val = v.split(": ", 1)
                    meta[i][k.strip()] = val.strip()
        if l.startswith("!Sample_description"):
            for i, v in enumerate(l.split("\t")[1:]):
                meta[i].setdefault("description", v.strip('"'))
    return data.apply(pd.to_numeric, errors="coerce"), pd.DataFrame(meta, index=cols)


def gpl96():
    a = pd.read_csv(BASE + "geo/GPL96.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"]).dropna()
    return a[~a["Gene symbol"].str.contains("///")].set_index("ID")["Gene symbol"]


def illumina(gpl):
    path = NEW + f"{gpl}_table.txt"
    with open(path, encoding="utf-8", errors="replace") as f:
        start = next(i for i, l in enumerate(f) if l.startswith("!platform_table_begin"))
    t = pd.read_csv(path, sep="\t", skiprows=start + 1, usecols=["ID", "Symbol"], dtype=str).dropna()
    return t[~t["ID"].str.startswith("!")].set_index("ID")["Symbol"]


def by_gene(data, pmap):
    d = data.loc[data.index.intersection(pmap.index)].copy()
    d["gene"] = pmap.reindex(d.index).values
    d["m"] = d.drop(columns="gene").mean(axis=1)
    return d.sort_values("m", ascending=False).drop_duplicates("gene").set_index("gene").drop(columns="m")


def diff_p(arr, bl, dl):
    a, b = arr.loc[:, bl], arr.loc[:, dl]
    return a.mean(axis=1) - b.mean(axis=1), pd.Series(stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue, index=arr.index)


judges = {}
data, meta = read_series(BASE + "geo/GSE4475_series_matrix.txt.gz")
arr = by_gene(data, gpl96())
with gzip.open(BASE + "geo/GSE4475_series_matrix.txt.gz", "rt") as f:
    morph = [d.strip('"').split(" : ", 1)[1].strip() for d in next(l for l in f if "Morphologic.diagnosis" in l).split("\t")[1:]]
morph = np.array(morph)
judges["GSE4475"] = diff_p(arr, np.isin(morph, ["BL", "atypical BL", "BL-leukemia"]), morph == "DLBCL")

data, meta = read_series(NEW + "GSE10172_series_matrix.txt.gz")
arr = by_gene(data, gpl96())
dx = meta[next(k for k in meta.columns if "Consensus morphological diagnosis" in k)].str.lower()
bl10 = dx.str.contains("burkitt|^bl").values
judges["GSE10172"] = diff_p(arr, bl10, dx.str.contains("dlbcl").values & ~bl10)

zs, bls, dls = [], [], []
for acc, gpl in [("GSE69049", "GPL8432"), ("GSE69051", "GPL14951")]:
    data, meta = read_series(NEW + f"{acc}_series_matrix.txt.gz")
    keep = ~meta["description"].fillna("").str.contains("replicate|relapse", case=False)
    arr = by_gene(data.loc[:, keep.values], illumina(gpl))
    arr = arr.sub(arr.mean(axis=1), axis=0).div(arr.std(axis=1).replace(0, 1), axis=0)
    dx = meta[keep]["clinic.diagnosis"].fillna("").str.lower()
    zs.append(arr); bls.append(dx.str.contains("burkitt").values); dls.append(dx.str.contains("diffuse large|dlbcl").values)
common = zs[0].index.intersection(zs[1].index)
leeds = pd.concat([z.loc[common] for z in zs], axis=1)
leeds.columns = range(leeds.shape[1])
judges["Leeds"] = diff_p(leeds, np.concatenate(bls), np.concatenate(dls))

# %% Part 1: the table
top = coef.reindex(coef.abs().sort_values(ascending=False).index[:50])
tab = pd.DataFrame({"rank": range(1, 51), "gene": names.reindex(top.index).values, "model_coef": top.values.round(4)})
tab["pushes_toward"] = np.where(tab.model_coef > 0, "DLBCL", "BL")
tab["expected_sign_BL_minus_DLBCL"] = np.where(tab.model_coef > 0, "-", "+")
tab["rnaseq_diff"] = d_rna.reindex(top.index).values.round(2)
for j, (d, p) in judges.items():
    tab[f"{j}_diff"] = d.reindex(tab.gene).values.round(2)
    tab[f"{j}_p"] = [float(f"{v:.2g}") if np.isfinite(v) else np.nan for v in p.reindex(tab.gene).values]
tab.to_csv(OUT + "stage11_top50_gene_replication_table.csv", index=False)
shutil.copy(OUT + "stage11_top50_gene_replication_table.csv",
            "./Burkitt_top50_genes_for_you.csv")
print(tab.head(8).to_string(index=False))
print("measurable in: " + ", ".join(f"{j} {tab[f'{j}_p'].notna().sum()}/50" for j in judges))

# %% Part 2: model kit
os.makedirs(KIT, exist_ok=True)
unver = lambda g: g.split(".")[0] + ("_PAR_Y" if g.endswith("_PAR_Y") else "")   # drop version, keep PAR_Y copies distinct
pd.DataFrame({"gene_id": [unver(g) for g in genes], "symbol": names.reindex(genes).values,
              "reference_sorted_log2": ref}).to_csv(KIT + "reference_genes.csv", index=False)
pd.DataFrame({"gene_id": [unver(g) for g in genes[sel]], "symbol": names.reindex(genes[sel]).values,
              "scaler_mean": sc.mean_, "scaler_scale": sc.scale_, "coef": clf.coef_[0]}).to_csv(KIT + "model_genes.csv", index=False)
json.dump({"intercept": float(clf.intercept_[0]), "positive_class": "DLBCL", "n_train_bl": len(BL), "n_train_dlbcl": len(NC),
           "trained": "2026-11-20, CGCI-BLGSP tumour samples vs NCICCR-DLBCL, GDC STAR-Counts tpm_unstranded"},
          open(KIT + "model_info.json", "w"), indent=2)
print("model kit written to", KIT)
