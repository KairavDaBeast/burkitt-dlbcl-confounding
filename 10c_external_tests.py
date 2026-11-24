"""Stage 10 tests, exactly as pre-registered in STAGE10_PREREGISTRATION.md.

A   frozen protein-coding model on CGCI-HTMCP-DLBCL (DLBCL sequenced by the BL lab, same protocol)
A2  old all-gene model on the same cohort
B   GSE10172 (pediatric, U133A), direct transfer
C   GSE69049 + GSE69051 (Leeds, Illumina DASL, FFPE), direct transfer, clinical diagnosis labels
D   within BCGSC: CGCI-BLGSP BL vs HTMCP-DLBCL (CV), plus a lab probe NCICCR vs HTMCP (both DLBCL)
Exploratory: within-BL FFPE vs frozen scores; BL-NOS-only cross-validation.

Run after 10b_clean_bl_cohort.py and the pipeline rerun: python 10c_external_tests.py
"""
import gzip, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = "data/"
NEW = BASE + "new/"
OUT = ""
N_TOP = 500
rng = np.random.default_rng(10)

# %% Training data (cleaned cohort) and the two frozen models
# read_pickle is safe here: both pickles were written on this PC from GDC downloads (05_ and 10_ scripts).
full = pd.read_pickle(BASE + "full/gdc_tpm_all.pkl")
proj = pd.read_csv(BASE + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
names = pd.read_csv(BASE + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"]
BL, NC, TC = (list(full.columns[(proj == p).values]) for p in ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC"])
y = np.array(["BL"] * len(BL) + ["DLBCL"] * len(NC), dtype=object)
ht_raw = pd.read_pickle(NEW + "htmcp_tpm.pkl")
ht_meta = pd.read_csv(NEW + "htmcp_files.csv").set_index("file_id").reindex(ht_raw.columns)
print(f"Training: BL {len(BL)}, DLBCL {len(NC)} | HTMCP-DLBCL {ht_raw.shape[1]}")


def ref_of(M):
    return np.sort(M, axis=0).mean(axis=1)


def qmap(M, ref):
    ranks = pd.DataFrame(M).rank(method="average", axis=0).values
    pos = np.arange(1, len(ref) + 1)
    return np.column_stack([np.interp(ranks[:, c], pos, ref) for c in range(M.shape[1])])


def build(gene_filter):
    raw = full.loc[gene_filter, BL + NC].dropna(how="any")
    M = np.log2(raw.values.astype(float) + 1)
    ref = ref_of(M)
    X = qmap(M, ref).T
    sel = np.argsort(-X.var(axis=0))[:N_TOP]
    sc = StandardScaler().fit(X[:, sel])
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[:, sel]), y)
    return {"genes": raw.index, "ref": ref, "sel": sel, "sc": sc, "clf": clf}


def score_rnaseq(model, raw):
    M = np.log2(raw.reindex(model["genes"]).fillna(0.0).values.astype(float) + 1)
    X = qmap(M, model["ref"]).T
    return model["clf"].predict_proba(model["sc"].transform(X[:, model["sel"]]))[:, 1]   # P(DLBCL)


pc_model = build(full.index.isin(names.index))
all_model = build(np.ones(len(full), bool))
print(f"protein-coding model genes {len(pc_model['genes'])}; all-gene model genes {len(all_model['genes'])}, "
      f"non-coding in its top-500: {int((~all_model['genes'][all_model['sel']].isin(names.index)).sum())}")

rows = []

# %% A / A2: HTMCP-DLBCL
for label, model in [("A: protein-coding model", pc_model), ("A2: old all-gene model", all_model)]:
    p = score_rnaseq(model, ht_raw)
    called = p >= 0.5
    dl_only = ht_meta["diagnosis"].fillna("").str.contains("large B-cell", case=False).values
    rows.append({"test": label, "cohort": "CGCI-HTMCP-DLBCL (BCGSC, same protocol as BL)", "n": len(p),
                 "metric": "fraction called DLBCL", "value": called.mean(),
                 "detail": f"{called.sum()}/{len(p)}; DLBCL-diagnosis subset {called[dl_only].sum()}/{dl_only.sum()}; "
                           f"median P(DLBCL) {np.median(p):.3f}"})
    print(f"{label}: {called.sum()}/{len(p)} called DLBCL ({called.mean():.1%}); median P(DLBCL) {np.median(p):.3f}")
    if label.startswith("A:"):
        pd.DataFrame({"file_id": ht_raw.columns, "diagnosis": ht_meta["diagnosis"].values, "age": ht_meta["age"].values,
                      "p_dlbcl": p}).to_csv(OUT + "stage10_htmcp_scores.csv", index=False)
        ages = pd.to_numeric(ht_meta["age"], errors="coerce").values
        print(f"   HTMCP ages: median {np.nanmedian(ages):.0f} (n={np.isfinite(ages).sum()}); "
              f"Spearman(P(DLBCL), age) {stats.spearmanr(p[np.isfinite(ages)], ages[np.isfinite(ages)]).statistic:.2f}")

# %% D: same lab (BCGSC), BL vs HTMCP-DLBCL; and a lab probe NCICCR vs HTMCP (both DLBCL)
pcg = pc_model["genes"]
L_bl = np.log2(full.loc[pcg, BL].values.astype(float) + 1)
L_nc = np.log2(full.loc[pcg, NC].values.astype(float) + 1)
L_ht = np.log2(ht_raw.reindex(pcg).fillna(0.0).values.astype(float) + 1)


def cv_bal(A, B, seed=42):
    X = np.hstack([A, B])
    X = qmap(X, ref_of(X)).T                                   # label-free normalization over these samples
    yy = np.r_[np.zeros(A.shape[1]), np.ones(B.shape[1])].astype(int)
    d = np.zeros(len(yy))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, yy):
        sel = np.argsort(-X[tr].var(axis=0))[:N_TOP]
        sc = StandardScaler().fit(X[tr][:, sel])
        d[te] = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr][:, sel]), yy[tr]).decision_function(
            sc.transform(X[te][:, sel]))
    return balanced_accuracy_score(yy, d > 0)


bal_same_lab = cv_bal(L_bl, L_ht)
probe_vals = [cv_bal(L_nc[:, np.random.default_rng(s).choice(len(NC), 3 * L_ht.shape[1], replace=False)], L_ht)
              for s in range(5)]
rows.append({"test": "D: same lab, BL vs HTMCP-DLBCL", "cohort": "BCGSC only", "n": len(BL) + L_ht.shape[1],
             "metric": "CV balanced accuracy", "value": bal_same_lab, "detail": "disease differs, lab constant (HIV status differs)"})
rows.append({"test": "D: lab probe, NCICCR vs HTMCP", "cohort": "two labs, both DLBCL", "n": 4 * L_ht.shape[1],
             "metric": "CV balanced accuracy", "value": float(np.mean(probe_vals)), "detail": "lab differs, disease constant"})
print(f"D: BL vs HTMCP within BCGSC, balanced accuracy {bal_same_lab:.3f}; NCICCR vs HTMCP lab probe {np.mean(probe_vals):.3f}")


# %% B / C: array cohorts, direct transfer of the frozen protein-coding model
def read_series(path):
    with gzip.open(path, "rt", errors="replace") as f:
        lines = f.read().splitlines()
    s, e = lines.index("!series_matrix_table_begin"), lines.index("!series_matrix_table_end")
    cols = [c.strip('"') for c in lines[s + 1].split("\t")[1:]]
    data = pd.DataFrame([l.split("\t") for l in lines[s + 2:e]]).set_index(0)
    data.index = data.index.str.strip('"')
    data.columns = cols
    data = data.apply(pd.to_numeric, errors="coerce")
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
    return data, pd.DataFrame(meta, index=cols)


def platform_map(gpl):
    if gpl == "GPL96":
        a = pd.read_csv(BASE + "geo/GPL96.annot.gz", sep="\t", skiprows=27, usecols=["ID", "Gene symbol"]).dropna()
        a = a[~a["Gene symbol"].str.contains("///")]
        return a.set_index("ID")["Gene symbol"]
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


sym = names.reindex(pcg[pc_model["sel"]]).values


def direct_scores(arr):
    on = np.isin(sym, arr.index)
    sub = arr.loc[sym[on]]
    sd = sub.std(axis=1, ddof=1).replace(0, 1)
    Z = np.zeros((arr.shape[1], N_TOP))
    Z[:, on] = np.nan_to_num(sub.sub(sub.mean(axis=1), axis=0).div(sd, axis=0).values.T)
    return pc_model["clf"].decision_function(Z), int(on.sum())          # > 0 = DLBCL


def auc_ci(dec, bl, dl):
    a = roc_auc_score(np.r_[np.ones(dl.sum()), np.zeros(bl.sum())], np.r_[dec[dl], dec[bl]])
    bi, di = np.where(bl)[0], np.where(dl)[0]
    boots = [roc_auc_score(np.r_[np.ones(len(di)), np.zeros(len(bi))],
                           np.r_[dec[rng.choice(di, len(di))], dec[rng.choice(bi, len(bi))]]) for _ in range(1000)]
    return a, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def add_auc(test, cohort, dec, bl, dl, n_genes, extra=""):
    a, lo, hi = auc_ci(dec, bl, dl)
    rows.append({"test": test, "cohort": cohort, "n": int(bl.sum() + dl.sum()), "metric": "AUC (BL vs DLBCL)",
                 "value": a, "detail": f"95% CI {lo:.3f}-{hi:.3f}; BL {bl.sum()} vs DLBCL {dl.sum()}; "
                                       f"{n_genes}/500 model genes on array{extra}"})
    print(f"{test}: AUC {a:.3f} ({lo:.3f}-{hi:.3f}), BL {bl.sum()} vs DLBCL {dl.sum()}, genes {n_genes}/500 {extra}")


# B: GSE10172 (pediatric)
data, meta = read_series(NEW + "GSE10172_series_matrix.txt.gz")
arr = by_gene(data, platform_map("GPL96"))
key = next(k for k in meta.columns if "Consensus morphological diagnosis" in k)
print("\nGSE10172 consensus diagnoses:", meta[key].value_counts().to_dict())
dx = meta[key].str.lower()
bl_m = dx.str.contains("burkitt|^bl|atypical bl").values
dl_m = dx.str.contains("dlbcl|diffuse large").values & ~bl_m
ages = pd.to_numeric(meta[next(k for k in meta.columns if k.startswith("Age"))], errors="coerce")
print(f"   ages: median {ages.median():.0f}, range {ages.min():.0f}-{ages.max():.0f}")
dec, ng = direct_scores(arr)
add_auc("B: pediatric same-lab", "GSE10172 (Kiel, U133A, children)", dec, bl_m, dl_m, ng)

# C: GSE69049 + GSE69051 (Leeds); each platform z-scored on its own, then pooled AUC and per platform
decs, bls, dls, adults, info = [], [], [], [], []
for acc, gpl in [("GSE69049", "GPL8432"), ("GSE69051", "GPL14951")]:
    data, meta = read_series(NEW + f"{acc}_series_matrix.txt.gz")
    keep = ~meta["description"].fillna("").str.contains("replicate|relapse", case=False)
    data, meta = data.loc[:, keep.values], meta[keep]
    arr = by_gene(data, platform_map(gpl))
    dx = meta["clinic.diagnosis"].fillna("").str.lower()
    bl_c = dx.str.contains("burkitt").values
    dl_c = dx.str.contains("diffuse large|dlbcl").values
    age = pd.to_numeric(meta.get("age"), errors="coerce").values
    dec, ng = direct_scores(arr)
    add_auc(f"C: Leeds {acc}", f"{acc} ({gpl}, DASL FFPE)", dec, bl_c, dl_c, ng)
    decs.append(dec); bls.append(bl_c); dls.append(dl_c); adults.append(age >= 18)
dec, bl_c, dl_c, ad = np.concatenate(decs), np.concatenate(bls), np.concatenate(dls), np.concatenate(adults)
add_auc("C: Leeds pooled", "GSE69049 + GSE69051", dec, bl_c, dl_c, ng, "")
add_auc("C: Leeds pooled, adults", "GSE69049 + GSE69051, age >= 18", dec, bl_c & ad, dl_c & ad, ng, "")

# %% Exploratory: within-BL FFPE vs frozen; BL-NOS-only CV
oof = pd.read_csv(OUT + "oof_scores_nested.csv", index_col="sample")
ch = pd.read_csv(BASE + "full/bl_chosen_files.csv").set_index("case_id")
bl_oof = oof[oof.label == "BL"].join(ch[["preservation", "diagnosis"]], how="left")
ff, fr = bl_oof[bl_oof.preservation == "FFPE"].oof_decision, bl_oof[bl_oof.preservation == "Frozen"].oof_decision
mw = stats.mannwhitneyu(ff, fr)
rows.append({"test": "Exploratory: within-BL FFPE vs frozen", "cohort": "CGCI-BLGSP", "n": len(ff) + len(fr),
             "metric": "Mann-Whitney p (oof score)", "value": mw.pvalue,
             "detail": f"median score FFPE {ff.median():.2f} (n={len(ff)}) vs frozen {fr.median():.2f} (n={len(fr)}); "
                       f"misclassified BL: {int((bl_oof.oof_decision > 0).sum())}"})
print(f"\nWithin-BL FFPE {ff.median():.2f} (n={len(ff)}) vs frozen {fr.median():.2f} (n={len(fr)}), p = {mw.pvalue:.3g}")

nos = [c for c in BL if str(ch.loc[c, "diagnosis"]).startswith("Burkitt lymphoma, NOS")]
L_nos = np.log2(full.loc[pcg, nos].values.astype(float) + 1)
bal_nos = cv_bal(L_nos, L_nc)
rows.append({"test": "Exploratory: BL-NOS only vs NCICCR", "cohort": "training", "n": len(nos) + len(NC),
             "metric": "CV balanced accuracy", "value": bal_nos, "detail": f"{len(nos)} BL-NOS patients"})
print(f"BL-NOS only ({len(nos)}) vs NCICCR: CV balanced accuracy {bal_nos:.4f}")

res = pd.DataFrame(rows)
res.to_csv(OUT + "stage10_results.csv", index=False)
print("\n" + res[["test", "n", "metric", "value"]].round(3).to_string(index=False))
print("saved stage10_results.csv, stage10_htmcp_scores.csv")
