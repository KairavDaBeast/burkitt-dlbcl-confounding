"""Stage 10d (post hoc, exploratory; NOT in the pre-registration).

Test A failed: the frozen model called 49 of 55 HTMCP-DLBCL samples BL. Question: is that a shift in the
score (lab offset) or a failure to rank? Within the BL lab (BCGSC), do HTMCP-DLBCL samples score as more
DLBCL-like than held-out BL samples?

5-fold over BL patients: train on the other BL folds + all NCICCR-DLBCL (same pipeline as the main model),
then score the held-out BL fold and all HTMCP-DLBCL. AUC (BL vs HTMCP) per fold, two ways:
  absolute  decision values as the model outputs them (threshold-free, so an offset does not matter)
  within    the held-out BL + HTMCP features z-scored together first (removes the shared-lab level)
Run: python 10d_same_lab_ranking.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

BASE = "data/"
OUT = ""
# read_pickle is safe here: both pickles were written on this PC from GDC downloads (05_ and 10_ scripts).
full = pd.read_pickle(BASE + "full/gdc_tpm_all.pkl")
proj = pd.read_csv(BASE + "full/samples_all.csv", index_col="sample")["project"].reindex(full.columns)
names = pd.read_csv(BASE + "gdc_tpm.csv.gz", index_col=0, usecols=[0, 1])["gene_name"]
BL, NC = (list(full.columns[(proj == p).values]) for p in ["CGCI-BLGSP", "NCICCR-DLBCL"])
genes = full.index[full.index.isin(names.index)]
L = lambda df: np.log2(df.reindex(genes).fillna(0.0).values.astype(float) + 1)
Lbl, Lnc, Lht = L(full[BL]), L(full[NC]), L(pd.read_pickle(BASE + "new/htmcp_tpm.pkl"))


def qmap(M, ref):
    ranks = pd.DataFrame(M).rank(method="average", axis=0).values
    pos = np.arange(1, len(ref) + 1)
    return np.column_stack([np.interp(ranks[:, c], pos, ref) for c in range(M.shape[1])])


rows = []
for k, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=42).split(BL)):
    Mtr = np.hstack([Lbl[:, tr], Lnc])
    ref = np.sort(Mtr, axis=0).mean(axis=1)
    Xtr = qmap(Mtr, ref).T
    ytr = np.r_[np.zeros(len(tr)), np.ones(Lnc.shape[1])]
    sel = np.argsort(-Xtr.var(axis=0))[:500]
    sc = StandardScaler().fit(Xtr[:, sel])
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr[:, sel]), ytr)
    Xb, Xh = qmap(Lbl[:, te], ref).T[:, sel], qmap(Lht, ref).T[:, sel]
    db, dh = clf.decision_function(sc.transform(Xb)), clf.decision_function(sc.transform(Xh))
    Z = StandardScaler().fit_transform(np.vstack([Xb, Xh]))
    dz = Z @ clf.coef_[0]
    t = np.r_[np.zeros(len(te)), np.ones(Xh.shape[0])]
    rows.append({"fold": k, "n_bl": len(te), "n_htmcp": Xh.shape[0],
                 "auc_absolute": roc_auc_score(t, np.r_[db, dh]), "auc_within_lab_z": roc_auc_score(t, dz),
                 "htmcp_called_dlbcl": int((dh > 0).sum()), "heldout_bl_called_bl": int((db < 0).sum())})
res = pd.DataFrame(rows)
print(res.round(3).to_string(index=False))
print(f"\nMean AUC absolute {res.auc_absolute.mean():.3f} | within-lab z-scored {res.auc_within_lab_z.mean():.3f}")
res.to_csv(OUT + "stage10d_same_lab_ranking.csv", index=False)
