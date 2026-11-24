"""Stage 10 data: new cohorts for testing the frozen protein-coding model.

1. CGCI-HTMCP-DLBCL (GDC, open STAR-Counts): HIV+ DLBCL sequenced at BCGSC with rRNA depletion and
   first-strand libraries, the same center and protocol as the BL training cohort (CGCI-BLGSP).
2. GDC metadata for the training files: sample type, preservation, library selection.
3. GEO series matrices for same-laboratory array cohorts with BL and DLBCL: GSE10172 (pediatric, GPL96),
   GSE4732 (Dave 2006), GSE69051 / GSE69049 (Sha 2015, Illumina DASL), GSE48097 (Kiel, GPL96).

Run once: python 10_download_new_cohorts.py
"""
import io, json, os, tarfile, urllib.request
import pandas as pd

BASE = "data/"
NEW = BASE + "new/"
os.makedirs(NEW, exist_ok=True)
API = "https://api.gdc.cancer.gov/"


def post(endpoint, body, raw=False):
    req = urllib.request.Request(API + endpoint, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    return r.read() if raw else json.load(r)["data"]


# %% 1. HTMCP-DLBCL STAR-Counts: file list with case metadata
filt = {"op": "and", "content": [
    {"op": "in", "content": {"field": "cases.project.project_id", "value": ["CGCI-HTMCP-DLBCL"]}},
    {"op": "in", "content": {"field": "files.data_type", "value": ["Gene Expression Quantification"]}},
    {"op": "in", "content": {"field": "files.analysis.workflow_type", "value": ["STAR - Counts"]}},
    {"op": "in", "content": {"field": "access", "value": ["open"]}}]}
fields = ("file_id,cases.case_id,cases.submitter_id,cases.diagnoses.primary_diagnosis,cases.diagnoses.age_at_diagnosis,"
          "cases.demographic.age_at_index,cases.demographic.gender,cases.samples.sample_type,cases.samples.tissue_type")
hits = post("files", {"filters": filt, "fields": fields, "size": 500})["hits"]
rows = []
for h in hits:
    c = h["cases"][0]
    rows.append({"file_id": h["file_id"], "case_id": c["case_id"], "submitter_id": c.get("submitter_id"),
                 "diagnosis": (c.get("diagnoses") or [{}])[0].get("primary_diagnosis"),
                 "age": (c.get("demographic") or {}).get("age_at_index"),
                 "sex": (c.get("demographic") or {}).get("gender"),
                 "sample_type": (c.get("samples") or [{}])[0].get("sample_type")})
meta = pd.DataFrame(rows).sort_values("file_id")
meta.to_csv(NEW + "htmcp_files.csv", index=False)
print(f"HTMCP-DLBCL: {len(meta)} files, {meta.case_id.nunique()} cases")
print(meta.diagnosis.value_counts().to_string())

# Download all files in one tar (GDC data endpoint), keep tpm_unstranded per file.
if not os.path.exists(NEW + "htmcp_tpm.pkl"):
    tar_bytes = post("data", {"ids": list(meta.file_id)}, raw=True)
    tpm = {}
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile() or not m.name.endswith(".tsv"):
                continue
            fid = m.name.split("/")[0]
            df = pd.read_csv(tf.extractfile(m), sep="\t", comment="#")
            df = df[df["gene_id"].str.startswith("ENSG")].set_index("gene_id")
            tpm[fid] = df["tpm_unstranded"]
    pd.DataFrame(tpm).to_pickle(NEW + "htmcp_tpm.pkl")
print("HTMCP TPM table saved:", os.path.exists(NEW + "htmcp_tpm.pkl"))

# %% 2. Training-file metadata (sample type, preservation; library from the source BAM read groups)
samples = pd.read_csv(BASE + "full/samples_all.csv")
ids = list(samples["file_id"])
hits = post("files", {"filters": {"op": "in", "content": {"field": "file_id", "value": ids}},
                      "fields": "file_id,cases.samples.sample_type,cases.samples.preservation_method,"
                                "cases.samples.portions.analytes.aliquots.aliquot_id,cases.case_id",
                      "size": 2000})["hits"]
tm = pd.DataFrame([{"file_id": h["file_id"],
                    "sample_type": h["cases"][0]["samples"][0].get("sample_type"),
                    "preservation": h["cases"][0]["samples"][0].get("preservation_method")} for h in hits])
tm = samples.merge(tm, on="file_id", how="left")
tm.to_csv(NEW + "training_file_metadata.csv", index=False)
print("\nTraining files by project and sample type:")
print(tm.groupby(["project", "sample_type"]).size().to_string())
print("\nTraining files by project and preservation:")
print(tm.groupby(["project", "preservation"]).size().to_string())

# %% 3. GEO series matrices
for acc in ["GSE10172", "GSE4732", "GSE69051", "GSE69049", "GSE48097"]:
    stub = acc[:-3] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}/{acc}/matrix/"
    listing = urllib.request.urlopen(url, timeout=120).read().decode()
    files = sorted({p.split('"')[0] for p in listing.split('href="')[1:] if p.split('"')[0].endswith("_series_matrix.txt.gz")})
    for f in files:
        out = NEW + f
        if not os.path.exists(out):
            urllib.request.urlretrieve(url + f, out)
        print(f"{acc}: {f} {os.path.getsize(out) / 1e6:.1f} MB")
