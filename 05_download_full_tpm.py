"""Stage 05 data: full-gene TPM for the training cohort, TCGA-DLBC, and CTSP-DLBCL1.

Same open GDC STAR-Counts files and column (tpm_unstranded) as Stage 01 and Stage 05.
Unlike 07_download_gdc_tpm.py, this keeps ALL genes, not just protein-coding ones, because
the Stage 03 model's top-variance genes include non-coding RNAs (e.g. MIR155HG).
One profile per patient (first file_id in sorted order). Failed files are retried, then skipped.

Output: data/full/gdc_tpm_all.pkl   (genes x samples, float32)
        data/full/samples_all.csv   (sample -> project)
Run:    python 05_download_full_tpm.py
"""
import io, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

OUT = "data/full/"
API = "https://api.gdc.cancer.gov/"
PROJECTS = ["CGCI-BLGSP", "NCICCR-DLBCL", "TCGA-DLBC", "CTSP-DLBCL1"]


def list_files(project):
    filt = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}}]}
    body = json.dumps({"filters": filt, "fields": "file_id,cases.case_id", "size": 5000}).encode()
    req = urllib.request.Request(API + "files", data=body, headers={"Content-Type": "application/json"})
    hits = json.load(urllib.request.urlopen(req, timeout=120))["data"]["hits"]
    one_per_patient = {}
    for h in sorted(hits, key=lambda h: h["file_id"]):
        one_per_patient.setdefault(h["cases"][0]["case_id"], h["file_id"])
    return [(fid, case, project) for case, fid in one_per_patient.items()]


def fetch_tpm(fid):
    last_err = None
    for attempt in range(4):
        try:
            raw = urllib.request.urlopen(API + "data/" + fid, timeout=300).read().decode()
            df = pd.read_csv(io.StringIO(raw), sep="\t", comment="#")
            return df.set_index("gene_id")["tpm_unstranded"].astype("float32")
        except Exception as e:
            last_err = e
            time.sleep(5 * (attempt + 1))
    print("SKIPPED", fid, last_err, flush=True)
    return None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = [f for p in PROJECTS for f in list_files(p)]
    print(len(files), "files to fetch", flush=True)
    cols, meta = {}, []
    with ThreadPoolExecutor(8) as pool:
        for i, ((fid, case, project), tpm) in enumerate(zip(files, pool.map(lambda f: fetch_tpm(f[0]), files))):
            if tpm is None:
                continue
            cols[case] = tpm
            meta.append({"sample": case, "project": project, "file_id": fid})
            if i % 100 == 0:
                print(i, "done", flush=True)
    mat = pd.DataFrame(cols)
    mat.to_pickle(OUT + "gdc_tpm_all.pkl")
    pd.DataFrame(meta).to_csv(OUT + "samples_all.csv", index=False)
    print("saved", mat.shape, pd.DataFrame(meta)["project"].value_counts().to_dict(), flush=True)
