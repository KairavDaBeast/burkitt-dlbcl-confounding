# Burkitt lymphoma vs DLBCL model kit

This kit lets a lab test a frozen classifier on its own RNA-seq data without sharing that data. Only a short summary of counts and AUCs comes back.

## What is in the folder

| File | Contents |
|---|---|
| `model_genes.csv` | The 500 genes, with scaler mean, scale and coefficient (positive pushes toward DLBCL) |
| `reference_genes.csv` | The 19,962 protein-coding genes and the sorted reference distribution used for quantile mapping |
| `model_info.json` | Intercept and training details |
| `score_cohort.py` | Scoring script (Python 3 with numpy, pandas and scikit-learn) |

## How to run it

1. Make a TPM table: genes in rows (Ensembl IDs, with or without version, or gene symbols) and samples in columns, saved as CSV or TSV.
2. Optionally, make `labels.csv` with columns `sample,label`, where label is BL or DLBCL.
3. Run `python score_cohort.py tpm.csv labels.csv`.
4. Send back only the printed SUMMARY block. `scores.csv` stays with you.

## How the model was built

- **Training data:** GDC STAR-Counts `tpm_unstranded`, with CGCI-BLGSP tumour samples (254 BL) against NCICCR-DLBCL (481).
- **Preprocessing:** protein-coding genes, log2(TPM+1), then quantile normalization to the training reference.
- **Classifier:** the top-500 variance genes, standardized, then L2 logistic regression.

## Important

- The model ranks samples well within one laboratory, but its 0.5 cut-off carries a laboratory offset. On DLBCL from the BL training lab it called 6 of 55 DLBCL.
- For that reason the script also reports a cohort-centred AUC.
- Self-check: on CGCI-HTMCP-DLBCL the script reproduces the published scores to within 0.0001.
