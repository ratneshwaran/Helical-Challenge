#!/usr/bin/env python3
import os
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from geneformer import TranscriptomeTokenizer  # Geneformer tokenizer

# ---------- optional caches ----------
os.environ["PIP_CACHE_DIR"] = "/cs/student/projects1/aibh/2024/rmaheswa/cache"
os.environ["HF_HOME"]       = "/cs/student/projects1/aibh/2024/rmaheswa/cache/huggingface"

print("Geneformer Transcriptome Tokenization (ALS snRNA)")
print("================================================")

# ---------- 1) Load AnnData (RAW counts; NOT log/CPM) ----------
adata_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/data/integrated_with_quiescence.h5ad"
print("Loading AnnData from:", adata_path)
adata = sc.read_h5ad(adata_path)

# Safety check: warn if matrix looks log-transformed
mtx_max = adata.X.max() if not hasattr(adata.X, "max") else adata.X.max()
if (hasattr(mtx_max, "A") and mtx_max.A.max() < 50) or (np.max(mtx_max) < 50):
    print("NOTE: Matrix max is <50. Ensure this is RAW counts, not log/CPM.")

# ---------- 2) Ensure Ensembl IDs in adata.var["ensembl_id"] ----------
# Option A: already present
if "ensembl_id" in adata.var.columns:
    pass

# Option B: some pipelines store them in "feature_id"
elif "feature_id" in adata.var.columns:
    adata.var["ensembl_id"] = adata.var["feature_id"]

# Option C: map symbols -> Ensembl using a local dict (if you have one)
else:
    gene_name_id_path = Path(__file__).parent.parent / "geneformer_001" / "geneformer" / "gene_name_id_dict.pkl"
    if gene_name_id_path.exists():
        print("Loading gene name->Ensembl mapping:", gene_name_id_path)
        with open(gene_name_id_path, "rb") as f:
            gene_name_id_dict = pickle.load(f)
        adata.var["ensembl_id"] = adata.var.index.astype(str)
        ens = adata.var["ensembl_id"].to_numpy().copy()
        converted = 0
        for i, g in enumerate(ens):
            if g in gene_name_id_dict:
                ens[i] = gene_name_id_dict[g]
                converted += 1
        adata.var["ensembl_id"] = ens
        print(f"-> Converted {converted} symbols to Ensembl IDs via mapping.")
    else:
        # last resort: assume var_names ARE Ensembl (strip version if present)
        adata.var["ensembl_id"] = adata.var_names.astype(str).str.replace(r"\.\d+$","", regex=True)
        print("WARNING: No mapping provided; assuming var_names are Ensembl (version stripped).")

# ---------- 3) Required obs fields ----------
# Geneformer needs per-cell total counts in 'n_counts'
if "total_counts" in adata.obs.columns:
    adata.obs["n_counts"] = adata.obs["total_counts"].astype(np.float64)
else:
    # fallback to sum over X
    if hasattr(adata.X, "A"):
        adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).ravel()
    else:
        adata.obs["n_counts"] = adata.X.sum(axis=1)

# Optional boolean used by some examples
adata.obs["filter_pass"] = True

print(f"Genes: {adata.n_vars}, cells: {adata.n_obs}")
print("Has 'ensembl_id' in var:", "ensembl_id" in adata.var.columns)
print("Added 'n_counts' from total_counts" if "total_counts" in adata.obs.columns else "Computed 'n_counts' by summing X")

# ---------- 4) Choose which obs metadata to carry into the .dataset ----------
# (Everything below comes from your example .obs)
obs_to_keep = [
    "Sample_ID", "Donor", "Region", "Sex",
    "Condition", "Group", "C9_pos",
    "CellClass", "CellType", "SubType",
    "Cellstates_LVL1", "Cellstates_LVL2", "Cellstates_LVL3",
    "total_counts", "log1p_total_counts",
    "total_counts_mt", "log1p_total_counts_mt", "pct_counts_mt",
    "n_genes",
    "split"
]

present = [k for k in obs_to_keep if k in adata.obs.columns]
missing = [k for k in obs_to_keep if k not in adata.obs.columns]
if missing:
    print("Missing (will be ignored):", missing)

# Build the mapping the tokenizer expects: {input_obs_key: output_name}
custom_attr_name_dict = {k: k for k in present}

# Optional: ensure categorical/string types so they survive round-trips nicely
for k in present:
    if adata.obs[k].dtype.kind in {"i","f"}:
        # leave numeric QC columns as-is
        continue
    adata.obs[k] = adata.obs[k].astype(str)

# ---------- 5) Save prepped AnnData and tokenize (V2) ----------
prepped_path = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/data/ALS_snRNA_raw_prepped.h5ad"
adata.write_h5ad(prepped_path)
print("Wrote prepped AnnData to:", prepped_path)

out_dir = "/cs/student/projects1/aibh/2024/rmaheswa/Dissertation/Geneformer/tokenized/"
Path(out_dir).mkdir(parents=True, exist_ok=True)

print("Tokenizing with Geneformer V2…")
tk = TranscriptomeTokenizer(
    custom_attr_name_dict=custom_attr_name_dict,
    nproc=4,
    model_version="V2"
)
tk.tokenize_data(
    data_directory=prepped_path,
    output_directory=out_dir,
    output_prefix="ALS",
    file_format="h5ad"
)
print(f"Done. Output: {out_dir}/ALS.dataset")

print("Next: use EmbExtractor / InSilicoPerturber with input_data_file set to the .dataset above.")
