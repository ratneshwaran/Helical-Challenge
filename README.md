# Helical Challenge: In-silico Perturbation Analysis using Geneformer V2

This project applies Geneformer V2 to simulate in-silico perturbations of ALS (Amyotrophic Lateral Sclerosis)–associated genes and interpret their effects in a biologically informed embedding space.

---

## Overview

The workflow follows the Helical Challenge tasks:

1. **Prepare the data**  
   - Tokenize the ALS/PN single-nucleus RNA-seq dataset.  
   - Generate a Geneformer-compatible `.dataset` file.

2. **Apply perturbations**  
   - Run in-silico perturbations targeting ALS-associated genes.  
   - Use Geneformer's InSilicoPerturber to simulate:
     - Knock-down (delete) and Knock-up (overexpress) perturbations.  
   - Process results into scalar cosine shifts indicating embedding-level effects.

3. **Interpret the embedding space**  
   - Extract embeddings from the pretrained Geneformer model.  
   - Visualize and analyze the latent space through:
     - UMAP projection  
     - Centroid shifts between ALS and PN  
     - KMeans clustering  
     - Neighbourhood analysis (ALS_nn_fraction)  
     - Perturbation effect magnitudes (KUeff values)

---

## Environment Setup

### Requirements

- Python 3.10 or higher  
- CUDA-capable GPU (recommended for Geneformer)  
- Packages listed in `requirements.txt`

### Installation

```bash
git clone <your-repository-url>
cd Helical_Challenge
python -m venv helical_challenge_venv
source helical_challenge_venv/bin/activate   # Windows: .\helical_challenge_venv\Scripts\activate
pip install -r requirements.txt
git clone https://huggingface.co/ctheodoris/Geneformer
cd Geneformer
pip install .
```

---

## Project Structure

```
Helical_Challenge/
│
├── data/
│   ├── prepped/                  # preprocessed .h5ad input data
│   └── tokenized/                # Geneformer tokenized dataset (.dataset)
│
├── Geneformer/                   # Geneformer model weights and code
│
├── results/
│   ├── isp_als/                  # in-silico perturbation (ISP) outputs
│   └── embeddings/               # EmbExtractor outputs (.csv / .pickle)
│
├── notebooks/
│   ├── in_silico_perturbation.ipynb
│   ├── apply_perturbations_to_ALS_genes.ipynb
│   └── interpret_embedding_space.ipynb
│
├── helpers/                      # optional helper scripts
│
├── requirements.txt
└── README.md
```

---

## Workflow Summary

### Step 1: Embedding Extraction

Run the EmbExtractor on the unperturbed dataset using Geneformer V2.

```python
emb_extractor = EmbExtractor(
    model_type="Pretrained",
    num_classes=1,
    emb_mode="cls",
    emb_layer=-1,
    forward_batch_size=FWD_BATCH,
    nproc=NPROC,
    model_version="V2",
)
emb_extractor.extract_embs(
    model_directory=MODEL_DIR,
    input_data_file=str(DATASET),
    output_directory=str(EMB_DIR),
    output_prefix="ALS_unperturbed",
)
```

### Step 2: In-Silico Perturbations

Simulate perturbations in ALS disease genes:

```python
run_isp_batched(
    ALS_GENES_ENSEMBL,
    perturb_type="overexpress",
    prefix="ALS_KU"
)
```

The resulting pickle files contain per-cell cosine similarities representing perturbation effects.

### Step 3: Interpret the Embedding Space

1. Load embeddings and metadata
2. Compute UMAP projection
3. Cluster and visualize ALS vs PN
4. Compute neighbourhood statistics and perturbation effect distributions

---

## Citation

If you use Geneformer, please cite:

Theodoris, C.V. *et al.*, "Geneformer: Foundation model for cell biology," *Nature*, 2024.
