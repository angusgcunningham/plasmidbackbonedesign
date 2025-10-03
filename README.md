```markdown
# Plasmid Backbone Design with DNA Language Models

> Clean, reproducible code for end-to-end *E. coli* plasmid backbone generation and QC using a DNA language model.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](#requirements)
[![Conda Envs](https://img.shields.io/badge/conda-2%20envs-44A833.svg)](#environments)
[![Reproducible](https://img.shields.io/badge/reproducible-yes-4A90E2.svg)](#reproducibility)

---

## Table of Contents
- [Overview](#overview)
- [What’s in this repo](#whats-in-this-repo)
- [What’s intentionally **not** in this repo](#whats-intentionally-not-in-this-repo)
- [Requirements](#requirements)
- [Environments](#environments)
- [Quickstart](#quickstart)
  - [1) Fine-tune / train](#1-fine-tune--train)
  - [2) Generate sequences](#2-generate-sequences)
  - [3) Run QC filters](#3-run-qc-filters)
  - [4) Analyse results & make figures](#4-analyse-results--make-figures)
- [Script Entry Points](#script-entry-points)
- [Recommended VS Code setup](#recommended-vs-code-setup)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Citing](#citing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview
This project implements an end-to-end pipeline for designing realistic plasmid backbones:

1. **Training / fine-tuning** a DNA LLM on curated circular plasmids.  
2. **Controlled generation** of candidate sequences (prompted seeds such as `ATG` or GFP cassette).  
3. **Bioinformatics QC**:  
   - ORI detection / validation  
   - ARG detection  
   - Repeat detection (hard gates on maximum repeat length)  
   - Heuristic two-stage pass filters (broad → strict)  
4. **Analysis**: k-mer divergence, annotation summaries, and figure creation.

The codebase is structured so you can **swap models**, **swap prompts**, and **re-run QC** on large batches quickly.

---

## What’s in this repo
```

src/                  # All reusable code (importable)
analysis/           # k-mer analysis, annotation wrappers, summaries
fasta/              # FASTA utilities (filtering, upsampling, token-length tools)
qc/                 # Two-stage QC, repeats, similarity scoring
training/           # Training & generation scripts
notebooks/            # Reproducible EDA/figure notebooks
envs/                 # Two conda environments (training/generation & QC)
docs/                 # Short notes / optional project docs

````

### Highlights
- **Two-stage QC** with configurable thresholds (counts, identity, coverage, repeat caps).
- **Repeat-aware filtering** to block synthesis-unfriendly candidates early.
- **K-mer & annotation** helpers to check realism shifts after prompting or FT.
- **Separated environments** for training vs QC (keeps deps lean and stable).

---

## What’s intentionally **not** in this repo
To keep the repo light, portable, and safe to publish:

- **No datasets** (FASTA, tarballs, raw metadata)  
- **No model checkpoints** (e.g., `.pt`, `.bin`, `.safetensors`)  
- **No generated results** (CSVs, plots, HTML reports)  
- **No third-party databases** (e.g., `plasmidfinder_db/`)  

> Keep those in sibling folders (or on object storage) and point the scripts to them via CLI args.

---

## Requirements
- Linux or macOS
- **Conda** (Miniconda/Anaconda/Mamba)
- Python **3.10+**
- (Optional) **VS Code** with Python + Jupyter extensions

---

## Environments
Two environments are provided to avoid dependency cross-talk:

- `envs/training_generation.yml` → model fine-tuning & generation  
- `envs/plasmid_qc.yml` → QC + bioinformatics/annotation tooling

Create them:
```bash
conda env create -f envs/training_generation.yml
conda env create -f envs/plasmid_qc.yml
````

> If you update packages, please re-export to these files so others can reproduce:
>
> ```bash
> conda env export --from-history > envs/training_generation.yml
> conda env export --from-history > envs/plasmid_qc.yml
> ```

---

## Quickstart

> **Tip:** Add the repo’s `src/` to `PYTHONPATH` so scripts can import modules cleanly:
>
> ```bash
> export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
> ```

### 1) Fine-tune / train

Activate the training env:

```bash
conda activate training-generation
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
```

Run one of the provided trainers (adapt CLI flags inside file as needed):

```bash
python src/training/20kPlasmidgpt.py
# or
python src/training/30kPlasmidGPT.py
```

Expected output: a model directory or checkpoint (stored **outside** the repo).

---

### 2) Generate sequences

```bash
conda activate training-generation
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Example: base model sampling
python src/training/generate_base.py \
  --out /path/to/out/generated_base.fasta \
  --max-length 4000 \
  --temperature 1.0 \
  --num-seqs 1000
```

You can also use `generate_base2.py` or `generate_ft4.py` depending on your workflow.

---

### 3) Run QC filters

```bash
conda activate plasmid-qc
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

# Two-stage QC on a FASTA of candidates
python src/qc/filter_qc_two_stage.py \
  --input /path/to/out/generated_base.fasta \
  --out   /path/to/qc_out/ \
  --strict
```

This will:

* annotate/detect ORIs and ARGs,
* compute repeat stats,
* write **CSV summaries** and a **passed FASTA** into the `--out` directory (outside the repo).

---

### 4) Analyse results & make figures

Open notebooks (prefer `nb_conda_kernels` or `ipykernel` inside your env):

* `notebooks/training_analysis.ipynb`
* `notebooks/kmer_divergence_analysis.ipynb`
* `notebooks/qcfigs.ipynb`
* `notebooks/sampling_outcomes.ipynb`

> If your data lives outside the repo, set an environment variable at the top of the notebook (e.g., `DATA_ROOT=...`) and reference it in paths.

---

## Script Entry Points

| Area     | Script/Module                              | What it does (typical I/O)                                               |
| -------- | ------------------------------------------ | ------------------------------------------------------------------------ |
| Training | `src/training/20kPlasmidgpt.py`            | Fine-tune DNA LLM (configure dataset/model paths inside).                |
| Training | `src/training/30kPlasmidGPT.py`            | Alternative fine-tuning run with different corpora/stride.               |
| Generate | `src/training/generate_base.py`            | Sample from base or FT model → FASTA (`--out`).                          |
| QC       | `src/qc/filter_qc_two_stage.py`            | Apply broad + strict filters; expects FASTA; outputs CSV + passed FASTA. |
| QC       | `src/qc/repeats.py`                        | Detect longest repeats (hard caps e.g. ≥50 bp).                          |
| QC       | `src/qc/similarity_score.py`               | Similarity metrics vs training sets (to curb near-duplicates).           |
| FASTA    | `src/fasta/bulk_filter_fasta.py`           | Subset/clean FASTA by id/length/regex.                                   |
| FASTA    | `src/fasta/upsample_data.py`               | Rotation-based upsampling for circular sequences.                        |
| Analysis | `src/analysis/kmer_analysis.py`            | K-mer divergence summaries for realism checks.                           |
| Analysis | `src/analysis/summarize_plannotate_gbk.py` | Collate annotations (e.g., ORI/ARG hits) into tidy tables.               |
| Analysis | `src/analysis/plannotate_wrapper.py`       | Helper wrapper to run/parse annotations for batches.                     |

> Run any script with `-h/--help` if implemented, or open the file to adjust parameters.

---

## Recommended VS Code setup

* Extensions: **Python**, **Jupyter**, **Pylance**.
* Workspace settings:

  * Set default interpreter to your conda env.
  * Add `${workspaceFolder}/src` to `"python.analysis.extraPaths"` or export `PYTHONPATH` in **Terminal > New** profile.
* Optional code quality:

  * Add a `.pre-commit-config.yaml` (Black, Ruff, isort) and run:

    ```bash
    pip install pre-commit
    pre-commit install
    ```

---

## Reproducibility

* **Conda envs** are versioned under `envs/`.
* **Determinism**: fix seeds where applicable in training/generation scripts.
* **Data & models**: store outside the repo; keep a small **text file of run parameters** next to outputs for auditability.
* **Paths**: prefer CLI args or env vars over hard-coding.

---

## Troubleshooting

**“Module not found …”**

* Ensure `src/` is on `PYTHONPATH`:
  `export PYTHONPATH="$(pwd)/src:$PYTHONPATH"`

**“Nothing happens / empty outputs in QC”**

* Check your input FASTA has sequences; confirm annotation tools’ paths if you use external binaries; try `--verbose` if available.

**“Too many repeats / sequences fail strict QC”**

* Relax repeat cap (for exploration), or reduce temperature / increase top-p during generation.

**Conda conflicts**

* Use `mamba env create -f …` or export **from-history** to minimise solver load.

---

## Citing

If this code or pipeline helps your work, please cite:

> **Plasmid Backbone Design with DNA Language Models**
> Angus G. Cunningham, 2025.
> MSc Thesis, University College London.

(If a DOI or thesis link is available, add it here.)

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE).

---

```
