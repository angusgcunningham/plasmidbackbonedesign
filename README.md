# Plasmid Backbone Design Pipeline

Reproducible end-to-end pipeline for training, sampling, QC, and analysis of plasmid backbones using DNA language models.

## Overview
This repository provides:
- **Training** of PlasmidGPT model on plasmid FASTA.
- **Sampling** of plasmid backbones with length controls.
- **QC** (ORI/ARG detection, repeats) and pass/fail filtering.
- **Analysis** (similarity scoring and summaries).

Training data and figure notebooks are available on **Zenodo** (see the Data section).

## Requirements
- Linux or macOS
- Conda (Miniconda/Anaconda/Mamba)
- Python 3.10+

## Environments
Create the two conda environments:

```bash
conda env create -f envs/training_generation.yml
conda env create -f envs/plasmid_qc.yml
```

## Data (Zenodo)
Training data and figure notebooks are hosted on Zenodo.

1. Download the archive from Zenodo: **<https://zenodo.org/records/17820200**
2. Extract into the repo (or point paths via config):

```bash
# Example (adjust to your local path)
tar -xzf plasmid_backbone_data.tar.gz -C /path/to/repo
```

Expected local layout (example):
```
./data/
  debug_fasta/
  train_fasta/
  oriVdb/
    oriV_db.nhr
    oriV_db.nin
    oriV_db.nsq
    oriV_refs.fasta
./notebooks/
```

## Configuration
Config presets live in `configs/`.
Key fields:
- `paths.dataset_input` (FASTA dir)
- `paths.tokenizer_json`
- `paths.base_model`
- `paths.model_output_dir`
- `generation.num_samples`
- `qc.oridb_prefix`, `qc.oridb_ref`

You can point to any config with `--config`.

## Quick Start (Full Run)
Run a full pipeline for a preset (example `ft15k`):

```bash
# training + sampling (GPU environment)
conda activate training_generation
python main.py --mode train --preset ft15k --config configs/ft15k.yaml --run-name myrun
python main.py --mode sample --sampling-preset ft --config configs/ft15k.yaml --run-name myrun

# QC + analysis (QC environment)
conda activate plasmid-qc
python main.py --mode qc --config configs/ft15k.yaml --run-name myrun
python main.py --mode analysis --config configs/ft15k.yaml --run-name myrun
```

## Output Layout
All outputs are written under `runs/<run_name>/`:

```
runs/<run_name>/
  models/<preset>/                 # trained model artifacts
  generations/                     # per-sequence FASTA + generations_metadata.csv
  qc/                              # qc_summary.csv, repeats.csv, aggregate_* + per-seq outputs
  qc/individual_sequences_qc/      # per-sequence QC folders
  analysis/                        # analysis_summary.csv + extra outputs
  logs/                            # pipeline.log
  manifest.json                    # resolved inputs/outputs
  config_used.yml                  # resolved config used for the run
```

## Running Individual Stages

```bash
# Train
conda activate training_generation
python main.py --mode train --preset ft15k --config configs/ft15k.yaml --run-name myrun

# Sample
conda activate training_generation
python main.py --mode sample --sampling-preset ft --config configs/ft15k.yaml --run-name myrun

# QC
conda activate plasmid-qc
python main.py --mode qc --config configs/ft15k.yaml --run-name myrun

# Analysis
conda activate plasmid-qc
python main.py --mode analysis --config configs/ft15k.yaml --run-name myrun
```

## Reproducibility Notes
- Use the provided conda environments.
- Keep models and data paths in configs for full provenance.
- `manifest.json` and `config_used.yml` capture resolved inputs and outputs per run.

## License
See `LICENSE`.
