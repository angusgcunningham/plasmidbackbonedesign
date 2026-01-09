from __future__ import annotations

import copy
import os
from typing import Any, Dict

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "Missing PyYAML dependency. Install with `pip install pyyaml` to load configs."
    ) from exc


DEFAULTS: Dict[str, Any] = {
    "run": {
        "name": "default",
        "dir": "runs/default",
    },
    "paths": {
        "dataset_input": "data/train_fasta",
        "tokenizer_json": "models/tokenizer.json",
        "base_model": "models/base_model.pt",
        "model_output_dir": "runs/default/models/model",
        "generation_output_dir": "runs/default/generations",
        "qc_output_dir": "runs/default/qc",
        "analysis_output_dir": "runs/default/analysis",
    },
    "training": {
        "ctx": 2048,
        "stride": None,
        "epochs": 3,
        "batch_size": 1,
        "grad_accum": 8,
        "learning_rate": 5e-5,
    },
    "generation": {
        "seed": "ATG",
        "num_samples": 100,
        "target_mean_bp": 4000,
        "target_spread_bp": 1500,
        "temperature": 1.0,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "max_new_tokens": 12000,
    },
    "qc": {
        "oridb_prefix": "data/oriVdb/oriV_db",
        "oridb_ref": "data/oriVdb/oriV_refs.fasta",
        "min_pident": 85.0,
        "min_scovs": 80.0,
        "min_len": 100,
        "threads": 4,
        "skip_prodigal": False,
        "filter": {
            "ori_low_identity": 85.0,
            "ori_low_cov": 80.0,
            "ori_low_count_min": 1,
            "ori_low_count_max": 1,
            "amr_low_identity": 85.0,
            "amr_low_cov": 80.0,
            "amr_low_count_min": 1,
            "amr_low_count_max": 1,
            "ori_strict_identity": 99.0,
            "ori_strict_cov": 99.0,
            "amr_strict_identity": 100.0,
            "amr_strict_cov": 100.0,
            "amr_strict_min": None,
            "amr_strict_all": False,
        },
    },
    "analysis": {
        "similarity": {
            "db_path": "data/plasmid_db",
            "max_target_seqs": 5,
        },
        "metadata": {
            "enabled": True,
        },
        "kmer": {
            "k": 4,
        },
        "plannotate": {
            "enabled": False,
            "yaml_file": None,
        },
    },
}


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def _expand_paths(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_paths(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj


def load_config(path: str, defaults: Dict[str, Any] | None = None) -> Dict[str, Any]:
    with open(path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    base = defaults if defaults is not None else {}
    merged = _deep_merge(base, data)
    return _expand_paths(merged)


def load_config_with_defaults(path: str) -> Dict[str, Any]:
    return load_config(path, defaults=DEFAULTS)


def default_config() -> Dict[str, Any]:
    return _expand_paths(copy.deepcopy(DEFAULTS))
