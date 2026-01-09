#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.utils.config import default_config, load_config_with_defaults


def _timestamp_run_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_config_path(
    config_path: Optional[str],
    preset: Optional[str],
    sampling_preset: Optional[str],
    mode: str,
) -> Optional[Path]:
    if config_path:
        return Path(config_path)
    if mode == "sample" and sampling_preset:
        return Path("configs") / f"sample_{sampling_preset}.yaml"
    if preset:
        return Path("configs") / f"{preset}.yaml"
    return None


def _rewrite_run_paths(cfg: Dict[str, Any], run_dir: Path) -> None:
    paths = cfg.get("paths", {})
    keys = [
        "model_output_dir",
        "generation_output_dir",
        "qc_output_dir",
        "analysis_output_dir",
    ]
    for key in keys:
        val = paths.get(key)
        if not isinstance(val, str):
            continue
        p = Path(val)
        if len(p.parts) >= 2 and p.parts[0] == "runs":
            paths[key] = str(run_dir.joinpath(*p.parts[2:]))


def _init_logger(log_path: Path):
    import logging

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


def _log_event(logger, event: Dict[str, Any]) -> None:
    logger.info(json.dumps(event, sort_keys=True))


def _mode_summary(mode: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = cfg.get("paths", {})
    summary = {
        "mode": mode,
        "inputs": {
            "dataset_input": paths.get("dataset_input"),
            "tokenizer_json": paths.get("tokenizer_json"),
            "base_model": paths.get("base_model"),
        },
        "outputs": {
            "model_output_dir": paths.get("model_output_dir"),
            "generation_output_dir": paths.get("generation_output_dir"),
            "qc_output_dir": paths.get("qc_output_dir"),
            "analysis_output_dir": paths.get("analysis_output_dir"),
        },
    }
    return summary


def _run_cmd(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    import subprocess

    subprocess.run(cmd, check=True)


def _script_path(*parts: str) -> str:
    return str(Path(*parts))

def _choose_generation_input(outdir: Path) -> Path:
    """
    Prefer a directory of per-sequence FASTAs if present.
    Fall back to generations.fasta if it exists as a real file.
    Raise a helpful error otherwise.
    """
    gen_dir = outdir / "generations"

    # Prefer per-sequence FASTAs (real files only; ignore broken symlinks)
    per_seq_fastas = sorted(
        p for p in gen_dir.glob("*.fasta")
        if p.name != "generations.fasta" and p.is_file()
    )
    if per_seq_fastas:
        return gen_dir  # directory input (qc_oriv_arg2.py supports this)

    # Fall back to single combined fasta if it exists and is readable
    combined = gen_dir / "generations.fasta"
    if combined.is_file():
        return combined

    # Helpful error
    listing = []
    if gen_dir.exists():
        for p in sorted(gen_dir.iterdir()):
            try:
                kind = "file" if p.is_file() else "dir" if p.is_dir() else "other"
            except OSError:
                kind = "unreadable"
            listing.append(f"{p.name}\t{kind}")
    msg = (
        f"Could not find generation FASTA input.\n"
        f"Tried:\n"
        f"  - directory with per-seq FASTAs: {gen_dir}\n"
        f"  - combined FASTA: {combined}\n"
        f"Directory listing ({gen_dir}):\n  " + "\n  ".join(listing)
    )
    raise FileNotFoundError(msg)

def main() -> int:
    ap = argparse.ArgumentParser(description="Plasmid pipeline entrypoint.")
    ap.add_argument("--mode", required=True, choices=["train", "sample", "qc", "analysis", "run"])
    ap.add_argument("--preset", choices=["ft15k", "ft35k"])
    ap.add_argument("--sampling-preset", choices=["base", "ft"])
    ap.add_argument("--config", help="Path to YAML config.")
    ap.add_argument("--run-name", default=_timestamp_run_name())
    ap.add_argument("--outdir", help="Override run output directory.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_name = args.run_name
    outdir = Path(args.outdir) if args.outdir else Path("runs") / run_name

    cfg_path = _resolve_config_path(args.config, args.preset, args.sampling_preset, args.mode)
    if cfg_path and cfg_path.exists():
        cfg = load_config_with_defaults(str(cfg_path))
    else:
        if cfg_path:
            print(f"Warning: config not found: {cfg_path}", file=sys.stderr)
        cfg = default_config()

    cfg["run"] = cfg.get("run", {})
    cfg["run"]["name"] = run_name
    cfg["run"]["dir"] = str(outdir)
    _rewrite_run_paths(cfg, outdir)

    for subdir in ["models", "generations", "qc", "analysis", "logs"]:
        (outdir / subdir).mkdir(parents=True, exist_ok=True)

    log_dir = outdir / "logs"
    log_path = log_dir / "pipeline.log"
    logger = _init_logger(log_path)

    summary = _mode_summary(args.mode, cfg)
    summary.update(
        {
            "run": cfg.get("run"),
            "config_path": str(cfg_path) if cfg_path else None,
            "outdir": str(outdir),
            "force": bool(args.force),
            "dry_run": bool(args.dry_run),
        }
    )

    _log_event(logger, {"event": "dirs_initialized", "run_dir": str(outdir)})
    _log_event(logger, {"event": "start", **summary})

    manifest_path = outdir / "manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    config_used_path = outdir / "config_used.yml"
    with open(config_used_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    print(json.dumps(summary, indent=2, sort_keys=True))

    paths = cfg.get("paths", {})
    generation = cfg.get("generation", {})
    qc = cfg.get("qc", {})
    analysis = cfg.get("analysis", {})

    if args.mode == "train":
        preset = args.preset or "ft15k"
        trainer = "15kplasmidgpt.py" if preset == "ft15k" else "35kplasmidgpt.py"
        cmd = [
            sys.executable,
            _script_path("src", "training", trainer),
            "--fasta-dir", paths.get("dataset_input", ""),
            "--tokenizer-json", paths.get("tokenizer_json", ""),
            "--model-path", paths.get("base_model", ""),
            "--run-name", run_name,
            "--preset", preset,
        ]
        _run_cmd(cmd, args.dry_run)
        return 0

    if args.mode == "sample":
        sampling_preset = args.sampling_preset or "base"
        script = "generate_base.py" if sampling_preset == "base" else "generate_ft.py"
        cmd = [
            sys.executable,
            _script_path("src", "training", script),
            "--run-name", run_name,
            "--preset", sampling_preset,
            "--tokenizer-json", paths.get("tokenizer_json", ""),
            "--output-dir", str(outdir / "generations"),
            "--generations-fasta", str(outdir / "generations" / "generations.fasta"),
            "--metadata-csv", str(outdir / "generations" / "generations_metadata.csv"),
            "--num-samples", str(generation.get("num_samples", 100)),
        ]
        if sampling_preset == "base":
            cmd += ["--base-pt", paths.get("base_model", "")]
        else:
            cmd += ["--ft-model-dir", paths.get("model_output_dir", "")]
        _run_cmd(cmd, args.dry_run)
        return 0

    if args.mode == "qc":
        gen_in = _choose_generation_input(outdir)
        _log_event(logger, {"event": "qc_input_selected", "qc_input": str(gen_in)})
        print(f"[qc] Using input: {gen_in}")

        cmd = [
            sys.executable,
            _script_path("src", "qc", "qc_oriv_arg2.py"),
            "--run-name", run_name,
            "--in", str(gen_in),
            "--outdir", str(outdir / "qc"),
            "--oridb_prefix", qc.get("oridb_prefix", ""),
        ]
        if qc.get("oridb_ref"):
            cmd += ["--oridb_ref", qc.get("oridb_ref")]
        cmd += [
            "--min_pident", str(qc.get("min_pident", 85.0)),
            "--min_scovs", str(qc.get("min_scovs", 80.0)),
            "--min_len", str(qc.get("min_len", 100)),
            "--threads", str(qc.get("threads", 1)),
        ]
        if qc.get("skip_prodigal"):
            cmd += ["--skip_prodigal"]
        _run_cmd(cmd, args.dry_run)

        repeats_cmd = [
            sys.executable,
            _script_path("src", "qc", "repeats2.py"),
            str(gen_in),
            "--run-name", run_name,
            "--circular",
        ]
        _run_cmd(repeats_cmd, args.dry_run)

        filt = qc.get("filter", {})
        filter_cmd = [
            sys.executable,
            _script_path("src", "qc", "filter_qc_two_stage2.py"),
            "--run-name", run_name,
            "--qc_out", str(outdir / "qc"),
            "--out_pass", str(outdir / "qc" / "passed.csv"),
            "--out_fail", str(outdir / "qc" / "failed.csv"),
            "--ori_low_identity", str(filt.get("ori_low_identity", 85.0)),
            "--ori_low_cov", str(filt.get("ori_low_cov", 80.0)),
            "--amr_low_identity", str(filt.get("amr_low_identity", 85.0)),
            "--amr_low_cov", str(filt.get("amr_low_cov", 80.0)),
            "--ori_low_count_min", str(filt.get("ori_low_count_min", 1)),
            "--ori_low_count_max", str(filt.get("ori_low_count_max", 1)),
            "--amr_low_count_min", str(filt.get("amr_low_count_min", 1)),
            "--amr_low_count_max", str(filt.get("amr_low_count_max", 1)),
            "--ori_strict_identity", str(filt.get("ori_strict_identity", 99.0)),
            "--ori_strict_cov", str(filt.get("ori_strict_cov", 99.0)),
            "--amr_strict_identity", str(filt.get("amr_strict_identity", 100.0)),
            "--amr_strict_cov", str(filt.get("amr_strict_cov", 100.0)),
        ]
        _run_cmd(filter_cmd, args.dry_run)
        return 0

    if args.mode == "analysis":
        sim = analysis.get("similarity", {})
        cmd = [
            sys.executable,
            _script_path("src", "analysis", "similarity_score.py"),
            "--run-name", run_name,
            "--input", str(outdir / "generations"),
            "--db-path", sim.get("db_path", ""),
            "--out-csv", str(outdir / "analysis" / "analysis_summary.csv"),
        ]
        _run_cmd(cmd, args.dry_run)
        return 0

    if args.mode == "run":
        base_cmd = [sys.executable, _script_path("main.py"), "--run-name", run_name]
        if args.config:
            base_cmd += ["--config", args.config]
        run_cmds = [
            base_cmd + ["--mode", "train", "--preset", args.preset or "ft15k"],
            base_cmd + ["--mode", "sample", "--sampling-preset", args.sampling_preset or "ft"],
            base_cmd + ["--mode", "qc"],
            base_cmd + ["--mode", "analysis"],
        ]
        for cmd in run_cmds:
            _run_cmd(cmd, args.dry_run)
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
