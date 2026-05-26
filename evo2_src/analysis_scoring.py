#!/usr/bin/env python3
import argparse
import csv
import statistics
from pathlib import Path


DEFAULT_GROUPS = {
    "atg": Path("evo2_src/results/evo2_scores/atg"),
    "gfp": Path("evo2_src/results/evo2_scores/gfp"),
    "randtok": Path("evo2_src/results/evo2_scores/randtok"),
}


def read_score_rows(csv_path: Path) -> list[tuple[str, int, float]]:
    rows: list[tuple[str, int, float]] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return rows
        if "mean_logprob_per_token" not in reader.fieldnames:
            raise ValueError(f"{csv_path} is missing mean_logprob_per_token")
        for row in reader:
            if not row:
                continue
            rows.append(
                (
                    row.get("id", csv_path.stem),
                    int(float(row.get("length_bp", 0) or 0)),
                    float(row["mean_logprob_per_token"]),
                )
            )
    return rows


def load_group_scores(group_dir: Path) -> list[tuple[str, int, float]]:
    if not group_dir.exists():
        raise FileNotFoundError(f"Missing score directory: {group_dir}")

    combined = group_dir / "generations.csv"
    if combined.is_file():
        return read_score_rows(combined)

    scores: list[tuple[str, int, float]] = []
    for csv_path in sorted(group_dir.glob("*.csv")):
        scores.extend(read_score_rows(csv_path))
    return scores


def summarize_group(name: str, rows: list[tuple[str, int, float]]) -> dict[str, float]:
    scores = [row[2] for row in rows]
    lengths = [row[1] for row in rows]
    if not scores:
        raise ValueError(f"No scores found for group: {name}")

    return {
        "group": name,
        "n": len(scores),
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
        "mean_length_bp": statistics.fmean(lengths),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Summarize Evo2 mean log-probability scores across ATG, GFP, and randtok result folders."
    )
    ap.add_argument(
        "--root",
        default="evo2_src/results/evo2_scores",
        help="Root directory containing atg/, gfp/, and randtok/ folders.",
    )
    ap.add_argument(
        "--out-csv",
        default=None,
        help="Optional CSV path for the summary table.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    groups = {name: root / path.name for name, path in DEFAULT_GROUPS.items()}

    summaries = []
    for group_name, group_dir in groups.items():
        rows = load_group_scores(group_dir)
        summary = summarize_group(group_name, rows)
        summary["path"] = str(group_dir)
        summaries.append(summary)

    best_group = max(summaries, key=lambda item: item["mean"])

    print("Evo2 score summary")
    print("==================")
    for summary in summaries:
        print(
            f"{summary['group']:8s} n={summary['n']:4d} "
            f"mean={summary['mean']:.6f} median={summary['median']:.6f} "
            f"stdev={summary['stdev']:.6f} min={summary['min']:.6f} max={summary['max']:.6f} "
            f"mean_length_bp={summary['mean_length_bp']:.1f}"
        )

    print()
    print(f"Best by mean score: {best_group['group']} ({best_group['mean']:.6f})")

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["group", "n", "mean", "median", "stdev", "min", "max", "mean_length_bp", "path"],
            )
            writer.writeheader()
            writer.writerows(summaries)
        print(f"\nSaved summary table to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
