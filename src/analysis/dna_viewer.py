# env used: training_generation
"""dna_viewer.py

Usage examples:
  python dna_viewer.py \
    --tokenizer-path /path/to/addgene_trained_dna_tokenizer.json \
    --gbk-path /path/to/individual.gbk

  python dna_viewer.py \
    --tokenizer-path /path/to/addgene_trained_dna_tokenizer.json \
    --gbk-dir /path/to/folder_gbk \
    --output /path/to/linear_plots_all_sequences.pdf

If --output is not provided, the script writes next to the input file/dir.
"""
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from Bio import SeqIO
from Bio.SeqFeature import FeatureLocation, CompoundLocation
from dna_features_viewer import GraphicFeature, GraphicRecord
from transformers import PreTrainedTokenizerFast, AutoTokenizer


PALETTE = {
    "CDS": "#4C78A8",
    "gene": "#9ECAE9",
    "rep_origin": "#F58518",
    "promoter": "#54A24B",
    "terminator": "#E45756",
    "RBS": "#72B7B2",
    "misc_feature": "#B279A2",
    "oriT": "#FF9DA6",
}
KEEP_TYPES = set(PALETTE) | {"misc_feature"}


def _segments_from_location(loc):
    """Yield (start0, end0) 0-based, end-exclusive segments."""
    if isinstance(loc, CompoundLocation):
        for part in loc.parts:
            yield from _segments_from_location(part)
    elif isinstance(loc, FeatureLocation):
        yield int(loc.start), int(loc.end)
    else:
        yield int(loc.start), int(loc.end)


def _is_fragment(feature) -> bool:
    q = feature.qualifiers or {}
    v = q.get("fragment", ["False"])
    if isinstance(v, list):
        v = v[0] if v else "False"
    return str(v).strip().lower() in {"true", "1", "yes"}


def graphic_record_from_biopython_record(record):
    """Build a GraphicRecord with pLannotate-style labels and fragment styling."""
    feats = []
    for f in record.features:
        if f.type not in KEEP_TYPES:
            continue

        q = f.qualifiers or {}
        label = (
            q.get("label", [None])[0]
            or q.get("gene", [None])[0]
            or q.get("product", [None])[0]
            or f.type
        )

        for a0, b0 in _segments_from_location(f.location):
            color = PALETTE.get(f.type, "#BBBBBB")
            frag = _is_fragment(f)

            if frag:
                facecolor = "white"
                linecolor = color
                labelcolor = "black"
            else:
                facecolor = color
                linecolor = "black"
                labelcolor = "white"

            feats.append(
                GraphicFeature(
                    start=a0,
                    end=b0,
                    strand=f.strand or 0,
                    label=label,
                    color=facecolor,
                    linecolor=linecolor,
                    linewidth=(1.2 if frag else 0.5),
                    label_color=labelcolor,
                )
            )

    return GraphicRecord(sequence_length=len(record.seq), features=feats)


def clean_dna(seq: str) -> str:
    return re.sub(r"[^ACGTNRYKMSWBDHV]", "", str(seq).upper())


def make_tokens_df(seq, tokenizer_path: str) -> pd.DataFrame:
    """Return token_idx, start_bp (1-based), end_bp, length_bp."""
    seq = clean_dna(seq)
    if tokenizer_path.endswith(".json"):
        tok = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    enc = tok(
        seq,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    ids = enc["input_ids"]
    spans = enc["offset_mapping"]
    rows = []
    for i, ((a0, b0), tid) in enumerate(zip(spans, ids)):
        rows.append(
            {
                "token_idx": i,
                "token_id": int(tid),
                "start_bp": a0 + 1,
                "end_bp": b0,
                "length_bp": b0 - a0,
            }
        )
    return pd.DataFrame(rows)


def plot_token_bar(
    ax,
    seq_len: int,
    tokens_df: pd.DataFrame,
    colors=("#1f77b4", "#ff7f0e"),
    edgecolor="white",
    edgewidth=0.9,
    show_indices=False,
    index_every=5,
):
    def to0(a1):
        return a1 - 1

    y, h = 0.5, 0.8
    for _, r in tokens_df.iterrows():
        idx = int(r.token_idx)
        x = to0(int(r.start_bp))
        w = int(r.length_bp)
        ax.broken_barh(
            [(x, w)],
            (y - h / 2, h),
            facecolors=colors[idx % len(colors)],
            edgecolors=edgecolor,
            linewidth=edgewidth,
        )
        if show_indices and idx % index_every == 0:
            ax.text(
                x + w / 2,
                y + h / 2,
                str(idx),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_ylim(0, 1.8)
    ax.set_yticks([])


def build_figure(record, tokenizer_path: str):
    seq_len = len(record.seq)
    grec = graphic_record_from_biopython_record(record)
    tokens_df = make_tokens_df(record.seq, tokenizer_path)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(12, 4.5),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 1], "hspace": 0.1},
    )

    grec.plot(ax=ax1, with_ruler=False, strand_in_label_threshold=4)
    ax1.set_ylabel("Features")
    ax1.margins(y=0)
    ax1.set_zorder(3)
    ax1.set_facecolor("none")

    gc = lambda s: 100.0 * len([c for c in s if c in "GC"]) / 50
    xx = np.arange(seq_len - 50)
    yy = [gc(record.seq[x : x + 50]) for x in xx]
    ax2.fill_between(xx + 25, yy, alpha=0.3)
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel("GC(%)")
    ax2.margins(y=0)
    ax2.set_ylim(0, 100)
    ax2.set_zorder(1)
    ax2.set_facecolor("none")

    plot_token_bar(
        ax3,
        seq_len,
        tokens_df,
        colors=("#444444", "#CCCCCC"),
        show_indices=True,
        index_every=5,
    )
    ax3.set_xlabel("Position (bp)")
    ax3.set_ylabel("Tokens")
    ax3.set_zorder(1)
    ax3.set_facecolor("none")

    return fig, ax1


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot annotated plasmid maps with GC and token tracks."
    )
    p.add_argument("--tokenizer-path", required=True, help="Path to tokenizer JSON or HF dir")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--gbk-path", help="Path to a single GenBank file")
    group.add_argument("--gbk-dir", help="Directory of GenBank files to batch")
    p.add_argument("--output", help="Output PDF path (optional)")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer_path = args.tokenizer_path

    if args.gbk_path:
        gbk_path = Path(args.gbk_path)
        record = SeqIO.read(gbk_path, "genbank")
        fig, ax1 = build_figure(record, tokenizer_path)
        ax1.set_title(gbk_path.stem, loc="left", fontsize=10, pad=8)
        plt.tight_layout()

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = gbk_path.with_name(f"{gbk_path.stem}_linear_plot.pdf")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        print(f"Wrote {output_path}")
        return

    gbk_dir = Path(args.gbk_dir)
    gbk_files = sorted(gbk_dir.glob("*.gbk"))
    if not gbk_files:
        raise FileNotFoundError(f"No .gbk files found in {gbk_dir}")

    if args.output:
        output_pdf = Path(args.output)
    else:
        output_pdf = gbk_dir / "linear_plots_all_sequences.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        for gbk_path in gbk_files:
            record = SeqIO.read(gbk_path, "genbank")
            fig, ax1 = build_figure(record, tokenizer_path)
            ax1.set_title(gbk_path.stem, loc="left", fontsize=10, pad=8)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {len(gbk_files)} pages to {output_pdf}")


if __name__ == "__main__":
    main()
