# plannotate_wrapper.py

import os
from plannotate.annotate import annotate
from plannotate.resources import get_seq_record
from plannotate.bokeh_plot import get_bokeh
from Bio import SeqIO

def run_plannotate_sequence(
    seq_or_filepath: str,
    output_html: str = None,
    is_detailed: bool = False,
    linear: bool = False,
    yaml_file: str = None,
):
    """
    Annotate an engineered plasmid (either by sequence string or by file path).

    Parameters
    ----------
    seq_or_filepath : str
        - If this is plain DNA (A/C/G/T…), pLannotate treats it as a raw sequence.
        - If this is a path ending in .fa/.fasta/.gbk/.gb (etc.), pLannotate will read that file.
    output_html : str, optional
        Path to write an HTML plasmid map. If None, no HTML file is generated.
    is_detailed : bool
        Use the more‐detailed (but slower) search algorithm.
    linear : bool
        Treat the input DNA as linear (default is circular plasmid).
    yaml_file : str, optional
        Path to a custom YAML database file; if not provided (None), pLannotate’s built‐in DB is used.

    Returns
    -------
    hits_df : pandas.DataFrame
        A DataFrame of annotation “hits” (features found on the plasmid).
    seq_record : Bio.SeqRecord.SeqRecord
        A Biopython SeqRecord object with the annotated features.
    bio_plot : bokeh.models.plots.Plot
        A Bokeh figure object representing the circular (or linear) plasmid map.
    """

    # 1) Load sequence (either raw string or a FASTA/GenBank file)
    if os.path.exists(seq_or_filepath):
        file_ext = os.path.splitext(seq_or_filepath)[1].lower()
        if file_ext in {".fa", ".fasta"}:
            seq_record = SeqIO.read(seq_or_filepath, "fasta")
            seq_str = str(seq_record.seq)
        elif file_ext in {".gb", ".gbk"}:
            seq_record = SeqIO.read(seq_or_filepath, "genbank")
            seq_str = str(seq_record.seq)
        else:
            raise ValueError(f"Unsupported file extension: {file_ext}")
    else:
        seq_str = seq_or_filepath
        seq_record = None

    # 2) Call pLannotate.annotate(), passing yaml_file ONLY if it’s not None
    if yaml_file:
        hits_df = annotate(
            seq_str,
            is_detailed=is_detailed,
            linear=linear,
            yaml_file=yaml_file,
        )
    else:
        hits_df = annotate(
            seq_str,
            is_detailed=is_detailed,
            linear=linear,
        )

    # 3) If we only had a raw string, convert hits_df → SeqRecord now
    if seq_record is None:
        seq_record = get_seq_record(hits_df, seq_str)

    # 4) Generate the Bokeh figure (circular or linear)
    bio_plot = get_bokeh(hits_df, linear=linear)

    # 5) If the user wants an HTML file, export it
    if output_html:
        from bokeh.embed import file_html
        from bokeh.resources import CDN

        html_content = file_html(bio_plot, CDN, "pLannotate Map")
        with open(output_html, "w") as f:
            f.write(html_content)

    return hits_df, seq_record, bio_plot