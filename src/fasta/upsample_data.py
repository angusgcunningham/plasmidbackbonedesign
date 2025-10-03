#!/usr/bin/env python3
# upsample_rotate_plasmids_resume_safe_with_metadata.py
from pathlib import Path
from Bio import SeqIO
import math, hashlib, random, os, tempfile, csv, re

# ── Settings ───────────────────────────────────────────────────────────
IN_DIR              = Path("/cs/student/projects1/aibh/2024/acunning/Projects/Data/dataset/fasta")
OUT_DIR             = Path("/cs/student/projects1/aibh/2024/acunning/Projects/Data/dataset/upsampled_fasta")
METADATA_PATH       = METADATA_PATH = Path("/cs/student/projects1/aibh/2024/acunning/Projects/Data/Ecoli_plasmid_ft.tsv")
MIN_LEN             = 0
UPSAMPLE_TARGET_BP  = 30000
UPSAMPLE_CUTOFF_BP  = 30000
MIN_ROTATIONS       = 1
MAX_ROTATIONS       = 15
EVEN_SPACING        = True
JITTER_FRAC         = 0.03
ADD_REVERSE_COMP    = False
DEDUP_ROTATIONS     = True
RANDOM_SEED_GLOBAL  = 42

# Column name guesses
META_ID_COLS       = ["Plasmid_ID"]
META_COMPLETE_COLS = ["Completeness"]

random.seed(RANDOM_SEED_GLOBAL)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────
def sanitize_id(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-", ".") else "_" for c in s)[:120]

def rotate(seq: str, offset: int) -> str:
    n = len(seq)
    if n == 0: return seq
    k = offset % n
    return seq[k:] + seq[:k]

def rc(seq: str) -> str:
    tbl = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(tbl)[::-1]

def md5(s: str) -> str:
    return hashlib.md5(s.encode("ascii", errors="ignore")).hexdigest()

def write_fasta_atomic(path: Path, header: str, seq: str):
    path_parent = path.parent
    path_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path_parent, delete=False) as tmp:
        tmp_name = tmp.name
        tmp.write(f">{header}\n")
        for i in range(0, len(seq), 80):
            tmp.write(seq[i:i+80] + "\n")
    os.replace(tmp_name, path)

# Treat only 'incomplete' as False; everything else counts as complete
def parse_boolish_complete(v):
    return str(v).strip().lower() != "incomplete"


def sniff_delimiter(path: Path) -> str:
    with open(path, "r", newline="") as fh:
        sample = fh.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            return dialect.delimiter
        except Exception:
            return ","  # fallback

def load_metadata_ids(path: Path):
    """Load sets of complete and incomplete IDs from CSV/TSV."""
    if not path or not path.exists():
        print("No metadata provided or path not found; treating all as complete.")
        return set(), set(), None, None

    delim = sniff_delimiter(path)
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        hdrs = [h.strip() for h in reader.fieldnames or []]
        # Pick columns
        id_col  = next((c for c in META_ID_COLS if c in hdrs), None)
        cmp_col = next((c for c in META_COMPLETE_COLS if c in hdrs), None)
        if not id_col or not cmp_col:
            print(f"Metadata header not found. Headers={hdrs}")
            print("Expected an ID column (e.g., plasmid_id/accession) and a completeness column.")
            return set(), set(), None, None

        complete_ids, incomplete_ids = set(), set()
        for row in reader:
            rid = (row.get(id_col) or "").strip()
            if not rid:
                continue
            flag = parse_boolish_complete(row.get(cmp_col))
            if flag is True:
                complete_ids.add(rid)
            elif flag is False:
                incomplete_ids.add(rid)
        print(f"Loaded metadata: {len(complete_ids)} complete, {len(incomplete_ids)} incomplete "
              f"(id_col='{id_col}', completeness_col='{cmp_col}', delim='{delim}')")
        return complete_ids, incomplete_ids, id_col, cmp_col

def accession_like_tokens(s: str):
    """Extract plausible accession tokens like CP088706.1, as well as alnum tokens."""
    toks = set()
    if not s:
        return toks
    # Accession patterns (very rough): two letters + digits + .digit
    for m in re.finditer(r"[A-Z]{2}\d+\.\d+", s):
        toks.add(m.group(0))
    # Also split on _-. and keep alnum/dot chunks
    for t in re.split(r"[_\-.|]", s):
        t = t.strip()
        if t and len(t) > 2:
            toks.add(t)
    return toks

def classify_by_metadata(filename_stem: str, rec_id: str, complete_ids: set, incomplete_ids: set):
    """Return 'complete' | 'incomplete' | 'unknown' based on metadata ID sets."""
    if complete_ids is None or incomplete_ids is None:
        return "unknown"
    # Candidate tokens
    cands = set()
    cands.add(filename_stem)
    cands.add(rec_id)
    # Common prefixes to strip (adjust as needed)
    for pref in ("GenBank_", "ADDGENE_", "ENA_", "NCBI_", "RefSeq_", "GCF_", "GCA_"):
        if filename_stem.startswith(pref):
            cands.add(filename_stem[len(pref):])
    # Tokenize
    cands |= accession_like_tokens(filename_stem)
    cands |= accession_like_tokens(rec_id)

    # Exact matches preferred
    for c in list(cands):
        if c in incomplete_ids:
            return "incomplete"
    for c in list(cands):
        if c in complete_ids:
            return "complete"

    # If no exact match, try suffix after first underscore (e.g., GenBank_CP088706.1 -> CP088706.1)
    parts = filename_stem.split("_", 1)
    if len(parts) == 2:
        suf = parts[1]
        if suf in incomplete_ids: return "incomplete"
        if suf in complete_ids:   return "complete"

    return "unknown"

def pick_offsets(nbp: int, n_rot: int, rng: random.Random) -> list[int]:
    if n_rot <= 0 or nbp <= 0:
        return []
    if EVEN_SPACING:
        step = nbp / (n_rot + 1)
        offs = [int(round((i+1) * step)) for i in range(n_rot)]
    else:
        offs = [rng.randrange(0, nbp) for _ in range(n_rot)]
    if JITTER_FRAC > 0:
        jitter = max(1, int(nbp * JITTER_FRAC))
        offs = [(o + rng.randint(-jitter, jitter)) % nbp for o in offs]
    return sorted(set(offs))

# ── Load metadata ──────────────────────────────────────────────────────
complete_ids, incomplete_ids, META_ID_USED, META_CMP_USED = load_metadata_ids(METADATA_PATH)

# ── Walk inputs ────────────────────────────────────────────────────────
inputs = sorted(list(IN_DIR.glob("*.fa")) + list(IN_DIR.glob("*.fasta")))
print(f"Found {len(inputs)} FASTA files in {IN_DIR.resolve()}")

total_written = 0
for fpath in inputs:
    done_marker = OUT_DIR / f"{fpath.stem}.__DONE__"
    if done_marker.exists():
        print(f"Skipping {fpath.name} (already done)")
        continue

    records = list(SeqIO.parse(str(fpath), "fasta"))
    if not records:
        done_marker.touch()
        continue

    base = fpath.stem
    print(f"→ {fpath.name}: {len(records)} record(s)")

    seen_hashes = set()

    try:
        for idx, rec in enumerate(records):
            seq = str(rec.seq).upper().replace(" ", "")
            if len(seq) < MIN_LEN:
                continue

            rec_id = sanitize_id(rec.id or f"{base}_{idx:04d}")

            # Determine completeness from metadata (only affects rotations)
            cls = classify_by_metadata(base, rec_id, complete_ids, incomplete_ids)
            is_incomplete = (cls == "incomplete")

            # Deterministic RNG per record
            seed_src = f"{base}||{rec_id}||{len(seq)}"
            local_seed = int(md5(seed_src), 16) % (2**32)
            rng = random.Random(local_seed)

            # Always write original once (resume-safe)
            out_name = f"{base}__{rec_id}__orig.fasta"
            out_path = OUT_DIR / out_name
            if not out_path.exists():
                if (not DEDUP_ROTATIONS) or (md5(seq) not in seen_hashes):
                    write_fasta_atomic(out_path, f"{rec_id}|source={base}|orig|len={len(seq)}", seq)
                    total_written += 1
                    if DEDUP_ROTATIONS:
                        seen_hashes.add(md5(seq))

            L = len(seq)

            # Decide how many rotations to add (skip if marked incomplete)
            if not is_incomplete and L < UPSAMPLE_CUTOFF_BP:
                raw_copies = math.ceil(UPSAMPLE_TARGET_BP / max(L, 1))
                n_rot = max(MIN_ROTATIONS, raw_copies - 1)
                n_rot = min(n_rot, MAX_ROTATIONS)
            else:
                n_rot = 0
                if is_incomplete:
                    print(f"   · {rec_id}: marked INCOMPLETE in metadata → no rotations")

            # Generate rotated variants
            for j, off in enumerate(pick_offsets(L, n_rot, rng)):
                rot = rotate(seq, off)
                rot_path = OUT_DIR / f"{base}__{rec_id}__rot{j}_off{off}.fasta"
                if not rot_path.exists():
                    if (not DEDUP_ROTATIONS) or (md5(rot) not in seen_hashes):
                        write_fasta_atomic(rot_path,
                                           f"{rec_id}|source={base}|rot={j}|off={off}|len={len(rot)}",
                                           rot)
                        total_written += 1
                        if DEDUP_ROTATIONS:
                            seen_hashes.add(md5(rot))

                if ADD_REVERSE_COMP:
                    rot_rc = rc(rot)
                    rot_rc_path = OUT_DIR / f"{base}__{rec_id}__rot{j}_off{off}_rc.fasta"
                    if not rot_rc_path.exists():
                        if (not DEDUP_ROTATIONS) or (md5(rot_rc) not in seen_hashes):
                            write_fasta_atomic(rot_rc_path,
                                               f"{rec_id}|source={base}|rot={j}|off={off}|rc|len={len(rot_rc)}",
                                               rot_rc)
                            total_written += 1
                            if DEDUP_ROTATIONS:
                                seen_hashes.add(md5(rot_rc))

        done_marker.touch()

    except OSError as e:
        print(f"[ERROR] While processing {fpath.name}: {e}")
        break  # stop so you can free space; rerun to resume

print(f"Done (this run). Wrote {total_written} FASTA files to {OUT_DIR.resolve()}")
