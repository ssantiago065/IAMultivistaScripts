"""All-subtype dataset for plan steps 12 and 9 (round 3).

Unlike dataset_preparing_expA.py, which reads each class folder in path order and stops
early (so DoS is one SYN Flood capture, Spoofing has no DNS spoofing, Mirai 14 of 29
captures), this script draws the per-class cap from EVERY capture:
  - the class quota (50,000 flows, or every flow if the class has fewer) is split equally
    among its subtypes (sub-folders; a class without sub-folders is one subtype), and each
    subtype's share equally among its capture files (capped by the file size, the surplus
    is redistributed);
  - each file is read in chunks and sampled uniformly at random without replacement.
Two bookkeeping columns are kept for the splits and dropped before training:
  capture   file name of the capture
  t_rank    position of the flow in its capture by Timestamp (0 = first, 1 = last)

    python Code/dataset_preparing_allsubtypes.py --flow-dir H:/Dataset/cleaned_dataset_flowbased \
        --out H:/Dataset/dataset_allsub_8clases.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_SAMPLES_PER_CLASS = 50_000
CHUNK_SIZE = 200_000
RANDOM_STATE = 42
VALID_CLASSES = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "BruteForce", "Web-Based"]
DROP = ["Flow ID", "Src IP", "Dst IP"]


def file_rows(report: Path) -> dict[str, int]:
    rep = pd.read_csv(report)
    return dict(zip(rep["Archivo"], rep["Filas Finales"]))


def fair_split(total: int, caps: list[int]) -> list[int]:
    """Splits `total` as equally as possible among bins with capacities `caps`."""
    alloc = [0] * len(caps)
    left, open_bins = total, [i for i, c in enumerate(caps) if c > 0]
    while left > 0 and open_bins:
        share = max(left // len(open_bins), 1)
        for i in list(open_bins):
            take = min(share, caps[i] - alloc[i], left)
            alloc[i] += take
            left -= take
            if alloc[i] == caps[i]:
                open_bins.remove(i)
            if left == 0:
                break
    return alloc


def class_quotas(class_dir: Path, rows: dict[str, int]) -> dict[Path, int]:
    files = sorted(class_dir.rglob("*.csv"), key=lambda p: str(p).lower())
    files = [f for f in files if rows.get(f.name, 0) > 0]
    groups: dict[Path, list[Path]] = {}
    for f in files:
        groups.setdefault(f.parent, []).append(f)
    subtypes = sorted(groups, key=str)
    sub_caps = [sum(rows[f.name] for f in groups[s]) for s in subtypes]
    sub_alloc = fair_split(min(TARGET_SAMPLES_PER_CLASS, sum(sub_caps)), sub_caps)
    quotas = {}
    for s, n in zip(subtypes, sub_alloc):
        fs = groups[s]
        for f, q in zip(fs, fair_split(n, [rows[f.name] for f in fs])):
            quotas[f] = q
    return quotas


def sample_file(path: Path, quota: int, n_rows: int, rng: np.random.RandomState) -> pd.DataFrame:
    keep = np.zeros(n_rows, dtype=bool)
    keep[rng.choice(n_rows, size=quota, replace=False)] = True
    parts, start = [], 0
    for chunk in pd.read_csv(path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        n = len(chunk)
        mask = keep[start:start + n] if start + n <= n_rows else np.pad(
            keep[start:n_rows], (0, max(0, start + n - n_rows)))
        parts.append(chunk[mask[:n]])
        start += n
    df = pd.concat(parts, ignore_index=True)
    ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    df["t_rank"] = ts.rank(method="first").fillna(0).to_numpy() / max(len(df), 1)
    df["capture"] = f"{path.parent.name}/{path.name}"
    return df.drop(columns=[c for c in DROP + ["Timestamp"] if c in df.columns])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flow-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    rows = file_rows(args.flow_dir / "cleaning_report_flowbased.csv")
    rng = np.random.RandomState(RANDOM_STATE)
    parts = []
    for cls in VALID_CLASSES:
        quotas = class_quotas(args.flow_dir / cls, rows)
        for f, q in quotas.items():
            if q:
                df = sample_file(f, q, rows[f.name], rng)
                df["Label"] = cls
                parts.append(df)
        print(f"{cls}: {sum(quotas.values())} flows from {sum(1 for q in quotas.values() if q)} captures",
              flush=True)
    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)
    tmp = args.out.with_suffix(".partial")
    out.to_csv(tmp, index=False)
    tmp.replace(args.out)                     # the batch runner waits for the final name
    print(f"Wrote {args.out} {out.shape}")
    print(out.groupby("Label")["capture"].nunique())


if __name__ == "__main__":
    main()
