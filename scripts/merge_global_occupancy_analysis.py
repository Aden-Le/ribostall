#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(processName)s  %(message)s",
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge per-site global occupancy analysis CSVs (E/P/A) into "
                    "a single CSV per analysis with a 'site' column prepended. "
                    "Output mirrors the layout consumed by the stall_sites R plotting scripts."
    )
    p.add_argument("--analysis-dir", default="results/global_occupancy/analysis",
                   help="Directory containing per-site subfolders (E/, P/, A/) of analysis CSVs.")
    p.add_argument("--out-dir", default="results/global_occupancy/analysis_corrected",
                   help="Directory to write merged CSVs (one per analysis, with a 'site' column).")
    p.add_argument("--sites", nargs="+", default=["E", "P", "A"],
                   help="Site folder names under --analysis-dir to merge, in concatenation order.")
    return p.parse_args()


def discover_basenames(analysis_dir: Path, sites: list[str]) -> list[str]:
    site_dirs = [analysis_dir / s for s in sites]
    for d in site_dirs:
        if not d.is_dir():
            raise FileNotFoundError(f"Expected site directory not found: {d}")
    file_sets = [{p.name for p in d.iterdir() if p.is_file() and p.suffix == ".csv"}
                 for d in site_dirs]
    common = sorted(set.intersection(*file_sets))
    only_in_one = set.union(*file_sets) - set.intersection(*file_sets)
    if only_in_one:
        logging.warning(
            f"Skipping {len(only_in_one)} CSV(s) not present in all site folders: "
            f"{sorted(only_in_one)}"
        )
    return common


def merge_one(analysis_dir: Path, sites: list[str], basename: str, out_dir: Path):
    frames = []
    for s in sites:
        df = pd.read_csv(analysis_dir / s / basename)
        df.insert(0, "site", s)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    out_path = out_dir / basename
    merged.to_csv(out_path, index=False)
    logging.info(f"Wrote {out_path}  ({len(merged)} rows from {len(sites)} sites)")


def main():
    args = parse_args()

    analysis_dir = Path(args.analysis_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # basenames are the CSV filenames (without path) that are present in all site subfolders. 
    # We only merge those that are present in all sites to avoid confusion about missing data.
    basenames = discover_basenames(analysis_dir, args.sites)
    logging.info(
        "Found %d analysis CSV(s) shared across all sites:\n%s",
        len(basenames),
        "\n".join(f"  - {bn}" for bn in sorted(basenames)),
    )

    for bn in basenames:
        merge_one(analysis_dir, args.sites, bn, out_dir)

    logging.info(f"Done. Merged {len(basenames)} file(s) into {out_dir}")


if __name__ == "__main__":
    main()
