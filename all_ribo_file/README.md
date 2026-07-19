# all_ribo_file/ — RiboFlow `.ribo` coverage file(s)

*A RiboFlow-generated `.ribo` HDF5 container holding RPM-normalized ribosome-profiling coverage plus CDS metadata — the primary read-coverage input to the pipeline.*

> **[ribostall](../README.md)** › all_ribo_file

---

## Overview

`mouse_all.ribo` (~30 MB) is a `.ribo` file produced by **RiboFlow** (see <https://github.com/ribosomeprofiling/riboflow>), the standard preprocessing workflow for ribosome-profiling data. The `.ribo` format is an HDF5 container: a single self-describing binary file that stores, per experiment × read-length × transcript, the **RPM-normalized read coverage** along each transcript, together with the CDS metadata (transcript boundaries, CDS start/stop, region annotations) needed to place a footprint relative to the coding sequence. Bundling every experiment and length into one indexed file is what lets the pipeline pull per-transcript coverage efficiently without re-parsing BAMs.

## Files

| File | Description | Approx. size |
|---|---|---|
| `mouse_all.ribo` | RiboFlow `.ribo` HDF5 — all mouse experiments, RPM-normalized coverage + CDS metadata | ~30 MB |

*(This is a multi-MB binary; inspect it with `ribopy` rather than opening it as text.)*

## How the pipeline reads it

The file is passed via the `--ribo` argument and read through the **`ribopy`** library. The Step-1 coverage extractor `adj_coverage.py` opens it to pull CDS-aligned coverage into the pipeline's pickle format, and the downstream stall-calling and occupancy scripts in [`../scripts/`](../scripts/README.md) consume it (in concert with the transcriptome reference) to annotate E/P/A codons and compute occupancy. Which experiment / read-length selection is used is controlled by those scripts' own arguments.

## Provenance

The `.ribo` format and toolchain come from the ribosomeprofiling project. Sample `.ribo` files (and the matching selected transcriptome — see [`../reference/README.md`](../reference/README.md)) are distributed as **ribograph_sampledata**; `mouse_all.ribo` pairs with the `appris_mouse_v2_selected` reference.

## See also

- [`../scripts/README.md`](../scripts/README.md) — the pipeline scripts that take `--ribo` (starting with `adj_coverage.py`).
- [`../README.md`](../README.md) — repository root.
