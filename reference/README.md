# reference/ — APPRIS transcriptome reference FASTA files

*Principal-isoform reference transcriptomes providing the CDS sequences that anchor E/P/A codon annotation and motif-background computation.*

> **[ribostall](../README.md)** › reference

---

## Overview

A reference FASTA in `ribostall` supplies, per transcript, the coding sequence (CDS) that the pipeline reads codons from. Given a footprint aligned to a CDS position, the pipeline uses these sequences to decode the codons occupying the ribosome's E, P, and A sites, to translate them to amino acids, and to compute the transcriptome-wide **background frequencies** against which observed occupancy/stall enrichment is measured. Without the reference, coverage is just numbers over positions; the reference is what turns a position into a codon and an amino acid.

## Files

| File | Organism | Storage | Approx. size |
|---|---|---|---|
| `appris_celegans_v1_selected_new.fa` | *C. elegans* | uncompressed FASTA | ~32 MB |
| `appris_mouse_v2_selected.fa.gz` | mouse | gzip-compressed FASTA | ~21 MB |

Both are **selected**, principal-isoform transcriptomes: for each gene, APPRIS's principal isoform is kept, so there is a single canonical CDS per gene rather than every annotated splice variant. This keeps codon-position bookkeeping unambiguous.

## Gzip handling

The FASTA reader in the pipeline (`ribostall/fasta.py`) **auto-detects gzip** by content, so a `.fa` and a `.fa.gz` are interchangeable — the mouse reference can stay compressed and the *C. elegans* one uncompressed with no change to how they are passed in. There is no need to decompress by hand.

## Which scripts consume it

Passed via the `--reference` argument to the stall-calling and occupancy scripts in [`../scripts/`](../scripts/README.md) — the `stall_sites_*` callers and `global_codon_occ.py` — where it is loaded once and used to build the per-transcript CDS lookup for codon annotation and background frequencies.

## Provenance

Both files are APPRIS principal-isoform selected transcriptomes. The mouse reference (`appris_mouse_v2_selected.fa.gz`) matches the transcriptome used by **ribograph_sampledata**, so it pairs directly with the sample `.ribo` file in [`../all_ribo_file/`](../all_ribo_file/README.md).

## See also

- [`../scripts/README.md`](../scripts/README.md) — the pipeline scripts that take `--reference`.
- [`../README.md`](../README.md) — repository root.
