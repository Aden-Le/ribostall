# test_data/ — bundled fixture CSVs for the R plotting suite

*Two small background-diff CSVs (amino-acid and codon) shipped as built-in TEST-mode fixtures so the R visualizers can be run and eyeballed without a full pipeline pass.*

> **[ribostall](../../README.md)** › [R_scripts](../README.md) › test_data

---

## Overview

These are per-timepoint **background-aware diff** result CSVs — the output family in which each experimental group's per-site enrichment is compared against its own transcriptome background, and the two groups' log2 enrichments are differenced (`delta_log2_enrichment`). They carry the exact schema the R scripts expect, one file per analysis level, and are wired in as the TEST-mode inputs of [`aa_codon_overlay.R`](../aa_codon_overlay.R) (and, by schema, usable for the `--effect-is-log2` mode of [`between_group_volcano.R`](../between_group_volcano.R)).

In `aa_codon_overlay.R` the two files are resolved via a commented-out TEST block near the top of the script: uncomment it and the script reads these copies (found relative to its own location) with no CLI arguments, so the plot code can be exercised standalone.

## Files

| File | Level | Rows drive |
|---|---|---|
| `per_timepoint_background_diff_aa.csv` | amino acid | the overlay **bars** / an AA-level volcano |
| `per_timepoint_background_diff_codon.csv` | codon | the overlay **dots** / a codon-level volcano |

## Column schema

Both files share the same layout; the only structural difference is the feature column (`amino_acid` vs `codon`).

| Column | Meaning |
|---|---|
| `site` | Ribosome site: `A`, `P`, or `E`. |
| `timepoint` | Experimental timepoint (e.g. `day_0`, `day_5`, `day_10`). |
| `amino_acid` / `codon` | The feature — one-letter amino acid, or 3-nt codon. |
| `BWM_count` | Count of the feature at the site in the BWM (treatment) group. |
| `BWM_total` | Total feature observations at the site in the BWM group. |
| `BWM_bg_freq` | Background frequency of the feature in the BWM group. |
| `control_count` | Count of the feature at the site in the control group. |
| `control_total` | Total feature observations at the site in the control group. |
| `control_bg_freq` | Background frequency of the feature in the control group. |
| `log2_enrich_BWM` | log2(observed / background) enrichment in the BWM group. |
| `log2_enrich_control` | log2(observed / background) enrichment in the control group. |
| `delta_log2_enrichment` | BWM minus control log2 enrichment — the **effect size** placed on the plots (default `--effect-col`). |
| `enrichment_ratio` | Linear ratio of the two groups' enrichments. |
| `p_value` | Raw p-value from the background-aware (conditional binomial) test. |
| `p_adj` | Benjamini-Hochberg FDR-adjusted p-value — drives significance shape/labels. |

## See also

- [`../README.md`](../README.md) — the R_scripts suite, including how `aa_codon_overlay.R` consumes these fixtures and how the background-diff CSVs are produced.
