---
input_csv: results/global_occupancy/analysis_corrected/codon_within_condition_binomial.csv
family: within_condition_binomial
test_type: One-sample binomial test (observed_count out of total_n vs H0: observed_freq = bg_freq, two-sided), BH-FDR within each (group, site) family of 62 codons (61 sense + in-frame stop TGA)
test_type_source: user-confirmed
n_tests: 1116
n_significant_fdr05: 1092
n_significant_fdr10: 1095
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "iid-violation-binomial", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bh-per-(group,site)", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bg-transcriptome-freq", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "stop-codon-instability", proposed_by: dylan, status: confirmed, why: "The 62nd codon is the in-frame stop TGA (TAA/TAG are absent). TGA carries the largest |log2_enrichment| in the file (site A/P/E mean +2.68/+2.18/+2.03, individual values up to +3.53) and sits in the concordant-enrichment table, but rests on 16-20 counts; occupancy at an in-frame stop is mechanistically distinct from sense-codon occupancy and its magnitude is noise-dominated. Read direction, not magnitude, for the three TGA cells."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "The checker rare-codon flag fires below observed_count 100 in any group; in this file it fires only on the three TGA cells (min counts 16-20). Every sense codon has min count >= 2573 (E:AGG), so no sense codon is rare; the rare-codon concern is confined to TGA and coincides with stop-codon-instability."}
caveats_considered: []
synced_from_olive_qmd: 2026-06-11
headline: "Codon-level within-group binomial vs bg_freq, 6 groups (BWM and control x d0/d5/d10) x 3 sites x 62 codons (61 sense + in-frame stop TGA) = 1116 tests, BH-FDR per (group, site) family of 62: 1092/1116 hits at FDR<0.05 (1095 at FDR<0.10); min p_adj underflows to exactly 0.0 (624/1116 cells at 0.0, min nonzero 5.93e-323 at BWM_day_0 P:CAG) at whole-transcriptome total_n (~2.1M), so every one of the 186 (site, codon) cells clears the #sig>=2 floor. Cross-group concordance dominates: 55 cells concordantly enriched, 102 concordantly depleted, 29 discordant across all 6 groups. The in-frame stop TGA carries the largest |log2_enrichment| (A/P/E means +2.68/+2.18/+2.03, up to +3.53) but rests on 16-20 counts (stop-codon-instability, rare-codon) - read direction not magnitude. Largest stable sense cells hold one sign across all 6: E:ATA -1.49, P:ATA -1.46, E:AAG +1.22, P:CTA -1.26. Rank by magnitude + reproducibility, not p."
user_directives:
  - "(per-CSV triage) Test-type confirmation: filename + columns (observed_count/total_n/observed_freq/bg_freq/log2_enrichment/p_value/p_adj) -> `Confirm binomial - both files` (one-sample exact binomial of observed occupancy frequency vs the transcriptome bg_freq, two-sided)."
  - "(per-CSV triage) CSV-specific caveats beyond the 4 family-wide -> confirmed `stop-codon-instability` and `rare-codon-low-count` for the codon file (both pre-checked by Dylan; user confirmed)."
  - "(per-CSV triage) Framing firmness -> initially `Firm`, revised to `Mixed` after confirming the completed stall_sites sister within_condition_binomial files were framed Mixed and the same iid-violation-binomial caveat is locked here; magnitude + cross-group reproducibility lead, tiny p-values are iid-inflated and not read at face value."
  - "(per-CSV triage) Top-hits source -> user directed Dylan to run `scripts/cross_group_concordance_tables.py` (block 18, `--min-sig 2`) directly; the three concordance tables below are transcribed from that output."
  - "(per-CSV triage) Spotlight -> none; rank by data alone per A.2.3."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-11 -> clean no-op: the three cross-group concordance tables already matched the .qmd on structure + values (bare one-letter aa column, TGA -> *, NO raw-p column); Headline / Numbers at a glance / Methods / all six caveats already agreed (the iid-violation-binomial caveat already carried this file's underflow 624/1116 cells to 0.0, smallest nonzero 5.93e-323 - never the stale ~1e-132). No asymptotic-with-ties entry (binomial). Olive-only sections (Biological interpretation, composite + individual plots, including the two Stage-6 individual-plot caption fixes) not imported. Numbers-in-prose audit: 0 corrections."
---

# Interpretation — codon_within_condition_binomial

> Source: `results/global_occupancy/analysis_corrected/codon_within_condition_binomial.csv`
> Family: `within_condition_binomial` (see [`_INDEX.md`](_INDEX.md))
> Test type: One-sample binomial vs bg_freq, BH-FDR per (group, site) family of 62 codons (source: user-confirmed)

## User directives
- (per-CSV triage) Test type -> "Confirm binomial - both files": one-sample exact binomial, `observed_count ~ Binomial(total_n, bg_freq)`, two-sided, BH-FDR per (group, site).
- (per-CSV triage) CSV-specific caveats beyond family-wide -> confirmed `stop-codon-instability` and `rare-codon-low-count` (both Dylan-proposed, user-confirmed).
- (per-CSV triage) Framing firmness -> Mixed. Initially answered Firm, then revised to Mixed after confirming the stall_sites sister within_condition_binomial files were Mixed and the same `iid-violation-binomial` caveat applies; magnitude + cross-group reproducibility lead, p magnitude does not.
- (per-CSV triage) Top-hits source -> user directed Dylan to run `cross_group_concordance_tables.py --min-sig 2` (block 18); the tables below are transcribed from its output.
- (per-CSV triage) Spotlight -> none; data-ranked only per A.2.3.
- (readback) "Reconciled shared content from the corrected .qmd on 2026-06-11" -> clean no-op. The three cross-group concordance tables already matched the .qmd on structure and values (bare one-letter `aa` column with TGA -> `*`, NO raw-p column, 55/102/29 split); Headline, Numbers at a glance, Methods, and all six caveats already agreed (the `iid-violation-binomial` caveat already carried the file-specific underflow - 624/1116 cells to 0.0, smallest nonzero 5.93e-323 - never the stall-sites carryover ~1e-132). No asymptotic-with-ties entry (binomial, not Wilcoxon). Olive-only sections (Biological interpretation, composite and individual plots, including the two Stage-6 individual-plot caption fixes) intentionally not imported. Numbers-in-prose audit: 0 corrections.

## Headline
Codon-level within-group binomial against `bg_freq`, 6 groups (BWM and control x d0/d5/d10) x 3 sites (A/P/E) x 62 codons (61 sense + the in-frame stop TGA) = 1116 tests; BH-FDR per (group, site) family of 62. 1092/1116 hits at `p_adj` < 0.05 (1095/1116 at `p_adj` < 0.10). File min `p_adj` underflows to exactly 0.0: 624 of 1116 cells return `p_adj` = 0.0 and the smallest nonzero `p_adj` is 5.93e-323 (BWM_day_0, P:CAG, `log2_enrichment` = -0.345, `observed_count` = 23483, `total_n` = 2110390). At whole-transcriptome `total_n` (~2.1M) every one of the 186 (site, codon) cells clears the `#sig` >= 2 floor, so significance separates nothing and ranking is by magnitude and reproducibility. The dominant structure is **cross-group concordance**: 55 of the 186 cells are concordantly enriched, 102 concordantly depleted, and 29 discordant across all 6 groups. The in-frame stop **TGA carries the largest `log2_enrichment` in the file** (site A/P/E means +2.68 / +2.18 / +2.03, individual values up to +3.53) but rests on 16-20 counts, so it carries `stop-codon-instability` and `rare-codon` - read its direction, not its magnitude. Among sense codons the largest-magnitude cells hold one sign across all 6 groups: E:ATA -1.49, P:ATA -1.46, P:CTA -1.26, E:AAG +1.22 (Lys), E:CTA -1.21, P:TTA -1.14, P:GGG -1.07. p_adj magnitudes are co-amplified by `iid-violation-binomial`; magnitude-plus-reproducibility is the anchor for reading.

## Top hits

The effect column is `log2_enrichment` = log2(observed `observed_freq` / `bg_freq`); positive means the codon is over-represented at that site in that group relative to its transcriptome abundance, negative under-represented. The three cross-group tables partition the 186 (site, codon) cells by the sign of `log2_enrichment` across the 6 groups: concordant enrichment (positive in all 6), concordant depletion (negative in all 6), and discordant (>= 1 sign disagreement). Each table shows every cell that reached `p_adj` < 0.05 in at least 2 of the 6 groups, a reproducibility floor on the `#sig` axis (`#sig` = groups with FDR<0.05, of 6), **not** a fixed row cap; here every cell clears it, so the three tables show all 55 / 102 / 29 cells. Rows are sorted by site (A / P / E), then `#sig` descending, then `min count` (smallest `observed_count` across the 6 groups) descending. Ranking is deliberately not by p magnitude (624 cells underflow to `p_adj` = 0.0). In the `log2_enrichment` column the six per-group values are on two lines: BWM day_0, day_5, day_10 first, then control day_0, day_5, day_10. The `aa` column is the single-letter amino-acid translation of the codon (no parenthetical expansion); the codon cell stays bare. The `flag` column carries `iid-amp` (a group `p_adj` < 1e-10) and `rare-codon` (a group `observed_count` < 100; here only the three TGA cells); no codon's `bg_freq` exceeds 0.05, so `bg-tight` never fires.

### Concordant enrichment: significant in >= 2 of 6 groups

| Site | codon | aa | log2_enrichment | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- | --- |
| A | GGA | G | +0.68, +0.46, +0.62 <br> +0.72, +0.48, +0.63 | 62966 | 6/6 | iid-amp |
| A | AAG | K | +1.01, +0.94, +0.89 <br> +1.07, +1.05, +0.85 | 58491 | 6/6 | iid-amp |
| A | GAT | D | +0.05, +0.38, +0.35 <br> +0.24, +0.40, +0.29 | 56787 | 6/6 | iid-amp |
| A | GAG | E | +0.59, +0.61, +0.76 <br> +0.72, +0.70, +0.73 | 51620 | 6/6 | iid-amp |
| A | TTC | F | +0.12, +0.20, +0.20 <br> +0.17, +0.16, +0.24 | 35763 | 6/6 | iid-amp |
| A | ATC | I | +0.27, +0.44, +0.42 <br> +0.49, +0.53, +0.40 | 31876 | 6/6 | iid-amp |
| A | GAC | D | +0.29, +0.40, +0.35 <br> +0.46, +0.34, +0.37 | 27933 | 6/6 | iid-amp |
| A | TAC | Y | +0.79, +0.44, +0.36 <br> +0.66, +0.42, +0.62 | 27482 | 6/6 | iid-amp |
| A | GTC | V | +0.36, +0.54, +0.54 <br> +0.61, +0.46, +0.48 | 23479 | 6/6 | iid-amp |
| A | CGT | R | +0.71, +0.78, +0.77 <br> +0.79, +0.60, +0.76 | 23296 | 6/6 | iid-amp |
| A | AGA | R | +0.13, +0.08, +0.15 <br> +0.14, +0.25, +0.08 | 20632 | 6/6 | iid-amp |
| A | GCC | A | +0.30, +0.29, +0.25 <br> +0.53, +0.05, +0.35 | 19827 | 6/6 | iid-amp |
| A | ACC | T | +0.21, +0.35, +0.26 <br> +0.54, +0.26, +0.30 | 15938 | 6/6 | iid-amp |
| A | CAC | H | +0.29, +0.17, +0.07 <br> +0.30, +0.06, +0.13 | 12443 | 6/6 | iid-amp |
| A | CGC | R | +0.95, +0.95, +0.92 <br> +1.01, +0.68, +1.01 | 12250 | 6/6 | iid-amp |
| A | TGA | * | +2.49, +2.99, +3.53 <br> +2.62, +3.09, +1.39 | 19 | 6/6 | iid-amp, rare-codon (BWM d0, BWM d5, ctrl d10) |
| P | GAT | D | +0.61, +0.57, +0.70 <br> +0.47, +0.70, +0.75 | 78068 | 6/6 | iid-amp |
| P | GGA | G | +0.86, +0.48, +0.60 <br> +0.76, +0.46, +0.65 | 64004 | 6/6 | iid-amp |
| P | AAG | K | +0.87, +0.80, +0.64 <br> +0.97, +0.62, +0.69 | 52116 | 6/6 | iid-amp |
| P | GAG | E | +0.42, +0.49, +0.69 <br> +0.60, +0.59, +0.68 | 49949 | 6/6 | iid-amp |
| P | TTC | F | +0.30, +0.31, +0.27 <br> +0.26, +0.35, +0.25 | 36033 | 6/6 | iid-amp |
| P | GTT | V | +0.28, +0.28, +0.23 <br> +0.22, +0.43, +0.17 | 34653 | 6/6 | iid-amp |
| P | AAC | N | +0.52, +0.50, +0.39 <br> +0.53, +0.46, +0.46 | 31624 | 6/6 | iid-amp |
| P | GCT | A | +0.15, +0.09, +0.10 <br> +0.09, +0.10, +0.12 | 30898 | 6/6 | iid-amp |
| P | ATC | I | +0.48, +0.50, +0.36 <br> +0.57, +0.55, +0.35 | 30757 | 6/6 | iid-amp |
| P | GAC | D | +0.51, +0.40, +0.47 <br> +0.43, +0.36, +0.46 | 29743 | 6/6 | iid-amp |
| P | CGT | R | +0.88, +0.84, +0.82 <br> +0.89, +0.79, +0.86 | 25013 | 6/6 | iid-amp |
| P | GTC | V | +0.54, +0.50, +0.56 <br> +0.57, +0.56, +0.41 | 22349 | 6/6 | iid-amp |
| P | TAC | Y | +0.25, +0.30, +0.15 <br> +0.36, +0.18, +0.23 | 21076 | 6/6 | iid-amp |
| P | GCC | A | +0.34, +0.27, +0.29 <br> +0.47, +0.17, +0.29 | 18966 | 6/6 | iid-amp |
| P | CTC | L | +0.05, +0.21, +0.16 <br> +0.17, +0.06, +0.02 | 18499 | 6/6 | iid-amp |
| P | ACC | T | +0.38, +0.40, +0.32 <br> +0.61, +0.28, +0.40 | 17104 | 6/6 | iid-amp |
| P | GGT | G | +0.39, +0.05, +0.22 <br> +0.16, +0.16, +0.22 | 16239 | 6/6 | iid-amp |
| P | CGC | R | +0.57, +0.63, +0.60 <br> +0.67, +0.45, +0.63 | 9434 | 6/6 | iid-amp |
| P | TGA | * | +1.40, +2.51, +3.14 <br> +1.83, +2.74, +1.46 | 20 | 6/6 | iid-amp, rare-codon (all) |
| E | AAG | K | +1.24, +1.13, +1.06 <br> +1.33, +1.40, +1.17 | 72657 | 6/6 | iid-amp |
| E | GGA | G | +0.56, +0.36, +0.43 <br> +0.64, +0.46, +0.44 | 55267 | 6/6 | iid-amp |
| E | GAT | D | +0.05, +0.11, +0.14 <br> +0.06, +0.21, +0.14 | 51079 | 6/6 | iid-amp |
| E | GAG | E | +0.55, +0.60, +0.59 <br> +0.69, +0.77, +0.65 | 48734 | 6/6 | iid-amp |
| E | CAA | Q | +0.07, +0.15, +0.12 <br> +0.10, +0.07, +0.07 | 35890 | 6/6 | iid-amp |
| E | ATC | I | +0.56, +0.56, +0.49 <br> +0.65, +0.44, +0.57 | 35715 | 6/6 | iid-amp |
| E | AAC | N | +0.54, +0.56, +0.47 <br> +0.65, +0.57, +0.56 | 33901 | 6/6 | iid-amp |
| E | GAC | D | +0.34, +0.32, +0.35 <br> +0.40, +0.28, +0.40 | 28600 | 6/6 | iid-amp |
| E | CTT | L | +0.20, +0.22, +0.17 <br> +0.08, +0.05, +0.10 | 28293 | 6/6 | iid-amp |
| E | GTC | V | +0.64, +0.70, +0.69 <br> +0.79, +0.51, +0.72 | 27838 | 6/6 | iid-amp |
| E | CTC | L | +0.35, +0.39, +0.47 <br> +0.35, +0.17, +0.42 | 24370 | 6/6 | iid-amp |
| E | AGA | R | +0.26, +0.14, +0.10 <br> +0.27, +0.31, +0.15 | 21749 | 6/6 | iid-amp |
| E | GCC | A | +0.44, +0.37, +0.37 <br> +0.56, +0.06, +0.44 | 20982 | 6/6 | iid-amp |
| E | CGT | R | +0.66, +0.57, +0.59 <br> +0.62, +0.60, +0.52 | 19730 | 6/6 | iid-amp |
| E | ACC | T | +0.51, +0.53, +0.40 <br> +0.70, +0.25, +0.52 | 18647 | 6/6 | iid-amp |
| E | CAC | H | +0.15, +0.28, +0.21 <br> +0.32, +0.11, +0.24 | 13429 | 6/6 | iid-amp |
| E | CGC | R | +0.66, +0.72, +0.68 <br> +0.74, +0.55, +0.65 | 9560 | 6/6 | iid-amp |
| E | TGA | * | +1.15, +2.35, +3.30 <br> +1.77, +2.47, +1.14 | 16 | 6/6 | iid-amp, rare-codon (all) |
| E | CCA | P | +0.66, +0.28, +0.15 <br> +0.49, +0.01, +0.26 | 39543 | 5/6 | iid-amp |
| E | AAA | K | +0.04, +0.03, +0.01 <br> +0.00, +0.12, +0.10 | 51363 | 4/6 | iid-amp |

### Concordant depletion: significant in >= 2 of 6 groups

| Site | codon | aa | log2_enrichment | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- | --- |
| A | ATT | I | -0.50, -0.28, -0.21 <br> -0.31, -0.07, -0.30 | 33940 | 6/6 | iid-amp |
| A | AAA | K | -0.45, -0.45, -0.45 <br> -0.53, -0.20, -0.51 | 33645 | 6/6 | iid-amp |
| A | AAT | N | -0.52, -0.23, -0.22 <br> -0.38, -0.06, -0.41 | 29099 | 6/6 | iid-amp |
| A | GCT | A | -0.09, -0.04, -0.06 <br> -0.05, -0.26, -0.03 | 27856 | 6/6 | iid-amp |
| A | GCA | A | -0.21, -0.50, -0.48 <br> -0.50, -0.57, -0.33 | 20493 | 6/6 | iid-amp |
| A | ACT | T | -0.39, -0.30, -0.35 <br> -0.37, -0.30, -0.38 | 18776 | 6/6 | iid-amp |
| A | TTT | F | -0.90, -0.77, -0.75 <br> -0.91, -0.74, -0.77 | 17139 | 6/6 | iid-amp |
| A | ACA | T | -0.56, -0.56, -0.62 <br> -0.73, -0.51, -0.61 | 17095 | 6/6 | iid-amp |
| A | TCT | S | -0.26, -0.26, -0.28 <br> -0.36, -0.32, -0.32 | 16856 | 6/6 | iid-amp |
| A | CAT | H | -0.11, -0.06, -0.11 <br> -0.18, -0.07, -0.10 | 16496 | 6/6 | iid-amp |
| A | TCA | S | -0.73, -0.65, -0.58 <br> -0.78, -0.62, -0.67 | 16354 | 6/6 | iid-amp |
| A | CAG | Q | -0.07, -0.06, -0.06 <br> -0.13, -0.17, -0.18 | 15735 | 6/6 | iid-amp |
| A | GTG | V | -0.12, -0.41, -0.42 <br> -0.31, -0.44, -0.26 | 15614 | 6/6 | iid-amp |
| A | TCG | S | -0.27, -0.31, -0.19 <br> -0.43, -0.36, -0.19 | 13232 | 6/6 | iid-amp |
| A | CGA | R | -0.24, -0.29, -0.29 <br> -0.62, -0.41, -0.29 | 12426 | 6/6 | iid-amp |
| A | CTG | L | -0.37, -0.52, -0.49 <br> -0.73, -0.67, -0.33 | 12361 | 6/6 | iid-amp |
| A | GGT | G | -0.16, -0.39, -0.31 <br> -0.37, -0.48, -0.49 | 9899 | 6/6 | iid-amp |
| A | TGT | C | -0.71, -0.58, -0.48 <br> -0.79, -0.46, -0.59 | 9689 | 6/6 | iid-amp |
| A | CCG | P | -0.47, -0.49, -0.51 <br> -0.80, -0.88, -0.44 | 9088 | 6/6 | iid-amp |
| A | GCG | A | -0.23, -0.44, -0.36 <br> -0.58, -0.60, -0.24 | 8930 | 6/6 | iid-amp |
| A | AGT | S | -0.91, -0.79, -0.68 <br> -0.97, -0.64, -0.80 | 8844 | 6/6 | iid-amp |
| A | ATA | I | -0.56, -0.42, -0.52 <br> -0.68, -0.26, -0.50 | 8740 | 6/6 | iid-amp |
| A | TTA | L | -0.48, -0.50, -0.54 <br> -0.68, -0.48, -0.51 | 8588 | 6/6 | iid-amp |
| A | ACG | T | -0.41, -0.42, -0.38 <br> -0.64, -0.41, -0.39 | 8577 | 6/6 | iid-amp |
| A | GTA | V | -0.48, -0.59, -0.65 <br> -0.58, -0.60, -0.60 | 8369 | 6/6 | iid-amp |
| A | AGC | S | -0.57, -0.54, -0.49 <br> -0.59, -0.47, -0.54 | 7121 | 6/6 | iid-amp |
| A | CCT | P | -0.57, -0.67, -0.75 <br> -0.94, -1.00, -0.67 | 7093 | 6/6 | iid-amp |
| A | GGC | G | -0.22, -0.69, -0.58 <br> -0.66, -0.90, -0.47 | 6112 | 6/6 | iid-amp |
| A | CTA | L | -0.69, -0.84, -0.87 <br> -1.10, -1.04, -0.84 | 5646 | 6/6 | iid-amp |
| A | CGG | R | -0.35, -0.47, -0.38 <br> -0.85, -0.70, -0.29 | 4799 | 6/6 | iid-amp |
| A | CCC | P | -0.35, -0.69, -0.89 <br> -0.83, -1.09, -0.56 | 3700 | 6/6 | iid-amp |
| A | AGG | R | -0.52, -0.72, -0.52 <br> -0.92, -0.56, -0.52 | 3319 | 6/6 | iid-amp |
| A | GGG | G | -0.86, -1.04, -0.75 <br> -1.12, -1.20, -0.86 | 3155 | 6/6 | iid-amp |
| P | GAA | E | -0.26, -0.22, -0.14 <br> -0.22, -0.10, -0.12 | 48327 | 6/6 | iid-amp |
| P | AAA | K | -0.30, -0.26, -0.26 <br> -0.28, -0.23, -0.25 | 40232 | 6/6 | iid-amp |
| P | CAA | Q | -0.13, -0.13, -0.21 <br> -0.15, -0.34, -0.26 | 28508 | 6/6 | iid-amp |
| P | ATG | M | -0.33, -0.18, -0.23 <br> -0.20, -0.24, -0.27 | 25872 | 6/6 | iid-amp |
| P | ACT | T | -0.11, -0.07, -0.12 <br> -0.14, -0.10, -0.09 | 23043 | 6/6 | iid-amp |
| P | TTG | L | -0.35, -0.18, -0.22 <br> -0.20, -0.13, -0.31 | 20613 | 6/6 | iid-amp |
| P | TTT | F | -0.62, -0.53, -0.52 <br> -0.75, -0.33, -0.58 | 19575 | 6/6 | iid-amp |
| P | GCA | A | -0.54, -0.74, -0.56 <br> -0.71, -0.65, -0.53 | 17855 | 6/6 | iid-amp |
| P | TCT | S | -0.27, -0.21, -0.30 <br> -0.35, -0.25, -0.38 | 16106 | 6/6 | iid-amp |
| P | ACA | T | -0.78, -0.73, -0.71 <br> -0.80, -0.72, -0.72 | 15828 | 6/6 | iid-amp |
| P | TCA | S | -0.77, -0.68, -0.67 <br> -0.80, -0.65, -0.73 | 15675 | 6/6 | iid-amp |
| P | CAG | Q | -0.35, -0.32, -0.25 <br> -0.34, -0.47, -0.25 | 15058 | 6/6 | iid-amp |
| P | GTG | V | -0.51, -0.49, -0.37 <br> -0.48, -0.37, -0.39 | 14247 | 6/6 | iid-amp |
| P | AGT | S | -0.49, -0.45, -0.42 <br> -0.58, -0.40, -0.36 | 11993 | 6/6 | iid-amp |
| P | TGT | C | -0.54, -0.41, -0.34 <br> -0.59, -0.30, -0.29 | 11896 | 6/6 | iid-amp |
| P | CGA | R | -0.37, -0.59, -0.40 <br> -0.79, -0.58, -0.38 | 11626 | 6/6 | iid-amp |
| P | TCC | S | -0.24, -0.12, -0.18 <br> -0.14, -0.32, -0.29 | 10793 | 6/6 | iid-amp |
| P | TCG | S | -0.68, -0.50, -0.38 <br> -0.65, -0.42, -0.50 | 10661 | 6/6 | iid-amp |
| P | CCG | P | -0.72, -0.64, -0.39 <br> -0.71, -0.65, -0.35 | 9652 | 6/6 | iid-amp |
| P | TGC | C | -0.37, -0.37, -0.36 <br> -0.34, -0.44, -0.31 | 9475 | 6/6 | iid-amp |
| P | TGG | W | -0.52, -0.51, -0.49 <br> -0.52, -0.73, -0.63 | 9188 | 6/6 | iid-amp |
| P | AGC | S | -0.31, -0.25, -0.28 <br> -0.29, -0.21, -0.27 | 8595 | 6/6 | iid-amp |
| P | CCT | P | -0.72, -0.63, -0.45 <br> -0.82, -0.61, -0.47 | 8165 | 6/6 | iid-amp |
| P | ACG | T | -0.75, -0.57, -0.41 <br> -0.72, -0.48, -0.47 | 8106 | 6/6 | iid-amp |
| P | GTA | V | -0.67, -0.71, -0.64 <br> -0.74, -0.59, -0.68 | 7901 | 6/6 | iid-amp |
| P | CTG | L | -1.06, -0.87, -0.86 <br> -1.12, -0.97, -1.02 | 7684 | 6/6 | iid-amp |
| P | GCG | A | -0.87, -0.81, -0.58 <br> -0.94, -0.68, -0.63 | 6798 | 6/6 | iid-amp |
| P | GGC | G | -0.42, -0.58, -0.36 <br> -0.63, -0.58, -0.47 | 6116 | 6/6 | iid-amp |
| P | TTA | L | -1.38, -1.05, -0.99 <br> -1.23, -0.90, -1.30 | 4961 | 6/6 | iid-amp |
| P | ATA | I | -1.63, -1.38, -1.36 <br> -1.57, -1.31, -1.50 | 4381 | 6/6 | iid-amp |
| P | CTA | L | -1.22, -1.21, -1.26 <br> -1.41, -1.19, -1.24 | 4286 | 6/6 | iid-amp |
| P | CGG | R | -1.07, -0.97, -0.67 <br> -1.30, -1.10, -0.81 | 3348 | 6/6 | iid-amp |
| P | CCC | P | -1.01, -0.95, -0.75 <br> -0.97, -1.06, -0.78 | 3253 | 6/6 | iid-amp |
| P | GGG | G | -0.99, -1.19, -0.84 <br> -1.19, -1.21, -0.97 | 2914 | 6/6 | iid-amp |
| P | AGG | R | -0.99, -0.97, -0.74 <br> -1.11, -1.09, -0.74 | 2859 | 6/6 | iid-amp |
| E | ATT | I | -0.22, -0.23, -0.31 <br> -0.32, -0.26, -0.34 | 33113 | 6/6 | iid-amp |
| E | ACT | T | -0.12, -0.13, -0.20 <br> -0.22, -0.34, -0.19 | 21505 | 6/6 | iid-amp |
| E | TTG | L | -0.28, -0.16, -0.15 <br> -0.28, -0.08, -0.27 | 21188 | 6/6 | iid-amp |
| E | TCA | S | -0.76, -0.55, -0.52 <br> -0.73, -0.64, -0.59 | 17306 | 6/6 | iid-amp |
| E | GCA | A | -0.57, -0.62, -0.55 <br> -0.72, -0.70, -0.59 | 17221 | 6/6 | iid-amp |
| E | ACA | T | -0.77, -0.60, -0.59 <br> -0.76, -0.68, -0.63 | 16867 | 6/6 | iid-amp |
| E | TCT | S | -0.42, -0.22, -0.22 <br> -0.40, -0.43, -0.36 | 16405 | 6/6 | iid-amp |
| E | TAT | Y | -0.48, -0.40, -0.52 <br> -0.56, -0.51, -0.55 | 15703 | 6/6 | iid-amp |
| E | TTT | F | -0.94, -0.87, -0.91 <br> -1.10, -0.97, -1.00 | 14675 | 6/6 | iid-amp |
| E | CAT | H | -0.41, -0.27, -0.23 <br> -0.35, -0.25, -0.32 | 14232 | 6/6 | iid-amp |
| E | GTG | V | -0.32, -0.41, -0.35 <br> -0.49, -0.28, -0.39 | 14218 | 6/6 | iid-amp |
| E | TCG | S | -0.55, -0.26, -0.14 <br> -0.49, -0.17, -0.23 | 12829 | 6/6 | iid-amp |
| E | TGG | W | -0.21, -0.27, -0.13 <br> -0.24, -0.09, -0.24 | 11982 | 6/6 | iid-amp |
| E | GGT | G | -0.15, -0.37, -0.12 <br> -0.27, -0.14, -0.25 | 11685 | 6/6 | iid-amp |
| E | CGA | R | -0.68, -0.59, -0.41 <br> -0.77, -0.39, -0.40 | 11515 | 6/6 | iid-amp |
| E | CTG | L | -0.61, -0.55, -0.46 <br> -0.77, -0.43, -0.48 | 11126 | 6/6 | iid-amp |
| E | CCG | P | -0.51, -0.59, -0.46 <br> -0.68, -0.52, -0.39 | 9417 | 6/6 | iid-amp |
| E | AGT | S | -0.97, -0.93, -0.72 <br> -1.07, -0.68, -0.82 | 8682 | 6/6 | iid-amp |
| E | AGC | S | -0.42, -0.36, -0.29 <br> -0.44, -0.29, -0.32 | 8256 | 6/6 | iid-amp |
| E | ACG | T | -0.80, -0.60, -0.45 <br> -0.80, -0.41, -0.51 | 7935 | 6/6 | iid-amp |
| E | GTA | V | -0.62, -0.69, -0.70 <br> -0.78, -0.69, -0.70 | 7814 | 6/6 | iid-amp |
| E | TGC | C | -0.62, -0.54, -0.51 <br> -0.63, -0.71, -0.64 | 7524 | 6/6 | iid-amp |
| E | TGT | C | -0.92, -0.87, -0.81 <br> -1.11, -0.84, -0.99 | 7344 | 6/6 | iid-amp |
| E | CCT | P | -0.57, -0.80, -0.70 <br> -0.91, -0.93, -0.63 | 7289 | 6/6 | iid-amp |
| E | GCG | A | -0.70, -0.68, -0.47 <br> -0.84, -0.56, -0.53 | 7265 | 6/6 | iid-amp |
| E | TTA | L | -1.09, -0.94, -0.93 <br> -1.13, -0.95, -1.03 | 5982 | 6/6 | iid-amp |
| E | GGC | G | -0.65, -0.74, -0.42 <br> -0.69, -0.61, -0.51 | 5958 | 6/6 | iid-amp |
| E | CTA | L | -1.26, -1.13, -1.06 <br> -1.41, -1.23, -1.17 | 4508 | 6/6 | iid-amp |
| E | ATA | I | -1.59, -1.42, -1.40 <br> -1.69, -1.38, -1.46 | 4507 | 6/6 | iid-amp |
| E | CGG | R | -1.02, -0.97, -0.72 <br> -1.15, -0.61, -0.65 | 3733 | 6/6 | iid-amp |
| E | CCC | P | -0.91, -0.96, -0.90 <br> -0.99, -1.13, -0.77 | 3258 | 6/6 | iid-amp |
| E | GGG | G | -1.16, -1.21, -0.91 <br> -1.17, -0.94, -0.95 | 2956 | 6/6 | iid-amp |
| E | AGG | R | -0.90, -1.03, -0.91 <br> -1.13, -0.74, -0.89 | 2573 | 6/6 | iid-amp |
| E | AAT | N | -0.29, -0.20, -0.19 <br> -0.30, -0.00, -0.19 | 33902 | 5/6 | iid-amp |

### Discordant: significant in >= 2 of 6 groups

| Site | codon | aa | log2_enrichment | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- | --- |
| A | CCA | P | +0.31, +0.15, +0.07 <br> +0.34, -0.14, +0.19 | 37808 | 6/6 | iid-amp |
| A | TTG | L | +0.26, +0.02, -0.02 <br> +0.06, +0.02, +0.13 | 28088 | 6/6 | iid-amp |
| A | CTC | L | +0.27, +0.25, +0.13 <br> +0.24, -0.08, +0.11 | 19621 | 6/6 | iid-amp |
| A | TGC | C | +0.10, -0.21, -0.18 <br> -0.16, -0.29, -0.03 | 11464 | 6/6 | iid-amp |
| A | ATG | M | -0.19, -0.01, +0.01 <br> -0.13, +0.09, -0.04 | 30255 | 5/6 | iid-amp |
| A | GTT | V | -0.28, -0.00, +0.02 <br> -0.03, +0.01, -0.08 | 29160 | 5/6 | iid-amp |
| A | CAA | Q | +0.04, +0.00, -0.11 <br> -0.01, -0.04, -0.24 | 28904 | 5/6 | iid-amp |
| A | CTT | L | +0.03, +0.05, -0.10 <br> -0.00, -0.21, -0.09 | 24827 | 5/6 | iid-amp |
| A | TAT | Y | -0.01, +0.09, -0.04 <br> -0.03, +0.22, +0.06 | 24005 | 5/6 | iid-amp |
| A | AAC | N | +0.00, +0.18, +0.17 <br> +0.20, +0.29, -0.03 | 22570 | 5/6 | iid-amp |
| A | TGG | W | +0.56, +0.03, -0.01 <br> +0.12, -0.13, +0.21 | 16363 | 5/6 | iid-amp |
| A | TCC | S | +0.10, +0.07, +0.00 <br> +0.17, -0.16, +0.08 | 13907 | 5/6 | iid-amp |
| A | GAA | E | +0.01, -0.01, +0.12 <br> -0.01, +0.24, +0.15 | 58315 | 4/6 | iid-amp |
| P | AAT | N | -0.02, +0.13, +0.06 <br> -0.05, +0.23, +0.15 | 43032 | 6/6 | iid-amp |
| P | CCA | P | +0.24, +0.10, +0.05 <br> +0.34, -0.05, +0.11 | 35687 | 6/6 | iid-amp |
| P | ATT | I | -0.16, -0.07, -0.24 <br> -0.19, +0.02, -0.27 | 34785 | 6/6 | iid-amp |
| P | TAT | Y | -0.02, +0.06, -0.06 <br> +0.02, +0.12, +0.08 | 24387 | 6/6 | iid-amp |
| P | CAT | H | -0.06, +0.06, -0.04 <br> -0.08, -0.05, -0.03 | 17328 | 6/6 | iid-amp |
| P | CAC | H | +0.14, +0.17, +0.03 <br> +0.22, -0.08, +0.06 | 11839 | 6/6 | iid-amp |
| P | CTT | L | +0.13, +0.26, +0.10 <br> +0.15, +0.20, -0.00 | 26445 | 5/6 | iid-amp |
| P | AGA | R | +0.03, -0.00, -0.10 <br> +0.06, -0.14, -0.02 | 19308 | 4/6 | iid-amp |
| E | TTC | F | +0.14, +0.19, +0.11 <br> +0.11, -0.06, +0.08 | 32092 | 6/6 | iid-amp |
| E | TAC | Y | +0.17, +0.27, +0.11 <br> +0.23, -0.04, +0.08 | 18956 | 6/6 | iid-amp |
| E | TCC | S | -0.07, +0.14, +0.09 <br> +0.09, -0.20, +0.04 | 13522 | 6/6 | iid-amp |
| E | GAA | E | -0.08, -0.02, +0.01 <br> -0.07, +0.04, +0.05 | 54315 | 5/6 | iid-amp |
| E | ATG | M | -0.12, -0.04, +0.01 <br> -0.07, +0.16, -0.02 | 30636 | 5/6 | iid-amp |
| E | GCT | A | +0.06, +0.02, +0.02 <br> -0.01, -0.20, -0.02 | 27992 | 5/6 | iid-amp |
| E | GTT | V | +0.13, +0.09, +0.04 <br> +0.03, -0.00, -0.01 | 30584 | 4/6 | iid-amp |
| E | CAG | Q | -0.04, +0.00, +0.09 <br> +0.01, +0.23, +0.09 | 19013 | 4/6 | iid-amp |

## Numbers at a glance
- `n_tests`: 1116 (6 groups x 3 sites x 62 codons; 61 sense + in-frame stop TGA)
- `n_significant` (adjusted-p < 0.05): 1092
- `n_significant` (adjusted-p < 0.10): 1095
- `min adjusted-p`: 0.0 (underflow) - 624 of 1116 cells return `p_adj` exactly 0.0; smallest nonzero `p_adj` is 5.93e-323 (BWM_day_0, P:CAG; log2 -0.345, k=23483, n=2110390). Under `iid-violation-binomial` this is not a magnitude claim.
- `p_floor`: n/a - no exact-test floor for the binomial at these n.
- Per-(group, site) hits at `p_adj` < 0.05 (each is 62 codons; A / P / E order; counts sum to 1092):
  - BWM_day_0: A 59/62, P 62/62, E 62/62
  - BWM_day_5: A 59/62, P 61/62, E 61/62
  - BWM_day_10: A 59/62, P 62/62, E 59/62
  - control_day_0: A 61/62, P 62/62, E 59/62
  - control_day_5: A 62/62, P 62/62, E 59/62
  - control_day_10: A 62/62, P 60/62, E 61/62

## Methods
Dylan proposed a one-sample binomial vs `bg_freq` (`observed_count ~ Binomial(total_n, bg_freq)`, two-sided) with BH-FDR within each (group, site) family of 62 codons; the user confirmed (`test_type_source: user-confirmed`). The 62 codons are the 61 sense codons plus the in-frame stop TGA (TAA and TAG are absent from this file). Effect column is `log2_enrichment` = log2(`observed_freq` / `bg_freq`); a count-weighted `weighted_log2_enrichment` is also present but not used for ranking. The null `bg_freq` is the transcriptome codon composition, so a positive value means "occupied more often than its transcriptomic abundance predicts," not "more occupied than control." The test answers "is codon X observed at this site at a different frequency than its transcriptome `bg_freq` in the same group?", *not* a BWM-vs-control or between-timepoint contrast (those are the Fisher and Wilcoxon families). Top hits are three cross-group tables partitioning the 186 (site, codon) cells by the sign of `log2_enrichment` across the 6 groups (concordant enrichment 55, concordant depletion 102, discordant 29), each showing every cell with `p_adj` < 0.05 in >= 2 of the 6 groups (all 186 cells clear this floor), sorted by site (A/P/E), `#sig` desc, `min count` desc, with an `aa` column added per the codon-table convention. Ranking is by magnitude and reproducibility, not p, because at `total_n` ~2.1M the binomial p collapses toward zero (624/1116 underflow to exactly 0.0).

## Caveats
### Confirmed
- **pseudorep** (family-wide) - replicates within a group are summed before `binomtest`; `total_n` is a pooled occupancy count rather than a sum of independent replicates, so the null treats correlated draws as independent.
- **iid-violation-binomial** (family-wide) - the binomial null assumes `total_n` iid Bernoulli draws at probability `bg_freq`; footprints are not iid (overlapping E/P/A windows on one transcript, transcript-level coverage variation). The practical effect compresses p toward zero (624/1116 cells underflow to 0.0; smallest nonzero 5.93e-323). `log2_enrichment` is the primary effect column, not p.
- **bh-per-(group, site)** (family-wide) - FDR is per (group, site), within each 62-codon family, not pooled across codons per group or across the merged E/P/A file.
- **bg-transcriptome-freq** (family-wide) - the null is the transcriptome codon composition (`bg_freq`), so "enrichment" means "occupied more than its transcriptomic abundance predicts," not vs uniform or vs control.
- **stop-codon-instability** (per-CSV) - the in-frame stop TGA carries the file's largest `log2_enrichment` (site A/P/E means +2.68 / +2.18 / +2.03, up to +3.53 in single groups) and sits in the concordant-enrichment table, but rests on 16-20 counts; occupancy at an in-frame stop is mechanistically distinct from sense-codon occupancy and the magnitude is noise-dominated. Read the direction (consistently positive across all 6 groups) as suggestive, not the magnitude; do not rank TGA alongside high-count sense codons.
- **rare-codon-low-count** (per-CSV) - the `rare-codon` flag (any group `observed_count` < 100) fires only on the three TGA cells (min counts 16-20). Every sense codon has `min count` >= 2573 (E:AGG), so no sense codon is low-count; the rare-codon concern is confined to TGA and coincides with `stop-codon-instability`.

### Considered but not applicable
*(none denied - both Dylan-proposed codon-specific caveats were confirmed; the 4 family-wide caveats were inherited from _INDEX.md.)*

## For Chumeng (joint-reading hooks)
- Family: `within_condition_binomial` - sister CSV in this family that should be reconciled: `aa_within_condition_binomial.csv` (amino-acid resolution; same design, same 4 family-wide caveats; no per-CSV caveats there).
- Open questions Chumeng should resolve at synthesis time, framed as falsifiers per A.2.7:
  - Many large depletions are low-usage synonyms (E:ATA -1.49, P:ATA -1.46, P/E:CTA, P/E:TTA, P/E:GGG, P:CGG, P:CCC) while the matching amino acid is enriched or near-zero at the same site (e.g. Lys: E:AAG +1.22 enriched vs P:AAA -0.30, A:AAA -0.45 depleted). **Does each aa-level cell in `aa_within_condition_binomial.csv` decompose into a clean synonym split here (one codon carrying the sign, its sisters opposing), or do synonyms move together?** A consistent split would mean the signal is codon-level decoding, not an amino-acid property - the central aa-vs-codon question for this family.
  - The in-frame stop **TGA is the largest-magnitude cell but rests on 16-20 counts.** **Does TGA reach a comparable extreme in any large-N Fisher file (`per_timepoint_fisher_codon`, `timepoint_fisher_within_condition_*`), or only in this low-count within-group binomial?** If only here, the TGA enrichment is a rare-event / instability artefact; if it recurs at high N, it is worth a readthrough-occupancy reading. Falsifier for `stop-codon-instability`.
  - These per-group binomials are blind to BWM-vs-control divergence. **Do the concordant cells (e.g. E:AAG, P:GAT, P:ATA) reappear with consistent direction in `per_timepoint_fisher_codon.csv`?** A concordant binomial cell that is null in the Fisher contrast at that (timepoint, site, codon) is a stable occupancy property; one that is also a large-OR Fisher hit needs a perturbation overlay (or is a large-N/imbalanced-N artefact).
  - File min `p_adj` is exactly 0.0 for 624 cells, the same iid-driven collapse as the aa file. **Does the codon magnitude ranking (TGA aside: ATA, CTA, TTA, GGG depleted; AAG enriched) match the magnitude ranking of the same codons in the between-condition / between-timepoint Wilcoxon files**, where N is small and the floor is the binding constraint instead? Agreement across the opposite-N-regime designs would separate a stable codon-occupancy pattern from an N-driven artefact.
