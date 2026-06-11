---
input_csv: results/global_occupancy/analysis_corrected/aa_within_condition_binomial.csv
family: within_condition_binomial
test_type: One-sample binomial test (observed_count out of total_n vs H0: observed_freq = bg_freq, two-sided), BH-FDR within each (group, site) family of 20 aa
test_type_source: user-confirmed
n_tests: 360
n_significant_fdr05: 349
n_significant_fdr10: 350
min_p_adj: 0.0
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "iid-violation-binomial", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bh-per-(group,site)", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bg-transcriptome-freq", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
caveats_considered:
  - {label: "rare-aa", proposed_by: dylan, status: denied, why: "The checker rare-k flag fires below observed_count 100 in any group; the smallest min count across all 60 (site, aa) cells is 9188 (P:W), far above threshold, so no aa cell is low-count.", user_note: "User confirmed no extra per-CSV caveats for the aa file."}
headline: "Within-group one-sample binomial vs bg_freq across 6 groups (BWM and control x d0/d5/d10) x 3 sites x 20 aa = 360 tests, BH-FDR per (group, site) family of 20: 349/360 hits at FDR<0.05 (350 at FDR<0.10); min p_adj underflows to exactly 0.0 (188/360 cells at 0.0, min nonzero 9.39e-323 at control_day_5 E:R) because total_n is whole-transcriptome (~2.1-2.4M), so significance is near-universal and uninformative for ranking. Dominant structure is cross-group concordance: of the 60 (site, aa) cells, 20 are concordantly enriched and 26 concordantly depleted across all 6 groups, 14 discordant; largest-magnitude cells hold one sign across all 6 (E:C -0.77, E:K +0.64, P:D +0.58, P:W -0.57). Top hits are three sign-partitioned cross-group tables; rank by log2_enrichment magnitude + cross-group reproducibility, not p."
synced_from_olive_qmd: 2026-06-10
user_directives:
  - "(per-CSV triage) Test-type confirmation: filename + columns (observed_count/total_n/observed_freq/bg_freq/log2_enrichment/p_value/p_adj) -> `Confirm binomial - both files` (one-sample exact binomial of observed occupancy frequency vs the transcriptome bg_freq, two-sided)."
  - "(per-CSV triage) CSV-specific caveats beyond the 4 family-wide -> none for the aa file (user confirmed); rare-aa was checked by Dylan and dismissed (min count 9188 >> rare-k 100)."
  - "(per-CSV triage) Framing firmness -> initially `Firm`, revised to `Mixed` after confirming the completed stall_sites sister within_condition_binomial files were framed Mixed and the same iid-violation-binomial caveat is locked here; magnitude + cross-group reproducibility lead, tiny p-values are iid-inflated and not read at face value."
  - "(per-CSV triage) Top-hits source -> user directed Dylan to run `scripts/cross_group_concordance_tables.py` (block 17, `--min-sig 2`) directly; the three concordance tables below are transcribed from that output."
  - "(per-CSV triage) Spotlight -> none; rank by data alone per A.2.3."
  - "(readback) \"Reconciled shared content from the corrected .qmd on 2026-06-10\" -> \"No per-CSV content change: the three cross-group concordance tables (partition 20/26/14, NO raw-p), Headline, Numbers, Methods, and caveats already matched the corrected .qmd cell-for-cell; no asymptotic-with-ties entry (binomial, not wilcoxon); Olive-only Biological-interpretation/composite/plots not imported. Dylan's iid-violation-binomial caveat already carried this file's figures (188/360 -> 0.0, smallest nonzero 9.39e-323); reworded only the shared _INDEX.md family-caveat's stale stall-sites ~1e-132 to a member-agnostic underflow statement.\""
---

# Interpretation — aa_within_condition_binomial

> Source: `results/global_occupancy/analysis_corrected/aa_within_condition_binomial.csv`
> Family: `within_condition_binomial` (see [`_INDEX.md`](_INDEX.md))
> Test type: One-sample binomial vs bg_freq, BH-FDR per (group, site) family of 20 aa (source: user-confirmed)

## User directives
- (per-CSV triage) Test type -> "Confirm binomial - both files": one-sample exact binomial, `observed_count ~ Binomial(total_n, bg_freq)`, two-sided, BH-FDR per (group, site).
- (per-CSV triage) CSV-specific caveats beyond family-wide -> none for the aa file; Dylan checked rare-aa and dismissed it (min count 9188 >> rare-k 100).
- (per-CSV triage) Framing firmness -> Mixed. Initially answered Firm, then revised to Mixed after confirming the stall_sites sister within_condition_binomial files were Mixed and the same `iid-violation-binomial` caveat applies; magnitude + cross-group reproducibility lead, p magnitude does not.
- (per-CSV triage) Top-hits source -> user directed Dylan to run `cross_group_concordance_tables.py --min-sig 2` (block 17); the tables below are transcribed from its output.
- (per-CSV triage) Spotlight -> none; data-ranked only per A.2.3.
- (readback) "Reconciled shared content from the corrected .qmd on 2026-06-10" -> "No per-CSV content change: the three cross-group concordance tables (partition 20/26/14, no raw-p column), Headline, Numbers, Methods, and caveats already matched the corrected `.qmd`; no asymptotic-with-ties entry (binomial, not wilcoxon); Olive-only sections not imported. Dylan's `iid-violation-binomial` caveat already carried this file's underflow figures (188/360 -> 0.0, smallest nonzero 9.39e-323); only the shared `_INDEX.md` family-caveat's stale `~1e-132` (a stall-sites carryover) was reworded to a member-agnostic underflow statement."

## Headline
Within-group binomial against `bg_freq`, 6 groups (BWM and control x d0/d5/d10) x 3 sites (A/P/E) x 20 aa = 360 tests; BH-FDR per (group, site) family of 20. 349/360 hits at `p_adj` < 0.05 (350/360 at `p_adj` < 0.10). File min `p_adj` underflows to exactly 0.0: 188 of 360 cells return `p_adj` = 0.0 and the smallest nonzero `p_adj` is 9.39e-323 (control_day_5, E:R, `log2_enrichment` = +0.148, `observed_count` = 137894, `total_n` = 2414916). At whole-transcriptome `total_n` (~2.1-2.4M) FDR significance is near-universal: every one of the 60 (site, aa) cells clears the `#sig` >= 2 floor, so significance separates almost nothing and ranking is by magnitude and reproducibility instead. The dominant structure of this file is **cross-group concordance**: of the 60 (site, aa) cells, 20 are concordantly enriched and 26 concordantly depleted across all 6 groups, with 14 discordant; the largest-magnitude cells hold one direction across all 6 groups (E:C -0.77, E:K +0.64, P:D +0.58, P:W -0.57; the largest discordant cell A:W spans only +0.56 to -0.13). See the three cross-group sub-tables in Top hits. p_adj magnitudes are co-amplified by `iid-violation-binomial` (within-transcript correlation of overlapping E/P/A windows breaks the binomial independence assumption) and by common-aa `bg_freq` tightness; magnitude-plus-reproducibility is the anchor for reading.

## Top hits

The effect column is `log2_enrichment` = log2(observed `observed_freq` / `bg_freq`); positive means the amino acid is over-represented at that site in that group relative to its transcriptome abundance, negative under-represented. Top hits are three **cross-group** sub-tables that partition the 60 (site, aa) cells by the sign of `log2_enrichment` across the 6 groups (BWM and control x d0/d5/d10): concordant enrichment (positive in all 6), concordant depletion (negative in all 6), and discordant (>= 1 sign disagreement). A cell enters the partition only if it has a value in all 6 groups (all 60 do). Each sub-table shows every cell that reached `p_adj` < 0.05 in at least 2 of the 6 groups, a reproducibility floor on the `#sig` axis (`#sig` = groups with FDR<0.05, of 6), **not** a fixed row cap; here every cell clears it, so the three tables show all 20 / 26 / 14 cells. Rows are sorted by site (A / P / E), then `#sig` descending, then `min count` (smallest `observed_count` across the 6 groups) descending. Ranking is deliberately not by p magnitude: `iid-violation-binomial` and common-aa `bg_freq` tightness inflate p for modest effects at this `total_n` (188 cells underflow to p_adj = 0.0). In the `log2_enrichment` column the six values are listed as `BWM d0, d5, d10` then `ctrl d0, d5, d10`. The `flag` column carries `iid-amp` (a group `p_adj` < 1e-10), `bg-tight` (the aa's `bg_freq` > 0.05, a tight binomial null), and would carry `rare-aa` for any cell below 100 counts (none here).

### Concordant enrichment: significant in >= 2 of 6 groups

| site | aa | log2_enrichment (BWM d0/d5/d10; ctrl d0/d5/d10) | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- |
| A | E | BWM +0.25, +0.26, +0.39; ctrl +0.30, +0.43, +0.39 | 109935 | 6/6 | iid-amp, bg-tight |
| A | K | BWM +0.32, +0.28, +0.24; ctrl +0.33, +0.44, +0.20 | 92136 | 6/6 | iid-amp, bg-tight |
| A | D | BWM +0.13, +0.39, +0.35; ctrl +0.31, +0.38, +0.32 | 84720 | 6/6 | iid-amp, bg-tight |
| A | G | BWM +0.35, +0.12, +0.27; ctrl +0.32, +0.10, +0.26 | 82132 | 6/6 | iid-amp, bg-tight |
| A | R | BWM +0.23, +0.21, +0.24; ctrl +0.17, +0.14, +0.23 | 76722 | 6/6 | iid-amp, bg-tight |
| A | Y | BWM +0.40, +0.25, +0.15; ctrl +0.31, +0.31, +0.33 | 51487 | 6/6 | iid-amp |
| P | D | BWM +0.58, +0.52, +0.63; ctrl +0.46, +0.60, +0.66 | 107811 | 6/6 | iid-amp, bg-tight |
| P | E | BWM +0.03, +0.08, +0.22; ctrl +0.14, +0.19, +0.23 | 98276 | 6/6 | iid-amp, bg-tight |
| P | K | BWM +0.28, +0.27, +0.17; ctrl +0.36, +0.17, +0.20 | 92348 | 6/6 | iid-amp, bg-tight |
| P | G | BWM +0.56, +0.21, +0.35; ctrl +0.43, +0.21, +0.38 | 89273 | 6/6 | iid-amp, bg-tight |
| P | N | BWM +0.20, +0.28, +0.19; ctrl +0.19, +0.32, +0.27 | 74656 | 6/6 | iid-amp |
| P | Y | BWM +0.10, +0.17, +0.04; ctrl +0.18, +0.15, +0.15 | 45463 | 6/6 | iid-amp |
| P | V | BWM +0.07, +0.06, +0.08; ctrl +0.05, +0.18, +0.00 | 79150 | 5/6 | iid-amp, bg-tight |
| E | K | BWM +0.65, +0.57, +0.53; ctrl +0.69, +0.78, +0.63 | 124020 | 6/6 | iid-amp, bg-tight |
| E | E | BWM +0.19, +0.24, +0.26; ctrl +0.26, +0.35, +0.30 | 103049 | 6/6 | iid-amp, bg-tight |
| E | D | BWM +0.15, +0.18, +0.21; ctrl +0.18, +0.23, +0.23 | 79679 | 6/6 | iid-amp, bg-tight |
| E | G | BWM +0.23, +0.04, +0.16; ctrl +0.27, +0.16, +0.14 | 75866 | 6/6 | iid-amp, bg-tight |
| E | R | BWM +0.10, +0.05, +0.08; ctrl +0.07, +0.15, +0.08 | 68860 | 6/6 | iid-amp, bg-tight |
| E | N | BWM +0.08, +0.13, +0.10; ctrl +0.13, +0.24, +0.13 | 67803 | 6/6 | iid-amp |
| E | Q | BWM +0.04, +0.10, +0.11; ctrl +0.07, +0.13, +0.08 | 54903 | 6/6 | iid-amp |

### Concordant depletion: significant in >= 2 of 6 groups

| site | aa | log2_enrichment (BWM d0/d5/d10; ctrl d0/d5/d10) | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- |
| A | L | BWM -0.02, -0.11, -0.19; ctrl -0.17, -0.28, -0.13 | 99131 | 6/6 | iid-amp, bg-tight |
| A | A | BWM -0.06, -0.14, -0.15; ctrl -0.09, -0.32, -0.06 | 77106 | 6/6 | iid-amp, bg-tight |
| A | S | BWM -0.43, -0.41, -0.37; ctrl -0.48, -0.44, -0.40 | 76314 | 6/6 | iid-amp, bg-tight |
| A | T | BWM -0.32, -0.26, -0.31; ctrl -0.30, -0.26, -0.31 | 60386 | 6/6 | iid-amp, bg-tight |
| A | F | BWM -0.29, -0.20, -0.19; ctrl -0.26, -0.21, -0.17 | 52902 | 6/6 | iid-amp |
| A | C | BWM -0.29, -0.40, -0.34; ctrl -0.48, -0.38, -0.31 | 21153 | 6/6 | iid-amp |
| A | P | BWM -0.01, -0.15, -0.23; ctrl -0.11, -0.47, -0.11 | 57760 | 5/6 | iid-amp |
| P | L | BWM -0.36, -0.20, -0.27; ctrl -0.29, -0.24, -0.39 | 82488 | 6/6 | iid-amp, bg-tight |
| P | A | BWM -0.10, -0.19, -0.12; ctrl -0.14, -0.18, -0.11 | 74517 | 6/6 | iid-amp, bg-tight |
| P | S | BWM -0.48, -0.39, -0.40; ctrl -0.49, -0.40, -0.45 | 73823 | 6/6 | iid-amp, bg-tight |
| P | T | BWM -0.28, -0.23, -0.24; ctrl -0.23, -0.26, -0.22 | 64081 | 6/6 | iid-amp, bg-tight |
| P | P | BWM -0.15, -0.22, -0.17; ctrl -0.09, -0.32, -0.13 | 56757 | 6/6 | iid-amp |
| P | Q | BWM -0.20, -0.19, -0.23; ctrl -0.22, -0.38, -0.26 | 43566 | 6/6 | iid-amp |
| P | M | BWM -0.33, -0.18, -0.23; ctrl -0.20, -0.24, -0.27 | 25872 | 6/6 | iid-amp |
| P | C | BWM -0.46, -0.39, -0.35; ctrl -0.47, -0.36, -0.30 | 21371 | 6/6 | iid-amp |
| P | W | BWM -0.52, -0.51, -0.49; ctrl -0.52, -0.73, -0.63 | 9188 | 6/6 | iid-amp |
| E | L | BWM -0.19, -0.13, -0.10; ctrl -0.25, -0.20, -0.18 | 95467 | 6/6 | iid-amp, bg-tight |
| E | S | BWM -0.54, -0.35, -0.31; ctrl -0.50, -0.43, -0.39 | 77000 | 6/6 | iid-amp, bg-tight |
| E | A | BWM -0.11, -0.16, -0.12; ctrl -0.14, -0.33, -0.13 | 73460 | 6/6 | iid-amp, bg-tight |
| E | I | BWM -0.05, -0.05, -0.12; ctrl -0.06, -0.12, -0.09 | 73335 | 6/6 | iid-amp, bg-tight |
| E | T | BWM -0.25, -0.19, -0.22; ctrl -0.22, -0.33, -0.20 | 64954 | 6/6 | iid-amp, bg-tight |
| E | F | BWM -0.29, -0.23, -0.30; ctrl -0.36, -0.43, -0.35 | 46767 | 6/6 | iid-amp |
| E | Y | BWM -0.16, -0.07, -0.21; ctrl -0.16, -0.29, -0.24 | 34659 | 6/6 | iid-amp |
| E | H | BWM -0.16, -0.03, -0.04; ctrl -0.05, -0.10, -0.07 | 27661 | 6/6 | iid-amp |
| E | C | BWM -0.78, -0.71, -0.67; ctrl -0.87, -0.78, -0.82 | 14868 | 6/6 | iid-amp |
| E | W | BWM -0.21, -0.27, -0.13; ctrl -0.24, -0.09, -0.24 | 11982 | 6/6 | iid-amp |

### Discordant: significant in >= 2 of 6 groups

| site | aa | log2_enrichment (BWM d0/d5/d10; ctrl d0/d5/d10) | min count | #sig | flag |
| --- | --- | --- | --- | --- | --- |
| A | V | BWM -0.11, -0.02, -0.02; ctrl +0.01, -0.05, -0.04 | 76622 | 6/6 | iid-amp, bg-tight |
| A | I | BWM -0.22, -0.03, -0.02; ctrl -0.05, +0.12, -0.07 | 74556 | 6/6 | iid-amp, bg-tight |
| A | N | BWM -0.30, -0.06, -0.06; ctrl -0.14, +0.08, -0.26 | 51669 | 6/6 | iid-amp |
| A | Q | BWM +0.01, -0.02, -0.09; ctrl -0.05, -0.08, -0.22 | 44639 | 5/6 | iid-amp |
| A | M | BWM -0.19, -0.01, +0.01; ctrl -0.13, +0.09, -0.04 | 30255 | 5/6 | iid-amp |
| A | H | BWM +0.06, +0.03, -0.03; ctrl +0.03, -0.02, -0.01 | 28939 | 5/6 | iid-amp |
| A | W | BWM +0.56, +0.03, -0.01; ctrl +0.12, -0.13, +0.21 | 16363 | 5/6 | iid-amp |
| P | F | BWM -0.08, -0.04, -0.07; ctrl -0.15, +0.05, -0.10 | 55608 | 6/6 | iid-amp |
| P | R | BWM +0.13, +0.08, +0.10; ctrl +0.08, -0.00, +0.13 | 71588 | 5/6 | iid-amp, bg-tight |
| P | I | BWM -0.06, +0.01, -0.14; ctrl -0.03, +0.08, -0.16 | 69923 | 5/6 | iid-amp, bg-tight |
| P | H | BWM +0.02, +0.10, -0.01; ctrl +0.05, -0.06, +0.01 | 29167 | 4/6 | iid-amp |
| E | V | BWM +0.08, +0.05, +0.04; ctrl +0.04, -0.02, +0.03 | 80454 | 6/6 | iid-amp, bg-tight |
| E | P | BWM +0.20, -0.11, -0.17; ctrl +0.02, -0.31, -0.07 | 59507 | 6/6 | iid-amp |
| E | M | BWM -0.12, -0.04, +0.01; ctrl -0.07, +0.16, -0.02 | 30636 | 5/6 | iid-amp |

## Numbers at a glance
- `n_tests`: 360 (6 groups x 3 sites x 20 aa)
- `n_significant` (adjusted-p < 0.05): 349
- `n_significant` (adjusted-p < 0.10): 350
- `min adjusted-p`: 0.0 (underflow) - 188 of 360 cells return `p_adj` exactly 0.0; smallest nonzero `p_adj` is 9.39e-323 (control_day_5, E:R; log2 +0.148, k=137894, n=2414916). Under `iid-violation-binomial` + `bg-tight` this is not a magnitude claim.
- `p_floor`: n/a - no exact-test floor for the binomial at these n.
- Per-(group, site) hits at `p_adj` < 0.05 (each is 20 aa; A / P / E order; counts sum to 349):
  - BWM_day_0: A 18/20, P 20/20, E 20/20
  - BWM_day_5: A 20/20, P 19/20, E 20/20
  - BWM_day_10: A 18/20, P 19/20, E 19/20
  - control_day_0: A 20/20, P 20/20, E 20/20
  - control_day_5: A 20/20, P 19/20, E 20/20
  - control_day_10: A 19/20, P 18/20, E 20/20

## Methods
Dylan proposed a one-sample binomial vs `bg_freq` (`observed_count ~ Binomial(total_n, bg_freq)`, two-sided) with BH-FDR computed within each (group, site) family of 20 aa; the user confirmed (`test_type_source: user-confirmed`). Effect column is `log2_enrichment` = log2(`observed_freq` / `bg_freq`); a count-weighted variant `weighted_log2_enrichment` is also present in the CSV but is not used for ranking here (the unweighted log2 is what the binomial p reflects). The null `bg_freq` is the transcriptome amino-acid composition, so a positive `log2_enrichment` means "occupied more often than its transcriptomic abundance predicts," not "more occupied than control." The test answers "is amino acid X observed at this site at a different frequency than its transcriptome `bg_freq` in the same group?", *not* "is X enriched in BWM relative to control" (that is the per-timepoint Fisher) and *not* "does the per-replicate frequency change between timepoints" (that is between-timepoint Wilcoxon). Top hits are three cross-group tables partitioning the 60 (site, aa) cells by the sign of `log2_enrichment` across the 6 groups (concordant enrichment 20, concordant depletion 26, discordant 14), each showing every cell with `p_adj` < 0.05 in >= 2 of the 6 groups (all 60 cells clear this floor), sorted by site (A/P/E), `#sig` desc, `min count` desc. Ranking is by magnitude and reproducibility, not p, because at `total_n` ~2.1-2.4M the binomial p collapses toward zero (188/360 underflow to exactly 0.0) for modest effects.

## Caveats
### Confirmed
- **pseudorep** (family-wide) - replicates within a group are summed before `binomtest`; `total_n` is a pooled occupancy count rather than a sum of independent replicates, so the null treats correlated draws as independent.
- **iid-violation-binomial** (family-wide) - the binomial null assumes `total_n` iid Bernoulli draws at probability `bg_freq`; ribosome footprints are not iid (one transcript contributes overlapping E/P/A windows and transcripts vary widely in coverage). The practical effect is to compress p toward zero for every cell with a non-trivial `log2_enrichment`; here that pushes 188/360 cells to exactly 0.0 and the smallest nonzero to 9.39e-323. `log2_enrichment` is the primary effect column, not p.
- **bh-per-(group, site)** (family-wide) - multiple-testing correction is per (group, site), so FDR control is within each 20-aa family, not pooled across the 60 aa per group or across the merged E/P/A file.
- **bg-transcriptome-freq** (family-wide) - the null frequency is the transcriptome amino-acid composition (`bg_freq`), so "enrichment" means "occupied more than its transcriptomic abundance predicts," not vs a uniform expectation or vs control.

### Considered but not applicable
- **rare-aa** - Dylan checked for low-count instability (the checker flags any cell below 100 counts in any group). The smallest `min count` across all 60 cells is 9188 (P:W), so no aa cell is low-count; the flag fires on no row. (The sister codon CSV does carry a rare-codon flag, but only on the stop codon TGA.)

## For Chumeng (joint-reading hooks)
- Family: `within_condition_binomial` - sister CSV in this family that should be reconciled: `codon_within_condition_binomial.csv` (codon resolution; same design, same 4 family-wide caveats; adds codon-specific `stop-codon-instability` and `rare-codon-low-count`).
- Open questions Chumeng should resolve at synthesis time, framed as falsifiers per A.2.7:
  - The 20 concordantly enriched and 26 concordantly depleted (site, aa) cells are this file's most reproduced cross-group signals (largest: E:K, P:D enriched; E:C, P:W depleted). **Does each re-appear at codon level in `codon_within_condition_binomial.csv` with consistent direction across all 6 groups, or does the aa-level sign split unevenly across synonyms** (suggesting one codon, not the aa, drives it)? A clean synonym split would move the reading from "aa property" toward "codon-level decoding."
  - The within-group binomial is by construction blind to BWM-vs-control divergence (each group is tested against its own `bg_freq`). **Do the same concordant cells reappear with consistent direction in `per_timepoint_fisher_aa.csv` (BWM-vs-control at each timepoint)?** A concordant binomial cell that is *null* in the Fisher contrast at that (timepoint, site, aa) is the simpler "stable occupancy property" reading; a concordant binomial cell that is *also* a large-OR Fisher hit needs a perturbation overlay (or is a large-N artefact - check imbalanced-N).
  - The 14 discordant cells (largest swing A:W, +0.56 to -0.13) are where sign disagrees across the 6 groups. **Does any discordant cell flip in a group-consistent way that a within-condition or per-timepoint Fisher contrast also picks up**, or is the discordance confined to small-magnitude cells near zero (then it is sign noise, not biology)?
  - File min `p_adj` is exactly 0.0 for 188 cells. **Does the magnitude ordering here (E:C, E:K, P:D, P:W on top) match the magnitude ordering of the same cells in any large-N Fisher file**, or does the Fisher pick a different set? Divergent orderings would mean the binomial magnitude is a within-group abundance readout that the between-group contrast does not see.
