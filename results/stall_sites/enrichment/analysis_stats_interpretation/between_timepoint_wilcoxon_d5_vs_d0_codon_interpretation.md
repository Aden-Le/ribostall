---
input_csv: results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_codon.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U / Wilcoxon rank-sum (two-sided)
test_type_source: user-confirmed
n_tests: 183
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.4357142857142857
p_floor: 0.02857142857142857
pseudoreplicated: null
synced_from_olive_qmd: 2026-06-01
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` of medians only."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "61 codons per site → BH wall ~3x more stringent than aa."}
  - {label: "low-count-rare-codon-instability", proposed_by: dylan, status: confirmed, why: "Top |effect| codons here include A-TTA (-1.341), P-GCG (-1.212), P-GGG (+1.001) — all median freq < 0.2%."}
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: ruled_out, why: "Empirically verified via scripts/_for_claude_mw_branch_audit.py --design between_timepoint --timepoints day5 day0 --level codon: only 2/183 (site,codon) tests have pooled-sample rank ties; scipy auto picked asymptotic for those 2 and exact for the other 181, matching the pipeline CSV to ~8e-17; forcing asymptotic for all 183 shifts raw p by at most 0.124 (median 0.006), flips zero raw-p<0.05 decisions and leaves 0 hits at FDR<0.05 (min p_adj A/P/E ~= 0.71/0.46/0.90 vs as-shipped 0.77/0.44/0.93)."}
headline: "No statistically significant differences at FDR<0.05 (0/183) for codon-level d5-vs-d0 MW with reps pooled across BWM and control per timepoint; min p_adj = 0.436 at site P — 4 floor rows (CTA -0.923, GAG +0.190, GCA -0.740, GCG -1.212). Five total floor rows file-wide (1 at A: ACT +0.227; 4 at P; 0 at E). Min raw p = 0.0286 (n=4 vs n=4 floor); 'no FDR hits' is structural per the locked floor caveat."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 codon files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-01 → adopted Olive's six per-(site, direction) Top-hits sub-tables in A/P/E order with the added `aa` and raw `p_value` columns; de-jargoned the Headline ('Firm null' → 'No statistically significant differences'); synced the `asymptotic-with-ties` ruled-out caveat. Every number enumerated against the .qmd/CSV — 0 corrections (all this-file values reconcile)."
---

# Interpretation — between_timepoint_wilcoxon_d5_vs_d0_codon

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d5_vs_d0_codon.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched 3 codon files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed.
- (readback) Reconciled shared content from the corrected .qmd on 2026-06-01: adopted Olive's six per-(site, direction) Top-hits sub-tables (A/P/E order) with the added `aa` and raw `p_value` columns; de-jargoned the Headline ("Firm null" → "No statistically significant differences"); synced the `asymptotic-with-ties` ruled-out caveat. Every number enumerated and verified against the .qmd/CSV — no corrections (all this-file values reconcile verbatim).

## Headline
No statistically significant differences at FDR<0.05 (0/183) for codon-level day_5 vs day_0 MW with BWM and control reps pooled within each timepoint (n=4 per side). Min p_adj = 0.436 at site P — 4 floor rows (CTA -0.923, GCG -1.212, GCA -0.740, GAG +0.190), giving BH wall 61*0.0286/4 = 0.436. File-wide: 5 floor rows (4 at P, 1 at A: ACT +0.227; 0 at E). Site E completely flat (min p_adj=0.926); site A min p_adj=0.775. Closest-to-significant codon-level signature: at site P, three depletions (CTA, GCG, GCA) and one enrichment (GAG) all reaching the floor between day_0 and day_5 reps. Treat as exploratory; the floor blocks formal FDR<0.05.

## Top hits

Effect column is `log2_FC` (day_5/day_0 median ratio); `p_value` is the raw Mann-Whitney p; `p_adj` is BH-corrected per A/P/E site (61-codon family). Each site is split into one sub-table per sign of effect (positive `log2_FC` = day_5-enriched, negative = day_5-depleted); within each, up to 5 rows ranked by raw `p_value` ascending, `|log2_FC|` descending as the tiebreaker. The `low-count` flag is applied when `min(median_day_5, median_day_0) < 0.005`. The `aa` column is the single-letter amino-acid translation of each codon.

### A site — enriched (day_5 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| ACT | T | +0.227 | 0.0286 | 0.775 | floor |
| AAC | N | +0.672 | 0.1143 | 0.775 |  |
| ATC | I | +0.470 | 0.2000 | 0.775 |  |
| TCT | S | +0.250 | 0.2000 | 0.775 |  |
| CAA | Q | +0.158 | 0.2000 | 0.775 |  |

### A site — depleted (day_5 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| TTA | L | -1.341 | 0.1143 | 0.775 | low-count |
| GCA | A | -0.713 | 0.1143 | 0.775 |  |
| GGC | G | -0.683 | 0.1143 | 0.775 | low-count |
| CTG | L | -0.652 | 0.1143 | 0.775 | low-count |
| TGG | W | -0.671 | 0.2000 | 0.775 |  |

### P site — enriched (day_5 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GAG | E | +0.190 | 0.0286 | 0.436 | floor |
| GGG | G | +1.001 | 0.1143 | 0.775 | low-count |
| TCT | S | +0.388 | 0.1143 | 0.775 |  |
| TCC | S | +0.444 | 0.2000 | 0.871 |  |
| GTC | V | +0.335 | 0.2000 | 0.871 |  |

### P site — depleted (day_5 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GCG | A | -1.212 | 0.0286 | 0.436 | floor, low-count |
| CTA | L | -0.923 | 0.0286 | 0.436 | floor, low-count |
| GCA | A | -0.740 | 0.0286 | 0.436 | floor, low-count |
| GGT | G | -0.115 | 0.0571 | 0.697 |  |
| AGG | R | -0.729 | 0.1143 | 0.775 | low-count |

### E site — enriched (day_5 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| TCC | S | +0.442 | 0.0571 | 0.926 |  |
| CAA | Q | +0.433 | 0.2000 | 0.926 |  |
| TCT | S | +0.385 | 0.2000 | 0.926 |  |
| CAG | Q | +0.312 | 0.2000 | 0.926 |  |
| ACT | T | +0.199 | 0.2000 | 0.926 |  |

### E site — depleted (day_5 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CCA | P | -0.525 | 0.1143 | 0.926 |  |
| ACC | T | -0.143 | 0.1143 | 0.926 |  |
| GCG | A | -0.610 | 0.2000 | 0.926 | low-count |
| CTA | L | -0.398 | 0.2000 | 0.926 | low-count |
| AGA | R | -0.270 | 0.2000 | 0.926 |  |

## Numbers at a glance
- `n_tests`: 183
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.436 (P-site, BH wall for 4 floor rows in 61-codon family)
- `min raw-p`: 0.02857 (= MW exact floor); 5 rows at the floor — A: ACT; P: CTA, GAG, GCA, GCG
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.775 (1 floor row)
  - P site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.436 (4 floor rows)
  - E site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.926 (0 floor rows)

## Methods
Dylan proposed Mann-Whitney U / Wilcoxon rank-sum two-sided on per-replicate codon frequencies, n=4 day_5 (BWM_day5_rep2/3 + control_day5_rep2/3) vs n=4 day_0 (BWM_day0_rep2/3 + control_day0_rep2/3), BH-FDR per site (each A/P/E site = 61-codon family); user confirmed. Effect column is `log2_FC` of medians; test statistic is `U_stat`. The test answers "do day_5 and day_0 reps differ in per-rep codon frequency at this site?", with BWM and control reps pooled within each timepoint. Does *not* answer "BWM-vs-control at any single day" (that is `per_timepoint_fisher_codon`) and does *not* answer "BWM moves from day_0 to day_5 holding condition fixed" (that is `timepoint_fisher_within_condition_d5_vs_d0_codon`).

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide).
- **condition-pooled-confound** (family-wide).
- **n=4-low-power** (family-wide).
- **p-floor-aware-headline** (family-wide).
- **weighted_log2_enrichment-absent** (per-CSV).
- **larger-bh-family** (per-CSV) — 61 codons per site means BH wall is ~3x further from FDR<0.05 vs aa-resolution. The aa-level d5_vs_d0 file gets to p_adj=0.571 with 1 floor row; here at codon level, 4 floor rows still only get to p_adj=0.436.
- **low-count-rare-codon-instability** (per-CSV) — extreme |effect| at A-TTA (-1.341), P-GCG (-1.212), P-GGG (+1.001), P-CTA (-0.923) all sit on rare codons (median freq < 0.2%).

### Considered but not applicable
- **asymptotic-with-ties** — empirically ruled out. Audit `scripts/_for_claude_mw_branch_audit.py --design between_timepoint --timepoints day5 day0 --level codon`: 2/183 (site,codon) tests have pooled-sample rank ties. scipy auto picked asymptotic for those 2 and exact for the other 181; recomputed p matches the pipeline CSV to ~8e-17. Forcing asymptotic for all 183 shifts raw p by at most 0.124 (median 0.006), flips zero raw-p<0.05 decisions, and leaves 0 hits at FDR<0.05 either branch (min p_adj A/P/E ~= 0.71/0.46/0.90 asymptotic vs as-shipped 0.77/0.44/0.93). Branch choice does not affect any FDR-level conclusion.

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs: `between_timepoint_wilcoxon_d5_vs_d0_aa.csv` (aa, same contrast), and the d10_vs_d0 / d10_vs_d5 pairs at both resolutions.
- Open questions Chumeng should resolve at synthesis time:
  - **The aa-level d5_vs_d0 file's only floor row is P-site C (+0.249) — but C has codons TGT, TGC, neither of which is at the floor here.** P-TGC: log2=+0.068, p=0.486. P-TGT: log2=+0.427, p=0.686. So the aa-level P-C signal is actually distributed across both Cys codons but neither individually clears the codon BH (3x more stringent at codon level). aa-level enrichment is real-but-codon-balanced, not single-codon-driven.
  - **Site P codon hits here (CTA depleted, GCG depleted, GCA depleted, GAG enriched) are NOT visible at the aa level**: P-A is +0.021 (p=1.0), P-E is +0.268 (p=0.343), P-L (codons including CTA) is +0.004 (p=0.686). This is the inverse of the previous question: codon-level signals at site P that aa-level aggregation hides. Only one or two synonymous codons of A/L/E carry the day-5-vs-day-0 shift.
  - Cross-contrast tracking — d10_vs_d0 codon's only P-site floor row was GAG (+0.387); d5_vs_d0 P-GAG is +0.190 (floor). The numerical sequence is consistent with several patterns (monotonic accumulation, biphasic, two independent floor coincidences). Does P-GAG show coherent direction in `per_timepoint_fisher_codon` and `timepoint_fisher_within_condition_*_codon` files? If yes across both contrast designs → Chumeng decides whether to elevate to a consensus feature; if only one design → reading is contrast-design-dependent.
  - Site E completely flat at every contrast in this family (d10_vs_d0 codon: 1 floor row, p_adj=0.93; d10_vs_d5 codon: 1 floor, p_adj=0.87; d5_vs_d0 codon: 0 floor, p_adj=0.93). Suggests that whatever timepoint shifts exist concentrate at the A and P sites, not E.
