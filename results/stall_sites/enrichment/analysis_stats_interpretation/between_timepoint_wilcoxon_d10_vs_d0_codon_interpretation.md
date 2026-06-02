---
input_csv: results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d0_codon.csv
family: between_timepoint_wilcoxon
test_type: Mann-Whitney U / Wilcoxon rank-sum (two-sided)
test_type_source: user-confirmed
n_tests: 183
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.4979591836734693
p_floor: 0.02857142857142857
pseudoreplicated: null
synced_from_olive_qmd: 2026-05-30
caveats:
  - {label: "mw-floor-blocking", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "condition-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "n=4-low-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "p-floor-aware-headline", proposed_by: family, status: confirmed, why: "Inherited from family `between_timepoint_wilcoxon` (see _INDEX.md)."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` of medians only; no count-weighted variant."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "~61 codons per site → BH-FDR family of ~61 vs ~20 at aa; with the same raw-p floor, codon BH wall is ~3x more stringent."}
  - {label: "low-count-rare-codon-instability", proposed_by: dylan, status: confirmed, why: "Rare codons (TTA, CTA, AGG, CGG, GCG, ATA) have median freqs < 0.2%; small absolute changes give large log2_FC swings. Top-|effect| codons here (TTA -1.20, GCA -1.07 at A; CTA -1.24, GCG -1.03 at P) all sit in this regime."}
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: ruled_out, why: "Empirically verified via scripts/_for_claude_mw_branch_audit.py --design between_timepoint --timepoints day10 day0 --level codon: only 4/183 (site,codon) tests have pooled-sample rank ties (ATA@E, CGG@E, TTA@E, TTA@P), all far from raw p<0.05; scipy auto picked asymptotic for those 4 and exact for the other 179, matching the pipeline CSV to ~1e-16; forcing asymptotic for all 183 flips zero raw-p<0.05 decisions and leaves 0 hits at FDR<0.05 (min p_adj A/P/E ~= 0.49/0.69/0.90 vs as-shipped 0.50/0.69/0.93)."}
headline: "No statistically significant differences at FDR<0.05 (0/183) for codon-level d10-vs-d0 MW with BWM and control reps pooled per timepoint; min raw p = 0.0286 (n=4 vs n=4 floor), 3 floor rows total — E-site TCC (+0.589), P-site GAG (+0.387), P-site TGC (+0.219). Min p_adj = 0.498 at A site (no floor row; the A-site BH wall is set by 14 codons at p_adj = 0.498 — 12 tied at raw p = 0.114 plus CGT and GCA at raw p = 0.057). 'No FDR hits' is structural per the locked floor caveat, not a biological negative."
user_directives:
  - "(triage, batched across all 6 family members) Test type confirmation — `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → `Confirm`."
  - "(triage, batched across the 3 codon files) CSV-specific caveats → `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-05-30 → adopted Olive's six per-(site, direction) Top-hits sub-tables in A/P/E order with the added `aa` and raw `p_value` columns; every number enumerated and verified against the .qmd/CSV, no values changed."
---

# Interpretation — between_timepoint_wilcoxon_d10_vs_d0_codon

> Source: `results/stall_sites/enrichment/analysis_stats/between_timepoint_wilcoxon_d10_vs_d0_codon.csv`
> Family: `between_timepoint_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage, batched) Test type: `MW / Wilcoxon rank-sum two-sided, n=4 vs n=4, BH-FDR per site` → "Confirm".
- (triage, batched 3 codon files) CSV-specific caveats: `weighted_log2_enrichment-absent`, `larger-bh-family`, `low-count-rare-codon-instability` confirmed.
- (readback) Reconciled shared content from the corrected .qmd on 2026-05-30: adopted Olive's six per-(site, direction) Top-hits sub-tables (A/P/E order) with the added `aa` and raw `p_value` columns. Every number enumerated and verified against the .qmd/CSV; no values changed.

## Headline
No statistically significant differences at FDR<0.05 (0/183) for codon-level day_10 vs day_0 MW with BWM and control reps pooled within each timepoint (n=4 per side). Three floor rows in the whole file: E-TCC (+0.589), P-GAG (+0.387), P-TGC (+0.219). Min p_adj = 0.498 at site A: the A-site BH wall is set by 14 codons reaching p_adj = 0.498 — 12 tied at raw p = 0.114 plus CGT and GCA at raw p = 0.057 — i.e. 0.114 * 61/14 ~= 0.498. Site A has the densest near-floor signal but no floor rows; site E has 1 floor row and site P has 2, yet min p_adj climbs to 0.93 (E) / 0.69 (P). Treat as exploratory leads only; the per-timepoint Fisher (`per_timepoint_fisher_codon`) at day_0 and day_10 is the test that can actually resolve a BWM-vs-control codon shift at either timepoint.

## Top hits

Effect column is `log2_FC` (day_10/day_0 median ratio); `p_value` is the raw Mann-Whitney p; `p_adj` is BH-corrected per A/P/E site (61-codon family). Each site is split into one sub-table per sign of effect (positive `log2_FC` = day_10-enriched, negative = day_10-depleted); within each, up to 5 rows ranked by raw `p_value` ascending, `|log2_FC|` descending as the tiebreaker. The `low-count` flag is applied when `min(median_day_10, median_day_0) < 0.005`. The `aa` column is the single-letter amino-acid translation of each codon.

### A site — enriched (day_10 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CGT | R | +0.276 | 0.0571 | 0.498 |  |
| CGC | R | +0.640 | 0.1143 | 0.498 |  |
| AAC | N | +0.595 | 0.1143 | 0.498 |  |
| ATC | I | +0.518 | 0.1143 | 0.498 |  |
| ACC | T | +0.295 | 0.1143 | 0.498 |  |

### A site — depleted (day_10 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GCA | A | -1.065 | 0.0571 | 0.498 | low-count |
| TTA | L | -1.201 | 0.1143 | 0.498 | low-count |
| GTG | V | -0.901 | 0.1143 | 0.498 |  |
| CTG | L | -0.800 | 0.1143 | 0.498 | low-count |
| GGC | G | -0.639 | 0.1143 | 0.498 | low-count |

### P site — enriched (day_10 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GAG | E | +0.387 | 0.0286 | 0.688 | floor |
| TGC | C | +0.219 | 0.0286 | 0.688 | floor |
| CCC | P | +0.298 | 0.0571 | 0.688 | low-count |
| GCC | A | +0.221 | 0.0571 | 0.688 |  |
| GGG | G | +1.147 | 0.1143 | 0.688 | low-count |

### P site — depleted (day_10 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GCG | A | -1.031 | 0.0571 | 0.688 | low-count |
| CTA | L | -1.242 | 0.1143 | 0.688 | low-count |
| GCA | A | -0.698 | 0.1143 | 0.688 | low-count |
| ATT | I | -0.578 | 0.1143 | 0.688 |  |
| GGT | G | -0.146 | 0.1143 | 0.688 |  |

### E site — enriched (day_10 > day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| TCC | S | +0.589 | 0.0286 | 0.930 | floor |
| CTC | L | +0.490 | 0.2000 | 0.930 |  |
| CAG | Q | +0.267 | 0.2000 | 0.930 |  |
| TGC | C | +0.220 | 0.2000 | 0.930 | low-count |
| GCT | A | +0.113 | 0.2000 | 0.930 |  |

### E site — depleted (day_10 < day_0)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| GCG | A | -1.204 | 0.0571 | 0.930 | low-count |
| AGG | R | -0.767 | 0.1143 | 0.930 | low-count |
| AAA | K | -0.468 | 0.1143 | 0.930 |  |
| CGA | R | -0.678 | 0.2000 | 0.930 | low-count |
| ATG | M | -0.384 | 0.2000 | 0.930 |  |

## Numbers at a glance
- `n_tests`: 183
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.498 (A site; BH wall set by 14 A-site codons at p_adj = 0.498 — 12 tied at raw p = 0.114 plus CGT and GCA at raw p = 0.057)
- `min raw-p`: 0.02857 (= MW exact floor); 3 rows at the floor (P-GAG +0.387, P-TGC +0.219, E-TCC +0.589)
- `p_floor`: 0.02857 (n=4 vs n=4 two-sided exact)
- Per-site BH families:
  - A site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.498 (no floor rows)
  - P site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.688 (2 floor rows)
  - E site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.930 (1 floor row)

## Methods
Dylan proposed Mann-Whitney U / Wilcoxon rank-sum two-sided on per-replicate frequencies, n=4 day_10 (BWM_day10_rep2/3 + control_day10_rep2/3) vs n=4 day_0, BH-FDR per site (each A/P/E site = 61-codon family); user confirmed. Effect column is `log2_FC` of medians; test statistic is `U_stat`. The test answers "do day_10 and day_0 reps differ in per-rep codon frequency at this site?", with BWM and control reps pooled within each timepoint. Does *not* answer "BWM-vs-control at any single day" (that is `per_timepoint_fisher_codon`) and does *not* answer "BWM moves from day_0 to day_10 holding condition fixed" (that is `timepoint_fisher_within_condition_d10_vs_d0_codon`).

## Caveats
### Confirmed
- **mw-floor-blocking** (family-wide) — MW two-sided exact p-floor for n=4 vs n=4 = 0.02857. With 61 codons per site, BH wall for one floor hit is 61 * 0.0286 ~= 1.74; for two floor hits at the same site, 0.85; for three, 0.57. Even all 61 codons tied at the floor would BH to 0.0286 — the only path to FDR<0.05.
- **condition-pooled-confound** (family-wide) — n=4 per timepoint = 2 BWM + 2 control reps; condition-by-time interactions cancel inside MW.
- **n=4-low-power** (family-wide) — the no-signal result is weakly informative.
- **p-floor-aware-headline** (family-wide) — the headline above explicitly states the floor.
- **weighted_log2_enrichment-absent** (per-CSV) — `log2_FC` of medians only.
- **larger-bh-family** (per-CSV) — 61 vs 20 means the codon BH wall is ~3x further from FDR<0.05 for any given count of floor hits than the aa BH wall for the same contrast. Empirically: aa min p_adj = 0.571 (1 floor row), codon min p_adj = 0.498 (no floor rows but a 14-codon block).
- **low-count-rare-codon-instability** (per-CSV) — top |effect| ranks at site A (TTA -1.20, GCA -1.07, CTG -0.80, GGC -0.64) and site P (CTA -1.24, GCG -1.03, GGG +1.15) are dominated by codons with median freq < 0.5%. A few absolute counts moving in/out of the stall set produces large log2_FC; do not over-read. (GTG -0.90 at A is the largest-|effect| A-site cell that is *not* low-count: min median ~0.006, above the 0.005 threshold.)

### Considered but not applicable
- **asymptotic-with-ties** — empirically ruled out. Audit `scripts/_for_claude_mw_branch_audit.py --design between_timepoint --timepoints day10 day0 --level codon`: 4/183 (site,codon) tests have pooled-sample rank ties (ATA@E, CGG@E, TTA@E, TTA@P), all far from raw p<0.05 (smallest tied-test p = 0.124 at TTA@P). scipy auto picked asymptotic for those 4 and exact for the other 179; recomputed p matches the pipeline CSV to ~1e-16. Forcing asymptotic for all 183 shifts raw p by at most 0.102 (median 0.006), flips zero raw-p<0.05 decisions, and leaves 0 hits at FDR<0.05 either branch (min p_adj A/P/E ~= 0.49/0.69/0.90 asymptotic vs as-shipped 0.50/0.69/0.93). Branch choice does not affect any FDR-level conclusion.

## For Chumeng (joint-reading hooks)
- Family: `between_timepoint_wilcoxon` — sister CSVs in this family that should be reconciled: `between_timepoint_wilcoxon_d10_vs_d0_aa.csv` (aa resolution, same contrast), `between_timepoint_wilcoxon_d10_vs_d5_aa/codon.csv`, `between_timepoint_wilcoxon_d5_vs_d0_aa/codon.csv`.
- Open questions Chumeng should resolve at synthesis time:
  - Codon-vs-aa for the day_10 vs day_0 contrast: P-aa C (+0.405, floor) — do its codons TGT and TGC behave alike here? TGC is at the floor (+0.219, raw p = 0.0286); TGT moves the same direction but larger and non-floor (+0.481, raw p = 0.486). The discrete-floor P-C signal is carried by the synonymous codon TGC specifically. Codon usage shift at P-site cysteine.
  - P-aa E (+0.373, near-floor) — synonyms GAA (raw p=1.0, +0.021) and GAG (floor, +0.387). The aa-level E signal is entirely on GAG, not GAA. Codon usage shift, not aa shift.
  - E-site TCC (+0.589, floor): is this the codon-level fingerprint of E-aa S (which was unremarkable at aa level)? S has 6 codons (TCT, TCC, TCA, TCG, AGT, AGC); only TCC is at the floor.
  - Site A's 14-codon "near-significant" block: if the same codons appear at floor in `per_timepoint_fisher_codon` (day_0 or day_10) with consistent direction → BWM-vs-control codon signal that MW couldn't resolve. If they don't → likely a between-day frequency drift independent of perturbation.
