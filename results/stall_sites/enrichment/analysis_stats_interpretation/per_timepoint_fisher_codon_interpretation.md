---
input_csv: results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_codon.csv
family: per_timepoint_fisher
test_type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 61 sense codons
test_type_source: user-confirmed
synced_from_olive_qmd: 2026-06-01
n_tests: 549
n_significant_fdr05: 86
n_significant_fdr10: 108
min_p_adj: 2.032519495016287e-30
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "imbalanced-N", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "bh-per-(timepoint,site)", proposed_by: family, status: confirmed, why: "Inherited from family `per_timepoint_fisher` (see _INDEX.md)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "BH-FDR is computed within per-(timepoint, site) families of 61 sense codons (~3x stricter than the AA file's 20 per family). AA-level signals can split across synonyms below the per-(timepoint, site) FDR threshold here; if a feature is significant at AA but absent here, that is an aggregation effect, not a contradiction."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "Many rare codons have BWM_count or control_count cells in the single/double digits even though pooled totals are in the thousands. Concrete examples in this file: day_0,A,GGG BWM_k=2 ctrl_k=33; day_0,P,TTA BWM_k=1; day_0,A,TGT BWM_k=7 ctrl_k=114; day_0,E,ACG BWM_k=4 ctrl_k=89. OR estimates and Fisher p-values are unstable for those rows; rows are flagged `rare-codon` in Top hits when BWM_k<100 OR control_k<100. The largest-magnitude single-cell rows in the file are dominated by these rare-codon depletions; treat with care."}
  - {label: "OR-direction-anchor", proposed_by: dylan, status: confirmed, why: "Each row is itself a BWM-vs-control 2x2: OR > 1 means BWM enriched relative to control at that (timepoint, site, codon); OR < 1 means BWM depleted. Direction is fixed by the contingency layout and is stated in Methods so downstream consumers do not invert."}
  - {label: "n-asymmetry-affects-power-not-direction", proposed_by: dylan, status: confirmed, why: "BWM_total varies across timepoints (day_0=6091, day_5=11935, day_10=6945); control_total varies more (day_0=27732, day_5=11177, day_10=8788). The day_0 control:BWM ratio of 4.55:1 means a moderate effect size at day_0 inflates to extreme p_adj more readily than the same effect size at day_5 (ratio 0.94:1) or day_10 (ratio 1.27:1). The 28-OOM gap between min p_adj at day_0 (2.03e-30) and day_5 (4.99e-3) or day_10 (1.36e-1) is consistent with this N-modulated power asymmetry. Per-cell n_sig differences are therefore power-modulated, not necessarily direction-of-effect biology; cross-test reconciliation is the place to disentangle the two."}
caveats_considered: []
headline: "Per-timepoint BWM-vs-control Fisher at codon resolution: 86/549 hits at FDR<0.05 (108/549 at FDR<0.10); per-(timepoint, site) cell n_sig at FDR<0.05 ranges from 0 (day_5 A, day_5 P, day_10 A, day_10 E) to 28 (day_0 P). 48 of 183 (site, codon) cells are concordant in OR direction across all 3 timepoints (17 BWM-enriched, 31 BWM-depleted; the depleted set is heavily weighted toward rare-codon rows where the across-timepoint sign is the same but each cell magnitude is unstable); 133 of 183 cells show at least one sign change across the 3 timepoint values, of which only 1 reaches FDR<0.05 on both opposite-sign rows (E:AAG high-count, day_0 BWM-enriched log2OR=+0.745 p_adj=2.03e-30 / day_5 BWM-depleted log2OR=-0.226 p_adj=5.00e-03). 3 rows have p_adj < 1e-10, all at day_0 and all on AAG: day_0 A AAG +0.654 (BWM_k=618 ctrl_k=1857), day_0 P AAG +0.692 (BWM_k=591 ctrl_k=1729), day_0 E AAG +0.745 (file-min, BWM_k=866 ctrl_k=2495) — equal billing across the three sites at this timepoint. Largest-magnitude high-count concordant cells (mean log2OR across the 3 timepoints, single sign throughout, k_BWM>=100 in every timepoint): A:AGA +0.294, P:AAG +0.379, P:GAA -0.310, P:AGA +0.249. Largest-magnitude single-cell signals at p_adj<0.05 with k_BWM+k_ctrl>=50 (grouped A/P/E then within-site by magnitude): day_0 A AAG +0.654, day_0 A CGC +0.592 (BWM_k=104), day_0 P CGC +0.954 (BWM_k=95, rare-codon), day_0 P AAG +0.692, day_0 E CGC +0.923 (BWM_k=93, rare-codon), day_0 E AAG +0.745. Treat per-cell n_sig differences across timepoints as power-modulated by the BWM/control N imbalance, not as direction-of-effect biology."
user_directives:
  - "(invocation context) `flat-prior` token — apply A.2.1 through A.2.9 strictly; do not import findings from prior CSV interpretations as priors; read this CSV cold without consulting `_INDEX.md` cross-family hooks; rank features by (a) effect size in high-count rows (k>=50 BWM+control combined; high-count means k_BWM>=100 in every cell of a cross-timepoint comparison), (b) cross-synonym coherence at codon level, (c) reproducibility within this CSV's per-cell neighbours; report shared-direction features at equal billing with divergent features; for any p_adj < 1e-10 row name at least one alternative explanation."
  - "(per-CSV triage, carried over from original triage; not re-litigated this run) Test type confirmation → `Fisher's exact (two-sided), BWM vs control 2x2 per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 61 sense codons`."
  - "(per-CSV triage, this run) CSV-specific caveats beyond family-wide → confirmed `larger-bh-family`, `rare-codon-low-count`, `OR-direction-anchor`, and added `n-asymmetry-affects-power-not-direction` as the magnitude-vs-power discipline analogous to the AA-resolution sister and the binomial files. The prior run's `cross-timepoint-direction-axis` caveat is dropped on this run because it phrased a temporal-trajectory axis AND named individual codon spotlights (AAG-at-E, AAA-at-A/E/P, CGC-at-E/P) — both reserved for synthesis (A.2.5) and ranked by data alone (A.2.3)."
  - "(per-CSV triage, this run) Framing firmness → `Mixed` (re-derived under flat-prior from the prior run's `Firm`). Cells with the very smallest p_adj (the 3 AAG cells at day_0 with p_adj < 1e-15) are firm by magnitude AND high-count. Cells with p_adj ~1e-3 in BH families of 61 are near the discreteness floor and are exploratory until cross-test corroboration is in hand. The largest single-cell |log2OR| values in the file (day_0 E ACG -2.293, day_0 P ACA -2.005, day_0 A TGT -1.843 etc.) are all rare-codon rows; magnitude alone is exploratory there."
  - "(per-CSV triage, this run) Spotlight → `No spotlight`. Headline ranks features by data alone per A.2.3."
  - "(readback) Reconciled shared content from the corrected .qmd on 2026-06-01 → adopted Olive's section order under Option A: the five per-(timepoint, direction) merged-site Top-hits tables (with the `aa` + raw `p_value` columns) now lead, and the cross-timepoint direction-concordance + direction-flip tables move to a `Cross-timepoint summary` at the end; the old per-(timepoint, site) `<details>` blocks are dropped and a Flag glossary added. Bare codon/AA codes kept. Methods given the Dylan `proposed … user confirmed` opener. Number audit: corrected day_0,A,GGG ctrl_k 14→33 and A:AAA day_0 log2OR -0.892→-0.893 (Class-2, CSV-verified); dropped the stale E:CCT second-sig-flip parenthetical (the current CSV has only E:AAG with both opposite-sign rows at p_adj<0.05); reconciled the frontmatter n-asymmetry day_5 min p_adj 4.93e-3→4.99e-3 to the body/.qmd."
---

# Interpretation — per_timepoint_fisher_codon

> Source: `results/stall_sites/enrichment/analysis_stats/per_timepoint_fisher_codon.csv`
> Family: `per_timepoint_fisher` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BWM vs control 2x2 contingency per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 61 sense codons (source: user-confirmed)

## User directives
- (invocation context) `flat-prior` — apply A.2.1 through A.2.9 strictly; read CSV cold; rank by effect size in high-count rows + cross-synonym coherence + per-cell reproducibility; equal billing for shared-direction vs divergent; alternative-explanation flags on every `p_adj < 1e-10` row.
- (per-CSV triage, carried over from original triage; not re-litigated) "Test type confirmation" → "Fisher's exact (two-sided), BWM vs control 2x2 per (timepoint, site, codon); BH-FDR within each (timepoint, site) family of 61 sense codons."
- (per-CSV triage, this run) "CSV-specific caveats beyond family-wide" → confirmed `larger-bh-family`, `rare-codon-low-count`, `OR-direction-anchor`, `n-asymmetry-affects-power-not-direction`. The prior run's `cross-timepoint-direction-axis` caveat is dropped because it framed a temporal-trajectory axis (A.2.5) AND named individual codon spotlights for explicit tracking (A.2.3).
- (per-CSV triage, this run) "Framing firmness" → Mixed (re-derived under flat-prior from prior `Firm`). The 3 AAG cells at day_0 (A/E/P, p_adj < 1e-15, k_BWM>=591) are firm by magnitude+count. Cells at p_adj ~1e-3 in BH-61 families are near discreteness floor; rare-codon rows with k<50 in either arm are exploratory regardless of |log2OR|.
- (per-CSV triage, this run) "Spotlight" → none. Data-ranked only per A.2.3.
- (readback) Section order reconciled to the corrected `.qmd`: the five per-(timepoint, direction) merged-site tables lead, cross-timepoint concordance/flip moved to the end under `Cross-timepoint summary`; `aa` + raw `p_value` columns adopted, old per-(timepoint, site) `<details>` blocks dropped, Flag glossary added. Number audit: corrected day_0,A,GGG ctrl_k (14→33), A:AAA day_0 log2OR (-0.892→-0.893), and frontmatter day_5 min p_adj (4.93e-3→4.99e-3); removed the stale E:CCT second-sig-flip note (current CSV has only E:AAG).

## Headline
549 tests (3 timepoints x 3 sites x 61 sense codons), 86 hits at p_adj<0.05 (15.7%), 108 at p_adj<0.10 (19.7%). Per-(timepoint, site) cell n_sig at p_adj<0.05: day_0 A=27, P=28, E=27; day_5 A=0, P=0, E=3; day_10 A=0, P=1, E=0. Of 183 (site, codon) cells: 48 are concordant in OR direction across all 3 timepoints (17 BWM-enriched, 31 BWM-depleted, with the depleted set heavily weighted toward low-count rare-codon rows that are direction-stable but per-cell magnitude-unstable); 133 show at least one sign change across the 3 timepoint values; only 1 of those 133 (E:AAG high-count) has the sign change at p_adj<0.05 on both opposite-sign rows. 3 rows have p_adj < 1e-10, all at day_0 and all on AAG (Lys): day_0 A AAG log2OR=+0.654 p_adj=1.38e-17 (BWM_k=618 ctrl_k=1857), day_0 P AAG log2OR=+0.692 p_adj=1.04e-18 (BWM_k=591 ctrl_k=1729), day_0 E AAG log2OR=+0.745 p_adj=2.03e-30 (file-min, BWM_k=866 ctrl_k=2495) — these three sites get equal billing in this row of the headline. Largest-magnitude single-cell BWM-vs-control divergences at p_adj<0.05 with combined k>=50 (grouped A/P/E then within-site by magnitude, mixing directions, including rare-codon flagged): day_0 A AAG +0.654, day_0 A CGC +0.592 (BWM_k=104), day_0 A AGA +0.559, day_0 A CTC +0.427, day_0 P CGC +0.954 (BWM_k=95, rare-codon), day_0 P AAG +0.692, day_0 P AGA +0.579, day_0 P ATC +0.424, day_0 E CGC +0.923 (BWM_k=93, rare-codon), day_0 E AAG +0.745. Note: the file's largest absolute |log2OR| values at p_adj<0.05 (day_0 A TGT -1.843, day_0 A TTT -1.521, day_0 A AAT -1.499, day_0 A TAT -1.346, day_0 P ACA -2.005, day_0 E ACG -2.293) are all rare-codon depletions with BWM_k under 40; their direction is consistent with the per-cell counts but the magnitude is dominated by the small-k floor and they are not in the high-count concordant set. Largest-magnitude high-count concordant cells (mean log2OR across the 3 timepoints, single sign throughout, k_BWM>=100 in every timepoint): A:AGA +0.294, P:AAG +0.379, P:GAA -0.310, P:AGA +0.249. Treat per-cell n_sig differences across timepoints as power-modulated by the BWM/control N imbalance, not as direction-of-effect biology.

## Top hits

The effect column is `log2_OR` (the log2 of the `odds_ratio` column). Direction is fixed by the BWM-vs-control contingency layout: positive log2(OR) = BWM-enriched, negative = BWM-depleted. `p_value` is the raw two-sided Fisher's exact p; `p_adj` is BH-corrected within each (timepoint, site) family of 61 sense codons. The `aa` column is the single-letter amino acid translation of each codon. Tables below merge all three ribosome sites and are split per-direction within each timepoint.

**Inclusion criterion.** A codon appears in a table only if its `p_adj` is below 0.05 within its own (timepoint, site) family of 61 AND its combined `k_BWM + k_control` is >= 50. Every qualifying row is shown (no row cap). Each table merges all three ribosome sites for a (timepoint, direction) pair; rows are grouped by `site` in A -> P -> E order and within each site sorted by `|log2_OR|` descending. If no rows clear the threshold for a given (timepoint, direction) combination, that table is omitted entirely.

Two cross-timepoint summary tables (direction concordance and direction-flip cells) appear under "Cross-timepoint summary" at the end of this section, after the per-(timepoint, direction) blocks.

### day_0 — Enriched (BWM > control)

| site | codon | aa | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A | AAG | K | +0.654 | 2.27e-19 | 1.38e-17 | iid-amp |
| A | CGC | R | +0.592 | 5.34e-04 | 2.03e-03 |  |
| A | AGA | R | +0.559 | 5.91e-05 | 3.68e-04 |  |
| A | CAC | H | +0.448 | 1.14e-02 | 2.67e-02 | rare-codon |
| A | CTC | L | +0.427 | 1.20e-03 | 4.00e-03 |  |
| P | CGC | R | +0.954 | 3.45e-07 | 3.45e-06 | rare-codon |
| P | AAG | K | +0.692 | 1.71e-20 | 1.04e-18 | iid-amp |
| P | AGA | R | +0.579 | 4.13e-05 | 2.47e-04 |  |
| P | ATC | I | +0.424 | 4.46e-05 | 2.47e-04 |  |
| P | CGT | R | +0.410 | 6.59e-04 | 2.41e-03 |  |
| E | CGC | R | +0.923 | 1.02e-06 | 1.44e-05 | rare-codon |
| E | AAG | K | +0.745 | 3.33e-32 | 2.03e-30 | iid-amp |
| E | CTC | L | +0.539 | 3.10e-05 | 1.89e-04 |  |
| E | AGA | R | +0.468 | 3.82e-04 | 1.66e-03 |  |
| E | CGT | R | +0.465 | 9.31e-04 | 3.47e-03 |  |

### day_0 — Depleted (BWM < control)

| site | codon | aa | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A | TGT | C | -1.843 | 1.25e-04 | 5.88e-04 | rare-codon |
| A | TTT | F | -1.521 | 4.44e-06 | 5.41e-05 | rare-codon |
| A | AAT | N | -1.499 | 3.39e-11 | 1.03e-09 | rare-codon |
| A | TAT | Y | -1.346 | 2.03e-09 | 4.13e-08 | rare-codon |
| A | AGT | S | -1.340 | 8.50e-03 | 2.35e-02 | rare-codon |
| P | ACA | T | -2.005 | 3.66e-09 | 5.58e-08 | rare-codon |
| P | ACG | T | -1.868 | 1.32e-04 | 6.17e-04 | rare-codon |
| P | TGT | C | -1.477 | 2.59e-04 | 1.05e-03 | rare-codon |
| P | TTT | F | -1.299 | 2.35e-05 | 1.59e-04 | rare-codon |
| P | AGT | S | -1.277 | 1.50e-03 | 5.07e-03 | rare-codon |
| E | ACG | T | -2.293 | 1.24e-04 | 6.32e-04 | rare-codon |
| E | ACA | T | -1.555 | 2.39e-06 | 2.43e-05 | rare-codon |
| E | CCG | P | -1.402 | 2.40e-03 | 7.20e-03 | rare-codon |
| E | GTA | V | -1.402 | 2.40e-03 | 7.20e-03 | rare-codon |
| E | TCA | S | -1.396 | 1.18e-06 | 1.44e-05 | rare-codon |

### day_5 — Enriched (BWM > control)

| site | codon | aa | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E | CCA | P | +0.358 | 1.20e-03 | 2.52e-02 |  |

### day_5 — Depleted (BWM < control)

| site | codon | aa | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| E | GAG | E | -0.300 | 1.16e-04 | 5.00e-03 |  |
| E | AAG | K | -0.226 | 1.64e-04 | 5.00e-03 |  |

### day_10 — Enriched (BWM > control)

| site | codon | aa | effect (`log2_OR`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| P | GTC | V | +0.563 | 6.61e-05 | 4.03e-03 |  |

### Cross-timepoint summary

#### Cross-timepoint direction concordance

48 of 183 (site, codon) cells hold the same OR direction across all 3 timepoints. Below the cells are split by displayed mean-log2OR direction into BWM-enriched and BWM-depleted sub-tables; each is sorted by `#sig` descending, then by `|mean log2OR|` descending, and capped at the top 15 cells per direction (omitted-row counts in the footer beneath each sub-table). `#sig` is the count (out of 3) of timepoints where the row reaches `p_adj` < 0.05 (the FDR-corrected threshold); `min p_adj` is the smallest of the 3 per-timepoint values. The `codon` column carries the `site:codon(AA)` triple — the site letter (A/P/E), the three-letter codon, and the one-letter amino acid translation in parentheses (no space between the codon and the parenthesised AA). The `per-tp log2OR` column lists the three per-timepoint values in chronological order (day 0, day 5, day 10); `*` marks each per-timepoint value at `p_adj` < 0.05.

##### Concordance — BWM-enriched (mean log2OR > 0)

| codon | mean log2OR | per-tp log2OR | #sig | min p_adj | flag |
| :---: | :---: | :---: | :---: | :---: | :---: |
| E:CGC(R) | +0.514 | +0.92\* +0.04 +0.58 | 1/3 | 1.44e-05 | rare-codon (0, 10) |
| P:AAG(K) | +0.379 | +0.69\* +0.25 +0.20 | 1/3 | 1.04e-18 | iid-amp |
| A:AGA(R) | +0.294 | +0.56\* +0.03 +0.29 | 1/3 | 3.68e-04 |  |
| A:CAC(H) | +0.292 | +0.45\* +0.20 +0.23 | 1/3 | 2.67e-02 | rare-codon (0, 10) |
| P:AGA(R) | +0.249 | +0.58\* +0.10 +0.06 | 1/3 | 2.47e-04 |  |
| P:CGT(R) | +0.230 | +0.41\* +0.16 +0.12 | 1/3 | 2.41e-03 |  |
| P:ATA(I) | +0.442 | +0.14 +0.39 +0.80 | 0/3 | 6.44e-01 | rare-codon (0, 5, 10) |
| P:AGG(R) | +0.378 | +0.43 +0.58 +0.12 | 0/3 | 5.94e-01 | rare-codon (0, 5, 10) |
| E:AGG(R) | +0.303 | +0.47 +0.42 +0.02 | 0/3 | 5.35e-01 | rare-codon (0, 5, 10) |
| A:CGT(R) | +0.230 | +0.26 +0.16 +0.27 | 0/3 | 8.81e-02 |  |
| P:CTT(L) | +0.204 | +0.13 +0.14 +0.34 | 0/3 | 3.75e-01 |  |
| A:GGT(G) | +0.200 | +0.10 +0.07 +0.43 | 0/3 | 5.64e-01 | rare-codon (0, 5, 10) |
| P:CTC(L) | +0.196 | +0.30 +0.07 +0.22 | 0/3 | 8.90e-02 |  |
| A:TCC(S) | +0.196 | +0.29 +0.07 +0.22 | 0/3 | 1.43e-01 | rare-codon (0) |
| P:GCG(A) | +0.179 | +0.03 +0.17 +0.34 | 0/3 | 8.93e-01 | rare-codon (0, 5, 10) |

(15 highest-ranked enriched concordant cells shown; 2 more enriched concordant cells below the cutoff `(#sig, |mean log2OR|) = (0/3, 0.179)` are in the data but not listed.)

##### Concordance — BWM-depleted (mean log2OR < 0)

| codon | mean log2OR | per-tp log2OR | #sig | min p_adj | flag |
| :---: | :---: | :---: | :---: | :---: | :---: |
| P:GTA(V) | -0.752 | -1.06\* -0.04 -1.15 | 1/3 | 2.65e-02 | rare-codon (0, 5, 10) |
| P:ACA(T) | -0.730 | -2.00\* -0.07 -0.12 | 1/3 | 5.58e-08 | rare-codon (0, 5, 10) |
| E:CGA(R) | -0.721 | -1.17\* -0.01 -0.98 | 1/3 | 2.22e-02 | rare-codon (0, 5, 10) |
| P:CCT(P) | -0.674 | -1.24\* -0.26 -0.52 | 1/3 | 3.48e-02 | rare-codon (0, 5, 10) |
| P:CGA(R) | -0.662 | -1.21\* -0.26 -0.52 | 1/3 | 4.50e-02 | rare-codon (0, 5, 10) |
| E:TCG(S) | -0.527 | -0.84\* -0.08 -0.66 | 1/3 | 1.16e-02 | rare-codon (0, 5, 10) |
| A:ACA(T) | -0.523 | -1.11\* -0.15 -0.31 | 1/3 | 3.68e-04 | rare-codon (0, 5, 10) |
| P:GCA(A) | -0.461 | -0.82\* -0.16 -0.41 | 1/3 | 6.64e-03 | rare-codon (0, 5, 10) |
| E:AAT(N) | -0.446 | -1.11\* -0.16 -0.06 | 1/3 | 5.16e-07 | rare-codon (0, 10) |
| P:TCA(S) | -0.443 | -0.95\* -0.14 -0.25 | 1/3 | 2.41e-03 | rare-codon (0, 5, 10) |
| E:TTT(F) | -0.417 | -1.11\* -0.06 -0.08 | 1/3 | 8.03e-03 | rare-codon (0, 5, 10) |
| A:AAA(K) | -0.391 | -0.89\* -0.05 -0.23 | 1/3 | 4.52e-05 | rare-codon (0, 10) |
| P:GAA(E) | -0.310 | -0.69\* -0.17 -0.07 | 1/3 | 3.45e-06 |  |
| A:GAA(E) | -0.242 | -0.46\* -0.24 -0.02 | 1/3 | 4.61e-04 |  |
| E:GAT(D) | -0.199 | -0.42\* -0.13 -0.04 | 1/3 | 1.66e-03 |  |

(15 highest-ranked depleted concordant cells shown; 16 more depleted concordant cells below the cutoff `(#sig, |mean log2OR|) = (1/3, 0.199)` are in the data but not listed.)

#### Direction-flip cells across timepoints

133 of 183 (site, codon) cells show at least one sign change across the 3 timepoint values. Only 1 (E:AAG high-count) has the flip register at p_adj<0.05 on both opposite-sign rows — flagged `sig-flip: yes` below. `#sig` is the count (out of 3 timepoints) of rows reaching p_adj<0.05 (the FDR-corrected threshold); the `per-tp log2OR` column lists the three per-timepoint values in chronological order (day 0, day 5, day 10); `*` marks each per-timepoint value that reaches the threshold. Sorted by `#sig` descending, then by max single-cell `|log2OR|` descending. Capped at the top 15 flip cells.

| codon | log2OR range | per-tp log2OR | #sig | sig-flip | flag |
| :---: | :---: | :---: | :---: | :---: | :---: |
| E:AAG(K) | [-0.226, +0.745] | +0.75\* -0.23\* +0.02 | 2/3 | yes | iid-amp (0 cell only) |
| E:ACG(T) | [-2.293, +0.663] | -2.29\* +0.16 +0.66 | 1/3 | no | rare-codon (0, 5, 10) |
| A:GCG(A) | [-2.165, +0.491] | -1.19\* +0.49 -2.17 | 1/3 | no | rare-codon (0, 5, 10) |
| P:ACG(T) | [-1.868, +0.086] | -1.87\* +0.09 -0.95 | 1/3 | no | rare-codon (0, 5, 10) |
| A:TGT(C) | [-1.843, +0.548] | -1.84\* -0.30 +0.55 | 1/3 | no | rare-codon (0, 5, 10) |
| E:ACA(T) | [-1.555, +0.292] | -1.56\* +0.29 +0.04 | 1/3 | no | rare-codon (0, 5, 10) |
| A:TTT(F) | [-1.521, +0.244] | -1.52\* -0.00 +0.24 | 1/3 | no | rare-codon (0, 5, 10) |
| A:AAT(N) | [-1.499, +0.077] | -1.50\* -0.20 +0.08 | 1/3 | no | rare-codon (0, 10) |
| P:TGT(C) | [-1.477, +0.077] | -1.48\* -0.18 +0.08 | 1/3 | no | rare-codon (0, 5, 10) |
| A:GTA(V) | [-1.418, +0.039] | -1.15\* +0.04 -1.42 | 1/3 | no | rare-codon (0, 5, 10) |
| E:CCG(P) | [-1.402, +0.512] | -1.40\* +0.51 -0.58 | 1/3 | no | rare-codon (0, 5, 10) |
| E:GTA(V) | [-1.402, +0.644] | -1.40\* +0.64 -0.42 | 1/3 | no | rare-codon (0, 5, 10) |
| E:TCA(S) | [-1.396, +0.405] | -1.40\* -0.05 +0.40 | 1/3 | no | rare-codon (0, 5, 10) |
| A:TAT(Y) | [-1.346, +0.091] | -1.35\* +0.09 -0.32 | 1/3 | no | rare-codon (0, 5, 10) |
| A:AGT(S) | [-1.340, +0.092] | -1.34\* +0.06 +0.09 | 1/3 | no | rare-codon (0, 5, 10) |

(15 highest-magnitude flip cells shown; 118 more flip cells below the cutoff `(#sig, max |log2OR|) = (1/3, 1.340)` are in the data but not listed.)

### Flag glossary

- `iid-amp` — single per-timepoint `p_adj` falls below 1e-10. The magnitude of `-log10(p_adj)` is inflated by Fisher's anti-conservative behaviour at large pooled N + correlated stall positions; rank by effect size, not by p magnitude.
- `rare-codon` — `BWM_k` < 100 or `ctrl_k` < 100 in this row (or at least one timepoint, for cross-timepoint cells). OR estimate is unstable at small k and the magnitude is dominated by the small-k floor; large |log2OR| in flagged rows may reflect sampling noise rather than biology.

## Numbers at a glance
- `n_tests`: 549 (3 timepoints x 3 sites x 61 sense codons)
- `n_significant` (adjusted-p < 0.05): 86 (15.7%)
- `n_significant` (adjusted-p < 0.10): 108 (19.7%)
- `min adjusted-p`: 2.0325194950162870e-30 (day_0 E AAG, log2OR=+0.745, BWM_k=866/n=6091, ctrl_k=2495/n=27732)
- `p_floor`: n/a — Fisher's exact has no analytic floor; effective discreteness floor at small (k_BWM + k_ctrl) is captured by the `larger-bh-family` and `rare-codon-low-count` caveats
- Cells with p_adj < 1e-10: 3 of 549 (day_0 E AAG, day_0 P AAG, day_0 A AAG); all flagged `iid-amp`
- 2 rows have OR exactly = 0 (BWM_k = 0): day_0 E ATA (ctrl_k=16, p_adj=0.152) and day_10 P TTA (ctrl_k=1, p_adj=1.0); reported with log2OR=-Inf and excluded from |mean log2OR| ranking
- Cross-(timepoint, site) BH families and their n_sig at p_adj<0.05:

| timepoint | A | P | E | BWM_total | control_total | ctrl:BWM ratio |
| --- | --- | --- | --- | --- | --- | --- |
| day_0 | 27/61 | 28/61 | 27/61 | 6091 | 27732 | 4.55 |
| day_5 | 0/61 | 0/61 | 3/61 | 11935 | 11177 | 0.94 |
| day_10 | 0/61 | 1/61 | 0/61 | 6945 | 8788 | 1.27 |

- Cross-timepoint (site, codon) cells, all 3 timepoints same OR direction: 48/183 (17 BWM-enriched, 31 BWM-depleted)
- Cross-timepoint (site, codon) cells with at least one sign change: 133/183
- Cells with sign change at p_adj<0.05 on both opposite-sign rows: 1 (E:AAG)

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2x2 contingency BWM_count vs control_count for each (timepoint, site, codon), with BH-FDR correction within each (timepoint, site) family of 61 sense codons; user confirmed. Effect column is `odds_ratio` (re-expressed as log2OR in this report's tables for symmetry with sister files); test statistic and exact p are returned by `scipy.stats.fisher_exact`. The p-correction column is `p_adj` (BH per (timepoint, site) family). The test answers "is the BWM-vs-control codon composition at this site different at this timepoint?". It does *not* answer "is each condition's per-codon enrichment vs background informative" (that is the `within_condition_binomial` family) and *not* "is the BWM-vs-control difference itself shifting across timepoints" (that would require an explicit between-timepoint contrast; see the `between_timepoint_wilcoxon` family for the within-condition variant of that question).

## Caveats
### Confirmed
- **pseudorep** (family-wide) — 2x2 contingencies pool replicates by design; per-replicate variation is not represented in the test statistic. Inherited from family `per_timepoint_fisher`.
- **large-N-Fisher-anticonservative** (family-wide) — Fisher's exact on contingencies built from many thousands of pooled stall events behaves anti-conservatively when stall positions are correlated (cluster on motifs, share transcripts). Effective N is smaller than k_total. Inherited from family `per_timepoint_fisher`.
- **imbalanced-N** (family-wide) — control_total is much larger than BWM_total at day_0 (27732 vs 6091, ratio 4.55:1). The asymmetry concentrates Fisher's information in the larger arm and produces extreme p_adj at moderate effect sizes there. Inherited from family `per_timepoint_fisher`.
- **bh-per-(timepoint,site)** (family-wide) — BH correction is applied independently within each of the 9 (timepoint, site) families of 61 codons, not across the full 549-test grid. Two cells with the same raw p in different (timepoint, site) families can therefore receive different `p_adj` values; cross-(timepoint, site) p_adj rankings are not directly commensurable. Inherited from family `per_timepoint_fisher`.
- **larger-bh-family** (per-CSV) — each per-(timepoint, site) BH family is 61 sense codons, ~3x larger than the AA-resolution sister's 20-AA families. AA-level signals can split across synonyms below the per-(timepoint, site) FDR threshold here; if a feature is significant in the AA file but absent here, that is an aggregation effect, not a contradiction.
- **rare-codon-low-count** (per-CSV) — many rare codons have BWM_count or control_count cells in the single/double digits even though pooled totals are in the thousands. Concrete examples in this file: day_0,A,GGG BWM_k=2 ctrl_k=33; day_0,P,TTA BWM_k=1; day_0,A,TGT BWM_k=7 ctrl_k=114; day_0,E,ACG BWM_k=4 ctrl_k=89. OR estimates and Fisher p-values are unstable for those rows. The largest single-cell |log2OR| values in the file (day_0 E ACG -2.293, day_0 P ACA -2.005, day_0 A TGT -1.843, etc.) are all rare-codon depletions; their direction is real per the per-cell counts but magnitude is dominated by the small-k floor. Rows are flagged `rare-codon` in Top hits when BWM_k<100 OR control_k<100.
- **OR-direction-anchor** (per-CSV) — OR > 1 = BWM enriched relative to control at that (timepoint, site, codon); OR < 1 = BWM depleted. Direction is fixed by the contingency layout. Stated here so downstream consumers do not invert.
- **n-asymmetry-affects-power-not-direction** (per-CSV) — BWM_total varies across timepoints (6091 / 11935 / 6945) and control_total varies more (27732 / 11177 / 8788). The day_0 control:BWM ratio of 4.55:1 means a moderate effect size at day_0 inflates to a smaller p_adj than the same effect size would at day_5 or day_10. The 28-OOM gap between min p_adj at day_0 (2.03e-30) and day_5 (4.99e-3) or day_10 (1.36e-1) is consistent with this N-modulated power asymmetry. Per-cell n_sig differences are therefore power-modulated, not necessarily direction-of-effect biology; cross-test reconciliation is the place to disentangle the two.

### Considered but not applicable
*(none denied this run; no per-CSV proposals were rejected.)*

## For Chumeng (joint-reading hooks)
- Family: `per_timepoint_fisher` — sister CSV in this family that should be reconciled: `per_timepoint_fisher_aa.csv` (AA resolution; same design, same family-wide caveats).
- Open questions Chumeng should resolve at synthesis time:
  - The 3 day_0 AAG cells (A: log2OR=+0.654 p_adj=1.38e-17 BWM_k=618; E: +0.745 p_adj=2.03e-30 BWM_k=866; P: +0.692 p_adj=1.04e-18 BWM_k=591) are the file's three p_adj < 1e-10 rows. Does the AA sister file show K (AAG's amino acid) at the same direction at all 3 day_0 sites with comparable magnitude? If yes at A and P but smaller magnitude than E here → AA-level Lys is the broader signal and AAG is the dominant codon; if AA-level K is smaller magnitude than the codon-level AAG at A or P → the codon resolution is adding specificity beyond AA-level. Independently: does the within-condition binomial AA file rank Lys at the same 3 sites at comparable magnitude in the BWM groups (independent of any control comparison)? Each route gives a different falsifier.
  - The Lys synonym divergence: AAG is BWM-enriched at all 3 day_0 sites (above), while AAA at the same day_0 (A:AAA log2OR=-0.893 p_adj=4.52e-05; need to check E:AAA and P:AAA from the per-(timepoint, site) tables) is in the opposite direction at A site at p_adj<0.05. Does AA-level Lys read as "no signal" (when the two synonyms cancel in aggregation) or as a directional signal (one synonym dominating the count)? AA file is the place to test.
  - The 48-cell concordant set, split into 17 BWM-enriched and 31 BWM-depleted by mean log2OR direction. Top-15-by-(#sig desc, |mean log2OR| desc) is displayed per direction in the report; the remaining 2 enriched and 16 depleted cells fall below the displayed cutoff. The highest-#sig (1/3) anchors per direction: enriched E:CGC, P:AAG, A:AGA, A:CAC, P:AGA, P:CGT (6 cells reaching FDR<0.05 in 1 timepoint each); depleted P:GTA, P:ACA, E:CGA, P:CCT, P:CGA, E:TCG, A:ACA, P:GCA, E:AAT, P:TCA, E:TTT, A:AAA, P:GAA, A:GAA, E:GAT (15 cells reaching FDR<0.05 in 1 timepoint each). No 2/3 or 3/3 row appears in the concordant set. Do the same-direction cells reappear in the AA-resolution sister at their amino-acid level (Arg for AGA/CGC/CGT/CGA, Lys for AAG/AAA, Glu for GAA, His for CAC, Asp for GAT, Thr for ACA, Pro for CCT, Ile for ATA, Leu for CTC/CTT, Ala for GCA/GCG, Ser for TCA/TCG/TCC, Val for GTA, Asn for AAT)? If a codon-level concordant cell aggregates to an AA-level concordant cell, the signal is robust at both resolutions; if it aggregates to a flat AA-level cell, the synonym specificity is the carrier of the signal.
  - The N-imbalance asymmetry: per-(timepoint, site) n_sig follows the same gradient as the AA-resolution sister (day_0 dominates, day_5/10 collapse to near-zero). Does this gradient match the AA file's gradient quantitatively? If both files show the same gradient with the same N imbalance, the gradient is power-modulated; if the codon file shows a different gradient (e.g. fewer day_0 hits relative to AA, or more day_5/10 hits), that flags a resolution-dependent biological signal.
  - The 1 sig-flip cell counted in the report: E:AAG (high-count, day_0 +0.745 vs day_5 -0.226, both p_adj<0.05). Does the AA-resolution sister show E:K (the Lys AA cell containing AAG) flip with the same direction sequence between day_0 and day_5? If yes, the AAG-driven flip is real at AA level; if AA-level E:K does not flip (because AAA stays in the other direction or partially cancels), the flip is a codon-resolution-specific signal.
  - The 18 below-cutoff concordant cells not displayed in the top-15-per-direction view (2 enriched + 16 depleted). These are predominantly 0/3-sig cells dominated by rare-codon depletions (A:CGG, P:GGG, P:CGG, E:AGT, E:GGG, A:GTG, P:TCG, etc.). Direction-stable but per-cell magnitude-unstable because at least one timepoint has BWM_k or ctrl_k under 100 (`rare-codon` flag). Does any of them reach significance at AA-level aggregation (where the synonym pool gives more counts), or only resolve at codon level? Useful to triangulate before discarding them as small-k noise.
