---
input_csv: results/stall_sites/enrichment/analysis_stats/between_condition_wilcoxon_codon.csv
family: between_condition_wilcoxon
test_type: Mann-Whitney U / Wilcoxon rank-sum (two-sided)
test_type_source: user-confirmed
n_tests: 183
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.26406926406926406
p_floor: null
pseudoreplicated: null
synced_from_olive_qmd: 2026-05-30
caveats:
  - {label: "timepoint-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "mw-floor-tight", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "n=6-modest-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "Per-site BH family is 61 codons (vs 20 AA at aa resolution). With raw-p exact-floor ~0.00216, the minimum BH-adjusted p achievable at a single sole floor-hit is now ~0.13 (0.00216*61/1) vs ~0.043 at AA. FDR<0.05 clearance is structurally harder. Empirically the file's min p_adj = 0.264 confirms the structural picture: smallest raw p (0.00433) at CGG@A BH-corrects to exactly 61*0.00433 = 0.264."}
  - {label: "low-count-rare-codon-instability", proposed_by: dylan, status: confirmed, why: "Rare codons (CGG@A median ~0.0009, GTA@A ~0.002, ATA@P ~0.001, GGG@P ~0.0008, TTA@P ~0.0002) have many tied zero-frequency reps; their `log2_FC` is volatile across small count fluctuations. Treat extreme |log2_FC| at low-count codons cautiously."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` (median ratio) only; no count-weighted enrichment. Acute at codon level because the rarity skew is wider (median freq ranges from ~0.0002 to ~0.12, three orders of magnitude)."}
caveats_considered:
  - {label: "synonymous-codon-redundancy", proposed_by: dylan, status: denied, why: "Several codons share an amino acid; if a real signal is AA-driven (e.g. lysine enrichment) it spreads across AAA/AAG synonyms and looks weaker per-codon than the AA-level test. Could mislead if the AA-level file showed a clean signal that the codon level cannot resolve.", user_note: "Denied in triage. Reasonable: the aa-level file also showed no FDR hits, so 'codon weakened the AA signal' is not the operative concern here."}
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: denied, why: "scipy.stats.mannwhitneyu with n=6+6 and possibly tied zero-frequency reps for rare codons may default to the asymptotic Z approximation rather than the exact distribution; would shift p-values away from the exact-floor bound (~0.00216).", user_note: "Denied after empirical audit (scripts/_for_claude_mw_branch_audit.py --level codon). Only 4 of 183 (site, codon) tests have any rank ties in the pooled 12-element sample (ATA@E, CGG@E, TTA@E, TTA@P — all rare codons, all far from raw p<0.05). For those 4 scipy auto-picked the asymptotic branch; for the remaining 179 it picked exact, and the pipeline CSV matches the auto choice to ~1e-16. A forced-asymptotic counterfactual flips zero raw-p<0.05 decisions and leaves the per-site BH conclusion unchanged (0 hits at FDR<0.05 either way; asymptotic is slightly more conservative, with min p_adj = 0.50/0.36/0.50 at A/P/E vs as-shipped 0.26/0.26/0.26). The branch choice does not affect any FDR-level conclusion in this file."}
headline: "No statistically significant differences at FDR<0.05 for codon-level BWM-vs-control under MW with timepoints pooled (0/183); 12 codons clear nominal raw p<0.05 across the 3 sites and four (CGG@A, TCA@P, GTA@P, AGT@E) hit the smallest possible BH-adjusted p of 0.264 — the discrete BH wall at n=61."
user_directives:
  - "(triage) Test type confirmation — same as aa variant, with the feature alphabet expanded to 61 sense codons → `Confirm`."
  - "(triage) CSV-specific caveats (multi-select; family-wide already locked) → `larger-bh-family`, `low-count-rare-codon-instability`, `weighted_log2_enrichment-absent` confirmed; `synonymous-codon-redundancy` and `asymptotic-with-ties` denied (the latter after empirical audit, see caveats_considered)."
  - "(triage) Framing firmness → `Mixed` (mirror aa file: firm overall null, flag CGG@A and other near-threshold features as exploratory leads to reconcile against per-timepoint Fisher contrasts)."
  - "(triage) Spotlight → `No spotlight` (default top-5-up + top-5-down per site; A-site main, E and P under <details>)."
  - "(readback) \"Reconciled shared content from the corrected .qmd on 2026-05-30\" -> \"Adopted Olive's per-(site, direction) Top-hits layout (six sub-tables in A/P/E order) with added `aa` and raw `p_value` columns; corrected the A-site For-Chumeng hook (AAA@A is BWM-depleted, opposite AAG@A); all table and prose numbers re-verified against the raw CSV, no other content changed.\""
---

# Interpretation — between_condition_wilcoxon_codon

> Source: `results/stall_sites/enrichment/analysis_stats/between_condition_wilcoxon_codon.csv`
> Family: `between_condition_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage) Test type confirmation: same test as aa variant, with the feature alphabet expanded to 61 sense codons → "Confirm".
- (triage) CSV-specific caveats: `larger-bh-family`, `low-count-rare-codon-instability`, `weighted_log2_enrichment-absent` confirmed; `synonymous-codon-redundancy` and `asymptotic-with-ties` denied (the latter after empirical audit).
- (triage) Framing firmness: "Mixed" — overall no-signal file but call out closest-to-significant features for cross-test follow-up.
- (triage) Spotlight: none — default per-site tables.
- (readback) Reconciled shared content from the corrected .qmd on 2026-05-30 → adopted Olive's per-(site, direction) Top-hits layout (six sub-tables, A/P/E) with `aa` + raw `p_value` columns; corrected the A-site For-Chumeng hook (AAA@A is BWM-depleted, not co-directional with AAG@A); all numbers re-verified against the raw CSV.

## Headline
No statistically significant differences at FDR<0.05 across 183 codon-level tests (3 sites x 61 sense codons) under Mann-Whitney with BWM (n=6) vs control (n=6) replicates pooled across day_0/5/10. Min `p_adj` = 0.264 — the discrete BH wall (0.00433 x 61 / 1) hit by four cells: **CGG@A** (BWM-depleted, log2_FC = -1.03), **TCA@P** (BWM-depleted, -0.32), **GTA@P** (BWM-depleted, -0.69), and **AGT@E** (BWM-depleted, -0.68). 12 codons clear raw p<0.05 across the 3 sites; all carry `nominal-only`. The codon resolution adds raw-p detail vs the aa file (smallest raw p drops from 0.0260 to 0.00433) but the larger BH family of 61 cancels the gain — FDR<0.05 remains out of reach. Treat as no signal conditional on `mw-floor-tight` + `larger-bh-family` + `timepoint-pooled-confound`; the named near-threshold codons are exploratory leads worth reconciling against per-timepoint Fisher contrasts.

## Top hits

Effect column is `log2_FC` (BWM/control median ratio). `p_value` is the raw Mann-Whitney p; `p_adj` is BH-corrected per A/P/E site (family of 61 codon tests). Each site group is split into one sub-table per sign of effect: positive `log2_FC` means BWM-enriched; negative means BWM-depleted. Within each sub-table, rows are up to 5 per direction ranked by raw `p_value` ascending, with `|log2_FC|` descending as the tiebreaker. The `low-count` flag is applied when `min(median_BWM, median_control) < 0.005`. The `aa` column is the single-letter amino acid translation of each codon.

### A site — Enriched (BWM > control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| AAG | K | +0.206 | 0.0260 | 0.528 | nominal-only |
| AGA | R | +0.192 | 0.0649 | 0.710 |  |
| CGT | R | +0.205 | 0.0931 | 0.710 |  |
| CTC | L | +0.265 | 0.2403 | 0.836 |  |
| TCC | S | +0.135 | 0.2403 | 0.836 |  |

### A site — Depleted (BWM < control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CGG | R | -1.028 | 0.00433 | 0.264 | nominal-only, low-count |
| GTA | V | -0.632 | 0.0260 | 0.528 | nominal-only, low-count |
| GAA | E | -0.220 | 0.0411 | 0.627 | nominal-only |
| GCG | A | -0.383 | 0.0931 | 0.710 | low-count |
| GTG | V | -0.318 | 0.0931 | 0.710 |  |

### P site — Enriched (BWM > control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| AAG | K | +0.331 | 0.0260 | 0.360 | nominal-only |
| ATA | I | +0.513 | 0.0411 | 0.360 | nominal-only, low-count |
| CGT | R | +0.235 | 0.0931 | 0.406 |  |
| AGA | R | +0.274 | 0.1320 | 0.424 |  |
| AGC | S | +0.131 | 0.1320 | 0.424 |  |

### P site — Depleted (BWM < control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| TCA | S | -0.320 | 0.00433 | 0.264 | nominal-only |
| GTA | V | -0.690 | 0.00866 | 0.264 | nominal-only, low-count |
| AGT | S | -0.880 | 0.0152 | 0.308 | nominal-only, low-count |
| CCA | P | -0.190 | 0.0411 | 0.360 | nominal-only |
| ACG | T | -0.715 | 0.0649 | 0.360 | low-count |

### E site — Enriched (BWM > control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| CGC | R | +0.619 | 0.0411 | 0.728 | nominal-only |
| CGT | R | +0.197 | 0.0931 | 0.728 |  |
| CTA | L | +0.401 | 0.2403 | 0.728 | low-count |
| TAC | Y | +0.128 | 0.2403 | 0.728 |  |
| TTC | F | +0.062 | 0.3095 | 0.728 |  |

### E site — Depleted (BWM < control)

| codon | aa | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| AGT | S | -0.676 | 0.00433 | 0.264 | nominal-only, low-count |
| TCG | S | -0.691 | 0.0649 | 0.728 | low-count |
| GAT | D | -0.183 | 0.0931 | 0.728 |  |
| CGA | R | -0.779 | 0.1320 | 0.728 | low-count |
| CTG | L | -0.596 | 0.1320 | 0.728 | low-count |

### Flag glossary
- `nominal-only` — raw p < 0.05 but `p_adj` >= 0.05. The row would have been flagged significant if tested in isolation, but did not survive BH-FDR correction across its grouping family.
- `low-count` — `min(median_BWM, median_control) < 0.005`. The codon's per-rep frequency is below the test's stability threshold; the `log2_FC` for this row is unstable and should not be over-read. At codon level the rare-codon median frequencies in this file span ~0.0002 to ~0.005, and large |log2_FC| at these cells is dominated by single-rep fluctuations.

## Numbers at a glance
- `n_tests`: 183
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.2641 (the discrete BH wall: 0.004329 × 61 / 1, hit by CGG@A, AGT@E, TCA@P, GTA@P)
- `min raw-p`: 0.00433 (CGG@A, AGT@E, TCA@P) — three orders of magnitude tighter than the aa file's 0.0260 minimum
- `nominal-only count`: 12 codons across 3 sites with raw p<0.05; none survive BH per-site
- `p_floor`: n/a — user did not flag a floor effect at the CSV level (family-wide `mw-floor-tight` instead)
- Per-site BH families:
  - A site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.264 (CGG, BWM-depleted)
  - P site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.264 (TCA / GTA, BWM-depleted)
  - E site: 61 tests, 0 hits at p_adj<0.05, min p_adj = 0.264 (AGT, BWM-depleted)

## Methods
Dylan proposed Mann-Whitney U / Wilcoxon rank-sum two-sided on per-replicate codon frequencies, n=6 BWM vs n=6 control with reps pooled across day_0/5/10, BH-FDR per A/P/E site (each site = own family of 61 codon hypotheses); user confirmed. Effect column is `log2_FC` of medians (`median_BWM/median_control`); test statistic is `U_stat`. The codon resolution is the natural follow-up to `between_condition_wilcoxon_aa.csv` and shares its design. The test answers "is codon X distributed at different per-rep frequencies between BWM and control replicates?", *not* "is codon X enriched at stall sites vs the transcriptome's codon usage" (within-condition binomial) and *not* a comparison vs an external null.

## Caveats
### Confirmed
- **timepoint-pooled-confound** (family-wide) — n=6 per condition treats day_0/5/10 reps as 6 control / 6 BWM replicates; time-by-condition interactions disappear into the noise.
- **mw-floor-tight** (family-wide) — exact-test p-floor for n=6 vs n=6 is ~0.00216, but the file's smallest observed raw p is 0.00433, ~2x above the exact floor. Per-site BH family of 61 puts the absolute minimum achievable p_adj at 0.132 even if a single codon hit the exact floor and all 60 others were uncorrelated. (Empirical audit confirms 179/183 tests are on scipy's exact branch and the remaining 4 — rare codons with ties — are on the asymptotic branch but not consequential; see `asymptotic-with-ties` under Considered but not applicable.)
- **n=6-modest-power** (family-wide) — null result is weakly informative.
- **larger-bh-family** (per-CSV) — at codon resolution per-site BH family expands from 20 (aa) to 61 (codon), tightening FDR clearance roughly 3x relative to the aa variant. Empirically the file's min p_adj = 0.264 = 0.00433 × 61, the discrete BH wall a sole-best-p codon hits.
- **low-count-rare-codon-instability** (per-CSV) — rare codons (CGG, GTA, ATA, AGT, GGG, TTA — flagged `low-count` in tables above) have median frequencies < 0.005 and many tied zero-frequency reps; their large |log2_FC| values are unstable, dominated by single-rep fluctuations. Three of the 4 codons hitting min p_adj are low-count (CGG, GTA, AGT); the rank-sum is on per-rep frequencies so this does not invalidate the test, but the effect-size column should not be over-read.
- **weighted_log2_enrichment-absent** (per-CSV) — `log2_FC` does not weight by absolute count; CGG@A (median ~0.0009) and AAG@A (median ~0.09) get the same column real estate despite a 100x abundance difference.

### Considered but not applicable
- **synonymous-codon-redundancy** (per-CSV) — Dylan flagged that AA-driven signals would dilute across synonymous codons and look weaker at codon level than at AA. User denied; reasonable because the aa-level sister file also produced 0 FDR hits, so dilution-from-AA is not the operative concern. The codon results are not weakening a clean AA signal — both resolutions show no signal.
- **asymptotic-with-ties** (per-CSV) — Dylan flagged that scipy may default to the asymptotic Z approximation under tied zero-frequency reps for rare codons, which would shift p-values away from the exact-floor bound. Denied after empirical audit (`scripts/_for_claude_mw_branch_audit.py --level codon`): only 4 of 183 (site, codon) tests have rank ties in the pooled 12-element sample (ATA@E, CGG@E, TTA@E, TTA@P; all rare codons, all far from raw p<0.05). For those 4 scipy auto-picked the asymptotic branch; for the remaining 179 it picked exact, and the pipeline CSV matches the auto choice to ~1e-16. A forced-asymptotic counterfactual flips zero raw-p<0.05 decisions and leaves the per-site BH conclusion unchanged (0 hits at FDR<0.05 either way; asymptotic is slightly more conservative, with min p_adj = 0.50/0.36/0.50 at A/P/E vs as-shipped 0.26/0.26/0.26). The branch choice does not affect any FDR-level conclusion in this file.

## For Chumeng (joint-reading hooks)
- Family: `between_condition_wilcoxon` — sister CSV: `between_condition_wilcoxon_aa.csv`.
- Open questions Chumeng should resolve at synthesis time:
  - Do the 4 codons hitting the BH wall (CGG@A, TCA@P, GTA@P, AGT@E, all BWM-depleted) reappear with consistent direction in `per_timepoint_fisher_codon.csv` at any timepoint? If at least one does, that's a real-but-small BWM-vs-control codon signal MW lacked the n to formalize.
  - **AA ↔ codon agreement at A-site enrichment**: aa file had K (lysine) closest-to-significant in the enriched direction at A site (raw p=0.065). In this codon file AAG@A is BWM-enriched (log2_FC=+0.21, raw p=0.026), but the other Lys synonym AAA@A is BWM-*depleted* (log2_FC=-0.47, raw p=0.132) — opposite direction — so the A-site Lys lead is AAG-specific, not a clean aa-level Lys signal. Neither codon clears FDR. Does the AAG-only A-site enrichment (and the AAG/AAA direction split) reproduce in `per_timepoint_fisher_codon.csv` at any timepoint?
  - **AA ↔ codon agreement at P-site depletion**: aa file had A and P (alanine, proline) BWM-depleted at P site (raw p=0.026, 0.041; min p_adj 0.411). Codon file has CCA (a P codon) BWM-depleted at P site (raw p=0.041, p_adj=0.360) and TCA (a S codon, not P) at the BH wall. Mixed: the AA "P depletion" partly survives via CCA, but the strongest codon-level hit at P-site (TCA) does not correspond to an aa-level top hit (S was not in aa top hits at P).
  - Are the four "BH-wall" codons all biological coincidences from the discrete BH wall, or is there a thread (e.g., NNS-like or rare-codon biased)? CGG, AGT, TCA, GTA are not obviously a class — Chumeng to evaluate.
