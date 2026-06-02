---
input_csv: results/stall_sites/enrichment/analysis_stats/between_condition_wilcoxon_aa.csv
family: between_condition_wilcoxon
test_type: Mann-Whitney U / Wilcoxon rank-sum (two-sided)
test_type_source: user-confirmed
n_tests: 60
n_significant_fdr05: 0
n_significant_fdr10: 0
min_p_adj: 0.41125541125541126
p_floor: null
pseudoreplicated: null
synced_from_olive_qmd: 2026-05-30
caveats:
  - {label: "timepoint-pooled-confound", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "mw-floor-tight", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "n=6-modest-power", proposed_by: family, status: confirmed, why: "Inherited from family `between_condition_wilcoxon` (see _INDEX.md)."}
  - {label: "no-near-nominal-signal", proposed_by: dylan, status: confirmed, why: "Smallest raw p in the file is 0.025974 (T at A site, log2_FC=-0.12); no AA at any site is differentially distributed at the per-replicate frequency level even at nominal p<0.05 in a way BH could lift. Empirical fact, not just a structural floor argument."}
  - {label: "weighted_log2_enrichment-absent", proposed_by: dylan, status: confirmed, why: "Effect column is `log2_FC` (median ratio) only; no count-weighted enrichment to penalize rare features. Column-shape note for downstream consumers."}
caveats_considered:
  - {label: "asymptotic-with-ties", proposed_by: dylan, status: denied, why: "scipy.stats.mannwhitneyu with n=6+6 and likely many tied zero-frequency reps for rare AAs would default to the asymptotic Z approximation rather than the exact distribution; would shift p-values away from the strict exact-floor bound (~0.00216).", user_note: "Denied in triage — not material at AA level. Empirical audit (scripts/_for_claude_mw_branch_audit.py) confirms: 0/60 tests have pooled-sample ties, scipy picked the exact branch for all 60, and forcing the asymptotic branch leaves the FDR conclusion unchanged (0 hits either way). The denial is correct; the original premise (tied zero-frequency reps) does not apply because every replicate has >= 13 stalls of every amino acid at every site."}
headline: "Firm null at FDR<0.05 for AA-level BWM-vs-control under MW with timepoints pooled (0/60); closest-to-significant features are A-site T depleted, E-site R enriched, and P-site A/P depleted, all at raw p ≈ 0.026-0.041 but p_adj ≥ 0.41."
user_directives:
  - "(triage) Test type confirmation — `Mann-Whitney U / Wilcoxon rank-sum on per-replicate frequencies, n=6 BWM vs n=6 control, BH-FDR per site` → `Confirm`."
  - "(triage) CSV-specific caveats (multi-select; family-wide already locked) → `no-near-nominal-signal`, `weighted_log2_enrichment-absent` confirmed; `asymptotic-with-ties` denied."
  - "(triage) Framing firmness → `Mixed` (firm overall null, but call out the closest-to-significant features as candidates worth checking against the per-timepoint Fisher results)."
  - "(triage) Spotlight → `No spotlight` (default top-5-up + top-5-down per site; A-site main, E and P under <details>)."
  - "(readback 2026-05-30) Reconciled the Top hits section to Olive's table structure (per-direction sub-tables A/P/E x Enriched/Depleted, plus the raw `p_value` column) from the corrected `.qmd`. Kept Dylan conventions (bare AA codes, terse headline, Methods provenance, Confirmed/Considered caveats); Olive-only sections (Composite, Overview, Biological interpretation, Plots) not imported. **Enumerated every prose/front-matter number and verified each** (shared numbers vs the `.qmd`; Dylan-only CSV-column numbers — the 0.065/0.065/0.093 near-nominal raw-p values, exact per-site min p_adj 0.41/0.52/0.52 — vs the raw CSV): all correct, no number changed (file stays 0/60 at FDR<0.05). Derivation/audit-sourced numbers (MW exact floor ~0.00216; `_for_claude_mw_branch_audit.py` outputs k=13/3149, 0/60 rank-ties, asymptotic 0.44/0.61/0.61) flagged as not CSV-verifiable. Minor: the shared 'W,C median ~0.5%' approximation slightly understates the CSV (~0.6-1.3%) but matches the `.qmd`, so left for a `.qmd`-side touch-up."
---

# Interpretation — between_condition_wilcoxon_aa

> Source: `results/stall_sites/enrichment/analysis_stats/between_condition_wilcoxon_aa.csv`
> Family: `between_condition_wilcoxon` (see [`_INDEX.md`](_INDEX.md))
> Test type: Mann-Whitney U / Wilcoxon rank-sum (source: user-confirmed)

## User directives
- (triage) Test type confirmation: `Mann-Whitney U / Wilcoxon rank-sum on per-replicate frequencies, n=6 BWM vs n=6 control, BH-FDR per site` → "Confirm".
- (triage) CSV-specific caveats: `no-near-nominal-signal`, `weighted_log2_enrichment-absent` confirmed; `asymptotic-with-ties` denied.
- (triage) Framing firmness: "Mixed" — firm overall null but call out closest-to-significant features for cross-test follow-up.
- (triage) Spotlight: none — default per-site tables, A-site main, E and P under details.
- (readback 2026-05-30) Top hits reconciled to Olive's `.qmd` table structure (per-direction sub-tables A/P/E x Enriched/Depleted + raw `p_value` column); numbers unchanged (0/60 at FDR<0.05). Dylan conventions kept (bare AA codes, terse headline, Methods provenance, Confirmed/Considered caveats); Olive-only sections not imported. Provenance in front-matter `synced_from_olive_qmd`.

## Headline
Firm null at FDR<0.05 across 60 AA-level tests (3 sites × 20 AAs) under Mann-Whitney with BWM (n=6) vs control (n=6) replicates pooled across day_0/5/10. Min `p_adj` = 0.41 at P-site (A and P, both BWM-depleted). The four nominal raw-p hits (T@A, R@E, A@P, P@P) all sit at the discrete tied raw p = 0.0260 or 0.0411 and clear `nominal-only` only — none survive BH per-site. Treat as a firm null conditional on `mw-floor-tight`, `n=6-modest-power`, and `timepoint-pooled-confound`; the four nominal hits are exploratory leads worth reconciling against the per-timepoint Fisher contrasts.

## Top hits

Per (site, direction): top 5 rows by raw `p_value` ascending, with `|log2_FC|` descending as the tiebreaker. `p_value` is the raw two-sided Mann-Whitney p; `p_adj` is BH-corrected per A/P/E site (family of 20 AAs). Positive `log2_FC` = BWM-enriched; negative = BWM-depleted. `nominal-only` flags rows with raw p < 0.05 but `p_adj` >= 0.05.

### A site - Enriched (BWM > control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| K | +0.157 | 0.0649 | 0.649 |  |
| R | +0.197 | 0.2403 | 0.909 |  |
| G | +0.154 | 0.5887 | 0.909 |  |
| L | +0.104 | 0.5887 | 0.909 |  |
| N | +0.018 | 0.5887 | 0.909 |  |

### A site - Depleted (BWM < control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| T | -0.122 | 0.0260 | 0.519 | nominal-only |
| P | -0.066 | 0.3939 | 0.909 |  |
| C | -0.019 | 0.4848 | 0.909 |  |
| Y | -0.174 | 0.5887 | 0.909 |  |
| S | -0.051 | 0.5887 | 0.909 |  |

### P site - Enriched (BWM > control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| K | +0.151 | 0.0649 | 0.433 |  |
| R | +0.289 | 0.1797 | 0.599 |  |
| Q | +0.118 | 0.4848 | 0.861 |  |
| I | +0.094 | 0.4848 | 0.861 |  |
| V | +0.070 | 0.6991 | 0.861 |  |

### P site - Depleted (BWM < control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| A | -0.164 | 0.0260 | 0.411 | nominal-only |
| P | -0.255 | 0.0411 | 0.411 | nominal-only |
| T | -0.109 | 0.0931 | 0.465 |  |
| S | -0.169 | 0.1320 | 0.528 |  |
| C | -0.126 | 0.3095 | 0.861 |  |

### E site - Enriched (BWM > control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| R | +0.292 | 0.0260 | 0.519 | nominal-only |
| W | +0.161 | 0.4848 | 0.999 |  |
| K | +0.099 | 0.4848 | 0.999 |  |
| F | +0.008 | 0.6991 | 0.999 |  |
| C | +0.047 | 0.8182 | 1.000 |  |

### E site - Depleted (BWM < control)

| feature | effect (`log2_FC`) | raw p (`p_value`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| D | -0.114 | 0.4848 | 0.999 |  |
| A | -0.042 | 0.4848 | 0.999 |  |
| N | -0.040 | 0.4848 | 0.999 |  |
| S | -0.246 | 0.5887 | 0.999 |  |
| Q | -0.165 | 0.5887 | 0.999 |  |

## Numbers at a glance
- `n_tests`: 60
- `n_significant` (adjusted-p < 0.05): 0
- `n_significant` (adjusted-p < 0.10): 0
- `min adjusted-p`: 0.4113 (P-site A and P, BWM-depleted)
- `min raw-p`: 0.02597 (4 ties: T@A, R@E, A@P; and 0.04113 at P@P) — the discrete MW tied minimum at this n with ties
- `p_floor`: n/a — user did not flag a floor effect at the CSV level (family-wide `mw-floor-tight` instead)
- Per-site BH families:
  - A site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.519
  - E site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.519
  - P site: 20 tests, 0 hits at p_adj<0.05, min p_adj = 0.411

## Methods
Dylan proposed Mann-Whitney U / Wilcoxon rank-sum two-sided on per-replicate frequencies, n=6 BWM (control_d0/d5/d10 reps2&3 from each, six total) vs n=6 control, BH-FDR per E/P/A site (each site = own family of 20 AA hypotheses); user confirmed. Effect column is `log2_FC` of medians (`median_BWM/median_control`); test statistic is `U_stat`. The test answers "is feature X distributed at different per-rep frequencies between BWM and control replicates?", *not* "is feature X enriched at stall sites vs background" (that is the within-condition binomial) and *not* "is feature X stalling more on day_d than day_e" (that is between-timepoint Wilcoxon).

## Caveats
### Confirmed
- **timepoint-pooled-confound** (family-wide) — n=6 per condition is built by treating control_d0/d5/d10 reps as 6 control replicates (and analogously for BWM); time-by-condition interactions are pushed into the noise. If the perturbation effect varies across day_0/5/10 (which the BWM-internal Fisher contrasts in `timepoint_fisher_within_condition` may show), pooling here masks it.
- **mw-floor-tight** (family-wide) — MW two-sided exact-test p-floor for n=6 vs n=6 is ~0.00216, but the file's smallest observed raw p is 0.0260 (≈ 2/77, asymptotic-with-ties territory), one order of magnitude above the exact floor. Per-site BH over 20 AA tests means even an exact-floor hit on one feature would BH to ~0.043 — feasible but requires the very smallest p in the family.
- **n=6-modest-power** (family-wide) — null result is weakly informative; an effect of biologically meaningful size could still be missed.
- **no-near-nominal-signal** (per-CSV) — the entire AA-level alphabet is null-looking: only 4 of 60 tests reach raw p<0.05 (all at the tied discrete minimum 0.0260 or 0.0411). Even relaxed to p<0.10 only 3 more are added (K@A and K@P at 0.065, T@P at 0.093). This is a substantive empirical observation, not a structural ceiling.
- **weighted_log2_enrichment-absent** (per-CSV) — the effect column is the median ratio `log2_FC`; no count-weighted variant. Rare AAs (e.g. W, C with median freq ~0.5%) get the same column real estate as common ones (K @ ~10%); when ranking by |effect| treat with care.

### Considered but not applicable
- **asymptotic-with-ties** (per-CSV) — Dylan flagged that scipy may default to the asymptotic Z approximation under tied zero-frequency reps for rare AAs, which would shift p-values away from the strict exact-floor bound; user denied as not material at AA resolution. Empirical audit (`scripts/_for_claude_mw_branch_audit.py`) confirms denial on stronger grounds than originally written: (1) the "tied zero-frequency reps" premise does not hold — every replicate observes at least 13 stalls of every amino acid at every site (rarest cell: W at P-site in BWM_day0_rep3, k=13/3149), so no per-rep frequency is exactly 0; (2) 0 of 60 (site, AA) tests have any rank ties in the pooled 12-element sample, and `scipy.stats.mannwhitneyu(method='auto')` picked the exact branch for all 60; recomputed p matches the pipeline CSV to ~1e-16; (3) counterfactual `method='asymptotic'` shifts raw p by at most 0.015, flips zero raw-p<0.05 decisions, and leaves per-site BH conclusions unchanged (0 hits at FDR<0.05 either way; asymptotic min p_adj = 0.44/0.61/0.61 at P/A/E vs exact 0.41/0.52/0.52 — asymptotic is slightly *more* conservative). The earlier "discrete clustering supports denial" wording was not the actual reason and has been retired.

## For Chumeng (joint-reading hooks)
- Family: `between_condition_wilcoxon` — sister CSV in this family that should be reconciled: `between_condition_wilcoxon_codon.csv` (codon resolution; same design, same caveats).
- Open questions Chumeng should resolve at synthesis time:
  - Do the 4 nominal raw-p hits here (T@A, R@E, A@P, P@P) reappear with consistent direction in `per_timepoint_fisher_aa.csv` at any timepoint? If yes → real but small BWM-vs-control signal that MW lacked the n to formalize.
  - Is the closest-to-significant locus consistently the P site (min p_adj = 0.411 vs 0.519 at A/E)? Cross-check: does P-site signal also dominate the within-condition binomial enrichment for either condition?
  - Codon vs AA agreement: do the codons synonymous with K, R, A, P at the same site (`between_condition_wilcoxon_codon.csv`) show the same direction? Aggregation-driven vs codon-specific.
