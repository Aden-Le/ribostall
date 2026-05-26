---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_codon.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 366
n_significant_fdr05: 88
n_significant_fdr10: 112
min_p_adj: 2.20e-19
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "Pervasive in BWM Top-enriched (6 of 13 BWM-enriched cells flagged: A:AAT, A:TGT, A:TTT, P:TTT, E:TCT, E:ACG) and across many control Top-depleted rows (23 of 45 control-depleted cells flagged, distributed across all three sites). Largest BWM-enrichment magnitudes are E:ACG `log2_OR`=+1.987, A:TGT `log2_OR`=+1.511, A:AAT `log2_OR`=+1.171, P:TTT `log2_OR`=+1.156, A:TTT `log2_OR`=+1.034 — all rare-codon-flagged (at least one timepoint cell below the codon-resolution stability threshold, typically `day_X_count` < 50). Rank-direction is reliable; `log2_OR` magnitude is unstable at small k."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "One zero-cell row: BWM,E,ATA `day_0`=0 → OR=inf (raw p=0.103, p_adj=0.274 ns). Reported in the source file but excluded from Top-hits ranking. `log2_OR` is the reported effect column for finite rows; positive/negative bands are symmetric by construction on that axis."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Major direction flips at codon resolution. A:AAG (BWM `log2_OR`=-0.256 FDR=9.3e-03 vs control `log2_OR`=+0.460 FDR=1.0e-12 — direction flip on a high-count codon, both significant). E:AAG (BWM `log2_OR`=-0.477 FDR=2.6e-10 vs control `log2_OR`=+0.494 FDR=2.2e-19 — direction flip, both massively significant; the strongest single divergence in the entire family). P:AAG (BWM `log2_OR`=-0.452 FDR=1.7e-06 vs control `log2_OR`=-0.009 p_adj=0.938 ns — BWM-only depletion). A:GAT (BWM `log2_OR`=+0.660 FDR=1.6e-06 vs control ns — BWM-only enrichment). A:AAT (BWM `log2_OR`=+1.171 FDR=1.3e-04 vs control ns — BWM-only enrichment, rare-codon-flagged)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "BH-FDR is per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Stricter per-sub-family correction; codon p_adj=0.05 is materially harder to clear than aa p_adj=0.05. The family-wide minimum p_adj of 2.20e-19 (control,E,AAG) sits in a 61-codon BH family — an aa-level p_adj=2e-19 in a 20-row BH family would carry comparable evidence. Read codon vs aa p_adj on comparable axes."}
caveats_considered: []
headline: "Codon-level d5 vs d0 within-condition Fisher: 88/366 hits at FDR<0.05 (22/183 BWM, 66/183 control); file-minimum `p_adj`=2.20e-19 at control,E,AAG (`log2_OR`=+0.494) — the strongest single signal in the entire `timepoint_fisher_within_condition` family. Three AAG (Lys) direction flips dominate: A:AAG (BWM `log2_OR`=-0.256 FDR=9.3e-03 vs control `log2_OR`=+0.460 FDR=1.0e-12), E:AAG (BWM `log2_OR`=-0.477 FDR=2.6e-10 vs control `log2_OR`=+0.494 FDR=2.2e-19), and P:AAG (BWM `log2_OR`=-0.452 FDR=1.7e-06 vs control ns p_adj=0.938) — BWM depletes AAG d0->d5 at all three sites while control enriches it at A and E. Control's enrichment side is led by basic/polar codons at A (AAG, ATC, AAC, GTC, GAG, CGC) and E (AAG, GAG, CGC, TCC, CAG, AAC) plus a 9-codon P-site enrichment cluster (ATC, GTC, GAG, AAC, CGC, TCC, GTT, TTC, CTC). Control's depletion is broad — 45 codons across the three sites, with the largest non-rare magnitudes at A:GCA (`log2_OR`=-1.329 FDR=1.6e-10), P:TAT (`log2_OR`=-0.927 FDR=3.4e-07), and E:AAA (`log2_OR`=-0.718 FDR=5.0e-10). BWM's top enrichments are dominated by rare-codon-flagged rows (E:ACG `log2_OR`=+1.987, A:TGT `log2_OR`=+1.511, A:AAT `log2_OR`=+1.171, P:TTT `log2_OR`=+1.156, A:TTT `log2_OR`=+1.034); the non-rare BWM enrichments cluster at A:GAT `log2_OR`=+0.660, E:CAA `log2_OR`=+0.780, A:AAC `log2_OR`=+0.676, and E:AAT `log2_OR`=+0.678. Cross-contrast hook for Chumeng: site-E AAG flip reproduces in the d10_vs_d0_codon sister (BWM `log2_OR`=-0.324 FDR=1.2e-03 / control `log2_OR`=+0.406 FDR=1.3e-10)."
user_directives:
  - "(resume probe) order → 'Codon-first pairing (Recommended)' — d5_vs_d0_codon is the fifth and final file processed this session"
  - "(resume probe) layout → 'Same layout for all 5 (Recommended)'"
  - "(resume probe) caveat flow → 'If Dylan thinks the flags are appropriate then prompt me with them' — Dylan proposes per-CSV, user confirms each"
  - "(triage) test type → user confirmed Fisher's exact, BH-FDR per (condition, site) (locked from family, not re-asked)"
  - "(triage) per-CSV caveats → user confirmed all four Dylan-proposed: rare-codon-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction, larger-bh-family"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — Dylan re-read; test design unchanged"
  - "(history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `larger-bh-family` caveat added. Numbers and Top hits unchanged from prior run after spot-check."
  - "(2026-05-25) Top hits rewritten to `log2_OR` effect column and the family-wide strict-`p_adj`<0.05 selection rule (no row cap, no nominal-only fallback); BWM table dropped from 10 to 22 rows and control table dropped from 10 to 66 rows under the no-cap rule. Numbers, `n_significant` counts, and `min_p_adj` unchanged. Headline + per-CSV caveat blocks + For-Chumeng hooks re-quoted to `log2_OR`."
  - "(2026-05-25) Stage 4 Dylan cross-check vs `.qmd` (source of truth) — value drift / row-membership / ordering all clean (88 rows reconcile to the source CSV under the family rule; checker block 16 confirms BWM 13+9 / control 21+45; tiebreak BWM,A,depleted TAC → TGG/GGA/AAG ordered correctly by |log2_OR| desc). Three Dylan-side narrative-drift fixes applied during the rebuild: (a) frontmatter `headline` broadened from the old single-AAG-cell framing to all three AAG flips + control's broad pattern + BWM's rare-codon dominance, matching the .qmd; (b) Cross-contrast hook re-quoted to `log2_OR` form for the d10_vs_d0_codon sister; (c) `control-vs-BWM-divergent-direction.why` re-quoted to `log2_OR`."
  - "(2026-05-25) Stage 3b/3c narrative audit on CSV 16 (user-requested) — three drift items fixed in both `.qmd` (source of truth) and this Dylan source: (i) control,P,AAG `p_adj=0.819` corrected to `0.938` and `log2_OR=-0.030` corrected to `-0.009` (raw CSV row is `0.9941105552 / p=0.9077 / p_adj=0.9385`); fixed across Headline, frontmatter `headline`, Bio-interp paragraph 1, and `control-vs-BWM-divergent-direction.why` (4 spots Dylan-side + 3 spots `.qmd`-side). (ii) Headline `six-codon P-site enrichment cluster` corrected to `nine-codon` (the listed 9 codons ATC/GTC/GAG/AAC/CGC/TCC/GTT/TTC/CTC match the Numbers-at-a-glance `control,P: 9 enriched` count). (iii) Numbers-at-a-glance + frontmatter `n_significant_fdr10: 110 (BWM 30, control 80)` corrected to `112 (BWM 32, control 80)` (BWM was undercounted by 2; control unchanged)."
---

# Interpretation — timepoint_fisher_within_condition_d5_vs_d0_codon

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_codon.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (resume probe) "Continuing the in-progress family — which order should I work the 5 remaining CSVs?" → "Codon-first pairing (Recommended)" — this file is fifth and final in the session.
- (resume probe) "Re-confirm the file-level layout choice for the remaining 5 files?" → "Same layout for all 5 (Recommended)"
- (resume probe) "Streamline caveat confirmation across the family?" → "If Dylan thinks the flags are appropriate then prompt me with them, I don't know myself if they are prevalent on a csv by csv basis"
- (triage) "Per-CSV caveats Dylan wants to flag for d5_vs_d0_codon. Which apply?" → user confirmed all four Dylan-proposed: `rare-codon-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`, `larger-bh-family`
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" → Dylan re-read; test design unchanged.
- (history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `larger-bh-family` caveat added. Numbers and Top hits unchanged from prior run after spot-check.
- (2026-05-25) Top hits rewritten to `log2_OR` effect column and the family-wide strict-`p_adj`<0.05 selection rule (no row cap, no nominal-only fallback). BWM table 10 → 22 rows (13 enriched + 9 depleted); control table 10 → 66 rows (21 enriched + 45 depleted). `n_significant` counts and `min_p_adj` unchanged.
- (2026-05-25) Stage 4 Dylan cross-check vs `.qmd` (source of truth) — value drift / row-membership / ordering all clean. Three Dylan-side narrative-drift fixes applied during the rebuild: (a) frontmatter `headline` broadened from the old single-AAG-cell framing to all three AAG flips + control's broad pattern + BWM's rare-codon dominance, matching the .qmd; (b) Cross-contrast hook re-quoted to `log2_OR` form for the d10_vs_d0_codon sister; (c) `control-vs-BWM-divergent-direction.why` re-quoted to `log2_OR`.

## Headline
Codon-level d5 vs d0 within-condition Fisher: 88/366 hits at FDR<0.05 (22/183 BWM, 66/183 control). The file minimum `p_adj` is 2.20e-19 at control,E,AAG (`log2_OR`=+0.494) — the strongest single signal in the entire `timepoint_fisher_within_condition` family. **Three AAG (Lys) direction flips dominate**: **A:AAG** (BWM `log2_OR`=-0.256 FDR=9.3e-03 vs control `log2_OR`=+0.460 FDR=1.0e-12), **E:AAG** (BWM `log2_OR`=-0.477 FDR=2.6e-10 vs control `log2_OR`=+0.494 FDR=2.2e-19), and **P:AAG** is BWM-only (BWM `log2_OR`=-0.452 FDR=1.7e-06 vs control `log2_OR`=-0.009 p_adj=0.938 ns) — BWM depletes AAG d0→d5 at all three sites while control enriches it at A and E. Control's enrichment side is led by basic/polar codons at A (AAG, ATC, AAC, GTC, GAG, CGC) and E (AAG, GAG, CGC, TCC, CAG, AAC) plus a 9-codon P-site enrichment cluster (ATC, GTC, GAG, AAC, CGC, TCC, GTT, TTC, CTC). Control's depletion is broad — 45 codons across the three sites, with the largest non-rare magnitudes at A:GCA (`log2_OR`=-1.329 FDR=1.6e-10), P:TAT (`log2_OR`=-0.927 FDR=3.4e-07), and E:AAA (`log2_OR`=-0.718 FDR=5.0e-10). BWM's top enrichments are dominated by **rare-codon-flagged rows** (E:ACG `log2_OR`=+1.987, A:TGT `log2_OR`=+1.511, A:AAT `log2_OR`=+1.171, P:TTT `log2_OR`=+1.156, A:TTT `log2_OR`=+1.034) — small-count Fisher instability is at least as plausible as biology for these rows. **Cross-contrast hook for Chumeng**: site-E AAG flip reproduces in the d10_vs_d0_codon sister (BWM `log2_OR`=-0.324 FDR=1.2e-03 / control `log2_OR`=+0.406 FDR=1.3e-10) — a coherent d0-anchored codon-level AAG signature across two contrasts.

## Top hits

Selection rule (family-level, updated 2026-05-25): all rows with `p_adj` < 0.05, no row cap, no raw-p fallback; rows grouped by direction (↑ then ↓), then by site in A → P → E order, then sorted by `p_adj` ascending with `|log2_OR|` descending as tiebreaker.

### BWM (n_sig FDR<0.05 = 22 / 183)

| direction | feature | aa | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| ↑ | A:GAT | D | +0.660 | 1.63e-06 |  |
| ↑ | A:AAT | N | +1.171 | 1.27e-04 | rare-codon-low-count |
| ↑ | A:AAC | N | +0.676 | 5.47e-04 |  |
| ↑ | A:TGT | C | +1.511 | 0.0480 | rare-codon-low-count |
| ↑ | A:ATT | I | +0.479 | 0.0480 |  |
| ↑ | A:TTT | F | +1.034 | 0.0488 | rare-codon-low-count |
| ↑ | P:TTT | F | +1.156 | 0.0135 | rare-codon-low-count |
| ↑ | P:AAT | N | +0.584 | 0.0397 |  |
| ↑ | E:CAA | Q | +0.780 | 8.53e-06 |  |
| ↑ | E:TCT | S | +0.813 | 0.0101 | rare-codon-low-count |
| ↑ | E:ACT | T | +0.599 | 0.0177 |  |
| ↑ | E:ACG | T | +1.987 | 0.0329 | rare-codon-low-count |
| ↑ | E:AAT | N | +0.678 | 0.0329 |  |
| ↓ | A:TAC | Y | -0.700 | 1.03e-06 |  |
| ↓ | A:TGG | W | -0.699 | 0.0093 |  |
| ↓ | A:GGA | G | -0.299 | 0.0093 |  |
| ↓ | A:AAG | K | -0.256 | 0.0093 |  |
| ↓ | A:GCC | A | -0.481 | 0.0182 |  |
| ↓ | P:AAG | K | -0.452 | 1.74e-06 |  |
| ↓ | P:GGA | G | -0.347 | 0.0039 |  |
| ↓ | E:AAG | K | -0.477 | 2.57e-10 |  |
| ↓ | E:AGA | R | -0.560 | 0.0042 |  |

### control (n_sig FDR<0.05 = 66 / 183)

| direction | feature | aa | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| ↑ | A:AAG | K | +0.460 | 1.00e-12 |  |
| ↑ | A:ATC | I | +0.606 | 9.17e-12 |  |
| ↑ | A:AAC | N | +0.591 | 7.48e-08 |  |
| ↑ | A:GTC | V | +0.419 | 1.11e-04 |  |
| ↑ | A:GAG | E | +0.280 | 2.17e-04 |  |
| ↑ | A:CGC | R | +0.439 | 0.0067 |  |
| ↑ | P:ATC | I | +0.439 | 2.16e-06 |  |
| ↑ | P:GTC | V | +0.513 | 6.33e-06 |  |
| ↑ | P:GAG | E | +0.337 | 4.06e-05 |  |
| ↑ | P:AAC | N | +0.377 | 5.23e-05 |  |
| ↑ | P:CGC | R | +0.654 | 2.48e-04 |  |
| ↑ | P:TCC | S | +0.593 | 0.0017 |  |
| ↑ | P:GTT | V | +0.297 | 0.0048 |  |
| ↑ | P:TTC | F | +0.245 | 0.0133 |  |
| ↑ | P:CTC | L | +0.305 | 0.0276 |  |
| ↑ | E:AAG | K | +0.494 | 2.20e-19 |  |
| ↑ | E:GAG | E | +0.423 | 3.72e-09 |  |
| ↑ | E:CGC | R | +0.633 | 7.42e-04 |  |
| ↑ | E:TCC | S | +0.456 | 0.0078 |  |
| ↑ | E:CAG | Q | +0.358 | 0.0173 |  |
| ↑ | E:AAC | N | +0.205 | 0.0364 |  |
| ↓ | A:GCA | A | -1.329 | 1.59e-10 |  |
| ↓ | A:TTG | L | -0.651 | 1.05e-06 |  |
| ↓ | A:TAT | Y | -0.818 | 1.98e-06 |  |
| ↓ | A:CTG | L | -1.216 | 2.98e-05 | rare-codon-low-count |
| ↓ | A:TGG | W | -0.735 | 3.96e-05 |  |
| ↓ | A:GCG | A | -1.586 | 1.03e-04 | rare-codon-low-count |
| ↓ | A:GGC | G | -1.246 | 2.17e-04 | rare-codon-low-count |
| ↓ | A:GTG | V | -0.653 | 2.88e-04 |  |
| ↓ | A:TTA | L | -1.717 | 0.0053 | rare-codon-low-count |
| ↓ | A:ACG | T | -0.979 | 0.0053 | rare-codon-low-count |
| ↓ | A:CCT | P | -1.224 | 0.0072 | rare-codon-low-count |
| ↓ | A:CCA | P | -0.276 | 0.0076 |  |
| ↓ | A:AAA | K | -0.347 | 0.0321 |  |
| ↓ | A:CGA | R | -0.797 | 0.0466 | rare-codon-low-count |
| ↓ | A:TAC | Y | -0.231 | 0.0466 |  |
| ↓ | A:ACA | T | -0.452 | 0.0469 |  |
| ↓ | P:TAT | Y | -0.927 | 3.40e-07 |  |
| ↓ | P:AAA | K | -0.669 | 3.40e-07 |  |
| ↓ | P:GCA | A | -1.035 | 1.42e-05 | rare-codon-low-count |
| ↓ | P:AAT | N | -0.531 | 5.23e-05 |  |
| ↓ | P:ACA | T | -0.744 | 0.0019 |  |
| ↓ | P:CCG | P | -0.962 | 0.0043 | rare-codon-low-count |
| ↓ | P:ATG | M | -0.403 | 0.0043 |  |
| ↓ | P:ATT | I | -0.350 | 0.0051 |  |
| ↓ | P:AGT | S | -0.929 | 0.0058 | rare-codon-low-count |
| ↓ | P:CAT | H | -0.517 | 0.0062 |  |
| ↓ | P:CTG | L | -0.901 | 0.0133 | rare-codon-low-count |
| ↓ | P:TTA | L | -2.059 | 0.0229 | rare-codon-low-count |
| ↓ | P:GCG | A | -1.227 | 0.0284 | rare-codon-low-count |
| ↓ | P:CCT | P | -0.869 | 0.0451 | rare-codon-low-count |
| ↓ | E:AAA | K | -0.718 | 4.96e-10 |  |
| ↓ | E:CCA | P | -0.589 | 3.10e-09 |  |
| ↓ | E:GCA | A | -1.075 | 1.91e-05 | rare-codon-low-count |
| ↓ | E:ACA | T | -0.814 | 0.0018 | rare-codon-low-count |
| ↓ | E:CGA | R | -1.277 | 0.0025 | rare-codon-low-count |
| ↓ | E:CCT | P | -1.521 | 0.0067 | rare-codon-low-count |
| ↓ | E:TAT | Y | -0.632 | 0.0067 |  |
| ↓ | E:TCA | S | -0.619 | 0.0067 |  |
| ↓ | E:CCG | P | -0.923 | 0.0172 | rare-codon-low-count |
| ↓ | E:GTA | V | -0.861 | 0.0271 | rare-codon-low-count |
| ↓ | E:GTT | V | -0.291 | 0.0271 |  |
| ↓ | E:GCG | A | -0.995 | 0.0319 | rare-codon-low-count |
| ↓ | E:CAT | H | -0.500 | 0.0364 |  |
| ↓ | E:CTG | L | -0.612 | 0.0462 | rare-codon-low-count |
| ↓ | E:TTA | L | -1.963 | 0.0497 | rare-codon-low-count |

## Numbers at a glance
- `n_tests`: 366 (183 per condition; BH families of 61 codons per (condition, site))
- `n_significant` (adjusted-p < 0.05): 88 (BWM 22, control 66)
- `n_significant` (adjusted-p < 0.10): 112 (BWM 32, control 80)
- `min adjusted-p`: 2.20e-19 (control,E,AAG)
- Per (condition, site) at FDR<0.05:
  - BWM,A: 11 (6 enriched / 5 depleted); min `p_adj` = 1.03e-06 (TAC depleted)
  - BWM,P: 4 (2 enriched / 2 depleted); min `p_adj` = 1.74e-06 (AAG depleted)
  - BWM,E: 7 (5 enriched / 2 depleted); min `p_adj` = 2.57e-10 (AAG depleted)
  - control,A: 22 (6 enriched / 16 depleted); min `p_adj` = 1.00e-12 (AAG enriched)
  - control,P: 23 (9 enriched / 14 depleted); min `p_adj` = 3.40e-07 (TAT depleted / AAA depleted tied)
  - control,E: 21 (6 enriched / 15 depleted); min `p_adj` = 2.20e-19 (AAG enriched) ← family-wide minimum
- Zero-cell row reported but excluded from sig results: BWM,E,ATA (`day_0`=0 → OR=inf; raw p=0.103, p_adj=0.274 ns).

## Methods
Same as the rest of the family. Fisher's exact (two-sided) on a 2x2 of (codon_count, total - codon_count) per timepoint within one condition; BH-FDR within each (condition, site) family of 61 codons. **One zero-cell row**: BWM,E,ATA `day_0`=0 → OR=inf (raw p=0.103, p_adj=0.274 ns); reported in the source file but excluded from Top-hits ranking. Counts (`day_X_count`) and totals (`day_X_total`) are summed across replicates before the test, hence the family-level pseudoreplication caveat. Top-hits selection rule (family-level, updated 2026-05-25): all rows with `p_adj` < 0.05, no row cap, no raw-p fallback.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)*. (Inherited.)
- **large-N-Fisher-anticonservative** *(family-wide)*. control,E,AAG `log2_OR`=+0.494 → `p_adj`=2.20e-19 — the smallest p_adj in this family. Rank by `log2_OR`, not p magnitude. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)*. (Inherited.)
- **within-condition-clean** *(family-wide)*. (Inherited.)
- **rare-codon-low-count** *(per-CSV)* — pervasive in BWM's top-enriched. **6 of 13 BWM-enriched cells flagged** (A:AAT, A:TGT, A:TTT, P:TTT, E:TCT, E:ACG) and **23 of 45 control-depleted cells flagged** (distributed across all three sites: TTA, GCG, CTG, CCT at all three; GGC and ACG at A; CGA at A and E; GCA, CCG at P and E; AGT at P; ACA and GTA at E). At least one timepoint cell for each flagged row is below the codon-resolution stability threshold (typically `day_X_count` < 50). The BWM-enriched headline magnitudes (E:ACG `log2_OR`=+1.987, A:TGT `log2_OR`=+1.511, A:AAT `log2_OR`=+1.171, P:TTT `log2_OR`=+1.156, A:TTT `log2_OR`=+1.034) are unstable; rank-direction is reliable but the precise `log2_OR` is sensitive to small-count Fisher's exact behaviour. Same caution applies to the long control-depleted rare list.
- **OR-direction-asymmetry** *(per-CSV)* — one zero-cell row (BWM,E,ATA, `day_0`=0 → OR=inf, ns). Excluded from Top-hits ranking. `log2_OR` is the reported effect column for finite rows; positive/negative bands are symmetric by construction on that axis.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — major direction flips at codon resolution; flagged for Chumeng's reconciliation.
  - **A:AAG**: BWM `log2_OR`=-0.256 (FDR=9.3e-03) vs control `log2_OR`=+0.460 (FDR=1.0e-12) — direction flip on a high-count codon, both significant.
  - **E:AAG**: BWM `log2_OR`=-0.477 (FDR=2.6e-10) vs control `log2_OR`=+0.494 (FDR=2.2e-19) — direction flip, both massively significant. **The strongest single divergence in the entire family.**
  - **P:AAG**: BWM `log2_OR`=-0.452 (FDR=1.7e-06) vs control `log2_OR`=-0.009 (p_adj=0.938 ns) — BWM-only depletion.
  - **A:GAT**: BWM `log2_OR`=+0.660 (FDR=1.6e-06) vs control ns — BWM-only enrichment.
  - **A:AAT**: BWM `log2_OR`=+1.171 (FDR=1.3e-04) vs control ns — BWM-only enrichment, rare-codon-flagged.
- **larger-bh-family** *(per-CSV)* — BH-FDR per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Read codon vs aa p_adj on comparable axes. The family-wide minimum `p_adj`=2.20e-19 (control,E,AAG) sits in a 61-codon sub-family — comparable evidence weight to an aa `p_adj`~2e-19 in a 20-row sub-family. A codon `p_adj`=0.05 here is materially harder to clear than an aa `p_adj`=0.05 in the aa companion, so cross-resolution comparisons should account for the BH family-size scaling.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: aa-level d5_vs_d0 (parent), codon-level d10_vs_d0 / d10_vs_d5.
- Open questions Chumeng should resolve:
  - **AAG cross-site / cross-contrast tracking question.** This file: BWM AAG depleted at A (`log2_OR`=-0.256 sig), E (`log2_OR`=-0.477 sig), P (`log2_OR`=-0.452 sig); control AAG enriched at A (`log2_OR`=+0.460 sig), E (`log2_OR`=+0.494 sig), P ns. d10_vs_d0_codon: same direction-flip at E (BWM `log2_OR`=-0.324 sig vs control `log2_OR`=+0.406 sig). d10_vs_d5_codon: BWM,E:AAG is in the fully-null BWM-E sub-family (smallest `p_adj`=0.114 ns). Does this numerical pattern across three contrasts support a single d0-anchored AAG signature, or could the two-vs-d0 cells share a common-baseline structure independent of biology? Does it map to the aa-level E:K cell (`timepoint_fisher_within_condition_d5_vs_d0_aa`: BWM E:K depleted / control E:K enriched, same direction)? Chumeng to weigh against alternative readings.
  - **Are AAG and AAA both contributing to the E:K aa-level flip, or is the signal AAG-specific?** This file: BWM E:AAG `log2_OR`=-0.477 sig, BWM E:AAA ns under strict `p_adj`<0.05. Control E:AAG `log2_OR`=+0.494 sig, control E:AAA `log2_OR`=-0.718 sig (depleted). So at site E, **AAG and AAA shift in opposite directions in control** (AAG↑ but AAA↓), while in BWM only AAG is significant. The aa-level E:K flip is a vector sum: control E:K enrichment reflects that AAG's enrichment outweighs AAA's depletion in count terms; BWM E:K depletion reflects AAG's depletion alone.
  - **BWM's heavy reliance on rare codons for top-enriched signal** (E:ACG, A:TGT, P:TGT, P:ACA, A:AAT, P:TTT, A:TTT, E:TCT) is itself a finding: it suggests BWM at d5 is shifting stall-site composition toward rare codons. Cross-check with `timepoint_fisher_within_condition_d10_vs_d0_codon` (the longer-window version): BWM,A,TGT is `log2_OR`≈+1.91 there too (p_adj=0.0148 in the prior odds_ratio framing). Pattern persists from d0→d5→d10 in BWM. Could be a real biological signal of perturbation pushing translation toward rare codons, OR a small-count artifact reproducible across reps. The combination of pseudoreplication + small absolute counts + same direction across two contrasts is suggestive but not conclusive.
  - **Control E:P (aa) was reported at p_adj=2.27e-12 in the aa companion**; the corresponding codon-level Pro picture here: control E:CCA `log2_OR`=-0.589 (FDR=3.1e-09), control E:CCT `log2_OR`=-1.521 (rare, FDR=6.7e-03), control E:CCG `log2_OR`=-0.923 (rare, FDR=1.7e-02). CCC is absent from the sig list at strict `p_adj`<0.05. Three of the four Pro synonyms move the same direction at site E in control, including the high-magnitude CCA cell. Coherent direction across synonyms is consistent with an amino-acid-level reading rather than a single-synonym codon-usage shift; Chumeng to weigh against the per-CSV `control-E-P-extreme-anticonservative` flag in the aa companion.
