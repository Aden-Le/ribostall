---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_codon.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 366
n_significant_fdr05: 22
n_significant_fdr10: 29
min_p_adj: 5.71e-09
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "Both BWM Top-depleted rows under the strict `p_adj` < 0.05 cutoff are rare-codon flagged: BWM,A,CTG (day_10=12, day_5=55, log2_OR=-1.419) and BWM,P,TTT (day_10=25, day_5=87, log2_OR=-1.023). `log2_OR` magnitude unstable; rank-direction reliable. Note also BWM site E is fully null at FDR<0.05 (smallest p_adj=0.114) — the BH-family scaling combined with small per-codon counts wipes out per-synonym significance even where aa-level effects exist."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "No zero-cell rows in this file (all day_X_count > 0). Caveat preserved as ranking-axis discipline only — `log2_OR` is the reported effect column."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Two codon-level direction flips with control significant and BWM trending opposite but ns: P:GTC (BWM log2_OR=+0.330 ns vs control log2_OR=-0.426 sig), E:GAG (BWM log2_OR=+0.107 ns vs control log2_OR=-0.292 sig). Common signal: site-A and site-P GGA enriched in both conditions (BWM A:GGA log2_OR=+0.382 / P:GGA log2_OR=+0.356; control A:GGA log2_OR=+0.435 / P:GGA log2_OR=+0.516)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "BH-FDR is per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Stricter per-sub-family correction; codon p_adj=0.05 is materially harder to clear than aa p_adj=0.05. Textbook case visible in this file: BWM site E is fully null at codon level (smallest p_adj=0.114) while the aa companion shows BWM,E:G/R/I sig at FDR<0.05 — the larger BH family wipes out per-synonym significance even when aggregate-aa signal exists. Read codon vs aa p_adj on comparable axes."}
caveats_considered: []
headline: "d10 vs d5 within-condition Fisher (codon-level): 22/366 hits at FDR<0.05 (4/183 BWM, 18/183 control); BWM site E is fully null at codon level (smallest p_adj=0.114) despite BWM,E:G/R/I being sig at the aa level — larger-bh-family caveat case study. Largest-magnitude shared-direction cell in this CSV: site-A and site-P GGA enrichment shared between BWM and control (BWM A:GGA log2_OR=+0.382 FDR<5e-4 / P:GGA log2_OR=+0.356 FDR<3e-3; control A:GGA log2_OR=+0.435 FDR<2e-6 / P:GGA log2_OR=+0.516 FDR<6e-9); within this CSV the aa-level Gly enrichment between d5->d10 concentrates on GGA, with other Gly synonyms ns. Largest-magnitude divergence cells: P:GTC (BWM log2_OR=+0.330 ns vs control log2_OR=-0.426 sig), E:GAG (BWM log2_OR=+0.107 ns vs control log2_OR=-0.292 sig)."
user_directives:
  - "(resume probe) order -> 'Codon-first pairing (Recommended)' — d10_vs_d5_codon is the third file processed this session"
  - "(resume probe) layout -> 'Same layout for all 5 (Recommended)'"
  - "(resume probe) caveat flow -> 'If Dylan thinks the flags are appropriate then prompt me with them' — Dylan proposes per-CSV, user confirms each"
  - "(triage) test type -> user confirmed Fisher's exact, BH-FDR per (condition, site) (locked from family, not re-asked)"
  - "(triage) per-CSV caveats -> user confirmed all four Dylan-proposed: rare-codon-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction, larger-bh-family"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — Dylan re-read; test design unchanged from rest of family"
  - "(history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `larger-bh-family` caveat was added. Numbers and Top hits unchanged from prior run after spot-check."
  - "(2026-05-23) Top hits rewritten to log2_OR effect column and the family-wide strict-`p_adj<0.05` selection rule (no row cap, no nominal-only fallback); BWM table dropped from 10 to 4 rows, control table dropped from 10 to 18 rows under the no-cap rule. Numbers, n_significant counts, and Methods unchanged."
  - "(2026-05-23) Stage 4 Dylan cross-check vs `.qmd` (source of truth) — value drift / row-membership / ordering all clean. Three Dylan-side narrative-drift fixes applied: (a) frontmatter `headline` `'site-P GGA enriched in both'` broadened to `'site-A and site-P GGA enrichment shared between BWM and control'` to match body Headline + qmd Headline (stale single-site framing from before the log2_OR rebuild broadened to both sites); (b) frontmatter `control-vs-BWM-divergent-direction.why` `'Common signal: P:GGA enriched in both'` similarly broadened to include site-A GGA; (c) body caveat `rare-codon-low-count` `'day_X_count < 30'` rewritten to qmd's framing `'at least one timepoint cell below the codon-resolution stability threshold (typically < 50)'` because day_5=55 (CTG) and day_5=87 (TTT) both exceed 30, making the `< 30` framing ambiguous and arguably false if read as 'all day_X counts'."
---

# Interpretation — timepoint_fisher_within_condition_d10_vs_d5_codon

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_codon.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (resume probe) "Continuing the in-progress family — which order should I work the 5 remaining CSVs?" -> "Codon-first pairing (Recommended)" — this file is third in the session.
- (resume probe) "Re-confirm the file-level layout choice for the remaining 5 files?" -> "Same layout for all 5 (Recommended)"
- (resume probe) "Streamline caveat confirmation across the family?" -> "If Dylan thinks the flags are appropriate then prompt me with them, I don't know myself if they are prevalent on a csv by csv basis"
- (triage) "Per-CSV caveats Dylan wants to flag for d10_vs_d5_codon. Which apply?" -> user confirmed all four Dylan-proposed: `rare-codon-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`, `larger-bh-family`
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" -> Dylan re-read; test design unchanged.
- (history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `larger-bh-family` caveat added. Numbers and Top hits unchanged from prior run after spot-check.
- (2026-05-23) Top hits rewritten to `log2_OR` effect column and the family-wide strict-`p_adj<0.05` selection rule (no row cap, no nominal-only fallback). Headline and caveat blocks re-quoted to `log2_OR`.
- (2026-05-23) Stage 4 Dylan cross-check vs `.qmd` (source of truth) — value drift / row-membership / ordering all clean. Three Dylan-side narrative-drift fixes applied: (a) frontmatter `headline` `'site-P GGA enriched in both'` -> `'site-A and site-P GGA enrichment shared between BWM and control'` (matches body + qmd; was stale single-site framing from before the log2_OR rebuild broadened to both sites); (b) frontmatter `control-vs-BWM-divergent-direction.why` `'Common signal: P:GGA enriched in both'` -> includes site-A GGA; (c) body caveat `rare-codon-low-count` `'day_X_count < 30'` -> qmd's framing `'at least one timepoint cell below the codon-resolution stability threshold (typically < 50)'` because day_5=55 (CTG) and day_5=87 (TTT) both exceed 30, so `< 30` was ambiguous and arguably false if read as 'all day_X counts'.

## Headline
d10 vs d5 within-condition Fisher (codon-level): 22/366 hits at FDR<0.05 (4/183 BWM, 18/183 control). BWM concentrates on **site-A and site-P GGA enrichment** (A:GGA `log2_OR`=+0.382 FDR<5e-4, P:GGA `log2_OR`=+0.356 FDR<2.4e-3) plus two rare-codon depletions (A:CTG `log2_OR`=-1.419 FDR=0.028; P:TTT `log2_OR`=-1.023 FDR=0.035). Control sweeps broadly across all three sites (A: 5 hits dominated by GGA enriched / CAA depleted / AAC depleted plus GCC and CCA enriched; E: 7 hits including GCC/CTC/CCA/ACC/GTC/ATC enriched and GAG depleted; P: 6 hits with GGA `log2_OR`=+0.516 FDR<6e-9 the leading feature plus CAA/ATT/GTC/CTT/ATC depleted). **The site-A and site-P GGA enrichment is the largest-magnitude shared-direction cell in this CSV**: same direction in BWM and control across both sites and at large magnitude. Notable direction flips with control significant and BWM trending opposite but ns: **P:GTC** (BWM `log2_OR`=+0.330 ns vs control `log2_OR`=-0.426 FDR=0.017); **E:GAG** (BWM `log2_OR`=+0.107 ns vs control `log2_OR`=-0.292 FDR=0.0069).

## Top hits

Selection rule (family-level, updated 2026-05-22): all rows with `p_adj` < 0.05, no row cap; rows grouped by `site` in A -> P -> E order then sorted by `p_adj` ascending with `|log2_OR|` descending as tiebreaker.

### BWM (n_sig FDR<0.05 = 4 / 183)

| direction | feature | aa | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| ↑ | A:GGA | G | +0.382 | 4.68e-04 |  |
| ↑ | P:GGA | G | +0.356 | 0.00239 |  |
| ↓ | A:CTG | L | -1.419 | 0.0276 | rare-codon-low-count |
| ↓ | P:TTT | F | -1.023 | 0.0347 | rare-codon-low-count |

BWM site E is fully null at codon level (smallest `p_adj` = 0.114). See the `larger-bh-family` caveat.

### control (n_sig FDR<0.05 = 18 / 183)

| direction | feature | aa | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| ↑ | A:GGA | G | +0.435 | 1.32e-06 |  |
| ↑ | A:GCC | A | +0.474 | 0.0154 |  |
| ↑ | A:CCA | P | +0.338 | 0.0320 |  |
| ↑ | P:GGA | G | +0.516 | 5.71e-09 |  |
| ↑ | E:GCC | A | +0.613 | 0.00111 |  |
| ↑ | E:CTC | L | +0.508 | 0.00111 |  |
| ↑ | E:CCA | P | +0.472 | 0.00111 |  |
| ↑ | E:ACC | T | +0.479 | 0.00691 |  |
| ↑ | E:GTC | V | +0.323 | 0.0338 |  |
| ↑ | E:ATC | I | +0.293 | 0.0401 |  |
| ↓ | A:CAA | Q | -0.857 | 1.02e-06 |  |
| ↓ | A:AAC | N | -0.568 | 3.00e-04 |  |
| ↓ | P:CAA | Q | -0.568 | 0.0118 |  |
| ↓ | P:ATT | I | -0.545 | 0.0128 |  |
| ↓ | P:GTC | V | -0.426 | 0.0170 |  |
| ↓ | P:CTT | L | -0.395 | 0.0390 |  |
| ↓ | P:ATC | I | -0.304 | 0.0390 |  |
| ↓ | E:GAG | E | -0.292 | 0.00691 |  |

## Numbers at a glance
- `n_tests`: 366 (183 per condition; BH families of 61 codons per (condition, site))
- `n_significant` (adjusted-p < 0.05): 22 (BWM 4, control 18)
- `n_significant` (adjusted-p < 0.10): 29 (BWM 9, control 20)
- `min adjusted-p`: 5.71e-09 (control,P,GGA)
- Per (condition, site) at FDR<0.05:
  - BWM,A: 2 (GGA enriched, CTG depleted); min p_adj = 4.68e-04
  - BWM,P: 2 (GGA enriched, TTT depleted); min p_adj = 0.00239
  - BWM,E: 0 (smallest p_adj = 0.114)
  - control,A: 5 (GGA, GCC, CCA enriched; CAA, AAC depleted); min p_adj = 1.02e-06 (CAA depleted)
  - control,P: 6 (GGA enriched; CAA, ATT, GTC, CTT, ATC depleted); min p_adj = 5.71e-09 (GGA enriched)
  - control,E: 7 (GCC, CTC, CCA, ACC, GTC, ATC enriched; GAG depleted); min p_adj = 0.00111 (3-way tie at GCC/CTC/CCA)

## Methods
Same as the rest of the family. **BWM site E is fully null at FDR<0.05** (smallest p_adj=0.114) — the d10_vs_d5 contrast has no detectable codon-level shift at site E in BWM, even though the aa-level d10_vs_d5 file showed E:G/E:R/E:I significant. This is consistent with codon-usage-shift features being washed out when aggregated to amino acids: small effects at multiple synonyms can sum to a sig aa-level effect even when no single synonym is sig.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — replicates summed; p anti-conservative. (Inherited.)
- **large-N-Fisher-anticonservative** *(family-wide)* — totals 6945/8788/11177/11935; control,P,GGA `log2_OR`=+0.516 -> p_adj=5.7e-9. Rank by `log2_OR`. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — 6 sub-families of 61, corrected independently. (Inherited.)
- **within-condition-clean** *(family-wide)*. (Inherited.)
- **rare-codon-low-count** *(per-CSV)* — both BWM Top-depleted rows under the strict `p_adj` < 0.05 cutoff have at least one timepoint cell below the codon-resolution stability threshold (typically `day_X_count` < 50): A:CTG (day_10=12 / day_5=55, `log2_OR`=-1.419) and P:TTT (day_10=25 / day_5=87, `log2_OR`=-1.023). Magnitude unstable; rank-direction reliable.
- **OR-direction-asymmetry** *(per-CSV)* — no zero-cell rows. Ranking-axis discipline only; `log2_OR` is the reported effect column (already log-transformed), so positive/negative bands are symmetric by construction.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — two direction flips with control significant and BWM trending opposite but ns: **P:GTC** (BWM `log2_OR`=+0.330 ns p_value=0.0104 / p_adj=0.159 vs control `log2_OR`=-0.426 FDR=0.017); **E:GAG** (BWM `log2_OR`=+0.107 ns vs control `log2_OR`=-0.292 FDR=0.0069). Several control-only-significant cells where BWM is in the same direction but ns (E:CTC, E:CCA, P:CAA, P:ATT, A:CAA, A:AAC).
- **larger-bh-family** *(per-CSV)* — BH-FDR per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Read codon vs aa p_adj on comparable axes. **Textbook case in this file**: BWM site E is fully null at FDR<0.05 (smallest p_adj=0.114) while the aa companion shows BWM,E:G/R/I significant — the larger BH family wipes out per-synonym significance even when aa-level signal exists. Implication: a codon p_adj=0.114 here may carry the same evidence weight as an aa p_adj~0.04 in the aa companion.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: aa-level d10_vs_d5 (parent), plus the d10_vs_d0 / d5_vs_d0 codon contrasts.
- Open questions Chumeng should resolve:
  - **Is the strong site-A/P GGA enrichment shared across all 4 Gly codons?** This file: A:GGA `log2_OR`=+0.382 in BWM (FDR<5e-4) is the only Gly codon at FDR<0.05 at site A; control A:GGA `log2_OR`=+0.435 (FDR<1.4e-6) is also the only A-site Gly codon at FDR<0.05 (GGC, GGG, GGT all ns). Same at site P: BWM P:GGA `log2_OR`=+0.356 (FDR<3e-3) is the only Gly codon to clear FDR; control P:GGA `log2_OR`=+0.516 (FDR<6e-9) is the leading Gly codon (GGT, GGC, GGG ns). **Conclusion: aa-level Gly enrichment between d5->d10 is GGA-codon-driven, not amino-acid-level**. This is a codon-usage signature.
  - **Why is BWM site E completely null at codon level in this contrast** (smallest p_adj=0.114) when the aa-level file shows BWM,E,G/R/I sig at FDR<0.05? Because the aa effects are spread across multiple synonymous codons (G across GGA/GGC/GGG/GGT etc.) and no single codon clears the per-(condition,site) FDR alone. Suggests aa-level aggregation gains power at site E for d10_vs_d5 in BWM.
  - **BWM hit-count comparison across codon contrasts in this family** (for Chumeng to weigh against the between_timepoint_wilcoxon family). This file: BWM has 4 codon-level FDR<0.05 hits in d10_vs_d5. Does `_d5_vs_d0_codon_interpretation` show a comparable, larger, or smaller BWM hit count? If d10_vs_d5 carries more BWM signal than d5_vs_d0 here, does that align with or contradict the per-contrast pattern visible in the between_timepoint_wilcoxon family?
  - **Direction-flip P:GTC** (BWM `log2_OR`=+0.330 ns vs control `log2_OR`=-0.426 sig) is novel to this contrast. Cross-check whether `between_condition_wilcoxon_codon` reports any near-significant signal on GTC at site P (codon-level GTC depletion in control over time would not appear in time-collapsed Wilcoxon, but feature-direction divergence might).
