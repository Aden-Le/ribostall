---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_codon.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 366
n_significant_fdr05: 88
n_significant_fdr10: 110
min_p_adj: 2.20e-19
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "Pervasive in BWM Top hits. **All 5 BWM Top-enriched** are rare-codon-flagged (E:ACG day_0=4 OR=3.96; A:TGT day_0=7 OR=2.85; P:TGT OR=2.42; A:AAT OR=2.25; P:TTT OR=2.23). **All 5 control Top-depleted** are rare-codon-flagged (TTA at all 3 sites with day_X_count 1-5; GCG at A; GCA at A more sparsely). Headline BWM enriched magnitudes (3.96/2.85/2.42/2.28/2.25) are unstable; rank-direction is reliable but the precise OR is sensitive to small-count Fisher's exact behavior."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "One zero-cell row: BWM,E,ATA day_0=0 → OR=inf (raw p=0.103, p_adj=0.274 ns). Reported in the file but excluded from Top hits ranking. Use log2(OR) for ordering finite rows; flag inf/0 cells explicitly."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Major direction flips at codon resolution. A:AAG (BWM↓0.84 sig vs control↑1.38 FDR=1.00e-12 — direction flip on a high-count codon, both heavily sig). E:AAG (BWM↓0.72 FDR=2.57e-10 vs control↑1.41 FDR=2.20e-19 — the strongest single divergence in the entire family). P:AAG (BWM↓0.73 FDR=1.74e-06 vs control 0.98 ns — BWM-only). A:GAT (BWM↑1.58 FDR=1.63e-06 vs control 0.98 ns — BWM-only). A:AAT (BWM↑2.25 FDR=1.27e-04 vs control 0.92 ns — BWM-only enrichment, rare-codon-flagged)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "BH-FDR is per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Stricter per-sub-family correction; codon p_adj=0.05 is materially harder to clear than aa p_adj=0.05. The family-wide minimum p_adj of 2.20e-19 (control,E,AAG) sits in a 61-codon BH family — an aa-level p_adj=2e-19 in a 20-row BH family would carry comparable evidence. Read codon vs aa p_adj on comparable axes."}
caveats_considered: []
headline: "Codon-level d5 vs d0 within-condition Fisher: 88/366 hits at FDR<0.05 (22/183 BWM, 66/183 control); min p_adj = 2.20e-19 at control,E,AAG (Lys). Largest-magnitude divergence cells: A,AAG (BWM↓0.84 FDR<0.01 vs control↑1.38 FDR<1e-12), E,AAG (BWM↓0.72 FDR<3e-10 vs control↑1.41 FDR<3e-19), P,AAG (BWM↓0.73 FDR<2e-6 vs control 0.98 ns). Largest-magnitude shared-direction cells: A,AAC↑ (BWM↑1.70 sig from d10_vs_d0 sister carries forward; control↑1.51 sig here) and the broad d5-vs-d0 enrichment pattern across A-site codons (control AAG↑1.38, ATC↑1.52, AAC↑1.51, GTC↑1.34). BWM's top-enriched codons are dominated by **rare codons with large effect** (E:ACG↑3.96, A:TGT↑2.85, P:TGT↑2.42, P:ACA↑2.28, A:AAT↑2.25, P:TTT↑2.23) — flagged via rare-codon-low-count; flag rare-codon-low-count + alternative-explanation: small-count Fisher instability is at least as plausible as biology for these rows. Cross-contrast hook for Chumeng: site-E AAG flip magnitude here (BWM 0.72 / control 1.41) is similar to d10_vs_d0_codon (BWM 0.80 / control 1.33)."
user_directives:
  - "(resume probe) order → 'Codon-first pairing (Recommended)' — d5_vs_d0_codon is the fifth and final file processed this session"
  - "(resume probe) layout → 'Same layout for all 5 (Recommended)'"
  - "(resume probe) caveat flow → 'If Dylan thinks the flags are appropriate then prompt me with them' — Dylan proposes per-CSV, user confirms each"
  - "(triage) test type → user confirmed Fisher's exact, BH-FDR per (condition, site) (locked from family, not re-asked)"
  - "(triage) per-CSV caveats → user confirmed all four Dylan-proposed: rare-codon-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction, larger-bh-family"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — Dylan re-read; test design unchanged"
  - "(history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `larger-bh-family` caveat added. Numbers and Top hits unchanged from prior run after spot-check."
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
- (history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `larger-bh-family` caveat added. Numbers and Top hits unchanged from prior run after spot-check."

## Headline
Codon-level d5 vs d0 within-condition Fisher: 88/366 hits at FDR<0.05 (22/183 BWM, 66/183 control). The minimum p_adj is 2.20e-19 at control,E,AAG (Lys). **Largest-magnitude divergence cells (per-cell)**: **A:AAG** (BWM↓0.84 FDR<0.01 vs control↑1.38 FDR<1e-12), **E:AAG** (BWM↓0.72 FDR<3e-10 vs control↑1.41 FDR<3e-19), and **P:AAG** is BWM-only (BWM↓0.73 FDR<2e-6 vs control 0.98 ns). Cross-contrast hook for Chumeng: the d10_vs_d0_codon E,AAG cell has similar direction (BWM↓0.80 / control↑1.33) — does the numerical sequence support a single d0-anchored pattern, or could the two-vs-d0 contrasts share a common-baseline structure independent of biology? **Alternative-explanation flags for the file-minimum p_adj rows**: control,E,AAG p_adj=2.20e-19 at OR=1.41 (35% relative shift) — large-N-Fisher anti-conservative + pseudoreplicated; treat p magnitude as direction-only. BWM's top-enriched codons are dominated by **rare codons with large effect** (E:ACG↑3.96, A:TGT↑2.85, P:TGT↑2.42, P:ACA↑2.28, A:AAT↑2.25, P:TTT↑2.23) — flag rare-codon-low-count + alternative-explanation: small-count Fisher instability is at least as plausible as biology. Control's enriched signal is led by site-A codons (AAG↑1.38, ATC↑1.52, AAC↑1.51, GTC↑1.34) and site-E AAG↑1.41.

## Top hits

### BWM (n_sig FDR<0.05 = 22 / 183)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | E:ACG (Thr) | 3.963 | 0.03294 | rare-codon-low-count |
| ↑ | A:TGT (Cys) | 2.849 | 0.04797 | rare-codon-low-count |
| ↑ | A:AAT (Asn) | 2.252 | 1.27e-04 |  |
| ↑ | P:TTT (Phe) | 2.229 | 0.01350 | rare-codon-low-count |
| ↑ | A:TTT (Phe) | 2.047 | 0.04879 | rare-codon-low-count |
| ↓ | A:TAC (Tyr) | 0.615 | 1.03e-06 |  |
| ↓ | A:TGG (Trp) | 0.616 | 0.00932 | rare-codon-low-count |
| ↓ | A:TGC (Cys) | 0.665 | 0.06456 | rare-codon-low-count, nominal-only |
| ↓ | E:AAG (Lys) | 0.718 | 2.57e-10 |  |
| ↓ | A:GCC (Ala) | 0.716 | 0.01823 |  |

### control (n_sig FDR<0.05 = 66 / 183)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | P:CGC (Arg) | 1.573 | 2.48e-04 |  |
| ↑ | E:CGC (Arg) | 1.551 | 7.42e-04 |  |
| ↑ | A:ATC (Ile) | 1.522 | 9.17e-12 |  |
| ↑ | A:AAC (Asn) | 1.506 | 7.48e-08 |  |
| ↑ | P:TCC (Ser) | 1.508 | 1.73e-03 |  |
| ↓ | P:TTA (Leu) | 0.240 | 0.02291 | rare-codon-low-count |
| ↓ | E:TTA (Leu) | 0.256 | 0.04974 | rare-codon-low-count |
| ↓ | A:TTA (Leu) | 0.304 | 0.00532 | rare-codon-low-count |
| ↓ | A:GCG (Ala) | 0.333 | 1.03e-04 | rare-codon-low-count |
| ↓ | A:GCA (Ala) | 0.398 | 1.59e-10 |  |

## Numbers at a glance
- `n_tests`: 366 (183 per condition; BH families of 61 codons per (condition, site))
- `n_significant` (adjusted-p < 0.05): 88 (BWM 22, control 66)
- `n_significant` (adjusted-p < 0.10): 110 (BWM 30, control 80)
- `min adjusted-p`: 2.20e-19 (control,E,AAG)
- Per (condition, site) at FDR<0.05:
  - BWM,A: 11; min p_adj = 1.03e-06 (TAC↓)
  - BWM,E: 7; min p_adj = 2.57e-10 (AAG↓)
  - BWM,P: 4; min p_adj = 1.74e-06 (AAG↓)
  - control,A: 22; min p_adj = 1.00e-12 (AAG↑)
  - control,E: 21; min p_adj = 2.20e-19 (AAG↑) ← family-wide minimum
  - control,P: 23; min p_adj = 3.40e-07 (AAA↓ / TAT↓ tied)

## Methods
Same as the rest of the family. **One zero-cell row**: BWM,E,ATA day_0=0 → OR=inf (raw p=0.103, p_adj=0.274 ns); reported but excluded from Top hits ranking. Top hits use |log2(OR)| ranking restricted to FDR<0.10 (110 candidates total).

## Caveats
### Confirmed
- **pseudorep** *(family-wide)*. (Inherited.)
- **large-N-Fisher-anticonservative** *(family-wide)*. control,E,AAG OR=1.41 → p_adj=2.20e-19 — the smallest p_adj in this family. Rank by log2(OR), not p magnitude. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)*. (Inherited.)
- **within-condition-clean** *(family-wide)*. (Inherited.)
- **rare-codon-low-count** *(per-CSV)* — pervasive in BWM's top-enriched. **All 5 BWM Top-enriched and 3 of 5 BWM Top-depleted** carry the flag; **all 5 control Top-depleted** are rare codons (TTA, GCG, plus GCA which is sparser at site A). The headline magnitudes for BWM enriched (3.96 / 2.85 / 2.42 / 2.28 / 2.25) are unstable; rank-direction is reliable but the precise OR is sensitive to small-count Fisher's exact behavior.
- **OR-direction-asymmetry** *(per-CSV)* — one zero-cell row (BWM,E,ATA, OR=inf). Excluded from Top hits.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng's reconciliation.
  - **A:AAG**: BWM↓0.84 (FDR=0.0093) vs control↑1.38 (FDR=1.00e-12) — direction flip on a high-count codon, both heavily significant.
  - **E:AAG**: BWM↓0.72 (FDR=2.57e-10) vs control↑1.41 (FDR=2.20e-19) — direction flip, both massively significant. **The strongest single divergence in the entire family.**
  - **P:AAG**: BWM↓0.73 (FDR=1.74e-06) vs control 0.98 (p_adj=0.819 ns) — BWM-only depletion.
  - **A:GAT**: BWM↑1.58 (FDR=1.63e-06) vs control 0.98 ns — BWM-only enrichment.
  - **A:AAT**: BWM↑2.25 (FDR=1.27e-04) vs control 0.92 ns — BWM-only enrichment, rare-codon-flagged.
- **larger-bh-family** *(per-CSV)* — BH-FDR per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Read codon vs aa p_adj on comparable axes. The family-wide minimum p_adj of 2.20e-19 (control,E,AAG) sits in a 61-codon sub-family — comparable evidence weight to an aa p_adj~2e-19 in a 20-row sub-family. A codon p_adj=0.05 here is materially harder to clear than an aa p_adj=0.05 in the aa companion, so cross-resolution comparisons should account for the BH family-size scaling.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: aa-level d5_vs_d0 (parent), codon-level d10_vs_d0 / d10_vs_d5.
- Open questions Chumeng should resolve:
  - **AAG cross-site / cross-contrast tracking question.** This file: BWM AAG depleted at A (0.84 sig), E (0.72 sig), P (0.73 sig); control AAG enriched at A (1.38 sig), E (1.41 sig), P null. d10_vs_d0_codon: same direction-flip at E (BWM↓0.80 vs control↑1.33, both sig). d10_vs_d5_codon: BWM E:AAG p_adj=0.114 ns. Does this numerical pattern across three contrasts support a single d0-anchored AAG signature, or could the two-vs-d0 cells share common-baseline structure independent of biology? Does it map to the aa-level E:K cell (`timepoint_fisher_within_condition_d5_vs_d0_aa`: BWM E:K↓0.75 / control E:K↑1.19, same direction)? Chumeng to weigh against alternative readings.
  - **Are AAG and AAA both contributing to the E:K aa-level flip, or is the signal AAG-specific?** This file: BWM E:AAG↓0.72 sig, BWM E:AAA↑1.01 ns. Control E:AAG↑1.41 sig, control E:AAA↓0.61 sig (depleted). So at site E, **AAG and AAA shift in opposite directions in control** (AAG↑ but AAA↓), while in BWM only AAG is significant. The aa-level E:K flip is a vector sum: control E:K↑1.19 reflects that AAG's enrichment outweighs AAA's depletion in count terms; BWM E:K↓0.75 reflects AAG's depletion alone.
  - **BWM's heavy reliance on rare codons for top-enriched signal** (E:ACG, A:TGT, P:TGT, P:ACA, A:AAT, P:TTT, A:TTT) is itself a finding: it suggests BWM at d5 is shifting stall-site composition toward rare codons. Cross-check with `timepoint_fisher_within_condition_d10_vs_d0_codon` (the longer-window version): BWM,A,TGT is OR=3.77 there too (p_adj=0.0148). Pattern persists from d0→d5→d10 in BWM. Could be a real biological signal of perturbation pushing translation toward rare codons, OR a small-count artifact reproducible across reps. The combination of pseudoreplication + small absolute counts + same direction across two contrasts is suggestive but not conclusive.
  - **Control E:P (aa) was reported at p_adj=2.27e-12 in the aa companion**; the corresponding codon-level Pro picture here: control E:CCA↓0.66 (FDR<3e-9), control E:CCT↓0.35 (rare, FDR<7e-3), control E:CCG↓0.50 (rare, FDR<2e-2). All four Pro codons are depleted at site E in control in this contrast. Coherent direction across synonyms is consistent with an amino-acid-level reading rather than a single-synonym codon-usage shift; Chumeng to weigh against the per-CSV `control-E-P-extreme-anticonservative` flag in the aa companion.
