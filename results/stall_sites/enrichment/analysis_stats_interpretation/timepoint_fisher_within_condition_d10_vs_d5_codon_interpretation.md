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
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "BWM Top hits include several small-count rows: A:CTG (12 vs 55), A:GCA (21 vs 72), P:TTT (25 vs 87), plus all 3 control,*,TTA depletion rows (counts 1-5). Magnitude unstable; rank-direction reliable. Note also BWM site E is fully null at FDR<0.05 (smallest p_adj=0.114) — the BH-family scaling combined with small per-codon counts wipes out per-synonym significance even where aa-level effects exist."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "No zero-cell rows in this file (all day_X_count > 0). Caveat preserved as ranking-axis discipline (use log2 axis)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Three codon-level direction flips: P:GTC (BWM 1.26 ns vs control↓0.74 sig), E:GAG (BWM 1.05 ns vs control↓0.82 sig), E:GCC (BWM 0.95 ns vs control↑1.53 sig). Common signal: P:GGA enriched in both (BWM 1.28 sig, control 1.43 sig)."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "BH-FDR is per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Stricter per-sub-family correction; codon p_adj=0.05 is materially harder to clear than aa p_adj=0.05. Textbook case visible in this file: BWM site E is fully null at codon level (smallest p_adj=0.114) while the aa companion shows BWM,E:G/R/I sig at FDR<0.05 — the larger BH family wipes out per-synonym significance even when aggregate-aa signal exists. Read codon vs aa p_adj on comparable axes."}
caveats_considered: []
headline: "d10 vs d5 within-condition Fisher (codon-level): 22/366 hits at FDR<0.05 (4/183 BWM, 18/183 control); BWM site E is fully null at codon level (smallest p_adj=0.114) despite BWM,E:G/R/I being sig at the aa level — larger-bh-family caveat case study. Largest-magnitude shared-direction cell in this CSV: site-P GGA enriched in both (BWM 1.28 / control 1.43, FDR<0.005 in both); within this CSV the aa-level Gly enrichment between d5→d10 concentrates on GGA, with other Gly synonyms ns. Largest-magnitude divergence cells: P:GTC (BWM 1.26 ns vs control↓0.74 sig), E:GAG (BWM 1.05 ns vs control↓0.82 sig), E:GCC (BWM 0.95 ns vs control↑1.53 sig)."
user_directives:
  - "(resume probe) order → 'Codon-first pairing (Recommended)' — d10_vs_d5_codon is the third file processed this session"
  - "(resume probe) layout → 'Same layout for all 5 (Recommended)'"
  - "(resume probe) caveat flow → 'If Dylan thinks the flags are appropriate then prompt me with them' — Dylan proposes per-CSV, user confirms each"
  - "(triage) test type → user confirmed Fisher's exact, BH-FDR per (condition, site) (locked from family, not re-asked)"
  - "(triage) per-CSV caveats → user confirmed all four Dylan-proposed: rare-codon-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction, larger-bh-family"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — Dylan re-read; test design unchanged from rest of family"
  - "(history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `larger-bh-family` caveat was added. Numbers and Top hits unchanged from prior run after spot-check."
---

# Interpretation — timepoint_fisher_within_condition_d10_vs_d5_codon

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d5_codon.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (resume probe) "Continuing the in-progress family — which order should I work the 5 remaining CSVs?" → "Codon-first pairing (Recommended)" — this file is third in the session.
- (resume probe) "Re-confirm the file-level layout choice for the remaining 5 files?" → "Same layout for all 5 (Recommended)"
- (resume probe) "Streamline caveat confirmation across the family?" → "If Dylan thinks the flags are appropriate then prompt me with them, I don't know myself if they are prevalent on a csv by csv basis"
- (triage) "Per-CSV caveats Dylan wants to flag for d10_vs_d5_codon. Which apply?" → user confirmed all four Dylan-proposed: `rare-codon-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`, `larger-bh-family`
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" → Dylan re-read; test design unchanged.
- (history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `larger-bh-family` caveat added. Numbers and Top hits unchanged from prior run after spot-check.

## Headline
d10 vs d5 within-condition Fisher (codon-level): 22/366 hits at FDR<0.05 (4/183 BWM, 18/183 control). BWM concentrates on **site-A and site-P GGA enrichment** (A:GGA↑1.30 FDR<5e-4, P:GGA↑1.28 FDR<2.4e-3) plus a rare-codon depletion (A:CTG↓0.37 FDR=0.028) and one rare-codon enrichment (P:TTT↑2.23 FDR=0.035). Control sweeps broadly across all three sites (A: 5 hits dominated by GGA/CAA/AAC/GCC/CCA; E: 7 hits including CCA/CTC/GCC/GAG/ACC↑ and GAG↓; P: 6 hits with GGA↑1.43 FDR<6e-9 the leading feature). **The site-A and site-P GGA enrichment is the largest-magnitude shared-direction cell in this CSV**: same direction in BWM and control across both sites and at large magnitude. Notable direction flip: **P:GTC** (BWM 1.26 ns vs control↓0.74 FDR=0.017).

## Top hits

### BWM (n_sig FDR<0.05 = 4 / 183)

Fewer than 10 candidates at FDR<0.10 (9 total). Top-5 each by raw p irrespective of cutoff.

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:GGA (Gly) | 1.303 | 4.68e-04 |  |
| ↑ | P:GGA (Gly) | 1.279 | 0.00239 |  |
| ↑ | A:CGT (Arg) | 1.291 | 0.05832 | nominal-only |
| ↑ | P:GAG (Glu) | 1.191 | 0.13886 | nominal-only |
| ↑ | A:GAG (Glu) | 1.175 | 0.07953 | nominal-only |
| ↓ | A:CTG (Leu) | 0.374 | 0.02759 | rare-codon-low-count |
| ↓ | P:TTT (Phe) | 0.492 | 0.03466 | rare-codon-low-count |
| ↓ | A:CAA (Gln) | 0.743 | 0.05832 | nominal-only |
| ↓ | A:GCA (Ala) | 0.500 | 0.05832 | rare-codon-low-count, nominal-only |
| ↓ | A:CTT (Leu) | 0.753 | 0.07300 | nominal-only |

### control (n_sig FDR<0.05 = 18 / 183)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | E:GCC (Ala) | 1.530 | 0.00111 |  |
| ↑ | P:GGA (Gly) | 1.430 | 5.71e-09 |  |
| ↑ | E:CTC (Leu) | 1.422 | 0.00111 |  |
| ↑ | A:GCC (Ala) | 1.389 | 0.01536 |  |
| ↑ | E:CCA (Pro) | 1.387 | 0.00111 |  |
| ↓ | A:CAA (Gln) | 0.552 | 1.02e-06 |  |
| ↓ | A:AAC (Asn) | 0.675 | 3.00e-04 |  |
| ↓ | P:CAA (Gln) | 0.675 | 0.01183 |  |
| ↓ | P:ATT (Ile) | 0.685 | 0.01283 |  |
| ↓ | A:AAT (Asn) | 0.712 | 0.08969 | nominal-only |

## Numbers at a glance
- `n_tests`: 366 (183 per condition; BH families of 61 codons per (condition, site))
- `n_significant` (adjusted-p < 0.05): 22 (BWM 4, control 18)
- `n_significant` (adjusted-p < 0.10): 29 (BWM 9, control 20)
- `min adjusted-p`: 5.71e-09 (control,P,GGA)
- Per (condition, site) at FDR<0.05:
  - BWM,A: 2 (GGA↑, CTG↓); min p_adj = 4.68e-04
  - BWM,E: 0 (smallest p_adj = 0.114); min p_adj = 0.114
  - BWM,P: 2 (GGA↑, TTT↓); min p_adj = 0.00239
  - control,A: 5; min p_adj = 1.02e-06 (CAA↓)
  - control,E: 7; min p_adj = 0.00111 (3-way tie at GCC/CTC/CCA)
  - control,P: 6; min p_adj = 5.71e-09 (GGA↑)

## Methods
Same as the rest of the family. **BWM site E is fully null at FDR<0.05** (smallest p_adj=0.114) — the d10_vs_d5 contrast has no detectable codon-level shift at site E in BWM, even though the aa-level d10_vs_d5 file showed E:G/E:R/E:I significant. This is consistent with codon-usage-shift features being washed out when aggregated to amino acids: small effects at multiple synonyms can sum to a sig aa-level effect even when no single synonym is sig.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — replicates summed; p anti-conservative. (Inherited.)
- **large-N-Fisher-anticonservative** *(family-wide)* — totals 6945/8788/11177/11935; control,P,GGA OR=1.43 → p_adj=5.7e-9. Rank by log2(OR). (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — 6 sub-families of 61, corrected independently. (Inherited.)
- **within-condition-clean** *(family-wide)*. (Inherited.)
- **rare-codon-low-count** *(per-CSV)* — 3 of 5 BWM Top-depleted rows have day_X_count < 60 (CTG: 12 vs 55; GCA: 21 vs 72; GTA in same file). Sister flag in BWM Top-enriched: P:TTT depleted at count 25 vs 87. Magnitude unstable.
- **OR-direction-asymmetry** *(per-CSV)* — no zero-cell rows. Ranking-axis discipline only.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — three direction flips: **P:GTC** (BWM 1.26 ns p=0.0104 → p_adj=0.159 vs control 0.74 FDR=0.017); **E:GAG** (BWM 1.05 ns vs control 0.82 FDR=0.0069); **E:GCC** (BWM 0.95 ns vs control 1.53 FDR=0.00111). Several control-only-significant cells where BWM is in the same direction but ns (E:CTC, E:CCA, P:CAA, P:ATT, A:CAA, A:AAC).
- **larger-bh-family** *(per-CSV)* — BH-FDR per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Read codon vs aa p_adj on comparable axes. **Textbook case in this file**: BWM site E is fully null at FDR<0.05 (smallest p_adj=0.114) while the aa companion shows BWM,E:G/R/I significant — the larger BH family wipes out per-synonym significance even when aa-level signal exists. Implication: a codon p_adj=0.114 here may carry the same evidence weight as an aa p_adj~0.04 in the aa companion.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: aa-level d10_vs_d5 (parent), plus the d10_vs_d0 / d5_vs_d0 codon contrasts.
- Open questions Chumeng should resolve:
  - **Is the strong site-A/P GGA enrichment shared across all 4 Gly codons?** This file: A:GGA↑1.30 in BWM (FDR<5e-4) is the only Gly codon at FDR<0.05 at site A; control A:GGA↑1.35 (FDR<2e-6) is also the only A-site Gly codon at FDR<0.05 (GGC, GGG, GGT all ns). Same at site P: BWM P:GGA↑1.28 (FDR<3e-3) is the only Gly codon to clear FDR; control P:GGA↑1.43 (FDR<6e-9) is the leading Gly codon (GGT, GGC, GGG ns). **Conclusion: aa-level Gly enrichment between d5→d10 is GGA-codon-driven, not amino-acid-level**. This is a codon-usage signature.
  - **Why is BWM site E completely null at codon level in this contrast** (smallest p_adj=0.114) when the aa-level file shows BWM,E,G/R/I sig at FDR<0.05? Because the aa effects are spread across multiple synonymous codons (G across GGA/GGC/GGG/GGT etc.) and no single codon clears the per-(condition,site) FDR alone. Suggests aa-level aggregation gains power at site E for d10_vs_d5 in BWM.
  - **BWM hit-count comparison across codon contrasts in this family** (for Chumeng to weigh against the between_timepoint_wilcoxon family). This file: BWM has 4 codon-level FDR<0.05 hits in d10_vs_d5. Does `_d5_vs_d0_codon_interpretation` show a comparable, larger, or smaller BWM hit count? If d10_vs_d5 carries more BWM signal than d5_vs_d0 here, does that align with or contradict the per-contrast pattern visible in the between_timepoint_wilcoxon family?
  - **Direction-flip P:GTC** (BWM 1.26 ns vs control↓0.74 sig) is novel to this contrast. Cross-check whether `between_condition_wilcoxon_codon` reports any near-significant signal on GTC at site P (codon-level GTC depletion in control over time would not appear in time-collapsed Wilcoxon, but feature-direction divergence might).
