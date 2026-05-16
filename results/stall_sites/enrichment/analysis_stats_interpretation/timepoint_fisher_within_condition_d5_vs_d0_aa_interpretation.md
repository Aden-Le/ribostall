---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_aa.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 120
n_significant_fdr05: 36
n_significant_fdr10: 47
min_p_adj: 2.27e-12
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Trp/Cys/Met sit below count=200 in BWM (W: 108 vs 89; C: 82 vs 34; M: 251 vs 122). BWM,A,W enters Top hits with day_0_count=89 — flagged. control,A,W (93 vs 382) and control,P,M (186 vs 607) also enter Top hits. Direction reliable; magnitude unstable for these rows."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "No zero-cell rows; caveat preserved as ranking-axis discipline (use log2(OR), not OR, when ordering effects)."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Three large-magnitude direction flips in this CSV. **E:K**: BWM↓0.75 (FDR<2e-9) vs control↑1.19 (FDR<8e-7); same direction-flip seen in d10_vs_d0_aa — Chumeng to weigh whether this reflects a d0-anchored divergence vs other patterns. **A:K** flip (BWM 0.89 ns vs control↑1.25 FDR<1e-7). **E:A** flip (BWM 1.02 ns vs control↓0.78 sig). Largest-magnitude shared-direction cells: A:N (both ↑, BWM 1.78/control 1.26 sig) and A:W (both ↓, BWM 0.62/control 0.60 sig — rare-aa flag)."}
  - {label: "control-E-P-extreme-anticonservative", proposed_by: dylan, status: confirmed, why: "control,E:P at p_adj=2.27e-12 is the smallest p_adj in the entire family for an OR of 0.647 — only ~35% relative depletion. The most extreme single-feature large-N anti-conservative case in the family. Per-CSV flag in addition to the family-wide large-N caveat to cap rhetoric on this row: rank by log2(OR) and interpret this row's p magnitude as direction-only; effect size carries the biological scale."}
caveats_considered: []
headline: "d5 vs d0 within-condition Fisher (aa-level): 36/120 hits at FDR<0.05 (15/60 BWM, 21/60 control). Largest-magnitude divergence cells: E:K (BWM↓0.75 FDR<2e-9 vs control↑1.19 FDR<8e-7), A:K (BWM 0.89 ns vs control↑1.25 FDR<1e-7), E:A (BWM 1.02 ns vs control↓0.78 sig). Largest-magnitude shared-direction cells: A:N both ↑ (BWM 1.78 / control 1.26, both sig) and A:W both ↓ (BWM 0.62 / control 0.60, both sig — rare-aa flag). control,E:P at p_adj=2.27e-12 (OR=0.647) is the file minimum — extreme large-N anti-conservative case flagged via control-E-P-extreme-anticonservative. Cross-contrast hook for Chumeng: E:K direction-flip magnitude is similar to the d10_vs_d0_aa cell (BWM 0.79 / control 1.13)."
user_directives:
  - "(resume probe) order → 'Codon-first pairing (Recommended)' — d5_vs_d0_aa is the fourth file processed this session"
  - "(resume probe) layout → 'Same layout for all 5 (Recommended)'"
  - "(resume probe) caveat flow → 'If Dylan thinks the flags are appropriate then prompt me with them' — Dylan proposes per-CSV, user confirms each"
  - "(triage) test type → user confirmed Fisher's exact, BH-FDR per (condition, site) (locked from family, not re-asked)"
  - "(triage) per-CSV caveats → user confirmed all four Dylan-proposed: rare-aa-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction, control-E-P-extreme-anticonservative"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — Dylan re-read; test design unchanged"
  - "(history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `control-E-P-extreme-anticonservative` caveat added. Numbers and Top hits unchanged from prior run after spot-check."
---

# Interpretation — timepoint_fisher_within_condition_d5_vs_d0_aa

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d5_vs_d0_aa.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (resume probe) "Continuing the in-progress family — which order should I work the 5 remaining CSVs?" → "Codon-first pairing (Recommended)" — this file is fourth in the session.
- (resume probe) "Re-confirm the file-level layout choice for the remaining 5 files?" → "Same layout for all 5 (Recommended)"
- (resume probe) "Streamline caveat confirmation across the family?" → "If Dylan thinks the flags are appropriate then prompt me with them, I don't know myself if they are prevalent on a csv by csv basis"
- (triage) "Per-CSV caveats Dylan wants to flag for d5_vs_d0_aa. Which apply?" → user confirmed all four Dylan-proposed: `rare-aa-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`, `control-E-P-extreme-anticonservative`
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" → Dylan re-read; test design unchanged.
- (history) Prior crashed run wrote this file via batch-propagation. This session re-triaged caveats per user's explicit per-CSV preference; new `control-E-P-extreme-anticonservative` caveat added. Numbers and Top hits unchanged from prior run after spot-check.

## Headline
d5 vs d0 within-condition Fisher (aa-level): **36/120 hits at FDR<0.05** (15/60 BWM, 21/60 control). **Largest-magnitude divergence cell**: E:K — BWM E:K↓0.75 FDR=1.95e-9 vs control E:K↑1.19 FDR=7.94e-7. Cross-contrast hook for Chumeng: the d10_vs_d0_aa cell (BWM 0.79 / control 1.13) carries similar direction and magnitude; d10_vs_d5 has E:K null in both conditions — Chumeng to weigh whether the numerical sequence supports a single d0-anchored pattern vs other readings. **Largest-magnitude shared-direction cells**: A:N enriched in both (BWM 1.78 vs control 1.26, both sig); A:W depleted in both (BWM 0.62 vs control 0.60, both sig — rare-aa flagged); A:Y depleted in both (0.73/0.76, both sig). **Other large-magnitude divergence cells**: **A:K** (BWM 0.89 ns vs control↑1.25 FDR<1e-7) and **E:A** (BWM 1.02 ns vs control↓0.78 FDR<6e-5).

## Top hits

### BWM (n_sig FDR<0.05 = 15 / 60)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:N (Asn) | 1.778 | 3.27e-08 |  |
| ↑ | E:Q (Gln) | 1.504 | 9.43e-06 |  |
| ↑ | E:S (Ser) | 1.426 | 5.45e-05 |  |
| ↑ | A:F (Phe) | 1.342 | 0.00341 |  |
| ↑ | A:D (Asp) | 1.301 | 2.29e-04 |  |
| ↓ | A:W (Trp) | 0.616 | 0.00357 | rare-aa-low-count |
| ↓ | A:Y (Tyr) | 0.726 | 2.82e-04 |  |
| ↓ | E:K (Lys) | 0.748 | 1.95e-09 |  |
| ↓ | E:R (Arg) | 0.759 | 1.09e-04 |  |
| ↓ | P:K (Lys) | 0.794 | 2.13e-04 |  |

### control (n_sig FDR<0.05 = 21 / 60)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:N (Asn) | 1.257 | 1.31e-04 |  |
| ↑ | A:I (Ile) | 1.247 | 8.54e-06 |  |
| ↑ | A:K (Lys) | 1.255 | 6.26e-08 |  |
| ↑ | P:V (Val) | 1.201 | 5.84e-04 |  |
| ↑ | E:K (Lys) | 1.194 | 7.94e-07 |  |
| ↓ | A:W (Trp) | 0.601 | 2.08e-05 | rare-aa-low-count |
| ↓ | E:P (Pro) | 0.647 | 2.27e-12 |  |
| ↓ | A:Y (Tyr) | 0.758 | 7.51e-06 |  |
| ↓ | P:M (Met) | 0.756 | 0.00570 | rare-aa-low-count |
| ↓ | A:L (Leu) | 0.783 | 4.96e-07 |  |

## Numbers at a glance
- `n_tests`: 120 (60 per condition)
- `n_significant` (adjusted-p < 0.05): 36 (BWM 15, control 21)
- `n_significant` (adjusted-p < 0.10): 47 (BWM 19, control 28)
- `min adjusted-p`: 2.27e-12 (control,E,P)
- Per (condition, site) at FDR<0.05:
  - BWM,A: 6 (N↑, D↑, Y↓, G↓, F↑, W↓); min p_adj = 3.27e-08
  - BWM,E: 5 (K↓, Q↑, S↑, R↓, G↓); min p_adj = 1.95e-09
  - BWM,P: 4 (K↓, G↓, R↓, Q↑); min p_adj = 2.13e-04
  - control,A: 9 (K↑, L↓, Y↓, I↑, W↓, P↓, N↑, A↓, E↑); min p_adj = 6.26e-08
  - control,E: 6 (P↓, K↑, A↓, E↑, T↓, R↑); min p_adj = 2.27e-12
  - control,P: 6 (V↑, P↓, M↓, K↓, H↓, E↑); min p_adj = 5.84e-04

## Methods
Same as the rest of the family. Top hits use |log2(OR)| ranking restricted to FDR<0.10 (47 candidates total — well above the 10-candidate threshold).

## Caveats
### Confirmed
- **pseudorep** *(family-wide)*. (Inherited.)
- **large-N-Fisher-anticonservative** *(family-wide)*. control,E,P OR=0.647 → p_adj=2.27e-12 — the smallest p_adj in the entire family. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)*. (Inherited.)
- **within-condition-clean** *(family-wide)*. (Inherited.)
- **rare-aa-low-count** *(per-CSV)* — BWM,A,W (78 vs 89), control,A,W (93 vs 382), control,P,M (186 vs 607) all enter Top hits. Direction is reliable; magnitude unstable for these rows.
- **OR-direction-asymmetry** *(per-CSV)* — no zero-cell rows. Ranking discipline only.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — flagged for Chumeng's reconciliation.
  - **E:K**: BWM↓0.75 (FDR<2e-9) vs control↑1.19 (FDR<8e-7) — direction flip, both massively significant. Reproduces d10_vs_d0 finding.
  - **A:K**: BWM 0.89 (p_adj=0.056, near-sig depleted) vs control↑1.25 (FDR<1e-7) — divergent direction, control side is decisively significant.
  - **E:A**: BWM 1.02 ns vs control↓0.78 FDR<6e-5 — control-only ↓.
  - Several control-only-significant cells (E:P↓, P:V↑, A:L↓, A:I↑, A:P↓) where BWM is in the same direction but ns.
- **control-E-P-extreme-anticonservative** *(per-CSV)* — control,E:P (Pro at site E, depleted) carries p_adj=2.27e-12 — the family-wide minimum p_adj — for an OR of 0.647 (~35% relative depletion only). The most extreme single-feature large-N anti-conservative case in the family. Cap rhetoric on this row: rank by log2(OR) and interpret this row's p magnitude as direction-only; effect size carries the biological scale. Cross-check against `between_condition_wilcoxon_aa` (E:P p_adj=0.27 ns there — the time-collapsed MW does not see this signal because it is a within-control time response, not a perturbation effect).

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: codon-level d5_vs_d0 (refinement), aa-level d10_vs_d0 / d10_vs_d5 (other contrasts).
- Open questions Chumeng should resolve:
  - **E:K cross-contrast tracking question.** The three contrasts give: d10_vs_d0_aa BWM 0.79 / control 1.13; d5_vs_d0_aa (this file) BWM 0.75 / control 1.19; d10_vs_d5_aa BWM 1.06 ns / control 0.95 ns. The numerical sequence is consistent with several patterns Chumeng should weigh — a d0-anchored divergence that persists at d10, a d0→d5 transition that plateaus, or two independent two-vs-d0 coincidences. Does `between_condition_wilcoxon_aa` show E:K? It reported p_adj=0.27 ns; if the d0-anchored reading is correct, time-pooling in MW would attenuate by averaging d0→d5 step against d5→d10 plateau. Chumeng to decide between readings.
  - **Cross-contrast A:N magnitudes** (provided for Chumeng's reconciliation, not as a synthesis claim): d10_vs_d0_aa BWM A:N=1.78, control A:N=1.26; d5_vs_d0_aa (this file) BWM A:N=1.78, control A:N=1.26; d10_vs_d5_aa BWM A:N=1.00 ns, control A:N=0.68 (opposite direction). Does this numerical pattern across three contrasts support a single BWM-anchored d0→d5 step (with d5→d10 plateau), or could the two-vs-d0 cells share common-baseline structure? Cross-check at codon level (sister CSV) — does AAC, AAT, or both carry the BWM signal?
  - **Site E in control has the smallest p_adj in the family** (E:P↓ at p_adj=2.27e-12). Cross-check whether `between_condition_wilcoxon_aa` E:P is anywhere near significant (collapsing time would mix d0 P-rich, d5 mid, d10 P-poor — the time-dependent depletion would attenuate). If between_condition shows E:P near-null, then control E:P depletion is a within-control time signature, not a perturbation effect.
  - **Cross-contrast A:K magnitudes** (for Chumeng's reconciliation): d10_vs_d0_aa BWM A:K=0.92 ns vs control A:K=1.14 sig (same direction, BWM ns); d5_vs_d0_aa (this file) BWM A:K=0.89 (p_adj=0.056) vs control A:K=1.25 (FDR<1e-7). Does the control A:K↑ trend from d0→d5 reproduce as a within-control time response in `per_timepoint_fisher_aa`, or does the cell read as a BWM-vs-control divergence at one or more timepoints? If yes to former → control-aligned time feature; if yes to latter → perturbation overlay. Chumeng to weigh.
