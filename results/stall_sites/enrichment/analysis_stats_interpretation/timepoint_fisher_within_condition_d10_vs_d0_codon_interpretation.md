---
input_csv: results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_codon.csv
family: timepoint_fisher_within_condition
test_type: Fisher's exact (two-sided), BH-FDR per (condition, site)
test_type_source: user-confirmed
n_tests: 366
n_significant_fdr05: 91
n_significant_fdr10: 103
min_p_adj: 1.402e-12
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "large-N-Fisher-anticonservative", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "bh-per-(condition,site)", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "within-condition-clean", proposed_by: family, status: confirmed, why: "Inherited from family `timepoint_fisher_within_condition` (see _INDEX.md)."}
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "Top-magnitude rows have day_X_count < 50 and are unstable: BWM,A,TGT (day_0=7, OR=3.77), BWM,P,TGT (day_0=11, OR=2.80), BWM,A,GCG (day_10=3, OR=0.26), BWM,P,GGG (day_0=1, OR=7.02 — extreme rank anchored on n=1), control,P,TTA (day_10=1, OR=0.10). Rank by OR but flag these rows so magnitude claims do not lean on n<50 cells."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "Two boundary rows: BWM,E,ATA (day_0=0 → OR=inf, p_adj=0.543, ns) and BWM,P,TTA (day_10=0 → OR=0, p_adj=0.978, ns). Both ns and excluded from Top hits ranking. Use log2(OR) for ordering finite rows; flag inf/0 cells explicitly."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Codon resolution sharpens the aa-level A:N, E:K, P:G divergences onto specific synonyms. Site-A,AAT: BWM↑1.94 sig vs control↓0.65 sig (opposite directions, both significant). Site-E,AAG: BWM↓0.80 sig vs control↑1.33 sig (opposite). Site-P,AAG: BWM↓0.69 (p_adj=7.5e-7) vs control flat 0.98. Site-P,GGA: BWM 1.06 ns vs control↑1.41 sig — the aa-level P:G divergence localizes to one Gly codon."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "BH-FDR is per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Stricter correction per sub-family: a codon p_adj=0.05 is materially harder to clear than an aa p_adj=0.05. Read codon vs aa p_adj on comparable axes."}
caveats_considered: []
headline: "Codon-level d10 vs d0 within-condition Fisher: 91/366 hits at FDR<0.05 (13/183 BWM, 78/183 control); min p_adj = 1.40e-12 at control,A,GGA. Largest-magnitude divergence cells: A,AAT (BWM↑1.94 vs control↓0.65, both sig opposite), E,AAG (BWM↓0.80 vs control↑1.33, both sig opposite), P,GGA (BWM 1.06 ns vs control↑1.41 sig). Largest-magnitude shared-direction cell: P,GAG both ↑ (BWM↑1.33 FDR<0.01 / control↑1.39 FDR<1e-7). Codon resolution also exposes synonymous-shift cells where aa-level signal concentrates on a subset of synonyms (Y→TAC only, K→AAG only at sites E and P, Q→CAA only at site E, E→GAG only at site P, control,A,G→GGA↑ but GGT↓ at FDR<0.05)."
user_directives:
  - "(resume probe) order → 'Codon-first pairing (Recommended)' — picking up the codon companion to the just-completed d10_vs_d0_aa first"
  - "(resume probe) layout → 'Same layout for all 5 (Recommended)' — both conditions in headline; Top hits split BWM vs control"
  - "(resume probe) caveat flow → 'If Dylan thinks the flags are appropriate then prompt me with them' — Dylan proposes per-CSV, user confirms each"
  - "(triage) test type → user confirmed Fisher's exact, BH-FDR per (condition, site) (locked from family, not re-asked)"
  - "(triage) per-CSV caveats → user confirmed all four Dylan-proposed: rare-codon-low-count, OR-direction-asymmetry, control-vs-BWM-divergent-direction, larger-bh-family"
  - "(invocation context) `read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging` — Dylan re-read; the test design (within-condition Fisher, BH per (condition, site)) is unchanged from the aa companion"
---

# Interpretation — timepoint_fisher_within_condition_d10_vs_d0_codon

> Source: `results/stall_sites/enrichment/analysis_stats/timepoint_fisher_within_condition_d10_vs_d0_codon.csv`
> Family: `timepoint_fisher_within_condition` (see [`_INDEX.md`](_INDEX.md))
> Test type: Fisher's exact (two-sided), BH-FDR per (condition, site) (source: user-confirmed)

## User directives
- (resume probe) "Continuing the in-progress family — which order should I work the 5 remaining CSVs?" → "Codon-first pairing (Recommended)"
- (resume probe) "Re-confirm the file-level layout choice for the remaining 5 files?" → "Same layout for all 5 (Recommended)"
- (resume probe) "Streamline caveat confirmation across the family?" → "If Dylan thinks the flags are appropriate then prompt me with them, I don't know myself if they are prevalent on a csv by csv basis"
- (triage) "Per-CSV caveats Dylan wants to flag for d10_vs_d0_codon. Which apply?" → user confirmed all four Dylan-proposed: `rare-codon-low-count`, `OR-direction-asymmetry`, `control-vs-BWM-divergent-direction`, `larger-bh-family`
- (invocation context) "read @shell_scripts/run_enrichment_stats.sh for further context before per csv triaging" → Dylan re-read; the test design is unchanged from the aa companion (within-condition Fisher, BH per (condition, site)).

## Headline
Strong codon-level signal at d10 vs d0 within-condition Fisher: 91/366 hits at FDR<0.05 (13/183 BWM, 78/183 control). Control's 78 codon hits dominate (6× the BWM count) and the smallest p_adj = 1.40e-12 sits at control,A,GGA (OR=1.43 — dominant statistical-design concern: the family-level large-N-Fisher-anticonservative caveat applies). **The aa-level direction divergences sharpen onto specific synonymous codons**: site-A AAT carries the A:N divergence (BWM↑1.94 sig vs control↓0.65 sig — both significant in opposite directions); site-E AAG carries the E:K divergence (BWM↓0.80 vs control↑1.33, both significant); site-P GGA alone carries the P:G divergence (BWM 1.06 ns vs control↑1.41 sig). **Codon resolution also exposes synonymous codon-usage shifts** where aa-level signal concentrates on a subset of synonyms: BWM,A:Y depletion is purely TAC (TAT flat); BWM,P:E enrichment is purely GAG (GAA flat); BWM,P:K depletion is purely AAG; BWM,E:Q enrichment is purely CAA; BWM,A:L depletion concentrates on CTC + CTG; control,A:G enrichment is GGA-only with GGT actually significantly *depleted* (mixed-direction codon-usage shift); control,A:R enrichment is split across CGC/CGT while CGA is depleted; control,P:E enrichment is GAG-only with GAA significantly *depleted*.

## Top hits

### BWM (n_sig FDR<0.05 = 13 / 183)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:TGT (Cys) | 3.771 | 0.01477 | rare-codon-low-count |
| ↑ | P:TGT (Cys) | 2.800 | 0.03509 | rare-codon-low-count |
| ↑ | A:AAT (Asn) | 1.945 | 0.01540 | rare-codon-low-count |
| ↑ | A:AAC (Asn) | 1.703 | 8.44e-04 |  |
| ↑ | E:CAA (Gln) | 1.583 | 0.00344 |  |
| ↓ | P:AAG (Lys) | 0.694 | 7.47e-07 |  |
| ↓ | A:CTG (Leu) | 0.404 | 0.05839 | nominal-only, rare-codon-low-count |
| ↓ | A:GCA (Ala) | 0.483 | 0.05839 | nominal-only, rare-codon-low-count |
| ↓ | A:GTG (Val) | 0.509 | 0.01540 |  |
| ↓ | A:TAC (Tyr) | 0.690 | 0.00270 |  |

Other significant BWM rows for completeness: A,GAT 1.36 (p_adj=0.0154); A,CTC 0.72 (p_adj=0.0397); E,AAG 0.80 (p_adj=0.00117); P,GAG 1.33 (p_adj=0.00744); P,GTC 1.38 (p_adj=0.0369). Two zero-cell rows (BWM,E,ATA OR=inf; BWM,P,TTA OR=0) are not significant and excluded from ranking per `OR-direction-asymmetry` discipline.

### control (n_sig FDR<0.05 = 78 / 183)

| direction | feature | effect (`odds_ratio`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| ↑ | A:CGC (Arg) | 1.814 | 1.82e-08 |  |
| ↑ | E:TCC (Ser) | 1.642 | 1.93e-05 |  |
| ↑ | E:CTC (Leu) | 1.641 | 1.64e-09 |  |
| ↑ | P:CGC (Arg) | 1.564 | 9.65e-04 |  |
| ↑ | P:TGC (Cys) | 1.498 | 0.00918 |  |
| ↓ | P:TTA (Leu) | 0.102 | 0.01126 | rare-codon-low-count |
| ↓ | A:TTA (Leu) | 0.276 | 0.00558 | rare-codon-low-count |
| ↓ | E:TAT (Tyr) | 0.384 | 3.96e-07 | rare-codon-low-count |
| ↓ | E:ACG (Thr) | 0.425 | 0.01007 | rare-codon-low-count |
| ↓ | A:TTT (Phe) | 0.468 | 1.98e-04 | rare-codon-low-count |

<details>
<summary>Additional notable control hits (synonymous-resolution view)</summary>

- **control,A,G enrichment splits between synonyms**: GGA OR=1.43 p_adj=1.40e-12 (file minimum) vs GGT OR=0.66 p_adj=0.0185 (significantly depleted). The aa-level "A:G enriched" headline obscures a clean codon-usage shift toward GGA at the expense of GGT.
- **control,A,R enrichment is synonym-heterogeneous**: CGC OR=1.81 p_adj=1.82e-08 and CGT OR=1.23 p_adj=0.0185 enriched; CGA OR=0.48 p_adj=0.0158 depleted; AGA / AGG / CGG flat. Net positive but mixed within the codon set.
- **control,P,E enrichment splits between synonyms**: GAG OR=1.39 p_adj=1.70e-08 (strongly enriched) vs GAA OR=0.83 p_adj=0.0342 (significantly *depleted*). The aa-level P:E↑ headline hides a clean GAG-vs-GAA opposite-direction split.
- **control,P top depletion landscape**: ATT 0.54 p_adj=2.16e-10, AAA 0.58 p_adj=2.04e-08, AAT 0.64 p_adj=1.09e-05, ATG 0.62 p_adj=1.09e-05, CAA 0.64 p_adj=3.73e-05, ACA 0.49 p_adj=1.24e-04 — broad small-residue / common-codon depletion at site P unique to control.

</details>

## Numbers at a glance
- `n_tests`: 366 (183 per condition; 61 codons × 3 sites × 2 conditions)
- `n_significant` (adjusted-p < 0.05): 91 (BWM 13, control 78)
- `n_significant` (adjusted-p < 0.10): 103 (BWM 16, control 87)
- `min adjusted-p`: 1.40e-12 (control, A, GGA)
- `p_floor`: n/a — Fisher with pooled N in the thousands has no meaningful floor; the dominant statistical-design concerns are `large-N-Fisher-anticonservative` (family-wide) and `larger-bh-family` (per-CSV) for codon scope.
- Per (condition, site) at FDR<0.05:
  - BWM,A: 7 sig (AAC↑, TAC↓, TGT↑, AAT↑, GAT↑, GTG↓, CTC↓); min p_adj = 8.44e-04
  - BWM,E: 2 sig (AAG↓, CAA↑); min p_adj = 0.00117
  - BWM,P: 4 sig (AAG↓, GAG↑, TGT↑, GTC↑); min p_adj = 7.47e-07
  - control,A: 29 sig; min p_adj = 1.40e-12 (GGA↑)
  - control,E: 29 sig; min p_adj = 1.30e-10 (AAG↑)
  - control,P: 20 sig; min p_adj = 1.65e-11 (GGA↑)

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2×2 of (codon_count, total − codon_count) at each timepoint within one condition; user confirmed (locked from the family). The script (`stall_sites_non_consensus_stats.py`) computes BH-FDR within each (condition, site) family of 61 codons at codon resolution (so each of the 6 sub-families is corrected independently, not against the full 366). Two zero-cell rows in BWM produce one infinite OR (E,ATA, day_0=0) and one zero OR (P,TTA, day_10=0); both are reported but excluded from magnitude ranking. The test does **not** answer whether the same codon shifts in BWM and control simultaneously — that is the user's job (see `control-vs-BWM-divergent-direction` caveat) and is the largest-magnitude reading from this file's cells. Counts (`day_X_count`) and totals (`day_X_total`) are summed across replicates before the test, hence the family-level pseudoreplication caveat.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — replicates summed before Fisher; p-values anti-conservative. (Inherited from `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* — pooled totals 6091, 6945, 8788, 27732. control,A,GGA at OR=1.43 hits p_adj=1.40e-12; this is the strongest signal in the file but only a 43% relative shift. `odds_ratio` (log-axis) is the dominant effect column; p magnitude alone over-states confidence. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — 6 sub-families of 61 codons each, corrected independently; `p_adj` means "corrected within this sub-family", not across the whole 366-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* — no condition or timepoint pooling; structurally cleaner than the Wilcoxon families. (Inherited.)
- **rare-codon-low-count** *(per-CSV)* — top-magnitude BWM rows include TGT,A (day_0=7), TGT,P (day_0=11), AAT,A (day_0=34), GCG,A (day_10=3), GGG,P (day_0=1, OR=7.02 — extreme rank anchored on n=1). Top control depletions include TTA at A and P (day_10=5, day_10=1). Rank by OR but flag these rows.
- **OR-direction-asymmetry** *(per-CSV)* — two zero-cell rows (BWM,E,ATA OR=inf; BWM,P,TTA OR=0); both are not significant and excluded from Top hits ranking. Use log2(OR) for ordering finite rows; flag inf/0 cells explicitly.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — site-A AAT (BWM↑1.94 vs control↓0.65, both FDR<0.05, opposite); site-E AAG (BWM↓0.80 vs control↑1.33, both FDR<0.05, opposite); site-P AAG (BWM↓0.69 at p_adj=7.5e-7 vs control flat 0.98); site-P GGA (BWM 1.06 ns vs control↑1.41 sig); site-P GAG (BWM↑1.33 sig and control↑1.39 sig — same direction at codon level, where the aa companion showed BWM stronger; the codon resolution shows control matches BWM here while AAG drives the K divergence). Flagged for Chumeng's reconciliation.
- **larger-bh-family** *(per-CSV)* — BH-FDR is per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. A codon p_adj=0.05 is materially harder to clear than an aa p_adj=0.05; read across resolutions on comparable axes.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: `timepoint_fisher_within_condition_d10_vs_d0_aa` (the aa-level parent of this codon refinement; done) and the d10_vs_d5 / d5_vs_d0 contrasts at the codon resolution.
- Open questions Chumeng should resolve at synthesis time:
  - **Synonymous codon-usage shift vs amino-acid-level shift.** Several aa-level signals at d10_vs_d0_aa concentrate on a single synonym at codon level (Y→TAC, P:E→GAG, P:K→AAG, E:K→AAG, E:Q→CAA, P:G→GGA in control). Where one synonym moves and the others are flat or move the opposite way (control,A,G: GGA↑ vs GGT↓; control,P,E: GAG↑ vs GAA↓; control,A,R: CGC/CGT↑ vs CGA↓), the signal is a codon-usage shift not an amino-acid shift. Chumeng should call these out as codon-bias signatures rather than amino-acid biology.
  - **The aa-level A:N divergence sharpens at codon level into opposite-direction codon preference**: BWM enriches both AAC (1.70 sig) and AAT (1.94 sig) at site A, but control specifically *depletes* AAT (0.65 sig) while leaving AAC flat (1.02 ns). The within-condition d10-vs-d0 trajectory at site A is therefore not just opposite in magnitude but opposite in codon preference between BWM and control.
  - **Site P AAG is the strongest BWM hit in the file (p_adj=7.5e-7, OR=0.69)** and is BWM-specific; control,P,AAG is essentially flat (0.98 ns). The aa companion buried this in P:K depletion alongside contributions from other site-P moves. The codon resolution promotes BWM,P,AAG depletion to a candidate first-order finding for the perturbation.
  - **Cross-resolution monotonicity check**: do the BWM site-A enrichments on AAC/AAT and depletions on TAC/CTC/CTG/GTG appear at the d5_vs_d0 codon contrast at intermediate magnitude? If yes → monotonic time response; if not → d10-only effect or biphasic. Same check for control's massive site-A signature (GGA, CAA, CGC, ATC, AAG).
  - **Per-cell observations on P-site GAG in this file** (for Chumeng to weigh against the between_timepoint_wilcoxon family): BWM,P,GAG OR=1.33 (FDR<0.01) and control,P,GAG OR=1.39 (FDR<1e-7); both conditions enrich GAG at site P from d0→d10 here. Does the same direction and magnitude reproduce in `per_timepoint_fisher_codon` at any timepoint, and does the within-condition binomial show GAG-at-P enriched as a stable baseline in every group? If yes across designs → Chumeng decides whether to elevate to a coordinated cross-family signal; if no → the within-condition Fisher and the timepoint-pooled MW reading cohere only on this design axis.
