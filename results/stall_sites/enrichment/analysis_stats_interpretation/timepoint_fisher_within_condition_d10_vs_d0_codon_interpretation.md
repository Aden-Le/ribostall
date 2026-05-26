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
  - {label: "rare-codon-low-count", proposed_by: dylan, status: confirmed, why: "Top-magnitude rows have day_X_count < 50 and are unstable: BWM,A,TGT (day_0=7, log2_OR=+1.915), BWM,P,TGT (day_0=11, log2_OR=+1.485), BWM,A,AAT (day_0=34, log2_OR=+0.960), control,P,TTA (day_10=1, log2_OR=-3.298), control,A,TTA (day_10=5, log2_OR=-1.855). Rank by log2_OR but flag these rows so magnitude claims do not lean on n<50 cells."}
  - {label: "OR-direction-asymmetry", proposed_by: dylan, status: confirmed, why: "Two boundary rows: BWM,E,ATA (day_0=0 → OR=inf, p_adj=0.543, ns) and BWM,P,TTA (day_10=0 → OR=0, p_adj=0.978, ns). Both ns and excluded from Top hits ranking. log2_OR is the reported effect column for finite rows; flag inf/0 cells explicitly."}
  - {label: "control-vs-BWM-divergent-direction", proposed_by: dylan, status: confirmed, why: "Codon resolution sharpens the aa-level A:N, E:K, P:G divergences onto specific synonyms. Site-A,AAT: BWM log2_OR=+0.960 sig vs control log2_OR=-0.616 sig (opposite directions, both significant). Site-E,AAG: BWM log2_OR=-0.324 sig vs control log2_OR=+0.406 sig (opposite). Site-P,AAG: BWM log2_OR=-0.527 (p_adj=7.5e-7) vs control flat log2_OR=-0.031. Site-P,GGA: BWM log2_OR=+0.008 ns vs control log2_OR=+0.493 sig — the aa-level P:G divergence localizes to one Gly codon."}
  - {label: "larger-bh-family", proposed_by: dylan, status: confirmed, why: "BH-FDR is per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. Stricter correction per sub-family: a codon p_adj=0.05 is materially harder to clear than an aa p_adj=0.05. Read codon vs aa p_adj on comparable axes."}
caveats_considered: []
headline: "Codon-level d10 vs d0 within-condition Fisher: 91/366 hits at FDR<0.05 (13/183 BWM, 78/183 control); min p_adj = 1.40e-12 at control,A,GGA (log2_OR=+0.520). Largest-magnitude divergence cells: A,AAT (BWM log2_OR=+0.960 vs control log2_OR=-0.616, both sig opposite), E,AAG (BWM log2_OR=-0.324 vs control log2_OR=+0.406, both sig opposite), P,GGA (BWM log2_OR=+0.008 ns vs control log2_OR=+0.493 sig). Largest-magnitude shared-direction cell: P,GAG both ↑ (BWM log2_OR=+0.410 FDR<0.01 / control log2_OR=+0.477 FDR<1e-7). Codon resolution also exposes synonymous-shift cells where aa-level signal concentrates on a subset of synonyms (Y→TAC only, K→AAG only at sites E and P, Q→CAA only at site E, E→GAG only at site P, control,A,G→GGA↑ but GGT↓ at FDR<0.05)."
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
Strong codon-level signal at d10 vs d0 within-condition Fisher: 91/366 hits at FDR<0.05 (13/183 BWM, 78/183 control). Control's 78 codon hits dominate (6× the BWM count) and the smallest p_adj = 1.40e-12 sits at control,A,GGA (`log2_OR`=+0.520 — dominant statistical-design concern: the family-level large-N-Fisher-anticonservative caveat applies). **The aa-level direction divergences sharpen onto specific synonymous codons**: site-A AAT carries the A:N divergence (BWM `log2_OR`=+0.960 sig vs control `log2_OR`=-0.616 sig — both significant in opposite directions); site-E AAG carries the E:K divergence (BWM `log2_OR`=-0.324 vs control `log2_OR`=+0.406, both significant); site-P GGA alone carries the P:G divergence (BWM `log2_OR`=+0.008 ns vs control `log2_OR`=+0.493 sig). **Codon resolution also exposes synonymous codon-usage shifts** where aa-level signal concentrates on a subset of synonyms: BWM,A:Y depletion is purely TAC (TAT flat); BWM,P:E enrichment is purely GAG (GAA flat); BWM,P:K depletion is purely AAG; BWM,E:Q enrichment is purely CAA; BWM,A:L depletion concentrates on CTC (CTG `p_adj`=0.058 ns under the strict `< 0.05` cutoff); control,A:G enrichment is GGA-only with GGT actually significantly *depleted* (mixed-direction codon-usage shift); control,A:R enrichment is split across CGC/CGT while CGA is depleted; control,P:E enrichment is GAG-only with GAA significantly *depleted*.

## Top hits

Selection rule (family-level, updated 2026-05-22): all rows with `p_adj` < 0.05, no row cap; rows grouped by `site` in A -> P -> E order then sorted by `p_adj` ascending with `|log2_OR|` descending as tiebreaker.

### BWM (n_sig FDR<0.05 = 13 / 183)

| direction | feature | aa | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| ↑ | A:AAC | N | +0.768 | 8.44e-04 |  |
| ↑ | A:TGT | C | +1.915 | 0.0148 | rare-codon-low-count |
| ↑ | A:AAT | N | +0.960 | 0.0154 | rare-codon-low-count |
| ↑ | A:GAT | D | +0.442 | 0.0154 |  |
| ↑ | P:GAG | E | +0.410 | 0.00744 |  |
| ↑ | P:TGT | C | +1.485 | 0.0351 | rare-codon-low-count |
| ↑ | P:GTC | V | +0.466 | 0.0369 |  |
| ↑ | E:CAA | Q | +0.663 | 0.00344 |  |
| ↓ | A:TAC | Y | -0.536 | 0.00270 |  |
| ↓ | A:GTG | V | -0.974 | 0.0154 |  |
| ↓ | A:CTC | L | -0.474 | 0.0397 |  |
| ↓ | P:AAG | K | -0.527 | 7.47e-07 |  |
| ↓ | E:AAG | K | -0.324 | 0.00117 |  |

Two zero-cell rows (BWM,E,ATA OR=inf; BWM,P,TTA OR=0) are not significant and excluded from ranking per `OR-direction-asymmetry` discipline.

### control (n_sig FDR<0.05 = 78 / 183)

| direction | feature | aa | effect (`log2_OR`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- | --- |
| ↑ | A:GGA | G | +0.520 | 1.40e-12 |  |
| ↑ | A:CGC | R | +0.859 | 1.82e-08 |  |
| ↑ | A:ATC | I | +0.516 | 3.47e-07 |  |
| ↑ | A:AAG | K | +0.353 | 1.07e-06 |  |
| ↑ | A:GCC | A | +0.399 | 0.00320 |  |
| ↑ | A:GAC | D | +0.299 | 0.00820 |  |
| ↑ | A:GTC | V | +0.302 | 0.0158 |  |
| ↑ | A:ACC | T | +0.334 | 0.0185 |  |
| ↑ | A:CGT | R | +0.302 | 0.0185 |  |
| ↑ | A:TTC | F | +0.260 | 0.0216 |  |
| ↑ | P:GGA | G | +0.493 | 1.65e-11 |  |
| ↑ | P:GAG | E | +0.477 | 1.70e-08 |  |
| ↑ | P:AAC | N | +0.412 | 4.09e-05 |  |
| ↑ | P:GAC | D | +0.388 | 4.75e-04 |  |
| ↑ | P:CGC | R | +0.645 | 9.65e-04 |  |
| ↑ | P:ACC | T | +0.381 | 0.00520 |  |
| ↑ | P:TGC | C | +0.583 | 0.00920 |  |
| ↑ | E:AAG | K | +0.406 | 1.30e-10 |  |
| ↑ | E:CTC | L | +0.715 | 1.64e-09 |  |
| ↑ | E:TCC | S | +0.716 | 1.93e-05 |  |
| ↑ | E:GTC | V | +0.427 | 6.12e-05 |  |
| ↑ | E:ATC | I | +0.293 | 0.00380 |  |
| ↑ | E:AAC | N | +0.267 | 0.00720 |  |
| ↑ | E:GGA | G | +0.229 | 0.00720 |  |
| ↑ | E:ACC | T | +0.337 | 0.00990 |  |
| ↑ | E:GCC | A | +0.355 | 0.0114 |  |
| ↑ | E:CGC | R | +0.478 | 0.0187 |  |
| ↑ | E:GAC | D | +0.244 | 0.0352 |  |
| ↓ | A:CAA | Q | -0.888 | 5.46e-10 |  |
| ↓ | A:TAT | Y | -0.955 | 1.07e-06 |  |
| ↓ | A:GCA | A | -1.065 | 1.45e-06 |  |
| ↓ | A:TTT | F | -1.094 | 1.98e-04 | rare-codon-low-count |
| ↓ | A:AAA | K | -0.637 | 2.94e-04 |  |
| ↓ | A:AAT | N | -0.616 | 9.93e-04 |  |
| ↓ | A:TTG | L | -0.474 | 0.00130 |  |
| ↓ | A:GAA | E | -0.360 | 0.00130 |  |
| ↓ | A:GTG | V | -0.632 | 0.00190 |  |
| ↓ | A:TCA | S | -0.829 | 0.00190 | rare-codon-low-count |
| ↓ | A:TGG | W | -0.604 | 0.00220 |  |
| ↓ | A:ACA | T | -0.692 | 0.00540 |  |
| ↓ | A:TTA | L | -1.855 | 0.00560 | rare-codon-low-count |
| ↓ | A:ACG | T | -0.954 | 0.0128 | rare-codon-low-count |
| ↓ | A:CGA | R | -1.067 | 0.0158 | rare-codon-low-count |
| ↓ | A:CTG | L | -0.734 | 0.0158 | rare-codon-low-count |
| ↓ | A:GGT | G | -0.610 | 0.0185 | rare-codon-low-count |
| ↓ | A:GCG | A | -0.958 | 0.0221 | rare-codon-low-count |
| ↓ | A:ATT | I | -0.308 | 0.0348 |  |
| ↓ | P:ATT | I | -0.895 | 2.16e-10 |  |
| ↓ | P:AAA | K | -0.798 | 2.04e-08 |  |
| ↓ | P:AAT | N | -0.648 | 1.09e-05 |  |
| ↓ | P:ATG | M | -0.681 | 1.10e-05 |  |
| ↓ | P:CAA | Q | -0.633 | 3.73e-05 |  |
| ↓ | P:ACA | T | -1.022 | 1.24e-04 | rare-codon-low-count |
| ↓ | P:TAT | Y | -0.683 | 4.25e-04 |  |
| ↓ | P:GCA | A | -0.747 | 0.00360 | rare-codon-low-count |
| ↓ | P:TTA | L | -3.298 | 0.0113 | rare-codon-low-count |
| ↓ | P:TGG | W | -0.699 | 0.0219 | rare-codon-low-count |
| ↓ | P:TTT | F | -0.624 | 0.0219 | rare-codon-low-count |
| ↓ | P:GAA | E | -0.277 | 0.0342 |  |
| ↓ | P:CTT | L | -0.296 | 0.0397 |  |
| ↓ | E:AAA | K | -0.736 | 7.83e-09 |  |
| ↓ | E:TAT | Y | -1.379 | 3.96e-07 | rare-codon-low-count |
| ↓ | E:GTG | V | -1.067 | 8.08e-05 | rare-codon-low-count |
| ↓ | E:ATT | I | -0.587 | 1.43e-04 |  |
| ↓ | E:TCA | S | -0.890 | 3.68e-04 | rare-codon-low-count |
| ↓ | E:GCA | A | -0.939 | 5.82e-04 | rare-codon-low-count |
| ↓ | E:GAT | D | -0.387 | 6.53e-04 |  |
| ↓ | E:AAT | N | -0.539 | 0.00170 |  |
| ↓ | E:GTT | V | -0.423 | 0.00210 |  |
| ↓ | E:CAT | H | -0.718 | 0.00670 | rare-codon-low-count |
| ↓ | E:ATG | M | -0.395 | 0.00720 |  |
| ↓ | E:ACG | T | -1.236 | 0.0101 | rare-codon-low-count |
| ↓ | E:CCG | P | -1.012 | 0.0127 | rare-codon-low-count |
| ↓ | E:ACA | T | -0.654 | 0.0140 | rare-codon-low-count |
| ↓ | E:TGG | W | -0.552 | 0.0207 |  |
| ↓ | E:TTG | L | -0.436 | 0.0226 |  |
| ↓ | E:TTT | F | -0.737 | 0.0288 | rare-codon-low-count |
| ↓ | E:GCG | A | -1.063 | 0.0298 | rare-codon-low-count |

<details>
<summary>Additional notable control hits (synonymous-resolution view)</summary>

- **control,A,G enrichment splits between synonyms**: GGA `log2_OR`=+0.520 p_adj=1.40e-12 (file minimum) vs GGT `log2_OR`=-0.610 p_adj=0.0185 (significantly depleted). The aa-level "A:G enriched" headline obscures a clean codon-usage shift toward GGA at the expense of GGT.
- **control,A,R enrichment is synonym-heterogeneous**: CGC `log2_OR`=+0.859 p_adj=1.82e-08 and CGT `log2_OR`=+0.302 p_adj=0.0185 enriched; CGA `log2_OR`=-1.067 p_adj=0.0158 depleted; AGA / AGG / CGG flat. Net positive but mixed within the codon set.
- **control,P,E enrichment splits between synonyms**: GAG `log2_OR`=+0.477 p_adj=1.70e-08 (strongly enriched) vs GAA `log2_OR`=-0.277 p_adj=0.0342 (significantly *depleted*). The aa-level P:E↑ headline hides a clean GAG-vs-GAA opposite-direction split.
- **control,P top depletion landscape**: ATT `log2_OR`=-0.895 p_adj=2.16e-10, AAA `log2_OR`=-0.798 p_adj=2.04e-08, AAT `log2_OR`=-0.648 p_adj=1.09e-05, ATG `log2_OR`=-0.681 p_adj=1.10e-05, CAA `log2_OR`=-0.633 p_adj=3.73e-05, ACA `log2_OR`=-1.022 p_adj=1.24e-04 — broad small-residue / common-codon depletion at site P unique to control.

</details>

## Numbers at a glance
- `n_tests`: 366 (183 per condition; 61 codons × 3 sites × 2 conditions)
- `n_significant` (adjusted-p < 0.05): 91 (BWM 13, control 78)
- `n_significant` (adjusted-p < 0.10): 103 (BWM 16, control 87)
- `min adjusted-p`: 1.40e-12 (control, A, GGA)
- `p_floor`: n/a — Fisher with pooled N in the thousands has no meaningful floor; the dominant statistical-design concerns are `large-N-Fisher-anticonservative` (family-wide) and `larger-bh-family` (per-CSV) for codon scope.
- Per (condition, site) at FDR<0.05:
  - BWM,A: 7 sig (AAC↑, AAT↑, GAT↑, TGT↑, CTC↓, GTG↓, TAC↓); min p_adj = 8.44e-04 (AAC↑)
  - BWM,P: 4 sig (GAG↑, GTC↑, TGT↑, AAG↓); min p_adj = 7.47e-07 (AAG↓)
  - BWM,E: 2 sig (CAA↑, AAG↓); min p_adj = 0.00117 (AAG↓)
  - control,A: 29 sig (10 enriched, 19 depleted); min p_adj = 1.40e-12 (GGA↑)
  - control,P: 20 sig (7 enriched, 13 depleted); min p_adj = 1.65e-11 (GGA↑)
  - control,E: 29 sig (11 enriched, 18 depleted); min p_adj = 1.30e-10 (AAG↑)

## Methods
Dylan proposed Fisher's exact (two-sided) on a 2×2 of (codon_count, total − codon_count) at each timepoint within one condition; user confirmed (locked from the family). The script (`stall_sites_non_consensus_stats.py`) computes BH-FDR within each (condition, site) family of 61 codons at codon resolution (so each of the 6 sub-families is corrected independently, not against the full 366). Two zero-cell rows in BWM produce one infinite OR (E,ATA, day_0=0) and one zero OR (P,TTA, day_10=0); both are reported but excluded from magnitude ranking. The test does **not** answer whether the same codon shifts in BWM and control simultaneously — that is the user's job (see `control-vs-BWM-divergent-direction` caveat) and is the largest-magnitude reading from this file's cells. Counts (`day_X_count`) and totals (`day_X_total`) are summed across replicates before the test, hence the family-level pseudoreplication caveat.

## Caveats
### Confirmed
- **pseudorep** *(family-wide)* — replicates summed before Fisher; p-values anti-conservative. (Inherited from `_INDEX.md`.)
- **large-N-Fisher-anticonservative** *(family-wide)* — pooled totals 6091, 6945, 8788, 27732. control,A,GGA at `log2_OR`=+0.520 hits p_adj=1.40e-12; this is the strongest signal in the file but only a 43% relative shift. `log2_OR` is the dominant effect column; p magnitude alone over-states confidence. (Inherited.)
- **bh-per-(condition,site)** *(family-wide)* — 6 sub-families of 61 codons each, corrected independently; `p_adj` means "corrected within this sub-family", not across the whole 366-row file. (Inherited.)
- **within-condition-clean** *(family-wide)* — no condition or timepoint pooling; structurally cleaner than the Wilcoxon families. (Inherited.)
- **rare-codon-low-count** *(per-CSV)* — top-magnitude BWM rows include TGT,A (day_0=7, `log2_OR`=+1.915), TGT,P (day_0=11, `log2_OR`=+1.485), AAT,A (day_0=34, `log2_OR`=+0.960). Top control depletions include TTA at A and P (day_10=5, `log2_OR`=-1.855 at A; day_10=1, `log2_OR`=-3.298 at P). Rank by `log2_OR` but flag these rows.
- **OR-direction-asymmetry** *(per-CSV)* — two zero-cell rows (BWM,E,ATA OR=inf; BWM,P,TTA OR=0); both are not significant and excluded from Top hits ranking. `log2_OR` is the reported effect column for finite rows; flag inf/0 cells explicitly.
- **control-vs-BWM-divergent-direction** *(per-CSV)* — site-A AAT (BWM `log2_OR`=+0.960 vs control `log2_OR`=-0.616, both FDR<0.05, opposite); site-E AAG (BWM `log2_OR`=-0.324 vs control `log2_OR`=+0.406, both FDR<0.05, opposite); site-P AAG (BWM `log2_OR`=-0.527 at p_adj=7.5e-7 vs control flat `log2_OR`=-0.031); site-P GGA (BWM `log2_OR`=+0.008 ns vs control `log2_OR`=+0.493 sig). Flagged for Chumeng's reconciliation. (Note: site-P GAG is co-enriched in both conditions — shared-direction, kept out of this divergent-direction caveat; see Top hits and Additional notable control hits for the GAG entry.)
- **larger-bh-family** *(per-CSV)* — BH-FDR is per (condition, site) family of 61 codons here vs 20 amino acids in the aa companion. A codon p_adj=0.05 is materially harder to clear than an aa p_adj=0.05; read across resolutions on comparable axes.

## For Chumeng (joint-reading hooks)
- Family: `timepoint_fisher_within_condition` — sister CSVs: `timepoint_fisher_within_condition_d10_vs_d0_aa` (the aa-level parent of this codon refinement; done) and the d10_vs_d5 / d5_vs_d0 contrasts at the codon resolution.
- Open questions Chumeng should resolve at synthesis time:
  - **Synonymous codon-usage shift vs amino-acid-level shift.** Several aa-level signals at d10_vs_d0_aa concentrate on a single synonym at codon level (Y→TAC, P:E→GAG, P:K→AAG, E:K→AAG, E:Q→CAA, P:G→GGA in control). Where one synonym moves and the others are flat or move the opposite way (control,A,G: GGA↑ vs GGT↓; control,P,E: GAG↑ vs GAA↓; control,A,R: CGC/CGT↑ vs CGA↓), the signal is a codon-usage shift not an amino-acid shift. Chumeng should call these out as codon-bias signatures rather than amino-acid biology.
  - **The aa-level A:N divergence sharpens at codon level into opposite-direction codon preference**: BWM enriches both AAC (`log2_OR`=+0.768 sig) and AAT (`log2_OR`=+0.960 sig) at site A, but control specifically *depletes* AAT (`log2_OR`=-0.616 sig) while leaving AAC flat (`log2_OR`~0 ns). The within-condition d10-vs-d0 trajectory at site A is therefore not just opposite in magnitude but opposite in codon preference between BWM and control.
  - **Site P AAG is the strongest BWM hit in the file (p_adj=7.5e-7, `log2_OR`=-0.527)** and is BWM-specific; control,P,AAG is essentially flat (`log2_OR`=-0.031 ns). The aa companion buried this in P:K depletion alongside contributions from other site-P moves. The codon resolution promotes BWM,P,AAG depletion to a candidate first-order finding for the perturbation.
  - **Cross-resolution monotonicity check**: do the BWM site-A enrichments on AAC/AAT and depletions on TAC/CTC/GTG appear at the d5_vs_d0 codon contrast at intermediate magnitude? If yes → monotonic time response; if not → d10-only effect or biphasic. Same check for control's massive site-A signature (GGA, CAA, CGC, ATC, AAG).
  - **Per-cell observations on P-site GAG in this file** (for Chumeng to weigh against the between_timepoint_wilcoxon family): BWM,P,GAG `log2_OR`=+0.410 (FDR<0.01) and control,P,GAG `log2_OR`=+0.477 (FDR<1e-7); both conditions enrich GAG at site P from d0→d10 here. Does the same direction and magnitude reproduce in `per_timepoint_fisher_codon` at any timepoint, and does the within-condition binomial show GAG-at-P enriched as a stable baseline in every group? If yes across designs → Chumeng decides whether to elevate to a coordinated cross-family signal; if no → the within-condition Fisher and the timepoint-pooled MW reading cohere only on this design axis.
