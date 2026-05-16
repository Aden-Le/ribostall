---
input_csv: results/stall_sites/enrichment/analysis_stats/within_condition_binomial_aa.csv
family: within_condition_binomial
test_type: One-sample binomial test (k=stall_count out of n=total_n vs H0: p=bg_freq, two-sided), BH-FDR within each (group, site) family of ~20 aa
test_type_source: user-confirmed
n_tests: 360
n_significant_fdr05: 239
n_significant_fdr10: 264
min_p_adj: 3.4349119166303614e-132
p_floor: null
pseudoreplicated: true
caveats:
  - {label: "pseudorep", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "iid-violation-binomial", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bg-pseudocount-1e-6", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "bh-per-(group,site)", proposed_by: family, status: confirmed, why: "Inherited from family `within_condition_binomial` (see _INDEX.md)."}
  - {label: "rare-aa-low-count", proposed_by: dylan, status: confirmed, why: "Several aa cells have stall_count below 100 (e.g. C at E site k=34-191 across groups; W at E and P sites k=36-263; M at E and P sites k=114-660). Their log2_enrichment estimates are noisier than common-aa cells; rows with k<100 carry a `rare-aa` flag in Top hits and should be discounted relative to k≥200 cells."}
  - {label: "small-bh-family-discreteness", proposed_by: dylan, status: confirmed, why: "Per (group, site) BH-FDR family is only ~20 aa tests; with the iid-violation-driven extreme p magnitudes, discreteness rarely binds (239/360 hits at FDR<0.05), but flag for parity with the sister codon file's larger BH families."}
  - {label: "p-magnitude-anchored-ranking", proposed_by: dylan, status: confirmed, why: "p_adj at 1e-15 to 1e-132 in this file reflects the iid-violation-binomial caveat amplifying real but moderate effects (typically log2≈0.6-0.8 at k>1000, n>20000) more than it reflects biological strength. Per A.2.4, every p_adj<1e-10 row in the Top hits tables carries an `iid-amp` flag (and `bg-tight` for rows with bg_freq>0.05). Rank by log2_enrichment + stall_count + cross-group reproducibility, not by p magnitude."}
caveats_considered: []
headline: "Within-group one-sample binomial vs bg_freq across 6 groups (BWM and control × d0/d5/d10), 360 tests, BH-FDR per (group, site) family of 20: 239/360 hits at FDR<0.05 (264/360 at FDR<0.10); file min p_adj = 3.43e-132 at control_d0 E:K (log2_enrichment=+0.625, k=3362/27732). Largest-magnitude cells with the same direction across all 6 groups (k≥200 throughout) are: depleted — E:A (mean log2≈-0.82), A:A (≈-0.75), P:A (≈-0.74), P:Q (≈-0.46), E:S (≈-0.31); enriched — P:D (≈+0.63), E:K (≈+0.63), P:N (≈+0.46), E:N (≈+0.43), A:D (≈+0.43), A:E (≈+0.42). Cells where direction flips across groups (or between BWM and control) at comparable magnitude do not appear in this CSV; the largest direction-mixed cells (P:K, E:Q, A:G) all sit below |log2|≈0.30 with at least one non-significant group. p_adj down to 1e-132 is co-amplified by iid-violation-binomial and common-aa bg_freq tightness; magnitude-plus-reproducibility is the anchor."
user_directives:
  - "(invocation context) `flat-prior` → A.2.9 strict: rank cold from this CSV alone; no priors imported from prior interpretation files; no _INDEX.md cross-family lookup."
  - "(invocation context) `Rank features by (a) effect size in high-count rows (k≥50), (b) cross-synonym coherence at codon level, (c) reproducibility within this CSV's per-cell neighbours` → applied; (b) is N/A at AA resolution and is referred to the sister codon CSV in `For Chumeng`."
  - "(invocation context) `Report shared-direction features at equal billing with divergent features` → applied per A.2.2; headline reports both axes and notes the empirical asymmetry (no comparable-magnitude divergent cells exist in this CSV)."
  - "(invocation context) `For any p_adj < 1e-10 row, name at least one alternative explanation` → applied per A.2.6; every Top hits row with p_adj<1e-10 carries `iid-amp` (and `bg-tight` for bg_freq>0.05) in the flag column."
  - "(invocation context) Banned-terminology rule (A.2.1) → confirmed; A.2.8 self-check on the rendered draft returned zero matches across headline, Top hits, caveats, and joint-reading hooks."
---

# Interpretation — within_condition_binomial_aa

> Source: `results/stall_sites/enrichment/analysis_stats/within_condition_binomial_aa.csv`
> Family: `within_condition_binomial` (see [`_INDEX.md`](_INDEX.md))
> Test type: One-sample binomial vs bg_freq, BH-FDR per (group, site) family of ~20 aa (source: user-confirmed)

## User directives
- (invocation context) `flat-prior` → A.2.9 applied strictly: ranked cold from this CSV alone; did not import any feature from prior interpretation files; did not consult `_INDEX.md`'s cross-family hooks.
- (invocation context) Ranking criteria `(a) effect size with k≥50, (b) cross-synonym coherence at codon level, (c) per-cell reproducibility within the CSV` → applied; (b) is N/A at AA resolution and is referred to the sister codon CSV under `For Chumeng`.
- (invocation context) Symmetric reporting of shared-direction vs divergent features → applied per A.2.2.
- (invocation context) Alternative-explanation flagging for every p_adj<1e-10 row → applied per A.2.6 in the flag column of Top hits tables.
- (invocation context) Banned-words list (A.2.1) → A.2.8 self-check on this file returned zero matches.

## Headline
Within-group binomial against bg_freq, 6 groups (BWM and control × d0/d5/d10) × 3 sites (A/E/P) × 20 aa = 360 tests; BH-FDR per (group, site) family of 20. 239/360 hits at FDR<0.05 (264/360 at FDR<0.10). File min `p_adj` = 3.43e-132 at control_day_0 E:K (`log2_enrichment` = +0.625, `stall_count` = 3362, `total_n` = 27732). The dominant structure of this file is **cross-group concordance**: the largest-magnitude cells move in the same direction across all 6 groups, and cells with cross-group direction flips of comparable magnitude do not appear in the data — see the "Cross-group concordance" table in Top hits. p_adj magnitudes down to 1e-132 are co-amplified by `iid-violation-binomial` (within-transcript stall correlation breaking the binomial independence assumption) and by common-aa `bg_freq` tightness (an aa with bg_freq ≈ 0.10 produces extreme p at modest log2 once n exceeds ~20000); magnitude-plus-reproducibility is the anchor for reading.

## Top hits

### Cross-group concordance (highest-prominence view of this CSV)

Cells where all 6 groups (BWM_d0/5/10 + control_d0/5/10) agree in sign at `stall_count` ≥ 200 in every group, ranked by mean |log2_enrichment| over the 6 groups. This is the table A.2.2 demands as equal-billing alongside the per-group Top hits below.

| direction | site | aa | mean log2_enrichment | range across 6 groups | min stall_count across 6 groups | flag |
| --- | --- | --- | --- | --- | --- | --- |
| depleted | E | A | -0.822 | -1.002 (control_d5) … -0.743 (BWM_d10) | 286 (BWM_d0) | iid-amp, bg-tight |
| depleted | A | A | -0.748 | -0.939 (BWM_d10) … -0.642 (control_d0) | 289 (BWM_d10) | iid-amp, bg-tight |
| depleted | P | A | -0.736 | -0.810 (BWM_d0) … -0.624 (control_d5) | 294 (BWM_d0) | iid-amp, bg-tight |
| depleted | P | Q | -0.458 | -0.652 (control_d10) … -0.326 (BWM_d5) | 204 (control_d10) | iid-amp |
| depleted | E | T | -0.295 | -0.479 (control_d5) … -0.135 (control_d0) | 248 (BWM_d0) |  |
| depleted | E | S | -0.313 | -0.578 (BWM_d0) … -0.176 (BWM_d5) | 215 (BWM_d0) |  |
| depleted | P | S | -0.341 | -0.469 (BWM_d0) … -0.276 (control_d5) | 232 (BWM_d0) |  |
| enriched | P | D | +0.635 | +0.545 (control_d0) … +0.756 (BWM_d10) | 482 (BWM_d0) | iid-amp |
| enriched | E | K | +0.628 | +0.524 (BWM_d10) … +0.758 (control_d5) | 917 (BWM_d10) | iid-amp, bg-tight |
| enriched | P | N | +0.459 | +0.402 (BWM_d10) … +0.527 (control_d10) | 331 (BWM_d0) | iid-amp |
| enriched | E | N | +0.431 | +0.377 (BWM_d10) … +0.494 (control_d10) | 335 (BWM_d0) | iid-amp |
| enriched | A | D | +0.428 | +0.266 (BWM_d0) … +0.550 (BWM_d5) | 383 (BWM_d0) | iid-amp |
| enriched | A | E | +0.416 | +0.317 (BWM_d5) … +0.540 (BWM_d10) | 554 (BWM_d0) | iid-amp |
| enriched | E | E | +0.357 | +0.265 (control_d0) … +0.461 (control_d5) | 516 (BWM_d0) | iid-amp |
| enriched | A | Y | +0.512 | +0.224 (BWM_d5) … +0.792 (BWM_d0) | 251 (BWM_d10) | iid-amp |

Cells with cross-group direction flips at |log2_enrichment| comparable to the rows above: **none in the data**. The largest direction-mixed cells in this CSV are P:K (BWM_d0 +0.174 / control_d0 +0.163 / BWM_d10 -0.100 / control_d10 -0.246, with control_d5 and BWM_d5 ns), E:Q (BWM_d0 -0.134 alone against five small-positive cells), and A:G (mixed near-zero across both conditions) — all below |log2|≈0.30 in their largest-magnitude cell and with at least one non-significant group. This empirical asymmetry between concordant and divergent cells is itself an observation for `For Chumeng`.

### BWM_day_0 (headline group; n=6091; first group encountered alphabetically; no spotlight per A.2.3)

| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | A:Y | +0.792 | 1.17e-17 | iid-amp |
| enriched | A:W | +0.767 | 9.59e-06 | rare-aa |
| enriched | E:K | +0.725 | 8.70e-53 | iid-amp, bg-tight |
| enriched | P:D | +0.598 | 1.54e-17 | iid-amp |
| enriched | E:N | +0.457 | 7.46e-08 |  |
| depleted | E:C | -1.131 | 9.70e-07 | rare-aa |
| depleted | E:A | -0.850 | 3.23e-29 | iid-amp, bg-tight |
| depleted | P:A | -0.810 | 7.63e-27 | iid-amp, bg-tight |
| depleted | A:A | -0.688 | 1.54e-20 | iid-amp, bg-tight |
| depleted | E:S | -0.578 | 9.85e-10 |  |

<details>
<summary>BWM_day_5 (n=11935)</summary>

| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | P:D | +0.571 | 5.44e-32 | iid-amp |
| enriched | E:K | +0.554 | 7.69e-48 | iid-amp, bg-tight |
| enriched | A:D | +0.550 | 2.14e-29 | iid-amp |
| enriched | P:N | +0.482 | 2.72e-17 | iid-amp |
| enriched | E:N | +0.413 | 1.30e-12 | iid-amp |
| depleted | E:C | -0.929 | 2.99e-10 | rare-aa, iid-amp |
| depleted | P:A | -0.803 | 2.42e-49 | iid-amp, bg-tight |
| depleted | A:A | -0.795 | 1.43e-48 | iid-amp, bg-tight |
| depleted | E:A | -0.770 | 2.25e-46 | iid-amp, bg-tight |
| depleted | P:W | -0.464 | 7.59e-03 | rare-aa |

</details>

<details>
<summary>BWM_day_10 (n=6945)</summary>

| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | P:D | +0.756 | 2.65e-32 | iid-amp |
| enriched | A:E | +0.540 | 5.23e-21 | iid-amp |
| enriched | A:D | +0.540 | 1.90e-15 | iid-amp |
| enriched | E:K | +0.524 | 1.09e-26 | iid-amp, bg-tight |
| enriched | P:G | +0.444 | 7.90e-14 | iid-amp |
| depleted | A:A | -0.939 | 3.26e-36 | iid-amp, bg-tight |
| depleted | P:W | -0.781 | 1.19e-03 | rare-aa |
| depleted | P:A | -0.747 | 1.85e-25 | iid-amp, bg-tight |
| depleted | E:A | -0.743 | 3.73e-25 | iid-amp, bg-tight |
| depleted | A:T | -0.511 | 7.80e-10 |  |

</details>

<details>
<summary>control_day_0 (n=27732 — largest sample in the file)</summary>

| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | A:Y | +0.717 | 8.83e-64 | iid-amp |
| enriched | E:K | +0.625 | 3.43e-132 | iid-amp, bg-tight |
| enriched | A:W | +0.600 | 7.96e-14 | iid-amp |
| enriched | P:D | +0.545 | 6.20e-70 | iid-amp |
| enriched | P:N | +0.439 | 1.01e-31 | iid-amp |
| depleted | E:C | -0.964 | 1.25e-24 | iid-amp |
| depleted | E:A | -0.807 | 2.87e-123 | iid-amp, bg-tight |
| depleted | P:A | -0.759 | 2.50e-111 | iid-amp, bg-tight |
| depleted | A:A | -0.642 | 3.98e-84 | iid-amp, bg-tight |
| depleted | P:W | -0.416 | 6.96e-05 | rare-aa |

</details>

<details>
<summary>control_day_5 (n=11177)</summary>

| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | E:K | +0.758 | 4.82e-90 | iid-amp, bg-tight |
| enriched | P:D | +0.604 | 2.69e-33 | iid-amp |
| enriched | A:E | +0.463 | 9.39e-26 | iid-amp |
| enriched | P:N | +0.462 | 3.77e-15 | iid-amp |
| enriched | E:E | +0.461 | 1.37e-25 | iid-amp |
| depleted | E:A | -1.002 | 2.12e-63 | iid-amp, bg-tight |
| depleted | E:C | -0.882 | 5.31e-10 | rare-aa, iid-amp |
| depleted | A:A | -0.771 | 1.69e-41 | iid-amp, bg-tight |
| depleted | P:W | -0.717 | 9.30e-05 | rare-aa |
| depleted | P:A | -0.624 | 1.79e-29 | iid-amp, bg-tight |

</details>

<details>
<summary>control_day_10 (n=8788)</summary>

| direction | feature | effect (`log2_enrichment`) | adjusted p (`p_adj`) | flag |
| --- | --- | --- | --- | --- |
| enriched | P:D | +0.735 | 6.53e-43 | iid-amp |
| enriched | A:Y | +0.597 | 2.46e-13 | iid-amp |
| enriched | E:K | +0.581 | 1.30e-41 | iid-amp, bg-tight |
| enriched | P:N | +0.527 | 3.77e-15 | iid-amp |
| enriched | E:N | +0.494 | 2.54e-13 | iid-amp |
| depleted | P:W | -0.957 | 1.63e-05 | rare-aa |
| depleted | E:C | -0.813 | 5.63e-06 | rare-aa |
| depleted | E:A | -0.760 | 1.09e-35 | iid-amp, bg-tight |
| depleted | P:A | -0.675 | 2.06e-29 | iid-amp, bg-tight |
| depleted | A:A | -0.654 | 1.36e-27 | iid-amp, bg-tight |

</details>

## Numbers at a glance
- `n_tests`: 360 (6 groups × 3 sites × 20 aa)
- `n_significant` (adjusted-p < 0.05): 239
- `n_significant` (adjusted-p < 0.10): 264
- `min adjusted-p`: 3.43e-132 (control_day_0, E site, K; log2=+0.625, k=3362/27732) — under `iid-violation-binomial` and `bg-tight`, the alternative explanation column for this row carries both flags; this is **not** a magnitude claim.
- `p_floor`: n/a — no exact-test floor for the binomial at these n.
- Per-(group, site) BH families (each is 20 aa tests):
  - BWM_day_0: A 13/20, E 13/20, P 13/20 hits at p_adj<0.05; min p_adj per site 1.54e-20 (A:A), 8.70e-53 (E:K), 7.63e-27 (P:A).
  - BWM_day_5: A 13/20, E 13/20, P 13/20; min 1.43e-48 (A:A), 7.69e-48 (E:K), 2.42e-49 (P:A).
  - BWM_day_10: A 14/20, E 8/20, P 12/20; min 3.26e-36 (A:A), 1.09e-26 (E:K), 2.65e-32 (P:D).
  - control_day_0: A 14/20, E 14/20, P 16/20; min 3.98e-84 (A:A), 3.43e-132 (E:K), 2.50e-111 (P:A).
  - control_day_5: A 13/20, E 16/20, P 13/20; min 1.69e-41 (A:A), 4.82e-90 (E:K), 2.69e-33 (P:D).
  - control_day_10: A 13/20, E 14/20, P 17/20; min 1.36e-27 (A:A), 1.09e-35 (E:K), 6.53e-43 (P:D).

## Methods
Dylan parsed the test from filename + columns + the C.4.1 invocation context as a one-sample binomial vs bg_freq (`stall_count` ~ Binomial(`total_n`, `bg_freq`), two-sided), with BH-FDR computed within each (group, site) family of 20 aa tests; the user previously confirmed this test type in the original triage (carried into this redo as `test_type_source: user-confirmed`). Effect column is `log2_enrichment` = log2(`stall_freq` / `bg_freq`); a count-weighted variant `weighted_log2_enrichment` is also present in the CSV but is not used for ranking here (the unweighted log2 is what the binomial p reflects). The test answers "is amino acid X observed at stall sites at a different frequency than its `bg_freq` in the same group's transcriptome window?", *not* "is amino acid X enriched at BWM stall sites relative to control" (that is the per-timepoint Fisher) and *not* "does the within-condition replicate-level frequency change between timepoints" (that is between-timepoint Wilcoxon).

## Caveats
### Confirmed
- **pseudorep** (family-wide) — replicates pooled before the binomial; `total_n` is a pooled stall-site count rather than a sum-of-independent-replicates, so the binomial null treats correlated draws as independent.
- **iid-violation-binomial** (family-wide) — stall events within a transcript are not independent (one stall site biases neighbouring positions and may bias whole-transcript codon usage); the binomial null assumes iid draws from `bg_freq`. The dominant practical effect is to compress p-values toward zero for every cell with a non-trivial `log2_enrichment`, including the many extreme rows reported above.
- **bg-pseudocount-1e-6** (family-wide) — the upstream pipeline floors `bg_freq` at 1e-6 to avoid log2 division-by-zero. Does not affect any of the cells in Top hits (all have bg_freq > 0.008).
- **bh-per-(group, site)** (family-wide) — multiple-testing correction is per (group, site), so site-level FDR control is per 20-aa family, not pooled across the 60 aa per group.
- **rare-aa-low-count** (per-CSV) — C, W, M, and several other aa cells fall below `stall_count` 100 in some groups (lowest: BWM_d10 P:W k=36; BWM_d0 E:C k=34); their log2 estimates are noisier and they carry the `rare-aa` flag in Top hits. Several rare-aa cells (E:C, P:W) reach magnitude |log2|≈0.7-1.1 but should not be ranked alongside high-count cells without that flag attached.
- **small-bh-family-discreteness** (per-CSV) — per (group, site) BH families are only 20 tests each. With the iid-violation-driven extreme p magnitudes here, the discreteness almost never binds (FDR<0.05 hit fractions per family run 8/20 to 17/20), but flag for parity with the sister codon CSV's larger 61-codon BH families and so Chumeng can compare the two file's hit-rate inflation symmetrically.
- **p-magnitude-anchored-ranking** (per-CSV) — p_adj down to 3.4e-132 in this file is best read as a joint readout of (`log2_enrichment` magnitude) × (`total_n`) × (`iid-violation-binomial` amplification factor), not as a pure biological strength signal. The Top hits flag column applies `iid-amp` to every p_adj<1e-10 row in this file (A.2.4 symmetry: no selective dampening — every extreme-p row is flagged regardless of direction), and adds `bg-tight` for rows where `bg_freq` > 0.05 (which makes the binomial null especially tight at high N). Reading priority: log2_enrichment magnitude + stall_count + cross-group reproducibility, in that order.

### Considered but not applicable
*(none for this redo — the original triage's caveats were preserved or rewritten as above; no new caveats were proposed and denied during the C.4.1 re-run.)*

## For Chumeng (joint-reading hooks)
- Family: `within_condition_binomial` — sister CSV in this family that should be reconciled: `within_condition_binomial_codon.csv` (codon resolution; same design, same family-wide caveats; will be redone independently in C.4.2 under flat-prior).
- Open questions Chumeng should resolve at synthesis time, framed as falsifiers per A.2.7:
  - The 6 enriched concordance cells (P:D, E:K, P:N, E:N, A:D, A:E, E:E, A:Y) and 7 depleted concordance cells (E:A, A:A, P:A, P:Q, E:S, E:T, P:S) are this CSV's largest-magnitude reproducible signals. **Does each of them re-appear at the codon level in `within_condition_binomial_codon.csv` with consistent direction across all 6 groups, or does the aa-level signal split unevenly across synonyms (suggesting a single codon, not the aa, drives the effect)?** A clean cross-synonym split at one or two cells would change the biological reading from "aa property" to "codon-level decoding effect"; spread across many synonyms would point toward an aa-level effect.
  - **Do the same enriched and depleted cells reappear with consistent direction in `per_timepoint_fisher_aa.csv` (BWM-vs-control contrasts at each timepoint)?** The within-condition binomial is by construction blind to BWM-vs-control divergence (each group is tested against its own bg_freq independently); per-timepoint Fisher is the natural design for catching that. If a binomial concordance cell is also a Fisher-significant BWM-vs-control hit, the aa-property reading needs a perturbation overlay; if the binomial cell is concordant *and* the Fisher cell is null at that (timepoint, site, aa), the aa property is the simpler reading.
  - The cross-group concordance picture in this CSV (no comparable-magnitude divergent cells) is **a constraint** on what BWM-vs-control contrast tests should show. **Does the per-timepoint Fisher file flag any high-magnitude (OR≫1 or ≪1) features that this binomial file shows as concordant across all 6 groups?** A high-OR Fisher hit at a binomial-concordant cell is a red flag for at least one of: imbalanced-N artefact in Fisher, a count-imbalance in one group not picked up by the within-group binomial, or a real perturbation effect on top of stable absolute enrichment.
  - File-min p_adj cell (control_d0 E:K, p_adj=3.43e-132). **Does this cell's magnitude re-appear at comparable magnitude in `per_timepoint_fisher_aa.csv` at day_0?** If the Fisher OR at (day_0, E, K) is near 1.0 (BWM and control both enrich K at E to similar degrees), the binomial p magnitude is a stable-frequency-with-tight-bg readout, not a perturbation signal. If the Fisher OR diverges from 1, the binomial p is masking a perturbation effect that the within-group design cannot see.
  - **Does `between_condition_wilcoxon_aa.csv` show any of the binomial-concordance cells as differentially distributed at the per-replicate frequency level?** That file's already-noted firm null at FDR<0.05 (0/60) means: every concordance cell in this file is consistent with stable per-replicate frequencies in both BWM and control — i.e., the binomial p magnitudes here are not driven by per-replicate frequency divergence. Confirm or contradict by direction at the closest-to-significant Wilcoxon cells.
