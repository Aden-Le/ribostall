# Between-Condition Enrichment Analysis: Methodology & Interpretation

## 1. Workflow Overview

**Input:** `enrichment_results/between_condition_wilcoxon.csv` produced by the enrichment pipeline

For each ribosome site (E, P, A):

1. Amino acid frequencies at stall sites are computed per replicate/timepoint for each condition (BWM and Control)
2. Frequencies are compared across conditions using the **Wilcoxon rank-sum test** (Mann-Whitney U)
3. P-values are **FDR-corrected** using the Benjamini-Hochberg method (`p_adj`)
4. Log2 fold-change is computed from median frequencies
5. Results are visualized as **sorted bar plots** using `R_scripts/between_condition_enrichment.R`

**Conditions compared:**
- **BWM** (treatment) vs **Control** (baseline)

**Ribosome sites:**
- **E-site** (exit): position -1 relative to stall
- **P-site** (peptidyl): position 0 (the stall position)
- **A-site** (aminoacyl): position +1 relative to stall

---

## 2. The Wilcoxon Rank-Sum Test

### What It Is

The Wilcoxon rank-sum test (also called the Mann-Whitney U test) is a **non-parametric** statistical test that compares two independent groups. It tests whether the distribution of values in one group tends to be higher or lower than the other.

### Why Non-Parametric?

- Does **not assume normality** of the underlying distributions
- Robust to outliers and skewed data
- Appropriate for the relatively small sample sizes typical of replicate-level frequency data

### What It Tests Here

For each amino acid at each site, the test compares the set of stall-site frequencies across all BWM replicates/timepoints against all Control replicates/timepoints. A significant result means the frequency of that amino acid at stall sites differs systematically between BWM and Control.

### Multiple Testing Correction

Raw p-values are adjusted using the **Benjamini-Hochberg** (BH) method to control the **false discovery rate** (FDR). The adjusted p-values (`p_adj`) account for the fact that 20 amino acids are tested at each site.

---

## 3. Log2 Fold-Change Interpretation

**Formula:** `log2(median_BWM / median_control)`

- **Positive log2_FC:** The amino acid is more frequent at stall sites in BWM than in Control (enriched in treatment)
- **Negative log2_FC:** The amino acid is more frequent at stall sites in Control than in BWM (depleted in treatment)
- **Zero:** No difference in median frequency between conditions

### Scale Reference

| log2_FC | Fold-Change | Interpretation |
|---------|-------------|----------------|
| +1.0    | 2.0x        | Twice as frequent in BWM |
| +0.5    | ~1.4x       | ~40% more frequent in BWM |
| 0       | 1.0x        | No difference |
| -0.5    | ~0.7x       | ~30% less frequent in BWM |
| -1.0    | 0.5x        | Half as frequent in BWM |

---

## 4. How to Read the Bar Plots

### Axes

- **x-axis:** Amino acids (single-letter codes), sorted by log2 fold-change from highest (left) to lowest (right)
- **y-axis:** Log2 fold-change (BWM vs Control). The zero line represents no difference between conditions

### Bar Colors

- **Blue (#2E86AB):** Enriched — amino acid is more frequent at stall sites in BWM (log2_FC >= 0)
- **Red (#E84855):** Depleted — amino acid is more frequent at stall sites in Control (log2_FC < 0)

### Significance Stars

Stars above or below each bar indicate statistical significance after FDR correction:

| Symbol | Threshold | Meaning |
|--------|-----------|---------|
| `***`  | p_adj < 0.001 | Highly significant |
| `**`   | p_adj < 0.01  | Very significant |
| `*`    | p_adj < 0.05  | Significant |
| (none) | p_adj >= 0.05 | Not significant |

### What to Look For

- **Tall blue bars with stars:** Amino acids strongly and significantly enriched at stall sites under BWM treatment — candidates for condition-specific stalling
- **Tall red bars with stars:** Amino acids strongly and significantly depleted under BWM — stalling may be reduced by treatment
- **Short bars without stars:** Minimal or non-significant differences between conditions
- **Sorting pattern:** The left-to-right ordering reveals the overall enrichment landscape at each site

### Uniform Y-Axis

All three site plots (E, P, A) share the same y-axis range. This enables direct visual comparison of effect sizes across sites — a bar of the same height means the same magnitude of fold-change regardless of which site it appears in.

---

## 5. Composite Layout

The composite plot arranges the three sites side by side:

```
| E-site | P-site | A-site |
```

- **Left to right:** E (exit) → P (peptidyl) → A (aminoacyl), following the order of the ribosome tunnel
- **Shared legend:** A single legend at the bottom applies to all three panels
- **Shared y-axis scale:** Enables cross-site comparison of fold-change magnitudes

This layout allows you to quickly identify:
- Which amino acids show consistent enrichment/depletion across all three sites
- Site-specific effects (e.g., an amino acid enriched only at the P-site)
- The overall magnitude of between-condition differences at each site

---

## 6. Output Structure

```
between_condition_output/
  individual/
    site_E_barplot.png       # E-site bar plot
    site_P_barplot.png       # P-site bar plot
    site_A_barplot.png       # A-site bar plot
  composite/
    EPA_barplot_composite.png  # All three sites side by side
```

When `--format both` is used, both `.pdf` and `.png` versions are generated for each plot.
