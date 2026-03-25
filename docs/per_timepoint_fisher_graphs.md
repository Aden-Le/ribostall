# Per-Timepoint Fisher Volcano Plots

## Workflow Overview

**Input:** `enrichment_results/per_timepoint_fisher.csv`

This visualization shows the results of Fisher's exact tests comparing amino acid frequencies at ribosome stall sites between BWM and control conditions, computed separately for each timepoint (Day 0, Day 5, Day 10) and ribosome site (E, P, A).

Each point on the volcano plot represents one amino acid tested at a specific timepoint and site combination.

## Understanding the Odds Ratio

The **odds ratio (OR)** quantifies how much more (or less) likely an amino acid is to appear at a stall site in BWM compared to control:

| Log2(OR)  | Odds Ratio | Interpretation                        |
|-----------|-----------|----------------------------------------|
| +2.0      | 4.0       | 4x more frequent in BWM stalls        |
| +1.0      | 2.0       | 2x more frequent in BWM stalls        |
| +0.5      | 1.41      | Moderately enriched in BWM            |
|  0.0      | 1.0       | No difference between BWM and control  |
| -0.5      | 0.71      | Moderately depleted in BWM            |
| -1.0      | 0.5       | 2x less frequent in BWM stalls        |
| -2.0      | 0.25      | 4x less frequent in BWM stalls        |

## How to Read the Volcano Plots

**Axes:**
- **X-axis:** Log2(Odds Ratio) — centered at 0 (no difference). Positive values indicate enrichment in BWM; negative values indicate depletion.
- **Y-axis:** -Log10(FDR-adjusted p-value) — higher values indicate stronger statistical significance.

**Reference lines:**
- **Vertical dashed lines** at Log2(OR) = -0.5 and +0.5 mark moderate effect size thresholds.
- **Horizontal dashed line** at -Log10(0.05) marks the FDR significance threshold.
- **Vertical dotted line** at 0 marks no enrichment/depletion.

**Points:**
- Colored by amino acid chemical class:
  - Red = Acidic (D, E)
  - Blue = Basic (K, R, H)
  - Green = Hydrophobic (A, V, I, L, M, F, W, Y)
  - Purple = Polar (C, N, Q, S, T)
  - Orange = Neutral (G, P)
- Triangle (▲) = Significant (FDR < 0.05)
- Circle (●) = Not significant
- Significant amino acids are labeled with their single-letter code.

**What to look for:**
- Points in the **upper-right** are significantly enriched in BWM stall sites.
- Points in the **upper-left** are significantly depleted in BWM stall sites.
- Points near the bottom or center show weak or non-significant differences.
- Extreme -Log10(p) values are capped at 50 to prevent axis distortion.

## Composite Layout

The 3×3 composite grid organizes all combinations:

|           | E-site | P-site | A-site |
|-----------|--------|--------|--------|
| **Day 0** | E, D0  | P, D0  | A, D0  |
| **Day 5** | E, D5  | P, D5  | A, D5  |
| **Day 10**| E, D10 | P, D10 | A, D10 |

All panels share uniform axis limits for direct comparison across timepoints and sites.

## Output Structure

```
outputs/per_timepoint_fisher_output/
├── individual/
│   ├── day_0_E_volcano.{png,pdf}
│   ├── day_0_P_volcano.{png,pdf}
│   ├── day_0_A_volcano.{png,pdf}
│   ├── day_5_E_volcano.{png,pdf}
│   ├── day_5_P_volcano.{png,pdf}
│   ├── day_5_A_volcano.{png,pdf}
│   ├── day_10_E_volcano.{png,pdf}
│   ├── day_10_P_volcano.{png,pdf}
│   └── day_10_A_volcano.{png,pdf}
└── composite/
    └── per_timepoint_fisher_composite.{png,pdf}
```
