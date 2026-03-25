# Within-Condition Enrichment Analysis: Methodology & Interpretation

## 1. Workflow Overview

**Input:** `enrichment_results/within_condition_enrichment.csv` produced by `stall_sites_non_consensus.py --enrichment`

For each experimental group (e.g., BWM_day_0) and ribosome site (E, P, A):

1. Amino acid frequencies at stall sites are computed from pooled replicate data
2. Frequencies are compared to genome-wide background frequencies
3. Statistical significance is assessed via **binomial test**
4. P-values are **FDR-corrected** using the Benjamini-Hochberg method (`p_adj`)
5. Log2 enrichment and weighted log2 enrichment are computed
6. Results are visualized as **volcano plots** using `R_scripts/within_condition_enrichment.R`

**Experimental groups:**
- BWM: BWM_day_0, BWM_day_5, BWM_day_10
- Control: control_day_0, control_day_5, control_day_10

**Ribosome sites:**
- **E-site** (exit): position -1 relative to stall
- **P-site** (peptidyl): position 0 (the stall position)
- **A-site** (aminoacyl): position +1 relative to stall

---

## 2. Log2 Enrichment vs Weighted Log2 Enrichment

### Log2 Enrichment (Unweighted)

**Formula:** `log2(observed_frequency / background_frequency)`

- Measures the **magnitude** of over- or under-representation of an amino acid at stall sites relative to its genome-wide frequency
- A value of 0 means the amino acid appears at stall sites exactly as often as expected
- Positive values = enriched (more common at stall sites)
- Negative values = depleted (less common at stall sites)

**Example:** Proline at the P-site with observed frequency 17.35% vs background 5.6%:
- log2(0.1735 / 0.056) = **1.63** (approximately 3.1x enriched)

**Limitation:** Treats a rare amino acid with 2 observations the same as a common one with 200, as long as the fold-change is equal.

### Weighted Log2 Enrichment

**Formula:** `observed_frequency x log2(observed_frequency / background_frequency)`

- Scales the enrichment by how frequently the amino acid actually appears at stall sites
- Combines **magnitude of enrichment** with **frequency of occurrence**

**Effect:**
- High enrichment + low frequency → downweighted (small weighted value)
- High enrichment + high frequency → preserved (large weighted value)

**Example comparison:**

| Amino Acid | Observed Freq | Log2 Enrichment | Weighted Log2 Enrichment |
|-----------|---------------|-----------------|--------------------------|
| X (rare)  | 1%            | 3.0             | 0.01 x 3.0 = **0.03**   |
| Y (common)| 15%           | 1.5             | 0.15 x 1.5 = **0.225**  |

Despite X having double the fold-enrichment, Y has a much larger weighted value because it actually contributes more to the overall stalling signal.

### When to Use Which

- **Unweighted:** Best for identifying which amino acids have the strongest fold-change enrichment, regardless of how common they are. Good for discovering potentially novel stall-inducing residues.
- **Weighted:** Best for identifying which amino acids contribute most to the overall stalling signal. Prioritizes biologically impactful amino acids that are both strongly enriched AND commonly observed.

---

## 3. Confidence Intervals

### Method

**Bayesian credible intervals** using the Beta distribution with **Jeffreys prior** (Beta(0.5, 0.5)).

### Unweighted CI Computation

1. Model stall site amino acid counts as **Binomial(n, p)** where n = total stalls, p = true proportion
2. Apply Jeffreys prior Beta(0.5, 0.5) → posterior is **Beta(count + 0.5, total - count + 0.5)**
3. Extract 95% credible interval on the proportion:
   - Lower: `qbeta(0.025, count + 0.5, total - count + 0.5)`
   - Upper: `qbeta(0.975, count + 0.5, total - count + 0.5)`
4. Transform to log2 enrichment scale: `log2(ci_proportion / background_frequency)`

### Weighted CI Computation

- Multiply unweighted CI bounds by the observed stall frequency:
  - `weighted_CI = stall_freq x ci_log2`
- This is an **approximate heuristic** scaling, not a formally derived Bayesian CI

### Interpretation on Volcano Plots

- **Horizontal error bars** show the 95% CI on the x-axis (enrichment)
- **Narrow bars** = high confidence in the enrichment estimate (large sample size)
- **Wide bars** = uncertain estimate (few observations)
- CIs crossing x = 0 suggest the direction of enrichment is uncertain
- CIs are on the **enrichment axis only**, not on the p-value axis

### Caveats

- The log2 transformation of proportion CIs is approximate (not exact quantiles on the log2 scale), but the error is modest for typical sample sizes
- Weighted CIs are heuristic — interpret as rough visual indicators, not strict statistical bounds
- Background frequency is treated as known (fixed), ignoring its own estimation uncertainty
- Each amino acid is modeled independently, ignoring the compositional constraint (proportions sum to 1)

---

## 4. How to Read the Volcano Plots

### Axes
- **x-axis:** Log2 enrichment (or weighted). Positive = enriched at stall sites, negative = depleted
- **y-axis:** -log10(adjusted p-value). Higher = more statistically significant

### Reference Lines
- **Horizontal dashed line:** Significance threshold at -log10(0.05) ~ 1.3. Points above this line are statistically significant after FDR correction
- **Vertical dashed lines:** Enrichment thresholds at +/- 0.5 (~1.4-fold change). Points beyond these lines show meaningful enrichment magnitude
- **Vertical dotted line:** x = 0, no enrichment

### Visual Encoding
- **Colors:** Amino acid biochemical class
  - Red = Acidic (D, E)
  - Blue = Basic (K, R, H)
  - Green = Hydrophobic (A, V, I, L, M, F, W, Y)
  - Purple = Polar (C, N, Q, S, T)
  - Orange = Neutral (G, P)
- **Shapes:** Triangle = significant (FDR < 0.05), Circle = not significant
- **Labels:** Only significant amino acids are labeled with their single-letter code

### Key Regions
- **Top-right quadrant:** Significantly **enriched** amino acids — potential stall-causing residues
- **Top-left quadrant:** Significantly **depleted** amino acids — disfavored at stall sites
- **Bottom half:** Not statistically significant (below the FDR threshold)

### Y-Axis Capping (`--y-cap`)

When a few amino acids have extremely small adjusted p-values, the resulting -log10 values can stretch the y-axis and compress the remaining points into a narrow band near the bottom. The `--y-cap` argument clamps any -log10(p_adj) value above the specified threshold to that threshold, effectively compressing the y-axis range.

**Example:** With `--y-cap 50`, a point with p_adj = 1e-80 (which would normally plot at y = 80) is capped to y = 50. Omitting the flag disables capping entirely.

### Uniform Axes
All plots within an enrichment type share the same axis ranges, enabling direct visual comparison across groups and sites. The axis range is determined by the group/site with the most extreme values.

---

## 5. Output Structure

```
within_condition_output/
  individual/
    unweighted/          # One plot per (group, site) combination
    weighted/
  composite/
    unweighted/
      BWM_volcano_grid       # 3x3: rows=days, cols=sites
      control_volcano_grid
      day_0_volcano_grid     # 2x3: rows=conditions, cols=sites
      day_5_volcano_grid
      day_10_volcano_grid
    weighted/
      (same structure)
  README.md              # This file
```
