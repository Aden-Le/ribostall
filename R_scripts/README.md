# R_scripts/ — the unified R visualization suite

*Rscript plotting programs that turn the pipeline's statistics CSVs into publication-grade volcano, bar, and overlay figures at both amino-acid and codon resolution.*

> **[ribostall](../README.md)** › R_scripts

---

## Overview

These are the four unified R visualizers of the `ribostall` pipeline, plus one shared constants file they all `source()`. Each plotting script:

- is a standalone `Rscript` program driven entirely by command-line arguments (`optparse` via the `argparse` package);
- is invoked by an `analyze_*.sh` launcher in `shell_scripts/` (the launcher supplies the CLI flags and points at the right CSV);
- reads a stats CSV produced by one of the `*_stats.py` scripts in `../scripts/`;
- writes both **individual** per-panel figures and **composite** (and sometimes **mega-composite**) grids into an output directory, in PDF and/or PNG.

Every script is run **from the repo root** — the shell launchers `cd` there first — so the hard-coded `source("R_scripts/aa_constants.R")` path resolves. (`aa_codon_overlay.R` additionally probes its own location so it also works when run from inside `R_scripts/`.)

A figure is a scatter of features (one point/bar per amino acid or codon) laid out per **E/P/A ribosome site**. Colour always encodes the amino-acid **property class**; shape or bar direction encodes **significance** and **effect direction**. All PDFs render through the `cairo_pdf` device so Unicode glyphs (→, ₂ subscripts, en-dashes) embed correctly on Windows.

### Level switch

Every script works at two resolutions, chosen with `--level {aa,codon}` (or, for the overlay, by taking one CSV of each). At `aa` level the feature column is `amino_acid`; at `codon` level it is `codon`, and each codon is decoded to its amino acid via `CODON2AA` before being coloured by property (stop codons `TAA/TAG/TGA` map to the grey `"Stop"` class).

---

## Contents

| Script | Plots | Driven by | Details |
|---|---|---|---|
| `aa_constants.R` | *(shared constants — not a plotter)* | `source()`d by all four | [↓](#aa_constantsr) |
| `between_group_volcano.R` | Between-group volcano (Fisher odds ratio **or** background-aware enrichment ratio) | `analyze_*_fisher_volcano.sh`, `analyze_*_background_diff_volcano.sh` | [↓](#between_group_volcanor) |
| `between_group_barplot.R` | Sorted log2-FC bar plots (Wilcoxon) | `analyze_*_wilcoxon.sh` | [↓](#between_group_barplotr) |
| `within_condition_volcano.R` | Within-condition binomial enrichment volcano (optional Beta-Jeffreys CIs) | `analyze_*_within_condition*.sh` | [↓](#within_condition_volcanor) |
| `aa_codon_overlay.R` | AA-bar + codon-dot overlay (background-diff) | overlay launcher (AA + codon bg-diff CSVs) | [↓](#aa_codon_overlayr) |

---

## `aa_constants.R`

A pure data file, sourced verbatim by all four plotting scripts. It holds only the values that are byte-identical across them; run-dependent labels (e.g. the FDR-driven significance label) stay in the individual scripts.

| Object | Type | Role |
|---|---|---|
| `AA_CLASS` | named vector, one-letter AA → class | Maps each amino acid to a property class: **Acidic** (D, E), **Basic** (K, R, H), **Hydrophobic** (A, V, I, L, M, F, W, Y), **Polar** (C, N, Q, S, T), **Neutral** (G, P). Drives point/bar **fill**. |
| `CLASS_COLORS` | named vector, class → hex | The Brewer-style palette: Acidic `#E41A1C`, Basic `#377EB8`, Hydrophobic `#4DAF4A`, Polar `#984EA3`, Neutral `#FF7F00`, and **Stop** `#666666` (used only at codon level for stop codons). |
| `CODON2AA` | named vector, codon → one-letter AA | The standard genetic code; `TAA/TAG/TGA` → `"*"`. Used to decode codon-level features to an amino acid so they can be coloured and (in the overlay) clustered. |
| `SITE_LABELS` | named vector, site code → label | `E → "E-site"`, `P → "P-site"`, `A → "A-site"`. Display labels for panel titles/subtitles. |

Because these live in one place, every plot in the suite uses an identical property colour scheme and site vocabulary — a codon dot and its amino-acid bar always share a colour.

---

## `between_group_volcano.R`

### Purpose

Volcano plots for a **between-group** comparison: log2 effect size on the x-axis against −log10(FDR) on the y-axis, one point per feature, faceted by a grouping column (typically `timepoint` or `condition`). One script drives two test families, selected by `--effect-col`:

- **Fisher's exact** — x = `log2(odds_ratio)` (the default; the linear `odds_ratio` column is log2-transformed).
- **Background-aware conditional binomial** — x = `delta_log2_enrichment`, already on a log2 scale, plotted directly with `--effect-is-log2`.

`--x-label`, `--title-test-label`, and `--composite-tag` keep the axis label, the composite title, and the composite filename honest for whichever test produced the CSV.

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the between-group CSV (codon or AA level). |
| `--outdir` | `between_group_volcano_output` | Output directory (an `individual/` and `composite/` subdir are created inside). |
| `--level` | `aa` | Analysis level: `aa` (feature column `amino_acid`) or `codon` (`codon`). |
| `--group-col` | `timepoint` | Column to facet individual/composite panels by (e.g. `timepoint` or `condition`). Ignored under `--flat-design`. |
| `--flat-design` | `FALSE` (flag) | Single-comparison input with no grouping column (only `site`) — e.g. a between-condition Fisher or a background-aware diff run **without** `--timepoints`. Plots one composite row of A/P/E panels. Mirrors `within_condition_volcano.R`'s flag of the same name. |
| `--comparison-label` | `BWM vs Control` | Comparison description used in titles. |
| `--effect-col` | `odds_ratio` | Effect-size column placed on the x-axis. log2-transformed unless `--effect-is-log2`. |
| `--effect-is-log2` | `FALSE` (flag) | Treat `--effect-col` as already log2-scaled (e.g. `delta_log2_enrichment`) and plot it directly. |
| `--x-label` | `""` | Override the x-axis label (plain text). Empty → the bold `Log₂ (Odds Ratio)` expression. |
| `--title-test-label` | `Fisher's Test` | Test name in the composite title (`<level> <label> (<comparison>)`). E.g. `Background-Aware Enrichment`. |
| `--composite-tag` | `fisher` | Test token embedded in the composite **filename** (`<level>_<tag>_composite`). E.g. `binomial` for the background-aware test. |
| `--format` | `both` | `pdf`, `png`, or `both`. |
| `--dpi` | `300` | DPI for PNG output. |

### Input CSV

A Fisher or background-diff stats CSV from `../scripts/` (e.g. `stall_sites_consensus_intersection_stats.py` → Fisher A3/A6, or `*_union_stats.py` → background-aware A4/A7; and the global-occupancy Fisher tests). Columns consumed: the feature column (`amino_acid` / `codon`), `site`, the `--effect-col` (`odds_ratio` or `delta_log2_enrichment`), `p_adj`, and the grouping column named by `--group-col` (absent under `--flat-design`).

### What it plots

- **Individual** (`individual/`): one volcano per group-value × site (faceted mode), or one per site (`--flat-design`).
- **Composite** (`composite/`): a single grid, rows = group values (numerically ordered when they look like `day_0/day_5/day_10`, else alphabetical), columns = A/P/E sites, legends collected once at the bottom. Filename `<level>_<composite-tag>_composite`.

### Under the hood

Each panel is built by `make_volcano()`. Points are `geom_point` coloured by `aa_class` (via `CLASS_COLORS`) and shaped by significance (`p_adj < 0.05` → filled triangle `17`, else circle `16`). Reference guides: dashed verticals at |log2| = 0.5 (effect thresholds), a dashed horizontal at −log10(0.05) (FDR line), and a dotted vertical at 0 (null effect). Only **significant** features are labelled, with `ggrepel` (`max.overlaps = 15`). Defensive caps keep degenerate points on-canvas: −log10(FDR) is capped at 50 and the log2 effect is clamped to ±10 (so `OR = 0`/`Inf` don't blow up the axis). x-limits are symmetric (±1.1 × max |effect|); y runs `[0, 1.1 × max]`. Composites are assembled with `patchwork::wrap_plots` + `plot_layout(guides = "collect")`; PDFs go through `cairo_pdf`.

### Related

Launchers: `analyze_*_fisher_volcano.sh` and `analyze_*_background_diff_volcano.sh` under
`../shell_scripts/c_elegans/stall_sites_consensus_intersection/` (Fisher),
`../shell_scripts/c_elegans/stall_sites_consensus_union/` (background-aware),
and `../shell_scripts/c_elegans/global_occupancy/`.

---

## `between_group_barplot.R`

### Purpose

Sorted bar plots of the **Wilcoxon** log2 fold-change per feature, per E/P/A site. Wilcoxon is the only test that feeds this plot: its coarse rank-test p-values suit bars over a volcano. Serves both between-condition and between-timepoint comparisons; the two `median_*` columns are read generically, so the script is agnostic to which two groups are compared.

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the Wilcoxon CSV (codon or AA level). |
| `--outdir` | `between_group_barplot_output` | Output directory (`individual/` and `composite/` created inside). |
| `--level` | `aa` | Analysis level: `aa` or `codon`. |
| `--comparison` | `BWM_vs_Control` | Comparison label, used in titles and file paths. Split on `_vs_` to build the direction subtitle (e.g. `Day_10_vs_Day_0`). |
| `--format` | `both` | `pdf`, `png`, or `both`. |
| `--dpi` | `300` | DPI for PNG output. |

### Input CSV

A Wilcoxon rank-sum stats CSV (stall-sites A2/A5 non-consensus Wilcoxons, or global-occupancy between-condition / between-timepoint Wilcoxons). Expected schema: `site`, the feature column, `median_<grpA>`, `median_<grpB>`, `log2_FC`, `U_stat`, `p_value`, `p_adj`. Only `site`, feature, `log2_FC`, and `p_adj` are actually plotted.

### What it plots

- **Individual** (`individual/`): one bar plot per site (`site_<A|P|E>_<level>_<comparison>_barplot`).
- **Composite** (`composite/`): A | P | E side by side (`APE_<level>_<comparison>_barplot_composite`).

### Under the hood

`make_barplot()` sorts features by `log2_FC` descending (locking the x-order via a factor), draws `geom_col` bars filled by `aa_class`, and places significance stars above/below each bar (`***` < 0.001, `**` < 0.01, `*` < 0.05, else none) nudged just past the bar tip. Bar **direction** conveys effect sign — up = enriched in the numerator group, down = enriched in the denominator — and the subtitle spells this out using ↑/↓/→ arrows typeset from the math `symbol()` engine (so they survive `cairo_pdf`). A uniform y-range with padding is shared across all three site panels for comparability. Codon plots are widened (64 bars) and their x-labels rotated 90°. Composites use `patchwork` with a collected legend.

### Related

Launcher: `analyze_stall_sites_non_consensus_wilcoxon.sh` under
`../shell_scripts/c_elegans/stall_sites_non_consensus/` (the only R-plot launcher for that stage — non-consensus runs A2/A5 Wilcoxons only), and the Wilcoxon launcher under `../shell_scripts/c_elegans/global_occupancy/`.

---

## `within_condition_volcano.R`

### Purpose

Volcano plots for the **within-condition binomial** enrichment test: each feature's observed frequency at a site is compared to its background frequency, giving a log2 enrichment (x-axis) against −log10(FDR) (y-axis). Supports both an **unweighted** and an **observed-frequency-weighted** enrichment axis, optional **Beta-Jeffreys** confidence-interval error bars, and grids over the full condition × timepoint cross-product (or per-group under a flat control-vs-treatment design).

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the within-condition enrichment CSV (codon or AA level). |
| `--outdir` | `within_condition_volcano_output` | Output directory (`individual/` and `composite/`, further split per enrichment type). |
| `--level` | `aa` | Analysis level: `aa` or `codon`. |
| `--show-ci` | `FALSE` (flag) | Draw Beta-Jeffreys CI horizontal error bars on each point. |
| `--mega-composite` | `FALSE` (flag) | Also emit an all-groups grid (rows = condition × timepoint, cols = sites). |
| `--flat-design` | `FALSE` (flag) | Flat control-vs-treatment input with no timepoint dimension (`group == condition`). Composites are built per group; the by-day composite is skipped. Used by the consensus stall-site pipeline. |
| `--enrichment-type` | `both` | Which axis to plot: `unweighted`, `weighted`, or `both`. |
| `--format` | `both` | `pdf`, `png`, or `both`. |
| `--dpi` | `300` | DPI for PNG output. |
| `--y-cap` | `50` | Cap −log10(p_adj) at this maximum. Required because `p_adj = 0` (saturated FDR) yields `Inf` and breaks plotting; a smaller value compresses the y-axis further. |

### Input CSV

A within-condition binomial stats CSV (`within_condition_binomial_{aa,codon}.csv` for stall sites, or `aa_within_condition_binomial.csv` / `codon_within_condition_binomial.csv` for global occupancy). Expected schema: `site`, `group`, `condition`, `timepoint`, the feature column, `observed_count`, `total_n`, `observed_freq`, `bg_freq`, `log2_enrichment`, `weighted_log2_enrichment`, `p_value`, `p_adj`.

### What it plots

- **Individual** (`individual/<etype>/`): one volcano per group × site, per enrichment type.
- **Composite** (`composite/<etype>/`): grids of A/P/E panels. In the default timepoint design, **by-condition** grids (rows = timepoint) and **by-day** grids (rows = condition) are both emitted; under `--flat-design`, one grid per group instead. `--mega-composite` adds a single all-groups grid (`all_groups_volcano_grid`).

### Under the hood

`make_volcano()` mirrors the between-group volcano (property-coloured points, significance-shaped, ggrepel labels on significant features, 0.5 / FDR / null reference lines). Two axes are prepared: **unweighted** (`log2_enrichment`) and **weighted** (`weighted_log2_enrichment = observed_freq × log2_enrichment`), each with its own symmetric x-limit that includes the CI bounds so error bars never run off-canvas. **Beta-Jeffreys CIs** are computed on the underlying proportion — `qbeta(0.025 / 0.975, observed_count + 0.5, total_n − observed_count + 0.5)` — then propagated through the `log2(prop / bg_freq)` transform (and multiplied by `observed_freq` for the weighted axis); with `--show-ci` they render as `geom_errorbarh`. The `--y-cap` clamps −log10(p_adj) (default 50) so saturated `p_adj = 0` points stay finite, and all log2 values are clamped to ±10. Composites use `patchwork` with collected legends; PDFs via `cairo_pdf`.

### Related

Launchers: `analyze_*_within_condition*.sh` under
`../shell_scripts/c_elegans/stall_sites_consensus_union/`,
`../shell_scripts/c_elegans/stall_sites_consensus_intersection/`,
and `../shell_scripts/c_elegans/global_occupancy/`.

---

## `aa_codon_overlay.R`

### Purpose

The fourth visualization: it fuses the amino-acid view and the codon view into a single figure, per E/P/A site × timepoint panel. **Bars** show each amino acid's effect; **dots** show each synonymous codon's effect clustered underneath its amino acid — so you can see at a glance whether a codon's signal tracks its amino acid's, or whether one synonymous codon carries the whole effect. Because it needs **both** the AA and codon CSVs, it takes two inputs rather than the single-input / `--level` pattern of the other scripts.

### Arguments

| Flag | Default | Description |
|---|---|---|
| `--input-aa` | `NULL` | Amino-acid background-diff CSV (drives the bars). Optional — leave unset to use the bundled `test_data/` fixtures via the in-script TEST toggle. |
| `--input-codon` | `NULL` | Codon background-diff CSV (drives the dots). Same TEST-toggle behaviour. |
| `--outdir` | `aa_codon_overlay_output` | Output directory (`individual/` and `composite/` created inside). |
| `--effect-col` | `delta_log2_enrichment` | Effect-size column, shared by both CSVs. |
| `--fdr` | `0.05` | FDR threshold (on `p_adj`) for the codon-dot significance shape. |
| `--cap` | `1.0` | Clamp codon dots with `|effect| > cap` to ±cap and print the true value beside them (compresses the y-axis). |
| `--comparison-label` | `BWM vs Control` | Comparison description used in titles. |
| `--format` | `both` | `pdf`, `png`, or `both`. |
| `--dpi` | `300` | DPI for PNG output. |

### Input CSVs

Two **background-diff** CSVs (one AA, one codon), each with `site`, `timepoint`, the feature column (`amino_acid` / `codon`), the `--effect-col` (`delta_log2_enrichment`), and `p_adj` — exactly the schema of the `test_data/` fixtures. See [`test_data/README.md`](./test_data/README.md) for the full column list.

### What it plots

- **Individual** (`individual/`): one wide overlay panel per site × timepoint (`site_<A|P|E>_<timepoint>_aa_codon_overlay`).
- **Composite** (`composite/`): a grid, rows = timepoint, cols = site (`aa_codon_overlay_composite`); missing site×timepoint cells become `plot_spacer()`s so the grid stays aligned.

### Under the hood

`make_overlay_plot()` builds each panel in two layers. **Bars** (`geom_rect`): one per amino acid, height = the AA effect, ordered left→right by descending AA effect, filled by property colour at 35 % alpha with a full-opacity same-colour border; the one-letter AA is labelled on top. **Dots** (`geom_point`): one per codon, x placed at an integer position **clustered under its amino acid** (sorted by codon effect within the AA), y = the codon effect, drawn solid black and shaped by significance (triangle if `p_adj < --fdr`, else circle). Each AA bar spans the x-range of its codons (computed from the per-codon min/max positions). Dots beyond `--cap` are clamped to ±cap with their true value printed alongside, and dashed grey lines mark the clamp boundary. Codons that fail to decode are dropped defensively. The significance legend label is built from `--fdr` at runtime (hence it lives in this script, not `aa_constants.R`). Composites via `patchwork`; PDFs via `cairo_pdf`.

### Related

Runs off the **background-aware diff** outputs (per-timepoint AA + codon), the same family that feeds the `--effect-is-log2` mode of `between_group_volcano.R`. The two bundled `test_data/` CSVs are the built-in TEST fixtures for this script.

---

## See also

- [`../scripts/README.md`](../scripts/README.md) — the Python pipeline entry points, including the `*_stats.py` scripts that produce every CSV consumed here.
- [`./test_data/README.md`](./test_data/README.md) — the bundled fixture CSVs (per-timepoint background-diff AA + codon).
- [`../shell_scripts/`](../shell_scripts/) — the `analyze_*.sh` launchers that wire these scripts into the pipeline, grouped by organism and stage:
  - `c_elegans/stall_sites_consensus_union/`
  - `c_elegans/stall_sites_consensus_intersection/`
  - `c_elegans/stall_sites_non_consensus/`
  - `c_elegans/global_occupancy/`
- [`../README.md`](../README.md) — repository root.
