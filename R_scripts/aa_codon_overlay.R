#!/usr/bin/env Rscript

# ============================================================
# Amino-Acid Bar + Codon Dot Overlay (4th visualization)
# Fuses the amino-acid view and the codon view into one plot.
#
# Per E/P/A site x timepoint panel:
#   - One BAR per amino acid, height = the AA effect size
#     (delta_log2_enrichment). Bars are ordered left -> right by
#     AA effect (descending). Fill = AA-property colour at reduced
#     alpha; border = the same property colour at full opacity.
#     The amino-acid letter is labelled on top of the bar.
#   - DOTS for each synonymous codon, placed at the codon's own
#     x-position (clustered under its amino acid) and y = the codon
#     effect (delta_log2_enrichment). Dots are solid black, shaped
#     by significance: triangle if p_adj < FDR, circle otherwise.
#
# Needs BOTH the amino-acid CSV (for bars) and the codon CSV (for
# dots), so it takes two inputs rather than the single-input /
# --level pattern of the other R scripts. Both files share the
# schema: site, timepoint, {amino_acid|codon}, ..., <effect-col>,
# ..., p_value, p_adj.
# ============================================================

library(argparse)
library(ggplot2)
library(ggtext)
library(dplyr)
library(patchwork)

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(
  description = "AA-bar + codon-dot overlay plot (background-diff CSVs)"
)

parser$add_argument("--input-aa",
                    default = NULL,
                    help = "Path to the amino-acid background-diff CSV (drives the bars). Optional: leave unset and use the TEST toggle in-script.")

parser$add_argument("--input-codon",
                    default = NULL,
                    help = "Path to the codon background-diff CSV (drives the dots). Optional: leave unset and use the TEST toggle in-script.")

parser$add_argument("--outdir",
                    default = "aa_codon_overlay_output",
                    help = "Output directory for plots")

parser$add_argument("--effect-col",
                    default = "delta_log2_enrichment",
                    help = "Name of the effect-size column (shared by both CSVs). Default 'delta_log2_enrichment'.")

parser$add_argument("--fdr",
                    type = "double",
                    default = 0.05,
                    help = "FDR threshold (on p_adj) for the codon-dot significance shape. Default 0.05.")

parser$add_argument("--cap",
                    type = "double",
                    default = 1.0,
                    help = "Clamp codon dots whose |effect| exceeds this to +/- cap, and label them with their true value. Compresses the y-axis. Default 1.0.")

parser$add_argument("--comparison-label",
                    default = "BWM vs Control",
                    help = "Label describing the comparison (used in titles)")

parser$add_argument("--format",
                    default = "both",
                    choices = c("pdf", "png", "both"),
                    help = "Output format: pdf, png, or both")

parser$add_argument("--dpi",
                    type = "integer",
                    default = 300L,
                    help = "DPI for PNG output")

args <- parser$parse_args()

# ============================================================
# Constants
# ============================================================

# AA_CLASS, CLASS_COLORS, CODON2AA, SITE_LABELS — shared verbatim
# with the other R_scripts (between_group_*.R / within_condition_*.R).
# script_dir resolves aa_constants.R (and the test_data/ copies below)
# whether the script is run from the repo root or from inside R_scripts/.
script_dir <- if (file.exists("aa_constants.R")) "." else "R_scripts"
source(file.path(script_dir, "aa_constants.R"))

# Significance legend label depends on the chosen FDR threshold.
SIG_LABEL  <- sprintf("Significant (FDR < %g)", args$fdr)
SHAPE_VALS <- c(17, 16)
names(SHAPE_VALS) <- c(SIG_LABEL, "Not significant")

comparison_label <- args$comparison_label
effect_col       <- args$effect_col

# ============================================================
# Input selection — PRODUCTION (default) vs TEST
# To run local tests: comment out the PRODUCTION block and
# uncomment the TEST block (no CLI args needed). The TEST inputs are
# the copies bundled in R_scripts/test_data/, found via script_dir.
# ============================================================

# --- PRODUCTION (reads CLI --input-aa / --input-codon) ---
input_aa    <- args$input_aa
input_codon <- args$input_codon

# --- TEST (uncomment to use the bundled copies in R_scripts/test_data/) ---
# input_aa    <- file.path(script_dir, "test_data", "per_timepoint_background_diff_aa.csv")
# input_codon <- file.path(script_dir, "test_data", "per_timepoint_background_diff_codon.csv")

if (is.null(input_aa) || is.null(input_codon)) {
  stop(paste0(
    "No input paths resolved. Either pass --input-aa AND --input-codon, ",
    "or uncomment the TEST block near the top of the script."
  ))
}

# ============================================================
# Read and Prepare Data
# ============================================================

read_level <- function(path, feature_col) {
  cat("Reading input:", path, "\n")
  df <- read.csv(path, stringsAsFactors = FALSE)
  for (col in c("site", "timepoint", feature_col, effect_col, "p_adj")) {
    if (!col %in% colnames(df)) {
      stop(sprintf("Expected column '%s' not found in %s. Found: %s",
                   col, path, paste(colnames(df), collapse = ", ")))
    }
  }
  df
}

aa_raw    <- read_level(input_aa, "amino_acid")
codon_raw <- read_level(input_codon, "codon")

# Bars: one effect per amino acid, classified by property.
aa_data <- aa_raw |>
  rename(aa = amino_acid, effect = !!effect_col) |>
  mutate(aa_class = AA_CLASS[aa]) |>
  select(site, timepoint, aa, effect, p_adj, aa_class)

# Dots: one effect per codon, decoded to its amino acid + property.
codon_data <- codon_raw |>
  rename(codon = codon, effect = !!effect_col) |>
  mutate(
    encoded_aa = CODON2AA[toupper(codon)],
    aa_class   = ifelse(encoded_aa == "*", "Stop", AA_CLASS[encoded_aa])
  ) |>
  select(site, timepoint, codon, encoded_aa, effect, p_adj, aa_class)

# Drop any codon that did not decode (defensive; should not happen).
n_bad <- sum(is.na(codon_data$encoded_aa))
if (n_bad > 0) {
  cat("  WARNING: dropping", n_bad, "codon rows that did not decode to an AA\n")
  codon_data <- codon_data |> filter(!is.na(encoded_aa))
}

cat("  AA rows:", nrow(aa_data),
    "| codon rows:", nrow(codon_data),
    "| sites:", paste(unique(aa_data$site), collapse = ", "),
    "| timepoints:", paste(unique(aa_data$timepoint), collapse = ", "), "\n")

# ============================================================
# Y-Axis Limits (codon dots clamped to +/- cap; bars sit well within)
# ============================================================

cap       <- args$cap
y_pad     <- 0.18 * cap                 # headroom for the clamp value labels
y_limits  <- c(-cap - y_pad, cap + y_pad)
y_breaks  <- pretty(y_limits, n = 6)
label_off <- diff(y_limits) * 0.03      # vertical offset for AA-on-bar labels

n_capped <- sum(abs(codon_data$effect) >= cap)
cat("  Codon dots clamped at +/-", cap, "(", n_capped, "of", nrow(codon_data),
    "beyond cap) | y-axis limits:", round(y_limits, 3), "\n")

# ============================================================
# Overlay Plot Function (one site x timepoint panel)
# ============================================================

make_overlay_plot <- function(aa_sub, codon_sub, title,
                              show_legend = TRUE, show_codon_labels = TRUE) {

  # Bar order: amino acids by effect, descending.
  aa_order <- aa_sub |> arrange(desc(effect)) |> pull(aa)
  aa_order <- aa_order[aa_order %in% codon_sub$encoded_aa]

  # Cluster codons under their amino acid: AA-effect order, then by
  # codon effect (descending) within the AA. Integer x-position per codon.
  codon_sub <- codon_sub |>
    filter(encoded_aa %in% aa_order) |>
    mutate(aa_rank = match(encoded_aa, aa_order)) |>
    arrange(aa_rank, desc(effect))
  codon_sub$codon  <- factor(codon_sub$codon, levels = codon_sub$codon)
  codon_sub$codon_x <- as.integer(codon_sub$codon)
  codon_sub$significance <- factor(
    ifelse(codon_sub$p_adj < args$fdr, SIG_LABEL, "Not significant"),
    levels = c(SIG_LABEL, "Not significant")
  )

  # Clamp dots whose |effect| exceeds the cap to +/- cap; the true value
  # is printed beside the clamped dot so nothing is silently hidden.
  codon_sub$capped    <- abs(codon_sub$effect) >= cap
  codon_sub$effect_pl <- pmax(pmin(codon_sub$effect, cap), -cap)
  cap_lab <- codon_sub[codon_sub$capped, , drop = FALSE]

  # Bar geometry: each AA spans the x-range of its codons.
  codon_pos <- codon_sub |>
    group_by(encoded_aa) |>
    summarise(xmin = min(codon_x) - 0.5,
              xmax = max(codon_x) + 0.5,
              center_x = mean(codon_x),
              .groups = "drop")
  bar_df <- aa_sub |>
    inner_join(codon_pos, by = c("aa" = "encoded_aa")) |>
    mutate(label_y = ifelse(effect >= 0, effect + label_off, effect - label_off))

  p <- ggplot() +

    # Bars: AA effect, translucent property fill, solid same-colour border.
    geom_rect(
      data = bar_df,
      aes(xmin = xmin, xmax = xmax, ymin = 0, ymax = effect,
          fill = aa_class, colour = aa_class),
      alpha = 0.35, linewidth = 0.6
    ) +

    geom_hline(yintercept = 0, linewidth = 0.4, colour = "grey40") +

    # Dashed lines marking the clamp boundary.
    geom_hline(yintercept = c(-cap, cap), linetype = "dashed",
               colour = "grey70", linewidth = 0.3) +

    # Dots: codon effect (clamped to +/- cap), black, triangle (sig) / circle (ns).
    geom_point(
      data = codon_sub,
      aes(x = codon_x, y = effect_pl, shape = significance),
      colour = "black", size = 2, stroke = 0.4
    ) +

    # True value beside each clamped dot.
    geom_text(
      data = cap_lab,
      aes(x = codon_x, y = effect_pl, label = sprintf("%.2f", effect)),
      vjust = ifelse(cap_lab$effect > 0, -0.6, 1.5),
      size = 2.6, colour = "grey25"
    ) +

    # Amino-acid letter on top of each bar.
    geom_text(
      data = bar_df,
      aes(x = center_x, y = label_y, label = aa),
      vjust = ifelse(bar_df$effect >= 0, 0, 1),
      size = 4, fontface = "bold", colour = "grey15"
    ) +

    scale_fill_manual(values = CLASS_COLORS, name = "Amino-acid property") +
    scale_colour_manual(values = CLASS_COLORS, guide = "none") +
    scale_shape_manual(values = SHAPE_VALS, name = NULL) +

    scale_x_continuous(
      breaks = if (show_codon_labels) codon_sub$codon_x else NULL,
      labels = if (show_codon_labels) as.character(codon_sub$codon) else NULL,
      expand = expansion(mult = 0.01)
    ) +
    scale_y_continuous(limits = y_limits, breaks = y_breaks) +

    labs(
      title    = title,
      subtitle = sprintf(paste0("Bars = amino-acid effect (ordered high to low); ",
                                "dots = codons, clamped at +/-%g (value shown)"), cap),
      x        = "Codon (clustered by amino acid)",
      y        = bquote(bold(Delta~log[2]~"enrichment"))
    ) +

    theme_classic(base_size = 13) +
    theme(
      plot.title         = element_text(face = "bold", size = 15, hjust = 0),
      plot.subtitle      = element_text(colour = "grey45", size = 11, hjust = 0,
                                        margin = margin(b = 8)),
      axis.title         = element_text(face = "bold"),
      axis.text          = element_text(colour = "grey20"),
      axis.text.x        = if (show_codon_labels) {
        element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7)
      } else {
        element_blank()
      },
      axis.ticks.x       = if (show_codon_labels) element_line(colour = "grey60")
                           else element_blank(),
      axis.line          = element_line(colour = "grey60"),
      axis.ticks.y       = element_line(colour = "grey60"),
      legend.position    = if (show_legend) "bottom" else "none",
      legend.text        = element_text(size = 11),
      panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.4),
      plot.margin        = margin(12, 16, 12, 12)
    )

  p
}

# ============================================================
# Helper: Save Plot
# ============================================================

save_plot <- function(p, filepath, width, height, format, dpi) {
  if (format %in% c("pdf", "both")) {
    # Use cairo_pdf rather than the classic "pdf" device so Unicode glyphs in
    # titles/axes/subtitles (e.g. the → arrow, the ₂ subscript, the – en-dash)
    # embed correctly. The base PostScript fonts drop them on Windows
    # (the "mbcsToSbcs ... substituted for <U+....>" warning at render time).
    ggsave(paste0(filepath, ".pdf"), plot = p,
           width = width, height = height, units = "in", device = cairo_pdf)
  }
  if (format %in% c("png", "both")) {
    ggsave(paste0(filepath, ".png"), plot = p,
           width = width, height = height, units = "in", device = "png", dpi = dpi)
  }
}

# ============================================================
# Order Sites and Timepoints (defensive)
# ============================================================

# Sites in canonical A | P | E order, keeping only those present.
sites <- intersect(c("A", "P", "E"), unique(aa_data$site))

# Timepoints sorted numerically when they embed digits (day_0 < day_5
# < day_10), falling back to a plain sort otherwise.
timepoints <- unique(aa_data$timepoint)
num_keys   <- suppressWarnings(as.numeric(gsub("[^0-9]", "", timepoints)))
if (all(!is.na(num_keys)) && any(num_keys != 0)) {
  timepoints <- timepoints[order(num_keys)]
} else {
  timepoints <- sort(timepoints)
}

# ============================================================
# Output Directories
# ============================================================

dir.create(file.path(args$outdir, "individual"),
           recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(args$outdir, "composite"),
           recursive = TRUE, showWarnings = FALSE)

# ============================================================
# Individual Plots (per site x timepoint)
# ============================================================

cat("\nGenerating individual plots...\n")
plot_count <- 0

for (tp in timepoints) {
  for (st in sites) {
    aa_sub    <- aa_data    |> filter(site == st, timepoint == tp)
    codon_sub <- codon_data |> filter(site == st, timepoint == tp)
    if (nrow(aa_sub) == 0 || nrow(codon_sub) == 0) next

    title <- paste0(SITE_LABELS[st], " – ", tp, " – ", comparison_label)
    p <- make_overlay_plot(aa_sub, codon_sub, title = title,
                           show_legend = TRUE, show_codon_labels = TRUE)

    filepath <- file.path(args$outdir, "individual",
                          paste0("site_", st, "_", tp, "_aa_codon_overlay"))
    save_plot(p, filepath, width = 16, height = 6,
              format = args$format, dpi = args$dpi)
    plot_count <- plot_count + 1
    cat("  Saved:", SITE_LABELS[st], "-", tp, "\n")
  }
}

# ============================================================
# Composite Grid (rows = timepoint, cols = site)
# ============================================================

cat("Generating composite plot...\n")

plot_list <- list()
for (tp in timepoints) {
  for (st in sites) {
    aa_sub    <- aa_data    |> filter(site == st, timepoint == tp)
    codon_sub <- codon_data |> filter(site == st, timepoint == tp)
    if (nrow(aa_sub) == 0 || nrow(codon_sub) == 0) {
      plot_list[[length(plot_list) + 1]] <- patchwork::plot_spacer()
    } else {
      plot_list[[length(plot_list) + 1]] <- make_overlay_plot(
        aa_sub, codon_sub,
        title = paste0(SITE_LABELS[st], " – ", tp),
        show_legend = FALSE, show_codon_labels = TRUE
      )
    }
  }
}

composite <- patchwork::wrap_plots(plot_list, ncol = length(sites)) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = paste0("AA bar / codon dot overlay – ", comparison_label),
    theme = theme(plot.title = element_text(hjust = 0.5, size = 18, face = "bold"))
  ) &
  theme(legend.position = "bottom", legend.text = element_text(size = 11))

filepath <- file.path(args$outdir, "composite", "aa_codon_overlay_composite")
save_plot(composite, filepath,
          width = max(length(sites) * 14, 14),
          height = max(length(timepoints) * 5.5 + 1.5, 6),
          format = args$format, dpi = args$dpi)
cat("  Saved composite plot\n")

# ============================================================
# Summary
# ============================================================

cat("\n============================================\n")
cat("Done! Generated", plot_count, "individual + 1 composite plot\n")
cat("Output directory:", args$outdir, "\n")
cat("Comparison:", comparison_label, "\n")
cat("Format:", args$format, "\n")
cat("============================================\n")
