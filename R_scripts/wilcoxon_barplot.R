#!/usr/bin/env Rscript

# ============================================================
# Wilcoxon Bar Plots (unified)
# Reads a Wilcoxon rank-sum CSV and generates sorted bar plots
# of log2 fold-change per feature (codon or AA), per E/P/A site.
#
# Handles both datasets:
#   - stall_sites (between-condition, between-timepoint)
#   - global_occupancy (between-condition, between-timepoint)
#
# Schema expected: site, {amino_acid|codon}, median_<grpA>, median_<grpB>,
# log2_FC, U_stat, p_value, p_adj. The two `median_*` columns are read
# generically — the script does not care which two groups are compared.
# ============================================================

library(argparse)
library(ggplot2)
library(ggtext)
library(dplyr)
library(patchwork)

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(description = "Generate bar plots for Wilcoxon enrichment results")

parser$add_argument("--input",
                    required = TRUE,
                    help = "Path to Wilcoxon CSV (codon or AA level)")

parser$add_argument("--outdir",
                    default = "wilcoxon_barplot_output",
                    help = "Output directory for plots")

parser$add_argument("--level",
                    default = "aa",
                    choices = c("codon", "aa"),
                    help = "Analysis level: codon or aa")

parser$add_argument("--comparison",
                    default = "BWM_vs_Control",
                    help = "Label for the comparison (used in titles, file paths)")

parser$add_argument("--format",
                    default = "both",
                    choices = c("pdf", "png", "both"),
                    help = "Output format: pdf, png, or both")

parser$add_argument("--dpi",
                    type = "integer",
                    default = 300L,
                    help = "DPI for PNG output")

args <- parser$parse_args()

feature_col <- ifelse(args$level == "aa", "amino_acid", "codon")

# ============================================================
# Constants
# ============================================================

PAL <- c("Enriched" = "#2E86AB", "Depleted" = "#E84855")
SITE_LABELS <- c("E" = "E-site", "P" = "P-site", "A" = "A-site")

level_label <- ifelse(args$level == "aa", "Amino Acid", "Codon")

# ============================================================
# Read and Prepare Data
# ============================================================

cat("Reading input:", args$input, "\n")
data <- read.csv(args$input, stringsAsFactors = FALSE)

if (!feature_col %in% colnames(data)) {
  stop(sprintf("Expected column '%s' not found in input CSV. Found: %s",
               feature_col, paste(colnames(data), collapse = ", ")))
}

# Select relevant columns. Rename feature_col -> "feature" so the rest of
# the script doesn't have to special-case AA vs codon.
data <- data |>
  rename(feature = !!feature_col) |>
  select(site, feature, log2_FC, p_adj)

cat("  Rows:", nrow(data), "| Sites:", paste(unique(data$site), collapse = ", "), "\n")

# ============================================================
# Compute Uniform Y-Axis Limits
# ============================================================

compute_limits <- function(values, padding_pct = 0.08, min_padding = 0.02, y_cap = NULL) {
  padding <- diff(range(values, na.rm = TRUE)) * padding_pct
  padding <- max(padding, min_padding)
  y_min <- min(values, na.rm = TRUE) - padding
  y_max <- max(values, na.rm = TRUE) + padding
  if (!is.null(y_cap)) {
    y_min <- max(y_min, -abs(y_cap))
    y_max <- min(y_max,  abs(y_cap))
  }
  c(y_min, y_max)
}

compute_breaks <- function(y_lim, n = 6) {
  pretty(y_lim, n = n)
}

y_limits <- compute_limits(data$log2_FC)
y_breaks <- compute_breaks(y_limits)

cat("  Y-axis limits:", round(y_limits, 3), "| Breaks:", length(y_breaks), "\n")

# ============================================================
# Bar Plot Function
# ============================================================

make_barplot <- function(plot_data, title, y_limits, y_breaks,
                         show_legend = TRUE) {

  plot_data <- plot_data |>
    arrange(desc(log2_FC)) |>
    mutate(
      feature  = factor(feature, levels = feature),
      bar_fill = ifelse(log2_FC >= 0, "Enriched", "Depleted"),
      sig_label = case_when(
        p_adj < 0.001 ~ "***",
        p_adj < 0.01  ~ "**",
        p_adj < 0.05  ~ "*",
        TRUE          ~ ""
      ),
      star_y = ifelse(log2_FC >= 0,
                      log2_FC + 0.02,
                      log2_FC - 0.02)
    )

  p <- ggplot(plot_data, aes(x = feature, y = log2_FC, fill = bar_fill)) +

    geom_col(width = 0.8, colour = "white", linewidth = 0.3) +

    geom_hline(yintercept = 0, linewidth = 0.4, colour = "grey40") +

    geom_text(
      aes(y = star_y, label = sig_label),
      size   = 5,
      vjust  = ifelse(plot_data$log2_FC >= 0, 0, 1),
      colour = "grey20",
      family = "sans"
    ) +

    scale_fill_manual(
      values = PAL,
      name   = NULL,
      limits = c("Enriched", "Depleted"),
      labels = c("Enriched" = "Enriched (log<sub>2</sub>FC ≥ 0)",
                 "Depleted"  = "Depleted (log<sub>2</sub>FC < 0)")
    ) +

    scale_y_continuous(
      limits = y_limits,
      breaks = y_breaks,
      expand = expansion(mult = c(0, 0))
    ) +

    labs(
      title    = title,
      subtitle = paste0("Bars sorted by log₂ fold-change (highest → lowest)"),
      x        = level_label,
      y        = bquote(bold(log[2]~"Fold-Change"))
    ) +

    theme_classic(base_size = 13) +
    theme(
      plot.title         = element_text(face = "bold", size = 15, hjust = 0),
      plot.subtitle      = element_text(colour = "grey45", size = 11, hjust = 0,
                                        margin = margin(b = 8)),
      axis.title         = element_text(face = "bold"),
      axis.text          = element_text(colour = "grey20"),
      axis.text.x        = element_text(
        angle = if (args$level == "codon") 90 else 0,
        hjust = if (args$level == "codon") 1 else 0.5,
        vjust = if (args$level == "codon") 0.5 else 1,
        size  = if (args$level == "codon") 8 else 11
      ),
      axis.line          = element_line(colour = "grey60"),
      axis.ticks         = element_line(colour = "grey60"),
      legend.position    = if (show_legend) "bottom" else "none",
      legend.text        = element_markdown(size = 11),
      panel.grid.major.y = element_line(colour = "grey92", linewidth = 0.4),
      plot.margin        = margin(12, 16, 12, 12)
    )

  return(p)
}

# ============================================================
# Helper: Save Plot
# ============================================================

save_plot <- function(p, filepath, width, height, format, dpi) {
  if (format %in% c("pdf", "both")) {
    ggsave(paste0(filepath, ".pdf"), plot = p,
           width = width, height = height, units = "in", device = "pdf")
  }
  if (format %in% c("png", "both")) {
    ggsave(paste0(filepath, ".png"), plot = p,
           width = width, height = height, units = "in", device = "png", dpi = dpi)
  }
}

# ============================================================
# Create Output Directories
# ============================================================

dir.create(file.path(args$outdir, "individual"),
           recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(args$outdir, "composite"),
           recursive = TRUE, showWarnings = FALSE)

comparison_label <- gsub("_", " ", args$comparison)

# Wider plot for codons (64 bars)
plot_width <- if (args$level == "codon") 14 else 7

sites <- c("E", "P", "A")

# ============================================================
# Generate Individual Plots (per-site)
# ============================================================

cat("\nGenerating individual plots...\n")
plot_count <- 0

for (st in sites) {
  plot_data <- data |> filter(site == st)

  title <- paste0(SITE_LABELS[st], " – ", level_label,
                  " – ", comparison_label)

  p <- make_barplot(plot_data, title = title,
                    y_limits = y_limits, y_breaks = y_breaks,
                    show_legend = TRUE)

  filepath <- file.path(args$outdir, "individual",
                        paste0("site_", st, "_", args$level, "_",
                               args$comparison, "_barplot"))
  save_plot(p, filepath, width = plot_width, height = 5,
            format = args$format, dpi = args$dpi)
  plot_count <- plot_count + 1

  cat("  Saved:", SITE_LABELS[st], "\n")
}

# ============================================================
# Composite Plot: E | P | A
# ============================================================

cat("Generating composite plot...\n")

plot_list <- list()
for (st in sites) {
  plot_data <- data |> filter(site == st)
  plot_list[[st]] <- make_barplot(plot_data,
                                  title = SITE_LABELS[st],
                                  y_limits = y_limits, y_breaks = y_breaks,
                                  show_legend = FALSE)
}

composite <- (plot_list[["E"]] | plot_list[["P"]] | plot_list[["A"]]) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = paste0(level_label, " – ", comparison_label),
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
    )
  ) &
  theme(legend.position = "bottom",
        legend.text = element_markdown(size = 11))

filepath <- file.path(args$outdir, "composite",
                      paste0("EPA_", args$level, "_", args$comparison,
                             "_barplot_composite"))
save_plot(composite, filepath, width = plot_width * 3, height = 6,
          format = args$format, dpi = args$dpi)

cat("  Saved composite plot\n")

# ============================================================
# Summary
# ============================================================

total <- plot_count + 1
cat("\n============================================\n")
cat("Done! Generated", total, "total plot files\n")
cat("Output directory:", args$outdir, "\n")
cat("Level:", args$level, "\n")
cat("Comparison:", args$comparison, "\n")
cat("Format:", args$format, "\n")
cat("============================================\n")
