#!/usr/bin/env Rscript

# ============================================================
# Global Occupancy Wilcoxon Bar Plots
# Reads Wilcoxon rank-sum test CSV and generates sorted bar
# plots for codon/AA occupancy fold-change between groups.
# Used for both between-condition and between-timepoint results.
# ============================================================

library(argparse)
library(ggplot2)
library(ggtext)
library(dplyr)
library(patchwork)

# ============================================================
# Test Input
# ============================================================

# INPUT_DIR <- "C:/Users/Aden Le/Documents/GitHub/ribostall/global_occupancy_results/analysis"
# OUTPUT_DIR <- "C:/Users/Aden Le/Documents/GitHub/ribostall/global_occupancy_results/wilcoxin_output"
# args <- list(level = "aa",
#              input = file.path(INPUT_DIR, "aa_wilcoxon_condition.csv"),
#              outdir = file.path(OUTPUT_DIR, "aa_condition"),
#              comparison = "BWM_vs_Control",
#              format = "png",
#              dpi = 300L
#              )

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(description = "Generate bar plots for global occupancy Wilcoxon results")

parser$add_argument("--input",
                    required = TRUE,
                    help = "Path to Wilcoxon CSV (codon or AA level)")

parser$add_argument("--outdir",
                    default = "global_occupancy_wilcoxon_output",
                    help = "Output directory for plots")

parser$add_argument("--level",
                    default = "aa",
                    choices = c("codon", "aa"),
                    help = "Analysis level: codon or aa")

parser$add_argument("--comparison",
                    default = "BWM_vs_Control",
                    help = "Label for the comparison (used in titles)")

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

PAL <- c("Enriched" = "#2E86AB", "Depleted" = "#E84855")

level_label <- ifelse(args$level == "aa", "Amino Acid", "Codon")

# ============================================================
# Read and Prepare Data
# ============================================================

cat("Reading input:", args$input, "\n")
data <- read.csv(args$input, stringsAsFactors = FALSE)

# Select relevant columns
data <- data |>
  select(unit, log2_FC, p_adj)

cat("  Rows:", nrow(data), "\n")

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

# Usage
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
      unit = factor(unit, levels = unit),
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

  p <- ggplot(plot_data, aes(x = unit, y = log2_FC, fill = bar_fill)) +

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
      labels = c("Enriched" = "Enriched (log<sub>2</sub>FC \u2265 0)",
                 "Depleted"  = "Depleted (log<sub>2</sub>FC < 0)")
    ) +

    scale_y_continuous(
      limits = y_limits,
      breaks = y_breaks,
      expand = expansion(mult = c(0, 0))
    ) +

    labs(
      title    = title,
      subtitle = paste0("Bars sorted by log\u2082 fold-change (highest \u2192 lowest)"),
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
# Create Output Directories & Generate Plot
# ============================================================

dir.create(file.path(args$outdir, "individual"),
           recursive = TRUE, showWarnings = FALSE)
file.path(args$outdir, "individual")
cat("\nGenerating bar plot...\n")

comparison_label <- gsub("_", " ", args$comparison)

title <- paste0("Global ", level_label, " Occupancy \u2013 ", comparison_label)

p <- make_barplot(data, title = title,
                  y_limits = y_limits, y_breaks = y_breaks,
                  show_legend = TRUE)

# Wider plot for codons (64 bars)
plot_width <- if (args$level == "codon") 14 else 7

filepath <- file.path(args$outdir, "individual",
                      paste0(args$level, "_", args$comparison, "_barplot"))

save_plot(p, filepath, width = plot_width, height = 5,
          format = args$format, dpi = args$dpi)

cat("  Saved bar plot\n")
p
# ============================================================
# Summary
# ============================================================

cat("\n============================================\n")
cat("Done! Generated 1 plot file\n")
cat("Output directory:", args$outdir, "\n")
cat("Level:", args$level, "\n")
cat("Comparison:", args$comparison, "\n")
cat("Format:", args$format, "\n")
cat("============================================\n")
