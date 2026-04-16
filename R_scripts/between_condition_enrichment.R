#!/usr/bin/env Rscript

# ============================================================
# Between-Condition Enrichment Bar Plots
# Reads between_condition_wilcoxon.csv and generates sorted
# bar plots for amino acid log2 fold-change (BWM vs Control)
# at each ribosome site (E, P, A).
# ============================================================

library(argparse)
library(ggplot2)
library(ggtext)
library(dplyr)
library(patchwork)

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(description = "Generate bar plots for between-condition amino acid enrichment")

parser$add_argument("--input",
                    default = "enrichment_results/between_condition_wilcoxon.csv",
                    help = "Path to between_condition_wilcoxon.csv")

parser$add_argument("--outdir",
                    default = "between_condition_output",
                    help = "Output directory for plots")

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

SITE_LABELS <- c("E" = "E-site", "P" = "P-site", "A" = "A-site")
PAL <- c("Enriched" = "#2E86AB", "Depleted" = "#E84855")

# ============================================================
# Read and Prepare Data
# ============================================================

cat("Reading input:", args$input, "\n")
data <- read.csv(args$input, stringsAsFactors = FALSE)

data <- data |>
  select(site, amino_acid, log2_FC, p_adj)

cat("  Rows:", nrow(data), "| Sites:", paste(unique(data$site), collapse = ", "), "\n")

# ============================================================
# Compute Uniform Y-Axis Limits
# ============================================================

global_y_min <- min(data$log2_FC) - 0.12
global_y_max <- max(data$log2_FC) + 0.12
y_limits <- c(global_y_min, global_y_max)

# Breaks at 0.2 intervals covering the full range
y_breaks <- seq(floor(global_y_min * 5) / 5, ceiling(global_y_max * 5) / 5, by = 0.2)

cat("  Y-axis limits:", round(y_limits, 3), "| Breaks:", length(y_breaks), "\n")

# ============================================================
# Bar Plot Function
# ============================================================

make_barplot <- function(plot_data, site_key, y_limits, y_breaks,
                         show_legend = TRUE) {

  # Sort by log2_FC descending and prepare columns
  plot_data <- plot_data |>
    arrange(desc(log2_FC)) |>
    mutate(
      amino_acid = factor(amino_acid, levels = amino_acid),
      bar_fill   = ifelse(log2_FC >= 0, "Enriched", "Depleted"),
      sig_label  = case_when(
        p_adj < 0.001 ~ "***",
        p_adj < 0.01  ~ "**",
        p_adj < 0.05  ~ "*",
        TRUE          ~ ""
      ),
      star_y = ifelse(log2_FC >= 0,
                      log2_FC + 0.02,
                      log2_FC - 0.02)
    )

  p <- ggplot(plot_data, aes(x = amino_acid, y = log2_FC, fill = bar_fill)) +

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
      title    = paste0(SITE_LABELS[site_key], " \u2013 Amino Acid Enrichment"),
      subtitle = "Bars sorted by log\u2082 fold-change (highest \u2192 lowest)",
      x        = "Amino Acid",
      y        = bquote(bold(log[2]~"Fold-Change"))
    ) +

    theme_classic(base_size = 13) +
    theme(
      plot.title         = element_text(face = "bold", size = 15, hjust = 0),
      plot.subtitle      = element_text(colour = "grey45", size = 11, hjust = 0,
                                        margin = margin(b = 8)),
      axis.title         = element_text(face = "bold"),
      axis.text          = element_text(colour = "grey20"),
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

# ============================================================
# Generate Individual Plots
# ============================================================

cat("\nGenerating individual plots...\n")

sites <- c("E", "P", "A")
plot_count <- 0

for (st in sites) {
  plot_data <- data |> filter(site == st)

  p <- make_barplot(plot_data, site_key = st,
                    y_limits = y_limits, y_breaks = y_breaks,
                    show_legend = TRUE)

  filepath <- file.path(args$outdir, "individual",
                        paste0("site_", st, "_barplot"))
  save_plot(p, filepath, width = 7, height = 5,
            format = args$format, dpi = args$dpi)
  plot_count <- plot_count + 1

  cat("  Saved:", SITE_LABELS[st], "\n")
}

cat("  Saved", plot_count, "individual plots\n")

# ============================================================
# Composite Plot: E | P | A
# ============================================================

cat("Generating composite plot...\n")

plot_list <- list()
for (st in sites) {
  plot_data <- data |> filter(site == st)
  plot_list[[st]] <- make_barplot(plot_data, site_key = st,
                                  y_limits = y_limits, y_breaks = y_breaks,
                                  show_legend = FALSE)
}

composite <- (plot_list[["E"]] | plot_list[["P"]] | plot_list[["A"]]) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Between-Condition Amino Acid Enrichment (BWM vs Control)",
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
    )
  ) &
  theme(legend.position = "bottom",
        legend.text = element_markdown(size = 11))

filepath <- file.path(args$outdir, "composite",
                      "EPA_barplot_composite")
save_plot(composite, filepath, width = 18, height = 6,
          format = args$format, dpi = args$dpi)

cat("  Saved composite plot\n")

# ============================================================
# Summary
# ============================================================

total <- plot_count + 1
cat("\n============================================\n")
cat("Done! Generated", total, "total plot files\n")
cat("Output directory:", args$outdir, "\n")
cat("Format:", args$format, "\n")
cat("============================================\n")
