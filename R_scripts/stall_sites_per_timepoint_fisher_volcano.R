#!/usr/bin/env Rscript

# ============================================================
# Per-Timepoint Fisher Volcano Plots
# Reads per_timepoint_fisher.csv and generates volcano plots
# for amino acid enrichment at ribosome stall sites (BWM vs control).
# ============================================================

library(argparse)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(dplyr)

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(description = "Generate volcano plots for per-timepoint Fisher enrichment")

parser$add_argument("--input",
                    default = "results/stall_sites/enrichment/per_timepoint_fisher_aa.csv",
                    help = "Path to per_timepoint_fisher_{aa,codon}.csv (emit by stall_sites_non_consensus_stats.py)")

parser$add_argument("--outdir",
                    default = "results/stall_sites/plots/per_timepoint_fisher",
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

AA_CLASS <- c(
  "D" = "Acidic", "E" = "Acidic",
  "K" = "Basic", "R" = "Basic", "H" = "Basic",
  "A" = "Hydrophobic", "V" = "Hydrophobic", "I" = "Hydrophobic",
  "L" = "Hydrophobic", "M" = "Hydrophobic", "F" = "Hydrophobic",
  "W" = "Hydrophobic", "Y" = "Hydrophobic",
  "C" = "Polar", "N" = "Polar", "Q" = "Polar", "S" = "Polar", "T" = "Polar",
  "G" = "Neutral", "P" = "Neutral"
)

CLASS_COLORS <- c(
  "Acidic"      = "#E41A1C",
  "Basic"       = "#377EB8",
  "Hydrophobic" = "#4DAF4A",
  "Polar"       = "#984EA3",
  "Neutral"     = "#FF7F00"
)

SITE_LABELS <- c("E" = "E-site", "P" = "P-site", "A" = "A-site")

# ============================================================
# Read and Prepare Data
# ============================================================

cat("Reading input:", args$input, "\n")
data <- read.csv(args$input, stringsAsFactors = FALSE)

cat("  Rows:", nrow(data), "| Timepoints:", paste(unique(data$timepoint), collapse = ", "),
    "| Sites:", paste(unique(data$site), collapse = ", "), "\n")

# Compute log2 odds ratio and -log10 adjusted p-value
data <- data |>
  mutate(
    log2_odds_ratio = log2(odds_ratio),
    neg_log10_p = -log10(p_adj)
  )

# Add amino acid class and significance columns
data <- data |>
  mutate(
    aa_class = AA_CLASS[amino_acid],
    significant = p_adj < 0.05,
    significance = ifelse(significant, "Significant (FDR < 0.05)", "Not significant"),
    neg_log10_p = pmin(neg_log10_p, 50)
  )

# ============================================================
# Compute Uniform Axis Limits
# ============================================================

x_abs_max <- max(abs(data$log2_odds_ratio), na.rm = TRUE) * 1.1
x_lim <- c(-x_abs_max, x_abs_max)

y_max <- max(data$neg_log10_p, na.rm = TRUE) * 1.1

cat("  Axis limits — x:", round(x_lim, 2), "| y: [0,", round(y_max, 2), "]\n")

# ============================================================
# Volcano Plot Function
# ============================================================

make_volcano <- function(plot_data, x_lim, y_max, title,
                         show_legend = TRUE) {

  p <- ggplot(plot_data,
              aes(x = log2_odds_ratio,
                  y = neg_log10_p)) +

    geom_point(aes(color = aa_class, shape = significance),
               alpha = 0.8, size = 2.5) +

    geom_vline(xintercept = c(-0.5, 0.5),
               linetype = "dashed", color = "gray50", alpha = 0.5) +

    geom_hline(yintercept = -log10(0.05),
               linetype = "dashed", color = "gray50", alpha = 0.5) +

    geom_vline(xintercept = 0,
               linetype = "dotted", color = "black", alpha = 0.3)

  # Labels for significant amino acids
  sig_data <- plot_data |> filter(significant)

  p <- p +
    geom_text_repel(
      data = sig_data,
      aes(label = amino_acid, color = aa_class),
      size = 3.5,
      box.padding = 0.5,
      point.padding = 0.3,
      force = 3,
      max.overlaps = 15,
      segment.color = "gray50",
      segment.size = 0.2,
      min.segment.length = 0,
      show.legend = FALSE
    ) +

    scale_color_manual(values = CLASS_COLORS, name = NULL) +

    scale_shape_manual(
      name = NULL,
      values = c("Significant (FDR < 0.05)" = 17, "Not significant" = 16)
    ) +

    coord_cartesian(xlim = x_lim, ylim = c(0, y_max)) +

    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = if (show_legend) "bottom" else "none",
      legend.box = "vertical",
      legend.box.just = "center",
      legend.box.spacing = unit(0.1, "cm"),
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10),
      legend.background = element_rect(fill = "white", color = NA),
      legend.key.size = unit(0.5, "cm"),
      legend.spacing.y = unit(0.05, "cm"),
      legend.margin = margin(t = 0, b = 0, unit = "cm"),
      axis.title = element_text(size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      plot.margin = margin(1, 1, 0.5, 1, "cm")
    ) +

    labs(
      title = title,
      x = expression(bold("Log"[2] ~ "(Odds Ratio)")),
      y = expression(bold("-Log"[10] ~ "(FDR)"))
    ) +

    guides(
      color = guide_legend(order = 1),
      shape = guide_legend(order = 2)
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

timepoints <- c("day_0", "day_5", "day_10")
sites <- c("E", "P", "A")
plot_count <- 0

for (tp in timepoints) {
  for (st in sites) {

    plot_data <- data |> filter(timepoint == tp, site == st)

    day_label <- gsub("_", " ", tp)
    day_label <- gsub("day ", "Day ", day_label)
    title <- paste0(SITE_LABELS[st], " | ", day_label)

    p <- make_volcano(
      plot_data,
      x_lim = x_lim,
      y_max = y_max,
      title = title,
      show_legend = TRUE
    )

    filepath <- file.path(args$outdir, "individual",
                          paste0(tp, "_", st, "_volcano"))
    save_plot(p, filepath, width = 7, height = 6,
              format = args$format, dpi = args$dpi)
    plot_count <- plot_count + 1
  }
}

cat("  Saved", plot_count, "individual plots\n")

# ============================================================
# Composite Plot: 3x3 Grid (Rows = Timepoints, Cols = Sites)
# ============================================================

cat("Generating composite plot...\n")

plot_list <- list()

for (tp in timepoints) {
  for (st in sites) {

    plot_data <- data |> filter(timepoint == tp, site == st)

    day_label <- gsub("_", " ", tp)
    day_label <- gsub("day ", "Day ", day_label)
    subtitle <- paste0(SITE_LABELS[st], " | ", day_label)

    p <- make_volcano(
      plot_data,
      x_lim = x_lim,
      y_max = y_max,
      title = subtitle,
      show_legend = FALSE
    )

    plot_list[[length(plot_list) + 1]] <- p
  }
}

# Assemble 3x3 grid (rows = timepoints, cols = sites)
composite <- wrap_plots(plot_list, ncol = 3, nrow = 3) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Per-Timepoint Fisher: Amino Acid Enrichment at Stall Sites (BWM vs Control)",
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
    )
  ) &
  theme(legend.position = "bottom")

filepath <- file.path(args$outdir, "composite",
                      "per_timepoint_fisher_composite")
save_plot(composite, filepath, width = 18, height = 16,
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
