#!/usr/bin/env Rscript

# ============================================================
# Within-Condition Enrichment Volcano Plots
# Reads pre-computed enrichment CSV and generates volcano plots
# for amino acid enrichment at ribosome stall sites.
# ============================================================

library(argparse)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(dplyr)

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(description = "Generate volcano plots for within-condition amino acid enrichment")

parser$add_argument("--input",
                    default = "results/stall_sites/enrichment/within_condition_enrichment_aa.csv",
                    help = "Path to within_condition_enrichment_{aa,codon}.csv (emit by stall_sites_non_consensus_stats.py)")

parser$add_argument("--outdir",
                    default = "results/stall_sites/plots/within_condition",
                    help = "Output directory for plots")

parser$add_argument("--show-ci",
                    action = "store_true",
                    default = FALSE,
                    help = "Show confidence interval error bars on plots")

parser$add_argument("--enrichment-type",
                    default = "both",
                    choices = c("unweighted", "weighted", "both"),
                    help = "Which enrichment type(s) to plot: unweighted, weighted, or both")

parser$add_argument("--format",
                    default = "both",
                    choices = c("pdf", "png", "both"),
                    help = "Output format: pdf, png, or both")

parser$add_argument("--dpi",
                    type = "integer",
                    default = 300L,
                    help = "DPI for PNG output")

parser$add_argument("--y-cap",
                    type = "double",
                    default = NULL,
                    help = "Cap -log10(p_adj) values at this maximum (clamps extreme values to compress the y-axis)")

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

cat("  Rows:", nrow(data), "| Groups:", length(unique(data$group)),
    "| Sites:", paste(unique(data$site), collapse = ", "), "\n")

# Compute confidence intervals (Beta distribution, Jeffreys prior)
data <- data |>
  mutate(
    ci_lower_prop = qbeta(0.025, stall_count + 0.5, total_n - stall_count + 0.5),
    ci_upper_prop = qbeta(0.975, stall_count + 0.5, total_n - stall_count + 0.5),
    ci_lower_log2 = log2(ci_lower_prop / bg_freq),
    ci_upper_log2 = log2(ci_upper_prop / bg_freq),
    ci_lower_weighted = stall_freq * ci_lower_log2,
    ci_upper_weighted = stall_freq * ci_upper_log2
  )

# Add amino acid class and significance columns
data <- data |>
  mutate(
    aa_class = AA_CLASS[amino_acid],
    significant = p_adj < 0.05,
    significance = ifelse(significant, "Significant (FDR < 0.05)", "Not significant"),
    neg_log10_p = -log10(p_adj)
  )

# Cap extreme -log10 p-values for display (if --y-cap provided)
if (!is.null(args$y_cap)) {
  y_cap <- args$y_cap
  capped_n <- sum(data$neg_log10_p > y_cap, na.rm = TRUE)
  data <- data |>
    mutate(neg_log10_p = pmin(neg_log10_p, y_cap))
  cat("  Y-axis cap:", y_cap, "| Capped", capped_n, "points\n")
}

# ============================================================
# Compute Uniform Axis Limits
# ============================================================

# Unweighted x-axis
x_vals_uw <- c(data$log2_enrichment, data$ci_lower_log2, data$ci_upper_log2)
x_abs_max_uw <- max(abs(x_vals_uw), na.rm = TRUE) * 1.1
x_lim_uw <- c(-x_abs_max_uw, x_abs_max_uw)

# Weighted x-axis
x_vals_w <- c(data$weighted_log2_enrichment, data$ci_lower_weighted, data$ci_upper_weighted)
x_abs_max_w <- max(abs(x_vals_w), na.rm = TRUE) * 1.1
x_lim_w <- c(-x_abs_max_w, x_abs_max_w)

# Shared y-axis
y_max <- max(data$neg_log10_p, na.rm = TRUE) * 1.1

cat("  Axis limits — unweighted x:", round(x_lim_uw, 2),
    "| weighted x:", round(x_lim_w, 2),
    "| y: [0,", round(y_max, 2), "]\n")

# ============================================================
# Volcano Plot Function
# ============================================================

make_volcano <- function(plot_data, x_col, ci_lower_col, ci_upper_col,
                         x_lim, y_max, title,
                         x_label = "Log2 Enrichment",
                         show_ci = FALSE, show_legend = TRUE) {

  p <- ggplot(plot_data,
              aes(x = .data[[x_col]],
                  y = neg_log10_p)) +

    geom_point(aes(color = aa_class, shape = significance),
               alpha = 0.8, size = 2.5) +

    geom_vline(xintercept = c(-0.5, 0.5),
               linetype = "dashed", color = "gray50", alpha = 0.5) +

    geom_hline(yintercept = -log10(0.05),
               linetype = "dashed", color = "gray50", alpha = 0.5) +

    geom_vline(xintercept = 0,
               linetype = "dotted", color = "black", alpha = 0.3)

  # Confidence interval error bars (optional)
  if (show_ci) {
    p <- p +
      geom_errorbarh(
        aes(xmin = .data[[ci_lower_col]],
            xmax = .data[[ci_upper_col]]),
        height = 0.1, alpha = 0.4
      )
  }

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
      x = x_label,
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
# Determine Enrichment Types to Plot
# ============================================================

enrichment_configs <- list()

if (args$enrichment_type %in% c("unweighted", "both")) {
  enrichment_configs[["unweighted"]] <- list(
    x_col = "log2_enrichment",
    ci_lower_col = "ci_lower_log2",
    ci_upper_col = "ci_upper_log2",
    x_lim = x_lim_uw,
    x_label = "Log2 Enrichment",
    label = "Unweighted"
  )
}

if (args$enrichment_type %in% c("weighted", "both")) {
  enrichment_configs[["weighted"]] <- list(
    x_col = "weighted_log2_enrichment",
    ci_lower_col = "ci_lower_weighted",
    ci_upper_col = "ci_upper_weighted",
    x_lim = x_lim_w,
    x_label = "Weighted Log2 Enrichment",
    label = "Weighted"
  )
}

# ============================================================
# Create Output Directories
# ============================================================

for (etype in names(enrichment_configs)) {
  dir.create(file.path(args$outdir, "individual", etype),
             recursive = TRUE, showWarnings = FALSE)
  dir.create(file.path(args$outdir, "composite", etype),
             recursive = TRUE, showWarnings = FALSE)
}

# ============================================================
# Generate Individual Plots
# ============================================================

cat("\nGenerating individual plots...\n")

groups <- sort(unique(data$group))
sites <- c("E", "P", "A")
plot_count <- 0

for (etype in names(enrichment_configs)) {
  cfg <- enrichment_configs[[etype]]

  for (grp in groups) {
    for (st in sites) {

      plot_data <- data |> filter(group == grp, site == st)

      # Format title: "E-site Enrichment | BWM Day 0"
      day_label <- gsub("_", " ", gsub(".*_(day)", "\\1", grp))
      day_label <- gsub("day ", "Day ", day_label)
      cond_label <- gsub("_day.*", "", grp)
      title <- paste0(SITE_LABELS[st], " ", cfg$label, " Enrichment | ",
                      cond_label, " ", day_label)

      p <- make_volcano(
        plot_data,
        x_col = cfg$x_col,
        ci_lower_col = cfg$ci_lower_col,
        ci_upper_col = cfg$ci_upper_col,
        x_lim = cfg$x_lim,
        y_max = y_max,
        title = title,
        x_label = cfg$x_label,
        show_ci = args$show_ci,
        show_legend = TRUE
      )

      filepath <- file.path(args$outdir, "individual", etype,
                            paste0(grp, "_", st, "_volcano"))
      save_plot(p, filepath, width = 7, height = 6,
                format = args$format, dpi = args$dpi)
      plot_count <- plot_count + 1
    }
  }
}

cat("  Saved", plot_count, "individual plots\n")

# ============================================================
# Composite Plots: Grouped by Condition
# ============================================================

cat("Generating composite plots by condition...\n")

conditions <- sort(unique(data$condition))
days <- c("day_0", "day_5", "day_10")
composite_count <- 0

for (etype in names(enrichment_configs)) {
  cfg <- enrichment_configs[[etype]]

  for (cond in conditions) {

    plot_list <- list()

    for (d in days) {
      grp <- paste0(cond, "_", d)

      for (st in sites) {
        plot_data <- data |> filter(group == grp, site == st)

        day_label <- gsub("_", " ", d)
        day_label <- gsub("day ", "Day ", day_label)
        subtitle <- paste0(SITE_LABELS[st], " | ", day_label)

        p <- make_volcano(
          plot_data,
          x_col = cfg$x_col,
          ci_lower_col = cfg$ci_lower_col,
          ci_upper_col = cfg$ci_upper_col,
          x_lim = cfg$x_lim,
          y_max = y_max,
          title = subtitle,
          x_label = cfg$x_label,
          show_ci = args$show_ci,
          show_legend = FALSE
        )

        plot_list[[length(plot_list) + 1]] <- p
      }
    }

    # Assemble 3x3 grid (rows = days, cols = sites)
    composite <- wrap_plots(plot_list, ncol = 3, nrow = 3) +
      plot_layout(guides = "collect") +
      plot_annotation(
        title = paste0(cond, ": ", cfg$label, " Amino Acid Enrichment at Stall Sites"),
        theme = theme(
          plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
        )
      ) &
      theme(legend.position = "bottom")

    filepath <- file.path(args$outdir, "composite", etype,
                          paste0(cond, "_volcano_grid"))
    save_plot(composite, filepath, width = 18, height = 16,
              format = args$format, dpi = args$dpi)
    composite_count <- composite_count + 1
  }
}

cat("  Saved", composite_count, "condition composite plots\n")

# ============================================================
# Composite Plots: Grouped by Day
# ============================================================

cat("Generating composite plots by day...\n")
composite_day_count <- 0

for (etype in names(enrichment_configs)) {
  cfg <- enrichment_configs[[etype]]

  for (d in days) {

    plot_list <- list()

    for (cond in conditions) {
      grp <- paste0(cond, "_", d)

      for (st in sites) {
        plot_data <- data |> filter(group == grp, site == st)

        subtitle <- paste0(SITE_LABELS[st], " | ", cond)

        p <- make_volcano(
          plot_data,
          x_col = cfg$x_col,
          ci_lower_col = cfg$ci_lower_col,
          ci_upper_col = cfg$ci_upper_col,
          x_lim = cfg$x_lim,
          y_max = y_max,
          title = subtitle,
          x_label = cfg$x_label,
          show_ci = args$show_ci,
          show_legend = FALSE
        )

        plot_list[[length(plot_list) + 1]] <- p
      }
    }

    day_label <- gsub("_", " ", d)
    day_label <- gsub("day ", "Day ", day_label)

    # Assemble 2x3 grid (rows = conditions, cols = sites)
    composite <- wrap_plots(plot_list, ncol = 3, nrow = 2) +
      plot_layout(guides = "collect") +
      plot_annotation(
        title = paste0(day_label, ": ", cfg$label, " Amino Acid Enrichment at Stall Sites"),
        theme = theme(
          plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
        )
      ) &
      theme(legend.position = "bottom")

    filepath <- file.path(args$outdir, "composite", etype,
                          paste0(d, "_volcano_grid"))
    save_plot(composite, filepath, width = 18, height = 11,
              format = args$format, dpi = args$dpi)
    composite_day_count <- composite_day_count + 1
  }
}

cat("  Saved", composite_day_count, "day composite plots\n")

# ============================================================
# Summary
# ============================================================

total <- plot_count + composite_count + composite_day_count
cat("\n============================================\n")
cat("Done! Generated", total, "total plot files\n")
cat("Output directory:", args$outdir, "\n")
cat("Enrichment type(s):", args$enrichment_type, "\n")
cat("Format:", args$format, "\n")
cat("Confidence intervals:", ifelse(args$show_ci, "shown", "hidden"), "\n")
cat("============================================\n")
