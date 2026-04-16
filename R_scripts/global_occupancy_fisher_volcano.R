#!/usr/bin/env Rscript

# ============================================================
# Global Occupancy Fisher Volcano Plots
# Reads Fisher's exact test CSV and generates volcano plots
# for codon/AA occupancy differences.
# Used for both per-timepoint and within-condition-timepoint results.
# ============================================================

library(argparse)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(dplyr)

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(description = "Generate volcano plots for global occupancy Fisher results")

parser$add_argument("--input",
                    required = TRUE,
                    help = "Path to Fisher CSV (codon or AA level)")

parser$add_argument("--outdir",
                    default = "global_occupancy_fisher_output",
                    help = "Output directory for plots")

parser$add_argument("--level",
                    default = "aa",
                    choices = c("codon", "aa"),
                    help = "Analysis level: codon or aa")

parser$add_argument("--group-col",
                    default = "timepoint",
                    help = "Column to group by for individual/composite plots (timepoint or condition)")

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

level_label <- ifelse(args$level == "aa", "Amino Acid", "Codon")

# ============================================================
# Read and Prepare Data
# ============================================================

cat("Reading input:", args$input, "\n")
data <- read.csv(args$input, stringsAsFactors = FALSE)

group_col <- args$group_col
cat("  Rows:", nrow(data), "| Groups:", paste(unique(data[[group_col]]), collapse = ", "), "\n")

# Compute log2 odds ratio and -log10 adjusted p-value
data <- data |>
  mutate(
    log2_odds_ratio = log2(odds_ratio),
    neg_log10_p = -log10(p_adj)
  )

# Add classification and significance columns
if (args$level == "aa") {
  data <- data |> mutate(aa_class = AA_CLASS[unit])
} else {
  CODON2AA <- c(
    "GCT"="A","GCC"="A","GCA"="A","GCG"="A",
    "CGT"="R","CGC"="R","CGA"="R","CGG"="R","AGA"="R","AGG"="R",
    "AAT"="N","AAC"="N","GAT"="D","GAC"="D","TGT"="C","TGC"="C",
    "GAA"="E","GAG"="E","CAA"="Q","CAG"="Q",
    "GGT"="G","GGC"="G","GGA"="G","GGG"="G",
    "CAT"="H","CAC"="H","ATT"="I","ATC"="I","ATA"="I",
    "TTA"="L","TTG"="L","CTT"="L","CTC"="L","CTA"="L","CTG"="L",
    "AAA"="K","AAG"="K","ATG"="M","TTT"="F","TTC"="F",
    "CCT"="P","CCC"="P","CCA"="P","CCG"="P",
    "TCT"="S","TCC"="S","TCA"="S","TCG"="S","AGT"="S","AGC"="S",
    "ACT"="T","ACC"="T","ACA"="T","ACG"="T","TGG"="W",
    "TAT"="Y","TAC"="Y","GTT"="V","GTC"="V","GTA"="V","GTG"="V"
  )
  data <- data |> mutate(
    encoded_aa = CODON2AA[toupper(unit)],
    aa_class = AA_CLASS[encoded_aa]
  )
}

data <- data |>
  mutate(
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

cat("  Axis limits -- x:", round(x_lim, 2), "| y: [0,", round(y_max, 2), "]\n")

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

  sig_data <- plot_data |> filter(significant)

  p <- p +
    geom_text_repel(
      data = sig_data,
      aes(label = unit, color = aa_class),
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

group_values <- unique(data[[group_col]])
group_values <- group_values[order(as.numeric(gsub("[^0-9]", "", group_values)))]
plot_count <- 0

for (gv in group_values) {
  plot_data <- data |> filter(.data[[group_col]] == gv)

  group_label <- gsub("_", " ", gv)
  group_label <- gsub("day ", "Day ", group_label)
  title <- paste0(level_label, " Occupancy | ", group_label,
                  " (", args$comparison_label, ")")

  p <- make_volcano(
    plot_data,
    x_lim = x_lim,
    y_max = y_max,
    title = title,
    show_legend = TRUE
  )

  filepath <- file.path(args$outdir, "individual",
                        paste0(gv, "_volcano"))
  save_plot(p, filepath, width = 7, height = 6,
            format = args$format, dpi = args$dpi)
  plot_count <- plot_count + 1
}

cat("  Saved", plot_count, "individual plots\n")

# ============================================================
# Composite Plot
# ============================================================

cat("Generating composite plot...\n")

plot_list <- list()

for (gv in group_values) {
  plot_data <- data |> filter(.data[[group_col]] == gv)

  group_label <- gsub("_", " ", gv)
  group_label <- gsub("day ", "Day ", group_label)
  subtitle <- group_label

  p <- make_volcano(
    plot_data,
    x_lim = x_lim,
    y_max = y_max,
    title = subtitle,
    show_legend = FALSE
  )

  plot_list[[length(plot_list) + 1]] <- p
}

n_groups <- length(group_values)
composite <- wrap_plots(plot_list, ncol = n_groups, nrow = 1) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = paste0("Global ", level_label, " Occupancy Fisher's Test (",
                   args$comparison_label, ")"),
    theme = theme(
      plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
    )
  ) &
  theme(legend.position = "bottom")

filepath <- file.path(args$outdir, "composite",
                      paste0(args$level, "_fisher_composite"))
save_plot(composite, filepath, width = 6 * n_groups, height = 7,
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
cat("Comparison:", args$comparison_label, "\n")
cat("Format:", args$format, "\n")
cat("============================================\n")
