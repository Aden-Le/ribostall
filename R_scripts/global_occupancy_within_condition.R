#!/usr/bin/env Rscript

# ============================================================
# Global Occupancy Within-Condition Bar Plots
# Reads within-condition binomial test CSV and generates sorted
# bar charts for codon/AA occupancy enrichment vs transcriptome
# background, with significance stars and Enriched/Depleted colouring.
# ============================================================

library(argparse)
library(ggplot2)
library(ggtext)
library(patchwork)
library(dplyr)

# ============================================================
# Test Input
# ============================================================

# INPUT_DIR <- "C:/Users/Aden Le/Documents/GitHub/ribostall/global_occupancy_results/analysis"
# OUTPUT_DIR <- "C:/Users/Aden Le/Documents/GitHub/ribostall/global_occupancy_results"
# args <- list(level = "aa",
#              input = file.path(INPUT_DIR, "aa_within_condition_binomial.csv"),
#              outdir = file.path(OUTPUT_DIR, "within_condition_output"),
#              enrichment_type = "unweighted",
#              format = "png",
#              dpi = 300L
#              )

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(description = "Generate bar plots for global occupancy within-condition enrichment")

parser$add_argument("--input",
                    required = TRUE,
                    help = "Path to within-condition binomial CSV (codon or AA level)")

parser$add_argument("--outdir",
                    default = "global_occupancy_within_condition_output",
                    help = "Output directory for plots")

parser$add_argument("--level",
                    default = "aa",
                    choices = c("codon", "aa"),
                    help = "Analysis level: codon or aa")

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
                    help = "Clamp y-axis to ±this value")

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

cat("  Rows:", nrow(data), "| Groups:", length(unique(data$group)), "\n")

data <- data |>
  mutate(
    sig_label = case_when(
      p_adj < 0.001 ~ "***",
      p_adj < 0.01  ~ "**",
      p_adj < 0.05  ~ "*",
      TRUE          ~ ""
    )
  )

# ============================================================
# Compute Uniform Axis Limits
# ============================================================

compute_limits <- function(values, y_cap = NULL) {
  # 8% * the difference between max and min of the values (1.2 - 0.8 = 0.4 * 0.08 = padding)
  padding <- diff(range(values, na.rm = TRUE)) * 0.08
  # Minimum 0.02 value to prevent 0 or near zero
  padding <- max(padding, 0.02)
  # The limits are the max and min +/- the padding
  y_min <- min(values, na.rm = TRUE) - padding
  y_max <- max(values, na.rm = TRUE) + padding
  if (!is.null(y_cap)) {
    # If a cap is provided change the limits so they don't exceed cap if greater or less than
    y_min <- max(y_min, -abs(y_cap))
    y_max <- min(y_max,  abs(y_cap))
  }
  c(y_min, y_max)
}

# Computes the limits for weighted and unweighted values
y_lim_uw <- compute_limits(data$log2_enrichment, args$y_cap)
y_lim_w <- compute_limits(data$weighted_log2,   args$y_cap)
# Gets the breaks
y_breaks_uw <- pretty(y_lim_uw, n = 6)
y_breaks_w  <- pretty(y_lim_w,  n = 6)

cat("  Axis limits -- unweighted y:", round(y_lim_uw, 3),
    "| weighted y:", round(y_lim_w, 4), "\n")

# ============================================================
# Bar Plot Function
# ============================================================

# If x_col = "log2_enrichment", then:
# .data[[x_col]]  ==  plot_data$log2_enrichment

make_barplot <- function(plot_data, x_col, y_lim, y_breaks,
                         title, y_label, show_legend) {
  # Padding for the stars to appear above graph
  star_padding <- diff(y_lim) * 0.03
  
  
  plot_data <- plot_data |>
    # Arranges data by highest to lowest value, where x_col is like log2_enrichment
    arrange(desc(.data[[x_col]])) |>
    mutate(
      # Locks order to prevent Alphabetical ordering
      unit     = factor(unit, levels = unit),
      # Tags as depleted or not depleted
      bar_fill = ifelse(.data[[x_col]] >= 0, "Enriched", "Depleted"),
      # Adds star padding 
      star_y   = ifelse(.data[[x_col]] >= 0, 
                        .data[[x_col]] + star_padding,
                        .data[[x_col]] - star_padding)
    )
  
  p <- ggplot(plot_data, aes(x = unit, y = .data[[x_col]], fill = bar_fill)) +

    geom_col(width = 0.8, colour = "white", linewidth = 0.3) +

    geom_hline(yintercept = 0, linewidth = 0.4, colour = "grey40") +

    geom_text(
      aes(y = star_y, label = sig_label),
      size   = 4,
      vjust  = 0.5,
      colour = "grey20",
      family = "sans"
    ) +

    scale_fill_manual(
      values = PAL,
      name   = NULL,
      limits = c("Enriched", "Depleted"),
      labels = c(
        "Enriched" = "Enriched (log<sub>2</sub> enrichment \u2265 0)",
        "Depleted"  = "Depleted (log<sub>2</sub> enrichment < 0)"
      )
    ) +

    scale_y_continuous(
      limits = y_lim,
      breaks = y_breaks,
      expand = expansion(mult = c(0, 0))
    ) +

    labs(
      title = title,
      x     = level_label,
      y     = y_label
    ) +

    theme_classic(base_size = 13) +
    theme(
      plot.title         = element_text(face = "bold", size = 14, hjust = 0.5),
      axis.title         = element_text(face = "bold"),
      axis.text          = element_text(colour = "grey20"),
      axis.text.x        = element_text(
                             angle = if (args$level == "codon") 45 else 0,
                             hjust = if (args$level == "codon") 1  else 0.5,
                             size  = if (args$level == "codon") 7  else 11
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
# Helper: Save Plot (Saves the plot as either a pdf, png, or both)
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
# Determine Enrichment Types to Plot (Controls what data is ploted based on Unweighted vs Weighted Flags)
# ============================================================

enrichment_configs <- list()

# This is the configs for unweighted
if (args$enrichment_type %in% c("unweighted", "both")) {
  enrichment_configs[["unweighted"]] <- list(
    x_col    = "log2_enrichment",
    y_lim    = y_lim_uw,
    y_breaks = y_breaks_uw,
    y_label  = bquote(bold(log[2]~"Enrichment")),
    label    = "Unweighted"
  )
}

# This is the configs for weighted
if (args$enrichment_type %in% c("weighted", "both")) {
  enrichment_configs[["weighted"]] <- list(
    x_col    = "weighted_log2",
    y_lim    = y_lim_w,
    y_breaks = y_breaks_w,
    y_label  = bquote(bold("Weighted"~log[2]~"Enrichment")),
    label    = "Weighted"
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
# Plot Dimensions
# ============================================================

panel_w <- if (args$level == "aa") 9 else 16
panel_h <- 5

# ============================================================
# Generate Individual Plots
# ============================================================

cat("\nGenerating individual plots...\n")
# Gets the conditions
groups <- sort(unique(data$group))
plot_count <- 0

# For unweighted, weighted, or both
for (etype in names(enrichment_configs)) {
  # The data
  cfg <- enrichment_configs[[etype]]
  
  # For each condition
  for (grp in groups) {
    plot_data <- data |> filter(group == grp)

    day_str    <- gsub(".*_(day)", "\\1", grp)
    day_label  <- gsub("day ", "Day ", gsub("_", " ", day_str))
    cond_label <- gsub("_day.*", "", grp)
    title <- paste0(level_label, " Occupancy ", cfg$label,
                    " | ", cond_label, " ", day_label)

    p <- make_barplot(plot_data,
                      x_col       = cfg$x_col,
                      y_lim       = cfg$y_lim,
                      y_breaks    = cfg$y_breaks,
                      title       = title,
                      y_label     = cfg$y_label,
                      show_legend = TRUE)

    filepath <- file.path(args$outdir, "individual", etype,
                          paste0(grp, "_barplot"))
    save_plot(p, filepath, width = panel_w, height = panel_h + 1,
              format = args$format, dpi = args$dpi)
    plot_count <- plot_count + 1
  }
}

cat("  Saved", plot_count, "individual plots\n")

# ============================================================
# Composite Plots: Grouped by Condition
# ============================================================

cat("Generating composite plots by condition...\n")

# Both BWM & Control
conditions      <- sort(unique(data$condition))
# day_0, day_5, day_10
days            <- unique(data$timepoint)[order(as.numeric(gsub("\\D", "", unique(data$timepoint))))]
composite_count <- 0

# For Unweighted or Weighted 
for (etype in names(enrichment_configs)) {
  # Get data
  cfg <- enrichment_configs[[etype]]
  
  # For each condition
  for (cond in conditions) {
    plot_list <- list()
    
    # For each day
    for (d in days) {
      # Ex: BWM_day_0 
      grp       <- paste0(cond, "_", d)
      plot_data <- data |> filter(group == grp)
      day_label <- gsub("day ", "Day ", gsub("_", " ", d))
      
      # each plot will have the identifier, 1, 2, 3
      plot_list[[length(plot_list) + 1]] <- make_barplot(
        plot_data,
        x_col       = cfg$x_col,
        y_lim       = cfg$y_lim,
        y_breaks    = cfg$y_breaks,
        title       = day_label,
        y_label     = cfg$y_label,
        show_legend = FALSE
      )
    }

    n_days <- length(days)
    # 3 Columns 1 Day composite
    composite <- wrap_plots(plot_list, ncol = n_days, nrow = 1) +
      plot_layout(guides = "collect") +
      plot_annotation(
        title = paste0(cond, ": ", cfg$label, " Global ",
                       level_label, " Occupancy Enrichment"),
        theme = theme(
          plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
        )
      ) &
      theme(legend.position = "bottom",
            legend.text = element_markdown(size = 11))

    filepath <- file.path(args$outdir, "composite", etype,
                          paste0(cond, "_barplot_grid"))
    save_plot(composite, filepath,
              width = panel_w * n_days, height = panel_h + 1.5,
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

# For un-weighted or weighted
for (etype in names(enrichment_configs)) {
  # Gets the data
  cfg <- enrichment_configs[[etype]]

  # For each day
  for (d in days) {
    plot_list <- list()
    
    # For each condition
    for (cond in conditions) {
      # BWM_day_0
      grp       <- paste0(cond, "_", d)
      plot_data <- data |> filter(group == grp)

      plot_list[[length(plot_list) + 1]] <- make_barplot(
        plot_data,
        x_col       = cfg$x_col,
        y_lim       = cfg$y_lim,
        y_breaks    = cfg$y_breaks,
        title       = cond,
        y_label     = cfg$y_label,
        show_legend = FALSE
      )
    }

    day_label <- gsub("day ", "Day ", gsub("_", " ", d))
    n_conds   <- length(conditions)
    
    # Composite by 2 columns
    composite <- wrap_plots(plot_list, ncol = n_conds, nrow = 1) +
      plot_layout(guides = "collect") +
      plot_annotation(
        title = paste0(day_label, ": ", cfg$label, " Global ",
                       level_label, " Occupancy Enrichment"),
        theme = theme(
          plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
        )
      ) &
      theme(legend.position = "bottom",
            legend.text = element_markdown(size = 11))

    filepath <- file.path(args$outdir, "composite", etype,
                          paste0(d, "_barplot_grid"))
    save_plot(composite, filepath,
              width = panel_w * n_conds, height = panel_h + 1.5,
              format = args$format, dpi = args$dpi)
    composite_day_count <- composite_day_count + 1
  }
}

cat("  Saved", composite_day_count, "day composite plots\n")

# ============================================================
# Mega Composite: All Groups (rows = conditions, cols = days)
# ============================================================

cat("Generating mega composite (all groups)...\n")
mega_count <- 0

for (etype in names(enrichment_configs)) {
  cfg <- enrichment_configs[[etype]]
  plot_list <- list()

  for (cond in conditions) {
    for (d in days) {
      grp       <- paste0(cond, "_", d)
      plot_data <- data |> filter(group == grp)
      day_label <- gsub("day ", "Day ", gsub("_", " ", d))

      plot_list[[length(plot_list) + 1]] <- make_barplot(
        plot_data,
        x_col       = cfg$x_col,
        y_lim       = cfg$y_lim,
        y_breaks    = cfg$y_breaks,
        title       = paste0(cond, " \u2013 ", day_label),
        y_label     = cfg$y_label,
        show_legend = FALSE
      )
    }
  }

  mega <- wrap_plots(plot_list, ncol = length(days), nrow = length(conditions)) +
    plot_layout(guides = "collect") +
    plot_annotation(
      title = paste0(cfg$label, " Global ", level_label, " Occupancy Enrichment \u2013 All Groups"),
      theme = theme(
        plot.title = element_text(hjust = 0.5, size = 20, face = "bold")
      )
    ) &
    theme(legend.position = "bottom",
          legend.text = element_markdown(size = 11))

  filepath <- file.path(args$outdir, "composite", etype, "all_groups_barplot_grid")
  save_plot(mega, filepath,
            width  = panel_w * length(days),
            height = (panel_h + 1) * length(conditions),
            format = args$format, dpi = args$dpi)
  mega_count <- mega_count + 1
}
cat("  Saved", mega_count, "mega composite plots\n")

# ============================================================
# Summary
# ============================================================

total <- plot_count + composite_count + composite_day_count + mega_count
cat("\n============================================\n")
cat("Done! Generated", total, "total plot files\n")
cat("Output directory:", args$outdir, "\n")
cat("Level:", args$level, "\n")
cat("Enrichment type(s):", args$enrichment_type, "\n")
cat("Format:", args$format, "\n")
cat("============================================\n")
