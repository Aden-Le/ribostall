#!/usr/bin/env Rscript

# ============================================================
# Within-Condition Enrichment Volcano Plots (unified)
# Reads pre-computed within-condition enrichment CSV and generates
# volcano plots for codon/AA enrichment vs background frequency.
#
# Handles both datasets:
#   - stall_sites (within_condition_binomial_{aa,codon}.csv)
#   - global_occupancy (aa_within_condition_binomial.csv,
#                       codon_within_condition_binomial.csv)
#
# Schema expected: site, group, condition, timepoint,
# {amino_acid|codon}, observed_count, total_n, observed_freq,
# bg_freq, log2_enrichment, weighted_log2_enrichment, p_value, p_adj.
# ============================================================

library(argparse)
library(ggplot2)
library(ggrepel)
library(patchwork)
library(dplyr)

# ============================================================
# Argument Parsing
# ============================================================

parser <- ArgumentParser(description = "Generate volcano plots for within-condition enrichment")

parser$add_argument("--input",
                    required = TRUE,
                    help = "Path to within-condition enrichment CSV (codon or AA level)")

parser$add_argument("--outdir",
                    default = "within_condition_volcano_output",
                    help = "Output directory for plots")

parser$add_argument("--level",
                    default = "aa",
                    choices = c("codon", "aa"),
                    help = "Analysis level: codon or aa")

parser$add_argument("--show-ci",
                    action = "store_true",
                    default = FALSE,
                    help = "Show Beta-Jeffreys confidence interval error bars on plots")

parser$add_argument("--mega-composite",
                    action = "store_true",
                    default = FALSE,
                    help = "Also emit an all-groups composite (rows = condition x timepoint, cols = sites)")

parser$add_argument("--flat-design",
                    action = "store_true",
                    default = FALSE,
                    help = "Flat control-vs-treatment input with no timepoint dimension (group == condition, timepoint == condition). Composites are built per group (rows = group, cols = sites) instead of condition x timepoint; the by-day composite is skipped. Use for the consensus stall-site pipeline.")

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
                    default = 50,
                    help = "Cap -log10(p_adj) values at this maximum (clamps extreme values to compress the y-axis). Default 50; pass a smaller value to compress further. Cap is required because p_adj=0 yields -log10=Inf and breaks plotting.")

args <- parser$parse_args()

feature_col <- ifelse(args$level == "aa", "amino_acid", "codon")

# ============================================================
# Constants
# ============================================================

# AA_CLASS, CLASS_COLORS, CODON2AA, SITE_LABELS — shared verbatim
# with the other R_scripts.
source("R_scripts/aa_constants.R")

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

cat("  Rows:", nrow(data), "| Groups:", length(unique(data$group)),
    "| Sites:", paste(unique(data$site), collapse = ", "), "\n")

# Compute Beta-Jeffreys CI on the underlying proportion (observed_count / total_n)
# and propagate through the log2 enrichment transform.
data <- data |>
  mutate(
    ci_lower_prop      = qbeta(0.025, observed_count + 0.5, total_n - observed_count + 0.5),
    ci_upper_prop      = qbeta(0.975, observed_count + 0.5, total_n - observed_count + 0.5),
    ci_lower_log2      = log2(ci_lower_prop / bg_freq),
    ci_upper_log2      = log2(ci_upper_prop / bg_freq),
    ci_lower_weighted  = observed_freq * ci_lower_log2,
    ci_upper_weighted  = observed_freq * ci_upper_log2
  )

# Add classification (always via AA_CLASS; for codon level, decode first).
if (args$level == "aa") {
  data <- data |> mutate(aa_class = AA_CLASS[.data[[feature_col]]])
} else {
  data <- data |>
    mutate(
      encoded_aa = CODON2AA[toupper(.data[[feature_col]])],
      aa_class   = ifelse(encoded_aa == "*", "Stop", AA_CLASS[encoded_aa])
    )
}

data <- data |>
  mutate(
    significant = p_adj < 0.05,
    significance = ifelse(significant, "Significant (FDR < 0.05)", "Not significant"),
    neg_log10_p = -log10(p_adj)
  )

# Y-cap on -log10(p_adj). Required because p_adj=0 (saturated FDR) yields
# Inf, which propagates into axis limits and breaks ggrepel.
y_cap <- args$y_cap
capped_n <- sum(data$neg_log10_p > y_cap, na.rm = TRUE)
data <- data |>
  mutate(neg_log10_p = pmin(neg_log10_p, y_cap))
cat("  Y-axis cap:", y_cap, "| Capped", capped_n, "points\n")

# Cap log2 enrichment values to keep ±Inf points on-canvas (mirrors the
# defensive cap used in between_group_volcano.R for sparse codon data).
data <- data |>
  mutate(
    log2_enrichment           = pmin(pmax(log2_enrichment,           -10), 10),
    weighted_log2_enrichment  = pmin(pmax(weighted_log2_enrichment,  -10), 10),
    ci_lower_log2             = pmin(pmax(ci_lower_log2,             -10), 10),
    ci_upper_log2             = pmin(pmax(ci_upper_log2,             -10), 10),
    ci_lower_weighted         = pmin(pmax(ci_lower_weighted,         -10), 10),
    ci_upper_weighted         = pmin(pmax(ci_upper_weighted,         -10), 10)
  )

# ============================================================
# Compute Uniform Axis Limits
# ============================================================

# Unweighted x-axis includes CI bounds so error bars never run off-canvas
x_vals_uw <- c(data$log2_enrichment, data$ci_lower_log2, data$ci_upper_log2)
x_abs_max_uw <- max(abs(x_vals_uw), na.rm = TRUE) * 1.1
x_lim_uw <- c(-x_abs_max_uw, x_abs_max_uw)

# Weighted x-axis
x_vals_w <- c(data$weighted_log2_enrichment, data$ci_lower_weighted, data$ci_upper_weighted)
x_abs_max_w <- max(abs(x_vals_w), na.rm = TRUE) * 1.1
x_lim_w <- c(-x_abs_max_w, x_abs_max_w)

# Shared y-axis
y_max <- max(data$neg_log10_p, na.rm = TRUE) * 1.1

cat("  Axis limits -- unweighted x:", round(x_lim_uw, 2),
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

  # Labels for significant features
  sig_data <- plot_data |> filter(significant)

  p <- p +
    geom_text_repel(
      data = sig_data,
      aes(label = .data[[feature_col]], color = aa_class),
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
      plot.title         = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position    = if (show_legend) "bottom" else "none",
      legend.box         = "vertical",
      legend.box.just    = "center",
      legend.box.spacing = unit(0.1, "cm"),
      legend.title       = element_text(size = 11, face = "bold"),
      legend.text        = element_text(size = 10),
      legend.background  = element_rect(fill = "white", color = NA),
      legend.key.size    = unit(0.5, "cm"),
      legend.spacing.y   = unit(0.05, "cm"),
      legend.margin      = margin(t = 0, b = 0, unit = "cm"),
      axis.title         = element_text(size = 12, face = "bold"),
      axis.text          = element_text(size = 10),
      plot.margin        = margin(1, 1, 0.5, 1, "cm")
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
# Determine Enrichment Types to Plot
# ============================================================

enrichment_configs <- list()

if (args$enrichment_type %in% c("unweighted", "both")) {
  enrichment_configs[["unweighted"]] <- list(
    x_col        = "log2_enrichment",
    ci_lower_col = "ci_lower_log2",
    ci_upper_col = "ci_upper_log2",
    x_lim        = x_lim_uw,
    x_label      = "Log2 Enrichment",
    label        = "Unweighted"
  )
}

if (args$enrichment_type %in% c("weighted", "both")) {
  enrichment_configs[["weighted"]] <- list(
    x_col        = "weighted_log2_enrichment",
    ci_lower_col = "ci_lower_weighted",
    ci_upper_col = "ci_upper_weighted",
    x_lim        = x_lim_w,
    x_label      = "Weighted Log2 Enrichment",
    label        = "Weighted"
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
sites <- c("A", "P", "E")
plot_count <- 0

for (etype in names(enrichment_configs)) {
  cfg <- enrichment_configs[[etype]]

  for (grp in groups) {
    for (st in sites) {

      plot_data <- data |> filter(group == grp, site == st)

      if (args$flat_design) {
        # No timepoint dimension: label by the group (condition) alone.
        group_label <- grp
      } else {
        day_label <- gsub("_", " ", gsub(".*_(day)", "\\1", grp))
        day_label <- gsub("day ", "Day ", day_label)
        cond_label <- gsub("_day.*", "", grp)
        group_label <- paste0(cond_label, " ", day_label)
      }
      title <- paste0(SITE_LABELS[st], " ", level_label, " ", cfg$label,
                      " Enrichment | ", group_label)

      p <- make_volcano(
        plot_data,
        x_col        = cfg$x_col,
        ci_lower_col = cfg$ci_lower_col,
        ci_upper_col = cfg$ci_upper_col,
        x_lim        = cfg$x_lim,
        y_max        = y_max,
        title        = title,
        x_label      = cfg$x_label,
        show_ci      = args$show_ci,
        show_legend  = TRUE
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
# Composite Plots
#   Layout depends on the input design:
#     - flat (--flat-design): one composite per group (rows = group,
#       cols = sites). No by-day composite (there is no timepoint axis).
#     - timepoint (default): by-condition and by-day grids over the
#       condition x timepoint cross-product.
# ============================================================

composite_count <- 0
composite_day_count <- 0
mega_count <- 0

if (args$flat_design) {

  # ---- Flat control-vs-treatment design (no timepoint) ----
  # `group` already identifies the only experimental axis, so filter on it
  # directly rather than reconstructing a condition_timepoint key.
  cat("Generating composite plots (flat design)...\n")

  for (etype in names(enrichment_configs)) {
    cfg <- enrichment_configs[[etype]]

    for (grp in groups) {

      plot_list <- list()

      for (st in sites) {
        plot_data <- data |> filter(group == grp, site == st)
        subtitle <- paste0(SITE_LABELS[st], " | ", grp)

        plot_list[[length(plot_list) + 1]] <- make_volcano(
          plot_data,
          x_col        = cfg$x_col,
          ci_lower_col = cfg$ci_lower_col,
          ci_upper_col = cfg$ci_upper_col,
          x_lim        = cfg$x_lim,
          y_max        = y_max,
          title        = subtitle,
          x_label      = cfg$x_label,
          show_ci      = args$show_ci,
          show_legend  = FALSE
        )
      }

      n_sites <- length(sites)
      composite <- wrap_plots(plot_list, ncol = n_sites, nrow = 1) +
        plot_layout(guides = "collect") +
        plot_annotation(
          title = paste0(grp, ": ", cfg$label, " ", level_label, " Enrichment"),
          theme = theme(
            plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
          )
        ) &
        theme(legend.position = "bottom")

      filepath <- file.path(args$outdir, "composite", etype,
                            paste0(grp, "_volcano_grid"))
      save_plot(composite, filepath,
                width  = 6 * n_sites,
                height = 5.5 * 1 + 1.5,
                format = args$format, dpi = args$dpi)
      composite_count <- composite_count + 1
    }
  }
  cat("  Saved", composite_count, "group composite plots\n")

  # Mega composite (all groups): rows = group, cols = sites.
  if (args$mega_composite) {
    cat("Generating mega composite (all groups)...\n")

    for (etype in names(enrichment_configs)) {
      cfg <- enrichment_configs[[etype]]
      plot_list <- list()

      for (grp in groups) {
        for (st in sites) {
          plot_data <- data |> filter(group == grp, site == st)
          plot_list[[length(plot_list) + 1]] <- make_volcano(
            plot_data,
            x_col        = cfg$x_col,
            ci_lower_col = cfg$ci_lower_col,
            ci_upper_col = cfg$ci_upper_col,
            x_lim        = cfg$x_lim,
            y_max        = y_max,
            title        = paste0(SITE_LABELS[st], " | ", grp),
            x_label      = cfg$x_label,
            show_ci      = args$show_ci,
            show_legend  = FALSE
          )
        }
      }

      n_groups <- length(groups)
      n_sites  <- length(sites)
      mega <- wrap_plots(plot_list, ncol = n_sites, nrow = n_groups) +
        plot_layout(guides = "collect") +
        plot_annotation(
          title = paste0(cfg$label, " ", level_label,
                         " Enrichment - All Groups"),
          theme = theme(
            plot.title = element_text(hjust = 0.5, size = 20, face = "bold")
          )
        ) &
        theme(legend.position = "bottom")

      filepath <- file.path(args$outdir, "composite", etype, "all_groups_volcano_grid")
      save_plot(mega, filepath,
                width  = 6 * n_sites,
                height = 5.5 * n_groups + 1.5,
                format = args$format, dpi = args$dpi)
      mega_count <- mega_count + 1
    }
    cat("  Saved", mega_count, "mega composite plots\n")
  }

} else {

# ============================================================
# Composite Plots: Grouped by Condition
# ============================================================

cat("Generating composite plots by condition...\n")

conditions <- sort(unique(data$condition))
# Order timepoints numerically (day_0, day_5, day_10) when possible.
days_raw <- unique(data$timepoint)
num_keys <- suppressWarnings(as.numeric(gsub("\\D", "", days_raw)))
if (all(!is.na(num_keys))) {
  days <- days_raw[order(num_keys)]
} else {
  days <- sort(days_raw)
}
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
          x_col        = cfg$x_col,
          ci_lower_col = cfg$ci_lower_col,
          ci_upper_col = cfg$ci_upper_col,
          x_lim        = cfg$x_lim,
          y_max        = y_max,
          title        = subtitle,
          x_label      = cfg$x_label,
          show_ci      = args$show_ci,
          show_legend  = FALSE
        )

        plot_list[[length(plot_list) + 1]] <- p
      }
    }

    n_days  <- length(days)
    n_sites <- length(sites)
    composite <- wrap_plots(plot_list, ncol = n_sites, nrow = n_days) +
      plot_layout(guides = "collect") +
      plot_annotation(
        title = paste0(cond, ": ", cfg$label, " ", level_label, " Enrichment"),
        theme = theme(
          plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
        )
      ) &
      theme(legend.position = "bottom")

    filepath <- file.path(args$outdir, "composite", etype,
                          paste0(cond, "_volcano_grid"))
    save_plot(composite, filepath,
              width  = 6 * n_sites,
              height = 5.5 * n_days + 1.5,
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
          x_col        = cfg$x_col,
          ci_lower_col = cfg$ci_lower_col,
          ci_upper_col = cfg$ci_upper_col,
          x_lim        = cfg$x_lim,
          y_max        = y_max,
          title        = subtitle,
          x_label      = cfg$x_label,
          show_ci      = args$show_ci,
          show_legend  = FALSE
        )

        plot_list[[length(plot_list) + 1]] <- p
      }
    }

    day_label <- gsub("_", " ", d)
    day_label <- gsub("day ", "Day ", day_label)

    n_conds <- length(conditions)
    n_sites <- length(sites)
    composite <- wrap_plots(plot_list, ncol = n_sites, nrow = n_conds) +
      plot_layout(guides = "collect") +
      plot_annotation(
        title = paste0(day_label, ": ", cfg$label, " ", level_label, " Enrichment"),
        theme = theme(
          plot.title = element_text(hjust = 0.5, size = 18, face = "bold")
        )
      ) &
      theme(legend.position = "bottom")

    filepath <- file.path(args$outdir, "composite", etype,
                          paste0(d, "_volcano_grid"))
    save_plot(composite, filepath,
              width  = 6 * n_sites,
              height = 5.5 * n_conds + 1.5,
              format = args$format, dpi = args$dpi)
    composite_day_count <- composite_day_count + 1
  }
}

cat("  Saved", composite_day_count, "day composite plots\n")

# ============================================================
# Mega Composite: All Groups (rows = condition x day, cols = sites)
# Optional, gated by --mega-composite to preserve parity with prior
# stall_sites behaviour (which never produced this plot type).
# ============================================================

mega_count <- 0
if (args$mega_composite) {
  cat("Generating mega composite (all groups)...\n")

  for (etype in names(enrichment_configs)) {
    cfg <- enrichment_configs[[etype]]
    plot_list <- list()

    for (cond in conditions) {
      for (d in days) {
        grp <- paste0(cond, "_", d)
        day_label <- gsub("_", " ", d)
        day_label <- gsub("day ", "Day ", day_label)

        for (st in sites) {
          plot_data <- data |> filter(group == grp, site == st)

          plot_list[[length(plot_list) + 1]] <- make_volcano(
            plot_data,
            x_col        = cfg$x_col,
            ci_lower_col = cfg$ci_lower_col,
            ci_upper_col = cfg$ci_upper_col,
            x_lim        = cfg$x_lim,
            y_max        = y_max,
            title        = paste0(SITE_LABELS[st], " | ", cond, " - ", day_label),
            x_label      = cfg$x_label,
            show_ci      = args$show_ci,
            show_legend  = FALSE
          )
        }
      }
    }

    n_groups <- length(conditions) * length(days)
    n_sites  <- length(sites)
    mega <- wrap_plots(plot_list, ncol = n_sites, nrow = n_groups) +
      plot_layout(guides = "collect") +
      plot_annotation(
        title = paste0(cfg$label, " ", level_label,
                       " Enrichment - All Groups"),
        theme = theme(
          plot.title = element_text(hjust = 0.5, size = 20, face = "bold")
        )
      ) &
      theme(legend.position = "bottom")

    filepath <- file.path(args$outdir, "composite", etype, "all_groups_volcano_grid")
    save_plot(mega, filepath,
              width  = 6 * n_sites,
              height = 5.5 * n_groups + 1.5,
              format = args$format, dpi = args$dpi)
    mega_count <- mega_count + 1
  }
  cat("  Saved", mega_count, "mega composite plots\n")
}

}  # end timepoint-design composites

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
cat("Confidence intervals:", ifelse(args$show_ci, "shown", "hidden"), "\n")
cat("Mega composite:", ifelse(args$mega_composite, "yes", "no"), "\n")
cat("============================================\n")
