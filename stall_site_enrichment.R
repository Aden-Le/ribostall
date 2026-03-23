compute_AA_enrichment <- function(file_list, prefix_list, working_dir) {
  
  results <- list()
  
  for (i in seq_along(file_list)) {
    
    counts_file <- file_list[i]
    prefix <- prefix_list[i]
    
    background_file <- gsub("_counts.csv", "_background.csv", counts_file)
    
    #-----------------------------
    # Read files
    
    AA_counts <- read.csv(file.path(working_dir, counts_file))
    
    AA_background_frequency <- read.csv(file.path(working_dir, background_file)) |>
      rename(
        AA = X,
        background_proportion = X0
      )
    
    #-----------------------------
    # Column titles
    
    col_titles <- c("AA", as.character(-10:6))
    colnames(AA_counts) <- col_titles
    
    # EPA sites
    AA_counts <- AA_counts |>
      select(AA, "-1", "0", "1") |>
      rename(
        E = "-1",
        P = "0",
        A = "1"
      )
    
    #-----------------------------
    # Site totals
    
    site_totals <- AA_counts |>
      summarise(across(E:A, sum))
    
    #-----------------------------
    # Significance tests
    
    AA_significance <- AA_counts |>
      left_join(AA_background_frequency, by = "AA") |>
      rowwise() |>
      mutate(
        E_pvalue = binom.test(E, site_totals$E, background_proportion)$p.value,
        P_pvalue = binom.test(P, site_totals$P, background_proportion)$p.value,
        A_pvalue = binom.test(A, site_totals$A, background_proportion)$p.value
      ) |>
      ungroup()
    
    AA_significance_adjusted <- AA_significance |>
      mutate(
        E_FDR = p.adjust(E_pvalue, method = "BH"),
        P_FDR = p.adjust(P_pvalue, method = "BH"),
        A_FDR = p.adjust(A_pvalue, method = "BH")
      ) |>
      select(AA, E_FDR, P_FDR, A_FDR)
    
    #-----------------------------
    # Confidence intervals
    
    AA_counts_with_ci <- AA_counts |>
      left_join(AA_background_frequency, by = "AA") |>
      rowwise() |>
      mutate(
        E_p_lower = qbeta(0.025, E + 0.5, site_totals$E - E + 0.5),
        E_p_upper = qbeta(0.975, E + 0.5, site_totals$E - E + 0.5),
        P_p_lower = qbeta(0.025, P + 0.5, site_totals$P - P + 0.5),
        P_p_upper = qbeta(0.975, P + 0.5, site_totals$P - P + 0.5),
        A_p_lower = qbeta(0.025, A + 0.5, site_totals$A - A + 0.5),
        A_p_upper = qbeta(0.975, A + 0.5, site_totals$A - A + 0.5),
        
        E_CI_lower = log2(E_p_lower / background_proportion),
        E_CI_upper = log2(E_p_upper / background_proportion),
        P_CI_lower = log2(P_p_lower / background_proportion),
        P_CI_upper = log2(P_p_upper / background_proportion),
        A_CI_lower = log2(A_p_lower / background_proportion),
        A_CI_upper = log2(A_p_upper / background_proportion)
      ) |>
      ungroup()
    
    #-----------------------------
    # Weighted enrichment
    
    AA_counts_psuedo <- AA_counts |>
      mutate(across(where(is.numeric), ~ .x + 0.5))
    
    AA_proportion <- AA_counts_psuedo |>
      mutate(across(where(is.numeric), ~ .x / sum(.x)))
    
    AA_enrichment <- AA_proportion |>
      left_join(AA_background_frequency, by = "AA") |>
      mutate(
        E_enrichment = E / background_proportion,
        P_enrichment = P / background_proportion,
        A_enrichment = A / background_proportion
      ) |>
      mutate(
        E_log2_enrichment = log2(E_enrichment),
        P_log2_enrichment = log2(P_enrichment),
        A_log2_enrichment = log2(A_enrichment)
      ) |>
      mutate(
        E_weighted_enrichment = E * E_log2_enrichment,
        P_weighted_enrichment = P * P_log2_enrichment,
        A_weighted_enrichment = A * A_log2_enrichment
      )
    
    AA_counts_with_ci_weighted <- AA_counts_with_ci |>
      mutate(
        E_Weighted_CI_lower = AA_enrichment$E * E_CI_lower,
        E_Weighted_CI_upper = AA_enrichment$E * E_CI_upper,
        P_Weighted_CI_lower = AA_enrichment$P * P_CI_lower,
        P_Weighted_CI_upper = AA_enrichment$P * P_CI_upper,
        A_Weighted_CI_lower = AA_enrichment$A * A_CI_lower,
        A_Weighted_CI_upper = AA_enrichment$A * A_CI_upper
      )
    
    #-----------------------------
    # Final outputs
    
    AA_results_weighted <- AA_enrichment |>
      select(AA, E_weighted_enrichment, P_weighted_enrichment, A_weighted_enrichment) |>
      left_join(AA_significance_adjusted, by = "AA") |>
      left_join(
        AA_counts_with_ci_weighted |>
          select(
            AA,
            E_Weighted_CI_lower, E_Weighted_CI_upper,
            P_Weighted_CI_lower, P_Weighted_CI_upper,
            A_Weighted_CI_lower, A_Weighted_CI_upper
          ),
        by = "AA"
      )
    
    AA_results_unweighted <- AA_enrichment |>
      select(AA, E_log2_enrichment, P_log2_enrichment, A_log2_enrichment) |>
      left_join(AA_significance_adjusted, by = "AA") |>
      left_join(
        AA_counts_with_ci_weighted |>
          select(
            AA,
            E_CI_lower, E_CI_upper,
            P_CI_lower, P_CI_upper,
            A_CI_lower, A_CI_upper
          ),
        by = "AA"
      )
    
    #-----------------------------
    # Store results
    
    results[[paste0(prefix, "_AA_results_weighted")]] <- AA_results_weighted
    results[[paste0(prefix, "_AA_results_unweighted")]] <- AA_results_unweighted
  }
  
  return(results)
}

file_list <- c(
  "BWM_day_0_counts.csv",
  "BWM_day_5_counts.csv",
  "BWM_day_10_counts.csv",
  "control_day_0_counts.csv",
  "control_day_5_counts.csv",
  "control_day_10_counts.csv"
)

prefix_list <- c(
  "BWM_day_0",
  "BWM_day_5",
  "BWM_day_10",
  "control_day_0",
  "control_day_5",
  "control_day_10"
)

WORKING_DIR <- "C:/Users/Aden Le/Documents/Sarinay Lab/ribo_stall_results/motif_csv/"

results <- compute_AA_enrichment(file_list, prefix_list, WORKING_DIR)

volcano_plot <- function(dataset, col_names,
                         title = "Volcano Plot",
                         enrichment_label = "Log2 Enrichment",
                         p_label = "FDR") {
  
  AA_col <- col_names[1]
  enrich_col <- col_names[2]
  p_col <- col_names[3]
  lower_col <- col_names[4]
  upper_col <- col_names[5]
  
  # Amino acid classes
  AA_CLASS <- c(
    "D" = "acidic", "E" = "acidic",
    "K" = "basic", "R" = "basic", "H" = "basic",
    "A" = "hydrophobic", "V" = "hydrophobic", "I" = "hydrophobic", 
    "L" = "hydrophobic", "M" = "hydrophobic", "F" = "hydrophobic", 
    "W" = "hydrophobic", "Y" = "hydrophobic",
    "C" = "polar", "N" = "polar", "Q" = "polar", "S" = "polar", "T" = "polar",
    "G" = "neutral", "P" = "neutral"
  )
  
  # Colors
  class_colors <- c(
    "acidic" = "#E41A1C",
    "basic" = "#377EB8",
    "hydrophobic" = "#4DAF4A",
    "polar" = "#984EA3",
    "neutral" = "#FF7F00"
  )
  
  # Prepare data
  dataset <- dataset |>
    mutate(
      class = AA_CLASS[.data[[AA_col]]],
      significance = ifelse(.data[[p_col]] < 0.05,
                            "Significant (FDR < 0.05)",
                            "Not significant"),
      to_label = ifelse(.data[[p_col]] < 0.05, TRUE, FALSE),
      label = ifelse(to_label, .data[[AA_col]], "")
    )
  
  p <- ggplot(dataset,
              aes(x = .data[[enrich_col]],
                  y = -log10(.data[[p_col]]))) +
    
    geom_point(aes(color = class, shape = significance),
               alpha = 0.8, size = 2.5) +
    
    geom_vline(xintercept = c(-0.5, 0.5),
               linetype = "dashed",
               color = "gray50",
               alpha = 0.5) +
    
    geom_hline(yintercept = -log10(0.05),
               linetype = "dashed",
               color = "gray50",
               alpha = 0.5) +
    
    geom_vline(xintercept = 0,
               linetype = "dotted",
               color = "black",
               alpha = 0.3) +
    
    # geom_errorbarh(
    #   aes(xmin = .data[[lower_col]],
    #       xmax = .data[[upper_col]]),
    #   height = 0.1,
    #   alpha = 0.5
    # ) +
    
    ggrepel::geom_text_repel(
      data = subset(dataset, to_label),
      aes(label = label, color = class),
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
    
    scale_color_manual(
      values = class_colors,
      name = NULL
    ) +
    
    scale_shape_manual(
      name = NULL,
      values = c(
        "Significant (FDR < 0.05)" = 17,
        "Not significant" = 16
      )
    ) +
    scale_y_continuous(
      limits = c(0, 8)
    ) +
    scale_x_continuous(
      limits = c(-5, 5),
      breaks = seq(-5, 5, 1)
    ) +

  # Theme improvements - with closer legends
  theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.position = "bottom",
    legend.box = "vertical",
    legend.box.just = "center",
    legend.box.spacing = unit(0.1, "cm"),  # Reduce space between legend boxes
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    legend.background = element_rect(fill = "white", color = NA),
    legend.key.size = unit(0.5, "cm"),  # Slightly smaller keys
    legend.spacing.y = unit(0.05, "cm"),  # Reduce vertical spacing
    legend.margin = margin(t = 0, b = 0, unit = "cm"),  # Remove legend margins
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    plot.margin = margin(1, 1, 0.5, 1, "cm")  # Reduced bottom margin
  ) +
    
    labs(
      title = title,
      x = enrichment_label,
      y = expression(bold("-Log"[10]~"(FDR)"))
    ) +
    
    guides(
      color = guide_legend(order = 1),
      shape = guide_legend(order = 2)
    )
  
  return(p)
}

colnames(results$BWM_day_0_AA_results_unweighted)

volcano_plot(
  results$BWM_day_0_AA_results_unweighted,
  c("AA", "E_log2_enrichment", "E_FDR", "E_CI_lower", "E_CI_upper"),
  title = "Volcano Plot: E-site Enrichment | BWM Day 0"
)

volcano_plot(
  results$BWM_day_0_AA_results_unweighted,
  c("AA", "P_log2_enrichment", "P_FDR", "P_CI_lower", "P_CI_upper"),
  title = "Volcano Plot: P-site Enrichment | BWM Day 0"
)

volcano_plot(
  results$BWM_day_0_AA_results_unweighted,
  c("AA", "A_log2_enrichment", "A_FDR", "A_CI_lower", "A_CI_upper"),
  title = "Volcano Plot: A-site Enrichment | BWM Day 0"
)