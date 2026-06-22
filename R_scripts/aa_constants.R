# ============================================================
# Shared AA / codon constants for the R_scripts plotting suite.
#
# Sourced verbatim by aa_codon_overlay.R, between_group_barplot.R,
# between_group_volcano.R, and within_condition_volcano.R via
#   source("R_scripts/aa_constants.R")
# Scripts are always run from the repo root (the shell_scripts
# wrappers cd there first), so the path is repo-root relative.
#
# Only values that are byte-identical across all four scripts live
# here. Run-dependent labels (e.g. SIG_LABEL / SHAPE_VALS, which are
# built from args$fdr) stay in the individual scripts.
# ============================================================

# Amino-acid -> property class. Drives bar/point fill colour.
AA_CLASS <- c(
  "D" = "Acidic", "E" = "Acidic",
  "K" = "Basic", "R" = "Basic", "H" = "Basic",
  "A" = "Hydrophobic", "V" = "Hydrophobic", "I" = "Hydrophobic",
  "L" = "Hydrophobic", "M" = "Hydrophobic", "F" = "Hydrophobic",
  "W" = "Hydrophobic", "Y" = "Hydrophobic",
  "C" = "Polar", "N" = "Polar", "Q" = "Polar", "S" = "Polar", "T" = "Polar",
  "G" = "Neutral", "P" = "Neutral"
)

# Property class -> colour. "Stop" is used only at codon level (stop codons).
CLASS_COLORS <- c(
  "Acidic"      = "#E41A1C",
  "Basic"       = "#377EB8",
  "Hydrophobic" = "#4DAF4A",
  "Polar"       = "#984EA3",
  "Neutral"     = "#FF7F00",
  "Stop"        = "#666666"
)

# Codon -> amino-acid decode, used to cluster/classify codon-level features.
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
  "TAT"="Y","TAC"="Y","GTT"="V","GTC"="V","GTA"="V","GTG"="V",
  "TAA"="*","TAG"="*","TGA"="*"
)

# Site code -> display label.
SITE_LABELS <- c("E" = "E-site", "P" = "P-site", "A" = "A-site")
