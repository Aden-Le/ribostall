#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

# --- mappings ---
CODON2AA = {
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R","AGA":"R","AGG":"R",
    "AAT":"N","AAC":"N","GAT":"D","GAC":"D","TGT":"C","TGC":"C",
    "GAA":"E","GAG":"E","CAA":"Q","CAG":"Q","GGT":"G","GGC":"G","GGA":"G","GGG":"G",
    "CAT":"H","CAC":"H","ATT":"I","ATC":"I","ATA":"I",
    "TTA":"L","TTG":"L","CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "AAA":"K","AAG":"K","ATG":"M","TTT":"F","TTC":"F",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S","AGT":"S","AGC":"S",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T","TGG":"W",
    "TAT":"Y","TAC":"Y","GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TAA":"*","TAG":"*","TGA":"*"
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert codon occupancy CSV to amino acid occupancy by summing counts."
    )
    p.add_argument("--input", required=True,
                   help="Path to codon occupancy CSV")
    p.add_argument("--out", default="aa_occupancy.csv",
                   help="Output CSV (default: aa_occupancy.csv)")
    return p.parse_args()


def codon_to_aa_occupancy(codon_csv_path, output_path):
    """
    Read codon occupancy CSV and convert to amino acid occupancy.
    Groups codons by their amino acid translation and sums occupancy counts.
    """
    # Read the codon occupancy CSV
    df = pd.read_csv(codon_csv_path)
    
    # Verify expected structure
    if df.columns[0] != "Codon":
        raise ValueError("First column must be 'Codon'")
    
    # Get all experiments (all columns except 'Codon' and 'Transcriptome')
    experiment_cols = [col for col in df.columns if col not in ["Codon", "Transcriptome"]]
    
    # Create a mapping from amino acid to list of codons
    # Example: {'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'], ...}
    aa_to_codons = {}
    for codon, aa in CODON2AA.items():
        if aa not in aa_to_codons:
            aa_to_codons[aa] = []
        aa_to_codons[aa].append(codon)
    
    # Group and sum by amino acid
    aa_results = []
    for aa in sorted(aa_to_codons.keys()):
        codons = aa_to_codons[aa]
        # Get rows for this amino acid's codons
        aa_rows = df[df["Codon"].isin(codons)]
        
        if len(aa_rows) == 0:
            continue
        
        # Sum across all codons for this amino acid
        row_dict = {"AA": aa}
        
        # Get Transcriptome value if it exists (sum of all codons)
        if "Transcriptome" in df.columns:
            row_dict["Transcriptome"] = aa_rows["Transcriptome"].sum()
        
        # Sum all experimental columns
        for col in experiment_cols:
            row_dict[col] = aa_rows[col].sum()
        
        aa_results.append(row_dict)
    
    # Create output dataframe
    aa_df = pd.DataFrame(aa_results)
    
    # Reorder columns: AA first, then Transcriptome (if exists), then experiments
    cols_order = ["AA"]
    if "Transcriptome" in aa_df.columns:
        cols_order.append("Transcriptome")
    cols_order.extend(experiment_cols)
    aa_df = aa_df[cols_order]
    
    # Write to CSV
    aa_df.to_csv(output_path, index=False)
    print(f"Amino acid occupancy written to {output_path}")
    print(f"Shape: {aa_df.shape[0]} amino acids x {len(experiment_cols)} experiments")


if __name__ == "__main__":
    args = parse_args()
    codon_to_aa_occupancy(args.input, args.out)
