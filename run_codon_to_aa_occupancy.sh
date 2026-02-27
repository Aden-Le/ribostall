#!/bin/bash

# Input codon occupancy CSV file
INPUT="./ribo_stall_results/codon_occupancy_all.csv"

# Output AA occupancy CSV file
OUTPUT="./ribo_stall_results/aa_occupancy_all.csv"

# Run the conversion script
python codon_to_aa_occupancy.py --input "$INPUT" --out "$OUTPUT"
