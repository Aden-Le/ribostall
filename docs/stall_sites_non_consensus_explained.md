# `stall_sites_non_consensus.py` — Detailed Walkthrough

This script detects **ribosome stall sites** from ribosome profiling (Ribo-seq) data and optionally runs amino acid enrichment analyses at the E/P/A sites of stalled ribosomes. Unlike a consensus-based approach, stalls are called **per-replicate** — there is no requirement for a stall to appear in multiple replicates to be retained.

---

## Table of Contents

1. [High-Level Pipeline](#1-high-level-pipeline)
2. [Argument Parsing](#2-argument-parsing)
3. [Parse Experimental Groups](#3-parse-experimental-groups)
4. [Load Coverage Data](#4-load-coverage-data)
5. [Load the Ribo Object](#5-load-the-ribo-object)
6. [Transcript Filtering (Per-Group)](#6-transcript-filtering-per-group)
7. [Codonize Coverage](#7-codonize-coverage)
8. [Call Stall Sites](#8-call-stall-sites)
9. [Export Stall Sites to CSV](#9-export-stall-sites-to-csv)
10. [Load CDS Ranges and Sequences](#10-load-cds-ranges-and-sequences)
11. [EPA Extraction](#11-epa-extraction)
12. [Background AA Frequencies](#12-background-aa-frequencies)
13. [Condition/Timepoint Mappings](#13-conditiontimepoint-mappings)
14. [Analysis 1: Within-Condition Enrichment (Binomial Test)](#14-analysis-1-within-condition-enrichment-binomial-test)
15. [Analysis 2: Between-Condition Wilcoxon Rank-Sum](#15-analysis-2-between-condition-wilcoxon-rank-sum)
16. [Analysis 3: Per-Timepoint Fisher's Exact Test](#16-analysis-3-per-timepoint-fishers-exact-test)
17. [Output Files](#17-output-files)

---

## 1. High-Level Pipeline

```
coverage.pickle.gz ──┐
                     ├──> Filter transcripts ──> Codonize ──> Call stalls ──> stall_sites.csv
   data.ribo ────────┘                                            │
                                                                  │  (if --enrichment)
   reference.fa ─────────────────────────────────────────────────>│
                                                                  ▼
                                                        Extract E/P/A amino acids
                                                                  │
                                           ┌──────────────────────┼──────────────────────┐
                                           ▼                      ▼                      ▼
                                   Analysis 1              Analysis 2              Analysis 3
                                  Binomial test          Wilcoxon rank-sum       Fisher's exact
                                (within-condition)     (between-condition)    (per-timepoint)
                                       │                      │                      │
                                       ▼                      ▼                      ▼
                              within_condition_     between_condition_     per_timepoint_
                              enrichment.csv        wilcoxon.csv           fisher.csv
```

---

## 2. Argument Parsing

The script accepts these CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--pickle` | (required) | Gzipped pickle of per-nt coverage arrays |
| `--ribo` | (required) | HDF5 `.ribo` file with transcript metadata |
| `--groups` | (required) | Semicolon-separated group definitions |
| `--tx_threshold` | 1.0 | Min reads/nt for a transcript to pass filtering |
| `--tx_min_reps` | 2 | Min replicates passing threshold per group |
| `--min_z` | 2.0 | Min z-score for a codon to be called a stall |
| `--min_reads` | 5 | Min raw read count for a stall |
| `--trim-start` | 20 | Codons to exclude from 5' end (initiation ramp) |
| `--trim-stop` | 10 | Codons to exclude from 3' end (termination) |
| `--pseudocount` | 0.5 | Pseudocount added before log2 in z-score |
| `--reference` | — | FASTA reference file (needed for `--enrichment`) |
| `--enrichment` | off | Flag to run E/P/A enrichment analyses |
| `--out-enrichment` | `../ribostall_results/enrichment` | Output directory |

**Example invocation:**
```bash
python stall_sites_non_consensus.py \
  --pickle coverage.pickle.gz \
  --ribo data.ribo \
  --groups "control_day_0:ctrl_rep1,ctrl_rep2;control_day_5:ctrl_rep3,ctrl_rep4;BWM_day_0:bwm_rep1,bwm_rep2;BWM_day_5:bwm_rep3,bwm_rep4" \
  --reference gencode.fa \
  --enrichment \
  --min_z 2.0 \
  --min_reads 5
```

---

## 3. Parse Experimental Groups

**Code:** `parse_groups(args.groups)`

Converts the CLI string into a dictionary mapping each experimental group to its replicate experiment names.

**Example:**

```
Input string:
  "control_day_0:ctrl_rep1,ctrl_rep2;BWM_day_5:bwm_rep3,bwm_rep4"

Output dict:
  {
    "control_day_0": ["ctrl_rep1", "ctrl_rep2"],
    "BWM_day_5":     ["bwm_rep3", "bwm_rep4"]
  }
```

This structure is used everywhere downstream to know which experiments belong together and to build condition/timepoint mappings.

A **reverse mapping** (`rep_to_group`) is also built:
```
  {
    "ctrl_rep1": "control_day_0",
    "ctrl_rep2": "control_day_0",
    "bwm_rep3":  "BWM_day_5",
    "bwm_rep4":  "BWM_day_5"
  }
```

---

## 4. Load Coverage Data

**Code:** `pickle.load(gzip.open(args.pickle, "rb"))`

The pickle file contains a nested dictionary of per-nucleotide read counts:

```
cov = {
  "ctrl_rep1": {
    "ENST00000456328|...": np.array([0, 0, 3, 5, 2, 1, 0, 8, ...]),  # one value per nt
    "ENST00000450305|...": np.array([1, 0, 0, 4, 7, ...]),
    ...  # ~20,000 transcripts
  },
  "ctrl_rep2": { ... },
  "bwm_rep3":  { ... },
  "bwm_rep4":  { ... },
  "failed_rep1": { ... },   # <-- present but ignored because not in --groups
}
```

Each array represents the number of ribosome footprint reads mapping to each nucleotide position along the entire transcript (5'UTR + CDS + 3'UTR). Only experiments listed in `--groups` are ever used.

---

## 5. Load the Ribo Object

**Code:** `Ribo(args.ribo, alias=None)`

The `.ribo` file (HDF5 format from `ribopy`) contains:
- Transcript names and their region boundaries (5'UTR, CDS, 3'UTR lengths)
- Metagene profiles
- Experiment metadata

`alias=None` means raw transcript names (e.g., `ENST00000456328|ENSG00000...|OTTHUMG...|OTTHUMT...|DDX11L1|...`) are used without remapping.

---

## 6. Transcript Filtering (Per-Group)

**Code:** `filter_tx()` from `functions_stall_sites.py`

**Goal:** Remove low-coverage transcripts that would produce noisy stall calls.

**How it works:**

For each group independently:
1. For each transcript, compute the mean reads/nt across the whole array for each replicate
2. Count how many replicates exceed `--tx_threshold` (default 1.0 read/nt)
3. Keep the transcript only if at least `--tx_min_reps` replicates pass

**Example with 2 replicates, threshold=1.0, min_reps=2:**

```
Transcript ENST000001 (length = 1000 nt):
  ctrl_rep1: mean = 2.3 reads/nt  ✓ passes
  ctrl_rep2: mean = 1.8 reads/nt  ✓ passes
  → 2/2 pass ≥ min_reps=2 → KEEP

Transcript ENST000002 (length = 500 nt):
  ctrl_rep1: mean = 0.4 reads/nt  ✗ fails
  ctrl_rep2: mean = 3.1 reads/nt  ✓ passes
  → 1/2 pass < min_reps=2 → DISCARD

Transcript ENST000003 (length = 2000 nt):
  ctrl_rep1: mean = 0.1 reads/nt  ✗ fails
  ctrl_rep2: mean = 0.05 reads/nt ✗ fails
  → 0/2 pass < min_reps=2 → DISCARD
```

**Key design choice: no intersection across groups.** Each group gets its own set of passing transcripts. `control_day_0` might retain 8,000 transcripts while `BWM_day_5` retains 7,500 — they don't need to match.

```
filt_tx_dict = {
  "control_day_0": {"ENST000001", "ENST000005", ...},   # 8,000 transcripts
  "BWM_day_5":     {"ENST000001", "ENST000003", ...},   # 7,500 transcripts
}
```

The coverage dict is then rebuilt so each experiment only contains its group's passing transcripts:

```
cov_filt = {
  "ctrl_rep1": { only transcripts in filt_tx_dict["control_day_0"] },
  "ctrl_rep2": { only transcripts in filt_tx_dict["control_day_0"] },
  "bwm_rep3":  { only transcripts in filt_tx_dict["BWM_day_5"] },
  "bwm_rep4":  { only transcripts in filt_tx_dict["BWM_day_5"] },
}
```

---

## 7. Codonize Coverage

**Code:** `codonize_counts_cds()` from `functions_stall_sites.py`

**Goal:** Convert nucleotide-resolution arrays into codon-resolution arrays by summing every 3 consecutive nucleotides.

**How it works:**

The input arrays already cover only the CDS (no UTRs). The function:
1. Trims to a multiple of 3
2. Reshapes into (n_codons, 3)
3. Sums across the 3-nt axis

**Example:**

```
Input (nt-level, CDS only):
  [2, 1, 0, | 5, 3, 1, | 0, 0, 8, | 3, 2, 1, | 0, 0, 0]
   codon 0     codon 1    codon 2    codon 3    codon 4

Reshape to (5, 3):
  [[2, 1, 0],    → sum = 3
   [5, 3, 1],    → sum = 9
   [0, 0, 8],    → sum = 8
   [3, 2, 1],    → sum = 6
   [0, 0, 0]]    → sum = 0

Output (codon-level):
  [3.0, 9.0, 8.0, 6.0, 0.0]
```

Each value now represents total ribosome footprint reads at one codon position. This matches the biological unit of interest — one codon = one amino acid = one step of translation.

**Result structure:**
```
codon_cov = {
  "ctrl_rep1": {
    "ENST000001": np.array([3.0, 9.0, 8.0, 6.0, 0.0, ...]),  # ~300 codons for a 900-nt CDS
    "ENST000005": np.array([...]),
    ...
  },
  "ctrl_rep2": { ... },
  ...
}
```

---

## 8. Call Stall Sites

**Code:** `call_stalls()` from `functions_stall_sites.py`

**Goal:** Identify codon positions with abnormally high ribosome occupancy within each transcript.

### Algorithm: `global_z_log`

For each transcript's codon array:

1. **Log-transform:** `v = log2(x + pseudocount)` — compresses dynamic range, makes the distribution more normal
2. **Z-score:** `z = (v - mean(v)) / std(v)` — standardizes so each position is measured in standard deviations from the mean

Then a codon is called as a **stall** if ALL of:
- `z >= min_z` (default 2.0) — statistically unusual
- `raw_count >= min_reads` (default 5) — enough evidence
- Position is NOT within `trim_start` (20) codons of start or `trim_stop` (10) codons of stop

**Example for one transcript (150 codons):**

```
Codon array (first 20 shown):
  index:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  ...
  reads: [45, 30, 22, 15, 8,  3,  2,  1,  0,  1,  2,  1,  0,  3, ...]
          ←── initiation ramp (trimmed, first 20 codons) ──→

Log2(x + 0.5):
  index:  0     1     2     3     4     5     6     7   ...
  log2:  [5.51, 4.93, 4.49, 3.95, 3.09, 1.81, 1.32, 0.58, ...]

Z-scores (after computing mean and std across ALL 150 codons):
  Suppose mean(log2) = 1.8, std(log2) = 1.2

  z[0] = (5.51 - 1.8) / 1.2 = 3.09   but index 0 < trim_start=20, so TRIMMED
  ...
  z[42] = (4.2 - 1.8) / 1.2 = 2.0    index 42 ≥ 20 and reads=18 ≥ 5 → STALL ✓
  z[43] = (1.5 - 1.8) / 1.2 = -0.25  z < 2.0 → not a stall
  z[87] = (5.8 - 1.8) / 1.2 = 3.33   reads=55, index OK → STALL ✓
  ...
  z[145] = (3.9 - 1.8) / 1.2 = 1.75  z < 2.0 → not a stall
  z[148] = (4.5 - 1.8) / 1.2 = 2.25  but index 148 ≥ 150-10=140... wait, 148 < 140? No, 148 ≥ 140 → TRIMMED

Output for this transcript:
  [
    {"index": 42, "obs": 18.0, "z": 2.0},
    {"index": 87, "obs": 55.0, "z": 3.33},
  ]
```

### Why log-transform?

Raw ribosome profiling counts are highly skewed — a few codons might have 100+ reads while most have 0-5. Log-transformation pulls in the outliers so the z-score isn't dominated by one extreme peak. The pseudocount (0.5) prevents `log2(0)`.

### Why trim edges?

- **Start (20 codons):** The "initiation ramp" — ribosomes pile up near the start codon as they begin translation. This is a known artifact, not a true stall.
- **Stop (10 codons):** Ribosomes slow down near the stop codon during termination. Again an artifact.

**Result structure:**
```
stalls = {
  "ctrl_rep1": {
    "ENST000001": [{"index": 42, "obs": 18.0, "z": 2.0}, {"index": 87, "obs": 55.0, "z": 3.33}],
    "ENST000005": [],             # no stalls found
    "ENST000009": [{"index": 112, "obs": 7.0, "z": 2.5}],
    ...
  },
  "ctrl_rep2": { ... },
  ...
}
```

---

## 9. Export Stall Sites to CSV

**Code:** `stalls_to_long_df()` from `functions_stall_sites.py`

Flattens the nested stalls dict into a tidy DataFrame:

```
| group          | replicate  | gene    | tx_id          | transcript          | pos_codon | obs  | z    |
|----------------|------------|---------|----------------|---------------------|-----------|------|------|
| control_day_0  | ctrl_rep1  | DDX11L1 | ENST00000456328| ENST...|...|DDX11L1 | 42        | 18.0 | 2.00 |
| control_day_0  | ctrl_rep1  | DDX11L1 | ENST00000456328| ENST...|...|DDX11L1 | 87        | 55.0 | 3.33 |
| control_day_0  | ctrl_rep1  | TP53    | ENST00000269305| ENST...|...|TP53    | 112       | 7.0  | 2.50 |
| control_day_0  | ctrl_rep2  | DDX11L1 | ENST00000456328| ENST...|...|DDX11L1 | 43        | 12.0 | 2.10 |
| BWM_day_5      | bwm_rep3   | TP53    | ENST00000269305| ENST...|...|TP53    | 115       | 9.0  | 2.80 |
| ...            | ...        | ...     | ...            | ...                 | ...       | ...  | ...  |
```

The transcript key is parsed to extract `tx_id` and `gene` from the pipe-delimited format (e.g., `ENST...|ENSG...|OTTHUMG...|OTTHUMT...|GENE_NAME|...`).

Saved to `enrichment/stall_sites.csv`.

---

## 10. Load CDS Ranges and Sequences

**Conditional:** Only runs if `--enrichment` is set.

### CDS Ranges — `get_cds_range_lookup()`

Extracts the CDS start/stop positions (in nucleotide coordinates) for each transcript from the `.ribo` file.

```
cds_range = {
  "DDX11L1":  (200, 1400),    # CDS starts at nt 200, ends at nt 1400
  "TP53":     (150, 1330),
  ...
}
```

### Sequences — `get_sequence()`

Reads the reference FASTA and builds a dict of full transcript sequences:

```
sequence = {
  "ENST00000456328|...|DDX11L1|...": "AGCTTAGCTTAG...ATCGATCG",   # full nt sequence
  "ENST00000269305|...|TP53|...":    "GCTAGCTAGCTA...NNNATCGN",
  ...
}
```

Together, `cds_range` and `sequence` let us extract the CDS nucleotide sequence and translate it to amino acids.

---

## 11. EPA Extraction

**Code:** `epa_triplet_counts()` from `functions_AA.py`

**Goal:** For each stall site, identify what amino acid is at the ribosome's **E-site** (exit), **P-site** (peptidyl-tRNA), and **A-site** (aminoacyl-tRNA).

### Ribosome Biology Context

When a ribosome stalls, it occupies 3 codon positions simultaneously:

```
mRNA: ... [codon i-1] [codon i] [codon i+1] ...
             E-site     P-site    A-site
             (exit)     (holding   (incoming
              ←          peptide)   tRNA)
                           ↑
                     stall called here
```

The P-site is where the stall codon index points. The E-site is one codon upstream (i-1), and the A-site is one codon downstream (i+1).

### Algorithm

For each experiment and each stall site:

1. Look up the transcript's CDS nucleotide sequence
2. Translate to amino acids: `translate_cds_nt_to_aa()` reads triplets and maps via `CODON2AA`
3. Extract the amino acid at positions `i-1` (E), `i` (P), `i+1` (A)
4. Accumulate counts

**Example:**

```
Transcript TP53, CDS amino acid sequence:
  Position: ... 40  41  42  43  44 ...
  AA:       ... G   K   P   L   F  ...

Stall at codon index 42 (P-site):
  E-site (42-1 = 41): K (Lysine)
  P-site (42):        P (Proline)
  A-site (42+1 = 43): L (Leucine)

This stall contributes:
  E_counter["K"] += 1
  P_counter["P"] += 1
  A_counter["L"] += 1
```

After processing all stalls in one experiment:

```
replicate_counts["ctrl_rep1"] = {
  "E": pd.Series({"A": 45, "C": 12, "D": 38, ..., "Y": 8}),   # 20 amino acids
  "P": pd.Series({"A": 30, "C": 5,  "D": 22, ..., "Y": 15}),
  "A": pd.Series({"A": 52, "C": 18, "D": 35, ..., "Y": 11}),
}
```

Each Series has 20 values (one per standard amino acid) representing how many times that amino acid appeared at that ribosome site across all stall sites in this replicate.

---

## 12. Background AA Frequencies

**Code:** `background_aa_freq()` from `functions_AA.py`

**Goal:** Compute the expected frequency of each amino acid across all codons in the filtered transcripts. This is the null expectation — if stalling were random, the E/P/A site composition would match the background.

### Algorithm

1. For each filtered transcript in the group, translate the entire CDS to amino acids
2. Count every amino acid occurrence
3. Normalize to frequencies (proportions summing to 1.0)
4. Add a tiny pseudocount (1e-6) to avoid zeros

**Example:**

```
Group "control_day_0" has 8,000 transcripts with a total of ~3,500,000 codons.

Raw counts:
  A (Ala): 280,000
  C (Cys):  70,000
  D (Asp): 185,000
  ...
  P (Pro): 195,000
  ...
  W (Trp):  42,000

Frequencies (after normalization):
  A: 0.080
  C: 0.020
  D: 0.053
  ...
  P: 0.056
  ...
  W: 0.012
```

Each group gets its own background because different groups may have different sets of filtered transcripts (and thus slightly different overall amino acid compositions).

---

## 13. Condition/Timepoint Mappings

The group name is split on `_` (first split only) to extract condition and timepoint:

```
Group name: "control_day_0"
  → condition: "control"
  → timepoint: "day_0"

Group name: "BWM_day_5"
  → condition: "BWM"
  → timepoint: "day_5"
```

This enables the three downstream analyses to stratify by condition and/or timepoint.

**Full mapping example (6 groups, 12 replicates):**

```
rep_to_condition:
  ctrl_rep1 → "control"    bwm_rep1 → "BWM"
  ctrl_rep2 → "control"    bwm_rep2 → "BWM"
  ctrl_rep3 → "control"    bwm_rep3 → "BWM"
  ctrl_rep4 → "control"    bwm_rep4 → "BWM"
  ctrl_rep5 → "control"    bwm_rep5 → "BWM"
  ctrl_rep6 → "control"    bwm_rep6 → "BWM"

rep_to_timepoint:
  ctrl_rep1 → "day_0"      bwm_rep1 → "day_0"
  ctrl_rep2 → "day_0"      bwm_rep2 → "day_0"
  ctrl_rep3 → "day_5"      bwm_rep3 → "day_5"
  ctrl_rep4 → "day_5"      bwm_rep4 → "day_5"
  ctrl_rep5 → "day_10"     bwm_rep5 → "day_10"
  ctrl_rep6 → "day_10"     bwm_rep6 → "day_10"
```

---

## 14. Analysis 1: Within-Condition Enrichment (Binomial Test)

**Code:** `within_condition_enrichment()` from `functions_enrichment.py`

**Question:** "Is Proline enriched at the P-site of stall sites in `control_day_0` compared to the background frequency of Proline in the genome?"

### Algorithm

For each group, for each E/P/A site, for each amino acid:

1. **Pool** counts across replicates in that group
2. Compute the **observed frequency** = count(AA) / total_stalls
3. Get the **expected frequency** from the group's background
4. Run a **two-sided binomial test**: `binomtest(k, n, p_background)`
5. Compute **log2 enrichment** = log2(observed_freq / background_freq)
6. Compute **weighted log2 enrichment** = observed_freq × log2_enrichment
7. Apply **Benjamini-Hochberg FDR** correction per group

### Detailed Example

```
Group: control_day_0 (2 replicates pooled)
Site:  P-site
AA:    Proline (P)

Pooled stall count at P-site:
  ctrl_rep1: P-site Proline count = 85
  ctrl_rep2: P-site Proline count = 92
  Pooled: k = 85 + 92 = 177

Total stall sites (all AAs at P-site, pooled):
  ctrl_rep1: 500 stalls
  ctrl_rep2: 520 stalls
  Total: n = 1020

Observed frequency:
  freq = 177 / 1020 = 0.1735

Background frequency of Proline in control_day_0 transcripts:
  bg_freq = 0.056

Log2 enrichment:
  log2(0.1735 / 0.056) = log2(3.098) = 1.63

Weighted log2 enrichment:
  0.1735 × 1.63 = 0.283
  (scales enrichment by how common the AA actually is at stall sites)

Binomial test:
  binomtest(177, 1020, 0.056, alternative="two-sided")
  → p-value = 2.3e-45  (extremely significant)
```

**Interpretation:** Proline appears at the P-site of stalls 3x more often than expected by chance. This makes biological sense — Proline is a known cause of ribosome stalling due to its rigid ring structure.

### What the 2x2 comparison looks like conceptually

```
                    Proline    Not Proline
At stall P-site:      177        843         (observed from stalls)
In genome:           5.6%       94.4%        (expected from background)

Binomial: Is 177/1020 = 17.4% significantly different from 5.6%?
→ YES (p < 0.001)
```

### Caveat: Low Replicate Count (n=2)

With more replicates (e.g., n=5+), you would compute per-replicate frequencies and use a replicate-aware test (one-sample t-test or Wilcoxon signed-rank) to compare against the background — this explicitly models between-replicate variance. With only n=2 biological replicates per group, there are not enough data points to estimate that variance. The binomial test is a count-based test that works even with n=1, so pooling the 2 replicates together is the pragmatic choice — it increases the total stall site count for more precise frequency estimates, but it cannot account for replicate-level variation (e.g., shared library prep or batch effects).

### Output DataFrame

```
| condition | group         | site | amino_acid | stall_count | total_n | stall_freq | bg_freq | log2_enrichment | weighted_log2_enrichment | p_value  | p_adj    |
|-----------|---------------|------|------------|-------------|---------|------------|---------|-----------------|--------------------------|----------|----------|
| control   | control_day_0 | P    | P          | 177         | 1020    | 0.1735     | 0.056   | 1.63            | 0.283                    | 2.3e-45  | 1.4e-43  |
| control   | control_day_0 | P    | D          | 120         | 1020    | 0.1176     | 0.053   | 1.15            | 0.135                    | 4.1e-20  | 1.2e-18  |
| control   | control_day_0 | E    | K          | 95          | 1020    | 0.0931     | 0.058   | 0.68            | 0.063                    | 1.2e-05  | 2.1e-04  |
| BWM       | BWM_day_5     | A    | G          | 40          | 980     | 0.0408     | 0.070   | -0.78           | -0.032                   | 3.5e-03  | 4.2e-02  |
| ...       | ...           | ...  | ...        | ...         | ...     | ...        | ...     | ...             | ...                      | ...      | ...      |
```

---

## 15. Analysis 2: Between-Condition Wilcoxon Rank-Sum

**Code:** `between_condition_wilcoxon()` from `functions_enrichment.py`

**Question:** "Does the P-site Proline frequency differ between BWM-treated and control replicates across ALL timepoints?"

### Algorithm

For each E/P/A site, for each amino acid:

1. Compute **per-replicate frequencies** (count / total for that replicate)
2. Split replicates into two groups by condition (e.g., 6 control vs 6 BWM)
3. Run **Wilcoxon rank-sum** (Mann-Whitney U) test comparing the two sets of frequencies
4. Compute **log2 fold change** = log2(median_BWM / median_control)
5. Apply **BH-FDR** correction across all tests

### Detailed Example

```
Site: P-site, Amino acid: Proline

Per-replicate P-site Proline frequencies:
  Control replicates (6 values):
    ctrl_rep1: 85/500  = 0.170
    ctrl_rep2: 92/520  = 0.177
    ctrl_rep3: 78/480  = 0.163
    ctrl_rep4: 88/510  = 0.173
    ctrl_rep5: 95/530  = 0.179
    ctrl_rep6: 80/490  = 0.163
    → median = 0.171

  BWM replicates (6 values):
    bwm_rep1: 120/500 = 0.240
    bwm_rep2: 115/490 = 0.235
    bwm_rep3: 108/470 = 0.230
    bwm_rep4: 125/510 = 0.245
    bwm_rep5: 110/480 = 0.229
    bwm_rep6: 118/500 = 0.236
    → median = 0.236

Log2 fold change:
  log2(0.236 / 0.171) = log2(1.380) = 0.464

Mann-Whitney U test:
  U=0, p=0.0022 (all BWM values > all control values)
```

**Interpretation:** BWM treatment increases Proline frequency at the P-site of stall sites by ~46% (log2 FC = 0.46). This is significant even with only 6 vs 6 replicates.

### Visual of the rank-sum comparison

```
Control freqs: [0.163, 0.163, 0.170, 0.173, 0.177, 0.179]
BWM freqs:     [0.229, 0.230, 0.235, 0.236, 0.240, 0.245]
                       ↑ no overlap ↑
                     → very significant
```

### Why Wilcoxon and not t-test?

- Non-parametric — no assumption of normality
- Robust to outliers
- Works well with small sample sizes (n=6 per group)
- Frequencies are bounded [0, 1], which can violate t-test assumptions

### Output DataFrame

```
| site | amino_acid | median_BWM | median_control | log2_FC | U_stat | p_value | p_adj  |
|------|------------|------------|----------------|---------|--------|---------|--------|
| P    | P          | 0.236      | 0.171          | 0.464   | 0.0    | 0.0022  | 0.033  |
| P    | D          | 0.125      | 0.098          | 0.351   | 2.0    | 0.0087  | 0.065  |
| E    | K          | 0.091      | 0.093          | -0.031  | 17.0   | 0.9372  | 0.978  |
| ...  | ...        | ...        | ...            | ...     | ...    | ...     | ...    |
```

---

## 16. Analysis 3: Per-Timepoint Fisher's Exact Test

**Code:** `per_timepoint_fisher()` from `functions_enrichment.py`

**Question:** "At day 5 specifically, is Proline at the P-site more frequent in BWM than control?"

### Algorithm

For each timepoint, for each E/P/A site, for each amino acid:

1. **Pool** counts across the 2 replicates within each condition at this timepoint
2. Build a **2x2 contingency table**
3. Run **Fisher's exact test** (two-sided)
4. Apply **BH-FDR** per timepoint

### Detailed Example

```
Timepoint: day_5
Site: P-site
AA: Proline

Pooled counts:
  Control day_5 (ctrl_rep3 + ctrl_rep4):
    Proline at P-site:     78 + 88 = 166
    Non-Proline at P-site: 402 + 422 = 824
    Total:                 990

  BWM day_5 (bwm_rep3 + bwm_rep4):
    Proline at P-site:     108 + 125 = 233
    Non-Proline at P-site: 362 + 385 = 747
    Total:                 980

2×2 contingency table (rows sorted alphabetically by condition — BWM first):
                  Proline    Not-Proline
  BWM:              233         747         = 980
  Control:          166         824         = 990
                   -----       -----
                    399        1571

Fisher's exact test (OR = (a*d)/(b*c) where table = [[a,b],[c,d]]):
  odds_ratio = (233 × 824) / (747 × 166)
             = 191,992 / 123,902
             = 1.549
  p-value = 8.2e-07

  Since OR > 1: BWM has HIGHER odds of Proline at the P-site than control
  (BWM has 1.55× the odds of Proline stalling at the P-site vs control)
```

### Interpreting the 2x2 table visually

```
                  Pro    ¬Pro
  BWM:           [233]   [747]     Pro rate: 23.8%
  Control:       [166]   [824]     Pro rate: 16.8%
                                            ↑ higher in BWM

  Odds ratio = 1.549 means BWM has 1.55× the odds of Proline vs control
```

### Interpreting the Odds Ratio

Because conditions are sorted alphabetically, **BWM is always row 0** and **control is always row 1** in the 2x2 table. The odds ratio is computed as:

```
OR = (BWM_AA × control_notAA) / (BWM_notAA × control_AA)
```

This means the odds ratio represents **BWM relative to control**:

| Odds Ratio | Meaning |
|------------|---------|
| OR > 1     | The amino acid is **more frequent** in BWM stall sites than in control |
| OR = 1     | No difference between BWM and control |
| OR < 1     | The amino acid is **less frequent** in BWM stall sites than in control |

**Magnitude examples:**
- OR = 2.0 — BWM has 2× the odds of this AA at the site compared to control
- OR = 1.5 — BWM has 50% higher odds than control
- OR = 0.5 — BWM has half the odds of control (equivalently, control has 2× the odds of BWM)
- OR = 0.75 — BWM has 25% lower odds than control

> **Note:** The odds ratio orientation depends entirely on which condition appears first in alphabetical sort. If your conditions were named differently (e.g., "treated" vs "untreated"), the first row — and therefore the reference for OR > 1 — would change accordingly.

### Caveat: Low Replicate Count (n=2)

With more replicates (e.g., n=5+ per condition per timepoint), you would compute per-replicate frequencies and use a replicate-aware test (Wilcoxon rank-sum or t-test) to compare conditions at each timepoint — this explicitly models between-replicate variance. With only n=2 biological replicates per condition per timepoint, there are not enough data points to estimate that variance. Fisher's exact test is a count-based test that works even with n=1, so pooling the 2 replicates together is the pragmatic choice — it increases the total stall site count for more precise frequency estimates, but it cannot account for replicate-level variation (e.g., shared library prep or batch effects). P-values may be too optimistic as a result. This analysis is exploratory — confirming timepoint-specific effects requires more replicates.

### Output DataFrame

```
| timepoint | site | amino_acid | BWM_count | BWM_total | control_count | control_total | odds_ratio | p_value  | p_adj    |
|-----------|------|------------|-----------|-----------|---------------|---------------|------------|----------|----------|
| day_5     | P    | P          | 233       | 980       | 166           | 990           | 1.549      | 8.2e-07  | 4.9e-05  |
| day_5     | P    | D          | 140       | 980       | 105           | 990           | 1.355      | 0.012    | 0.144    |
| day_0     | P    | P          | 165       | 990       | 160           | 1000          | 1.048      | 0.724    | 0.951    |
| day_10    | E    | K          | 72        | 940       | 88            | 960           | 0.823      | 0.215    | 0.645    |
| ...       | ...  | ...        | ...           | ...           | ...       | ...       | ...        | ...      | ...      |
```

---

## 17. Output Files

All files are written to `--out-enrichment` (default: `../ribostall_results/enrichment/`).

### Always produced

| File | Description |
|---|---|
| `stall_sites.csv` | All stall sites across all replicates (long format) |

### Produced with `--enrichment`

| File | Description |
|---|---|
| `replicate_aa_frequencies.csv` | Per-replicate, per-site (E/P/A), per-AA stall counts and frequencies |
| `input_data/{rep}_epa_counts.csv` | Wide-format AA×site count matrix per replicate |
| `input_data/per_group_background_aa.csv` | Background AA counts/frequencies per group |
| `within_condition_enrichment.csv` | Analysis 1 results (binomial test) |
| `between_condition_wilcoxon.csv` | Analysis 2 results (Wilcoxon rank-sum) |
| `per_timepoint_fisher.csv` | Analysis 3 results (Fisher's exact test) |

### Example `replicate_aa_frequencies.csv`

```
replicate,group,condition,timepoint,site,amino_acid,stall_count,total_stall_sites,stall_freq
ctrl_rep1,control_day_0,control,day_0,E,A,45,500,0.090
ctrl_rep1,control_day_0,control,day_0,E,C,12,500,0.024
ctrl_rep1,control_day_0,control,day_0,E,D,38,500,0.076
...
ctrl_rep1,control_day_0,control,day_0,P,A,30,500,0.060
ctrl_rep1,control_day_0,control,day_0,P,P,85,500,0.170    ← Proline enriched at P-site
...
```

### Example `input_data/ctrl_rep1_epa_counts.csv`

```
amino_acid,E,P,A
A,45,30,52
C,12,5,18
D,38,22,35
...
P,28,85,20     ← 85 Proline at P-site vs 28 at E and 20 at A
...
W,3,2,5
Y,8,15,11
```

---

## Summary of Statistical Tests

| Analysis | Test | Compares | Unit of observation | Multiple testing correction |
|---|---|---|---|---|
| **1. Within-condition** | Binomial test | Stall AA freq vs genome background | Pooled counts per group | BH-FDR per group |
| **2. Between-condition** | Wilcoxon rank-sum | Per-replicate frequencies, all timepoints combined | 6 values per condition | BH-FDR globally |
| **3. Per-timepoint** | Fisher's exact | Pooled counts, BWM vs control at each timepoint | 2×2 table per timepoint | BH-FDR per timepoint |
