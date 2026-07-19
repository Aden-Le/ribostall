# scripts/ — Pipeline entry points

*The command-line front door to ribostall: the small set of runnable scripts that turn a `.ribo` file into CDS-aligned coverage, called stall sites, occupancy tables, and the statistics computed over them.*

> **[ribostall](../README.md)** › scripts

These are the CLI scripts a user actually runs — usually not by hand but through the thin launchers under [`../shell_scripts/`](../shell_scripts/README.md), which fill in paths, group specs, and thresholds for a given organism and stage. Every script here is a `main()` wrapped in `argparse`; each inserts the repo root onto `sys.path` (`sys.path.insert(0, ...parent.parent)`) so it can import the shared [`ribostall`](../ribostall/README.md) package, then delegates the real work to that package. The scripts own all user-facing output (`print`, `logging`); the package stays quiet (see the project [`CLAUDE.md`](../CLAUDE.md) — no `print()` inside `ribostall/`).

The pipeline splits deliberately into **call scripts** and **stats scripts**. All the heavy, environment-bound I/O — opening the HDF5 `.ribo` via `ribopy`, reading the reference FASTA, computing P-site offsets and per-transcript backgrounds — lives in the *call* half (`adj_coverage.py`, `stall_sites_*.py` without the `_stats` suffix, `global_codon_occ.py`, `internal_stop_codons.py`). Those write plain CSVs. The *stats* half (`*_stats.py`) consumes only those CSVs, imports no `ribopy`, and therefore runs on a laptop that has never seen the `.ribo` file. This is why, for example, the per-group background frequencies are precomputed and written to `per_group_background_{codon,aa}.csv` by the call script rather than recomputed in the stats script.

---

## Contents

| Script | Role | Details |
|---|---|---|
| [`adj_coverage.py`](#adj_coveragepy) | Ingestion | [→](#adj_coveragepy) |
| [`stall_sites_consensus_union.py`](#stall_sites_consensus_unionpy) | Stall-site calling (consensus, union tx) | [→](#stall_sites_consensus_unionpy) |
| [`stall_sites_consensus_intersection.py`](#stall_sites_consensus_intersectionpy) | Stall-site calling (consensus, intersection tx) | [→](#stall_sites_consensus_intersectionpy) |
| [`stall_sites_non_consensus.py`](#stall_sites_non_consensuspy) | Stall-site calling (per-replicate) | [→](#stall_sites_non_consensuspy) |
| [`stall_sites_consensus_union_stats.py`](#stall_sites_consensus_union_statspy) | Stall-site stats (A1/A4/A7, background-aware) | [→](#stall_sites_consensus_union_statspy) |
| [`stall_sites_consensus_intersection_stats.py`](#stall_sites_consensus_intersection_statspy) | Stall-site stats (A1/A3/A6, Fisher) | [→](#stall_sites_consensus_intersection_statspy) |
| [`stall_sites_non_consensus_stats.py`](#stall_sites_non_consensus_statspy) | Stall-site stats (A2/A5, Wilcoxon) | [→](#stall_sites_non_consensus_statspy) |
| [`global_codon_occ.py`](#global_codon_occpy) | Global occupancy (call) | [→](#global_codon_occpy) |
| [`global_codon_occ_stats.py`](#global_codon_occ_statspy) | Global occupancy (stats) | [→](#global_codon_occ_statspy) |
| [`internal_stop_codons.py`](#internal_stop_codonspy) | Supplementary diagnostic | [→](#internal_stop_codonspy) |

---

## Domain concepts these scripts embody

A short glossary that the sections below assume.

- **P-site offset & CDS-aligned coverage.** A ribosome-protected footprint (a read of length `L`) has its 5′ end some fixed number of nucleotides upstream of the codon in the ribosome's P-site. That number — the **P-site offset** — depends on `L`. "Adjusting" coverage means shifting each length's per-nucleotide read pileup by its offset so that position `i` in the output array corresponds to the codon actually decoded there, then slicing to the CDS (coding sequence). The output of `adj_coverage.py` is a dict `{experiment: {transcript: np.ndarray}}` where the array is CDS-length and P-site (or A-site) aligned.
- **Codonization.** Summing the three in-frame nucleotides of each codon into one number, turning an nt-resolution array into a codon-resolution array.
- **z-score stall calling on the trimmed elongation body.** Within one transcript, the first `--trim-start` codons (the initiation ramp) and last `--trim-stop` codons (the termination region) are excluded, and a per-codon z-score is computed against the body's mean/SD (with a `--pseudocount`). A codon that clears `--min_z` and `--min_reads` is a **stall site**. This is done per replicate.
- **Consensus vs non-consensus.** *Non-consensus* keeps every biological replicate as an independent observation. *Consensus* collapses replicates within a group to a single reproducible stall set (a site must recur in `--stall_min_reps_per_group` replicates within `--tol`, and near-duplicate sites within `--min_sep` are merged, preferring the downstream one). The two **consensus** variants differ only in how the transcript universe is chosen: **union** lets each group keep its own filtered transcript set; **intersection** restricts every group to the transcripts that pass filtering in *all* groups.
- **E/P/A annotation.** A translating ribosome holds three codons: the E (exit) site upstream, the P (peptidyl) site, and the A (aminoacyl) site downstream. Given a stall's codon index, each script derives the codon and amino acid at all three sites (`--basis` sets the register: with `P`, E=−1/P=0/A=+1 codons; with `A`, E=−2/P=−1/A=0).
- **A1–A7 statistical-test taxonomy.** The stats scripts share one enrichment engine but each selects a different subset of tests, keyed to what the upstream design licenses:
  - **A1** — within-condition binomial (observed feature share vs that group's sequence background).
  - **A2** — between-condition Wilcoxon rank-sum on per-replicate frequencies.
  - **A3** — between-condition Fisher's exact on pooled counts.
  - **A4** — between-condition *background-aware* diff (each condition normalized to its own background first).
  - **A5** — between-timepoint Wilcoxon (per-replicate, pooled across conditions).
  - **A6** — between-timepoint Fisher within each condition.
  - **A7** — between-timepoint background-aware diff (pooled across conditions).

  Why the split: *union* backgrounds differ per group, so a raw Fisher would be confounded → it runs the **background-aware** A1/A4/A7. *Intersection* forces one shared transcript universe, making raw shares comparable (Fisher is fair) but making the background identical across groups (background-aware diff degenerates) → it runs **Fisher** A1/A3/A6. *Non-consensus* keeps replicates independent, so it may only run the tests that never pool replicates into a pseudoreplicate → the **Wilcoxons** A2/A5. Occupancy normalizes every condition to one shared transcriptome background, so its background-aware diffs never apply — it runs A1/A2/A3/A5/A6 (no A4/A7).
- **Occupancy normalization (rate / proportion / rpm).** Global occupancy reports each experiment's read counts per codon/AA four ways: `raw` (summed reads), `rate` (raw divided by the transcriptome background count for that codon — a per-occurrence occupancy), `proportion` (each rate divided by the sum of rates across features, so they sum to 1), and `rpm` (raw scaled to reads per million of that experiment's total).

---

## Ingestion

### `adj_coverage.py`

**Purpose.** Step 1 of the pipeline. Reads a `.ribo` file and produces the CDS-aligned, length-summed, P-site- (or A-site-) adjusted coverage dict that every downstream call script consumes. Parallelized across experiments.

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--ribo` | str | — | yes | Path to `.ribo` file. |
| `--min-len` | int | — | yes | Minimum read length, inclusive. |
| `--max-len` | int | — | yes | Maximum read length, inclusive. |
| `--site-type` | str | `None` | no | Site type passed to the offset computation. |
| `--search-window` | 2 ints (`LO HI`) | `None` | no | Position window relative to the landmark for offset search, e.g. `--search-window -60 -30`. |
| `--return-site` | str | — | yes | Adjust coverage to the `P`- or `A`-site (passed through to `get_offset`). |
| `--alias` | flag | `False` | no | Use `apris_human_alias` (set when the `.ribo` uses mouse/human aliasing). |
| `--procs` | int | `1` | no | Number of parallel worker processes (experiments run in parallel). |
| `--batch-size` | int | `0` | no | Transcript batch size to bound memory; `0` = process all transcripts at once per length. |
| `--out` | str | `coverage_bulk_perlen_perexp.pkl.gz` | no | Output gzipped-pickle path. |

**Under the hood**

1. Open a single `Ribo` handle (aliased if `--alias`) to gather metadata: the list of `experiments`, the `transcript_names`, and the CDS-range lookup via `get_cds_range_lookup` (cast to signed ints defensively).
2. For each experiment, precompute a per-length P-site offset table `{L: offset}` by calling `get_offset(ribo0, exp, min_len, max_len, site_type, search_window, return_site)`. These are logged and stashed in `exp_offsets`.
3. Launch a `multiprocessing.Pool` of `--procs` workers. Each worker's initializer (`_init_worker`) opens its own `Ribo` handle once and stashes the shared CDS ranges and transcript list as process globals.
4. Each experiment is one task (`_process_experiment`). It pre-allocates a zero `int64` array per transcript sized to that transcript's CDS window (`stop − start`).
5. It then iterates read **length** by length (bulk per-length I/O: one `get_coverage(range_lower=L, range_upper=L)` HDF5 call per `L`). For each length it applies that length's offset `ps`: for transcript `t` it takes the window `raw[start−ps : stop−ps]`, zero-padded by `_safe_window` if it runs off either end, and adds it in place to the output array. `--batch-size` optionally chunks the transcript loop to bound memory.
6. `starmap` gathers results in order into `{exp: {tx: np.ndarray}}`, which is pickled with `gzip` to `--out`. An empty `.ribo` (no experiments) writes an empty dict and returns.

**Inputs.** One `.ribo` HDF5 file.

**Outputs.** `--out` (default `coverage_bulk_perlen_perexp.pkl.gz`) — a gzipped pickle of `{experiment: {transcript: np.ndarray}}`, each array CDS-length and P/A-aligned.

**Notable algorithms / notes.** The per-length bulk I/O is the key efficiency trick: coverage is read once per read length (not once per transcript), then fanned out to all transcripts. `_safe_window` guarantees the sliced window always matches the pre-allocated CDS length even when the offset pushes it past the transcript boundary, and a final defensive length-match copies into a zero array if shapes still disagree. All keys/values are cast to plain Python ints up front to avoid `uint` overflow when subtracting offsets.

**Related.** Package: [`ribostall.sequence`](../ribostall/README.md) (`get_cds_range_lookup`, `get_offset`, `apris_human_alias`). Runner: [`../shell_scripts/c_elegans/adj_coverage/README.md`](../shell_scripts/c_elegans/adj_coverage/README.md).

---

## Stall-site calling

All three call scripts share the same skeleton — load coverage + `.ribo` + FASTA, filter transcripts, codonize, call stalls per replicate, annotate E/P/A, optionally drop stop-codon windows, write two CSVs plus two per-group background CSVs. They differ in *how the transcript universe and the replicate axis are handled*, which in turn dictates which stats script consumes their output.

### `stall_sites_consensus_union.py`

**Purpose.** Step 2a call (union variant). Detects stall sites with cross-replicate **consensus** while letting **each group keep its own filtered transcript set** (no cross-group intersection). Emits stats-ready E/P/A CSVs plus per-group backgrounds.

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--pickle` | str | — | yes | Path to coverage `pickle.gz`. |
| `--ribo` | str | — | yes | Path to ribo file. |
| `--reference` | str | — | yes | Reference FASTA for CDS-sequence lookup (E/P/A annotation). |
| `--groups` | str | — | yes | Groups, e.g. `'groupA:rep1,rep2;groupB:rep3,rep4'`. |
| `--tx_threshold` | float | `1.0` | no | Minimum reads/nt in the CDS for the transcript filter. |
| `--tx_min_reps_per_group` | str | — | yes | Per-group minimum replicates passing `--tx_threshold`; must name every declared group, e.g. `'control:2;treatment:1'`. |
| `--min_z` | float | `1.0` | no | Minimum z-score to pass as a stall. |
| `--min_reads` | int | `2` | no | Minimum reads to pass as a stall. |
| `--trim-start` | int | `20` | no | Exclude first N codons (initiation ramp). |
| `--trim-stop` | int | `10` | no | Exclude last N codons (termination region). |
| `--pseudocount` | float | `0.5` | no | Pseudocount for stall calling. |
| `--stall_min_reps_per_group` | str | — | yes | Per-group minimum supporting replicates for consensus; must name every declared group, e.g. `'control:2;treatment:1'`. |
| `--tol` | int | `0` | no | Tolerance window for matching sites across replicates. |
| `--min_sep` | int | `7` | no | Minimum separation between consensus sites; prefer downstream when closer. |
| `--psite-offset` | int | `0` | no | Codon offset applied to each stall index before deriving E/P/A. |
| `--basis` | choice `P`/`A` | `P` | no | E/P/A register (`P`: E=−1,P=0,A=+1; `A`: E=−2,P=−1,A=0). |
| `--drop-stop-codons` | choice `True`/`False` | `True` | no | Drop stall windows whose E/P/A hits a stop codon (TAA/TAG/TGA). |
| `--out-dir` | str | `results/stall_sites_consensus_union/raw` | no | Output directory for the CSVs. |

**Under the hood**

1. Parse `--groups` into `{group: [reps]}` and build `rep_to_group`. Parse both `--stall_min_reps_per_group` and `--tx_min_reps_per_group` into per-group ints, validating that every declared group is named (no global fallback) and warning if any threshold exceeds a group's replicate count.
2. Load the coverage pickle; drop any experiments not declared in `--groups`. Open the `Ribo` object, look up CDS ranges (`get_cds_range_lookup`), and load reference sequences (`get_sequence`).
3. Print per-replicate coverage summaries and write a per-replicate coverage-density KDE plot (`plot_coverage_density`) into `--out-dir`.
4. **Transcript filtering (union):** for each group independently, `filter_tx` keeps transcripts with ≥ `--tx_threshold` reads/nt over the trimmed body in ≥ that group's `tx_min_reps`. Each group keeps its own set; no intersection. Coverage is then restricted per replicate to its group's transcripts.
5. **Codonize** every kept array (`codonize_counts_cds`).
6. **Call stalls per replicate** with `call_stalls` (z-score on the trimmed body, gated by `--min_z`/`--min_reads`, `--pseudocount`).
7. **Consensus per group:** `consensus_stalls_across_reps` keeps sites recurring in ≥ that group's `stall_min_reps` within `--tol`, collapsing near-duplicates within `--min_sep`.
8. Flatten to a long dataframe (`consensus_to_long_df`); because consensus yields one set per group, `df["replicate"] = df["group"]` so each group acts as a single "replicate" downstream.
9. **Annotate E/P/A** (`annotate_stalls_epa`) with `--psite-offset` and `--basis`, dropping rows where E/P/A falls outside the CDS or hits an unknown codon. If `--drop-stop-codons True`, also drop windows whose E/P/A codon is a stop.
10. Write the two stall CSVs, then compute **per-group backgrounds** (`background_codon_freq`, `background_aa_freq`) over each group's filtered transcript set and write them.

**Inputs.** Coverage pickle (from `adj_coverage.py`), `.ribo`, reference FASTA.

**Outputs** (into `--out-dir`): `stall_sites_codon.csv`, `stall_sites_aa.csv`, `per_group_background_codon.csv`, `per_group_background_aa.csv`, `coverage_density.png`.

**Notable algorithms / notes.** Union backgrounds *differ per group* (each group has its own transcript set), which is exactly why the matching stats script runs the **background-aware** A1/A4/A7 rather than Fisher — a raw Fisher across differing transcript universes would confound a shift in the expressed transcriptome with differential stalling.

**Related.** Package: [`ribostall.stall_sites`](../ribostall/README.md), [`ribostall.amino_acids`](../ribostall/README.md), [`ribostall.sequence`](../ribostall/README.md), [`ribostall.enrichment`](../ribostall/README.md). Stats sibling: [`stall_sites_consensus_union_stats.py`](#stall_sites_consensus_union_statspy). Runner: [`../shell_scripts/c_elegans/stall_sites_consensus_union/README.md`](../shell_scripts/c_elegans/stall_sites_consensus_union/README.md), [`../shell_scripts/mouse/stall_sites_consensus_union/README.md`](../shell_scripts/mouse/stall_sites_consensus_union/README.md).

---

### `stall_sites_consensus_intersection.py`

**Purpose.** Step 2a call (intersection variant). Identical to the union script *except* that after per-group filtering it restricts **every** group to the transcripts that pass filtering in **all** groups, so all conditions share one transcript universe.

**Arguments.** Identical set to [`stall_sites_consensus_union.py`](#stall_sites_consensus_unionpy), with one difference: `--out-dir` defaults to `results/stall_sites_consensus_intersection/raw`. (Same flags, types, defaults, and required-ness for everything else.)

**Under the hood.** Steps 1–3 and 5–10 are identical to the union script. The only divergence is step 4:

4. **Transcript filtering (intersection):** filter each group independently with `filter_tx`, then compute `common_txs = ∩ (filtered sets)` and reassign *every* group's transcript set to that shared intersection. A warning is logged if the intersection is empty (no stalls will be called). All groups' coverage is then restricted to `common_txs`.

Because the background frequencies in step 10 are computed over this shared set, they come out **identical across groups**.

**Inputs / Outputs.** Same as the union script but under `results/stall_sites_consensus_intersection/raw` by default.

**Notable algorithms / notes.** The shared transcript universe makes between-condition stall counts apples-to-apples, so the sibling stats script runs **Fisher** (A1/A3/A6). The identical backgrounds make the background-aware diff degenerate (the shared background cancels), which is why the background-aware tests live only in the union stats script.

**Related.** Same package modules as the union script. Stats sibling: [`stall_sites_consensus_intersection_stats.py`](#stall_sites_consensus_intersection_statspy). Runner: [`../shell_scripts/c_elegans/stall_sites_consensus_intersection/README.md`](../shell_scripts/c_elegans/stall_sites_consensus_intersection/README.md), [`../shell_scripts/mouse/stall_sites_consensus_intersection/README.md`](../shell_scripts/mouse/stall_sites_consensus_intersection/README.md).

---

### `stall_sites_non_consensus.py`

**Purpose.** Step 2b call. Calls stall sites **per replicate** and keeps every replicate as an independent row — no consensus collapse. The style reference for Python conventions in this repo.

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--pickle` | str | — | yes | Path to coverage `pickle.gz`. |
| `--ribo` | str | — | yes | Path to ribo file. |
| `--reference` | str | — | yes | Reference FASTA for CDS-sequence lookup. |
| `--groups` | str | — | yes | Groups, e.g. `'groupA:rep1,rep2;groupB:rep3,rep4'`. |
| `--tx_threshold` | float | `1.0` | no | Minimum reads/nt in the CDS for the transcript filter. |
| `--tx_min_reps_per_group` | str | — | yes | Per-group minimum replicates passing `--tx_threshold`; must name every declared group. |
| `--min_z` | float | `2.0` | no | Minimum z-score to pass as a stall. **(Note: default 2.0, higher than the consensus scripts' 1.0.)** |
| `--min_reads` | int | `5` | no | Minimum reads to pass as a stall. **(Note: default 5, higher than the consensus scripts' 2.)** |
| `--trim-start` | int | `20` | no | Exclude first N codons (initiation ramp). |
| `--trim-stop` | int | `10` | no | Exclude last N codons (termination region). |
| `--pseudocount` | float | `0.5` | no | Pseudocount for stall calling. |
| `--psite-offset` | int | `0` | no | Codon offset applied before deriving E/P/A. |
| `--basis` | choice `P`/`A` | `P` | no | E/P/A register. |
| `--drop-stop-codons` | choice `True`/`False` | `True` | no | Drop stall windows whose E/P/A hits a stop codon. |
| `--out-dir` | str | `results/stall_sites_non_consensus/raw` | no | Output directory. |

Note there is **no** `--stall_min_reps_per_group`, `--tol`, or `--min_sep` here — those govern consensus, which this script does not perform for the output (only for a diagnostic; see below).

**Under the hood**

1. Parse `--groups` and `--tx_min_reps_per_group` (per-group, validated as in the consensus scripts).
2. Load coverage, `Ribo`, CDS ranges, and reference sequences. Report per-replicate **body** coverage (trimming `--trim-start`/`--trim-stop` codons, i.e. `×3` nt) and write the coverage-density plot.
3. **Transcript filtering (per-group, no intersection):** `filter_tx` per group; each group keeps its own set.
4. **Sequence-resolution sanity check:** confirm every filtered transcript resolves to both a CDS range and a FASTA sequence (accounting for `|`-delimited transcript names), warning otherwise.
5. Restrict coverage per replicate to its group's transcripts, **codonize**, and **call stalls per replicate** (`call_stalls`).
6. **Reproducibility diagnostic** (logging only): for groups with ≥2 replicates, compute the consensus fraction (`consensus_stalls_across_reps` at `min_support=2, tol=0`) versus the union of sites, printing the % reproducible. This does not affect the output — every per-replicate stall is kept.
7. Flatten with `stalls_to_long_df` (keeping the replicate identity), annotate E/P/A, optionally drop stop-codon windows.
8. Write the two stall CSVs and the two per-group background CSVs (over each group's filtered set).

**Inputs.** Coverage pickle, `.ribo`, reference FASTA.

**Outputs** (into `--out-dir`): `stall_sites_codon.csv`, `stall_sites_aa.csv`, `per_group_background_codon.csv`, `per_group_background_aa.csv`, `coverage_density.png`.

**Notable algorithms / notes.** Because replicates stay independent, the only statistically valid tests are ones that never pool replicates into a pseudoreplicate — the **Wilcoxons** (A2/A5). The count-collapsing tests are structurally impossible here and live in the consensus stats scripts. The higher `--min_z`/`--min_reads` defaults reflect that per-replicate calls are noisier than consensus calls.

**Related.** Package: same modules as the consensus scripts, plus `stalls_to_long_df`. Stats sibling: [`stall_sites_non_consensus_stats.py`](#stall_sites_non_consensus_statspy). Runner: [`../shell_scripts/c_elegans/stall_sites_non_consensus/README.md`](../shell_scripts/c_elegans/stall_sites_non_consensus/README.md), [`../shell_scripts/mouse/stall_sites_non_consensus/README.md`](../shell_scripts/mouse/stall_sites_non_consensus/README.md).

---

## Stall-site stats

Each stats script consumes **one** of the two CSVs written by its sibling call script (run it twice — once for `stall_sites_codon.csv`, once for `stall_sites_aa.csv`); the level (codon vs AA) is auto-detected from the columns via `detect_level`, and outputs are suffixed `_codon`/`_aa`. None of these import `ribopy`. Each has per-analysis `--<name> true|false` toggles (default `true`); a skipped analysis is announced and writes no CSV.

### `stall_sites_consensus_union_stats.py`

**Purpose.** Runs the **background-aware** count-collapsing tests on a union-consensus CSV: **A1** (within-condition binomial), **A4** (between-condition background-aware diff), **A7** (between-timepoint background-aware diff, pooled across conditions).

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--stall-sites` | str | — | yes | Path to `stall_sites_codon.csv` or `stall_sites_aa.csv`. |
| `--groups` | str | — | yes | Groups; consensus sets replicate == group name, e.g. `'control:control;treatment:treatment'`, or timepoint-carrying names like `'treatment_day_0:treatment_day_0;...'`. |
| `--background` | str | — | yes | Path to `per_group_background_{level}.csv` from the call script. |
| `--out-dir` | str | `results/stall_sites_consensus_union/analysis` | no | Output directory. |
| `--headline-condition` | str | `None` | no | Condition treated as numerator in A4; positive `delta_log2_enrichment` = more enriched vs background here. Default alphabetical. A7 is fixed later-vs-earlier and ignores this. |
| `--timepoints` | str | `None` | no | Comma-separated chronological labels, e.g. `'day_0,day_5,day_10'`. With ≥2, A4 slices per-timepoint and A7 runs across day-pairs; without it, A4 is one comparison and A7 is skipped. **Not** sorted automatically. |
| `--within-condition` | choice `true`/`false` | `true` | no | A1 toggle. |
| `--between-condition-background-diff` | choice `true`/`false` | `true` | no | A4 toggle. |
| `--between-timepoint-background-diff` | choice `true`/`false` | `true` | no | A7 toggle (needs `--timepoints` ≥2). |

**Under the hood**

1. Read the CSV; `detect_level` returns the level, the three site columns, the alphabet (61 sense codons or the AA order), and the feature column name.
2. Parse `--groups`; derive `rep_to_condition` (condition = group name before the first `_`). If `--timepoints`, build and validate the timepoint mapping (no auto-sort; warn about undeclared timepoints).
3. `build_replicate_counts` tallies per-"replicate" (= group) counts of each feature at each of E/P/A. Load and reindex the per-group background frequencies from `--background`.
4. **A1:** `within_condition_enrichment` — binomial test of each group's observed feature share at each site against that group's background frequency; FDR-adjusted.
5. **A4:** `between_condition_background_diff` — each condition normalized to its own background, then compared (`delta_log2_enrichment`). With `--timepoints`, slice to one timepoint at a time, feeding that timepoint's own per-condition background (FDR per (timepoint, site)), and tag rows with a `timepoint` column.
6. **A7** (timepoints only): pool replicate counts *across conditions* within each timepoint, build a count-weighted pooled background per timepoint, and run `between_timepoint_background_diff` for each later-vs-earlier day-pair (`build_timepoint_pairs`), stacking into one CSV tagged by a `comparison` column.

**Inputs.** One stall CSV + the matching `per_group_background_{level}.csv`.

**Outputs** (into `--out-dir`, `{suffix}` = `codon` or `aa`): `within_condition_binomial_{suffix}.csv` (A1); A4 writes `per_timepoint_background_diff_{suffix}.csv` (with `--timepoints`) or `between_condition_background_diff_{suffix}.csv` (flat); A7 writes `between_timepoint_background_diff_{suffix}.csv`.

**Notable algorithms / notes.** The consensus design has one set per (condition, timepoint) cell, so pooling at n=1 is a no-op, not a pseudoreplicate — hence these count-collapsing tests are legitimate here but forbidden on per-replicate (non-consensus) data. `--timepoints` is optional: the union consensus is not flat-only.

**Related.** Package: [`ribostall.enrichment`](../ribostall/README.md), [`ribostall.stats_cli`](../ribostall/README.md). Call sibling: [`stall_sites_consensus_union.py`](#stall_sites_consensus_unionpy). Plots: [`../R_scripts/README.md`](../R_scripts/README.md). Runner: [`../shell_scripts/c_elegans/stall_sites_consensus_union/README.md`](../shell_scripts/c_elegans/stall_sites_consensus_union/README.md).

---

### `stall_sites_consensus_intersection_stats.py`

**Purpose.** Runs the **Fisher** count-collapsing tests on an intersection-consensus CSV: **A1** (within-condition binomial), **A3** (between-condition Fisher's exact), **A6** (between-timepoint Fisher within each condition).

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--stall-sites` | str | — | yes | Path to `stall_sites_codon.csv` or `stall_sites_aa.csv`. |
| `--groups` | str | — | yes | Groups; consensus sets replicate == group name (flat or timepoint-carrying). |
| `--background` | str | `None` | **conditional** | `per_group_background_{level}.csv`. Used **only** by A1; Fisher (A3/A6) needs none. Required only when A1 runs (the default); a Fisher-only run may omit it. |
| `--out-dir` | str | `results/stall_sites_consensus_intersection/analysis` | no | Output directory. |
| `--headline-condition` | str | `None` | no | Numerator condition for A3; positive log2 odds ratio = enriched here. Validated up front against the two conditions. A6 ignores it. |
| `--timepoints` | str | `None` | no | Chronological labels. With ≥2, A3 slices per-timepoint and A6 runs across day-pairs; without it, A3 is one Fisher and A6 is skipped. Not auto-sorted. |
| `--within-condition` | choice `true`/`false` | `true` | no | A1 toggle. |
| `--between-condition-fisher` | choice `true`/`false` | `true` | no | A3 toggle. |
| `--between-timepoint-fisher` | choice `true`/`false` | `true` | no | A6 toggle (needs `--timepoints` ≥2). |

**Under the hood**

1. **Fail fast:** if `--within-condition true` (default) but `--background` is `None`, exit with a clear message — A1 needs the background; a Fisher-only run (`--within-condition false`) may omit it.
2. Read the CSV and detect level. Parse groups, derive conditions, validate `--headline-condition` (must name one of the two conditions) and `--timepoints` up front.
3. `build_replicate_counts` as above. Load the per-group backgrounds **only if A1 runs** (skipped entirely otherwise — the Fisher tests never touch them).
4. **A1:** `within_condition_enrichment` (identical to the union script's A1).
5. **A3:** with `--timepoints`, `per_timepoint_fisher` (one Fisher per (timepoint, feature, site), tagged by `timepoint`); flat, `between_condition_fisher` (pooled across the whole condition). Positive log2 odds ratio = enriched in the headline condition.
6. **A6** (timepoints only): for each later-vs-earlier day-pair, `between_timepoint_fisher_within_condition` compares timepoints *within* each condition; each pair is written to its own CSV.

**Inputs.** One stall CSV; the background CSV only when A1 runs.

**Outputs** (into `--out-dir`): `within_condition_binomial_{suffix}.csv` (A1); A3 writes `per_timepoint_fisher_{suffix}.csv` (timepoints) or `between_condition_fisher_{suffix}.csv` (flat); A6 writes one `timepoint_fisher_within_condition_{tag}_{suffix}.csv` per day-pair.

**Notable algorithms / notes.** The A1/A3/A6 labels are this pipeline's own IDs and do **not** map onto the "Analysis 1/2/…/7" section numbering inside `ribostall/enrichment.py` (that module numbers its functions independently and is shared across all three stall stats scripts). Cross-reference by function name, not number. The intersection's shared transcript universe is what makes Fisher a fair between-group test.

**Related.** Package: [`ribostall.enrichment`](../ribostall/README.md), [`ribostall.stats_cli`](../ribostall/README.md). Call sibling: [`stall_sites_consensus_intersection.py`](#stall_sites_consensus_intersectionpy). Plots: [`../R_scripts/README.md`](../R_scripts/README.md). Runner: [`../shell_scripts/c_elegans/stall_sites_consensus_intersection/README.md`](../shell_scripts/c_elegans/stall_sites_consensus_intersection/README.md).

---

### `stall_sites_non_consensus_stats.py`

**Purpose.** Runs **only** the two per-replicate Wilcoxon tests that never collapse replicate counts: **A2** (between-condition Wilcoxon rank-sum) and **A5** (between-timepoint Wilcoxon, pooled across conditions). No background is read.

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--stall-sites` | str | — | yes | Path to `stall_sites_codon.csv` or `stall_sites_aa.csv`. |
| `--groups` | str | — | yes | Groups, e.g. `'groupA:rep1,rep2;groupB:rep3,rep4'` (real per-replicate groups here, not group==name). |
| `--out-dir` | str | `results/stall_sites_non_consensus/analysis` | no | Output directory. |
| `--headline-condition` | str | `None` | no | Numerator/direction reference for A2; positive `log2_FC` = higher per-replicate stall frequency here. Default alphabetical. A5 is fixed later-vs-earlier and ignores it. |
| `--timepoints` | str | `None` | no | Chronological labels. With ≥2, A5 runs over each later-vs-earlier day-pair; without it, A5 is skipped and only A2 runs. Not auto-sorted. |
| `--between-condition-wilcoxon` | choice `true`/`false` | `true` | no | A2 toggle. |
| `--between-timepoint-wilcoxon` | choice `true`/`false` | `true` | no | A5 toggle (needs `--timepoints` ≥2). |

Note: **no** `--background` argument — neither Wilcoxon uses one.

**Under the hood**

1. Read the CSV, detect level, parse groups, derive conditions and (optional) timepoints.
2. `build_replicate_counts` on the *real* per-replicate rows.
3. **Feasibility gate** (Decision 9): the Wilcoxons need ≥2 replicates per (condition, timepoint) cell. `min_reps_per_cell = min(len(reps))`; non-consensus normally satisfies this, so a skip here signals misuse (e.g. n=1 consensus data fed in by mistake).
4. **A2:** `between_condition_wilcoxon` — rank-sum on per-replicate feature frequencies between the two conditions, FDR-adjusted.
5. **A5** (timepoints only): for each later-vs-earlier day-pair (`build_timepoint_pairs`), `between_timepoint_wilcoxon` pools replicates *across conditions* within each timepoint; one CSV per pair.

**Inputs.** One stall CSV (per-replicate). No background.

**Outputs** (into `--out-dir`): `between_condition_wilcoxon_{suffix}.csv` (A2); one `between_timepoint_wilcoxon_{tag}_{suffix}.csv` per day-pair (A5).

**Notable algorithms / notes.** This script contains no count-collapsing code at all — pseudoreplication is impossible by construction. That structural split is the whole reason the pipeline separates non-consensus (Wilcoxon-only) from consensus (count-collapsing) stats.

**Related.** Package: [`ribostall.enrichment`](../ribostall/README.md), [`ribostall.stats_cli`](../ribostall/README.md). Call sibling: [`stall_sites_non_consensus.py`](#stall_sites_non_consensuspy). Plots (A2/A5 Wilcoxon bar plots): [`../R_scripts/README.md`](../R_scripts/README.md). Runner: [`../shell_scripts/c_elegans/stall_sites_non_consensus/README.md`](../shell_scripts/c_elegans/stall_sites_non_consensus/README.md).

---

## Global occupancy

### `global_codon_occ.py`

**Purpose.** Step 3 call. Computes **global** (transcriptome-wide) codon and amino-acid occupancy per experiment at each of the E/P/A sites, normalized four ways (raw / rate / proportion / rpm) against a shared transcriptome background.

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--ribo` | str | — | yes | Path to `.ribo` file. |
| `--pickle` | str | — | yes | Gzipped pickle of coverage `{exp: {tx: np.ndarray}}` (CDS coverage). |
| `--reference` | str | — | yes | Reference FASTA/2bit used by `get_sequence()`. |
| `--out-dir` | str | `results/global_occupancy` | no | Output directory (CSVs go under `raw/`). |
| `--trim-start` | int | `0` | no | Exclude the first N codons of each CDS. |
| `--trim-stop` | int | `0` | no | Exclude the last N codons of each CDS. |
| `--use-human-alias` | flag | `False` | no | Use `apris_human_alias` when opening the Ribo file. |
| `--drop-stop-codons` | choice `True`/`False` | `True` | no | Exclude stop codons before computing occupancy (absent from CSVs and all totals/rates/proportions/rpm). |
| `--groups` | str | `None` | no | `group:rep1,rep2;...` used only to filter the coverage dict to declared replicates. |

**Under the hood**

1. Open the `Ribo` object (aliased if `--use-human-alias`), load the coverage pickle, and optionally filter to `--groups` replicates.
2. Look up CDS ranges and reference sequences. Build the set of stop codons to exclude (empty if `--drop-stop-codons False`).
3. **Background:** iterate once over every transcript's CDS, walking codons in-frame with `iter_trimmed_codons` (respecting `--trim-start`/`--trim-stop`), accumulating `transcriptome_codon_counts` (skipping stops when configured).
4. **Per-experiment occupancy:** for each experiment × transcript, and for each site E/P/A, call `iter_trimmed_site_counts(cds_seq, cov, trim_start, trim_stop, SITE_SHIFT[site])`. `SITE_SHIFT` is `{E:−3, P:0, A:+3}` nt: for a ribosome whose P-site sits at a CDS position, the read count in that P-site window is attributed to the codon *identity* shifted by the site offset. Sum counts per (experiment, site, site-codon), skipping zeros and stops.
5. Aggregate the transcriptome background to the AA level (`aggregate_to_aa`, a shared background).
6. For each site, build the codon-level frame (`build_codon_df`) and the AA-level frame (`build_aa_df`, codon occupancy re-aggregated per experiment). Each frame carries, per experiment, the four normalizations: `raw`, `rate` (raw ÷ background count), `proportion` (rate ÷ Σ rates), `rpm` (raw ÷ experiment total × 1e6).

**Inputs.** `.ribo`, coverage pickle, reference FASTA.

**Outputs** (into `--out-dir/raw/`): `codon_occupancy_E.csv`, `codon_occupancy_P.csv`, `codon_occupancy_A.csv`, `aa_occupancy_E.csv`, `aa_occupancy_P.csv`, `aa_occupancy_A.csv`. Codon frames carry columns `Codon, AminoAcid, Transcriptome, {exp}_raw, {exp}_rate, {exp}_proportion, {exp}_rpm`; AA frames drop `Codon`.

**Notable algorithms / notes.** Occupancy normalizes every experiment/condition to *one shared transcriptome background*, which is why the downstream stats have no background-aware diff (A4/A7): there is nothing per-group to normalize against differently.

**Related.** Package: [`ribostall.global_occupancy`](../ribostall/README.md) (`iter_trimmed_codons`, `iter_trimmed_site_counts`, `parse_groups`, `aggregate_to_aa`), [`ribostall.sequence`](../ribostall/README.md), [`ribostall.amino_acids`](../ribostall/README.md). Stats sibling: [`global_codon_occ_stats.py`](#global_codon_occ_statspy). Runner: [`../shell_scripts/c_elegans/global_occupancy/README.md`](../shell_scripts/c_elegans/global_occupancy/README.md).

---

### `global_codon_occ_stats.py`

**Purpose.** Step 3 stats. Runs the statistical tests on the raw occupancy CSVs for **one level** (codon or AA), processing all E/P/A sites in a single invocation and writing a merged `analysis/` tree with a prepended `site` column. The per-site frames are an internal intermediate and are **not** exported.

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--raw-dir` | str | `results/global_occupancy/raw` | no | Directory of raw occupancy CSVs (reads `{level}_occupancy_{site}.csv`). |
| `--analysis-dir` | str | `results/global_occupancy/analysis` | no | Output directory for merged CSVs. |
| `--level` | choice `codon`/`aa` | — | yes | Occupancy level to process. |
| `--sites` | 1+ strings | `["E", "P", "A"]` | no | Sites to process, in concatenation order for the merged tree. |
| `--groups` | str | — | yes | Groups, e.g. `'groupA:rep1,rep2;groupB:rep3,rep4'`. |
| `--headline-condition` | str | `None` | no | Numerator for A2 (positive `log2_FC` = higher occupancy here) and A3 (positive log2 odds ratio = enriched here). Default alphabetical. |
| `--timepoints` | str | `None` | no | Chronological labels. Declares a timepoint axis: A3 slices per-tp, A5/A6 run per day-pair; when absent A5/A6 are skipped and A3 is one pooled Fisher. Not auto-sorted. |
| `--within-condition` | choice `true`/`false` | `true` | no | A1 toggle. |
| `--between-condition-wilcoxon` | choice `true`/`false` | `true` | no | A2 toggle (auto-skips when n<2/condition). |
| `--between-condition-fisher` | choice `true`/`false` | `true` | no | A3 toggle (per-tp when `--timepoints`). |
| `--between-timepoint-wilcoxon` | choice `true`/`false` | `true` | no | A5 toggle (needs `--timepoints` ≥2). |
| `--between-timepoint-fisher` | choice `true`/`false` | `true` | no | A6 toggle (needs `--timepoints` ≥2). |

**Under the hood**

1. Parse groups; derive `rep_to_group`. Timepoint handling: if `--timepoints` is absent but any group name contains `_`, **exit** (the names look like they carry timepoints); otherwise `rep_to_condition = rep_to_group` and A5/A6 are skipped. If `--timepoints` is present, split conditions off the group names, build/validate `rep_to_timepoint`.
2. Create a **temp** per-site directory (`tempfile.mkdtemp`) — the per-site analysis CSVs are an intermediate, never exported.
3. For each site in `--sites`, read `{level}_occupancy_{site}.csv`, auto-detect the feature column (`Codon` → `codon`, `AminoAcid` → `amino_acid`), and **fail fast** if any declared replicate's `{rep}_raw` column is missing (a `--groups`/`--raw-dir` mismatch). Build the stats-input dicts: `raw_for_stats` from `{rep}_raw`, `rates_for_stats` from `{rep}_proportion`, and the transcriptome frequencies from the `Transcriptome` column.
4. Run the selected analyses per site (`run_site_analyses`):
   - **A1** `within_condition_binomial_occupancy`.
   - **A2** `between_condition_wilcoxon_occupancy` — feasible only with exactly two conditions each having ≥2 reps.
   - **A3** `per_timepoint_fisher_occupancy` (with timepoints) or `between_condition_fisher_occupancy` (pooled). Prints a pseudoreplication warning — pooling biological replicates makes p-values anti-conservative.
   - **A5** `between_timepoint_wilcoxon_occupancy` per day-pair (skips a pair with <2 reps in a timepoint).
   - **A6** `between_timepoint_fisher_within_condition` per day-pair (also warns about pseudoreplication).
5. **Merge:** for each analysis basename, re-read the per-site CSVs *from disk*, prepend a `site` column, and concatenate in `--sites` order (E→P→A) into `--analysis-dir`. Re-reading (rather than reusing in-memory frames) reproduces the old two-step merge byte-for-byte. Finally the temp per-site dir is deleted.

**Inputs.** The six raw occupancy CSVs under `--raw-dir` for the chosen `--level`.

**Outputs** (into `--analysis-dir`, one merged CSV per analysis that ran, each prefixed by `{level}_` and carrying a `site` column): `{level}_within_condition_binomial.csv` (A1); `{level}_wilcoxon_condition.csv` (A2); `{level}_per_timepoint_fisher.csv` or `{level}_between_condition_fisher.csv` (A3); `{level}_wilcoxon_timepoint_{tag}.csv` (A5); `{level}_timepoint_fisher_within_condition_{tag}.csv` (A6).

**Notable algorithms / notes.** The A-numbering skips A4/A7 because occupancy uses one shared transcriptome background, so the background-aware diffs the stall pipelines run as A4/A7 do not apply. The Fisher analyses print explicit pseudoreplication warnings because, unlike the consensus stall data, occupancy pooling of biological replicates is a real pseudoreplicate (p-values anti-conservative).

**Related.** Package: [`ribostall.global_occupancy`](../ribostall/README.md), [`ribostall.stats_cli`](../ribostall/README.md). Call sibling: [`global_codon_occ.py`](#global_codon_occpy). Plots: [`../R_scripts/README.md`](../R_scripts/README.md). Runner: [`../shell_scripts/c_elegans/global_occupancy/README.md`](../shell_scripts/c_elegans/global_occupancy/README.md).

---

## Supplementary

### `internal_stop_codons.py`

**Purpose.** A read-only **diagnostic** (it does not feed the occupancy or stall-site pipelines). It finds transcripts whose CDS carries an in-frame stop codon *inside* the coding body — i.e. a stop that survives after the terminal stop is trimmed off — and pulls the P-site read count at each such stop as evidence it is translated through. Internal stops are candidates for translational recoding (TGA→selenocysteine, TAG→pyrrolysine) or annotation problems.

**Arguments**

| Flag | Type | Default | Required | Description |
|---|---|---|---|---|
| `--ribo` | str | — | yes | Path to `.ribo` file. |
| `--pickle` | str | — | yes | Gzipped pickle of coverage `{rep: {tx: np.ndarray}}` (CDS coverage). |
| `--reference` | str | — | yes | Reference FASTA used by `get_sequence()`. |
| `--out-dir` | str | `results/diagnostics/internal_stops` | no | Output directory. |
| `--trim-start` | int | `0` | no | Exclude the first N codons before scanning. |
| `--trim-stop` | int | `1` | no | Exclude the last N codons before scanning. The terminal stop is the last codon, so the default `1` trims it off; pass `0` to keep it (then every transcript is reported). |
| `--use-human-alias` | flag | `False` | no | Use `apris_human_alias` when opening the Ribo file. |
| `--groups` | str | `None` | no | `group:rep1,rep2;...` used only to filter the coverage dict to declared replicates. |

**Under the hood**

1. Set up logging to both console and a dated log file under `scripts/logs/` (the shared repo format).
2. Open the `Ribo` object, load coverage, optionally filter to `--groups` replicates, and look up CDS ranges + sequences.
3. **Scan:** for every CDS, walk codons in-frame with `iter_trimmed_codons` (dropping the first `--trim-start` and last `--trim-stop` codons); "internal" is defined purely by the trim window, so any stop it still yields sits in the body. Record `(codon_index, nt_index, stop_codon)` per hit.
4. **Coverage evidence:** for each internal stop at CDS nt index `i`, sum `cov[i:i+3]` per replicate (the same P-site window model as the occupancy pipeline; guarded against short/missing arrays).
5. Build a long table (one row per occurrence, with `frac_position`, `codons_to_end`, the recoding annotation, per-replicate reads, and a total) and a per-transcript summary. Both are sorted most-covered-first (strongest read-through candidates on top).

**Inputs.** `.ribo`, coverage pickle, reference FASTA.

**Outputs** (into `--out-dir`): `internal_stop_codons_long.csv` (one row per internal-stop occurrence) and `internal_stop_codons_by_transcript.csv` (one row per transcript). Also a dated log under `scripts/logs/` and a console scan summary.

**Notable algorithms / notes.** Folding "trim off the terminal stop" and "find the leftover stops" into one operation (via the trim window) keeps the logic clean rather than special-casing the last codon. `STOP_RECODING` annotates TGA and TAG; TAA has no documented recoding and is reported unannotated.

**Related.** Package: [`ribostall.global_occupancy`](../ribostall/README.md) (`iter_trimmed_codons`, `parse_groups`), [`ribostall.sequence`](../ribostall/README.md), [`ribostall.amino_acids`](../ribostall/README.md) (`STOP_CODONS`). Runner: [`../shell_scripts/c_elegans/internal_stop_codons/README.md`](../shell_scripts/c_elegans/internal_stop_codons/README.md).

---

## See also

- [`../README.md`](../README.md) — repository overview and pipeline map.
- [`../ribostall/README.md`](../ribostall/README.md) — the shared analysis package these scripts drive (coverage, stall calling, enrichment, occupancy, stats kernels, CLI helpers).
- [`../R_scripts/README.md`](../R_scripts/README.md) — the R plotting layer that visualizes the `analysis/` CSVs (volcano and bar plots).
- [`../shell_scripts/README.md`](../shell_scripts/README.md) — the per-organism, per-stage Bash launchers that invoke these scripts with concrete arguments.
