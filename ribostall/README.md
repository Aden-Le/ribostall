# ribostall/ — Shared analysis package

*The importable core library of the ribostall ribosome-profiling pipeline: sequence and offset extraction, stall calling, motif/E-P-A analysis, and the statistical kernels shared by both the stall-site and global-occupancy pipelines.*

> **[ribostall](../README.md)** › ribostall (package)

---

## What this package is

`ribostall/` is the reusable library the pipeline is built on. Everything with a
CLI — argument parsing, file I/O, logging, and every `print()` — lives one level
up in [`../scripts/`](../scripts/README.md). This package is meant to be
*imported*, not run.

That split is a hard project rule: **no `print()` may appear inside
`ribostall/`.** The functions here are side-effect-free with respect to
user-facing output, with exactly one sanctioned exception — a `logging.warning`
(or higher) for a data-integrity check the caller cannot reasonably perform
itself (e.g. a CDS/coverage length mismatch detected inside a generator, in
`global_occupancy.iter_trimmed_site_counts`). Informational logging
(`logging.info`, `logging.debug`) and all normal output belong in the scripts.
`stats_cli` follows the same discipline for errors: it raises `ValueError`
rather than calling `sys.exit`, leaving the exit decision to the script.

The one wrinkle worth flagging up front: `amino_acids.py` and
`sequence.get_offset`/`windows_aa` were written before this rule hardened and
still contain a few `print()`/plotting calls. Treat those as legacy — new code
in this package must not add output.

---

## Architecture / import graph

The package is layered. The bottom layer is pure data (a FASTA reader and a
codon table); the middle layer turns coverage arrays into stall calls and E/P/A
annotations; the top layer is the statistics, where a single kernel module
(`stats_core`) is the source of truth for every test, and two thin *adapter*
modules wrap those kernels into the exact column layouts each pipeline emits.

```
                      ┌───────────────────────────────────────────┐
   data / reference   │  fasta.py            amino_acids.py        │
                      │  (FastaFile,          (CODON2AA, AA_ORDER,  │
                      │   FastaEntry)          SENSE_CODONS, …,     │
                      │      │                 PWM/logo, EPA count) │
                      │      ▼                                      │
                      │  sequence.py  ◄── ribopy (.ribo handles)    │
                      │  (get_sequence, get_cds_range_lookup,       │
                      │   get_offset — metagene peak calibration)   │
                      └───────────────────────────────────────────┘
                                     │
   calling / shaping   ┌─────────────▼─────────────┐
                       │  stall_sites.py           │
                       │  (filter_tx, codonize,    │
                       │   call_stalls, consensus, │
                       │   *_to_long_df)           │
                       └─────────────┬─────────────┘
                                     │
   statistics          ┌────────────▼────────────────────────────┐
                       │  stats_core.py   ← SINGLE SOURCE OF TRUTH │
                       │  binom_row · wilcoxon_row · fisher_row ·  │
                       │  background_diff_row · bh_fdr             │
                       └───────┬───────────────────────┬──────────┘
                               │                       │
                     ┌─────────▼────────┐    ┌─────────▼─────────────┐
                     │  enrichment.py   │    │  global_occupancy.py  │
                     │  (stall-site     │    │  (occupancy adapters  │
                     │   adapters A1–A7)│    │   A1–A6 + coverage     │
                     └──────────────────┘    │   iterators)          │
                                             └───────────────────────┘

   shared CLI helpers   stats_cli.py  (parse_groups, parse_timepoints,
                        build_timepoint_pairs, detect_level, build_replicate_counts …)
                        — imported by both adapters and by the stats scripts
```

Two things make this graph work:

- **`stats_core` is imported by both adapters.** `enrichment.py` (stall sites)
  and `global_occupancy.py` (occupancy) used to re-implement the same three
  tests with slightly different scaffolding. The per-row math now lives once in
  `stats_core`; the adapters differ only in how they pool counts and name
  columns. A fix to the Fisher table, the FDR family, or the background-diff
  math lands in every consumer at once.
- **`stats_cli` is the single source of truth for CLI parsing.** `parse_groups`
  is even re-exported from `global_occupancy` so old
  `from ribostall.global_occupancy import parse_groups` imports keep working.
  Likewise `bh_fdr` is re-exported from both `enrichment` and `global_occupancy`.

---

## Contents

| Module | Responsibility | Details |
|---|---|---|
| `__init__.py` | Package marker — intentionally empty; import the modules below directly (e.g. `from ribostall.stats_core import binom_row`). | — |
| `fasta.py` | Streaming FASTA reader/writer (`.gz`-aware), a hand-rolled parse state machine | [→](#fastapy) |
| `sequence.py` | Pull transcript sequences and CDS ranges from a `.ribo` handle; calibrate per-read-length P/A-site offsets from metagene peaks | [→](#sequencepy) |
| `amino_acids.py` | The codon table and AA-class constants; translation, background frequencies, AA windows, E/P/A triplet counting, PWM/sequence-logo plotting | [→](#amino_acidspy) |
| `stall_sites.py` | Transcript filtering, codonizing coverage, z-score stall calling, cross-replicate consensus, long-format flattening | [→](#stall_sitespy) |
| `stats_core.py` | **The statistical kernels** — one-row-at-a-time binomial, Wilcoxon, Fisher, and background-aware diff, plus BH-FDR | [→](#stats_corepy) |
| `stats_cli.py` | Cross-script CLI/input helpers: group & timepoint parsing, level detection, replicate-count building | [→](#stats_clipy) |
| `enrichment.py` | Stall-site statistics adapter — Analyses A1–A7 over E/P/A composition + a coverage-density plot | [→](#enrichmentpy) |
| `global_occupancy.py` | Global-occupancy statistics adapter — Analyses A1–A6 + P-site coverage iterators | [→](#global_occupancypy) |

---

### `fasta.py`

**Purpose.** A dependency-light FASTA reader used by `sequence.get_sequence` to
map transcript names to nucleotide sequences. It transparently handles gzipped
files and can be iterated one entry at a time so a whole genome need not sit in
memory.

**Classes**

- **`FastaEntry(header, sequence)`** — one record. `reverse_complement()`
  rewrites `self.sequence` in place (A↔T, C↔G, N→N, case-preserving), raising
  `IOError` on an unrecognized base. `__str__` re-serializes to FASTA with a
  fixed 50-nt line wrap (do not change the chunk size — the class comment marks
  it load-bearing).

- **`FastaFile(file)`** — the reader. If `file` ends in `.gz` it opens with
  `gzip.open("rt")`, otherwise plain `open("rt")`; it also works as a context
  manager (`__enter__`/`__exit__`). The parse is a small **state machine driven
  through `__getitem__`**: iterating the open file, it accumulates sequence
  lines into `current_sequence` and, on hitting the *next* `>` header, packages
  the previous record into a `FastaEntry` and returns it — so header *n* is
  emitted when header *n+1* is seen. The final record is flushed after EOF, and
  a subsequent call raises `IndexError` (which is what makes `for e in
  FastaFile(...)` terminate). Header text is split on whitespace and only the
  first token is kept.

---

### `sequence.py`

**Purpose.** The bridge between `ribopy`/`.ribo` files and the rest of the
pipeline: it resolves transcript sequences and CDS boundaries, and calibrates
ribosome-footprint offsets. Depends on `ribopy` and on `fasta.FastaFile`.

**Constants.** `CODON_NT = 3` (one codon in nt).

**Functions**

- **`get_sequence(ribo_object, reference_file_path, alias) -> dict`** — reads the
  reference FASTA into `{header: sequence}`, then subsets to the ribo object's
  `transcript_names`. When `alias` is truthy, transcript keys are reduced to the
  5th `|`-delimited field (`transcript.split("|")[4]`, the APPRIS-style alias);
  otherwise the full transcript names are used. Returns
  `{transcript_or_alias: sequence}`.

- **`get_cds_range_lookup(ribo_object) -> dict`** — returns `{transcript:
  (cds_start, cds_stop)}` by reading region boundaries from the ribo HDF5 handle
  (`get_region_boundaries`, taking `boundary[1]` = the CDS range for each
  reference) and zipping them against reference names (alias-mapped when the ribo
  object carries an alias). These `(start, stop)` nt offsets are what every
  downstream module slices the CDS with.

- **`apris_human_alias(x)`** — the `x.split("|")[4]` alias helper as a standalone
  callable.

- **`get_offset(ribo_object, exp, mmin, mmax, landmark, search_window=None, return_site="P") -> {read_length: offset}`**
  — **metagene peak-detection offset calibration.** For an experiment `exp` and
  read-length range `[mmin, mmax]`, it pulls the metagene profile
  (`sum_lengths=False`, `sum_references=True`) around a `landmark` — `"start"`
  or `"stop"`. The algorithm:
  1. Normalizes the (possibly `MultiIndex`) metagene frame so columns are
     integer positions relative to the landmark and rows are grouped by read
     length.
  2. Uses a default search window of `(-25, -10)` for `start` or `(-60, -30)`
     for `stop` (overridable).
  3. For each read length, sums the profile, finds the position of the peak
     within the window (`idxmax`), and sets `base_offset = -peak_pos + 1` — the
     distance from the 5′ end to the landmark-defined site.
  4. A `start` landmark natively yields **P-site** offsets and `stop` yields
     **A-site** offsets; if `return_site` differs from that inferred site, every
     offset is shifted by ±`CODON_NT` (A→P subtracts 3, P→A adds 3).

  Read lengths whose windowed profile is empty or all-zero are dropped.

---

### `amino_acids.py`

**Purpose.** The genetic-code lookup layer plus the motif/E-P-A analysis and
sequence-logo plotting that operate on stall calls. (Note: some functions here
still `print()` and draw matplotlib figures — legacy code predating the
no-`print()` rule.)

**Key constants**

| Constant | What it is |
|---|---|
| `CODON2AA` | The standard genetic code, `{codon: one-letter AA}`; stops map to `"*"`. |
| `AA_ORDER` | The 20 amino acids in alphabetical one-letter order (`"ACDEFGHIKLMNPQRSTVWY"`, no stop) — the canonical row/column order everywhere. |
| `SENSE_CODONS` | The 61 non-stop codons, sorted — the codon-level alphabet. Derived from `CODON2AA`. |
| `STOP_CODONS` | `["TAA", "TAG", "TGA"]`, sorted — used by the `--drop-stop-codons` flag. |
| `AA_CLASS` | `{AA: class}` over `acidic / basic / hydrophobic / polar / neutral`. |
| `CLASS_COLORS` | `{class: hex}` palette used to colour sequence logos and bar plots. |

**Functions**

- **`translate_cds_nt_to_aa(cds_nt) -> str`** — upper-cases, maps `U→T`, walks
  the sequence in codon steps and translates via `CODON2AA`, emitting `"X"` for
  any unknown/ambiguous triplet. Trailing partial codons are dropped.

- **`windows_aa(consensus_group, cds_range, sequence, flank_left=10,
  flank_right=10, psite_offset_codons=0) -> list[list[str]]`** — builds fixed-
  width AA windows of length `flank_left + 1 + flank_right` centered on each
  stall codon (offset by `psite_offset_codons` to land on the P-site). Windows
  that run off either CDS end, or that contain a stop (`*`) or ambiguous (`X`)
  residue, are dropped. Handles the `tx`-vs-alias key mismatch by falling back to
  `tx.split("|")[4]`.

- **`count_matrix(win_list, aa_order=AA_ORDER, flank_left=10, flank_right=10) -> DataFrame`**
  — turns a window list into an AA × relative-position count matrix (columns run
  `-flank_left … +flank_right`); asserts every window is the expected width.

- **`background_aa_freq(transcripts, cds_range, sequence, aa_order=AA_ORDER, *, trim_start=0, trim_stop=0) -> (Series, Series)`**
  — the AA-level background. Counts every residue over the *callable* CDS body
  (first `trim_start` and last `trim_stop` codons dropped so the background
  mirrors `call_stalls`' elongation window), then returns a pseudocounted,
  L1-normalized frequency Series (`(count + 1e-6) / (sum + 1e-6·|A|)`) *and* the
  raw count Series. **`background_codon_freq(...)`** is the exact codon-level
  twin over `SENSE_CODONS`.

- **`epa_triplet_counts(consensus_group, cds_range, sequence, *,
  psite_offset_codons=0, basis="P", drop_stop_windows=True, aa_order=AA_ORDER)`**
  — **the E/P/A triplet counter.** For every stall codon it reads the three
  residues at the E, P and A sites — relative offsets `(-1, 0, +1)` under
  `basis="P"` or `(-2, -1, 0)` under `basis="A"` — skipping any triplet that
  runs off the CDS or (optionally) contains `*`/`X`. Returns four objects: the
  full 20³ `Series` indexed by a `MultiIndex(E, P, A)`, and the three per-site
  marginal count Series over `aa_order`.

- **`annotate_stalls_epa(df, cds_range, sequence, *, psite_offset_codons=0,
  basis="P", drop_invalid=True) -> (df_codon, df_aa)`** — the row-wise
  counterpart used by the calling scripts: given a long stall frame (`transcript`
  + `pos_codon`), it annotates each row with `E_codon/P_codon/A_codon` and
  `E_aa/P_aa/A_aa`, caching each transcript's codon list so a CDS is split once.
  Only ambiguous (`X`) triplets are treated as invalid; **stop codons are kept
  here on purpose** — stop removal is owned once by the callers'
  `--drop-stop-codons` flag. Returns two frames sharing the same (post-filter)
  rows, one carrying the codon columns and one the AA columns.

- **`pwm_position_weighted_log2(counts_pos, bg_freq, pseudocount=0.5) -> DataFrame`**
  — turns a position × AA count matrix into the position-weight matrix
  `W = p · log₂(p / bg)`, where `p` is the pseudocounted column-normalized
  probability and `bg` the background frequency (clipped to ≥1e-12). Columns with
  no counts stay zero. This is the per-position information-content quantity a
  sequence logo draws.

- **`plot_logo(weight_mat, title="", aa_class=None, ax=None, ylim=None)`** —
  renders `W` as a `logomaker` sequence logo, colouring glyphs by AA class
  (`CLASS_COLORS`) and shading the E/P/A span (positions −1, 0, +1).

- **`epa_enrichment(counts_epa, bg_aa_freq, pseudocount=0.5) -> Series`** — the
  triplet-level enrichment `W(E,P,A) = p(EPA) · log₂( p(EPA) / (q_E q_P q_A) )`,
  where the null triplet probability is the product of the three per-site
  background frequencies and `p` is the pseudocounted normalized triplet
  frequency.

- **`epa_pairwise_matrix(counts_epa, pair="EP") -> DataFrame`** — collapses the
  third site to a 20×20 heatmap matrix for `"EP"`, `"PA"` or `"EA"`.

- **`plot_top_triplets_multi(epa_enrich_by_group, groups=…, N=25, …)`** — grouped
  bar plot of the top-`N` triplets by mean enrichment across groups; returns
  `(fig, ax, pivot)` and can dump the Triplet × Group table to CSV.

---

### `stall_sites.py`

**Purpose.** The calling layer: it turns per-nucleotide coverage vectors into
codon sums, calls stalls by a global z-score, reconciles them across replicates,
and flattens the result to tidy long tables. Depends only on `numpy`, `pandas`
and `bisect`.

**Functions**

- **`filter_tx(cov_by_exp, reps, min_reps=2, threshold=1.0, trim_start=0, trim_stop=0) -> list`**
  — keeps transcripts present in *all* `reps` where at least `min_reps` of them
  have mean coverage per-nt above `threshold` over the elongation body (CDS with
  the first `trim_start` and last `trim_stop` codons trimmed, in nt). This is the
  expression gate that defines the analysable transcript set.

- **`codonize_counts_cds(x_nt, frame=0) -> np.ndarray`** — sums a CDS-only per-nt
  coverage vector into per-codon counts: it aligns to `frame`, trims the tail to
  a whole number of codons, reshapes to `(codons, 3)` and sums each row.

- **`global_z_log(x, pseudocount=0.5) -> np.ndarray`** — the transcript-wide
  z-score on `log₂(x + pc)`; returns all-zeros when the sd is 0.

- **`call_stalls(x_codon, min_z=1.0, min_obs=2, trim_start=10, trim_stop=10, pseudocount=0.5, trim_edges=None) -> list[dict]`**
  — **the stall caller.** Its key design choice: it **trims the initiation ramp
  and termination region *before* computing the z-score**, so the null (mean,
  sd) reflects only the elongation body and a genuine mid-CDS peak isn't measured
  against an inflated std. A codon is a stall if its body z ≥ `min_z` *and* its
  raw count ≥ `min_obs`. Returned indices are mapped back to the original
  (untrimmed) `x_codon` coordinates; each hit is `{"index", "obs", "z"}`.
  `trim_edges` is a legacy shortcut that sets both `trim_start` and `trim_stop`.

- **`consensus_stalls_across_reps(stalls_by_exp, reps, *, min_support=2, tol=0, min_sep=0, conflict_resolution="keep_both") -> dict`**
  — **cross-replicate consensus with tolerance matching.** Over the transcripts
  common to all `reps`, it forms the union of every replicate's stall indices as
  candidates. For each candidate it counts replicate *support* using a
  **`bisect`-based tolerance match**: `bisect_left(arr, c - tol)` locates the
  first index ≥ `c - tol` and it counts as support if that index is also
  `≤ c + tol`. A candidate is accepted when support ≥ `min_support`; its recorded
  position is the median of the supporting hits when `tol > 0`, else the
  candidate itself. Sites closer than `min_sep` are reconciled by
  `conflict_resolution` — `"keep_both"` (default, ignore `min_sep`),
  `"downstream"`, `"upstream"`, `"merge_median"`, or `"drop_both"`. Returns
  `{tx: sorted unique consensus indices}`.

- **`parse_key(k) -> (full, tx_id, gene)`** — splits a `|`-delimited transcript
  key into its full string, transcript id (field 0) and gene (field 5).

- **`consensus_to_long_df(consensus) -> DataFrame`** — flattens the nested
  `{group: {tx: [positions]}}` consensus into rows of
  `group, transcript, tx_id, gene, pos_codon`, sorted by
  `group, gene, tx_id, pos_codon`.

- **`stalls_to_long_df(stalls_by_exp, rep_to_group=None) -> DataFrame`** —
  flattens the per-replicate `{rep: {tx: [stall_dict]}}` structure into columns
  `group, replicate, gene, tx_id, transcript, pos_codon, obs, z`. It tolerates
  several aliases for the position key (`index/idx/pos/position/codon/codon_idx/
  codon_index`) and raises `TypeError`/`KeyError` if a stall record is malformed.

---

### `stats_core.py`

**Purpose.** **The single source of statistical truth.** Every test in the whole
pipeline is one of these kernels; each computes *one row's* worth of statistics
and returns a plain dict of generic keys, so the two adapters can drop the
values into their own historical column order without duplicating the math.
Depends only on `numpy` and `scipy.stats`.

**FDR helpers**

- **`bh_fdr(p_values) -> np.ndarray`** — Benjamini-Hochberg adjusted p-values via
  `scipy.stats.false_discovery_control(method="bh")`; an empty input returns a
  copy.

- **`apply_bh_fdr(df, group_cols=None) -> DataFrame`** — adds a `p_adj` column.
  With no `group_cols` it corrects the whole `p_value` column at once; with
  `group_cols` it does `groupby(..., sort=False)` and corrects **within each
  family** (e.g. per E/P/A site), then concatenates preserving first-appearance
  order. Because BH is order-independent within a family, the per-row `p_adj`
  depends only on family membership — the caller's final `sort_values` alone sets
  output row order.

**Per-row test kernels**

- **`binom_row(k, n, p_bg) -> dict`** — the within-condition test. Computes the
  observed frequency `k/n`, `log2_enrichment = log2((k/n) / p_bg)`, the
  frequency-weighted `weighted_log2_enrichment`, and a two-sided
  `scipy.stats.binomtest(k, n, p_bg)` p-value. Guards keep the log finite when
  `freq` or `p_bg` is 0.

- **`wilcoxon_row(values_a, values_b) -> dict`** — a two-sided Mann-Whitney U
  (Wilcoxon rank-sum). Medians default to 0.0 on empty input; `log2_FC =
  log2(median_a / median_b)` only when both medians are positive; the test runs
  only when **both arms have ≥ 2 values**, otherwise `(U_stat, p_value) = (nan,
  1.0)`. Used for per-replicate comparisons (the biologically honest test — it
  keeps replicates as independent observations).

- **`fisher_row(count_a, total_a, count_b, total_b) -> dict`** — a two-sided
  Fisher's exact on the 2×2 table `[[count_a, total_a-count_a], [count_b,
  total_b-count_b]]`; returns `odds_ratio, p_value` (`(nan, 1.0)` on
  `ValueError`). This compares *raw shares* between two pools.

- **`background_diff_row(stall_count_headline, stall_total_headline, bg_freq_headline, stall_count_other, stall_total_other, bg_freq_other) -> dict`**
  — **the background-aware between-condition test**, and the most subtle kernel
  here. Instead of comparing raw stall-site shares (Fisher), it compares each
  condition's enrichment *over its own background*. Per condition:

  ```
  expected_count = stall_total · bg_freq        # count if stalling tracked background
  enrichment     = stall_count / expected_count # == the within-condition (A1) enrichment
  ```

  The effect size is the difference of the two log₂ enrichments,
  `delta_log2_enrichment = log2(enrich_headline) − log2(enrich_other)`
  (`enrichment_ratio = 2^delta`). The **test** is the exact conditional test for
  two equal Poisson rates with background as the exposure/offset: conditioning on
  `combined_count = stall_count_headline + stall_count_other`,

  ```
  null_share = expected_count_headline / (expected_count_headline + expected_count_other)
  stall_count_headline | combined_count  ~  Binomial(combined_count, null_share)
  p_value = binomtest(stall_count_headline, combined_count, null_share)
  ```

  Backgrounds are treated as known (they are estimated from millions of codons)
  and passed in already pseudocounted and strictly positive, so `expected_count`
  is never zero. **The convergence property is the whole point:** when the two
  backgrounds are equal, `null_share` reduces to the raw stall-total split and
  the result *converges to `fisher_row`*; the two diverge only when the
  transcriptome composition shifts between conditions — which is itself the
  diagnostic for a compositional confound. Returns the two expected counts, the
  two log₂ enrichments, `delta_log2_enrichment`, `enrichment_ratio`,
  `null_share`, `observed_share`, and `p_value`; a degenerate `combined_count == 0`
  yields `delta = 0`, `p = 1.0`.

---

### `stats_cli.py`

**Purpose.** The genuinely cross-script CLI/input helpers, shared by all four
stats entry points so a fix (e.g. to timepoint pairing) lands everywhere at once.
No `print`, no `sys.exit` — errors raise `ValueError`. Depends on `pandas` and on
`amino_acids` (`AA_ORDER`, `SENSE_CODONS`).

**Functions**

- **`parse_groups(groups_arg) -> {group: [rep, …]}`** — parses
  `"groupA:rep1,rep2;groupB:rep3,rep4"` (blocks split on `;`, name/reps on `:`,
  reps on `,`).
- **`parse_timepoints(timepoints_arg) -> [tp, …]`** — splits `"day_0,day_5,day_10"`
  on commas, **order preserved** (no sorting — critical, since a string sort
  places `day_10` before `day_5`).
- **`timepoint_token(label) -> str`** — `"day_10" → "d10"` short filename tag;
  any other label passes through unchanged.
- **`build_timepoint_pairs(timepoint_order) -> [(time_a, time_b, tag), …]`** —
  from a chronological list, yields every **later-vs-earlier** pair (`time_a` is
  always the later timepoint), in a fixed order (latest first). For
  `[day_0, day_5, day_10]` → `(day_10,day_0,d10_vs_d0)`, `(day_10,day_5,d10_vs_d5)`,
  `(day_5,day_0,d5_vs_d0)`.
- **`build_rep_to_timepoint(rep_to_group) -> {rep: timepoint}`** — derives a
  timepoint from each group name by splitting once on `_` (`"BWM_day_0" →
  "day_0"`); a flat name with no underscore maps to itself.
- **`validate_timepoints(timepoint_order, rep_to_timepoint) -> set`** —
  cross-checks declared timepoints against those present in the data; raises
  `ValueError` on a declared-but-absent timepoint, and returns the set of
  *undeclared* timepoints (present in data, absent from the list) for the caller
  to warn about.
- **`detect_level(df) -> (level, (E_col,P_col,A_col), alphabet, feature_col_name)`**
  — the level-parametric switch: returns `("codon", ("E_codon","P_codon","A_codon"),
  SENSE_CODONS, "codon")` or the amino-acid analogue depending on which columns
  the input CSV carries; raises `ValueError` if neither triple is present. This is
  how one code path serves both codon and AA analyses.
- **`build_replicate_counts(df, site_cols, alphabet) -> {rep: {"E":Series,"P":Series,"A":Series}}`**
  — groups a long stall frame by `replicate` and, per site column,
  `value_counts()` reindexed to the full `alphabet` (missing units filled 0). This
  is the exact `replicate_counts` structure every `enrichment.py` adapter expects.

---

### `enrichment.py`

**Purpose.** The **stall-site statistics adapter** — the E/P/A composition
analyses. Each function pools the `replicate_counts` structure the appropriate
way, calls one `stats_core` kernel per row, and lays out the historical column
order. **Every E/P/A site is tested independently and is its own BH-FDR family**
(the `apply_bh_fdr` `group_cols` reflect this). All functions take a
`feature_col` (`"amino_acid"` or `"codon"`) so one implementation serves both
levels, and most take a `headline_condition` that fixes the direction of the
effect (positive = higher in the headline). `bh_fdr` is re-exported here for
backward compatibility.

The catalog (the docstring's Analyses 1–7):

| Function | Test (kernel) | Comparison | FDR family |
|---|---|---|---|
| `within_condition_enrichment` | binomial (`binom_row`) | pooled group vs its own AA/codon background | per `(group, site)` |
| `between_condition_wilcoxon` | Wilcoxon (`wilcoxon_row`) | per-replicate freq, condition A vs B (n=6 vs 6) | per `site` |
| `between_timepoint_wilcoxon` | Wilcoxon | per-replicate freq, `time_a` vs `time_b`, **pooled across conditions** | per `site` |
| `between_timepoint_fisher_within_condition` | Fisher (`fisher_row`) | pooled `time_a` vs `time_b`, **within each condition** | per `(condition, site)` |
| `per_timepoint_fisher` | Fisher | pooled A vs B **within each timepoint** | per `(timepoint, site)` |
| `between_condition_fisher` | Fisher | pooled A vs B, timepoint-free | per `site` |
| `between_condition_background_diff` | background diff (`background_diff_row`) | enrichment-over-own-background, A vs B | per `site` |
| `between_timepoint_background_diff` | background diff | enrichment-over-own-background, `time_a` vs `time_b`, pooled across conditions | per `site` |

Recurring mechanics worth knowing:

- **Wilcoxon functions** first convert each replicate's per-site counts to
  frequencies (`counts / total`) and then compare *per-replicate* values — the
  test that respects biological replication. `between_timepoint_wilcoxon` pools
  replicates across conditions at each timepoint.
- **Fisher functions** *pool* replicates into one contingency table per bucket
  (a documented pseudoreplication caveat repeated in every Fisher docstring:
  interpret p-values cautiously). `per_timepoint_fisher` additionally re-orders
  the output timepoint blocks to the caller-declared order via a `_tp_order` rank
  (so `day_5` isn't sorted after `day_10`).
- **Background-diff functions** take a `bg_freq_per_cond` / `bg_freq_per_timepoint`
  dict, apply a `1e-6` pseudocount per unit (mirroring
  `within_condition_enrichment`), and delegate to `background_diff_row`. The
  timepoint variant carries the timepoint *labels* in `time_a`/`time_b` columns
  (rather than baking them into column names) so several day-pairs concatenate
  into one CSV with identical columns.

Also here: **`plot_coverage_density(cov, groups, out_dir, trim_start=0,
trim_stop=0)`** — a diagnostic that KDE-plots the per-transcript mean coverage
(reads/nt, on the elongation body) for every replicate onto one figure and writes
`coverage_density.png`.

---

### `global_occupancy.py`

**Purpose.** The **global-occupancy statistics adapter** plus the coverage
iterators that build occupancy from `.ribo` P-site coverage. It runs the same
tests as `enrichment.py` but over transcriptome-wide read sums instead of
stall-site composition, so its A1–A6 catalog omits the background-diff analyses
(A4/A7 are N/A here). `bh_fdr` and `parse_groups` are re-exported.

**Statistical functions** (all `feature_col`-parametric, kernels from
`stats_core`):

| Function | Test | Comparison | FDR family |
|---|---|---|---|
| `within_condition_binomial_occupancy` | binomial | group's read share vs **transcriptome** share | per `group` |
| `between_condition_wilcoxon_occupancy` | Wilcoxon | per-replicate normalized rate, A vs B | global |
| `between_timepoint_wilcoxon_occupancy` | Wilcoxon | per-replicate rate, `time_a` vs `time_b`, pooled across conditions | global |
| `between_timepoint_fisher_within_condition` | Fisher | pooled `time_a` vs `time_b`, within each condition | per `condition` |
| `between_condition_fisher_occupancy` | Fisher | pooled A vs B, timepoint-free | global |
| `per_timepoint_fisher_occupancy` | Fisher | pooled A vs B within each timepoint | per `timepoint` |

Notes specific to occupancy: the binomial background is the **transcriptome
frequency** (`transcriptome_counts[unit] / total`), not a stall-site background;
raw read sums are float-valued but integer-equal, so counts/totals go through
`int(round(...))` before the exact tests; and the two Wilcoxon variants FDR-
correct globally (`apply_bh_fdr(df)` with no group cols), since occupancy has one
comparison per unit rather than per E/P/A site.

**Coverage iterators & aggregation** (the machinery `global_codon_occ.py` uses to
turn coverage into `{unit: read_sum}`):

- **`iter_trimmed_codons(seq, trim_start_codons, trim_stop_codons)`** — yields
  `(codon_str, cds_nt_idx)` over the trimmed CDS, skipping any codon containing
  `N`.
- **`iter_trimmed_site_counts(cds_seq, cov, trim_start_codons, trim_stop_codons,
  site_shift)`** — the **E/P/A occupancy iterator**. Coverage is P-site-offset:
  the value at `cov[X:X+3]` is the read count for ribosomes whose P-site sits on
  the codon at CDS nt `X`. For each P-site position it reads that same count and
  reports whichever codon the `site_shift` selects — `-3` = E-site (one codon
  upstream), `0` = P-site, `+3` = A-site — so the caller runs it three times to
  accumulate E/P/A totals. It bounds-checks the P-site window and the shifted
  codon, skips `N`-containing codons, and (the one sanctioned warning in this
  package) `logging.warning`s once if the CDS and coverage lengths disagree.
- **`aggregate_to_aa(codon_dict) -> dict`** — sums a `{codon: value}` map up to
  `{AA: value}` via `CODON2AA`, dropping stop codons.

---

## See also

- [`../README.md`](../README.md) — repository root: pipeline overview and the
  full three-stage process → analyze → visualize flow.
- [`../scripts/README.md`](../scripts/README.md) — the top-level CLI scripts that
  import this package (the only place with `print()`, argument parsing, and file
  I/O).
