# adj_coverage — Step 1: extract CDS-aligned coverage

*Pipeline entry point: turn a C. elegans `.ribo` file into a gzipped per-transcript P-site coverage pickle that every downstream stage consumes.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [c_elegans](../README.md) › adj_coverage

This is **Step 1** of the pipeline. It runs `scripts/adj_coverage.py`, which reads the raw Ribo-seq HDF5 (`.ribo`) file, computes a read-length-specific P-site (or A-site) offset from the coverage metagene, applies that offset to align each read to a codon, and writes the resulting CDS-aligned coverage as a gzipped pickle (`<basename>_coverage.pkl.gz`) next to the `.ribo`. Every Step 2 / Step 3 stage — the three stall-site trees, global occupancy, and the internal-stop diagnostic — begins by loading this pickle, so nothing else can run until this does.

There is no stats runner or plot launcher here: coverage extraction is a single one-shot step.

## Contents

| File | Kind | Summary |
|---|---|---|
| `run_adj_coverage_all.sh` | `run_*` (drives Python) | Runs `adj_coverage.py` on the single named C. elegans `.ribo` file and writes `<basename>_coverage.pkl.gz`. |

---

### `run_adj_coverage_all.sh`

**What it does.** Activates the conda env, self-locates to the repo root, checks that the configured `.ribo` file exists, derives the output pickle name from it, then builds and `eval`s a `python3 scripts/adj_coverage.py …` command. Unlike the array-based launchers in later stages, this one assembles the invocation as a **string** (`CMD="python3 …"`), conditionally appends flags, echoes it, and runs it with `eval $CMD`.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `RIBO_DIR="./all_ribo_file"` | Directory holding the `.ribo` files. |
| `RIBO_FILE="$RIBO_DIR/C_elegan_all_02_04_2026.ribo"` | The **exact** input `.ribo`. Named explicitly (not globbed) because `all_ribo_file/` holds multiple organisms — a `*.ribo` glob would also match the mouse file. |
| `MIN_LEN=26`, `MAX_LEN=38` | Read-length window (in nt) to keep when building the offset metagene and coverage. |
| `RETURN_SITE="P"` | Which ribosomal site to report coverage for — `P`-site (default) or `A`-site. |
| `SITE_TYPE="start"` | The landmark the offset is measured against. For `start`, the 5′-end offset peak is searched in the window below. |
| `SEARCH_WINDOW="-25 -10"` | The nt window (relative to the landmark) in which the offset peak is located. Passed as two space-separated integers. |
| `USE_ALIAS="no"` | Set to `yes` if the `.ribo` uses the human/mouse `appris_human_alias` naming; adds `--alias`. |
| `PROCS=64` | Parallel workers — experiments inside the one `.ribo` are processed in parallel. |

**Command built.** Invokes `scripts/adj_coverage.py`. The base string is:

```bash
CMD="python3 scripts/adj_coverage.py \
  --ribo $RIBO --min-len $MIN_LEN --max-len $MAX_LEN \
  --return-site $RETURN_SITE --out $OUT --procs $PROCS"
[ "$USE_ALIAS" = "yes" ] && CMD="$CMD --alias"
CMD="$CMD --site-type $SITE_TYPE"
CMD="$CMD --search-window $SEARCH_WINDOW"
eval $CMD
```

`$OUT` is computed as `${RIBO_DIR}/${BASENAME}_coverage.pkl.gz`, where `BASENAME` is the `.ribo` filename without its extension. `--alias` is appended only when `USE_ALIAS=yes`; `--site-type` and `--search-window` are always appended. `eval` is required so the two-token `$SEARCH_WINDOW` ("-25 -10") expands into two separate arguments.

**Inputs / Outputs.**

- **Input:** the `.ribo` at `$RIBO_FILE` (verified to exist; the script exits with an error if not).
- **Output:** the gzipped coverage pickle `all_ribo_file/C_elegan_all_02_04_2026_coverage.pkl.gz` — the file every downstream stage's `run_*.sh` picks up (usually via `ls "$RIBO_DIR"/*_coverage.pkl.gz | head -1`, or, in the internal-stop scan, derived from the `.ribo` name).

**Related.** Underlying Python: `scripts/adj_coverage.py` — see [scripts/README.md](../../../scripts/README.md).

## See also

- [c_elegans stage index](../README.md) — the full pipeline order; every later stage depends on the pickle this one writes.
- [scripts/README.md](../../../scripts/README.md) — `adj_coverage.py` documentation.
- [shell_scripts/README.md](../../README.md) — the shell orchestration layer.
- [ribostall root README](../../../README.md) — project overview.
