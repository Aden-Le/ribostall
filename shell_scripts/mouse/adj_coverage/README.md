# adj_coverage — Step 1: extract CDS-aligned coverage (mouse)

*Wraps `scripts/adj_coverage.py` to turn `mouse_all.ribo` into a gzipped coverage pickle that every downstream mouse stage consumes.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [mouse](../README.md) › adj_coverage

---

## What this stage is

This is the first step of the pipeline. `adj_coverage.py` reads the HDF5 `.ribo` file, selects reads in a length band, computes the P-site (or A-site) offset from a start-codon metagene, aligns coverage to the CDS, and writes a per-experiment coverage dict as a gzipped pickle. Every later mouse stage (stall calling, occupancy, the internal-stop diagnostic) reads that pickle, so this must run first.

**Output:** `all_ribo_file/mouse_all_coverage.pkl.gz` — written next to the input `.ribo`, named `<basename>_coverage.pkl.gz`. There is no `results/mouse/` tree at this step; the pickle lives beside the ribo file so downstream scripts can derive its path from `RIBO_FILE`.

## Contents

| File | Role | Summary |
|---|---|---|
| `run_adj_coverage_all.sh` | `run_*` | Builds and `eval`s the single `adj_coverage.py` call for `mouse_all.ribo`. |

---

### `run_adj_coverage_all.sh`

**What it does.** A self-contained launcher for one `.ribo` file. It uses the standard ribostall Bash pattern:

- **Shebang + self-location.** `#!/bin/bash`; it computes its own directory with `SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` and then `cd "$SCRIPT_DIR/../../.."` to reach the repo root (three levels up from `shell_scripts/mouse/adj_coverage/`). Every path in the command is therefore repo-relative and the script runs correctly no matter where it is invoked from.
- **Conda activation.** `source ${HOME}/miniconda3/etc/profile.d/conda.sh` then `conda activate ribostall_env`.
- **Command build via string + `eval`.** Unlike the later stages (which use Bash arrays), this script assembles the command as a string in `CMD=...`, conditionally appends optional flags, `echo`s it, and runs `eval $CMD`.

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `RIBO_DIR` | `./all_ribo_file` | Directory holding the `.ribo`. |
| `RIBO_FILE` | `$RIBO_DIR/mouse_all.ribo` | The exact single input (not a glob). The mouse reference is *not* used here — it is consumed by downstream sequence steps. |
| `MIN_LEN` / `MAX_LEN` | `27` / `34` | Read-length band. |
| `RETURN_SITE` | `P` | Return P-site coverage. |
| `SITE_TYPE` | `start` | Metagene anchor for offset estimation. |
| `SEARCH_WINDOW` | `-25 -10` | Offset search window (two integers). |
| `USE_ALIAS` | `no` | If `yes`, adds `--alias` (human/mouse `appris_human_alias`); this run leaves it off. |
| `PROCS` | `64` | Parallel workers (experiments processed in parallel within the one `.ribo`). |

**Command built.** It derives the output path `OUT="${RIBO_DIR}/${BASENAME}_coverage.pkl.gz"` (with `BASENAME=$(basename "$RIBO" .ribo)`), guards that the input `.ribo` exists, then builds:

```bash
CMD="python3 scripts/adj_coverage.py --ribo $RIBO --min-len 27 --max-len 34 \
  --return-site P --out $OUT --procs 64"
# USE_ALIAS=no → --alias NOT appended
CMD="$CMD --site-type start"
CMD="$CMD --search-window -25 -10"
eval $CMD
```

Because `USE_ALIAS="no"`, the `--alias` flag is omitted; `--site-type` and `--search-window` are always appended.

**Inputs / Outputs.**
- **In:** `all_ribo_file/mouse_all.ribo`.
- **Out:** `all_ribo_file/mouse_all_coverage.pkl.gz`.

**Related.** This pickle is the required input to every mouse stage below. Downstream scripts re-derive its path as `"${RIBO_DIR}/$(basename "$RIBO_FILE" .ribo)_coverage.pkl.gz"` rather than globbing, so a co-present C. elegans pickle is never picked up by accident.

---

## See also

- [Python entry points](../../../scripts/README.md) — `adj_coverage.py` details.
- [mouse organism README](../README.md) — the flat-design overview and pipeline order.
- [C. elegans adj_coverage](../../c_elegans/adj_coverage/README.md) — the timepoint-organism counterpart.
- [shell_scripts top level](../../README.md) · [Repository root](../../../README.md).
