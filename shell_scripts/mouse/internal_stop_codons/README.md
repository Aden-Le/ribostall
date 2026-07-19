# internal_stop_codons — diagnostic: in-frame internal stop scan (mouse)

*Wraps `scripts/internal_stop_codons.py` to scan every CDS for in-frame internal stop codons left in the coding body and pull the per-replicate P-site read count at each one.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [mouse](../README.md) › internal_stop_codons

---

## What this stage is

A **read-only diagnostic**, not a pipeline step that feeds anything downstream. It scans each CDS for in-frame internal stop codons (`TAA`/`TAG`/`TGA`) that remain inside the coding body after the terminal stop is trimmed off, and reports the per-replicate P-site read count at each. It can run any time after Step 1 (it only needs the coverage pickle, the ribo file, and the reference). It writes two CSVs and touches nothing else.

**Output:** two CSVs under `results/mouse/internal_stop_codons/`.

The scan window matters: the terminal stop is the last CDS codon, so `TRIM_STOP=1` would remove it and leave only genuine internal stops; `TRIM_STOP=0` keeps the terminal stop (then every transcript is reported). This run uses `TRIM_START=20`, `TRIM_STOP=10`.

## Contents

| File | Role | Summary |
|---|---|---|
| `run_internal_stop_codons.sh` | `run_*` | Builds and runs the single `internal_stop_codons.py` call for the mouse dataset. |

---

### `run_internal_stop_codons.sh`

**What it does.** Standard ribostall launcher pattern: self-locates to repo root (`cd "$SCRIPT_DIR/../../.."`), activates `ribostall_env`, derives the coverage pickle from `RIBO_FILE` (not a glob), guards that it exists, echoes a summary banner, then runs `internal_stop_codons.py` via a `CMD=(...)` Bash array.

**CONFIG.**

| Variable | Value | Meaning |
|---|---|---|
| `RIBO_DIR` / `RIBO_FILE` | `./all_ribo_file` / `…/mouse_all.ribo` | Ribo file (pickle derived from it). |
| `EXP_GROUPS` | `control:AA_3,AA_4;treatment:Ch_WAA2` | Flat 2-vs-1; filters the coverage dict to the declared read-count columns. |
| `TRIM_START` | `20` | Codons trimmed from the CDS start before scanning. |
| `TRIM_STOP` | `10` | Codons trimmed from the CDS end (removes the terminal stop and its neighborhood). |
| `REFERENCE_FILE` | `./reference/appris_mouse_v2_selected.fa.gz` | CDS sequences to scan. |
| `OUT_DIR` | `./results/mouse/internal_stop_codons` | Where the two CSVs land. |

The pickle path is derived as `PICKLE="${RIBO_DIR}/$(basename "$RIBO_FILE" .ribo)_coverage.pkl.gz"` → `all_ribo_file/mouse_all_coverage.pkl.gz`.

**Command built.**

```bash
python3 scripts/internal_stop_codons.py \
  --ribo all_ribo_file/mouse_all.ribo \
  --pickle all_ribo_file/mouse_all_coverage.pkl.gz \
  --reference reference/appris_mouse_v2_selected.fa.gz \
  --groups 'control:AA_3,AA_4;treatment:Ch_WAA2' \
  --trim-start 20 --trim-stop 10 \
  --out-dir results/mouse/internal_stop_codons
```

**Inputs / Outputs.**
- **In:** `mouse_all.ribo`, `mouse_all_coverage.pkl.gz`, `appris_mouse_v2_selected.fa.gz`.
- **Out:** two CSVs in `results/mouse/internal_stop_codons/`.

**Related.** Depends only on Step 1's coverage pickle; independent of the stall-site and occupancy trees. There is no stats step or plot launcher — the CSVs are the deliverable.

---

## See also

- [Python entry points](../../../scripts/README.md) — `internal_stop_codons.py`.
- [mouse organism README](../README.md) — flat-design overview.
- [adj_coverage stage](../adj_coverage/README.md) — produces the coverage pickle this scan consumes.
- [shell_scripts top level](../../README.md) · [Repository root](../../../README.md).
