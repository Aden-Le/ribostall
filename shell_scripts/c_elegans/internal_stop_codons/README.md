# internal_stop_codons — CDS in-frame stop diagnostic

*Read-only QC scan: find in-frame internal stop codons (TAA / TAG / TGA) left inside the coding body after the terminal stop is trimmed, and report the per-replicate P-site read count at each one.*

> **[ribostall](../../../README.md)** › [shell_scripts](../../README.md) › [c_elegans](../README.md) › internal_stop_codons

This stage sits off the main analysis line — it is a **diagnostic**, not a stall-calling or occupancy step. It scans every CDS for in-frame stop codons that remain inside the coding body once the *terminal* stop is trimmed off. A genuine internal stop is biologically notable (it can flag annotation problems, read-through, or non-canonical decoding), and the scan pulls the per-replicate P-site read count at each one so you can see whether ribosomes are actually parked there. It writes two CSVs and touches nothing else in the pipeline.

## Contents

| File | Kind | Summary |
|---|---|---|
| `run_internal_stop_codons.sh` | `run_*` (drives Python) | Runs `internal_stop_codons.py`: scans CDS bodies for in-frame TAA/TAG/TGA and reports per-replicate P-site counts; writes two CSVs. |

There is no stats runner or plot launcher — the scan is a single diagnostic step.

---

### `run_internal_stop_codons.sh`

**What it does.** Activates `ribostall_env`, self-locates to the repo root (`cd "$SCRIPT_DIR/../../.."`), then **derives the coverage pickle name from the `.ribo` filename** — `${RIBO_DIR}/$(basename "$RIBO_FILE" .ribo)_coverage.pkl.gz` — rather than globbing, because `all_ribo_file/` holds multiple organisms and a glob could grab the wrong pickle. It checks the pickle exists (errors out if not), echoes a banner, then builds a `CMD=(python3 …)` array and runs it as `"${CMD[@]}"`.

**CONFIG.**

| Variable | Meaning |
|---|---|
| `RIBO_DIR="./all_ribo_file"` | Directory holding the `.ribo` and its `_coverage.pkl.gz`. |
| `RIBO_FILE="$RIBO_DIR/C_elegan_all_02_04_2026.ribo"` | The exact C. elegans `.ribo`; the pickle path is derived from it. |
| `EXP_GROUPS` | The six `(condition, timepoint)` cells, `rep2`/`rep3` each — filters the coverage dict to declared replicates (the read-count columns). |
| `TRIM_START=20` | Codons trimmed from the CDS start before scanning. |
| `TRIM_STOP=10` | Codons trimmed from the CDS end. The terminal stop is the last CDS codon, so `TRIM_STOP=1` would already remove it and any stop still found is a true internal stop; `TRIM_STOP=0` keeps the terminal stop (then every transcript is reported). |
| `REFERENCE_FILE="./reference/appris_celegans_v1_selected_new.fa"` | APPRIS C. elegans FASTA — provides the CDS sequence scanned for stops. |
| `OUT_DIR="./results/c_elegans/internal_stop_codons"` | Where the two CSVs are written. |

**Command built.**

```bash
CMD=(python3 scripts/internal_stop_codons.py \
  --ribo "$RIBO_FILE" \
  --pickle "$PICKLE" \
  --reference "$REFERENCE_FILE" \
  --groups "$EXP_GROUPS" \
  --trim-start "$TRIM_START" \
  --trim-stop "$TRIM_STOP" \
  --out-dir "$OUT_DIR")
"${CMD[@]}"
```

where `$PICKLE` is the derived `C_elegan_all_02_04_2026_coverage.pkl.gz`.

**Inputs / Outputs.**

- **Input:** the `.ribo`, its derived coverage pickle, and the reference FASTA.
- **Output:** two CSVs written to `$OUT_DIR` (`results/c_elegans/internal_stop_codons/`) — the internal-stop hits and their per-replicate P-site read counts.

**Related.** Underlying Python: `scripts/internal_stop_codons.py` — see [scripts/README.md](../../../scripts/README.md).

## See also

- [c_elegans stage index](../README.md) — where this diagnostic sits relative to the main pipeline.
- [adj_coverage](../adj_coverage/README.md) — Step 1, which writes the coverage pickle this scan reads.
- [scripts/README.md](../../../scripts/README.md) — `internal_stop_codons.py` documentation.
- [shell_scripts/README.md](../../README.md) · [ribostall root README](../../../README.md).
