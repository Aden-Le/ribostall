"""Tests for scripts/cross_tp_summary_checker.py.

Coverage:
  * Pure helpers (_tp_sort_key, _tp_token, _rare_label_with_tps,
    _concordance_flag, _flip_flag, _fmt_signed, _fmt_p).
  * _aggregate_cells: builds the per-(site, feature) cell aggregates,
    including rare_tps (chronological list of timepoints tripping the
    rare-k threshold), n_sig, same_sign, sig_flip, and incomplete-cell
    flagging.
  * End-to-end: rendered `rare-aa (d0, d10)` suffix appears in the
    Concordance and Direction-flip tables when expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import cross_tp_summary_checker as cts


# -----------------------------------------------------------------------------
# Pure helpers
# -----------------------------------------------------------------------------

class TestTpSortKey:
    @pytest.mark.parametrize("tp,expected", [
        ("day_0", 0.0), ("day_5", 5.0), ("day_10", 10.0),
        ("day_-2", -2.0), ("day_3.5", 3.5),
    ])
    def test_numeric(self, tp, expected):
        assert cts._tp_sort_key(tp) == expected

    def test_no_digits_pushes_to_end(self):
        assert cts._tp_sort_key("baseline") == float("inf")


class TestTpToken:
    @pytest.mark.parametrize("tp,token", [
        ("day_0", "d0"), ("day_5", "d5"), ("day_10", "d10"),
    ])
    def test_known(self, tp, token):
        assert cts._tp_token(tp) == token

    def test_unknown_falls_back_to_str(self):
        assert cts._tp_token("day_42") == "day_42"


class TestRareLabelWithTps:
    def test_single_tp(self):
        assert cts._rare_label_with_tps("rare-aa", ["day_10"]) == "rare-aa (d10)"

    def test_multi_tp_preserves_order(self):
        # Caller is responsible for chronological order; helper just joins.
        assert cts._rare_label_with_tps(
            "rare-codon", ["day_0", "day_5", "day_10"]
        ) == "rare-codon (d0, d5, d10)"


class TestConcordanceFlag:
    def test_multi_tp_coherent(self):
        assert cts._concordance_flag(
            n_sig=2, rare_tps=[], any_iid_amp=False, rare_label="rare-aa"
        ) == "multi-tp-coherent"

    def test_single_tp_driven(self):
        assert cts._concordance_flag(
            n_sig=1, rare_tps=[], any_iid_amp=False, rare_label="rare-aa"
        ) == "single-tp-driven"

    def test_no_significance_label_when_nsig_zero(self):
        # n_sig=0 alone -> empty string (no significance, no caveats).
        assert cts._concordance_flag(
            n_sig=0, rare_tps=[], any_iid_amp=False, rare_label="rare-aa"
        ) == ""

    def test_all_flags_compose_semicolon(self):
        assert cts._concordance_flag(
            n_sig=3, rare_tps=["day_0", "day_10"],
            any_iid_amp=True, rare_label="rare-codon",
        ) == "multi-tp-coherent; iid-amp; rare-codon (d0, d10)"

    def test_rare_with_no_sig_no_iid(self):
        assert cts._concordance_flag(
            n_sig=0, rare_tps=["day_5"], any_iid_amp=False, rare_label="rare-aa",
        ) == "rare-aa (d5)"


class TestFlipFlag:
    def test_no_caveats_returns_empty(self):
        assert cts._flip_flag(rare_tps=[], any_iid_amp=False, rare_label="rare-aa") == ""

    def test_iid_only(self):
        assert cts._flip_flag(rare_tps=[], any_iid_amp=True, rare_label="rare-aa") == "iid-amp"

    def test_rare_only(self):
        assert cts._flip_flag(
            rare_tps=["day_0", "day_10"], any_iid_amp=False, rare_label="rare-codon"
        ) == "rare-codon (d0, d10)"

    def test_both(self):
        assert cts._flip_flag(
            rare_tps=["day_5"], any_iid_amp=True, rare_label="rare-aa",
        ) == "iid-amp; rare-aa (d5)"


class TestFormatHelpers:
    def test_fmt_signed_positive(self):
        assert cts._fmt_signed(0.123) == "+0.12"

    def test_fmt_signed_negative_precision(self):
        assert cts._fmt_signed(-1.234567, prec=3) == "-1.235"

    def test_fmt_signed_nan(self):
        assert cts._fmt_signed(float("nan")) == "nan"

    def test_fmt_p_scientific(self):
        assert cts._fmt_p(1.5e-7) == "1.50e-07"

    def test_fmt_p_nan(self):
        assert cts._fmt_p(float("nan")) == "nan"


# -----------------------------------------------------------------------------
# _aggregate_cells
# -----------------------------------------------------------------------------

def _fisher_row(site: str, tp: str, feat_val: str, *,
                or_: float, p_adj: float,
                bwm_k: int, ctrl_k: int) -> dict:
    return {
        "site": site, "timepoint": tp, "amino_acid": feat_val,
        "odds_ratio": or_, "p_value": p_adj * 0.5, "p_adj": p_adj,
        "BWM_count": bwm_k, "control_count": ctrl_k,
    }


class TestAggregateCells:

    @staticmethod
    def _csv_with_one_concordant_one_flip_one_rare() -> pd.DataFrame:
        """Three (site, AA) cells across day_0/5/10:
          - A:K  concordant enriched, no caveats, all sig
          - A:N  flip cell (+, +, -), 0 sig
          - P:W  concordant enriched, rare at d0 + d10
        """
        rows = []
        # A:K concordant, all sig p_adj<0.05
        for tp, or_, p in [("day_0", 2.0, 1e-8), ("day_5", 1.8, 1e-7), ("day_10", 1.6, 1e-6)]:
            rows.append(_fisher_row("A", tp, "K", or_=or_, p_adj=p,
                                    bwm_k=300, ctrl_k=300))
        # A:N flip (sign +/+/-), no sig
        for tp, or_ in [("day_0", 1.5), ("day_5", 1.2), ("day_10", 0.6)]:
            rows.append(_fisher_row("A", tp, "N", or_=or_, p_adj=0.5,
                                    bwm_k=200, ctrl_k=200))
        # P:W concordant enriched, rare at d0 (BWM<100) and d10 (ctrl<100)
        rows.append(_fisher_row("P", "day_0", "W", or_=1.2, p_adj=0.3,
                                bwm_k=40, ctrl_k=200))   # rare via BWM
        rows.append(_fisher_row("P", "day_5", "W", or_=1.1, p_adj=0.4,
                                bwm_k=120, ctrl_k=150))  # not rare
        rows.append(_fisher_row("P", "day_10", "W", or_=1.3, p_adj=0.3,
                                bwm_k=110, ctrl_k=60))   # rare via control
        return pd.DataFrame(rows)

    def test_rare_tps_list_correct(self):
        df = self._csv_with_one_concordant_one_flip_one_rare()
        cells, tps = cts._aggregate_cells(
            df, feat="amino_acid", rare_k=100,
            iid_amp_thresh=1e-10, sig_thresh=0.05,
        )
        assert tps == ["day_0", "day_5", "day_10"]
        cell = cells.set_index("cell")
        assert cell.loc["A:K", "rare_tps"] == []
        assert cell.loc["A:N", "rare_tps"] == []
        assert cell.loc["P:W", "rare_tps"] == ["day_0", "day_10"]

    def test_same_sign_and_nsig(self):
        df = self._csv_with_one_concordant_one_flip_one_rare()
        cells, _ = cts._aggregate_cells(
            df, feat="amino_acid", rare_k=100,
            iid_amp_thresh=1e-10, sig_thresh=0.05,
        )
        cell = cells.set_index("cell")
        # A:K: signs all +, n_sig=3 (use bool() because pandas yields numpy.bool_).
        assert bool(cell.loc["A:K", "same_sign"])
        assert cell.loc["A:K", "n_sig"] == 3
        # A:N: log2 of OR 1.5, 1.2, 0.6 -> log2(0.6) < 0; not same sign
        assert not bool(cell.loc["A:N", "same_sign"])
        assert cell.loc["A:N", "n_sig"] == 0
        # P:W: all ORs > 1 -> same sign
        assert bool(cell.loc["P:W", "same_sign"])
        assert cell.loc["P:W", "n_sig"] == 0

    def test_sig_flip_detection(self):
        # Build a cell with sig+ at d0 and sig- at d10.
        rows = [
            _fisher_row("A", "day_0", "X", or_=2.0, p_adj=1e-5,
                        bwm_k=300, ctrl_k=300),    # sig +
            _fisher_row("A", "day_5", "X", or_=1.05, p_adj=0.5,
                        bwm_k=300, ctrl_k=300),    # ns
            _fisher_row("A", "day_10", "X", or_=0.4, p_adj=1e-4,
                        bwm_k=300, ctrl_k=300),    # sig -
        ]
        cells, _ = cts._aggregate_cells(
            pd.DataFrame(rows), feat="amino_acid", rare_k=100,
            iid_amp_thresh=1e-10, sig_thresh=0.05,
        )
        row = cells.set_index("cell").loc["A:X"]
        assert bool(row["sig_flip"])
        assert row["sig_pos_tps"] == ["day_0"]
        assert row["sig_neg_tps"] == ["day_10"]

    def test_or_zero_dropped(self):
        # Cell A:Z has OR=0 at d0 -> log2 = NaN -> row dropped, cell incomplete.
        rows = [
            _fisher_row("A", "day_0", "Z", or_=0.0, p_adj=0.5,
                        bwm_k=200, ctrl_k=200),
            _fisher_row("A", "day_5", "Z", or_=1.5, p_adj=0.5,
                        bwm_k=200, ctrl_k=200),
            _fisher_row("A", "day_10", "Z", or_=1.4, p_adj=0.5,
                        bwm_k=200, ctrl_k=200),
        ]
        cells, tps = cts._aggregate_cells(
            pd.DataFrame(rows), feat="amino_acid", rare_k=100,
            iid_amp_thresh=1e-10, sig_thresh=0.05,
        )
        cell = cells.set_index("cell").loc["A:Z"]
        # 2 valid TPs out of 3 -> incomplete.
        assert cell["n_valid_tp"] == 2
        # rare_tps zips against `timepoints` order; the dropped d0 has no row
        # in the per_tp arrays, so it's NaN and skipped by the (not NaN) guard.
        assert cell["rare_tps"] == []

    def test_iid_amp_threshold(self):
        rows = [
            _fisher_row("A", "day_0", "Y", or_=2.0, p_adj=1e-12,
                        bwm_k=300, ctrl_k=300),
            _fisher_row("A", "day_5", "Y", or_=1.5, p_adj=1e-3,
                        bwm_k=300, ctrl_k=300),
            _fisher_row("A", "day_10", "Y", or_=1.3, p_adj=1e-3,
                        bwm_k=300, ctrl_k=300),
        ]
        cells, _ = cts._aggregate_cells(
            pd.DataFrame(rows), feat="amino_acid", rare_k=100,
            iid_amp_thresh=1e-10, sig_thresh=0.05,
        )
        cell = cells.set_index("cell").loc["A:Y"]
        assert bool(cell["any_iid_amp"])

    def test_rare_threshold_strict_less_than(self):
        # k exactly at threshold (100) is NOT rare.
        rows = [
            _fisher_row("A", "day_0", "Q", or_=1.5, p_adj=0.5,
                        bwm_k=100, ctrl_k=100),
            _fisher_row("A", "day_5", "Q", or_=1.5, p_adj=0.5,
                        bwm_k=100, ctrl_k=100),
            _fisher_row("A", "day_10", "Q", or_=1.5, p_adj=0.5,
                        bwm_k=100, ctrl_k=100),
        ]
        cells, _ = cts._aggregate_cells(
            pd.DataFrame(rows), feat="amino_acid", rare_k=100,
            iid_amp_thresh=1e-10, sig_thresh=0.05,
        )
        assert cells.set_index("cell").loc["A:Q", "rare_tps"] == []


# -----------------------------------------------------------------------------
# End-to-end: rare-aa suffix appears in stdout
# -----------------------------------------------------------------------------

class TestPrintersEndToEnd:

    @staticmethod
    def _build_csv() -> pd.DataFrame:
        # A:K concordant enriched (sig at d0 and d10 -> multi-tp-coherent)
        # P:W concordant enriched, rare at d0 + d10 -> rare-aa (d0, d10)
        rows = []
        for tp, or_, p in [("day_0", 2.0, 1e-8), ("day_5", 1.8, 0.5), ("day_10", 1.6, 1e-6)]:
            rows.append(_fisher_row("A", tp, "K", or_=or_, p_adj=p,
                                    bwm_k=300, ctrl_k=300))
        rows.append(_fisher_row("P", "day_0", "W", or_=1.2, p_adj=0.3,
                                bwm_k=40, ctrl_k=200))
        rows.append(_fisher_row("P", "day_5", "W", or_=1.1, p_adj=0.4,
                                bwm_k=120, ctrl_k=150))
        rows.append(_fisher_row("P", "day_10", "W", or_=1.3, p_adj=0.3,
                                bwm_k=110, ctrl_k=60))
        return pd.DataFrame(rows)

    def test_concordance_table_renders_rare_suffix_and_significance(self, capsys):
        df = self._build_csv()
        cells, tps = cts._aggregate_cells(
            df, feat="amino_acid", rare_k=100,
            iid_amp_thresh=1e-10, sig_thresh=0.05,
        )
        cts._print_concordance(cells, tps, rare_label="rare-aa")
        out = capsys.readouterr().out
        # rare suffix renders with the chronological d0/d10 list.
        assert "rare-aa (d0, d10)" in out
        # A:K has n_sig=2 -> multi-tp-coherent.
        assert "multi-tp-coherent" in out
        # Cell labels appear.
        assert "A:K" in out and "P:W" in out

    def test_flip_table_renders_when_present(self, capsys):
        # Build a cell that flips sign and trips rare at one TP.
        rows = [
            _fisher_row("A", "day_0", "K", or_=2.0, p_adj=0.5,
                        bwm_k=50, ctrl_k=300),   # rare via BWM<100, +
            _fisher_row("A", "day_5", "K", or_=1.5, p_adj=0.5,
                        bwm_k=200, ctrl_k=200),  # +
            _fisher_row("A", "day_10", "K", or_=0.6, p_adj=0.5,
                        bwm_k=200, ctrl_k=200),  # -
        ]
        cells, tps = cts._aggregate_cells(
            pd.DataFrame(rows), feat="amino_acid", rare_k=100,
            iid_amp_thresh=1e-10, sig_thresh=0.05,
        )
        cts._print_flip(cells, tps, rare_label="rare-aa",
                        top_flip=15, sig_thresh=0.05)
        out = capsys.readouterr().out
        assert "Direction-flip cells across timepoints" in out
        assert "A:K" in out
        assert "rare-aa (d0)" in out
