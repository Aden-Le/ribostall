"""Tests for scripts/dylan_table_checker.py.

Strategy: build small handcrafted DataFrames where the expected pick list,
sort order, and filter outcome are computable by hand. Family-level
`select_*` functions print directly (no return value), so end-to-end tests
use pytest's `capsys` fixture and assert on substrings of stdout.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

import dylan_table_checker as dtc


# -----------------------------------------------------------------------------
# Helpers (private to the tests)
# -----------------------------------------------------------------------------

def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _lines(captured: str) -> list[str]:
    return [ln.rstrip() for ln in captured.splitlines()]


def _section_index(lines: list[str], header: str) -> int:
    """Return index of `### {header}` line; -1 if absent."""
    target = f"### {header}"
    for i, ln in enumerate(lines):
        if ln.strip() == target:
            return i
    return -1


# -----------------------------------------------------------------------------
# _feature_col
# -----------------------------------------------------------------------------

class TestFeatureCol:
    def test_aa(self):
        assert dtc._feature_col(_df([{"amino_acid": "K"}])) == "amino_acid"

    def test_codon(self):
        assert dtc._feature_col(_df([{"codon": "AAG"}])) == "codon"

    def test_missing(self):
        with pytest.raises(ValueError, match="Neither"):
            dtc._feature_col(_df([{"other": 1}]))


# -----------------------------------------------------------------------------
# _tp_rank
# -----------------------------------------------------------------------------

class TestTpRank:
    @pytest.mark.parametrize("tp,expected", [
        ("day_0", 0),
        ("day_5", 5),
        ("day_10", 10),
        ("day_100", 100),
    ])
    def test_numeric(self, tp, expected):
        assert dtc._tp_rank(tp) == expected

    def test_no_digits_pushes_to_end(self):
        # No digits -> 1e9 so the row sorts last.
        assert dtc._tp_rank("baseline") == 10**9


# -----------------------------------------------------------------------------
# _sort_by_site
# -----------------------------------------------------------------------------

class TestSortBySite:
    def test_canonical_apE_order(self):
        # Input rows in alphabetical (A, E, P) order; expect canonical A, P, E.
        df = _df([
            {"site": "A", "x": 1},
            {"site": "E", "x": 2},
            {"site": "P", "x": 3},
        ])
        out = dtc._sort_by_site(df)
        assert list(out["site"]) == ["A", "P", "E"]

    def test_with_timepoint_outer(self):
        # Across two TPs, A/P/E order must hold inside each TP, and TPs in
        # chronological order.
        df = _df([
            {"site": "E", "timepoint": "day_10", "x": 1},
            {"site": "A", "timepoint": "day_0", "x": 2},
            {"site": "P", "timepoint": "day_0", "x": 3},
            {"site": "A", "timepoint": "day_10", "x": 4},
            {"site": "P", "timepoint": "day_10", "x": 5},
            {"site": "E", "timepoint": "day_0", "x": 6},
        ])
        out = dtc._sort_by_site(df, "timepoint")
        assert list(zip(out["timepoint"], out["site"])) == [
            ("day_0", "A"), ("day_0", "P"), ("day_0", "E"),
            ("day_10", "A"), ("day_10", "P"), ("day_10", "E"),
        ]

    def test_no_site_column_returns_unchanged(self):
        df = _df([{"x": 1}, {"x": 2}])
        out = dtc._sort_by_site(df)
        pd.testing.assert_frame_equal(out, df)


# -----------------------------------------------------------------------------
# _wilcoxon_median_cols + _wilcoxon_with_lowcount
# -----------------------------------------------------------------------------

class TestWilcoxonHelpers:
    def test_finds_two_median_cols(self):
        df = _df([{"median_BWM": 0.1, "median_control": 0.2, "x": 1}])
        assert dtc._wilcoxon_median_cols(df) == ("median_BWM", "median_control")

    def test_rejects_wrong_count(self):
        with pytest.raises(ValueError):
            dtc._wilcoxon_median_cols(_df([{"median_only": 1}]))

    def test_lowcount_flag(self):
        df = _df([
            {"median_BWM": 0.010, "median_control": 0.020},  # min=0.010, not low
            {"median_BWM": 0.001, "median_control": 0.020},  # min=0.001, low
            {"median_BWM": 0.004, "median_control": 0.005},  # min=0.004, low
            {"median_BWM": 0.005, "median_control": 0.005},  # min=0.005, NOT low (strict <)
        ])
        out, a, b = dtc._wilcoxon_with_lowcount(df, threshold=0.005)
        assert a == "median_BWM" and b == "median_control"
        assert list(out["low_count"]) == [False, True, True, False]
        assert list(out["min_median"]) == [0.010, 0.001, 0.004, 0.005]


# -----------------------------------------------------------------------------
# select_wilcoxon (family #1-#8)
# -----------------------------------------------------------------------------

class TestSelectWilcoxon:

    @staticmethod
    def _make_csv() -> pd.DataFrame:
        # Site A: 4 enriched + 4 depleted, with distinct raw p so ranking is
        # unambiguous. Include one low-count row.
        rows = []
        # Enriched (log2_FC > 0), ranked by p_value ascending
        for i, (aa, p) in enumerate([("K", 1e-6), ("R", 1e-5), ("L", 1e-4),
                                      ("G", 1e-3), ("V", 1e-2), ("I", 5e-2)]):
            rows.append({
                "site": "A", "amino_acid": aa,
                "median_BWM": 0.05, "median_control": 0.02,
                "log2_FC": 1.5 - 0.1 * i,
                "p_value": p, "p_adj": p * 2,
            })
        # Depleted (log2_FC < 0)
        for i, (aa, p) in enumerate([("D", 2e-6), ("E", 2e-5), ("N", 2e-4),
                                      ("Q", 2e-3), ("S", 2e-2), ("T", 9e-2)]):
            rows.append({
                "site": "A", "amino_acid": aa,
                "median_BWM": 0.02, "median_control": 0.05,
                "log2_FC": -(1.5 - 0.1 * i),
                "p_value": p, "p_adj": p * 2,
            })
        # One low-count row at site P to verify the audit picks it up
        rows.append({
            "site": "P", "amino_acid": "W",
            "median_BWM": 0.001, "median_control": 0.020,
            "log2_FC": 0.3, "p_value": 0.4, "p_adj": 0.4,
        })
        return _df(rows)

    def test_top5_per_direction_and_lowcount_audit(self, capsys):
        df = self._make_csv()
        dtc.select_wilcoxon(df, top_n=5, low_count_threshold=0.005)
        out = capsys.readouterr().out

        # Low-count audit fires for the single P:W row.
        assert "1 of 13 rows fall below the threshold" in out
        assert "W" in out

        # Site A Enriched: top-5 by p ascending = K, R, L, G, V (NOT I).
        a_enriched = out.split("### site A -- Enriched")[1].split("###")[0]
        assert " K " in a_enriched and " R " in a_enriched and " L " in a_enriched
        assert " G " in a_enriched and " V " in a_enriched
        assert " I " not in a_enriched  # 6th, dropped by top-5 cap

        # Site A Depleted: top-5 by p ascending = D, E, N, Q, S (NOT T).
        a_depleted = out.split("### site A -- Depleted")[1].split("###")[0]
        assert " D " in a_depleted and " E " in a_depleted and " N " in a_depleted
        assert " Q " in a_depleted and " S " in a_depleted
        assert " T " not in a_depleted

    def test_empty_direction_prints_no_rows(self, capsys):
        # Build a single-site CSV with only enriched rows so the depleted
        # sub-table comes up empty.
        df = _df([
            {"site": "A", "amino_acid": "K",
             "median_BWM": 0.05, "median_control": 0.02,
             "log2_FC": 1.0, "p_value": 0.01, "p_adj": 0.02},
        ])
        dtc.select_wilcoxon(df)
        out = capsys.readouterr().out
        a_depleted = out.split("### site A -- Depleted")[1]
        assert "(no rows clear the filter)" in a_depleted


# -----------------------------------------------------------------------------
# select_fisher_aa (family #9)
# -----------------------------------------------------------------------------

class TestSelectFisherAA:

    @staticmethod
    def _make_csv() -> pd.DataFrame:
        rows = []
        # day_0, site A: K (sig, +large), R (sig, +small), N (sig, -large),
        #                S (ns, +tiny), D (ns, -tiny)
        rows.extend([
            {"site": "A", "timepoint": "day_0", "amino_acid": "K",
             "odds_ratio": 2.0, "p_value": 1e-10, "p_adj": 1e-9,
             "BWM_count": 200, "BWM_total": 5000,
             "control_count": 100, "control_total": 5000},
            {"site": "A", "timepoint": "day_0", "amino_acid": "R",
             "odds_ratio": 1.5, "p_value": 1e-5, "p_adj": 1e-4,
             "BWM_count": 150, "BWM_total": 5000,
             "control_count": 100, "control_total": 5000},
            {"site": "A", "timepoint": "day_0", "amino_acid": "N",
             "odds_ratio": 0.5, "p_value": 1e-8, "p_adj": 1e-7,
             "BWM_count": 50, "BWM_total": 5000,
             "control_count": 100, "control_total": 5000},
            {"site": "A", "timepoint": "day_0", "amino_acid": "S",
             "odds_ratio": 1.05, "p_value": 0.5, "p_adj": 0.5,
             "BWM_count": 105, "BWM_total": 5000,
             "control_count": 100, "control_total": 5000},
            {"site": "A", "timepoint": "day_0", "amino_acid": "Z",
             "odds_ratio": 0.0, "p_value": 1e-3, "p_adj": 1e-2,
             "BWM_count": 0, "BWM_total": 5000,
             "control_count": 50, "control_total": 5000},  # OR=0 -> dropped
        ])
        return _df(rows)

    def test_padj_filter_and_OR0_drop(self, capsys):
        dtc.select_fisher_aa(self._make_csv(), top_n=5, p_thresh=0.05)
        out = capsys.readouterr().out

        # Section per (timepoint, site, direction). Enriched at day_0/A has K, R.
        enr = out.split("### day_0, site A -- Enriched")[1].split("###")[0]
        assert " K " in enr and " R " in enr
        # S has p_adj=0.5, dropped by filter.
        assert " S " not in enr
        # Z has OR=0 -> NaN log2_OR -> dropped.
        assert " Z " not in enr

        # Depleted at day_0/A has only N.
        dep = out.split("### day_0, site A -- Depleted")[1].split("###")[0]
        assert " N " in dep

    def test_rare_aa_audit_and_flag_column(self, capsys):
        # Two enriched rows at day_0/A: K is well above the threshold (no
        # flag); rare_K has BWM_count=40 (below 100, flagged).
        rows = [
            {"site": "A", "timepoint": "day_0", "amino_acid": "K",
             "odds_ratio": 2.0, "p_value": 1e-8, "p_adj": 1e-7,
             "BWM_count": 300, "BWM_total": 5000,
             "control_count": 200, "control_total": 5000},
            {"site": "A", "timepoint": "day_0", "amino_acid": "W",
             "odds_ratio": 3.0, "p_value": 1e-6, "p_adj": 1e-5,
             "BWM_count": 40, "BWM_total": 5000,
             "control_count": 200, "control_total": 5000},
        ]
        dtc.select_fisher_aa(_df(rows), top_n=5, p_thresh=0.05, rare_k=100)
        out = capsys.readouterr().out

        # Pre-filter audit fires: 1 of 2 rows trips rare-aa.
        assert "### rare-aa audit" in out
        assert "1 of 2 rows fall below the threshold" in out

        # Enriched sub-table includes the `rare_aa` column with True/False.
        enr = out.split("### day_0, site A -- Enriched")[1].split("###")[0]
        assert "rare_aa" in enr
        assert "BWM_count" in enr and "control_count" in enr
        # K row carries False; W row carries True.
        k_line = [ln for ln in enr.splitlines() if " K " in ln][0]
        w_line = [ln for ln in enr.splitlines() if " W " in ln][0]
        assert k_line.rstrip().endswith("False")
        assert w_line.rstrip().endswith("True")

    def test_rare_k_threshold_is_strict_less_than(self, capsys):
        # k exactly at the threshold (100) is NOT rare.
        rows = [
            {"site": "A", "timepoint": "day_0", "amino_acid": "K",
             "odds_ratio": 2.0, "p_value": 1e-8, "p_adj": 1e-7,
             "BWM_count": 100, "BWM_total": 5000,
             "control_count": 100, "control_total": 5000},
        ]
        dtc.select_fisher_aa(_df(rows), rare_k=100)
        out = capsys.readouterr().out
        assert "0 of 1 rows fall below the threshold" in out
        enr = out.split("### day_0, site A -- Enriched")[1].split("###")[0]
        k_line = [ln for ln in enr.splitlines() if " K " in ln][0]
        assert k_line.rstrip().endswith("False")

    def test_top_n_cap_per_direction(self, capsys):
        # Build 6 enriched cells in one (TP, site) and verify top_n=5 caps.
        rows = [
            {"site": "A", "timepoint": "day_0", "amino_acid": chr(ord("A") + i),
             "odds_ratio": 1.5 + 0.1 * (6 - i),  # K=largest, then descending
             "p_value": 1e-3, "p_adj": 1e-3,
             "BWM_count": 200, "BWM_total": 5000,
             "control_count": 100, "control_total": 5000}
            for i in range(6)
        ]
        # Re-label so OR magnitudes are distinct and ranked by |log2_OR|.
        for i, r in enumerate(rows):
            r["odds_ratio"] = 5.0 - 0.5 * i  # row0: 5.0, row1: 4.5, ..., row5: 2.5
        dtc.select_fisher_aa(_df(rows), top_n=5)
        out = capsys.readouterr().out
        enr = out.split("### day_0, site A -- Enriched")[1].split("###")[0]
        # 6 candidates, top 5 by |log2_OR| -> the lowest-OR row (last alphabet
        # letter, OR=2.5) must be dropped.
        dropped_letter = chr(ord("A") + 5)
        assert f" {dropped_letter} " not in enr


# -----------------------------------------------------------------------------
# select_fisher_codon (family #10)
# -----------------------------------------------------------------------------

class TestSelectFisherCodon:

    def test_kmin_audit_and_filter(self, capsys):
        rows = [
            # Combined k = 200; passes k_min=50; enriched.
            {"site": "A", "timepoint": "day_0", "codon": "AAG",
             "odds_ratio": 2.0, "p_value": 1e-8, "p_adj": 1e-7,
             "BWM_count": 100, "BWM_total": 5000,
             "control_count": 100, "control_total": 5000},
            # Combined k = 30; FAILS k_min=50; would be enriched if not for filter.
            {"site": "A", "timepoint": "day_0", "codon": "TTT",
             "odds_ratio": 3.0, "p_value": 1e-6, "p_adj": 1e-5,
             "BWM_count": 20, "BWM_total": 5000,
             "control_count": 10, "control_total": 5000},
            # Combined k = 150; passes; depleted.
            {"site": "A", "timepoint": "day_0", "codon": "CGA",
             "odds_ratio": 0.4, "p_value": 1e-7, "p_adj": 1e-6,
             "BWM_count": 50, "BWM_total": 5000,
             "control_count": 100, "control_total": 5000},
        ]
        dtc.select_fisher_codon(_df(rows), top_n=5, p_thresh=0.05, k_min=50)
        out = capsys.readouterr().out

        # Audit announces 1 row excluded by k_min.
        assert "1 of 3 rows fall below the threshold" in out

        enr = out.split("### day_0, site A -- Enriched")[1].split("###")[0]
        assert "AAG" in enr
        assert "TTT" not in enr  # excluded by k_min
        dep = out.split("### day_0, site A -- Depleted")[1].split("###")[0]
        assert "CGA" in dep

    def test_rare_codon_is_independent_of_kmin(self, capsys):
        # combined_k = 60 passes k_min=50, but split 55/5 means ctrl_k=5 is
        # < 100 -> rare_codon=True. The two filters are orthogonal: k_min
        # gates inclusion in Top hits, rare_k is annotation only.
        rows = [
            {"site": "A", "timepoint": "day_0", "codon": "AAG",
             "odds_ratio": 4.0, "p_value": 1e-5, "p_adj": 1e-4,
             "BWM_count": 55, "BWM_total": 5000,
             "control_count": 5, "control_total": 5000},
        ]
        dtc.select_fisher_codon(_df(rows), top_n=5, k_min=50,
                                p_thresh=0.05, rare_k=100)
        out = capsys.readouterr().out
        assert "### rare-codon audit" in out
        assert "1 of 1 rows fall below the threshold" in out
        enr = out.split("### day_0, site A -- Enriched")[1].split("###")[0]
        assert "rare_codon" in enr
        aag_line = [ln for ln in enr.splitlines() if "AAG" in ln][0]
        assert aag_line.rstrip().endswith("True")


# -----------------------------------------------------------------------------
# _tfwc_day_cols + _tfwc_with_rareflag
# -----------------------------------------------------------------------------

class TestTfwcHelpers:

    def test_finds_d10_vs_d0_cols(self):
        df = _df([{"day_10_count": 1, "day_0_count": 2, "x": 0}])
        assert dtc._tfwc_day_cols(df) == ("day_10_count", "day_0_count")

    def test_finds_d5_vs_d0_cols(self):
        # Dynamic detection: same helper must work for the d5_vs_d0 layout.
        df = _df([{"day_5_count": 1, "day_0_count": 2, "x": 0}])
        assert dtc._tfwc_day_cols(df) == ("day_5_count", "day_0_count")

    def test_rejects_wrong_count(self):
        with pytest.raises(ValueError, match="expected 2 day_"):
            dtc._tfwc_day_cols(_df([{"day_10_count": 1}]))

    def test_flag_uses_per_condition_threshold(self):
        df = _df([
            {"condition": "BWM",     "day_10_count":  99, "day_0_count": 500},  # BWM < 100  -> flagged
            {"condition": "BWM",     "day_10_count": 500, "day_0_count": 500},  # BWM >= 100 -> not
            {"condition": "control", "day_10_count": 199, "day_0_count": 500},  # ctrl < 200 -> flagged
            {"condition": "control", "day_10_count": 500, "day_0_count": 500},  # ctrl >= 200 -> not
        ])
        out = dtc._tfwc_with_rareflag(df, low_bwm=100, low_control=200,
                                      day_a="day_10_count", day_b="day_0_count")
        assert list(out["rare_low_count"]) == [True, False, True, False]
        assert list(out["min_day_count"]) == [99, 500, 199, 500]

    def test_threshold_is_strict_less_than(self):
        # day_count == threshold is NOT flagged (matches the existing
        # wilcoxon low-count and fisher rare-k rules).
        df = _df([
            {"condition": "BWM",     "day_10_count": 100, "day_0_count": 500},  # == 100, not flagged
            {"condition": "control", "day_10_count": 200, "day_0_count": 500},  # == 200, not flagged
        ])
        out = dtc._tfwc_with_rareflag(df, low_bwm=100, low_control=200,
                                      day_a="day_10_count", day_b="day_0_count")
        assert list(out["rare_low_count"]) == [False, False]

    def test_flag_checks_min_across_both_day_cols(self):
        # The flag is min(day_a, day_b) < threshold, so either column dipping
        # below trips it.
        df = _df([
            {"condition": "BWM", "day_10_count":  50, "day_0_count": 500},  # day_a low
            {"condition": "BWM", "day_10_count": 500, "day_0_count":  50},  # day_b low
            {"condition": "BWM", "day_10_count": 500, "day_0_count": 500},  # neither low
        ])
        out = dtc._tfwc_with_rareflag(df, low_bwm=100, low_control=200,
                                      day_a="day_10_count", day_b="day_0_count")
        assert list(out["rare_low_count"]) == [True, True, False]


# -----------------------------------------------------------------------------
# select_tfwc (family #11-#16)
# -----------------------------------------------------------------------------

class TestSelectTfwc:

    def test_hard_cutoff_no_topn_cap(self, capsys):
        # 12 enriched rows all clearing p_adj<0.05; with no top-N cap, all 12
        # should appear in the Enriched sub-table. (Previous rule capped at 5.)
        rows = [
            {"condition": "BWM", "site": "A", "amino_acid": f"X{i}",
             "odds_ratio": 2.0 + 0.01 * i,
             "p_value": 1e-3, "p_adj": 0.01,
             "day_10_count": 500, "day_0_count": 500}
            for i in range(12)
        ]
        dtc.select_tfwc(_df(rows), p_thresh=0.05)
        out = capsys.readouterr().out
        enr = out.split("### BWM, site A -- Enriched")[1].split("###")[0]
        for i in range(12):
            assert f"X{i}" in enr

    def test_hard_cutoff_excludes_above_threshold(self, capsys):
        # Two rows: one clears p_adj<0.05, one does not. The above-threshold
        # row must NOT appear in any Top-hits sub-table -- no fallback.
        rows = [
            {"condition": "BWM", "site": "A", "amino_acid": "K",
             "odds_ratio": 2.0,
             "p_value": 1e-5, "p_adj": 0.01,  # passes
             "day_10_count": 500, "day_0_count": 500},
            {"condition": "BWM", "site": "A", "amino_acid": "R",
             "odds_ratio": 2.5,
             "p_value": 0.05, "p_adj": 0.08,  # fails
             "day_10_count": 500, "day_0_count": 500},
        ]
        dtc.select_tfwc(_df(rows), p_thresh=0.05)
        out = capsys.readouterr().out
        enr = out.split("### BWM, site A -- Enriched")[1].split("###")[0]
        assert " K " in enr
        assert " R " not in enr  # excluded by hard cutoff, no fallback
        # The Enriched section header advertises the active cutoff.
        assert "all rows with p_adj < 0.05" in out

    def test_p_thresh_threshold_is_strict_less_than(self, capsys):
        # p_adj == p_thresh is NOT kept (matches every other audit's strict <).
        # When the only row is excluded, the (condition, site) groupby is empty
        # so no sub-table header prints.
        rows = [
            {"condition": "BWM", "site": "A", "amino_acid": "K",
             "odds_ratio": 2.0,
             "p_value": 0.05, "p_adj": 0.05,  # == threshold, excluded
             "day_10_count": 500, "day_0_count": 500},
        ]
        dtc.select_tfwc(_df(rows), p_thresh=0.05)
        out = capsys.readouterr().out
        # Sanity: select_tfwc ran (the pre-filter audit always prints).
        assert "### rare-low-count audit" in out
        # No per-(condition, site) sub-table -- the row was excluded.
        assert "### BWM, site A" not in out

    def test_p_thresh_override_changes_membership(self, capsys):
        # Same row, two thresholds. p_adj=0.07 fails 0.05 but passes 0.10.
        rows = [
            {"condition": "BWM", "site": "A", "amino_acid": "K",
             "odds_ratio": 2.0,
             "p_value": 0.05, "p_adj": 0.07,
             "day_10_count": 500, "day_0_count": 500},
        ]
        df = _df(rows)

        # Default 0.05 -> row excluded -> no sub-table for (BWM, A).
        dtc.select_tfwc(df, p_thresh=0.05)
        out = capsys.readouterr().out
        assert "### BWM, site A" not in out

        # Threshold raised to 0.10 -> row passes -> sub-table prints with K.
        dtc.select_tfwc(df, p_thresh=0.10)
        out = capsys.readouterr().out
        enr = out.split("### BWM, site A -- Enriched")[1].split("###")[0]
        assert " K " in enr

    def test_ranking_by_padj_then_abs_effect(self, capsys):
        # Three enriched rows in one (condition, site) cell. Ranking is p_adj
        # ascending, then |log2_OR| descending as tiebreaker.
        rows = [
            # smallest p_adj -> first
            {"condition": "BWM", "site": "A", "amino_acid": "A1",
             "odds_ratio": 1.5, "p_value": 1e-3, "p_adj": 0.001,
             "day_10_count": 500, "day_0_count": 500},
            # tied p_adj with A3 -> larger |log2_OR| wins -> A2 before A3
            {"condition": "BWM", "site": "A", "amino_acid": "A2",
             "odds_ratio": 4.0, "p_value": 1e-3, "p_adj": 0.020,
             "day_10_count": 500, "day_0_count": 500},
            {"condition": "BWM", "site": "A", "amino_acid": "A3",
             "odds_ratio": 1.5, "p_value": 1e-3, "p_adj": 0.020,
             "day_10_count": 500, "day_0_count": 500},
        ]
        dtc.select_tfwc(_df(rows), p_thresh=0.05)
        out = capsys.readouterr().out
        enr = out.split("### BWM, site A -- Enriched")[1].split("###")[0]
        # Confirm A1 prints first (smallest p_adj), then A2 (larger |log2_OR|
        # at tied p_adj), then A3.
        order = [aa for aa in ["A1", "A2", "A3"] if aa in enr]
        idx = [enr.index(aa) for aa in order]
        assert order == ["A1", "A2", "A3"]
        assert idx == sorted(idx)

    def test_rare_low_count_audit_and_flag_column(self, capsys):
        # Two BWM enriched rows at site A: K is well-sampled; rare_W has
        # day_10=40 (below 100, flagged). Both clear FDR<0.10 so they enter
        # the Top-hits sub-table with the `rare_low_count` column attached.
        rows = [
            {"condition": "BWM", "site": "A", "amino_acid": "K",
             "odds_ratio": 2.0, "p_value": 1e-8, "p_adj": 1e-7,
             "day_10_count": 300, "day_0_count": 200},
            {"condition": "BWM", "site": "A", "amino_acid": "W",
             "odds_ratio": 3.0, "p_value": 1e-6, "p_adj": 1e-5,
             "day_10_count":  40, "day_0_count": 200},
        ]
        dtc.select_tfwc(_df(rows), p_thresh=0.05,
                        low_bwm=100, low_control=200)
        out = capsys.readouterr().out

        # Pre-filter audit fires: 1 of 2 rows trips rare-low-count.
        assert "### rare-low-count audit" in out
        assert "1 of 2 rows fall below the threshold" in out
        # Audit header echoes the active rule (used for cross-check vs the qmd).
        assert "< 100 for BWM" in out and "< 200 for control" in out

        # Enriched sub-table includes `rare_low_count` + the day_*_count columns.
        enr = out.split("### BWM, site A -- Enriched")[1].split("###")[0]
        assert "rare_low_count" in enr
        assert "day_10_count" in enr and "day_0_count" in enr
        # K -> False (well-sampled); W -> True (rare).
        k_line = [ln for ln in enr.splitlines() if " K " in ln][0]
        w_line = [ln for ln in enr.splitlines() if " W " in ln][0]
        assert k_line.rstrip().endswith("False")
        assert w_line.rstrip().endswith("True")

    def test_per_condition_threshold_in_one_call(self, capsys):
        # Mixed-condition CSV: BWM uses 100, control uses 200. Build one rare
        # row per condition that would NOT be flagged if the thresholds were
        # swapped, so the audit count uniquely fixes the per-condition routing.
        rows = [
            # day_10=99, BWM -> flagged (99 < 100). With control's 200 it would
            # also be flagged, so this row doesn't disambiguate.
            {"condition": "BWM", "site": "A", "amino_acid": "K",
             "odds_ratio": 2.0, "p_value": 1e-3, "p_adj": 1e-3,
             "day_10_count": 99, "day_0_count": 500},
            # day_10=150, control -> flagged (150 < 200). With BWM's 100 it
            # would NOT be flagged -> only flagged if the control branch is hit.
            {"condition": "control", "site": "A", "amino_acid": "R",
             "odds_ratio": 2.0, "p_value": 1e-3, "p_adj": 1e-3,
             "day_10_count": 150, "day_0_count": 500},
            # day_10=120, BWM -> NOT flagged (120 >= 100). With control's 200
            # it WOULD be flagged -> only NOT flagged if the BWM branch is hit.
            {"condition": "BWM", "site": "A", "amino_acid": "G",
             "odds_ratio": 2.0, "p_value": 1e-3, "p_adj": 1e-3,
             "day_10_count": 120, "day_0_count": 500},
        ]
        dtc.select_tfwc(_df(rows), p_thresh=0.05,
                        low_bwm=100, low_control=200)
        out = capsys.readouterr().out
        # Exactly 2 of 3 -> per-condition routing wired correctly.
        assert "2 of 3 rows fall below the threshold" in out

    def test_threshold_override_via_args(self, capsys):
        # Mirrors the real-data boundary case (control,P,Q at day_10=204).
        # With the default 200 threshold the row is NOT flagged; raising to
        # 250 flips it to True.
        rows = [{
            "condition": "control", "site": "P", "amino_acid": "Q",
            "odds_ratio": 0.7, "p_value": 1e-4, "p_adj": 1e-3,
            "day_10_count": 204, "day_0_count": 865,
        }]
        df = _df(rows)

        dtc.select_tfwc(df, p_thresh=0.05,
                        low_bwm=100, low_control=200)
        out = capsys.readouterr().out
        assert "0 of 1 rows fall below the threshold" in out

        dtc.select_tfwc(df, p_thresh=0.05,
                        low_bwm=100, low_control=250)
        out = capsys.readouterr().out
        assert "1 of 1 rows fall below the threshold" in out

    def test_handles_d5_vs_d0_layout(self, capsys):
        # The day_*_count columns are detected dynamically, so a CSV using
        # the d5_vs_d0 layout works without any per-CSV configuration.
        rows = [{
            "condition": "BWM", "site": "A", "amino_acid": "K",
            "odds_ratio": 2.0, "p_value": 1e-5, "p_adj": 1e-4,
            "day_5_count": 50, "day_0_count": 500,
        }]
        dtc.select_tfwc(_df(rows), p_thresh=0.05,
                        low_bwm=100, low_control=200)
        out = capsys.readouterr().out
        assert "min(day_5_count, day_0_count)" in out
        assert "1 of 1 rows fall below the threshold" in out


# -----------------------------------------------------------------------------
# main() — CLI default routing for tfwc thresholds
# -----------------------------------------------------------------------------

class TestMainTfwcDefaults:
    """The AA-vs-codon default routing lives in main() (not select_tfwc), so
    these tests cover the CLI entry point. The audit-header line echoes the
    active thresholds, which lets us assert defaults without coupling to
    internal call args."""

    @staticmethod
    def _write_csv(tmp_path, feature_col: str):
        rows = [{
            "condition": "BWM", "site": "A", feature_col: "K",
            "odds_ratio": 2.0, "p_value": 1e-3, "p_adj": 1e-3,
            "day_10_count": 500, "day_0_count": 500,
        }]
        path = tmp_path / f"tfwc_{feature_col}.csv"
        _df(rows).to_csv(path, index=False)
        return path

    def test_aa_defaults_to_100_and_200(self, tmp_path, capsys, monkeypatch):
        csv = self._write_csv(tmp_path, "amino_acid")
        monkeypatch.setattr(sys, "argv",
                            ["dylan_table_checker.py", str(csv), "--family", "tfwc"])
        assert dtc.main() == 0
        out = capsys.readouterr().out
        assert "< 100 for BWM, < 200 for control" in out

    def test_codon_defaults_to_50_and_50(self, tmp_path, capsys, monkeypatch):
        csv = self._write_csv(tmp_path, "codon")
        monkeypatch.setattr(sys, "argv",
                            ["dylan_table_checker.py", str(csv), "--family", "tfwc"])
        assert dtc.main() == 0
        out = capsys.readouterr().out
        assert "< 50 for BWM, < 50 for control" in out

    def test_cli_flag_overrides_default(self, tmp_path, capsys, monkeypatch):
        csv = self._write_csv(tmp_path, "amino_acid")
        monkeypatch.setattr(sys, "argv", [
            "dylan_table_checker.py", str(csv), "--family", "tfwc",
            "--rare-low-bwm-threshold", "75",
            "--rare-low-control-threshold", "250",
        ])
        assert dtc.main() == 0
        out = capsys.readouterr().out
        assert "< 75 for BWM, < 250 for control" in out

    def test_p_adj_default_is_005(self, tmp_path, capsys, monkeypatch):
        csv = self._write_csv(tmp_path, "amino_acid")
        monkeypatch.setattr(sys, "argv",
                            ["dylan_table_checker.py", str(csv), "--family", "tfwc"])
        assert dtc.main() == 0
        out = capsys.readouterr().out
        assert "all rows with p_adj < 0.05" in out

    def test_p_adj_cli_override(self, tmp_path, capsys, monkeypatch):
        csv = self._write_csv(tmp_path, "amino_acid")
        monkeypatch.setattr(sys, "argv", [
            "dylan_table_checker.py", str(csv), "--family", "tfwc",
            "--tfwc-p-adj-threshold", "0.10",
        ])
        assert dtc.main() == 0
        out = capsys.readouterr().out
        assert "all rows with p_adj < 0.1" in out


# -----------------------------------------------------------------------------
# select_binom_codon (family #18)
# -----------------------------------------------------------------------------

class TestSelectBinomCodon:

    def test_observed_count_filter_and_padj_filter(self, capsys):
        rows = [
            # Passes k_min=50 AND p_adj<0.05; enriched.
            {"group": "BWM_day0", "site": "A", "codon": "AAG",
             "log2_enrichment": 1.5,
             "observed_count": 200, "p_value": 1e-5, "p_adj": 1e-4},
            # FAILS k_min=50; would otherwise pass.
            {"group": "BWM_day0", "site": "A", "codon": "TTT",
             "log2_enrichment": 2.0,
             "observed_count": 30, "p_value": 1e-5, "p_adj": 1e-4},
            # Passes k_min but FAILS p_adj.
            {"group": "BWM_day0", "site": "A", "codon": "GGG",
             "log2_enrichment": 0.5,
             "observed_count": 100, "p_value": 0.5, "p_adj": 0.5},
            # Passes both; depleted.
            {"group": "BWM_day0", "site": "A", "codon": "CGA",
             "log2_enrichment": -1.2,
             "observed_count": 80, "p_value": 1e-4, "p_adj": 1e-3},
        ]
        dtc.select_binom_codon(_df(rows), top_n=5, k_min=50, p_thresh=0.05)
        out = capsys.readouterr().out

        enr = out.split("### BWM_day0, site A -- Enriched")[1].split("###")[0]
        assert "AAG" in enr
        assert "TTT" not in enr and "GGG" not in enr

        dep = out.split("### BWM_day0, site A -- Depleted")[1].split("###")[0]
        assert "CGA" in dep


# -----------------------------------------------------------------------------
# select_binom_aa (family #17 — provisional rule + warning)
# -----------------------------------------------------------------------------

class TestSelectBinomAA:

    def test_warning_printed_and_codon_rule_applied(self, capsys):
        rows = [
            {"group": "BWM_day0", "site": "A", "amino_acid": "K",
             "log2_enrichment": 1.0,
             "observed_count": 200, "p_value": 1e-3, "p_adj": 1e-2},
            {"group": "BWM_day0", "site": "A", "amino_acid": "R",
             "log2_enrichment": 0.8,
             "observed_count": 30, "p_value": 1e-3, "p_adj": 1e-2},  # k_min fail
        ]
        dtc.select_binom_aa(_df(rows), top_n=5, k_min=50, p_thresh=0.05)
        out = capsys.readouterr().out
        assert "rule for AA binomial is TBD" in out
        enr = out.split("### BWM_day0, site A -- Enriched")[1].split("###")[0]
        assert " K " in enr
        assert " R " not in enr


# -----------------------------------------------------------------------------
# Family dispatch
# -----------------------------------------------------------------------------

class TestFamilyMap:
    def test_all_six_families_registered(self):
        assert set(dtc.FAMILY_MAP.keys()) == {
            "wilcoxon", "fisher_aa", "fisher_codon",
            "tfwc", "binom_aa", "binom_codon",
        }
