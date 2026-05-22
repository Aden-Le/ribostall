"""Tests for scripts/dylan_table_checker.py.

Strategy: build small handcrafted DataFrames where the expected pick list,
sort order, and filter outcome are computable by hand. Family-level
`select_*` functions print directly (no return value), so end-to-end tests
use pytest's `capsys` fixture and assert on substrings of stdout.
"""

from __future__ import annotations

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
# select_tfwc (family #11-#16)
# -----------------------------------------------------------------------------

class TestSelectTfwc:

    def test_fallback_when_few_significant(self, capsys):
        # condition=BWM, site=A: only 2 rows with p_adj<0.10 -> fallback to raw-p
        # ranking with no FDR cutoff. Build 12 rows total so the pool is
        # non-trivial.
        rows = []
        for i in range(2):
            rows.append({
                "condition": "BWM", "site": "A",
                "amino_acid": f"E{i}",  # bogus AA labels are fine
                "odds_ratio": 2.0,
                "p_value": 1e-5, "p_adj": 0.01,  # passes 0.10 cutoff
            })
        for i in range(10):
            rows.append({
                "condition": "BWM", "site": "A",
                "amino_acid": f"N{i}",
                "odds_ratio": 1.1,
                "p_value": 0.2 + 0.01 * i, "p_adj": 0.5,  # all fail 0.10
            })
        dtc.select_tfwc(_df(rows), top_n=5, p_thresh=0.10, min_candidates=10)
        out = capsys.readouterr().out
        assert "fallback to raw-p ranking" in out

    def test_normal_path_when_enough_significant(self, capsys):
        # >=10 candidates at p_adj<0.10 -> use FDR-cut pool, rank by |log2_OR|.
        rows = []
        for i in range(12):
            rows.append({
                "condition": "BWM", "site": "A",
                "amino_acid": f"X{i}",
                "odds_ratio": 2.0 + 0.1 * i,
                "p_value": 1e-3, "p_adj": 0.01,
            })
        dtc.select_tfwc(_df(rows), top_n=5, p_thresh=0.10, min_candidates=10)
        out = capsys.readouterr().out
        assert "fallback to raw-p ranking" not in out


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
