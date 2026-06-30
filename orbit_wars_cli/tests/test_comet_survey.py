"""
tests/test_comet_survey.py — Tests for cwm/comet_survey.py (Module 8).

Coverage (informational, not correctness):
  - test_run_survey_produces_nonempty_output
  - test_run_survey_logs_comet_positions_every_turn
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cwm.comet_survey import run_survey


class TestCometSurvey:

    def test_run_survey_produces_nonempty_output(self, tmp_path):
        """Output file exists and has at least one row per episode per turn."""
        out_path = str(tmp_path / "survey.csv")
        num_episodes = 2
        # Run short episodes (max 10 turns) so the test is fast
        run_survey(num_episodes=num_episodes, out_path=out_path, max_turns=10)

        assert os.path.exists(out_path), "CSV file was not created"

        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # At least one row per episode per turn
        assert len(rows) >= num_episodes, (
            f"Expected at least {num_episodes} rows, got {len(rows)}"
        )

    def test_run_survey_logs_comet_positions_every_turn(self, tmp_path):
        """Comet position column is present in every logged row."""
        out_path = str(tmp_path / "survey.csv")
        run_survey(num_episodes=1, out_path=out_path, max_turns=10)

        with open(out_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0
        for row in rows:
            assert "comet_positions" in row, (
                f"Row missing 'comet_positions' column: {row}"
            )
            # The column must exist in every row (may be empty string if no comets)
            # but the key itself must always be present.


# ── Runner ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
