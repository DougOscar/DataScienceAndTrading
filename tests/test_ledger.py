"""Ledger: append-only studies, tamper-evident chain, one-shot holdout, trial store."""

from __future__ import annotations

import json
from datetime import date

import polars as pl
import pytest

from quantlab import ledger
from quantlab.ledger import LedgerError


def _study(**over):
    base = dict(study_id="fbs-0007-a1", book="FBS", system="donchian", issue=7, attempt=1,
                dev_window=["2016-05-02", "2025-05-14"], cost_model_version="fbs-v0",
                cv_scheme="CPCV(10,2)")
    return {**base, **over}


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(ledger, "git_commit", lambda: "abc1234")


def test_study_id_format_and_attempt_limit():
    assert ledger.new_study_id("fbs", 7, 2) == "fbs-0007-a2"
    with pytest.raises(LedgerError):
        ledger.new_study_id("FBS", 7, 4)


def test_create_and_fold_events(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    ledger.log_event("fbs-0007-a1", "trials", ledger_dir=tmp_path, n_trials=480)
    ledger.log_event("fbs-0007-a1", "gates", ledger_dir=tmp_path, gates={"dsr": 0.97})
    s = ledger.studies(tmp_path)["fbs-0007-a1"]
    assert s["n_trials"] == 480 and s["gates"] == {"dsr": 0.97}
    assert s["events"] == ["study_created", "trials", "gates"]
    assert ledger.system_trial_count("FBS", "donchian", tmp_path) == 480


def test_duplicate_study_and_missing_fields_rejected(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    with pytest.raises(LedgerError, match="already exists"):
        ledger.create_study(ledger_dir=tmp_path, **_study())
    with pytest.raises(LedgerError, match="missing"):
        ledger.create_study(ledger_dir=tmp_path, study_id="x")
    with pytest.raises(LedgerError, match="unknown study"):
        ledger.log_event("nope", "note", ledger_dir=tmp_path)


def test_chain_detects_tampering(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    ledger.log_event("fbs-0007-a1", "trials", ledger_dir=tmp_path, n_trials=10)
    ledger.log_event("fbs-0007-a1", "decision", ledger_dir=tmp_path, decision="fail")
    ledger.verify_chain(tmp_path)

    path = tmp_path / ledger.STUDIES_FILE
    lines = path.read_text().splitlines()
    row = json.loads(lines[1])
    row["n_trials"] = 1                       # quietly shrink the trial count
    lines[1] = json.dumps(row, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(LedgerError, match="chain broken"):
        ledger.verify_chain(tmp_path)


def test_deleting_a_row_is_detected(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    ledger.log_event("fbs-0007-a1", "note", ledger_dir=tmp_path, text="a")
    ledger.log_event("fbs-0007-a1", "note", ledger_dir=tmp_path, text="b")
    path = tmp_path / ledger.STUDIES_FILE
    lines = path.read_text().splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n")
    with pytest.raises(LedgerError):
        ledger.verify_chain(tmp_path)


def test_holdout_is_one_shot(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    assert not ledger.is_holdout_unlocked("FBS", "donchian", tmp_path)
    with pytest.raises(LedgerError, match="pass band"):
        ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                     pass_band={}, ledger_dir=tmp_path)
    ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                 pass_band={"sharpe_p10": 0.4}, ledger_dir=tmp_path)
    assert ledger.is_holdout_unlocked("fbs", "donchian", tmp_path)
    assert not ledger.is_holdout_unlocked("B3", "donchian", tmp_path)
    with pytest.raises(LedgerError, match="cannot be unlocked twice"):
        ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                     pass_band={"sharpe_p10": 0.4}, ledger_dir=tmp_path)


def test_unlock_refused_on_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "git_commit", lambda: "abc1234-dirty")
    ledger.create_study(ledger_dir=tmp_path, **_study())
    with pytest.raises(LedgerError, match="dirty"):
        ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                     pass_band={"x": 1}, ledger_dir=tmp_path)


def test_trial_recorder_counts_every_trial_and_builds_return_matrix(tmp_path):
    dates = [date(2020, 1, d) for d in (6, 7, 8)]
    with ledger.TrialRecorder("fbs-0007-a1", studies_dir=tmp_path, flush_every=2) as rec:
        for k in range(5):
            status = "pruned" if k == 3 else "ok"
            rets = pl.DataFrame({"date": dates, "ret": [0.001 * k, -0.001, 0.0]})
            rec.add({"n": 10 + k}, {"sharpe": 0.1 * k}, status=status, returns=rets)
    assert rec.n_trials == 5
    trials = ledger.load_trials("fbs-0007-a1", tmp_path)
    assert trials.height == 5 and trials["status"].to_list().count("pruned") == 1
    wide = ledger.load_trial_returns("fbs-0007-a1", tmp_path)
    assert wide.shape == (3, 6)                          # date + 5 trial columns

    # Re-opening continues numbering instead of overwriting earlier parts.
    rec2 = ledger.TrialRecorder("fbs-0007-a1", studies_dir=tmp_path)
    assert rec2.n_trials == 5
    assert rec2.add({"n": 99}, {"sharpe": 0.0}) == 5
