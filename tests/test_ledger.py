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


def _band(sid="fbs-0007-a1", **over):
    """A registered-band stand-in whose horizon comes from the real manifest (metadata only), with
    the fixed construction fields a registrable band carries (R2-3)."""
    from quantlab import data, stats
    return {**data.holdout_horizon("FBS", "EURUSD"), "sharpe_lo": 0.4, "p_pass_zero_edge": 0.1,
            "seed": stats.holdout_band_seed(sid), "n_boot_requested": stats.HOLDOUT_BAND_N_BOOT,
            "n_power_requested": stats.HOLDOUT_BAND_N_POWER, **over}


def _registered(tmp_path, sid="fbs-0007-a1", **over):
    band = _band(sid, **over)
    ledger.register_holdout_band(study_id=sid, band=band, reason="S5", ledger_dir=tmp_path)
    return band


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
                                     pass_band={}, user_confirmation="UNLOCK HOLDOUT FBS/donchian", ledger_dir=tmp_path)
    with pytest.raises(LedgerError, match="registered band"):     # nothing registered yet
        ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1", pass_band=_band(),
                                     user_confirmation="UNLOCK HOLDOUT FBS/donchian", ledger_dir=tmp_path)
    band = _registered(tmp_path)
    ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                 pass_band=band, user_confirmation="UNLOCK HOLDOUT FBS/donchian", ledger_dir=tmp_path)
    assert ledger.is_holdout_unlocked("fbs", "donchian", tmp_path)
    assert not ledger.is_holdout_unlocked("B3", "donchian", tmp_path)
    with pytest.raises(LedgerError, match="cannot be unlocked twice"):
        ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                     pass_band=band, user_confirmation="UNLOCK HOLDOUT FBS/donchian", ledger_dir=tmp_path)


def test_unlock_refused_on_dirty_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "git_commit", lambda: "abc1234-dirty")
    ledger.create_study(ledger_dir=tmp_path, **_study())
    band = _registered(tmp_path)
    with pytest.raises(LedgerError, match="dirty"):
        ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                     pass_band=band, user_confirmation="UNLOCK HOLDOUT FBS/donchian", ledger_dir=tmp_path)


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


def test_unlock_requires_exact_user_phrase(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    band = _registered(tmp_path)
    for bad in ("", "unlock holdout FBS/donchian", "UNLOCK HOLDOUT FBS/other"):
        with pytest.raises(LedgerError, match="type exactly"):
            ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                         pass_band=band, user_confirmation=bad, ledger_dir=tmp_path)
    row = ledger.record_holdout_unlock(book="FBS", system="donchian", study_id="fbs-0007-a1",
                                       pass_band=band, user_confirmation=ledger.unlock_phrase("fbs", "donchian"),
                                       ledger_dir=tmp_path)
    assert row["user_confirmation"] == "UNLOCK HOLDOUT FBS/donchian"


def test_system_effective_trials_sums_prior_attempts(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    ledger.log_event("fbs-0007-a1", "trials", ledger_dir=tmp_path, n_trials=400)
    ledger.log_event("fbs-0007-a1", "gates", ledger_dir=tmp_path, effective_trials=31.5)
    ledger.create_study(ledger_dir=tmp_path, **_study(study_id="fbs-0007-a2", attempt=2))
    ledger.log_event("fbs-0007-a2", "trials", ledger_dir=tmp_path, n_trials=50)   # no gates yet
    assert ledger.system_effective_trials("FBS", "donchian", ledger_dir=tmp_path) == 81.5
    assert ledger.system_effective_trials("FBS", "donchian", exclude_study="fbs-0007-a2",
                                          ledger_dir=tmp_path) == 31.5


def test_system_prior_trials_uses_own_counts_and_skips_later_attempts(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    ledger.log_event("fbs-0007-a1", "trials", ledger_dir=tmp_path, n_trials=400)
    ledger.create_study(ledger_dir=tmp_path, **_study(study_id="fbs-0007-a2", attempt=2))
    ledger.log_event("fbs-0007-a2", "trials", ledger_dir=tmp_path, n_trials=50)
    # m1: a gates row carries the study's own count; cumulative fields are never summed
    ledger.log_event("fbs-0007-a2", "gates", ledger_dir=tmp_path, n_trials_study=50, n_trials_dsr=450,
                     effective_trials_study=12.0, effective_trials=40.0)
    ledger.create_study(ledger_dir=tmp_path, **_study(study_id="fbs-0007-a3", attempt=3))
    tot, used = ledger.system_prior_trials("FBS", "donchian", exclude_study="fbs-0007-a3", max_attempt=3,
                                           ledger_dir=tmp_path)
    assert tot == 450 and used == ["fbs-0007-a1", "fbs-0007-a2"]
    tot, used = ledger.system_prior_trials("FBS", "donchian", exclude_study="fbs-0007-a1", max_attempt=1,
                                           ledger_dir=tmp_path)
    assert tot == 0 and used == []
    assert ledger.system_effective_trials("FBS", "donchian", exclude_study="fbs-0007-a3",
                                          ledger_dir=tmp_path) == 400 + 12.0


def test_study_events_filters_by_study_and_event(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study())
    ledger.log_event("fbs-0007-a1", "gates", ledger_dir=tmp_path, verdict="FAIL", gate_run=1)
    ledger.log_event("fbs-0007-a1", "note", ledger_dir=tmp_path, text="x")
    ledger.log_event("fbs-0007-a1", "gates", ledger_dir=tmp_path, verdict="PASS", gate_run=2)
    ev = ledger.study_events("fbs-0007-a1", "gates", ledger_dir=tmp_path)
    assert [e["verdict"] for e in ev] == ["FAIL", "PASS"]
    assert len(ledger.study_events("fbs-0007-a1", ledger_dir=tmp_path)) == 4


def test_related_prior_trials_by_name_or_issue_in_ledger_order(tmp_path, clean_tree):
    """N2: every OTHER study that shares the normalised system name or the issue — whenever it was
    created (R2 minor N2b: re-gating an earlier attempt counts the later ones)."""
    ledger.create_study(ledger_dir=tmp_path, **_study(system="Donchian-20"))
    ledger.log_event("fbs-0007-a1", "trials", ledger_dir=tmp_path, n_trials=400)
    ledger.create_study(ledger_dir=tmp_path, **_study(study_id="fbs-0009-a1", issue=9, system="donchian_20"))
    ledger.log_event("fbs-0009-a1", "trials", ledger_dir=tmp_path, n_trials=30)
    ledger.create_study(ledger_dir=tmp_path, **_study(study_id="fbs-0010-a1", issue=10, system="rsi"))
    ledger.log_event("fbs-0010-a1", "trials", ledger_dir=tmp_path, n_trials=999)
    ledger.create_study(ledger_dir=tmp_path, **_study(study_id="fbs-0007-a1b", issue=7, system="breakout"))
    tot, used, own = ledger.related_prior_trials("fbs-0007-a1b", ledger_dir=tmp_path)
    assert tot == 400 and used == ["fbs-0007-a1"] and own["system"] == "breakout"
    tot, used, _ = ledger.related_prior_trials("fbs-0009-a1", ledger_dir=tmp_path)
    assert tot == 400 and used == ["fbs-0007-a1"]
    tot, used, _ = ledger.related_prior_trials("fbs-0007-a1", ledger_dir=tmp_path)
    assert tot == 30 and used == ["fbs-0009-a1", "fbs-0007-a1b"]   # created later: counted too
    with pytest.raises(LedgerError, match="unknown study"):
        ledger.related_prior_trials("nope", ledger_dir=tmp_path)


def test_data_dependence_flag_never_resets_and_trial_source_is_stored(tmp_path, clean_tree):
    ledger.create_study(ledger_dir=tmp_path, **_study(method="sobol", candidate_set_data_dependent=False))
    assert ledger.candidate_set_data_dependent("fbs-0007-a1", ledger_dir=tmp_path) is False
    ledger.log_event("fbs-0007-a1", "trials", ledger_dir=tmp_path, candidate_set_data_dependent=True)
    ledger.log_event("fbs-0007-a1", "selection", ledger_dir=tmp_path, candidate_set_data_dependent=False)
    assert ledger.candidate_set_data_dependent("fbs-0007-a1", ledger_dir=tmp_path) is True
    ledger.create_study(ledger_dir=tmp_path, **_study(study_id="fbs-0007-a2", attempt=2, method="tpe"))
    assert ledger.candidate_set_data_dependent("fbs-0007-a2", ledger_dir=tmp_path) is True
    with ledger.TrialRecorder("s1", tmp_path / "studies") as rec:
        rec.add({"a": 1}, {"sharpe": 0.1}, source="tpe")
        rec.add({"a": 2}, {"sharpe": 0.2})
    assert ledger.load_trials("s1", tmp_path / "studies")["source"].to_list() == ["tpe", None]
