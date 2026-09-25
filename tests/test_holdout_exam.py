"""Holdout exam: horizon from the manifest, PASS / FAIL / NOT_DECISIVE, re-exam rules (DESIGN §4.4,
user decision 2026-09-24, red-team N5).

* The band horizon = holdout start → end of the available data (locked year + newer exports),
  read from ``data/manifest.json`` metadata only.
* All criteria pass but P(pass | zero edge) > 0.30 → NOT_DECISIVE (the system waits for more
  data); any criterion fails → FAIL (killed, never re-examined).
* A band may be rebuilt for newer data before an unlock (logged), never after it.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quantlab import config, data, ledger
from quantlab import gates as G
from quantlab import stats as S
from quantlab.ledger import LedgerError

from test_gates import make_study, register  # noqa: E402  (tests/ is on sys.path under pytest)

FBS = config.get_book("FBS")
LOCKED_END = datetime(2026, 5, 15, 10, 36)          # the FBS export today: exactly the locked year
TWO_YEARS_END = datetime(2027, 5, 14, 23, 59)       # a later export: one year of newer data


# --------------------------------------------------------------------------- fixtures
def _manifest_entry(symbol: str, market: str, end: datetime, start: datetime = datetime(2016, 5, 2)) -> dict:
    rel = f"data/{market}/{symbol}_M1_DOES_NOT_EXIST.parquet"       # never opened
    return rel, {"file": rel, "market": market, "broker": "x", "symbol": symbol, "timeframe": "M1",
                 "kind": "bars", "rows": 1, "start": str(start), "end": str(end)}


@pytest.fixture
def manifest(monkeypatch, tmp_path):
    """Point config.MANIFEST_PATH at a hand-written manifest; returns a setter for EURUSD's end."""
    path = tmp_path / "manifest.json"

    def _set(eur_end: datetime = LOCKED_END, extra: list[tuple[str, str, datetime]] = ()):
        entries = dict([_manifest_entry("EURUSD", "forex", eur_end),
                        *[_manifest_entry(s, m, e) for s, m, e in extra]])
        path.write_text(json.dumps(entries))
        monkeypatch.setattr(config, "MANIFEST_PATH", path)
        return path

    _set()
    return _set


@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(ledger, "git_commit", lambda: "abc1234")


class _WfoStudy:
    """Minimal band source: one WFO OOS series with an exact annualised Sharpe."""

    def __init__(self, sr: float, n: int = 1500, seed: int = 0, sd: float = 0.006):
        z = np.random.default_rng(seed).standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        self.mu, self.sd = sr / math.sqrt(260) * sd, sd
        r = self.mu + sd * z
        d = [date(2018, 1, 1) + timedelta(days=i) for i in range(n)]
        self.wfo_oos = pl.DataFrame({"date": d, "ret": r})
        self.cpcv_paths = None


def _band(sr: float, horizon, *, seed: int = 0, n_power: int = 600) -> S.HoldoutBand:
    return S.holdout_band(_WfoStudy(sr, seed=seed), horizon, 260.0, n_boot=1500, n_power=n_power,
                          trades_per_day=0.4, seed=seed)


def reg_band(src, horizon, study_id: str) -> dict:
    """A band built the one registrable way (R2-3): fixed n_boot / n_power, seed from the study id."""
    return S.holdout_band(src, horizon, 260.0, n_boot=S.HOLDOUT_BAND_N_BOOT, n_power=S.HOLDOUT_BAND_N_POWER,
                          trades_per_day=0.4, seed=S.holdout_band_seed(study_id)).as_dict()


# --------------------------------------------------------------------------- 1. horizon from the manifest
def test_horizon_comes_from_manifest_metadata_only(manifest, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("holdout_horizon must not open a Parquet file")
    monkeypatch.setattr(pl, "read_parquet", _boom)
    monkeypatch.setattr(pl, "scan_parquet", _boom)

    h = data.holdout_horizon("FBS", "EURUSD")
    expect = int(np.busday_count(date(2025, 5, 15), date(2026, 5, 16)))
    assert h["horizon_days"] == expect == 262
    assert h["horizon_end"] == "2026-05-15 10:36:00" and h["holdout_start"] == "2025-05-15 00:00:00"
    assert h["newer_data_included"] is False and h["horizon_source"] == "manifest"
    assert h["locked_end"] == "2026-05-15 00:00:00" and h["symbols"] == ["EURUSD"]

    manifest(TWO_YEARS_END)                                        # a newer export
    h2 = data.holdout_horizon("FBS", "EURUSD")
    assert h2["newer_data_included"] is True
    assert h2["horizon_days"] == int(np.busday_count(date(2025, 5, 15), date(2027, 5, 15)))
    assert 515 <= h2["horizon_days"] <= 525


def test_horizon_multi_symbol_crypto_and_b3(manifest):
    manifest(TWO_YEARS_END, extra=[("GBPUSD", "forex", LOCKED_END), ("BTCUSD", "crypto", datetime(2026, 5, 14, 12)),
                                   ("WIN", "b3", datetime(2026, 4, 28, 17))])
    h = data.holdout_horizon("FBS", ["EURUSD", "GBPUSD"])            # the earliest leg bounds the exam
    assert h["horizon_end"] == "2026-05-15 10:36:00" and h["newer_data_included"] is False
    c = data.holdout_horizon("FBS", "BTCUSD")                        # crypto: calendar days
    assert c["periods_per_year"] == 365.0 and c["horizon_days"] == (date(2026, 5, 14) - date(2025, 5, 15)).days + 1
    b = data.holdout_horizon("B3", "WIN")                            # B3: 252/260 of weekdays
    wd = int(np.busday_count(date(2025, 4, 29), date(2026, 4, 29)))
    assert b["horizon_days"] == round(wd * 252 / 260) and b["newer_data_included"] is False
    with pytest.raises(ValueError, match="no M1 bars"):
        data.holdout_horizon("FBS", "NOPE")


def test_gates_band_uses_manifest_horizon_and_explicit_override(manifest):
    study, trades = make_study(seed=3)
    ev = None
    rep = G.evaluate_gates(study, ev, periods_per_year=260.0, selected_trades=trades, 
                           ledger_dir=register(study), holdout_symbols="EURUSD")
    hb = rep.holdout_band
    assert hb["horizon_source"] == "manifest" and hb["horizon_days"] == 262
    assert hb["horizon_end"] == "2026-05-15 10:36:00" and hb["newer_data_included"] is False
    assert "2026-05-15 10:36:00" in rep.to_markdown()
    study2, trades2 = make_study(seed=3)
    rep2 = G.evaluate_gates(study2, None, periods_per_year=260.0, selected_trades=trades2, 
                            ledger_dir=register(study2), holdout_symbols="EURUSD", holdout_days=100)
    assert rep2.holdout_band["horizon_days"] == 100 and rep2.holdout_band["horizon_source"] == "explicit"


# --------------------------------------------------------------------------- 2. verdict
def test_holdout_status_rule_and_readonly_threshold():
    assert S.DECISIVE_MAX_ZERO_EDGE_PASS == 0.30 and G.HOLDOUT_MAX_ZERO_EDGE_PASS == 0.30
    assert S.holdout_status(False, 0.01) == "FAIL"
    assert S.holdout_status(True, 0.30) == "PASS"
    assert S.holdout_status(True, 0.31) == "NOT_DECISIVE"
    assert S.holdout_status(True, None) == "NOT_DECISIVE"           # unknown power is never decisive
    assert S.holdout_status(True, float("nan")) == "NOT_DECISIVE"
    with pytest.raises(ValueError, match="fixed"):
        S.holdout_band(_WfoStudy(2.0), 260, n_boot=200, n_power=0, max_zero_edge_pass=0.9)


def test_not_decisive_when_zero_edge_pass_probability_above_030():
    weak = _band(1.2, 260)                                            # calibration: 1 y never decisive at SR 1.2
    assert weak.p_pass_zero_edge > 0.30 and weak.decisive is False
    good = np.full(260, 0.0012) + np.random.default_rng(1).normal(0, 1e-4, 260)   # clears every criterion
    r = S.holdout_check(weak, good, 104)
    assert r["criteria_pass"] is True and r["status"] == "NOT_DECISIVE" and r["pass"] is False
    strong = _band(2.8, 260)
    assert strong.p_pass_zero_edge <= 0.30
    r2 = S.holdout_check(strong, good, 104)
    assert r2["criteria_pass"] and r2["status"] == "PASS" and r2["pass"] is True
    bad = np.full(260, -0.001)
    for b in (weak, strong):
        assert S.holdout_check(b, bad, 104)["status"] == "FAIL"
    with pytest.raises(ValueError, match="built for"):                # wrong span for this band
        S.holdout_check(strong, good[:150], 60)


def test_decisiveness_improves_with_horizon_on_a_synthetic_edge():
    """Calibration v1.2 §8.2 (1 y → 2 y zero-edge pass): SR 1.2 0.58 → 0.38; SR 1.9 0.27 → 0.07;
    SR 2.8 0.03 → 0.00.  Our iid synthetic edges must show the same ordering and verdicts at the
    clear-cut points (not decisive at SR 1.2 / 1 y; decisive at SR 1.9 / 2 y and SR 2.8)."""
    p = {(sr, h): _band(sr, h, seed=7).p_pass_zero_edge for sr in (1.2, 1.9, 2.8) for h in (260, 520)}
    for sr in (1.2, 1.9, 2.8):
        assert p[(sr, 520)] < p[(sr, 260)], (sr, p)
    assert p[(1.2, 260)] > 0.30
    assert p[(1.9, 520)] <= 0.30 and p[(2.8, 260)] <= 0.30 and p[(2.8, 520)] <= 0.05
    assert p[(1.2, 260)] > p[(1.9, 260)] > p[(2.8, 260)]


# --------------------------------------------------------------------------- 3. ledger life cycle
def _setup(tmp_path, sr: float):
    ledger.create_study(ledger_dir=tmp_path, study_id="fbs-0042-a1", book="FBS", system="probe", issue=42,
                        attempt=1, dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t")
    src = _WfoStudy(sr, seed=3)
    band = reg_band(src, data.holdout_horizon("FBS", "EURUSD"), "fbs-0042-a1")
    ledger._register_holdout_band(study_id="fbs-0042-a1", band=band, reason="S5 pre-registration", ledger_dir=tmp_path)
    return src, band


def _unlock(tmp_path, band):
    return ledger._record_holdout_unlock(book="FBS", system="probe", study_id="fbs-0042-a1", pass_band=band,
                                       user_confirmation=ledger.unlock_phrase("FBS", "probe"), ledger_dir=tmp_path)


def _exam(tmp_path, series, n_trades):
    return G.run_holdout_exam(book="FBS", system="probe", study_id="fbs-0042-a1", holdout_daily=series,
                              n_trades=n_trades, periods_per_year=260.0, ledger_dir=tmp_path)


def test_fail_blocks_any_reexam(manifest, clean_tree, tmp_path):
    _, band = _setup(tmp_path, 2.8)
    _unlock(tmp_path, band)
    res = _exam(tmp_path, np.full(band["horizon_days"], -0.001), 100)
    assert res["status"] == "FAIL"
    st = ledger.holdout_state("FBS", "probe", tmp_path)
    assert st["killed"] and not st["pending"] and ledger.studies(tmp_path)["fbs-0042-a1"]["system_state"] == "killed"
    manifest(TWO_YEARS_END)                                          # even with new data …
    with pytest.raises(LedgerError, match="killed"):
        ledger._register_holdout_band(study_id="fbs-0042-a1", band={**band, "horizon_end": "2027-05-14 23:59:00"},
                                     reason="more data", ledger_dir=tmp_path)
    with pytest.raises(LedgerError, match="FAIL can never be re-examined"):
        _unlock(tmp_path, band)
    with pytest.raises(G.GateError, match="no unlocked exam"):
        _exam(tmp_path, np.full(band["horizon_days"], 0.001), 100)


def test_band_rebuild_before_unlock_is_logged_and_after_unlock_raises(manifest, clean_tree, tmp_path):
    study, trades = make_study(seed=4)
    ld = register(study, tmp_path)
    rep = G.evaluate_gates(study, None, periods_per_year=260.0, selected_trades=trades,
                           ledger_dir=ld, holdout_symbols="EURUSD", log=True)   # S5: band registered
    assert ledger.registered_holdout_band(study.study_id, ledger_dir=ld)["horizon_days"] == 262
    with pytest.raises(G.GateError, match="no newer data"):          # same horizon: no band shopping
        G.rebuild_holdout_band(study, periods_per_year=260.0, reason="x", ledger_dir=ld)

    manifest(TWO_YEARS_END)                                          # newer data exported before the unlock
    old = rep.holdout_band
    with pytest.raises(LedgerError, match="newer data was exported"):   # stale band cannot be unlocked
        ledger._record_holdout_unlock(book="FBS", system="toy", study_id=study.study_id, pass_band=old,
                                     user_confirmation=ledger.unlock_phrase("FBS", "toy"), ledger_dir=ld)
    new = G.rebuild_holdout_band(study, periods_per_year=260.0, reason="EURUSD export to 2027-05-14",
                                 ledger_dir=ld)
    ev = ledger.study_events(study.study_id, "holdout_band_registered", ledger_dir=ld)
    assert len(ev) == 1 and ev[0]["band_version"] == 1 and ev[0]["reason"] == "EURUSD export to 2027-05-14"
    assert ev[0]["replaces_horizon_end"] == "2026-05-15 10:36:00" and ev[0]["newer_data_included"] is True
    assert new["horizon_days"] > 500 and new["seed"] == old["seed"]
    assert ledger.registered_holdout_band(study.study_id, ledger_dir=ld) == ev[0]["band"]
    with pytest.raises(LedgerError, match="registered band"):        # the old band is no longer valid
        ledger._record_holdout_unlock(book="FBS", system="toy", study_id=study.study_id, pass_band=old,
                                     user_confirmation=ledger.unlock_phrase("FBS", "toy"), ledger_dir=ld)
    row = ledger._record_holdout_unlock(book="FBS", system="toy", study_id=study.study_id,
                                       pass_band=ev[0]["band"], user_confirmation=ledger.unlock_phrase("FBS", "toy"),
                                       ledger_dir=ld)
    assert row["exam"] == 1 and row["horizon_end"] == "2027-05-14 23:59:00" and row["newer_data_included"] is True

    manifest(datetime(2027, 9, 1))                                   # more data after the unlock
    with pytest.raises(LedgerError, match="never be rebuilt after the unlock"):
        G.rebuild_holdout_band(study, periods_per_year=260.0, reason="late", ledger_dir=ld)


def test_not_decisive_waits_and_reexam_uses_full_longer_span(manifest, clean_tree, tmp_path):
    src, band = _setup(tmp_path, 1.2)
    assert band["p_pass_zero_edge"] > 0.30
    _unlock(tmp_path, band)
    # holdout data access is limited to the band's horizon until the exam is decisive
    assert ledger.holdout_access_end("FBS", "probe", tmp_path) == datetime(2026, 5, 15, 10, 37)
    n1 = band["horizon_days"]
    good = src.mu + 1e-4 * np.random.default_rng(2).standard_normal(4 * n1)   # a steady live edge
    r1 = _exam(tmp_path, good[:n1], 0.4 * n1)
    assert r1["criteria_pass"] and r1["status"] == "NOT_DECISIVE"
    st = ledger.holdout_state("FBS", "probe", tmp_path)
    assert st["last_status"] == "NOT_DECISIVE" and not st["killed"] and not st["pending"]
    s = ledger.studies(tmp_path)["fbs-0042-a1"]
    assert s["system_state"] == "holdout_pending" and s["holdout_horizon_days"] == n1

    # no re-exam on the same data: the band must reach newer data first
    with pytest.raises(LedgerError, match="past the last exam|re-exam band must"):
        _unlock(tmp_path, band)
    with pytest.raises(LedgerError, match="already covered"):
        ledger._register_holdout_band(study_id="fbs-0042-a1", band=band, reason="again", ledger_dir=tmp_path)

    manifest(TWO_YEARS_END)                                          # a year of newer data exported
    h2 = data.holdout_horizon("FBS", "EURUSD")
    band2 = reg_band(src, h2, "fbs-0042-a1")
    assert band2["p_pass_zero_edge"] < band["p_pass_zero_edge"]      # longer horizon → more decisive
    ledger._register_holdout_band(study_id="fbs-0042-a1", band=band2, reason="renewing holdout: +1 y",
                                 ledger_dir=tmp_path)
    row = _unlock(tmp_path, band2)
    assert row["exam"] == 2 and row["kind"].startswith("re-exam")
    n2 = band2["horizon_days"]
    with pytest.raises(ValueError, match="built for"):               # must judge the FULL span, not the new year
        _exam(tmp_path, good[:n2 - n1], 0.4 * (n2 - n1))
    r2 = _exam(tmp_path, good[:n2], 0.4 * n2)
    exams = ledger.holdout_state("FBS", "probe", tmp_path)["exams"]
    assert [e["exam"] for e in exams] == [1, 2]
    assert [e["horizon_days"] for e in exams] == [n1, n2]
    assert exams[1]["horizon_end"] == "2027-05-14 23:59:00" and exams[1]["newer_data_included"] is True
    assert r2["status"] in ("PASS", "NOT_DECISIVE")
    ledger.verify_chain(tmp_path)


def test_exam_record_rejects_foreign_band_or_forged_status(manifest, clean_tree, tmp_path):
    _, band = _setup(tmp_path, 2.8)
    _unlock(tmp_path, band)
    series = np.full(band["horizon_days"], 0.0012) + np.random.default_rng(1).normal(0, 1e-4, band["horizon_days"])
    other = {**band, "sharpe_lo": -99.0}
    with pytest.raises(LedgerError, match="band stored with the unlock"):
        ledger.record_holdout_exam(book="FBS", system="probe", study_id="fbs-0042-a1",
                                   result=S.holdout_check(other, series, 100), ledger_dir=tmp_path)
    forged = S.holdout_check(band, np.full(band["horizon_days"], -0.001), 100)
    forged["status"] = "PASS"
    with pytest.raises(LedgerError, match="does not follow"):
        ledger.record_holdout_exam(book="FBS", system="probe", study_id="fbs-0042-a1", result=forged,
                                   ledger_dir=tmp_path)


def test_unlock_requires_manifest_horizon(manifest, clean_tree, tmp_path):
    ledger.create_study(ledger_dir=tmp_path, study_id="fbs-0043-a1", book="FBS", system="p2", issue=43, attempt=1,
                        dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t")
    band = reg_band(_WfoStudy(2.8), 262, "fbs-0043-a1")               # explicit horizon (tests only)
    band["horizon_end"] = "2026-05-15 10:36:00"
    ledger._register_holdout_band(study_id="fbs-0043-a1", band=band, reason="S5", ledger_dir=tmp_path)
    with pytest.raises(LedgerError, match="not taken from the data manifest"):
        ledger._record_holdout_unlock(book="FBS", system="p2", study_id="fbs-0043-a1", pass_band=band,
                                     user_confirmation=ledger.unlock_phrase("FBS", "p2"), ledger_dir=tmp_path)


def test_load_bars_serves_holdout_only_up_to_the_exam_horizon(monkeypatch):
    """Until a decisive PASS, an unlocked system sees holdout bars only up to its band's horizon
    (data exported later stays locked until the band is rebuilt and a re-exam unlocked)."""
    reads = []
    monkeypatch.setattr(ledger, "log_holdout_read", lambda **kw: reads.append(kw))
    acc = {"end": datetime(2025, 6, 2), "symbols": ["EURUSD"], "study_id": "fbs-0042-a1", "system": "probe"}
    monkeypatch.setattr(ledger, "holdout_access", lambda b, s, d=None: acc if s == "probe" else None)
    bars = data.load_bars("EURUSD", "H1", end="2025-07-01", include_holdout=True, system="probe")
    assert bars["ts"].max() >= FBS.holdout_start and bars["ts"].max() < datetime(2025, 6, 2)
    assert reads and reads[-1]["symbol"] == "EURUSD" and reads[-1]["end"] == datetime(2025, 6, 2)
    acc["end"] = None                                                               # after a PASS
    bars = data.load_bars("EURUSD", "H1", end="2025-07-01", include_holdout=True, system="probe")
    assert bars["ts"].max() >= datetime(2025, 6, 30)
