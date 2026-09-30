"""Tests for quantlab.systems (system-folder scaffolding, the notebook template, and the
QUANTLAB_*_DIR config overrides those two lean on -- DESIGN §1/§3/§10 Phase 2)."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import nbformat
import pytest

from quantlab import config, systems


# =============================================================================================
# config: env overrides, read at call time by ledger / opt / gates (never captured as a default
# argument value) -- see quantlab/config.py's _env_path docstring. Exercised via a *subprocess*
# (not monkeypatch): these constants are set once at import time, so the only faithful way to
# prove the env var is honoured is a fresh interpreter that reads it before `import quantlab`.
# =============================================================================================
# Prints the four directory constants only -- no filesystem writes at all -- so it is always
# safe to run with *any* env (including no override, which resolves to the real, git-tracked
# research/ledger etc.; this script must never write there).
_PATHS_SCRIPT = """
import json
from quantlab import config

print(json.dumps({
    "ledger_dir": str(config.LEDGER_DIR), "studies_dir": str(config.STUDIES_DIR),
    "systems_dir": str(config.SYSTEMS_DIR), "vault_dir": str(config.VAULT_DIR),
}))
"""

# Exercises ledger.create_study / gates.resolve_studies_dir / ledger.TrialRecorder / opt's trial
# loader -- real writes, so every caller of this script MUST override all four directories to
# locations under a tmp_path (never the defaults) or it will pollute the real, git-tracked ledger.
_OVERRIDE_SCRIPT = """
import json
from quantlab import config, gates, ledger, opt

out = {
    "ledger_dir": str(config.LEDGER_DIR), "studies_dir": str(config.STUDIES_DIR),
    "systems_dir": str(config.SYSTEMS_DIR), "vault_dir": str(config.VAULT_DIR),
}

ledger.create_study(
    study_id="fbs-9999-a1", book="FBS", system="_config_override_probe", issue=9999, attempt=1,
    dev_window=["2016-05-02", "2025-05-14"], cost_model_version="test-v0", cv_scheme="CPCV(1,1)",
    git_commit="deadbeef", data_manifest_sha="deadbeef",
)
out["ledger_file_exists"] = (config.LEDGER_DIR / "studies.jsonl").exists()
out["resolved_studies_dir"] = str(gates.resolve_studies_dir("fbs-9999-a1"))

rec = ledger.TrialRecorder("fbs-9999-a1")
rec.add({"x": 1}, {"sharpe": 1.0}, source="grid")
rec.flush()
out["trial_store_dir"] = str(rec.dir)
out["opt_loaded_trials"] = len(opt._load_existing("fbs-9999-a1", None))

print(json.dumps(out))
"""


def _run_script(script: str, env_overrides: dict[str, str]) -> dict:
    env = {**os.environ, **env_overrides}
    result = subprocess.run(
        [str(config.ROOT / ".venv" / "bin" / "python"), "-c", script],
        cwd=str(config.ROOT), env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_config_dir_overrides_are_absolute_and_read_at_call_time(tmp_path):
    real_ledger_file = config.RESEARCH_DIR / "ledger" / "studies.jsonl"
    before = real_ledger_file.read_text() if real_ledger_file.exists() else None

    ledger_dir, studies_dir, systems_dir, vault_dir = (
        tmp_path / "led", tmp_path / "stud", tmp_path / "sys", tmp_path / "vault",
    )
    out = _run_script(_OVERRIDE_SCRIPT, {
        "QUANTLAB_LEDGER_DIR": str(ledger_dir), "QUANTLAB_STUDIES_DIR": str(studies_dir),
        "QUANTLAB_SYSTEMS_DIR": str(systems_dir), "QUANTLAB_VAULT_DIR": str(vault_dir),
    })

    assert out["ledger_dir"] == str(ledger_dir)
    assert out["studies_dir"] == str(studies_dir)
    assert out["systems_dir"] == str(systems_dir)
    assert out["vault_dir"] == str(vault_dir)
    assert out["ledger_file_exists"] is True
    # gates.resolve_studies_dir (no `studies_dir` recorded on the study) falls back to
    # config.STUDIES_DIR read at *its own* call time, not whatever ledger.py saw at import.
    assert out["resolved_studies_dir"] == str(studies_dir)
    # opt._load_existing / ledger.TrialRecorder resolve the same override independently.
    assert out["trial_store_dir"] == str(studies_dir / "fbs-9999-a1")
    assert out["opt_loaded_trials"] == 1
    # And the real, git-tracked ledger was never touched by any of this.
    after = real_ledger_file.read_text() if real_ledger_file.exists() else None
    assert after == before


def test_config_dir_override_relative_path_resolves_against_root():
    # The paths-only script never writes anywhere, so it's safe to point ROOT-relative even
    # though nothing under ROOT actually exists at that name.
    out = _run_script(_PATHS_SCRIPT, {"QUANTLAB_LEDGER_DIR": "a_relative_ledger_dir_for_tests"})
    assert out["ledger_dir"] == str(config.ROOT / "a_relative_ledger_dir_for_tests")
    assert not (config.ROOT / "a_relative_ledger_dir_for_tests").exists()


def test_config_dirs_default_to_research_dir_without_any_override():
    out = _run_script(_PATHS_SCRIPT, {})
    assert out["ledger_dir"] == str(config.RESEARCH_DIR / "ledger")
    assert out["studies_dir"] == str(config.RESEARCH_DIR / "studies")
    assert out["systems_dir"] == str(config.RESEARCH_DIR / "systems")
    assert out["vault_dir"] == str(config.ROOT / "DocumentationVault" / "systems")


# =============================================================================================
# create_system / system_dir / find_system / read_card
# =============================================================================================
_CARD = """## Donchian Breakout Probe
**Book:** FBS · **Component type:** full system
**Idea (<=200 chars):** Enter on a 20-bar Donchian breakout, exit on the opposite channel.
**Mechanism:** trend continuation after a volatility-driven range breakout.
**Falsifiable prediction:** breakouts on trending FX pairs outperform random entries with the
same exit; a flat edge kills it.
**Rules:** long when close > 20-bar high (excluding the current bar), short on the mirror,
exit on the opposite channel touch.
**Risk semantics:** type A (DESIGN §5) + ATR(14) stop at entry, never widened.
**Markets & timeframes:** EURUSD H1. **Expected trades/year (per symbol):** ~40.
**Free parameters & prior ranges:** lookback: [10, 40] -- channel length.
**Cost sensitivity:** gross edge/trade ~8 pips vs ~0.9 pip daytime spread.
**Null / benchmark for component tests:** random entries with the same exit/holding-time dist.
**Sources:** Faber (2013), donchian.
**Provenance:** literature
"""


def _card(risk_line: str | None = "**Risk semantics:** type A (DESIGN §5) + ATR stop.") -> str:
    body = "## Probe\n**Idea:** a probe card for tests.\n"
    if risk_line is not None:
        body += risk_line + "\n"
    return body


def test_create_system_layout_and_refuses_to_overwrite(tmp_path):
    d = systems.create_system("FBS", 7, "donchian_breakout", name="Donchian Breakout",
                              card_markdown=_CARD, systems_dir=tmp_path)

    assert d == tmp_path / "FBS" / "0007_donchian_breakout"
    assert d.is_dir()
    assert (d / "results").is_dir()
    assert (d / "hypothesis.md").is_file()
    assert (d / "donchian_breakout.ipynb").is_file()

    nb = nbformat.read(d / "donchian_breakout.ipynb", as_version=4)
    nbformat.validate(nb)

    with pytest.raises(FileExistsError):
        systems.create_system("FBS", 7, "donchian_breakout", name="Donchian Breakout",
                              card_markdown=_CARD, systems_dir=tmp_path)


def test_create_system_writes_front_matter_and_preserves_card_verbatim(tmp_path):
    d = systems.create_system("FBS", 7, "donchian_breakout", name="Donchian Breakout",
                              card_markdown=_CARD, systems_dir=tmp_path)
    text = (d / "hypothesis.md").read_text()
    assert text.startswith("---\n")
    assert _CARD.strip() in text
    assert "## Donchian Breakout Probe" in text


def test_create_system_rejects_bad_issue_or_slug(tmp_path):
    with pytest.raises(ValueError, match="issue"):
        systems.create_system("FBS", -1, "ok_slug", name="x", card_markdown=_card(),
                              systems_dir=tmp_path)
    with pytest.raises(ValueError, match="slug"):
        systems.create_system("FBS", 1, "Not-A-Slug", name="x", card_markdown=_card(),
                              systems_dir=tmp_path)


def test_create_system_requires_a_risk_semantics_declaration(tmp_path):
    with pytest.raises(ValueError, match="risk"):
        systems.create_system("FBS", 8, "no_risk_type", name="x", card_markdown=_card(risk_line=None),
                              systems_dir=tmp_path)


@pytest.mark.parametrize("letter", ["A", "B", "C", "D"])
def test_create_system_parses_each_risk_type(tmp_path, letter):
    d = systems.create_system(
        "FBS", 1, f"probe_{letter.lower()}", name="x",
        card_markdown=_card(f"**Risk semantics:** type {letter} + details."),
        systems_dir=tmp_path,
    )
    card = systems.read_card(d / "hypothesis.md")
    assert card["risk_type"] == letter


def test_read_card_round_trip(tmp_path):
    d = systems.create_system("B3", 12, "wdo_probe", name="WDO Probe", card_markdown=_CARD,
                              systems_dir=tmp_path)
    card = systems.read_card(d / "hypothesis.md")

    assert card["issue"] == 12
    assert card["book"] == "B3"
    assert card["slug"] == "wdo_probe"
    assert card["name"] == "WDO Probe"
    assert card["status"] == "testing"
    assert card["risk_type"] == "A"
    assert card["date"]  # ISO date string
    assert "## Donchian Breakout Probe" in card["body"]
    assert card["path"] == str(d / "hypothesis.md")


def test_read_card_requires_front_matter(tmp_path):
    p = tmp_path / "no_front_matter.md"
    p.write_text("# just a card, no YAML header\n")
    with pytest.raises(ValueError, match="front matter"):
        systems.read_card(p)


def test_system_dir_shape(tmp_path):
    d = systems.system_dir("FBS", 3, "some_system", systems_dir=tmp_path)
    assert d == tmp_path / "FBS" / "0003_some_system"


def test_find_system_locates_and_disambiguates(tmp_path):
    systems.create_system("FBS", 1, "unique_slug", name="x", card_markdown=_card(), systems_dir=tmp_path)
    found = systems.find_system("unique_slug", systems_dir=tmp_path)
    assert found == tmp_path / "FBS" / "0001_unique_slug"

    with pytest.raises(FileNotFoundError):
        systems.find_system("nope_not_created", systems_dir=tmp_path)

    systems.create_system("B3", 2, "unique_slug", name="x", card_markdown=_card(), systems_dir=tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        systems.find_system("unique_slug", systems_dir=tmp_path)


# =============================================================================================
# the notebook template: valid nbformat, has every heading, and is exactly what the builder
# function produces (so a hand-edit to the checked-in .ipynb would fail review, DESIGN §10).
# =============================================================================================
_EXPECTED_HEADINGS = [f"## {i}." for i in range(9)]  # "## 0." .. "## 8."


def test_template_notebook_is_valid_nbformat_with_every_heading():
    nb = nbformat.read(systems.TEMPLATE_PATH, as_version=4)
    nbformat.validate(nb)
    markdown_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for heading in _EXPECTED_HEADINGS:
        assert heading in markdown_text, f"missing heading {heading!r}"


def test_template_notebook_on_disk_matches_the_builder_function():
    built = systems.build_template_notebook()
    on_disk = nbformat.read(systems.TEMPLATE_PATH, as_version=4)
    built_sources = [(c.cell_type, c.source) for c in built.cells]
    disk_sources = [(c.cell_type, c.source) for c in on_disk.cells]
    assert built_sources == disk_sources, (
        "research/systems/_template/template.ipynb is stale -- regenerate it with "
        "quantlab.systems.write_template_notebook() after editing _system_cells()"
    )


def test_create_system_copies_the_template_verbatim(tmp_path):
    d = systems.create_system("FBS", 4, "template_copy_probe", name="x", card_markdown=_card(),
                              systems_dir=tmp_path)
    copied = nbformat.read(d / "template_copy_probe.ipynb", as_version=4)
    template = nbformat.read(systems.TEMPLATE_PATH, as_version=4)
    assert [(c.cell_type, c.source) for c in copied.cells] == [(c.cell_type, c.source) for c in template.cells]


# =============================================================================================
# §0-§2 execute end to end (nbconvert's ExecutePreprocessor). Needs a real strategy module
# reachable by `quantlab.strategies.get_strategy` from inside a *separate kernel process* (a
# monkeypatched __path__ in this test process would not be visible there), so this writes one
# temporary file directly under the real quantlab/strategies/ package and removes it in `finally`.
# =============================================================================================
_PROBE_STRATEGY_SOURCE = '''"""Ephemeral probe strategy for tests/test_systems.py's notebook-execution test only."""
from dataclasses import dataclass

import polars as pl

from quantlab.contracts import Params, RiskType


@dataclass(frozen=True)
class ProbeParams(Params):
    period: int = 10


class Strategy:
    name = "{slug}"
    risk_type = RiskType.C
    params_cls = ProbeParams

    def __init__(self, params=None):
        self.params = params or ProbeParams()

    def signals(self, bars):
        high, low, close = pl.col("high"), pl.col("low"), pl.col("close")
        prev_close = close.shift(1)
        tr = pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())
        atr = tr.rolling_mean(self.params.period, min_samples=self.params.period)
        state = pl.when(close > close.shift(1)).then(1).otherwise(-1)
        return bars.select(
            pl.when(atr.is_null()).then(None)
            .when(state != state.shift(1)).then(state).otherwise(None)
            .cast(pl.Int8).alias("signal"),
            (atr * 1.5).alias("stop_dist"),
            pl.lit(None, dtype=pl.Float64).alias("target_dist"),
        )
'''


@pytest.fixture
def real_probe_strategy_module():
    """Writes (and guarantees removal of) a tiny real strategy module under the actual
    quantlab/strategies/ package -- see the module-level comment above."""
    slug = f"probe_{uuid.uuid4().hex[:8]}"
    module_path = config.ROOT / "quantlab" / "strategies" / f"{slug}.py"
    module_path.write_text(_PROBE_STRATEGY_SOURCE.format(slug=slug))
    try:
        yield slug
    finally:
        module_path.unlink(missing_ok=True)
        pycache = module_path.parent / "__pycache__"
        if pycache.is_dir():
            for f in pycache.glob(f"{slug}.*.pyc"):
                f.unlink(missing_ok=True)


def _kernel_available(name: str = "python3") -> str | None:
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        KernelSpecManager().get_kernel_spec(name)
    except Exception as exc:  # pragma: no cover - environment-dependent
        return str(exc)
    return None


def test_notebook_sections_0_to_2_execute_end_to_end(tmp_path, monkeypatch, real_probe_strategy_module):
    pytest.importorskip("ipykernel")
    reason = _kernel_available("python3")
    if reason is not None:
        pytest.skip(f"no 'python3' Jupyter kernel available: {reason}")
    from nbconvert.preprocessors import ExecutePreprocessor

    slug = real_probe_strategy_module
    d = systems.create_system(
        "FBS", 9001, slug, name="Notebook Probe",
        card_markdown=_card("**Risk semantics:** type C (no hard stop)."),
        systems_dir=tmp_path,
    )
    nb = nbformat.read(d / f"{slug}.ipynb", as_version=4)

    # Kernel subprocesses inherit os.environ; PYTHONPATH makes `import quantlab` resolve from a
    # cwd (the system folder, set via resources below) that isn't the repo root.
    monkeypatch.setenv("PYTHONPATH", str(config.ROOT))
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    ep.preprocess(nb, resources={"metadata": {"path": str(d)}})

    errors = [
        (i, out.get("ename"), out.get("evalue"))
        for i, cell in enumerate(nb.cells)
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]
    assert not errors, errors

    stream_text = "".join(
        out.get("text", "") for cell in nb.cells for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    )
    assert "look-ahead audit clean" in stream_text
    assert "RiskType.C" in stream_text


def test_create_system_reads_symbols_and_timeframe_from_the_card(tmp_path):
    from quantlab import systems
    card = ("# Toy\n**Book:** FBS\n**Risk semantics:** type A + ATR stop\n"
            "**Symbols:** EURUSD, GBPUSD   **Timeframe:** H4\n")
    d = systems.create_system("FBS", 7, "toy_market", name="Toy", card_markdown=card, systems_dir=tmp_path)
    front = systems.read_card(d / "hypothesis.md")
    assert front["symbols"] == ["EURUSD", "GBPUSD"] and front["symbol"] == "EURUSD" and front["timeframe"] == "H4"
    placeholder = card.replace("EURUSD, GBPUSD", "…").replace("H4", "…")
    d2 = systems.create_system("FBS", 8, "toy_placeholder", name="Toy", card_markdown=placeholder, systems_dir=tmp_path)
    front2 = systems.read_card(d2 / "hypothesis.md")
    assert "symbol" not in front2 and "timeframe" not in front2
