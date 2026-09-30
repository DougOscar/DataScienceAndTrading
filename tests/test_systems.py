"""Tests for quantlab.systems (system-folder scaffolding, the notebook template, and the
QUANTLAB_*_DIR config overrides those two lean on -- DESIGN §1/§3/§10 Phase 2)."""

from __future__ import annotations

import json
import os
import shutil
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


def test_notebook_sections_0_to_2_execute_end_to_end(monkeypatch, real_probe_strategy_module):
    pytest.importorskip("ipykernel")
    reason = _kernel_available("python3")
    if reason is not None:
        pytest.skip(f"no 'python3' Jupyter kernel available: {reason}")
    from nbconvert.preprocessors import ExecutePreprocessor

    slug = real_probe_strategy_module
    # #33: §0's sys.path fix walks UP from the notebook's own folder looking for `quantlab/` --
    # that only finds anything when the folder is actually nested under the repo root, the same
    # as every real (or sandboxed, e.g. research/dryrun/systems/) system folder is. An arbitrary
    # tmp_path used to work here only because the test set PYTHONPATH by hand; now that the
    # notebook finds its own root, this test has to put the folder somewhere that root exists --
    # a gitignored scratch folder inside the repo (never the real research/systems/, so a crash
    # mid-test can't leave a fake system behind).
    import tempfile
    scratch_root = config.ROOT / ".pytest_scratch"
    scratch_root.mkdir(exist_ok=True)
    sdir = Path(tempfile.mkdtemp(prefix="systems_", dir=scratch_root))
    d = systems.system_dir("FBS", 9001, slug, systems_dir=sdir)
    try:
        systems.create_system(
            "FBS", 9001, slug, name="Notebook Probe",
            card_markdown=_card("**Risk semantics:** type C (no hard stop)."), systems_dir=sdir,
        )
        nb = nbformat.read(d / f"{slug}.ipynb", as_version=4)

        # No PYTHONPATH, and no editable install either -- the kernel subprocess's only way to
        # `import quantlab` is §0's own sys.path walk-up. Explicitly scrub any PYTHONPATH this
        # test process inherited, so a passing run proves the notebook's own fix, not the
        # environment's.
        monkeypatch.delenv("PYTHONPATH", raising=False)
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
    finally:
        shutil.rmtree(sdir, ignore_errors=True)


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


@pytest.mark.parametrize("line", ["**Risk semantics:** type **A**. Fixed-fraction 1 % risk",
                                  "**Risk semantics:** type `C` (no stop)", "**Risk semantics:** type B"])
def test_risk_type_tolerates_markdown_emphasis(tmp_path, line):
    """Dry run #23: the scout wrote 'type **A**' and create_system could not parse it."""
    from quantlab import systems
    d = systems.create_system("FBS", 9, "toy_emph", name="T", card_markdown=f"# T\n{line}\n", systems_dir=tmp_path)
    assert systems.read_card(d / "hypothesis.md")["risk_type"] == line.split("type")[1].strip(" *`.")[0]


# =============================================================================================
# set_front_matter / find_study_id / log_baseline / card_space_corners (#28, #35, #31)
# =============================================================================================
def test_set_front_matter_merges_and_preserves_body(tmp_path):
    d = systems.create_system("FBS", 5, "front_matter_probe", name="x", card_markdown=_card(),
                              systems_dir=tmp_path)
    path = d / "hypothesis.md"
    before_body = systems.read_card(path)["body"]

    systems.set_front_matter(path, study_id="fbs-0005-a1", extra=1)
    card = systems.read_card(path)
    assert card["study_id"] == "fbs-0005-a1"
    assert card["extra"] == 1
    assert card["risk_type"] == "A"          # untouched existing keys survive
    assert card["body"] == before_body       # card body preserved verbatim

    systems.set_front_matter(path, study_id="fbs-0005-a2")   # overwrite, no duplicate keys
    assert systems.read_card(path)["study_id"] == "fbs-0005-a2"


def test_set_front_matter_requires_front_matter(tmp_path):
    p = tmp_path / "no_front_matter.md"
    p.write_text("# no header\n")
    with pytest.raises(ValueError, match="front matter"):
        systems.set_front_matter(p, study_id="x")


def _create_study(ledger_dir, **over):
    from quantlab import ledger
    fields = dict(
        study_id="fbs-0099-a1", book="FBS", system="probe_baseline", issue=99, attempt=1,
        dev_window=["2016-05-02", "2020-01-01"], cost_model_version="test-v0", cv_scheme="CPCV(1,1)",
        git_commit="deadbeef", data_manifest_sha="deadbeef",
    )
    fields.update(over)
    return ledger.create_study(ledger_dir=ledger_dir, **fields)


def test_find_study_id_matches_book_system_issue(tmp_path):
    ledger_dir = tmp_path / "ledger"
    assert systems.find_study_id("FBS", 99, "probe_baseline", ledger_dir=ledger_dir) is None
    _create_study(ledger_dir)
    assert systems.find_study_id("FBS", 99, "probe_baseline", ledger_dir=ledger_dir) == "fbs-0099-a1"
    # a different issue / a different (normalised) system does not match
    assert systems.find_study_id("FBS", 100, "probe_baseline", ledger_dir=ledger_dir) is None
    assert systems.find_study_id("FBS", 99, "not_this_system", ledger_dir=ledger_dir) is None
    # the highest attempt wins (DESIGN §4.5: a later attempt supersedes an earlier, killed one)
    _create_study(ledger_dir, study_id="fbs-0099-a2", attempt=2)
    assert systems.find_study_id("FBS", 99, "probe_baseline", ledger_dir=ledger_dir) == "fbs-0099-a2"


def test_log_baseline_is_a_noop_without_a_study_and_idempotent_with_one(tmp_path):
    from quantlab import ledger
    ledger_dir = tmp_path / "ledger"

    # no study yet -- not a study row, nothing written (#28)
    out = systems.log_baseline("FBS", 99, "probe_baseline", params={"lookback": 60},
                               edge_breakdown={"gross_points_per_trade": 1.0}, cost_model_version="v0",
                               ledger_dir=ledger_dir)
    assert out is None
    assert not (ledger_dir / "studies.jsonl").exists()

    _create_study(ledger_dir)
    logged = systems.log_baseline("FBS", 99, "probe_baseline", params={"lookback": 60},
                                  edge_breakdown={"gross_points_per_trade": 1.0}, cost_model_version="v0",
                                  ledger_dir=ledger_dir)
    assert logged is not None
    assert logged["event"] == "baseline" and logged["study_id"] == "fbs-0099-a1"
    assert logged["params"] == {"lookback": 60}
    assert logged["cost_model_version"] == "v0"

    events = ledger.study_events("fbs-0099-a1", ledger_dir=ledger_dir)
    assert [e["event"] for e in events] == ["study_created", "baseline"]

    # idempotent: a second call never logs a second baseline event
    again = systems.log_baseline("FBS", 99, "probe_baseline", params={"lookback": 60},
                                 edge_breakdown={}, cost_model_version="v0", ledger_dir=ledger_dir)
    assert again is None
    assert len(ledger.study_events("fbs-0099-a1", ledger_dir=ledger_dir)) == 2


_CARD_WITH_SPACE = '''## Probe
**Free parameters (pre-registered; reviewed by the red team at S2):** 2 parameters.
| name | type | range [lo, hi] (or levels) | plateau scale |
|---|---|---|---|
| lookback | int | [10, 120], grid step 10 (12 levels) | relative (r = 0.20) |
| stop_mult | float | [1.0, 4.0], grid step 0.5 (7 levels) | relative (r = 0.20) |

Some prose after the table.
'''


def test_card_space_corners_parses_the_free_parameters_table():
    corners = systems.card_space_corners({"body": _CARD_WITH_SPACE})
    assert len(corners) == 4   # 2**2 parameters
    assert {"lookback": 10, "stop_mult": 1.0} in corners
    assert {"lookback": 120, "stop_mult": 4.0} in corners
    assert all(isinstance(c["lookback"], int) for c in corners)
    assert all(isinstance(c["stop_mult"], float) for c in corners)
    # a raw markdown string works too, not just a read_card()-shaped dict
    assert systems.card_space_corners(_CARD_WITH_SPACE) == corners


def test_card_space_corners_is_empty_without_a_parseable_table():
    assert systems.card_space_corners({"body": "no free parameters table here"}) == []
    assert systems.card_space_corners({"body": ""}) == []
    categorical = '''**Free parameters:**
| name | type | range [lo, hi] (or levels) |
|---|---|---|
| mode | categorical | fast, slow |
'''
    assert systems.card_space_corners({"body": categorical}) == []
