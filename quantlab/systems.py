"""Scaffolding for one system's research folder (DESIGN §1, §3, §10 Phase 2).

``research/systems/<book>/<issue#>_<slug>/`` holds everything one hypothesis produces:
``hypothesis.md`` (the approved card, verbatim, plus a small YAML front-matter header),
``<slug>.ipynb`` (a copy of the generic template — :func:`build_template_notebook`), and a
``results/`` folder for figures and ``metrics.json``. This module only creates and locates that
folder; it never writes strategy logic (that's ``quantlab/strategies/<slug>.py``, S2) or runs
any stage of the pipeline.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from . import config

__all__ = [
    "create_system", "system_dir", "find_system", "read_card",
    "build_template_notebook", "write_template_notebook", "TEMPLATE_PATH",
]

# The template is a fixed, checked-in repo asset -- always read from the real research tree,
# even when QUANTLAB_SYSTEMS_DIR points a sandbox's *output* somewhere else (config._env_path).
TEMPLATE_DIR = config.RESEARCH_DIR / "systems" / "_template"
TEMPLATE_PATH = TEMPLATE_DIR / "template.ipynb"

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# DESIGN §5 risk types A-D, declared on the card as e.g. "**Risk semantics:** type A + ..."
# (quant-scout's card template, .claude/agents/quant-scout.md). Matches the first "type <letter>"
# token on the same line as "risk semantics" (case-insensitive); does not span lines, so a
# multi-line reformat of the card needs the declaration back on one line.
_RISK_TYPE_RE = re.compile(r"risk\s+semantics[^\n]*?\btype\s*[:\-]?\s*([ABCD])\b", re.IGNORECASE)


# "**Symbols:** EURUSD, GBPUSD   **Timeframe:** H1" (quant-scout card template). Placeholders ("…")
# are ignored, so an incomplete card leaves the fields out and §3+ of the notebook waits for them.
_SYMBOLS_RE = re.compile(r"\*\*Symbols?:\*\*\s*([^*\n]+)", re.IGNORECASE)
_TIMEFRAME_RE = re.compile(r"\*\*Timeframe:\*\*\s*([A-Za-z]+\d+)\b", re.IGNORECASE)
_SYMBOL_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._]*$")


def _extract_market(card_markdown: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = _SYMBOLS_RE.search(card_markdown)
    if m:
        syms = [t.strip() for t in re.split(r"[,;/\s]+", m.group(1)) if t.strip()]
        syms = [t for t in syms if _SYMBOL_TOKEN_RE.match(t)]
        if syms:
            out["symbols"], out["symbol"] = syms, syms[0]
    m = _TIMEFRAME_RE.search(card_markdown)
    if m:
        out["timeframe"] = m.group(1).upper()
    return out


def _check_issue(issue: Any) -> int:
    if isinstance(issue, bool) or not isinstance(issue, int) or issue < 0:
        raise ValueError(f"issue must be a non-negative int (the hypothesis issue number), got {issue!r}")
    return int(issue)


def _check_slug(slug: str) -> str:
    if not _SLUG_RE.match(slug):
        raise ValueError(
            f"slug {slug!r} is invalid: must match {_SLUG_RE.pattern!r} (lowercase letters, digits, "
            "underscores, starting with a letter) -- it becomes both a directory name component and "
            "a Python module name, quantlab.strategies.<slug>"
        )
    return slug


def _extract_risk_type(card_markdown: str, *, slug: str) -> str:
    # markdown emphasis around the letter ("type **A**", "type `A`") is common in real cards
    m = _RISK_TYPE_RE.search(re.sub(r"[*`]", "", card_markdown))
    if not m:
        raise ValueError(
            f"create_system({slug!r}): could not find a risk-semantics declaration in card_markdown "
            "(expected a line such as '**Risk semantics:** type A ...', DESIGN §5 / the quant-scout "
            "card template) -- every hypothesis card must declare its risk type"
        )
    return m.group(1).upper()


# --------------------------------------------------------------------------- lookups
def system_dir(book: str, issue: int, slug: str, *, systems_dir: Path | None = None) -> Path:
    """``<systems_dir or config.SYSTEMS_DIR>/<book>/<issue:04d>_<slug>`` (read at call time)."""
    b = config.get_book(book)
    issue = _check_issue(issue)
    slug = _check_slug(slug)
    base = Path(systems_dir) if systems_dir is not None else config.SYSTEMS_DIR
    return base / b.name / f"{issue:04d}_{slug}"


def find_system(slug: str, *, systems_dir: Path | None = None) -> Path:
    """The system folder for ``slug``, searched across every book under ``systems_dir``.

    Raises ``FileNotFoundError`` if none exists, ``ValueError`` if more than one does (an issue
    number was reused for the same slug under two different books -- disambiguate with
    :func:`system_dir`)."""
    slug = _check_slug(slug)
    base = Path(systems_dir) if systems_dir is not None else config.SYSTEMS_DIR
    matches = sorted(p for p in base.glob(f"*/????_{slug}") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"no system folder for slug {slug!r} under {base}")
    if len(matches) > 1:
        raise ValueError(f"slug {slug!r} is ambiguous under {base}: {matches}; use system_dir(book, issue, slug)")
    return matches[0]


# --------------------------------------------------------------------------- hypothesis.md
_FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)


def read_card(path: str | Path) -> dict[str, Any]:
    """Parse ``hypothesis.md``'s YAML front matter (written by :func:`create_system`).

    Returns the front-matter mapping (``issue``, ``book``, ``slug``, ``name``, ``date``,
    ``status``, ``risk_type``, plus anything else a later stage adds to it -- e.g. ``symbol``/
    ``timeframe`` once strategy-engineer picks them at S3) with two extra keys: ``body`` (the
    card markdown after the front matter, stripped of leading/trailing blank lines) and
    ``path`` (the file read, as a string).
    """
    p = Path(path)
    text = p.read_text()
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        raise ValueError(f"{p}: no YAML front matter found (expected '---\\n...\\n---' at the top of the file)")
    front = yaml.safe_load(m.group(1)) or {}
    if not isinstance(front, dict):
        raise ValueError(f"{p}: front matter must be a YAML mapping, got {type(front).__name__}")
    body = text[m.end():]
    return {**front, "body": body.strip("\n"), "path": str(p)}


def create_system(book: str, issue: int, slug: str, *, name: str, card_markdown: str,
                  systems_dir: Path | None = None) -> Path:
    """Create ``research/systems/<book>/<issue:04d>_<slug>/`` (checkpoint A: card approved).

    Writes ``hypothesis.md`` (the header below, then ``card_markdown`` verbatim), an empty
    ``results/`` folder, and ``<slug>.ipynb`` (a copy of :data:`TEMPLATE_PATH`, generated by
    :func:`build_template_notebook`). Refuses to overwrite an existing folder (``FileExistsError``).

    The YAML front-matter header written to ``hypothesis.md``::

        issue: <int>
        book: <FBS|B3>
        slug: <slug>
        name: <name>
        date: <today, ISO>
        status: testing
        risk_type: <A|B|C|D>          # parsed out of card_markdown, DESIGN §5

    ``risk_type`` is parsed from a "**Risk semantics:** type X ..." line in ``card_markdown``
    (the quant-scout card template) -- every hypothesis card must declare it, so a card missing
    the line raises rather than silently recording an unknown type.
    """
    import nbformat

    b = config.get_book(book)
    issue = _check_issue(issue)
    slug = _check_slug(slug)
    risk_type = _extract_risk_type(card_markdown, slug=slug)

    d = system_dir(b.name, issue, slug, systems_dir=systems_dir)
    if d.exists():
        raise FileExistsError(f"{d} already exists; create_system refuses to overwrite a system folder")

    front = {
        "issue": issue, "book": b.name, "slug": slug, "name": name,
        "date": date.today().isoformat(), "status": "testing", "risk_type": risk_type,
        **_extract_market(card_markdown),
    }
    front_yaml = yaml.safe_dump(front, sort_keys=False)
    hypothesis_text = f"---\n{front_yaml}---\n\n{card_markdown}"
    if not hypothesis_text.endswith("\n"):
        hypothesis_text += "\n"

    d.mkdir(parents=True)
    (d / "results").mkdir()
    (d / "hypothesis.md").write_text(hypothesis_text)

    if TEMPLATE_PATH.exists():
        nb = nbformat.read(TEMPLATE_PATH, as_version=4)
    else:
        nb = build_template_notebook()
    nbformat.write(nb, d / f"{slug}.ipynb")
    return d


# --------------------------------------------------------------------------- notebook template
def _system_cells() -> list[dict[str, Any]]:
    """The §0-§8 cells shared by every system notebook. A pure function of nothing: the
    template is completely generic (every cell reads the ``SYSTEM``/``CARD`` dicts built in §0
    from that system's own ``hypothesis.md``), so it is safe to copy byte-for-byte into each new
    system folder (:func:`create_system`) rather than re-rendered per system.
    """
    from nbformat.v4 import new_code_cell, new_markdown_cell

    setup_code = '''
from pathlib import Path

from quantlab import config
from quantlab.systems import read_card

# Jupyter's default working directory is the notebook's own folder; nbconvert callers should
# pass resources={"metadata": {"path": <this folder>}} to ExecutePreprocessor if that isn't
# already true. hypothesis.md lives right next to this notebook.
SYSTEM_DIR = Path.cwd()
CARD = read_card(SYSTEM_DIR / "hypothesis.md")
SYSTEM = {
    "book": CARD["book"],
    "issue": CARD["issue"],
    "slug": CARD["slug"],
    "name": CARD.get("name", CARD["slug"]),
    "risk_type": CARD.get("risk_type"),
    "status": CARD.get("status", "testing"),
    # Set by strategy-engineer once the instrument/timeframe is picked (S3): add `symbol:` /
    # `timeframe:` (and optionally `start:` / `end:`) to hypothesis.md's front matter.
    "symbol": CARD.get("symbol"),
    "timeframe": CARD.get("timeframe"),
    "start": CARD.get("start"),
    "end": CARD.get("end"),
}
LEDGER_DIR = config.LEDGER_DIR
STUDIES_DIR = config.STUDIES_DIR
RESULTS_DIR = SYSTEM_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SYSTEM
'''.strip("\n")

    hypothesis_code = '''
from IPython.display import Markdown, display

display(Markdown(f"**Status:** {SYSTEM['status']}  ·  **Risk type:** {SYSTEM['risk_type']}  ·  "
                 f"**Issue:** #{SYSTEM['issue']}"))
display(Markdown(CARD.get("body", "")))
'''.strip("\n")

    implementation_code = '''
import inspect

from quantlab.costs import InstrumentSpec
from quantlab.strategies import get_strategy
from quantlab.testing import (
    assert_engine_causal,
    assert_no_lookahead,
    assert_strategy_source_clean,
    synthetic_bars,
)

StrategyCls = get_strategy(SYSTEM["slug"])
assert_strategy_source_clean(inspect.getfile(StrategyCls))

# Same class/Params resolution order as quantlab.evaluators.RuleEvaluator.build_strategy: a
# strategy is built from its declared prior (default) parameters here, so this audit exercises
# the real class with the card's own prior values, not an arbitrary stand-in.
_params_cls = getattr(StrategyCls, "params_cls", None) or getattr(StrategyCls, "Params", None)
_factory = (lambda: StrategyCls(_params_cls())) if _params_cls is not None else StrategyCls

# Generic, book/timeframe-agnostic bars/spec for the audit -- the real per-system test module
# (tests/, written alongside this notebook at S2) additionally runs assert_engine_causal on real
# dev data with m1=/timeframe= for the system's actual instrument.
_audit_bars = synthetic_bars(600, seed=0, timeframe="H1")
assert_no_lookahead(_factory, _audit_bars)

_audit_spec = InstrumentSpec(
    symbol="AUDIT", digits=5, point=1e-5, contract_size=100_000.0, tick_size=1e-5,
    volume_min=0.01, volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD",
    swap_mode="points", swap_long=0.0, swap_short=0.0, swap_3day=2,
    commission_per_lot_rt=0.0, stops_level=0.0, calibrated=True,
)
assert_engine_causal(_factory(), _audit_bars, _audit_spec, n_checks=8, seed=1, min_history=60)

print(f"{StrategyCls.name}: look-ahead audit clean (risk_type={StrategyCls.risk_type})")
'''.strip("\n")

    baseline_code = '''
if SYSTEM.get("symbol") and SYSTEM.get("timeframe"):
    from quantlab.evaluators import RuleEvaluator

    evaluator = RuleEvaluator(
        StrategyCls, symbol=SYSTEM["symbol"], timeframe=SYSTEM["timeframe"], book=SYSTEM["book"],
        start=SYSTEM.get("start"), end=SYSTEM.get("end"),
    )
    baseline = evaluator({})   # {} = the card's prior (default) parameters, costs on
    print(evaluator.describe())
    baseline.metrics
else:
    evaluator = None
    baseline = None
    print("S3 pending: add `symbol:` / `timeframe:` to hypothesis.md's front matter, then re-run.")
'''.strip("\n")

    optimisation_code = '''
if evaluator is not None:
    from quantlab import opt

    # S4 (optimization-architect): declare the real search space here -- every numeric Param
    # needs plateau_scale="relative" or plateau_step=... (DESIGN §4.2, pre-registered on the
    # card) -- then run_study(...). Left unset in the template so it never launches a study with
    # placeholder bounds.
    space = None   # opt.SearchSpace((opt.IntParam(...), ...))
    study = None
    if space is not None:
        study = opt.run_study(
            evaluator, space, book=SYSTEM["book"], system=SYSTEM["slug"], issue=SYSTEM["issue"], attempt=1,
        )
else:
    space = None
    study = None
    print("S4 pending: needs a baseline (S3) first.")
'''.strip("\n")

    validation_code = '''
if study is not None:
    import polars as pl

    from quantlab import gates

    gate_report = gates.evaluate_gates(study, evaluator, periods_per_year=evaluator.periods_per_year, log=True)
    gate_table = pl.DataFrame([
        {"gate": r.gate, "value": r.display if r.display is not None else r.value,
         "threshold": r.threshold, "status": r.status, "interpretation": r.interpretation}
        for r in gate_report.rows
    ])
    gate_table
else:
    gate_report = None
    gate_table = None
    print("S5 pending: needs an optimisation study (S4) first.")
'''.strip("\n")

    redteam_code = '''
FINDINGS_PATH = RESULTS_DIR / "redteam_findings.md"
if FINDINGS_PATH.exists():
    from IPython.display import Markdown, display

    display(Markdown(FINDINGS_PATH.read_text()))
else:
    print(f"S6 pending: no findings file yet at {FINDINGS_PATH}")
'''.strip("\n")

    report_code = '''
if study is not None and gate_report is not None:
    from quantlab import report

    ts = report.tear_sheet(study.study_id, evaluator=evaluator, risk_type=SYSTEM.get("risk_type"))
    ts.write_results(RESULTS_DIR)
    report.write_card(
        ts, slug=SYSTEM["slug"], name=SYSTEM["name"], idea=CARD.get("idea", ""),
        status=SYSTEM.get("status", "testing"), issue=SYSTEM["issue"], book=SYSTEM["book"],
    )
    ts.headline
else:
    ts = None
    print("S7 pending: needs validation (S5) first.")
'''.strip("\n")

    holdout_code = '''
# Status only. The holdout is unlocked once, by the user, via `/unlock-holdout <system>` -- never
# from this notebook. This cell never calls quantlab.data with include_holdout=True.
print(f"S8 holdout: system status = {SYSTEM.get('status')!r} (book {SYSTEM['book']}). "
     "Run `/unlock-holdout` yourself when checkpoint B is reached (DESIGN §3, §4.4).")
'''.strip("\n")

    sections = [
        ("markdown", "# System research notebook\n\nCopied verbatim from "
         "`research/systems/_template/template.ipynb` by `quantlab.systems.create_system` "
         "(DESIGN §1/§3/§10 Phase 2). Every stage cell below is a thin wrapper over `quantlab`; "
         "heavy logic lives in the library, not here. All of it reads the `SYSTEM`/`CARD` dicts "
         "built in §0 from this folder's own `hypothesis.md` -- nothing below is specific to one "
         "system, including this heading; the system's name/slug is shown once §0 runs."),
        ("markdown", "## 0. Setup"), ("code", setup_code),
        ("markdown", "## 1. Hypothesis"), ("code", hypothesis_code),
        ("markdown", "## 2. Implementation"), ("code", implementation_code),
        ("markdown", "## 3. Baseline"), ("code", baseline_code),
        ("markdown", "## 4. Optimisation"), ("code", optimisation_code),
        ("markdown", "## 5. Validation"), ("code", validation_code),
        ("markdown", "## 6. Red-team findings"), ("code", redteam_code),
        ("markdown", "## 7. Report"), ("code", report_code),
        ("markdown", "## 8. Holdout"), ("code", holdout_code),
    ]
    cells = []
    for kind, text in sections:
        cells.append(new_markdown_cell(text) if kind == "markdown" else new_code_cell(text))
    return cells


def build_template_notebook() -> Any:
    """Build the generic system notebook (nbformat v4 ``NotebookNode``) -- see :func:`_system_cells`.

    Written out to :data:`TEMPLATE_PATH` by :func:`write_template_notebook`; kept as a builder
    function (rather than a hand-edited ``.ipynb``) so it goes through ordinary code review, per
    DESIGN §10 Phase 2."""
    from nbformat.v4 import new_notebook

    return new_notebook(
        cells=_system_cells(),
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )


def write_template_notebook(path: Path | None = None) -> Path:
    """Write :func:`build_template_notebook` to ``path`` (default :data:`TEMPLATE_PATH`)."""
    import nbformat

    p = Path(path) if path is not None else TEMPLATE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_template_notebook(), p)
    return p
