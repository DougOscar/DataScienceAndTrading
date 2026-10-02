"""Scaffolding for one system's research folder (DESIGN §1, §3, §10 Phase 2).

``research/systems/<book>/<issue#>_<slug>/`` holds everything one hypothesis produces:
``hypothesis.md`` (the approved card, verbatim, plus a small YAML front-matter header),
``<slug>.ipynb`` (a copy of the generic template — :func:`build_template_notebook`), and a
``results/`` folder for figures and ``metrics.json``. This module only creates and locates that
folder; it never writes strategy logic (that's ``quantlab/strategies/<slug>.py``, S2) or runs
any stage of the pipeline.
"""

from __future__ import annotations

import itertools
import re
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from . import config

__all__ = [
    "create_system", "system_dir", "find_system", "read_card", "set_front_matter",
    "find_study_id", "log_baseline", "card_space_corners",
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


def set_front_matter(path: str | Path, **fields: Any) -> Path:
    """Merge ``fields`` into ``hypothesis.md``'s YAML front matter in place (#35).

    Used by the notebook template's S4 cell to record the ``study_id`` the first time
    ``opt.run_study`` (or a ledger lookup, :func:`find_study_id`) resolves one, so a later stage
    — or a later "Run all", possibly a fresh kernel — can read it straight from :func:`read_card`
    instead of re-scanning the ledger. The card body (everything after the front matter) is
    preserved byte-for-byte; existing front-matter keys not in ``fields`` are kept.
    """
    p = Path(path)
    text = p.read_text()
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        raise ValueError(f"{p}: no YAML front matter found (expected '---\\n...\\n---' at the top of the file)")
    front = yaml.safe_load(m.group(1)) or {}
    if not isinstance(front, dict):
        raise ValueError(f"{p}: front matter must be a YAML mapping, got {type(front).__name__}")
    front.update(fields)
    front_yaml = yaml.safe_dump(front, sort_keys=False)
    p.write_text(f"---\n{front_yaml}---\n" + text[m.end():])
    return p


def find_study_id(book: str, issue: int, slug: str, *, ledger_dir: Path | None = None) -> str | None:
    """The study already in the ledger for this system (book + issue + slug), or ``None``.

    Lets a stage section find "does the ledger already have a study for this system" without a
    ``study_id`` recorded in ``hypothesis.md``'s front matter yet (:func:`set_front_matter` only
    records one *after* the first successful lookup/``run_study`` — a fresh clone, or a kernel
    that never ran S4 this session, has no other way to find it). Matches book +
    :func:`ledger.normalise_system` (the same identity ``ledger.system_prior_trials`` uses),
    narrowed to this ``issue`` (DESIGN's own identity key, §4.4) since a slug is occasionally
    reused across books/issues in tests. When more than one attempt exists (the DESIGN §4.5 kill
    rule: attempt 2 supersedes a killed attempt 1), the highest ``attempt`` wins.
    """
    from . import ledger
    b = config.get_book(book).name
    issue = _check_issue(issue)
    sysn = ledger.normalise_system(slug)
    best: tuple[int, str] | None = None
    for sid, state in ledger.studies(ledger_dir=ledger_dir).items():
        if state.get("book") != b or ledger.normalise_system(state.get("system")) != sysn:
            continue
        if state.get("issue") is not None and int(state["issue"]) != issue:
            continue
        att = int(state.get("attempt") or 0)
        if best is None or att > best[0]:
            best = (att, sid)
    return best[1] if best else None


def log_baseline(book: str, issue: int, slug: str, *, params: dict[str, Any], edge_breakdown: dict[str, Any],
                 cost_model_version: str, ledger_dir: Path | None = None) -> dict[str, Any] | None:
    """Log the S3 baseline (#28) to the ledger: params, the edge/cost breakdown
    (:func:`evaluators.edge_breakdown`) and the cost-model version.

    No optimisation study exists yet the first time S3 runs (DESIGN §3: S3 precedes S4), and
    ``ledger.log_event`` only appends to a study that already has a ``study_created`` row —
    creating one here (``ledger.create_study``) would misrepresent a baseline read as an
    optimisation study, complete with its own ``attempt``/``cv_scheme``/``dev_window``
    bookkeeping that a baseline has no use for. So this only ever attaches an ordinary event —
    never a new study row — to the system's study **once one exists** (:func:`find_study_id`):
    on a fresh system this is a no-op returning ``None`` (S3 is independently re-runnable, #35,
    so a later "Run all" — after S4 has created the study — logs it then). Idempotent: never logs
    a second ``baseline`` event for the same study.
    """
    from . import ledger
    study_id = find_study_id(book, issue, slug, ledger_dir=ledger_dir)
    if study_id is None:
        return None
    if ledger.study_events(study_id, "baseline", ledger_dir=ledger_dir):
        return None
    return ledger.log_event(
        study_id, "baseline", ledger_dir=ledger_dir, book=config.get_book(book).name, system=slug,
        issue=_check_issue(issue), params=dict(params), edge_breakdown=dict(edge_breakdown),
        cost_model_version=str(cost_model_version),
    )


# --------------------------------------------------------------------------- free-parameter grid corners
_FREE_PARAMS_ANCHOR_RE = re.compile(r"free\s+parameters", re.IGNORECASE)
_RANGE_BRACKET_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def card_space_corners(card: Any) -> list[dict[str, Any]]:
    """The 2^n corners of the card's pre-registered free-parameter grid (#31): every combination
    of each numeric parameter's declared low/high, read straight off the card's "Free parameters"
    markdown table — no ``opt.SearchSpace`` needed (that isn't built until S4), so the S2
    look-ahead audit can run at the grid's extremes, not just the prior, as soon as the card
    exists.

    Best-effort and silent: a card with no such table, or a row whose range cell isn't a
    ``"[lo, hi]"`` pair (a categorical, a "see levels" note, ...), contributes nothing to the
    result. Returns ``[]`` rather than raising, so a missing/malformed table degrades to
    "audit the prior only" (the pre-#31 behaviour), not a broken notebook.

    ``card``: a dict with a ``"body"`` key (:func:`read_card`'s return, or the ``CARD`` dict the
    notebook template builds from it) or a raw markdown string.
    """
    text = card.get("body", "") if isinstance(card, dict) else str(card)
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if _FREE_PARAMS_ANCHOR_RE.search(ln)), None)
    if start is None:
        return []

    rows: list[tuple[str, str, str]] = []
    in_table = False
    for ln in lines[start + 1:]:
        s = ln.strip()
        if not s.startswith("|"):
            if in_table:
                break
            continue
        in_table = True
        cells = [re.sub(r"[*`]", "", c).strip() for c in s.strip("|").split("|")]
        if len(cells) < 3 or cells[0].lower() == "name" or all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue   # header row, or a "|---|---|...|" separator row (with or without :-alignment)
        rows.append((cells[0], cells[1].lower(), cells[2]))

    params: dict[str, tuple[Any, Any]] = {}
    for name, kind, range_cell in rows:
        if kind not in ("int", "float") or not name:
            continue
        m = _RANGE_BRACKET_RE.search(range_cell)
        if not m:
            continue
        cast = int if kind == "int" else float
        try:
            params[name] = (cast(float(m.group(1))), cast(float(m.group(2))))
        except ValueError:
            continue

    if not params:
        return []
    names = list(params)
    return [dict(zip(names, combo)) for combo in itertools.product(*(params[n] for n in names))]


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
import sys
from pathlib import Path

# Jupyter's default working directory is the notebook's own folder; nbconvert callers should
# pass resources={"metadata": {"path": <this folder>}} to ExecutePreprocessor if that isn't
# already true. hypothesis.md lives right next to this notebook.
SYSTEM_DIR = Path.cwd()

# #33: `quantlab` is not pip-installed, and a fresh kernel subprocess (nbconvert/nbclient) has no
# PYTHONPATH of its own -- walk up from this folder to whichever ancestor directory holds the
# `quantlab` package (the repo root) and put that on sys.path, so the import below resolves
# whatever the kernel's cwd or install, with no environment setup required.
_root = SYSTEM_DIR
while not (_root / "quantlab" / "__init__.py").is_file():
    if _root.parent == _root:
        raise RuntimeError(f"could not find a 'quantlab' package in any parent of {SYSTEM_DIR}")
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from quantlab import config
from quantlab.systems import read_card

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
    # Set by S4 (quantlab.systems.set_front_matter) the first time it resolves this system's
    # study -- read here so a later stage / a later "Run all" doesn't have to re-scan the ledger.
    "study_id": CARD.get("study_id"),
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
from dataclasses import asdict

from quantlab.costs import InstrumentSpec
from quantlab.strategies import get_strategy
from quantlab.systems import card_space_corners
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

# #31: audit the prior AND every corner of the card's pre-registered grid, when it declares one
# parseable this way (opt.SearchSpace itself isn't built until S4) -- {} = the prior (no override).
_param_overrides = [{}]
if _params_cls is not None:
    _param_overrides += card_space_corners(CARD)

_audit_bars = synthetic_bars(600, seed=0, timeframe="H1")
_audit_spec = InstrumentSpec(
    symbol="AUDIT", digits=5, point=1e-5, contract_size=100_000.0, tick_size=1e-5,
    volume_min=0.01, volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD",
    swap_mode="points", swap_long=0.0, swap_short=0.0, swap_3day=2,
    commission_per_lot_rt=0.0, stops_level=0.0, calibrated=True,
)
for _override in _param_overrides:
    if _params_cls is not None:
        _factory = lambda _o=_override: StrategyCls(_params_cls(**{**asdict(_params_cls()), **_o}))
    else:
        _factory = StrategyCls
    assert_no_lookahead(_factory, _audit_bars)
    assert_engine_causal(_factory(), _audit_bars, _audit_spec, n_checks=8, seed=1, min_history=60)

print(f"{StrategyCls.name}: look-ahead audit clean at the prior and {len(_param_overrides) - 1} "
     f"grid corner(s) (risk_type={StrategyCls.risk_type})")
'''.strip("\n")

    baseline_code = '''
if SYSTEM.get("symbol") and SYSTEM.get("timeframe"):
    from quantlab.evaluators import RuleEvaluator, edge_breakdown
    from quantlab.systems import log_baseline

    evaluator = RuleEvaluator(
        StrategyCls, symbol=SYSTEM["symbol"], timeframe=SYSTEM["timeframe"], book=SYSTEM["book"],
        start=SYSTEM.get("start"), end=SYSTEM.get("end"),
    )
    baseline = evaluator({})   # {} = the card's prior (default) parameters, costs on
    print(evaluator.describe())
    print(baseline.metrics)                              # #24: explicit -- this is the `if`
                                                            # branch, so auto-display never fires
    edge = edge_breakdown(baseline, cost=evaluator.cost)   # #25: gross/net/spread/swap/slippage
    print(edge)                                            # per trade, trades/year, cost % gross

    # #28: an audit-trail event, not a new study row -- see quantlab.systems.log_baseline.
    _params_used = _params_cls().as_dict() if _params_cls is not None else {}
    _logged = log_baseline(
        SYSTEM["book"], SYSTEM["issue"], SYSTEM["slug"],
        params=_params_used, edge_breakdown=edge, cost_model_version=evaluator.cost_version,
    )
    if _logged is None:
        print("S3: no study in the ledger yet for this system -- baseline not logged to the "
             "ledger (re-run this cell after S4; #35, S3 is independently re-runnable), or a "
             "'baseline' event is already logged for it.")
    else:
        print(f"S3: baseline logged to the ledger under {_logged['study_id']}")
else:
    evaluator = None
    baseline = None
    edge = None
    print("S3 pending: add `symbol:` / `timeframe:` to hypothesis.md's front matter, then re-run.")
'''.strip("\n")

    optimisation_code = '''
if evaluator is not None:
    from quantlab import opt
    from quantlab.systems import find_study_id, set_front_matter

    # S4 (optimization-architect): declare the real search space here -- every numeric Param
    # needs plateau_scale="relative" or plateau_step=... (DESIGN §4.2, pre-registered on the
    # card) -- then run_study(...). Left unset in the template so it never launches a study with
    # placeholder bounds.
    space = None   # opt.SearchSpace((opt.IntParam(...), ...))

    # #35: a full "Run all" must not repeat a logged run_study -- load the ledger's own study
    # (found by id, else by a book/issue/slug ledger lookup) instead of re-running it blind.
    study_id = SYSTEM.get("study_id") or find_study_id(SYSTEM["book"], SYSTEM["issue"], SYSTEM["slug"])
    if study_id is not None:
        study = opt.load_study(study_id)
        print(f"S4: loaded existing study {study_id} read-only (run_study skipped, #35)")
    elif space is not None:
        study = opt.run_study(
            evaluator, space, book=SYSTEM["book"], system=SYSTEM["slug"], issue=SYSTEM["issue"], attempt=1,
        )
        study_id = study.study_id
        print(f"S4: ran a new study {study_id}")
    else:
        study = None
        print("S4 pending: no search space declared yet.")

    if study_id is not None and CARD.get("study_id") != study_id:
        set_front_matter(SYSTEM_DIR / "hypothesis.md", study_id=study_id)   # #35
        CARD["study_id"] = study_id
    SYSTEM["study_id"] = study_id
else:
    space = None
    study = None
    study_id = None
    print("S4 pending: needs a baseline (S3) first.")
'''.strip("\n")

    validation_code = '''
if evaluator is not None:
    import polars as pl

    from quantlab import gates, ledger, opt
    from quantlab.systems import find_study_id

    # Resolved independently of §4's own `study` variable (#35/#68: this cell must work even if
    # §4 did not run this kernel session -- only the ledger/trial-store, loaded by id, are trusted).
    study_id = SYSTEM.get("study_id") or find_study_id(SYSTEM["book"], SYSTEM["issue"], SYSTEM["slug"])
    study = opt.load_study(study_id) if study_id is not None else None

    if study is None:
        gate_report = None
        gate_table = None
        print("S5 pending: needs an optimisation study (S4) first.")
    elif ledger.study_events(study.study_id, "gates"):
        gate_report = None
        gate_table = None
        print(f"S5: {study.study_id} already has a logged gates event -- "
             "evaluate_gates(log=True) skipped (#35: a full 'Run all' must never log it twice).")
    else:
        gate_report = gates.evaluate_gates(study, evaluator, periods_per_year=evaluator.periods_per_year, log=True)
        gate_table = pl.DataFrame([
            {"gate": r.gate, "value": r.display if r.display is not None else r.value,
             "threshold": r.threshold, "status": r.status, "interpretation": r.interpretation}
            for r in gate_report.rows
        ])
    gate_table
else:
    study = None
    gate_report = None
    gate_table = None
    print("S5 pending: needs a baseline (S3) first.")
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
if evaluator is not None:
    from quantlab import ledger, report
    from quantlab.systems import find_study_id

    # #68: resolved independently of §4/§5's own variables -- report.tear_sheet only needs a
    # study_id + evaluator, reading everything else (including the gates event) from the ledger
    # and trial store itself, so this cell works even run alone in a fresh kernel.
    study_id = SYSTEM.get("study_id") or find_study_id(SYSTEM["book"], SYSTEM["issue"], SYSTEM["slug"])
    if study_id is not None and ledger.study_events(study_id, "gates"):
        ts = report.tear_sheet(study_id, evaluator=evaluator, risk_type=SYSTEM.get("risk_type"))
        ts.write_results(RESULTS_DIR)
        report.write_card(
            ts, slug=SYSTEM["slug"], name=SYSTEM["name"], idea=CARD.get("idea", ""),
            status=SYSTEM.get("status", "testing"), issue=SYSTEM["issue"], book=SYSTEM["book"],
        )
    else:
        ts = None
        print("S7 pending: needs validation (S5, a logged gates event) first.")
else:
    ts = None
    print("S7 pending: needs a baseline (S3) first.")
ts.headline if ts is not None else None
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
