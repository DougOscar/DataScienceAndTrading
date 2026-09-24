"""Append-only research ledger (DESIGN §8).

Two git-tracked JSONL files under ``research/ledger/``:

* ``studies.jsonl`` — one *event* per line (``study_created``, ``gates``,
  ``decision``, ``note`` …).  A study's current state is the fold of its events;
  nothing is ever rewritten.
* ``holdout_access.jsonl`` — one ``holdout_unlock`` event per system, ever.

Every line carries ``seq`` and ``prev`` (SHA-256 of the previous raw line), so
edits or deletions anywhere in the file are detectable with :func:`verify_chain`.

Per-trial detail (parameters, metrics, return series) is too large for git and
lives in ``research/studies/<study_id>/`` as Parquet parts written by
:class:`TrialRecorder`.  The raw trial count feeds the deflation statistics.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from . import config

STUDIES_FILE = "studies.jsonl"
HOLDOUT_FILE = "holdout_access.jsonl"
GENESIS = "0" * 64

REQUIRED_STUDY_FIELDS = (
    "study_id", "book", "system", "issue", "attempt", "dev_window",
    "cost_model_version", "cv_scheme",
)


class LedgerError(RuntimeError):
    pass


# --------------------------------------------------------------------------- utils
def _ledger_dir(ledger_dir: Path | None) -> Path:
    d = Path(ledger_dir) if ledger_dir is not None else config.LEDGER_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Exclusive advisory lock so concurrent writers can't interleave lines."""
    lock = path.with_suffix(path.suffix + ".lock")
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _raw_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [ln for ln in path.read_text().splitlines() if ln.strip()]


def _append(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    with _locked(path):
        lines = _raw_lines(path)
        prev = _sha(lines[-1]) if lines else GENESIS
        row = {"seq": len(lines), "prev": prev, "at": _now(), **event}
        line = json.dumps(row, sort_keys=True, default=str)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return row


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in _raw_lines(path)]


def verify_chain(ledger_dir: Path | None = None) -> None:
    """Raise :class:`LedgerError` if any ledger file was edited, reordered or truncated mid-chain."""
    d = _ledger_dir(ledger_dir)
    for name in (STUDIES_FILE, HOLDOUT_FILE):
        prev = GENESIS
        for i, line in enumerate(_raw_lines(d / name)):
            row = json.loads(line)
            if row.get("seq") != i or row.get("prev") != prev:
                raise LedgerError(f"{name}: chain broken at line {i + 1}")
            prev = _sha(line)


def git_commit() -> str:
    """Current HEAD (with ``-dirty`` suffix when the tree has uncommitted changes)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "quantlab", "research/systems"],
                               cwd=config.ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def manifest_sha() -> str:
    p = config.MANIFEST_PATH
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16] if p.exists() else "none"


# --------------------------------------------------------------------------- studies
def new_study_id(book: str, issue: int, attempt: int) -> str:
    """``fbs-0007-a2`` style id: book, issue number, attempt (1-3)."""
    if not 1 <= attempt <= 3:
        raise LedgerError("attempt must be 1..3 (DESIGN §4.5 kill rule)")
    return f"{config.get_book(book).name.lower()}-{issue:04d}-a{attempt}"


def create_study(*, ledger_dir: Path | None = None, **fields: Any) -> dict[str, Any]:
    missing = [f for f in REQUIRED_STUDY_FIELDS if f not in fields]
    if missing:
        raise LedgerError(f"study is missing fields: {missing}")
    config.get_book(fields["book"])
    path = _ledger_dir(ledger_dir) / STUDIES_FILE
    if any(r["study_id"] == fields["study_id"] and r["event"] == "study_created" for r in _read(path)):
        raise LedgerError(f"study {fields['study_id']} already exists; use a new attempt id")
    fields.setdefault("git_commit", git_commit())
    fields.setdefault("data_manifest_sha", manifest_sha())
    return _append(path, {"event": "study_created", **fields})


def log_event(study_id: str, event: str, *, ledger_dir: Path | None = None, **payload: Any) -> dict[str, Any]:
    """Append an event (``trials``, ``gates``, ``decision``, ``note``…) to an existing study."""
    if event == "study_created":
        raise LedgerError("use create_study()")
    path = _ledger_dir(ledger_dir) / STUDIES_FILE
    if not any(r["study_id"] == study_id and r["event"] == "study_created" for r in _read(path)):
        raise LedgerError(f"unknown study {study_id}")
    return _append(path, {"event": event, "study_id": study_id, **payload})


def studies(ledger_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Fold events into the current state of each study (later events override keys)."""
    state: dict[str, dict[str, Any]] = {}
    for row in _read(_ledger_dir(ledger_dir) / STUDIES_FILE):
        sid = row["study_id"]
        body = {k: v for k, v in row.items() if k not in ("seq", "prev")}
        if row["event"] == "study_created":
            state[sid] = {**body, "events": ["study_created"]}
        else:
            state[sid].update({k: v for k, v in body.items() if k not in ("event", "at")})
            state[sid]["events"].append(row["event"])
    return state


def system_trial_count(book: str, system: str, ledger_dir: Path | None = None) -> int:
    """Raw number of trials logged across all attempts of one system."""
    return sum(int(s.get("n_trials", 0)) for s in studies(ledger_dir).values()
               if s["book"] == config.get_book(book).name and s["system"] == system)


def system_effective_trials(book: str, system: str, *, exclude_study: str | None = None,
                            ledger_dir: Path | None = None) -> float:
    """Sum of the effective trials logged by the gate evaluation of every other study of a
    system (diagnostic since DESIGN v1.2 — the DSR gate uses raw counts, see
    :func:`system_prior_trials`).  Uses each study's *own* ``effective_trials_study`` when
    logged (m1: the cumulative ``effective_trials`` of older rows included the prior they were
    given, so summing them double-counts); else ``effective_trials``; else raw ``n_trials``."""
    total = 0.0
    for sid, s in studies(ledger_dir).items():
        if sid == exclude_study or s["book"] != config.get_book(book).name or s["system"] != system:
            continue
        eff = s.get("effective_trials_study", s.get("effective_trials"))
        total += float(eff) if eff is not None else float(s.get("n_trials", 0))
    return total


def study_trial_count(state: dict[str, Any]) -> float:
    """A study's own raw trial count from its folded ledger state: ``n_trials_study`` (logged by
    the gates, m1) else ``n_trials`` (logged by the optimizer's ``trials`` event) else
    ``n_trials_planned``; 0 if none."""
    for k in ("n_trials_study", "n_trials", "n_trials_planned"):
        v = state.get(k)
        if v is not None:
            return float(v)
    return 0.0


def system_prior_trials(book: str, system: str, *, exclude_study: str | None = None,
                        max_attempt: int | None = None,
                        ledger_dir: Path | None = None) -> tuple[float, list[str]]:
    """Raw trials of the *other* studies of a system (DESIGN §0.3 / §4.2 v1.2 DSR: N = this
    study's raw trials + earlier attempts').  Studies with ``attempt > max_attempt`` are
    ignored (re-validating attempt 1 must not count attempt 2).  Returns (total, study ids)."""
    b = config.get_book(book).name
    total, used = 0.0, []
    for sid, s in studies(ledger_dir).items():
        if sid == exclude_study or s.get("book") != b or s.get("system") != system:
            continue
        att = s.get("attempt")
        if max_attempt is not None and att is not None and int(att) > int(max_attempt):
            continue
        total += study_trial_count(s)
        used.append(sid)
    return total, used


def study_events(study_id: str, event: str | None = None, *,
                 ledger_dir: Path | None = None) -> list[dict[str, Any]]:
    """Raw ledger rows of one study (optionally one event type), in order."""
    return [r for r in _read(_ledger_dir(ledger_dir) / STUDIES_FILE)
            if r.get("study_id") == study_id and (event is None or r.get("event") == event)]


def created_row(study_id: str, *, ledger_dir: Path | None = None) -> dict[str, Any] | None:
    """The study's ``study_created`` row (raw, with ``seq``), or None if the study is not in the ledger."""
    for r in _read(_ledger_dir(ledger_dir) / STUDIES_FILE):
        if r.get("study_id") == study_id and r.get("event") == "study_created":
            return r
    return None


def normalise_system(name: Any) -> str:
    """System name for cross-study matching (N2): casefold, strip every non-alphanumeric
    character (``"SMA-Cross"``, ``"sma_cross"`` and ``"smacross"`` are the same system)."""
    return re.sub(r"[^0-9a-z]", "", str(name or "").casefold())


def related_prior_trials(study_id: str, *, ledger_dir: Path | None = None
                         ) -> tuple[float, list[str], dict[str, Any]]:
    """Raw trials of every OTHER study created **before** ``study_id`` (ledger order) that shares
    either the normalised system name (:func:`normalise_system`) or the issue number with it,
    whatever its attempt label or book (red-team N2: relabelling attempt / system / issue must
    not shrink the DSR's N).  Each study contributes its own count (:func:`study_trial_count`).

    Returns ``(total, study ids used, this study's created row)``.  Raises :class:`LedgerError`
    if ``study_id`` is not in the ledger."""
    rows = _read(_ledger_dir(ledger_dir) / STUDIES_FILE)
    own = next((r for r in rows if r.get("study_id") == study_id and r.get("event") == "study_created"), None)
    if own is None:
        raise LedgerError(f"unknown study {study_id}")
    sysn = normalise_system(own.get("system"))
    issue = own.get("issue")
    state = studies(ledger_dir)
    total, used = 0.0, []
    for r in rows:
        if r.get("event") != "study_created" or r["seq"] >= own["seq"] or r["study_id"] == study_id:
            continue
        same_sys = bool(sysn) and normalise_system(r.get("system")) == sysn
        same_issue = issue is not None and r.get("issue") is not None and int(r["issue"]) == int(issue)
        if same_sys or same_issue:
            total += study_trial_count(state.get(r["study_id"], {}))
            used.append(r["study_id"])
    return total, used, own


def candidate_set_data_dependent(study_id: str, *, ledger_dir: Path | None = None) -> bool:
    """True if the study's candidate set was ever data-dependent according to the ledger (B3 /
    N1): the created row says so or has ``method == "tpe"``, or ANY later event of the study set
    ``candidate_set_data_dependent=True`` or recorded TPE-sourced trials (``n_by_source``).
    Once True it can never be reset by a later event."""
    for r in study_events(study_id, ledger_dir=ledger_dir):
        if r.get("candidate_set_data_dependent") is True:
            return True
        if r.get("event") == "study_created" and r.get("method") == "tpe":
            return True
        src = r.get("n_by_source")
        if isinstance(src, dict) and int(src.get("tpe", 0) or 0) > 0:
            return True
    return False


# --------------------------------------------------------------------------- holdout
def holdout_unlocks(ledger_dir: Path | None = None) -> list[dict[str, Any]]:
    return _read(_ledger_dir(ledger_dir) / HOLDOUT_FILE)


def is_holdout_unlocked(book: str, system: str, ledger_dir: Path | None = None) -> bool:
    b = config.get_book(book).name
    return any(r["book"] == b and r["system"] == system for r in holdout_unlocks(ledger_dir))


def unlock_phrase(book: str, system: str) -> str:
    return f"UNLOCK HOLDOUT {config.get_book(book).name}/{system}"


def record_holdout_unlock(*, book: str, system: str, study_id: str, pass_band: dict[str, Any],
                          user_confirmation: str, ledger_dir: Path | None = None) -> dict[str, Any]:
    """One-shot unlock (DESIGN §4.4).  Only ``/unlock-holdout`` should call this.

    ``user_confirmation`` must be the exact :func:`unlock_phrase` **typed by the user** in the
    conversation; the skill passes it through verbatim and it is stored in the chained log.
    Code cannot tell a human from an agent, so this makes every unlock explicit and auditable
    rather than impossible to fake — agents must never compose the phrase themselves.
    """
    b = config.get_book(book).name
    if user_confirmation != unlock_phrase(b, system):
        raise LedgerError(f"unlock requires the user to type exactly: {unlock_phrase(b, system)!r}")
    if is_holdout_unlocked(b, system, ledger_dir):
        raise LedgerError(f"holdout already used for {b}/{system}; it cannot be unlocked twice")
    if study_id not in studies(ledger_dir):
        raise LedgerError(f"unknown study {study_id}")
    if not pass_band:
        raise LedgerError("a pre-registered pass band is required before unlocking")
    commit = git_commit()
    if commit.endswith("-dirty"):
        raise LedgerError("commit the system's code before unlocking (tree is dirty)")
    return _append(_ledger_dir(ledger_dir) / HOLDOUT_FILE, {
        "event": "holdout_unlock", "book": b, "system": system, "study_id": study_id,
        "git_commit": commit, "pass_band": pass_band, "user_confirmation": user_confirmation,
    })


# --------------------------------------------------------------------------- trials
class TrialRecorder:
    """Buffers trials and flushes them as Parquet parts under ``research/studies/<id>/``.

    Every evaluated configuration must go through :meth:`add` — including failed
    and pruned ones — because the raw count feeds DSR/PBO (DESIGN §0.3).
    Returns are optional per trial but required for PBO/CSCV and effective-N.
    """

    def __init__(self, study_id: str, studies_dir: Path | None = None, flush_every: int = 200):
        self.study_id = study_id
        self.dir = (Path(studies_dir) if studies_dir else config.STUDIES_DIR) / study_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self._rows: list[dict[str, Any]] = []
        self._returns: dict[str, pl.Series] = {}
        self._part = len(list(self.dir.glob("trials-*.parquet")))
        self.n_trials = self.count()

    def count(self) -> int:
        parts = sorted(self.dir.glob("trials-*.parquet"))
        return sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in parts)

    def add(self, params: dict[str, Any], metrics: dict[str, Any], *,
            status: str = "ok", returns: pl.DataFrame | None = None, source: str | None = None) -> int:
        """Record one trial; ``returns`` = frame with ``date`` + ``ret`` (daily % equity);
        ``source`` = how the configuration was proposed (grid / sobol / random / tpe — N1)."""
        trial_id = self.n_trials
        row = {
            "trial_id": trial_id, "status": status,
            "params": json.dumps(params, sort_keys=True, default=str),
            **{f"m_{k}": (float(v) if v is not None else None) for k, v in metrics.items()},
        }
        if source is not None:
            row["source"] = str(source)
        self._rows.append(row)
        if returns is not None:
            self._returns[f"t{trial_id}"] = returns
        self.n_trials += 1
        if len(self._rows) >= self.flush_every:
            self.flush()
        return trial_id

    def flush(self) -> None:
        if not self._rows:
            return
        pl.DataFrame(self._rows, infer_schema_length=None).write_parquet(self.dir / f"trials-{self._part:05d}.parquet")
        if self._returns:
            long = pl.concat([df.select("date", "ret").with_columns(pl.lit(k).alias("trial"))
                              for k, df in self._returns.items()])
            long.write_parquet(self.dir / f"returns-{self._part:05d}.parquet")
        self._part += 1
        self._rows.clear()
        self._returns.clear()

    def __enter__(self) -> "TrialRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()


def load_trials(study_id: str, studies_dir: Path | None = None) -> pl.DataFrame:
    d = (Path(studies_dir) if studies_dir else config.STUDIES_DIR) / study_id
    parts = sorted(d.glob("trials-*.parquet"))
    return pl.concat([pl.read_parquet(p) for p in parts], how="diagonal") if parts else pl.DataFrame()


def load_trial_returns(study_id: str, studies_dir: Path | None = None) -> pl.DataFrame:
    """Wide matrix: ``date`` × one column per trial (input to CSCV/PBO and effective-N)."""
    d = (Path(studies_dir) if studies_dir else config.STUDIES_DIR) / study_id
    parts = sorted(d.glob("returns-*.parquet"))
    if not parts:
        return pl.DataFrame()
    long = pl.concat([pl.read_parquet(p) for p in parts])
    return long.pivot(on="trial", index="date", values="ret").sort("date")
