"""Append-only research ledger (DESIGN §8).

Two git-tracked JSONL files under ``research/ledger/``:

* ``studies.jsonl`` — one *event* per line (``study_created``, ``gates``,
  ``decision``, ``note`` …).  A study's current state is the fold of its events;
  nothing is ever rewritten.
* ``holdout_access.jsonl`` — ``holdout_unlock`` and ``holdout_exam`` events (one unlock per
  system family, except that a NOT_DECISIVE exam may be followed by a re-exam on a longer
  horizon; DESIGN §4.4, see the "holdout" section below; a FAIL is final) and ``holdout_read``
  events (every holdout read served by ``quantlab.data``: system, symbol, range — R2-2).

Every line carries ``seq`` and ``prev`` (SHA-256 of the previous raw line), so
edits or deletions anywhere in the file are detectable with :func:`verify_chain`.

Per-trial detail (parameters, metrics, return series) is too large for git and
lives in ``research/studies/<study_id>/`` as Parquet parts written by
:class:`TrialRecorder`.  The raw trial count feeds the deflation statistics.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import polars as pl

from . import config

STUDIES_FILE = "studies.jsonl"
HOLDOUT_FILE = "holdout_access.jsonl"
READS_FILE = "holdout_reads.jsonl"       # R3-5: holdout reads, kept out of the unlock/exam chain
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


def _last_line(path: Path) -> str | None:
    """The last non-empty line of ``path``, read from the end of the file (R3-5: appends no longer
    re-read the whole file)."""
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        chunk = 4096
        while True:
            start = max(0, size - chunk)
            fh.seek(start)
            buf = fh.read(size - start)
            lines = [ln for ln in buf.split(b"\n") if ln.strip()]
            # the first piece may be a partial line unless we read from the start of the file
            if start == 0 or len(lines) >= 2:
                return lines[-1].decode() if lines else None
            chunk *= 4


def _append(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    with _locked(path):
        last = _last_line(path)
        prev = _sha(last) if last is not None else GENESIS
        seq = int(json.loads(last)["seq"]) + 1 if last is not None else 0
        row = {"seq": seq, "prev": prev, "at": _now(), **event}
        line = json.dumps(row, sort_keys=True, default=str)
        with open(path, "a") as fh:
            fh.write(line + "\n")
    return row


_READ_CACHE: dict[str, tuple[int, int, list[dict[str, Any]]]] = {}


def _read(path: Path) -> list[dict[str, Any]]:
    """Parsed rows of a ledger file, cached per (size, mtime) (R3-5).  Callers must not mutate them."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return []
    key = str(path)
    got = _READ_CACHE.get(key)
    if got is not None and got[0] == st.st_size and got[1] == st.st_mtime_ns:
        return got[2]
    rows = [json.loads(ln) for ln in _raw_lines(path)]
    _READ_CACHE[key] = (st.st_size, st.st_mtime_ns, rows)
    return rows


def verify_chain(ledger_dir: Path | None = None) -> None:
    """Raise :class:`LedgerError` if any ledger file was edited, reordered or truncated mid-chain."""
    d = _ledger_dir(ledger_dir)
    for name in (STUDIES_FILE, HOLDOUT_FILE, READS_FILE):
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
    iss = fields.get("issue")
    if isinstance(iss, bool) or not isinstance(iss, int) or iss < 0:
        raise LedgerError(f"issue must be a non-negative int (the hypothesis issue number), got {iss!r} "
                          f"(red-team R3-4: a study without an issue escapes the holdout family)")
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
    """A study's own raw trial count from its folded ledger state: the LARGER of ``n_trials_study``
    (logged by the gates, m1) and ``n_trials`` (logged by the optimizer's ``trials`` event) — a
    gates event can never lower the optimizer's count (red-team R2-4) — else
    ``n_trials_planned``; 0 if none."""
    vals = [float(state[k]) for k in ("n_trials_study", "n_trials") if state.get(k) is not None]
    if vals:
        return max(vals)
    v = state.get("n_trials_planned")
    return float(v) if v is not None else 0.0


def logged_trial_count(study_id: str, *, ledger_dir: Path | None = None) -> float:
    """This study's evaluated-trial count as logged by the optimizer (R2-4): the latest ``trials``
    event's ``n_trials`` minus its ``n_invalid`` (never-evaluated configurations), or 0 when the
    study has no ``trials`` event."""
    ev = study_events(study_id, "trials", ledger_dir=ledger_dir)
    if not ev:
        return 0.0
    last = ev[-1]
    return float(last.get("n_trials") or 0) - float(last.get("n_invalid") or 0)


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
    return [copy.deepcopy(r) for r in _read(_ledger_dir(ledger_dir) / STUDIES_FILE)
            if r.get("study_id") == study_id and (event is None or r.get("event") == event)]


def created_row(study_id: str, *, ledger_dir: Path | None = None) -> dict[str, Any] | None:
    """The study's ``study_created`` row (raw, with ``seq``), or None if the study is not in the ledger."""
    for r in _read(_ledger_dir(ledger_dir) / STUDIES_FILE):
        if r.get("study_id") == study_id and r.get("event") == "study_created":
            return copy.deepcopy(r)
    return None


def normalise_system(name: Any) -> str:
    """System name for cross-study matching (N2): casefold, strip every non-alphanumeric
    character (``"SMA-Cross"``, ``"sma_cross"`` and ``"smacross"`` are the same system)."""
    return re.sub(r"[^0-9a-z]", "", str(name or "").casefold())


def related_prior_trials(study_id: str, *, ledger_dir: Path | None = None
                         ) -> tuple[float, list[str], dict[str, Any]]:
    """Raw trials of every OTHER study in the ledger — whenever it was created (red-team R2 N2b:
    re-gating attempt 1 after attempts 2–5 exist counts them too) — that shares either the
    normalised system name (:func:`normalise_system`) or the issue number with ``study_id``,
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
        if r.get("event") != "study_created" or r["study_id"] == study_id:
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
# DESIGN §4.4 (decided 2026-09-24).  The holdout exam covers the locked year PLUS all newer data
# available at the unlock, with a band pre-registered for exactly that horizon.  Each exam ends
# PASS, FAIL or NOT_DECISIVE (all criteria pass but P(pass | zero edge) of the band > 0.30).
#
# Life cycle of one system (all rows are chained, append-only):
#
#   S5 gates → band registered: the band of the study's FIRST gates run that produced one
#              (``gates`` event), or a ``holdout_band_registered`` event.  Later gate runs log
#              their band as a diagnostic only (red-team R2-3: no re-rolling at S5).
#   [newer data exported] → band REBUILT for the longer horizon and re-registered
#                           (``holdout_band_registered``, reason logged) — only before an unlock
#   /unlock-holdout      → ``holdout_unlock`` (exam 1; the user's typed phrase; the band must be
#                           the registered one, built with the fixed seed / n_boot, for the study's
#                           registered symbols, and not stale versus the manifest)
#   judge                → ``holdout_exam`` (status + horizon) in holdout_access.jsonl, mirrored
#                           as a ``holdout_exam`` event on the study
#   FAIL                 → killed: no band rebuild, no unlock, no exam can ever follow
#   PASS                 → final (promotion may proceed)
#   NOT_DECISIVE         → the system waits (``holdout_pending``).  When more data is exported the
#                           band is rebuilt for the new, longer horizon (strictly later
#                           ``horizon_end``) BEFORE any of the new data is read, then the user
#                           unlocks exam n+1, which re-judges the FULL holdout span (holdout start
#                           → new end).  This is not a retry: nothing overturns a FAIL, and the
#                           new exam is judged on a band fixed before its new data was visible.
#
# Identity (red-team R2-2): holdout state is keyed on the system FAMILY — every holdout row whose
# (book, normalised system name) equals this system's, OR whose issue is one of the issues of
# this system's studies (or of the study at hand).  A kill, a pass or a pending exam anywhere in
# the family blocks a fresh unlock for all of its members, so renaming a system ("probe" →
# "Probe" / "probe_v2" on the same issue) does not buy a new exam.
#
# Access (R2-2): ``quantlab.data`` serves holdout bars only to an unlocked family, only for the
# symbols registered for the unlocked study (traded symbols + their conversion legs), and — until
# a PASS — only up to the latest unlock's ``horizon_end`` (:func:`holdout_access`).  Every such
# read is logged as a ``holdout_read`` event (system, symbol, range) in holdout_access.jsonl.
HOLDOUT_STATUSES = ("PASS", "FAIL", "NOT_DECISIVE")
SYSTEM_STATE_AFTER_EXAM = {"PASS": "holdout_passed", "FAIL": "killed", "NOT_DECISIVE": "holdout_pending"}
HOLDOUT_EXAM_OVERDUE_DAYS = 14          # an unlock with no recorded exam after this many days is reported
_HOLDOUT_EXAM_EVENTS = ("holdout_unlock", "holdout_exam")


def _canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def holdout_unlocks(ledger_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every ``holdout_unlock`` / ``holdout_exam`` row of ``holdout_access.jsonl`` (reads live in
    ``holdout_reads.jsonl`` — see :func:`holdout_reads`)."""
    return [r for r in _read(_ledger_dir(ledger_dir) / HOLDOUT_FILE)
            if r.get("event", "holdout_unlock") in _HOLDOUT_EXAM_EVENTS]


def holdout_reads(ledger_dir: Path | None = None) -> list[dict[str, Any]]:
    """Every logged holdout read (``holdout_read`` events in ``holdout_reads.jsonl``: book, system
    the caller named, the unlocked system / study it resolved to, symbol, range)."""
    return list(_read(_ledger_dir(ledger_dir) / READS_FILE))


def _created_rows(ledger_dir: Path | None) -> dict[str, dict[str, Any]]:
    return {r["study_id"]: r for r in _read(_ledger_dir(ledger_dir) / STUDIES_FILE)
            if r.get("event") == "study_created"}


def _as_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _book_of(v: Any) -> str | None:
    try:
        return config.get_book(v).name
    except Exception:  # noqa: BLE001 — a malformed row cannot match
        return None


def holdout_family(book: str, system: str, ledger_dir: Path | None = None, *, study_id: str | None = None,
                   issue: Any = None) -> dict[str, Any]:
    """The key set of a system's holdout family (R2-2 / R3-4), **within one book**: the closure of
    "shares a normalised system name OR an issue" over the book's studies, starting from
    ``system`` (+ ``issue`` / ``study_id``'s issue).  Symmetric: every member has the same
    family.  Returns ``book``, ``systems`` (normalised names), ``issues``, ``created`` rows."""
    b = config.get_book(book).name
    created = _created_rows(ledger_dir)
    rows = [r for r in created.values() if _book_of(r.get("book")) == b]
    systems: set[str] = {normalise_system(system)} - {""}
    issues: set[int] = set()
    if _as_int(issue) is not None:
        issues.add(_as_int(issue))
    if study_id and study_id in created and _book_of(created[study_id].get("book")) == b:
        if _as_int(created[study_id].get("issue")) is not None:
            issues.add(_as_int(created[study_id]["issue"]))
        systems.add(normalise_system(created[study_id].get("system")))
    changed = True
    while changed:
        changed = False
        for r in rows:
            ns, iss = normalise_system(r.get("system")), _as_int(r.get("issue"))
            if (ns and ns in systems) or (iss is not None and iss in issues):
                if ns and ns not in systems:
                    systems.add(ns)
                    changed = True
                if iss is not None and iss not in issues:
                    issues.add(iss)
                    changed = True
    return {"book": b, "systems": systems, "issues": issues, "created": created}


def _row_in_family(r: dict[str, Any], fam: dict[str, Any]) -> bool:
    if _book_of(r.get("book")) != fam["book"]:
        return False
    if normalise_system(r.get("system")) in fam["systems"]:
        return True
    iss = _as_int(r.get("issue"))
    if iss is None:
        iss = _as_int(fam["created"].get(r.get("study_id"), {}).get("issue"))
    return iss is not None and iss in fam["issues"]


def holdout_history(book: str, system: str, ledger_dir: Path | None = None, *, study_id: str | None = None,
                    issue: Any = None) -> list[dict[str, Any]]:
    """The unlock / exam rows of a system's holdout FAMILY (:func:`holdout_family`), in order."""
    fam = holdout_family(book, system, ledger_dir, study_id=study_id, issue=issue)
    return [r for r in holdout_unlocks(ledger_dir) if _row_in_family(r, fam)]


def is_holdout_unlocked(book: str, system: str, ledger_dir: Path | None = None) -> bool:
    return any(r.get("event", "holdout_unlock") == "holdout_unlock"
               for r in holdout_history(book, system, ledger_dir))


def holdout_state(book: str, system: str, ledger_dir: Path | None = None, *, study_id: str | None = None,
                  issue: Any = None) -> dict[str, Any]:
    """Fold of the holdout rows of a system's family (R2-2 / R3-4: within the book, the closure of
    shared normalised system names and shared issues — see :func:`holdout_family`); the state is
    the most restrictive over the whole family.

    ``n_unlocks`` / ``n_exams``; ``pending`` (unlocked, exam not yet recorded); ``last_status``
    (None / PASS / FAIL / NOT_DECISIVE); ``killed`` (any FAIL); ``passed`` (a PASS);
    ``last_horizon_end`` (of the latest exam); ``study_id`` (of the first unlock);
    ``last_unlock`` (row); ``systems`` (the system names seen in the family's rows);
    ``pending_days`` (days since the pending unlock) and ``exam_overdue`` (pending for more than
    :data:`HOLDOUT_EXAM_OVERDUE_DAYS` — reported only, it changes nothing)."""
    rows = holdout_history(book, system, ledger_dir, study_id=study_id, issue=issue)
    unlocks = [r for r in rows if r.get("event", "holdout_unlock") == "holdout_unlock"]
    exams = [r for r in rows if r.get("event") == "holdout_exam"]
    last = exams[-1] if exams else None
    pending = len(unlocks) > len(exams)
    pending_days = None
    if pending and unlocks[-1].get("at"):
        try:
            t = datetime.strptime(unlocks[-1]["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            pending_days = (datetime.now(timezone.utc) - t).total_seconds() / 86400.0
        except ValueError:
            pending_days = None
    return {"n_unlocks": len(unlocks), "n_exams": len(exams), "pending": pending,
            "last_status": last.get("status") if last else None,
            "killed": any(e.get("status") == "FAIL" for e in exams),
            "passed": any(e.get("status") == "PASS" for e in exams),
            "last_horizon_end": last.get("horizon_end") if last else None,
            "study_id": unlocks[0].get("study_id") if unlocks else None,
            "last_unlock": unlocks[-1] if unlocks else None, "exams": exams,
            "systems": sorted({str(r.get("system")) for r in rows}),
            "pending_days": pending_days,
            "exam_overdue": bool(pending_days is not None and pending_days > HOLDOUT_EXAM_OVERDUE_DAYS)}


def holdout_access(book: str, system: str, ledger_dir: Path | None = None) -> dict[str, Any] | None:
    """What ``quantlab.data`` may serve from the holdout to ``system`` (R2-2 / R3-5), or None when
    ``system`` is not (normalised) the system of an unlocked study — family members and unknown
    aliases get nothing.  ``symbols``: the symbols registered for the unlocked study (traded +
    conversion legs, stored on the latest unlock row); ``end``: exclusive timestamp limit — the
    latest unlock's ``horizon_end`` + 1 minute until a PASS, None after a decisive PASS (newer
    data then belongs to the decay review, §4.6) or for a legacy row without a horizon;
    ``study_id`` / ``system`` of that unlock."""
    st = holdout_state(book, system, ledger_dir)
    if st["n_unlocks"] == 0:
        return None
    u = st["last_unlock"] or {}
    if not normalise_system(system) or normalise_system(u.get("system")) != normalise_system(system):
        return None
    he = u.get("horizon_end")
    end = None if (st["passed"] or not he) else datetime.fromisoformat(str(he)) + timedelta(minutes=1)
    syms = list(u.get("symbols") or []) + [s for s in (u.get("conversion_legs") or []) if s not in (u.get("symbols") or [])]
    return {"end": end, "symbols": syms, "study_id": u.get("study_id"), "system": u.get("system")}


def holdout_access_end(book: str, system: str, ledger_dir: Path | None = None) -> datetime | None:
    """Latest holdout timestamp ``quantlab.data`` may serve to an unlocked system (exclusive); see
    :func:`holdout_access`.  None = no extra limit (not unlocked, after a PASS, legacy row)."""
    acc = holdout_access(book, system, ledger_dir)
    return None if acc is None else acc["end"]


_LOGGED_READS: set[tuple] = set()


def log_holdout_read(*, book: str, system: str, study_id: str | None, symbol: str, timeframe: str,
                     start: Any, end: Any, unlocked_system: str | None = None,
                     ledger_dir: Path | None = None) -> dict[str, Any] | None:
    """Append a ``holdout_read`` event (R2-2) to ``holdout_reads.jsonl`` (its own chained file, R3-5):
    who (the system named by the caller, the unlocked study it resolved to) read which symbol over
    which range.  An identical read already logged by this process is not repeated."""
    key = (str(_ledger_dir(ledger_dir)), book, system, study_id, symbol, timeframe, str(start), str(end))
    if key in _LOGGED_READS:
        return None
    _LOGGED_READS.add(key)
    return _append(_ledger_dir(ledger_dir) / READS_FILE, {
        "event": "holdout_read", "book": book, "system": system, "study_id": study_id, "symbol": symbol,
        "unlocked_system": unlocked_system,
        "timeframe": timeframe, "start": None if start is None else str(start), "end": str(end)})


def unlock_phrase(book: str, system: str) -> str:
    return f"UNLOCK HOLDOUT {config.get_book(book).name}/{system}"


def registered_holdout_band(study_id: str, *, ledger_dir: Path | None = None) -> dict[str, Any] | None:
    """The study's pre-registered holdout band (R2-3): the latest ``holdout_band_registered`` event,
    else the band of the study's FIRST ``gates`` event that produced one (later gate runs never
    replace it), else None."""
    reg = study_events(study_id, "holdout_band_registered", ledger_dir=ledger_dir)
    if reg:
        return copy.deepcopy(reg[-1]["band"])
    for r in study_events(study_id, "gates", ledger_dir=ledger_dir):
        if r.get("holdout_band"):
            return copy.deepcopy(r["holdout_band"])
    return None


def study_symbols(study_id: str, *, ledger_dir: Path | None = None) -> tuple[list[str], list[str]]:
    """(traded symbols, conversion legs) registered for a study (R2-2): from its ``study_created``
    row (``symbols`` / ``conversion_legs``, written by ``opt.run_study`` for a real evaluator),
    else from its registered holdout band.  ([], []) when none is registered yet."""
    row = created_row(study_id, ledger_dir=ledger_dir) or {}
    if row.get("symbols"):
        syms = [row["symbols"]] if isinstance(row["symbols"], str) else list(row["symbols"])
        return syms, list(row.get("conversion_legs") or [])
    band = registered_holdout_band(study_id, ledger_dir=ledger_dir)
    if band and band.get("symbols"):
        syms = [band["symbols"]] if isinstance(band["symbols"], str) else list(band["symbols"])
        return syms, list(band.get("conversion_legs") or [])
    return [], []


def record_mechanism_review(study_id: str, *, passed: bool, note: str,
                            ledger_dir: Path | None = None) -> dict[str, Any]:
    """Record the human S5 judgement of the mechanism gate (component ablation vs the card) when it
    was not run as a callable.  ``gates.unlock_holdout`` accepts a MANUAL mechanism gate only if the
    latest review ``passed``."""
    if not str(note).strip():
        raise LedgerError("a mechanism review needs a note (what was checked against the card)")
    return log_event(study_id, "mechanism_review", ledger_dir=ledger_dir, passed=bool(passed), note=str(note))


def _study_system(study_id: str, ledger_dir: Path | None) -> tuple[str, str]:
    row = created_row(study_id, ledger_dir=ledger_dir)
    if row is None:
        raise LedgerError(f"unknown study {study_id}")
    return config.get_book(row["book"]).name, row["system"]


def check_band_construction(study_id: str, band: dict[str, Any], *, ledger_dir: Path | None = None) -> None:
    """Raise :class:`LedgerError` unless ``band`` was built the one pre-registered way (R2-3): seed =
    ``stats.holdout_band_seed(study_id)``, requested ``n_boot`` = ``stats.HOLDOUT_BAND_N_BOOT`` and
    ``n_power`` = ``stats.HOLDOUT_BAND_N_POWER``, and — when the study already has registered
    symbols — for exactly those symbols."""
    from .stats import HOLDOUT_BAND_N_BOOT, HOLDOUT_BAND_N_POWER, holdout_band_seed
    want = {"seed": holdout_band_seed(study_id), "n_boot_requested": HOLDOUT_BAND_N_BOOT,
            "n_power_requested": HOLDOUT_BAND_N_POWER}
    bad = {k: (band.get(k), v) for k, v in want.items() if _as_int(band.get(k)) != v}
    if bad:
        raise LedgerError(f"{study_id}: the band was not built with the fixed construction (band, required): {bad}. "
                          f"Seed and n_boot are read-only (red-team R2-3); build it with gates.evaluate_gates / "
                          f"gates.rebuild_holdout_band")
    syms, _legs = study_symbols(study_id, ledger_dir=ledger_dir)
    bsyms = band.get("symbols")
    bsyms = [bsyms] if isinstance(bsyms, str) else list(bsyms or [])
    if syms and sorted(bsyms) != sorted(syms):
        raise LedgerError(f"{study_id}: the band is for symbols {bsyms} but the study's registered symbols are "
                          f"{syms} (R2-3: symbols come from the ledger)")


def _register_holdout_band(*, study_id: str, band: dict[str, Any], reason: str,
                           ledger_dir: Path | None = None) -> dict[str, Any]:
    """Pre-register (or re-register) the holdout pass band of a study — event
    ``holdout_band_registered`` with ``band_version``, horizon and ``reason``.

    Private (R3-1): reachable only through ``gates.rebuild_holdout_band``, which computes the band
    itself from the verified study (the first band is registered by the first gates run).
    Refuses (:class:`LedgerError`) when:

    * the band was not built with the fixed seed / n_boot / n_power, or for other symbols than
      the study's registered ones (:func:`check_band_construction`, R2-3);
    * the system FAMILY's holdout is unlocked and its exam not yet recorded — a band can never
      be rebuilt after the unlock it is judged by;
    * the family already has a FAIL (killed) or a PASS (final);
    * after a NOT_DECISIVE exam, the band does not reach **past** that exam's ``horizon_end``;
    * the band does not end strictly later than the currently registered band (a rebuild needs
      new data — re-rolling the bootstrap on the same horizon is band shopping);
    * the band has no ``horizon_days`` / ``horizon_end`` / ``horizon_source``."""
    b, system = _study_system(study_id, ledger_dir)
    missing = [k for k in ("horizon_days", "horizon_end", "horizon_source") if band.get(k) in (None, "")]
    if missing:
        raise LedgerError(f"band is missing its horizon fields {missing}; build it with a horizon from "
                          f"data.holdout_horizon (DESIGN §4.4)")
    check_band_construction(study_id, band, ledger_dir=ledger_dir)
    st = holdout_state(b, system, ledger_dir, study_id=study_id)
    if st["killed"]:
        raise LedgerError(f"{b}/{system} failed its holdout (killed; family {st['systems']}); no band may be registered")
    if st["passed"]:
        raise LedgerError(f"{b}/{system} already passed its holdout (family {st['systems']}); the band is final")
    if st["pending"]:
        raise LedgerError(f"{b}/{system}: the holdout is unlocked and its exam is not recorded (family "
                          f"{st['systems']}); a band can never be rebuilt after the unlock")
    new_end = datetime.fromisoformat(str(band["horizon_end"]))
    if st["last_horizon_end"] and new_end <= datetime.fromisoformat(str(st["last_horizon_end"])):
        raise LedgerError(f"{b}/{system}: the last exam (NOT_DECISIVE) already covered up to "
                          f"{st['last_horizon_end']}; a re-exam band must reach newer data (ends {new_end})")
    cur = registered_holdout_band(study_id, ledger_dir=ledger_dir)
    if cur and cur.get("horizon_end") and new_end <= datetime.fromisoformat(str(cur["horizon_end"])):
        raise LedgerError(f"{study_id}: the registered band already covers up to {cur['horizon_end']}; "
                          f"rebuild only when newer data exists (new end {new_end})")
    version = len(study_events(study_id, "holdout_band_registered", ledger_dir=ledger_dir)) + 1
    return log_event(study_id, "holdout_band_registered", ledger_dir=ledger_dir, band=band,
                     band_version=version, reason=str(reason),
                     horizon_days=band["horizon_days"], horizon_end=band["horizon_end"],
                     newer_data_included=band.get("newer_data_included"),
                     p_pass_zero_edge=band.get("p_pass_zero_edge"),
                     replaces_horizon_end=(cur or {}).get("horizon_end"))


def _check_band_current(b: str, band: dict[str, Any]) -> dict[str, Any]:
    """The band's horizon must be manifest-sourced and still equal the horizon the manifest gives
    NOW for its symbols + conversion legs: later → newer data was exported (rebuild first);
    earlier or a different day count → the manifest changed under the band (H6).  The manifest
    SHA is re-checked: a changed SHA with an unchanged horizon is allowed (other files changed)
    and recorded.  Returns ``{"manifest_sha_band", "manifest_sha_now", "manifest_changed"}``."""
    if band.get("horizon_source") != "manifest" or not band.get("symbols"):
        raise LedgerError("the pass band's horizon was not taken from the data manifest "
                          f"(horizon_source={band.get('horizon_source')!r}); rebuild it with "
                          "gates.rebuild_holdout_band before unlocking (DESIGN §4.4)")
    from . import data  # lazy: data imports ledger
    now = data.holdout_horizon(b, band["symbols"], periods_per_year=band.get("periods_per_year"),
                               legs=band.get("conversion_legs") or ())
    now_end, band_end = datetime.fromisoformat(now["horizon_end"]), datetime.fromisoformat(str(band["horizon_end"]))
    if now_end > band_end:
        raise LedgerError(f"newer data was exported (manifest ends {now['horizon_end']}, band built to "
                          f"{band['horizon_end']}); rebuild and re-register the band with "
                          f"gates.rebuild_holdout_band BEFORE unlocking")
    if now_end < band_end or int(now["horizon_days"]) != int(band["horizon_days"]):
        raise LedgerError(f"the data manifest no longer supports the band's horizon (manifest ends {now['horizon_end']}, "
                          f"{now['horizon_days']} days; band {band['horizon_end']}, {band['horizon_days']} days): the "
                          f"manifest changed under the band (red-team H6) — investigate before unlocking")
    sha_now, sha_band = manifest_sha(), band.get("manifest_sha")
    return {"manifest_sha_band": sha_band, "manifest_sha_now": sha_now,
            "manifest_changed": bool(sha_band and sha_band != sha_now)}


def register_holdout_band(**_: Any) -> None:
    """Removed (red-team R3-1): a band is registered only by the first gates run
    (``gates.evaluate_gates(..., log=True)``) or by ``gates.rebuild_holdout_band``."""
    raise LedgerError("ledger.register_holdout_band is not public (red-team R3-1): the first band comes from "
                      "gates.evaluate_gates(..., log=True); a newer horizon from gates.rebuild_holdout_band")


def record_holdout_unlock(**_: Any) -> None:
    """Removed (red-team R3-1): use ``gates.unlock_holdout``, which re-verifies the study, recomputes
    the gates and the band, and only then writes the unlock row."""
    raise LedgerError("ledger.record_holdout_unlock is not public (red-team R3-1): use gates.unlock_holdout, which "
                      "recomputes the verdict and the band from the verified study before unlocking")


def _record_holdout_unlock(*, book: str, system: str, study_id: str, pass_band: dict[str, Any],
                           user_confirmation: str, ledger_dir: Path | None = None,
                           verification: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unlock a holdout exam (DESIGN §4.4).  Private: only ``gates.unlock_holdout`` (called by
    ``/unlock-holdout`` after it recomputed the gates and the band, R3-1) writes the row.

    ``user_confirmation`` must be the exact :func:`unlock_phrase` **typed by the user** in the
    conversation; the skill passes it through verbatim and it is stored in the chained log.
    Code cannot tell a human from an agent, so this makes every unlock explicit and auditable
    rather than impossible to fake — agents must never compose the phrase themselves.

    ``system`` must be the study's system (ledger row).  State is read for the whole system
    FAMILY (R2-2: same normalised name, or a shared issue): a kill, a pass or a pending exam of
    any member blocks the unlock.  ``pass_band`` must be the study's registered band
    (:func:`registered_holdout_band`), built with the fixed construction for the study's
    registered symbols (:func:`check_band_construction`), with a manifest-sourced horizon that
    still equals what the manifest gives now (else rebuild it first; the manifest SHA is
    re-checked).  Exam 1 is the first unlock.  A further unlock (exam n+1) is allowed **only**
    after a NOT_DECISIVE exam, for the same study, with a band re-registered for a strictly later
    ``horizon_end``; it re-judges the full holdout span.  The unlock row stores the symbols
    (traded + conversion legs) that ``quantlab.data`` may then serve."""
    b = config.get_book(book).name
    if user_confirmation != unlock_phrase(b, system):
        raise LedgerError(f"unlock requires the user to type exactly: {unlock_phrase(b, system)!r}")
    row = created_row(study_id, ledger_dir=ledger_dir)
    if row is None:
        raise LedgerError(f"unknown study {study_id}")
    if config.get_book(row["book"]).name != b or row.get("system") != system:
        raise LedgerError(f"study {study_id} belongs to {row['book']}/{row.get('system')}, not {b}/{system}")
    st = holdout_state(b, system, ledger_dir, study_id=study_id)
    fam = f" (holdout family {st['systems']})" if st["systems"] else ""
    if st["killed"]:
        raise LedgerError(f"{b}/{system} failed its holdout (killed){fam}; a FAIL can never be re-examined")
    if st["passed"]:
        raise LedgerError(f"holdout already passed for {b}/{system}{fam}; it cannot be unlocked again")
    if st["pending"]:
        raise LedgerError(f"holdout already used for {b}/{system}{fam} (exam {st['n_unlocks']} pending); it cannot "
                          f"be unlocked twice")
    if st["n_unlocks"] and st["study_id"] != study_id:
        raise LedgerError(f"a re-exam must use the frozen study {st['study_id']}, not {study_id}")
    if not pass_band:
        raise LedgerError("a pre-registered pass band is required before unlocking")
    reg = registered_holdout_band(study_id, ledger_dir=ledger_dir)
    if reg is None or _canon_json(reg) != _canon_json(pass_band):
        raise LedgerError("pass_band is not the study's registered band (ledger.registered_holdout_band); "
                          "register it first (S5: gates.evaluate_gates(..., log=True))")
    check_band_construction(study_id, pass_band, ledger_dir=ledger_dir)
    if st["n_unlocks"] and (not pass_band.get("horizon_end") or datetime.fromisoformat(str(pass_band["horizon_end"]))
                            <= datetime.fromisoformat(str(st["last_horizon_end"]))):
        raise LedgerError(f"re-exam band must reach past the last exam's horizon {st['last_horizon_end']}")
    man = _check_band_current(b, pass_band)
    commit = git_commit()
    if commit.endswith("-dirty"):
        raise LedgerError("commit the system's code before unlocking (tree is dirty)")
    syms, legs = study_symbols(study_id, ledger_dir=ledger_dir)
    legs = sorted(set(legs) | set(pass_band.get("conversion_legs") or []))
    exam = st["n_unlocks"] + 1
    return _append(_ledger_dir(ledger_dir) / HOLDOUT_FILE, {
        "event": "holdout_unlock", "book": b, "system": system, "study_id": study_id, "issue": row.get("issue"),
        "git_commit": commit, "pass_band": pass_band, "user_confirmation": user_confirmation,
        "exam": exam, "kind": "initial" if exam == 1 else "re-exam after NOT_DECISIVE",
        "horizon_days": pass_band.get("horizon_days"), "horizon_end": pass_band.get("horizon_end"),
        "newer_data_included": pass_band.get("newer_data_included"),
        "symbols": syms, "conversion_legs": [s for s in legs if s not in syms], **man,
        "verification": verification,
    })


def record_holdout_exam(*, book: str, system: str, study_id: str, result: dict[str, Any],
                        ledger_dir: Path | None = None) -> dict[str, Any]:
    """Record the verdict of the pending exam (``holdout_exam`` in holdout_access.jsonl, mirrored
    on the study).  ``result`` is ``stats.holdout_check``'s output; its ``band`` must be the band
    of the pending unlock, and its status is re-derived here from the criteria and the band's
    ``p_pass_zero_edge`` (``stats.holdout_status``) — a mismatch raises."""
    from .stats import holdout_status  # lazy: keep the ledger import-light
    b = config.get_book(book).name
    st = holdout_state(b, system, ledger_dir, study_id=study_id)
    if not st["pending"]:
        raise LedgerError(f"{b}/{system}: no unlocked exam is pending")
    unlock = st["last_unlock"]
    if unlock.get("study_id") != study_id:
        raise LedgerError(f"the pending exam belongs to study {unlock.get('study_id')}, not {study_id}")
    band = result.get("band")
    if band is None or _canon_json(band) != _canon_json(unlock.get("pass_band")):
        raise LedgerError("the exam was not judged against the band stored with the unlock")
    status = result.get("status")
    if status not in HOLDOUT_STATUSES:
        raise LedgerError(f"status must be one of {HOLDOUT_STATUSES}, got {status!r}")
    expect = holdout_status(bool(all(result["checks"].values())), band.get("p_pass_zero_edge"))
    if status != expect:
        raise LedgerError(f"status {status!r} does not follow from the criteria and the band ({expect!r})")
    exam = unlock.get("exam", st["n_unlocks"])
    payload = {"exam": exam, "status": status, "horizon_days": unlock.get("horizon_days"),
               "horizon_end": unlock.get("horizon_end"), "newer_data_included": unlock.get("newer_data_included"),
               "checks": result.get("checks"), "values": result.get("values"),
               "p_pass_zero_edge": band.get("p_pass_zero_edge"), "n_days": result.get("n_days"),
               "system_state": SYSTEM_STATE_AFTER_EXAM[status]}
    row = _append(_ledger_dir(ledger_dir) / HOLDOUT_FILE,
                  {"event": "holdout_exam", "book": b, "system": unlock.get("system", system),
                   "study_id": study_id, "issue": unlock.get("issue"), **payload})
    log_event(study_id, "holdout_exam", ledger_dir=ledger_dir,
              **{("holdout_" + k if k in ("exam", "status", "horizon_days", "horizon_end") else k): v
                 for k, v in payload.items() if k in ("exam", "status", "horizon_days", "horizon_end",
                                                      "newer_data_included", "system_state")})
    return row


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


# --------------------------------------------------------------------------- study artifacts (R3-2)
STUDY_ARTIFACTS = ("cpcv_paths", "wfo_oos", "wfo_params", "trade_counts", "entry_counts")
EMPTY_FRAME_SHA = "empty"


def frame_sha256(df: pl.DataFrame | None) -> str:
    """Content hash of a frame in its row order (R3-2): column names (sorted), each column cast to a
    canonical type (numbers → float64 with NaN for null, dates/datetimes → int, anything else →
    string) and hashed.  None or a frame with no rows / columns → :data:`EMPTY_FRAME_SHA`."""
    if df is None or df.width == 0 or df.height == 0:
        return EMPTY_FRAME_SHA
    import numpy as np
    h = hashlib.sha256()
    h.update(str(df.height).encode())
    for c in sorted(df.columns):
        s = df[c]
        h.update(b"\x1e" + c.encode())
        if s.dtype == pl.Date:
            arr = s.cast(pl.Int32).cast(pl.Float64).fill_null(float("nan")).to_numpy()
        elif isinstance(s.dtype, pl.Datetime):
            arr = s.dt.epoch("ms").cast(pl.Float64).fill_null(float("nan")).to_numpy()
        elif s.dtype.is_numeric() or s.dtype == pl.Boolean:
            arr = s.cast(pl.Float64).fill_null(float("nan")).to_numpy()
        else:
            h.update("\x1f".join("\x00" if v is None else str(v) for v in s.to_list()).encode())
            continue
        h.update(np.ascontiguousarray(arr, dtype="<f8").tobytes())
    return h.hexdigest()


def write_study_artifacts(study_id: str, frames: dict[str, pl.DataFrame | None],
                          studies_dir: Path | None = None) -> dict[str, str]:
    """Write the study's OOS artifacts (``cpcv_paths``, ``wfo_oos``, ``wfo_params``, ``trade_counts``,
    ``entry_counts``) as ``artifact-<name>.parquet`` in its store and return their
    :func:`frame_sha256` (logged in the ``selection`` event by ``opt.run_study``)."""
    d = (Path(studies_dir) if studies_dir else config.STUDIES_DIR) / study_id
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for name in STUDY_ARTIFACTS:
        df = frames.get(name)
        p = d / f"artifact-{name}.parquet"
        sha = frame_sha256(df)
        if sha == EMPTY_FRAME_SHA:
            p.unlink(missing_ok=True)
        else:
            df.write_parquet(p)
        out[name] = sha
    return out


def load_study_artifact(study_id: str, name: str, studies_dir: Path | None = None) -> pl.DataFrame | None:
    """A stored OOS artifact (see :func:`write_study_artifacts`), or None when absent / empty."""
    if name not in STUDY_ARTIFACTS:
        raise ValueError(f"unknown artifact {name!r}")
    p = (Path(studies_dir) if studies_dir else config.STUDIES_DIR) / study_id / f"artifact-{name}.parquet"
    return pl.read_parquet(p) if p.exists() else None


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
