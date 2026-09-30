"""``report.tear_sheet(study_id, ...)`` -> :class:`TearSheet` (DESIGN §7, report-builder agent).

Orchestrates ``_loading`` (ledger / trial store reads), ``_metrics`` (values + one-line
interpretations + the applicability table) and ``_figures`` (matplotlib) into one object the
notebook template calls exactly as specified.  Gate *values* are always the ones logged by
``evaluate_gates(..., log=True)``; this module never calls ``evaluate_gates`` itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .. import stats as st
from ..contracts import RiskType
from ..gates import GATE_LABELS, GATE_ORDER, GATE_THRESHOLDS
from . import _card, _figures, _loading, _metrics

_BOOK_PPY = {"FBS": 260.0, "B3": 252.0}


def _json_default(v: Any) -> Any:
    if isinstance(v, RiskType):
        return v.value
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


@dataclass
class TearSheet:
    study_id: str
    risk_type: RiskType
    headline: dict[str, Any]
    metrics: dict[str, Any]
    applicability: pl.DataFrame
    interpretations: dict[str, str]
    gate_table: pl.DataFrame
    risk_profile: tuple[str, dict[str, Any]]

    # everything below is private figure-building context, not part of the public "value" API
    _daily: pl.DataFrame | None = field(default=None, repr=False)
    _ppy: float = field(default=260.0, repr=False)
    _cpcv_paths: pl.DataFrame | None = field(default=None, repr=False)
    _wfo_params: pl.DataFrame | None = field(default=None, repr=False)
    _plateau_detail: dict[str, Any] | None = field(default=None, repr=False)
    _cost_curve: tuple[np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    # ----------------------------------------------------------------- figures
    def figures(self, out_dir: str | Path) -> dict[str, Path]:
        """Write every figure this study has data for as a PNG under ``out_dir`` and return
        ``{name: path}``.  A figure with no underlying data (e.g. no CPCV paths stored) is
        silently skipped, not an error."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        builders: list[tuple[str, Any]] = []
        if self._daily is not None and self._daily.height >= 2:
            builders += [
                ("equity_underwater", lambda: _figures.fig_equity_underwater(self._daily)),
                ("sharpe_yearly", lambda: _figures.fig_yearly_sharpe(self._daily, self._ppy)),
                ("sharpe_monthly", lambda: _figures.fig_monthly_sharpe_band(self._daily, self._ppy)),
                ("monthly_heatmap", lambda: _figures.fig_monthly_heatmap(self._daily)),
                ("dd_distribution", lambda: _figures.fig_dd_distribution(self._daily)),
            ]
        builders += [
            ("cpcv_fan", lambda: _figures.fig_cpcv_path_fan(self._cpcv_paths)),
            ("plateau_heatmap", lambda: _figures.fig_plateau_heatmap(self._plateau_detail)),
            ("cost_sensitivity", lambda: _figures.fig_cost_sensitivity(
                *(self._cost_curve or (None, None)),
                threshold=GATE_THRESHOLDS["cost_stress_sharpe"][1])),
            ("wfo_drift", lambda: _figures.fig_wfo_param_drift(self._wfo_params)),
        ]
        import matplotlib.pyplot as plt
        for name, build in builders:
            fig = build()
            if fig is None:
                continue
            p = out / f"{name}.png"
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths[name] = p
        return paths

    # ----------------------------------------------------------------- markdown
    def to_markdown(self) -> str:
        h = self.headline
        band = h.get("mean_monthly_band", (float("nan"),) * 3)
        L = [f"# Tear sheet — {self.study_id}", "",
            f"**Class:** {h.get('class')} — {h.get('mean_monthly_pct', float('nan')):.1f}%/month at the "
            f"10% DD budget (bootstrap p5/p50/p95 = {band[0]:.2%}/{band[1]:.2%}/{band[2]:.2%})  ",
            f"**Gate verdict:** {h.get('gate_verdict')}  ",
            f"**Risk type:** {self.risk_type.value}  ", ""]
        risks = h.get("key_risks") or []
        if risks:
            L += ["## Key risks", ""] + [f"- {r}" for r in risks] + [""]
        L += ["## Metrics applicability", "", "| Metric | Shown | Why |", "|---|---|---|"]
        for r in self.applicability.iter_rows(named=True):
            L.append(f"| {r['metric']} | {'yes' if r['shown'] else 'no'} | {r['why']} |")
        L += ["", "## Metrics", "", "| Metric | Value | Interpretation |", "|---|---|---|"]
        for k, v in self.metrics.items():
            if isinstance(v, dict):
                continue
            disp = f"{v:.4g}" if isinstance(v, float) else str(v)
            L.append(f"| {k} | {disp} | {self.interpretations.get(k, '')} |")
        L += ["", "## Gates (as logged in the ledger)", "", "| Gate | Value | Threshold | Status |", "|---|---|---|---|"]
        for r in self.gate_table.iter_rows(named=True):
            val = "—" if r["value"] is None else f"{r['value']:.3g}" if isinstance(r["value"], float) else r["value"]
            L.append(f"| {r['label']} | {val} | {r['threshold']} | {r['status']} |")
        rp_label, rp = self.risk_profile
        L += ["", "## Risk profile", "",
             f"**{rp_label}** — MaxDD {rp.get('max_dd', float('nan')):.1%}, "
             f"longest DD {rp.get('longest_dd_days', float('nan')) / 21.0:.1f} months, "
             f"max losing streak {rp.get('max_losing_streak', 0):.0f}, "
             f"skew {rp.get('skew', float('nan')):.2f}, CVaR95 {rp.get('cvar95', float('nan')):.2%}", ""]
        return "\n".join(L)

    # ----------------------------------------------------------------- write_results
    def write_results(self, results_dir: str | Path) -> dict[str, Any]:
        out = Path(results_dir)
        out.mkdir(parents=True, exist_ok=True)
        fig_paths = self.figures(out / "figures")
        payload = {"study_id": self.study_id, "risk_type": self.risk_type.value, "headline": self.headline,
                  "metrics": self.metrics, "interpretations": self.interpretations,
                  "applicability": self.applicability.to_dicts(), "gate_table": self.gate_table.to_dicts(),
                  "risk_profile": {"label": self.risk_profile[0], "evidence": self.risk_profile[1]},
                  "figures": {k: str(v) for k, v in fig_paths.items()}}
        metrics_json = out / "metrics.json"
        metrics_json.write_text(json.dumps(payload, indent=2, default=_json_default, allow_nan=True))
        md_path = out / "tear_sheet.md"
        md_path.write_text(self.to_markdown())
        return {"metrics_json": metrics_json, "tear_sheet_md": md_path, "figures": fig_paths}


def _gate_table(gates_event: dict[str, Any] | None) -> pl.DataFrame:
    schema = {"gate": pl.Utf8, "label": pl.Utf8, "value": pl.Float64, "status": pl.Utf8, "threshold": pl.Utf8}
    if gates_event is None:
        return pl.DataFrame(schema=schema)
    gv, gs = gates_event.get("gates") or {}, gates_event.get("gate_status") or {}
    rows = []
    for g in GATE_ORDER:
        thr = GATE_THRESHOLDS.get(g)
        thr_txt = f"{thr[0]} {thr[1]}" if thr else ""
        v = gv.get(g)
        rows.append({"gate": g, "label": GATE_LABELS.get(g, g),
                    "value": float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None,
                    "status": gs.get(g), "threshold": thr_txt})
    return pl.DataFrame(rows, schema=schema)


def _key_risks(gates_event: dict[str, Any] | None, holdout: dict[str, Any],
              app_rows: list[dict[str, Any]], budget: Any) -> list[str]:
    risks: list[str] = []
    if gates_event is None:
        risks.append("Not validated yet: no gates event logged for this study.")
    else:
        gv, gs = gates_event.get("gates") or {}, gates_event.get("gate_status") or {}
        failed = [g for g in GATE_ORDER if gs.get(g) == "FAIL"]
        if failed:
            risks.append("Failed gate(s): " + ", ".join(GATE_LABELS.get(g, g) for g in failed) + ".")
        mys = gv.get("max_year_share")
        if isinstance(mys, (int, float)) and np.isfinite(mys) and mys > 0.40:
            risks.append(f"PnL concentration: the best year carries {mys:.0%} of total PnL.")
    if not holdout.get("unlocked"):
        risks.append("Holdout not unlocked: the figures above are development-data only.")
    elif holdout.get("status") == "FAIL":
        risks.append("Holdout FAILED: this system is killed.")
    elif holdout.get("status") == "NOT_DECISIVE":
        risks.append("Holdout NOT_DECISIVE: waiting for more data before a final verdict.")
    excluded_trade = [r["metric"] for r in app_rows if not r["shown"] and "no trades" in r["why"]]
    if excluded_trade:
        risks.append("Trade-level metrics unavailable: no evaluator was given to the tear sheet.")
    if budget is None:
        risks.append("Could not solve a 10% DD budget leverage (bootstrapped drawdown never reaches it).")
    return risks


def tear_sheet(study_id: str, *, evaluator: Any = None, ledger_dir: Path | None = None,
               studies_dir: Path | None = None, risk_type: Any = None, holdout: Any = None) -> TearSheet:
    row = _loading.created_row(study_id, ledger_dir)
    sd = _loading.resolve_studies_dir(row, studies_dir)
    sel_event = _loading.selection_event(study_id, ledger_dir)
    trial_id = _loading.selected_trial_id(sel_event)
    selected_params = (sel_event or {}).get("selected_params") or {}
    daily = _loading.selected_daily(study_id, trial_id, sd)
    trades = _loading.resolve_trades(evaluator, selected_params)
    rtype = _loading.resolve_risk_type(risk_type, evaluator, row)
    ppy = float(getattr(evaluator, "periods_per_year", 0.0) or 0.0) or _BOOK_PPY.get(
        str(row.get("book", "")).upper(), 260.0)

    gates_event = _loading.latest_gates_event(study_id, ledger_dir)
    budget = None
    if daily is not None and daily.height >= 20:
        try:
            budget = st.return_at_dd_budget(daily, 0.10, periods_per_year=ppy, rng=np.random.default_rng(0))
        except Exception:  # noqa: BLE001 -- classification degrades to "not available"
            budget = None
    pbo = _loading.pbo_diagnostic(study_id, sd)
    wfo_recent = _loading.wfo_recent_sharpe(study_id, sd, ppy)
    holdout_data = _loading.holdout_info(row, ledger_dir, holdout, study_id)
    cpcv_paths = _loading.load_artifact(study_id, "cpcv_paths", sd)
    wfo_params = _loading.load_artifact(study_id, "wfo_params", sd)
    plateau_detail = (gates_event or {}).get("plateau")
    cost_curve = _loading.cost_sensitivity_curve(evaluator, selected_params, ppy)

    metrics: dict[str, Any] = {}
    interpretations: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for mdict, idict, rlist in (
        _metrics.returns_and_risk(daily, ppy, budget),
        _metrics.costs(trades),
        _metrics.robustness(gates_event, pbo, holdout_data, wfo_recent),
        _metrics.trade_level(rtype, trades, daily["date"] if daily is not None else None),
    ):
        metrics.update(mdict)
        interpretations.update(idict)
        rows.extend(rlist)

    if daily is not None and daily.height >= 2:
        risk_profile = _card.risk_profile_label(daily, trades)
    else:
        risk_profile = ("not available", {})

    headline = {
        "study_id": study_id, "book": row.get("book"), "system": row.get("system"),
        "issue": row.get("issue"), "risk_type": rtype.value,
        "class": budget.label if budget is not None else "not available",
        "mean_monthly": budget.mean_monthly if budget is not None else float("nan"),
        "mean_monthly_pct": (budget.mean_monthly * 100.0) if budget is not None else float("nan"),
        "mean_monthly_band": budget.mean_monthly_band if budget is not None else (float("nan"),) * 3,
        "gate_verdict": gates_event.get("verdict") if gates_event else "not validated",
        "robustness": {k: metrics.get(k) for k in ("dsr", "cscv_oos_loss", "oos_sharpe", "wfo_oos",
                                                    "wfo_oos_recent", "plateau", "cost_stress_sharpe",
                                                    "positive_years", "max_year_share", "holdout_status", "pbo")},
        "risk_profile": risk_profile[0],
    }
    headline["key_risks"] = _key_risks(gates_event, holdout_data, rows, budget)

    return TearSheet(
        study_id=study_id, risk_type=rtype, headline=headline, metrics=metrics,
        applicability=pl.DataFrame(rows, schema={"metric": pl.Utf8, "shown": pl.Boolean, "why": pl.Utf8}),
        interpretations=interpretations, gate_table=_gate_table(gates_event), risk_profile=risk_profile,
        _daily=daily, _ppy=ppy, _cpcv_paths=cpcv_paths, _wfo_params=wfo_params,
        _plateau_detail=plateau_detail, _cost_curve=cost_curve,
    )
