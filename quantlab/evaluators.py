"""Evaluators: map one parameter set to an :class:`contracts.Outcome` (DESIGN §4, Phase 1).

Two implementations of the :class:`contracts.Evaluator` protocol live here:

* :class:`RuleEvaluator` — the production path for rule-based systems:
  ``strategy.signals`` → ``engine.run_backtest(m1=, timeframe=)`` →
  ``sizing.apply_sizing`` → ``sizing.daily_equity`` → daily returns + trade stats,
  over the **development window only**.
* :class:`SyntheticEvaluator` — a planted-surface generator with a known ground truth
  (parameter → annualised Sharpe).  Used by the optimiser's tests and by the Phase 1
  "validate the validator" exit test (a zero-edge system must fail, a planted edge
  must pass).  Fully data-driven (no callables), so it pickles into worker processes.

Both are *picklable configuration objects*: nothing heavy is stored on the instance.
:class:`RuleEvaluator` loads bars / M1 / instrument spec / conversion rates lazily on
first use and caches them in a module-level, per-process dict, so a process pool that
receives the evaluator once in its initializer loads the data once per worker
(the performance-engineer pattern, DESIGN §7) and never pickles a DataFrame per task.

Parameter → strategy mapping (RuleEvaluator)
--------------------------------------------
``strategy_cls`` must expose its parameter dataclass as ``strategy_cls.params_cls``
(preferred) or ``strategy_cls.Params``; the evaluator builds
``strategy_cls(params_cls(**{**fixed_params, **params}))``.  A strategy class with
neither attribute is called as ``strategy_cls(**{**fixed_params, **params})``.
``fixed_params`` are the parameters the mechanism pins down (not searched, not free
parameters for the budget); the search space only ever sees the free ones.

Sizing by risk type (DESIGN §5)
-------------------------------
``sizing="auto"`` (default) picks the mode from ``strategy_cls.risk_type``:

* **A** → ``fixed_fraction`` at ``risk_fraction`` (every trade must carry a stop);
* **B** → ``fixed_lots`` at ``lots``;
* **C** (no hard stop) → ``fixed_lots`` at ``lots``: there is no stop distance to risk-size
  on, so a constant exposure is the only well-defined choice; returns are then
  reported on the nominal account like any other system;
* **D** (continuous / vol-targeted) → ``NotImplementedError``: the Phase 0 engine is a
  single discrete-position loop, it cannot express a continuous position.

``sizing="fixed_fraction"`` / ``"fixed_lots"`` override the automatic choice.
``lots=None`` means 1.0 lot (≈1x notional on the 100k nominal account for FX majors).

Warm-up (no look-ahead)
-----------------------
Indicators need history before the evaluation start.  The evaluator loads up to
``warmup_bars`` extra bars *before* ``start`` (dev data only — the window can only
extend backwards), computes signals on ``warm-up + evaluation`` bars, then **slices
the signal frame back to the evaluation bars** and runs the engine only there.  The
engine therefore starts flat at the first evaluation bar; daily returns begin on
the server day of ``start``.  Because every strategy is causal (``testing.
assert_no_lookahead``), row *i* of the signal frame depends only on rows ≤ *i*, so
the extra history can only change the *state* of indicators at ``start`` (as it
would live), never leak anything after it.  With ``start=None`` the evaluation
begins at the first available bar and indicators simply warm up inside the window
(null signals until then).

Holdout
-------
``end=None`` resolves to the book's holdout start (exclusive upper bound, DESIGN §4.1);
an ``end`` past it raises immediately.  ``include_holdout`` is never passed to
``data.load_bars`` / ``data.conversion_rate``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, ClassVar, Optional

import numpy as np
import polars as pl

from . import config, data, metrics
from .contracts import SIGNAL_COLUMNS, Outcome, RiskType
from .costs import CostModel, load_instrument
from .engine import run_backtest
from .sizing import apply_sizing, daily_equity

__all__ = ["RuleEvaluator", "SyntheticEvaluator", "trade_stats", "clear_cache"]


# --------------------------------------------------------------------------- helpers
def _to_dt(v: datetime | date | str | None) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.replace(tzinfo=None)
    if isinstance(v, date):
        return datetime(v.year, v.month, v.day)
    return datetime.fromisoformat(str(v))


def trade_stats(trades: pl.DataFrame, *, calendar_days: bool = False) -> dict[str, float]:
    """Trade-level statistics from a sized trades frame (skipped trades excluded).

    ``hold_days_max`` / ``hold_days_p95`` count *trading* days between the entry and
    exit server dates (weekdays; calendar days when ``calendar_days``, e.g. crypto) —
    this is what the CPCV purge/embargo is sized from.
    """
    t = trades
    if "skipped" in t.columns:
        t = t.filter(~pl.col("skipped"))
    n = t.height
    out: dict[str, float] = {"n_trades": float(n)}
    if n == 0:
        out.update(win_rate=float("nan"), profit_factor=float("nan"), expectancy_points=float("nan"),
                   avg_hold_bars=float("nan"), hold_days_max=0.0, hold_days_p95=0.0)
        return out
    pnl = t["pnl_ccy"].to_numpy() if "pnl_ccy" in t.columns else t["pnl_points"].to_numpy()
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    out["win_rate"] = float(np.mean(pnl > 0))
    out["profit_factor"] = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else float("nan"))
    out["expectancy_points"] = float(t["pnl_points"].mean()) if "pnl_points" in t.columns else float("nan")
    out["avg_hold_bars"] = float(t["bars_held"].mean()) if "bars_held" in t.columns else float("nan")
    ed = t["entry_ts"].dt.date().to_numpy().astype("datetime64[D]")
    xd = t["exit_ts"].dt.date().to_numpy().astype("datetime64[D]")
    hold = (xd - ed).astype(np.int64) if calendar_days else np.busday_count(ed, xd)
    out["hold_days_max"] = float(hold.max())
    out["hold_days_p95"] = float(np.quantile(hold, 0.95))
    return out


# --------------------------------------------------------------------------- per-process data cache
@dataclass
class _Data:
    bars_sig: pl.DataFrame        # warm-up + evaluation bars (signals are computed on this)
    offset: int                   # index of the first evaluation bar inside bars_sig
    bars_eval: pl.DataFrame       # evaluation bars only (engine / sizing / MTM run on this)
    m1: Optional[pl.DataFrame]
    spec: Any
    rates: dict[str, np.ndarray]  # ccy -> rate to account ccy at each bars_eval ts_utc
    ts_utc_ms: np.ndarray         # bars_eval ts_utc as epoch-ms int64 (rate lookup)


_CACHE: dict[tuple, _Data] = {}


def clear_cache() -> None:
    """Drop every cached data bundle in this process (tests / memory pressure)."""
    _CACHE.clear()


# --------------------------------------------------------------------------- RuleEvaluator
@dataclass
class RuleEvaluator:
    """Evaluator for rule-based strategies over the dev window (see module docstring).

    Picklable: holds configuration only.  ``__call__(params, cost=None)`` → Outcome with
    ``daily`` (date, ret), sized ``trades`` and ``metrics`` = ``metrics.summary`` + trade
    stats (``n_trades, win_rate, profit_factor, expectancy_points, avg_hold_bars,
    hold_days_max, hold_days_p95, n_skipped``).  ``cost=`` overrides the configured
    :class:`CostModel` for that call only (cost-stress gate).
    """

    strategy_cls: Any
    symbol: str
    timeframe: str
    book: str = "FBS"
    start: Any = None
    end: Any = None
    sizing: str = "auto"
    risk_fraction: float = 0.01
    lots: Optional[float] = None
    cost: CostModel = field(default_factory=CostModel)
    use_m1: bool = True
    fixed_params: Optional[dict[str, Any]] = None
    warmup_bars: int = 500
    periods_per_year_override: Optional[float] = None

    requires_refit: ClassVar[bool] = False

    def __post_init__(self) -> None:
        b = config.get_book(self.book)
        self.book = b.name
        end = _to_dt(self.end)
        if end is not None and end > b.holdout_start:
            raise ValueError(
                f"RuleEvaluator is dev-window only: end={end} is past the {b.name} holdout start "
                f"{b.holdout_start} (DESIGN §4.1)."
            )
        if self.sizing not in ("auto", "fixed_fraction", "fixed_lots"):
            raise ValueError(f"sizing must be 'auto', 'fixed_fraction' or 'fixed_lots', got {self.sizing!r}")
        self._mode()  # fail fast on risk type D

    # ---------------------------------------------------------------- descriptors
    @property
    def periods_per_year(self) -> float:
        if self.periods_per_year_override is not None:
            return float(self.periods_per_year_override)
        return metrics.PERIODS_PER_YEAR[self._market()]

    @property
    def cost_version(self) -> str:
        return self.cost.version

    @property
    def end_dt(self) -> datetime:
        return _to_dt(self.end) or config.get_book(self.book).holdout_start

    @property
    def dev_window(self) -> tuple[str, str]:
        """(first, last) calendar date the evaluation can cover (last = day before ``end``)."""
        start = _to_dt(self.start)
        if start is None:
            cat = data.catalog(self.book).filter(pl.col("symbol") == self.symbol)
            start = min(cat["start"].to_list()) if cat.height else None
        last = self.end_dt - timedelta(microseconds=1)
        return (start.date().isoformat() if start else "data-start", last.date().isoformat())

    def describe(self) -> dict[str, Any]:
        return {
            "evaluator": "RuleEvaluator", "strategy": getattr(self.strategy_cls, "name", self.strategy_cls.__name__),
            "symbol": self.symbol, "timeframe": self.timeframe, "book": self.book,
            "start": str(self.start), "end": str(self.end_dt), "sizing": self._mode(),
            "risk_fraction": self.risk_fraction, "lots": self._lots(), "use_m1": self.use_m1,
            "fixed_params": self.fixed_params or {}, "warmup_bars": self.warmup_bars,
            "cost_model_version": self.cost.version,
        }

    def _market(self) -> str:
        m = self.__dict__.get("_market_name")
        if m is None:
            cat = data.catalog(self.book).filter(pl.col("symbol") == self.symbol)
            if cat.height == 0:
                raise ValueError(f"{self.symbol!r} not found in the {self.book} catalog")
            m = self.__dict__["_market_name"] = cat["market"][0]
        return m

    def _mode(self) -> str:
        rt = getattr(self.strategy_cls, "risk_type", None)
        rt = RiskType(rt) if rt is not None else None
        if rt is RiskType.D:
            raise NotImplementedError(
                "RuleEvaluator: risk type D (continuous / vol-targeted position) is not expressible with "
                "the Phase 0 discrete single-position engine."
            )
        if self.sizing != "auto":
            return self.sizing
        return "fixed_fraction" if rt is RiskType.A else "fixed_lots"

    def _lots(self) -> float:
        return 1.0 if self.lots is None else float(self.lots)

    # ---------------------------------------------------------------- data
    def _key(self) -> tuple:
        return (self.symbol, self.timeframe, self.book, str(_to_dt(self.start)), str(self.end_dt),
                int(self.warmup_bars), bool(self.use_m1 and self.timeframe != "M1"))

    def prepare(self) -> None:
        """Load and cache this evaluator's data in the current process (pool initializer hook)."""
        self._data()

    def _data(self) -> _Data:
        key = self._key()
        d = _CACHE.get(key)
        if d is not None:
            return d
        start, end = _to_dt(self.start), self.end_dt
        # Dev data only: never include_holdout; end <= holdout_start was checked in __post_init__.
        bars_all = data.load_bars(self.symbol, self.timeframe, book=self.book, start=None, end=end)
        if bars_all.height == 0:
            raise ValueError(f"no {self.symbol} {self.timeframe} bars before {end}")
        i0 = 0 if start is None else int(bars_all["ts"].search_sorted(start, side="left"))
        if i0 >= bars_all.height:
            raise ValueError(f"no {self.symbol} {self.timeframe} bars in [{start}, {end})")
        w0 = max(0, i0 - int(self.warmup_bars))
        bars_sig = bars_all.slice(w0)
        bars_eval = bars_all.slice(i0)
        m1 = None
        if self.use_m1 and self.timeframe != "M1":
            # Starts exactly at the first evaluation bar's open → every HTF bar fully covered.
            m1 = data.load_bars(self.symbol, "M1", book=self.book, start=bars_eval["ts"][0], end=end)
        spec = load_instrument(self.symbol, book=self.book)
        acct = config.get_book(self.book).account_currency
        rates: dict[str, np.ndarray] = {}
        for ccy in {spec.quote_ccy, spec.swap_ccy}:
            if ccy in ("ACCOUNT", acct):
                continue
            # One as-of lookup at every evaluation-bar open (apply_sizing only ever asks for
            # bars['ts_utc'] gathered at entry/exit idx) — identical values to calling
            # conversion_rate per trial, without re-reading the M1 cross per trial.
            rates[ccy] = data.conversion_rate(ccy, acct, bars_eval["ts_utc"], book=self.book).to_numpy()
        ts_ms = bars_eval["ts_utc"].dt.epoch("ms").to_numpy()
        d = _Data(bars_sig=bars_sig, offset=i0 - w0, bars_eval=bars_eval, m1=m1, spec=spec,
                  rates=rates, ts_utc_ms=ts_ms)
        _CACHE[key] = d
        return d

    def _rate_fn(self, d: _Data):
        book = self.book

        def rate_fn(ccy: str, account_ccy: str, ts_utc: pl.Series) -> np.ndarray:
            if ccy in d.rates:
                q = ts_utc.dt.epoch("ms").to_numpy()
                pos = np.searchsorted(d.ts_utc_ms, q)
                pos_c = np.clip(pos, 0, len(d.ts_utc_ms) - 1)
                if np.all(d.ts_utc_ms[pos_c] == q):
                    return d.rates[ccy][pos_c]
            return data.conversion_rate(ccy, account_ccy, ts_utc, book=book).to_numpy()

        return rate_fn

    # ---------------------------------------------------------------- evaluation
    def build_strategy(self, params: dict[str, Any]):
        p = {**(self.fixed_params or {}), **params}
        pcls = getattr(self.strategy_cls, "params_cls", None) or getattr(self.strategy_cls, "Params", None)
        return self.strategy_cls(pcls(**p)) if pcls is not None else self.strategy_cls(**p)

    def __call__(self, params: dict[str, Any], *, cost: Any = None) -> Outcome:
        cost = self.cost if cost is None else cost
        d = self._data()
        strat = self.build_strategy(params)
        sig_full = strat.signals(d.bars_sig)
        if sig_full.height != d.bars_sig.height:
            raise ValueError(f"strategy returned {sig_full.height} signal rows for {d.bars_sig.height} bars")
        sig = sig_full.select(list(SIGNAL_COLUMNS)).slice(d.offset)
        tf = self.timeframe if d.m1 is not None else None
        res = run_backtest(d.bars_eval, sig, d.spec, cost, m1=d.m1, timeframe=tf)
        b = config.get_book(self.book)
        mode = self._mode()
        sized = apply_sizing(
            res.trades, d.spec, mode=mode, equity0=b.nominal_equity,
            risk_fraction=self.risk_fraction if mode == "fixed_fraction" else None,
            lots=self._lots() if mode == "fixed_lots" else None,
            rate_fn=self._rate_fn(d), account_ccy=b.account_currency, bars=d.bars_eval,
        )
        eq = daily_equity(d.bars_eval, res, sized, equity0=b.nominal_equity)
        daily = eq.select("date", "ret")
        ppy = self.periods_per_year
        m = metrics.summary(daily, ppy)
        m.update(trade_stats(sized, calendar_days=self._market() == "crypto"))
        m["n_skipped"] = float(sized["skipped"].sum()) if "skipped" in sized.columns else 0.0
        return Outcome(daily=daily, trades=sized, metrics=m)


# --------------------------------------------------------------------------- SyntheticEvaluator
def _stable_seed(params: dict[str, Any], seed: int) -> int:
    h = hashlib.sha256((json.dumps(params, sort_keys=True, default=str) + f"|{seed}").encode()).hexdigest()
    return int(h[:16], 16)


@dataclass
class SyntheticEvaluator:
    """Planted-surface evaluator with a known ground truth (tests, validate-the-validator).

    True annualised Sharpe of a parameter set ``p``::

        SR(p) = base_sharpe + Σ_b height_b · shape_b(u(p))

    where ``u(p)`` maps each parameter in ``bounds`` linearly to [0, 1] and each bump is
    ``{"center": {name: value}, "height": h, "width": w, "kind": "gauss"|"box"}``
    (gauss: ``exp(-½‖(u-c)/w‖²)``; box: 1 inside the Chebyshev ball of half-width ``w``).
    Parameters not in ``bounds`` are ignored by the surface.

    Daily returns: ``vol · (SR(p)/√ppy + √rho·z_common + √(1-rho)·z_idio(p))`` on
    ``n_days`` weekdays from ``start``.  ``z_common`` is shared by all trials (seeded by
    ``seed``), ``z_idio`` is seeded by a stable hash of the params, so results are
    deterministic in any process.  ``rho=1`` makes the measured Sharpe an exact
    monotone function of SR(p) (clean planted-surface tests); lower ``rho`` adds
    realistic selection noise.  ``regimes`` optionally replaces the bumps after a given
    fraction of the sample: ``({"from": 0.5, "bumps": [...], "base_sharpe": 0.0}, ...)``.

    ``trades_per_year`` evenly spaced synthetic trades (``hold_days`` long) give the
    trade-count / holding-period inputs.  ``cost`` (per call) may be a number: annual
    Sharpe units subtracted from SR(p).  ``error_when``: param subsets that raise.
    """

    bounds: dict[str, tuple[float, float]]
    bumps: tuple = ()
    base_sharpe: float = 0.0
    n_days: int = 2340
    start: str = "2016-05-02"
    vol: float = 0.005
    rho: float = 1.0
    seed: int = 0
    trades_per_year: float = 100.0
    hold_days: int = 3
    error_when: tuple = ()
    regimes: tuple = ()
    book: str = "FBS"
    periods_per_year: float = 260.0
    cost_version: str = "synthetic"

    requires_refit: ClassVar[bool] = False

    @property
    def dev_window(self) -> tuple[str, str]:
        d = self._dates()
        return (str(d[0]), str(d[-1]))

    def describe(self) -> dict[str, Any]:
        return {"evaluator": "SyntheticEvaluator", "bounds": self.bounds, "bumps": list(self.bumps),
                "base_sharpe": self.base_sharpe, "n_days": self.n_days, "rho": self.rho, "seed": self.seed}

    def prepare(self) -> None:
        pass

    def _dates(self) -> np.ndarray:
        d0 = np.datetime64(self.start, "D")
        d0 = np.busday_offset(d0, 0, roll="forward")
        return np.busday_offset(d0, np.arange(self.n_days), roll="forward")

    def _unit(self, params: dict[str, Any]) -> dict[str, float]:
        return {k: (float(params[k]) - lo) / (hi - lo) if hi > lo else 0.0
                for k, (lo, hi) in self.bounds.items() if k in params}

    @staticmethod
    def _surface(u: dict[str, float], bumps, base: float) -> float:
        sr = float(base)
        for b in bumps:
            c = {k: v for k, v in b["center"].items()}
            w = float(b.get("width", 0.1))
            diffs = []
            for k, cv in c.items():
                if k in u:
                    diffs.append(u[k] - cv)
            dz = np.asarray(diffs, dtype=float) / w
            if b.get("kind", "gauss") == "box":
                sr += float(b["height"]) * float(np.all(np.abs(dz) <= 1.0 + 1e-12))
            else:
                sr += float(b["height"]) * float(np.exp(-0.5 * np.sum(dz ** 2)))
        return sr

    def _unit_center(self, bumps) -> tuple:
        out = []
        for b in bumps:
            c = {k: ((float(v) - self.bounds[k][0]) / (self.bounds[k][1] - self.bounds[k][0])
                     if k in self.bounds and self.bounds[k][1] > self.bounds[k][0] else 0.0)
                 for k, v in b["center"].items()}
            out.append({**b, "center": c})
        return tuple(out)

    def true_sharpe(self, params: dict[str, Any], frac: float = 0.0) -> float:
        """Ground-truth annualised Sharpe at sample fraction ``frac`` (regimes)."""
        bumps, base = self.bumps, self.base_sharpe
        for r in sorted(self.regimes, key=lambda r: r["from"]):
            if frac >= r["from"]:
                bumps, base = r.get("bumps", ()), r.get("base_sharpe", self.base_sharpe)
        return self._surface(self._unit(params), self._unit_center(bumps), base)

    def __call__(self, params: dict[str, Any], *, cost: Any = None) -> Outcome:
        for cond in self.error_when:
            if all(params.get(k) == v for k, v in cond.items()):
                raise RuntimeError(f"planted error for {params}")
        n = self.n_days
        dates = self._dates()
        frac = np.arange(n) / n
        cuts = sorted(self.regimes, key=lambda r: r["from"])
        sr = np.full(n, self.true_sharpe(params, 0.0))
        for r in cuts:
            sr[frac >= r["from"]] = self.true_sharpe(params, r["from"])
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            sr = sr - float(cost)
        zc = np.random.default_rng(self.seed).standard_normal(n)
        zi = np.random.default_rng(_stable_seed(params, self.seed)).standard_normal(n)
        ret = self.vol * (sr / math.sqrt(self.periods_per_year)
                          + math.sqrt(self.rho) * zc + math.sqrt(max(0.0, 1.0 - self.rho)) * zi)
        daily = pl.DataFrame({"date": dates.astype("datetime64[D]"), "ret": ret}).with_columns(
            pl.col("date").cast(pl.Date))
        n_tr = int(round(self.trades_per_year * n / self.periods_per_year))
        if n_tr > 0:
            ex_i = np.unique(np.linspace(min(self.hold_days, n - 1), n - 1, n_tr).round().astype(int))
            en_i = np.maximum(ex_i - self.hold_days, 0)
            trades = pl.DataFrame({
                "entry_ts": dates[en_i].astype("datetime64[ms]"),
                "exit_ts": dates[ex_i].astype("datetime64[ms]"),
            })
        else:
            trades = pl.DataFrame({"entry_ts": [], "exit_ts": []},
                                  schema={"entry_ts": pl.Datetime("ms"), "exit_ts": pl.Datetime("ms")})
        m = metrics.summary(daily, self.periods_per_year)
        m.update({"n_trades": float(trades.height), "hold_days_max": float(self.hold_days),
                  "hold_days_p95": float(self.hold_days)})
        return Outcome(daily=daily, trades=trades, metrics=m)
