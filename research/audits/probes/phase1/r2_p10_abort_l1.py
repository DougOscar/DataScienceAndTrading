"""R2-P10 (48e696f) -- abort-on-uniform-errors (can it hide trials?) and the L1 evaluation-start clamp
(look-ahead?).  Synthetic evaluator for the abort part; dev-window bars only for L1."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r1_common import ExploitEval, tmpdirs
from quantlab import opt, ledger, gates as G
from quantlab.evaluators import SyntheticEvaluator

print("== abort_on_uniform_errors")
space = opt.SearchSpace([opt.IntParam("a", 0, 6), opt.IntParam("b", 0, 6)])
class Boom(ExploitEval):
    def __call__(self, params, *, cost=None):
        raise RuntimeError("data file missing")
class FirstTwoBoom(ExploitEval):
    def __call__(self, params, *, cost=None):
        if params["a"] == 0 and params["b"] in (0, 1):
            raise RuntimeError("lookback too short")
        return super().__call__(params, cost=cost)
inner = SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)}, rho=0.5, seed=1)
ld, sd = tmpdirs()
for cls, sid in ((Boom, "ab-a1"), (FirstTwoBoom, "ab-a2")):
    try:
        opt.run_study(cls(inner), space, book="FBS", system="ab", issue=77, attempt=int(sid[-1]), study_id=sid,
                      n_jobs=1, wfo=None, ledger_dir=ld, studies_dir=sd)
        print(f"  {sid} {cls.__name__}: completed")
    except Exception as e:
        print(f"  {sid} {cls.__name__}: {type(e).__name__}: {str(e)[:110]}")
    s = ledger.studies(ld)[sid]
    tev = ledger.study_events(sid, "trials", ledger_dir=ld)
    print(f"     ledger: trials events {[(e.get('status'), e.get('n_trials'), e.get('n_error')) for e in tev]}; "
          f"study_trial_count = {ledger.study_trial_count(s)}; recorded rows = {ledger.TrialRecorder(sid, sd).count()}")
r3 = opt.run_study(ExploitEval(inner), space, book="FBS", system="ab", issue=77, attempt=3, study_id="ab-a3", n_jobs=1,
                   wfo=None, ledger_dir=ld, studies_dir=sd)
tot, used, _ = ledger.related_prior_trials("ab-a3", ledger_dir=ld)
print(f"  attempt 3 after the two aborted studies: prior = {tot} from {used}")
r4 = opt.run_study(FirstTwoBoom(inner), space, book="FBS", system="ab", issue=77, attempt=4, study_id="ab-a4", n_jobs=1,
                   wfo=None, ledger_dir=ld, studies_dir=sd, abort_on_uniform_errors=False)
print(f"  ab-a4 with abort disabled: {r4.trials.height} trials, errors {int((r4.trials['status'] == 'error').sum())}")

print("== L1 clamp: eval start vs the conversion series' first bar; independent of the end date?")
from quantlab import data, evaluators
from sma_strategy import SmaCross, SmaParams
class Sma(SmaCross):
    params_cls = SmaParams
for sym, tf in (("USDCHF", "H1"), ("AUDNZD", "H1"), ("EURUSD", "H1")):
    spec = __import__("quantlab.costs", fromlist=["x"]).load_instrument(sym)
    out = []
    for end in ("2017-06-01", "2020-01-01", "2025-05-14"):
        ev = evaluators.RuleEvaluator(Sma, sym, tf, start=None, end=end)
        out.append(ev.eval_start_effective())
    first = data.load_bars(sym, tf, end="2016-06-01")["ts"][0]
    avail = data.conversion_available_from(spec.quote_ccy, "USD")
    print(f"  {sym} {tf}: first {tf} bar {first}; quote {spec.quote_ccy} conversion available from {avail}; "
          f"series_start(M1) {data.series_start(sym)}; eval_start_effective for end 2017/2020/2025 = {out}")
import polars as pl
from datetime import datetime
for sym, ccy in (("USDCHF", "CHF"), ("AUDNZD", "NZD")):
    t0 = data.conversion_available_from(ccy, "USD")
    for dt in (-1, 0, 1):
        ts = pl.Series([t0 + __import__("datetime").timedelta(minutes=dt)]).dt.replace_time_zone("EET").dt.convert_time_zone("UTC")
        try:
            r = data.conversion_rate(ccy, "USD", ts)
            m1 = data.load_bars(data.conversion_symbols(ccy, "USD")[0], "M1", end=t0 + __import__("datetime").timedelta(minutes=3))
            print(f"  {ccy}->USD at available_from{dt:+d}min: rate {r[0]:.6f}; first M1 rows {m1.select('ts','open','close').head(2).rows()}")
        except Exception as e:
            print(f"  {ccy}->USD at available_from{dt:+d}min: {type(e).__name__}: {str(e)[:80]}")
