"""S6 red team: analyse the diagnostic grids (s6_grid_diag.py output). Read-only; no evaluations.

For each symbol x {net, gross}: Sharpe surface stats; for EURUSD gross also the study's own
selection procedure (plateau, CPCV, WFO) replayed on the gross matrix with the study's configs.
"""
import sys, warnings
from pathlib import Path
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from s5_load_study import load_study  # noqa
from quantlab import opt, stats

G = Path(sys.argv[1])
PPY = 260.0


def sh(x):
    x = np.asarray(x, float); s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(PPY)) if s > 0 else 0.0


def main():
    st = load_study()
    # (1) reproduction check of the net EURUSD grid vs the trial store
    net = pl.read_parquet(G / "EURUSD_net.parquet")
    tr = st.trials.sort("trial_id")
    # map store trial ids -> (lookback, stop_mult, hold) -> my grid index
    gi = pl.read_parquet(G / "EURUSD_net_trials.parquet")
    key = {(r["lookback"], float(r["stop_mult"]), r["hold"]): r["trial"] for r in gi.iter_rows(named=True)}
    R = st.returns
    mx = 0.0
    for r in tr.iter_rows(named=True):
        k = (r["param_lookback"], float(r["param_stop_mult"]), r["param_hold"])
        a = R.select("date", pl.col(f"t{r['trial_id']}").alias("a")).join(
            net.select("date", pl.col(f"t{key[k]}").alias("b")), on="date", how="full", coalesce=True).fill_null(0.0)
        mx = max(mx, float((a["a"] - a["b"]).abs().max()))
    print(f"[repro] net EURUSD grid vs trial store: max|diff| = {mx:.3g} over {tr.height} configs")

    sel = st.selected_params
    for f in sorted(G.glob("*_*.parquet")):
        if f.stem.endswith("_trials"):
            continue
        m = pl.read_parquet(f); t = pl.read_parquet(G / f"{f.stem}_trials.parquet")
        cols = [c for c in m.columns if c != "date"]
        S = np.array([sh(m[c].to_numpy()) for c in cols])
        idx = {int(c[1:]): j for j, c in enumerate(cols)}
        ksel = t.filter((pl.col("lookback") == sel["lookback"]) & (pl.col("stop_mult") == sel["stop_mult"])
                        & (pl.col("hold") == sel["hold"]))["trial"][0]
        best = int(np.argmax(S)); bt = t.filter(pl.col("trial") == int(cols[best][1:])).row(0, named=True)
        by = t.with_columns(pl.Series("sr", [S[idx[i]] for i in t["trial"]])).group_by("lookback").agg(
            pl.col("sr").median()).sort("lookback")
        print(f"\n[{f.stem}] n={len(S)} median SR {np.median(S):+.3f}  p90 {np.quantile(S, .9):+.3f}  max {S.max():+.3f} "
              f"at {bt['lookback']},{bt['stop_mult']},{bt['hold']}  share>0 {np.mean(S > 0):.2f}  share>0.5 {np.mean(S > .5):.3f}  "
              f"selected-config SR {S[idx[ksel]]:+.3f}  median trades {t['n_trades'].median():.0f}")
        print("   median SR by lookback:", {r[0]: round(r[1], 2) for r in by.iter_rows()})

    # (2) the study's own procedure on the GROSS EURUSD matrix (same trials frame/space/configs)
    gm = pl.read_parquet(G / "EURUSD_gross.parquet")
    ren = {}
    for r in tr.iter_rows(named=True):
        k = (r["param_lookback"], float(r["param_stop_mult"]), r["param_hold"])
        ren[f"t{key[k]}"] = f"t{r['trial_id']}"
    gm = gm.rename(ren).select("date", *[f"t{i}" for i in tr["trial_id"]])
    gm = st.returns.select("date").join(gm, on="date", how="left").fill_null(0.0)
    space = st.meta["search_space_obj"]
    obj = opt.Objective(kind="block_sharpe", n_blocks=4, lam=0.5, min_trades=200)
    cv = opt.CPCVConfig(n_groups=10, k_test=2)
    wfo = opt.WFOConfig(refit_every="3mo", window="anchored", min_train="2y")
    tc = st.meta["trade_counts"]
    for name, M in (("net(store)", st.returns), ("gross", gm)):
        paths, splits, cm = opt.cpcv_paths(M, st.trials, space, cv, objective=obj, trade_counts=tc, periods_per_year=PPY)
        psr = paths.group_by("path_id").agg(pl.col("ret")).with_columns(
            pl.col("ret").map_elements(lambda s: sh(s.to_numpy()), return_dtype=pl.Float64).alias("sr"))["sr"].to_numpy()
        w, wp, wm = opt.walk_forward(M, st.trials, space, wfo, objective=obj, trade_counts=tc, periods_per_year=PPY)
        wr = w["ret"].to_numpy(); third = len(wr) // 3
        pb = stats.pbo(M.drop("date").to_numpy(), n_splits=16) if hasattr(stats, "pbo") else None
        print(f"\n[procedure on EURUSD {name}] CPCV median path SR {np.median(psr):+.3f} (paths>0 {np.mean(psr > 0):.2f}); "
              f"WFO OOS SR {sh(wr):+.3f}, recent third {sh(wr[-third:]):+.3f}; picks {wp.select([c for c in wp.columns if c.startswith('param_')]).unique().height} distinct")
        if pb is not None:
            print("   pbo:", {k: pb[k] for k in pb if k in ("pbo", "prob_oos_loss")} if isinstance(pb, dict) else pb)
    # DSR hurdle for the best gross config (N=840 raw), for scale
    gS = np.array([sh(gm[c].to_numpy()) for c in gm.columns if c != "date"])
    try:
        from quantlab.stats import deflated_sharpe  # noqa
    except Exception:
        pass
    print(f"\n[gross EURUSD] best-of-840 SR {gS.max():+.3f} vs the study's DSR hurdle 1.07 (N=840, T=2352 d)")


if __name__ == "__main__":
    main()
