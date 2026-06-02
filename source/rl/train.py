"""Training / resume / evaluation orchestration for the RL trading agent.

Everything that needs ``torch`` / ``stable_baselines3`` lives here and is
imported lazily, so ``import source`` (and the non-RL notebooks) never pay for
the heavy deps. The notebook calls:

* :func:`train_one_seed` — train (or resume) one PPO/DQN run with checkpoint +
  eval callbacks;
* :func:`latest_checkpoint` — discover where a cell left off;
* :func:`evaluate_policy_to_signals` — roll a trained policy deterministically
  and return a Backtester-ready ``signal`` array aligned to ``df``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd

from .env import TradingEnv, ACTION_TO_SIGNAL

if TYPE_CHECKING:
    from ..strategy import DeepRLTradingParams


# --------------------------------------------------------------------- deps
def _sb3():
    """Import stable-baselines3 lazily; raise an actionable error if missing."""
    try:
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.callbacks import (
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "This step needs PyTorch + stable-baselines3 + gymnasium. Install "
            "them into a CUDA-capable venv (see the notebook §0 / README):\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu126 torch\n"
            '  pip install "stable-baselines3[extra]>=2.3" "gymnasium>=0.29"'
        ) from e
    return dict(
        PPO=PPO, DQN=DQN,
        CheckpointCallback=CheckpointCallback, EvalCallback=EvalCallback,
        Monitor=Monitor, DummyVecEnv=DummyVecEnv, SubprocVecEnv=SubprocVecEnv,
    )


def cuda_is_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ------------------------------------------------------------------- envs
def make_env_fn(
    df: pd.DataFrame, params: "DeepRLTradingParams", mode: str, seed: int
) -> Callable[[], object]:
    """Return a thunk building a Monitor-wrapped :class:`TradingEnv`."""
    sb = _sb3()
    Monitor = sb["Monitor"]

    def _thunk():
        env = TradingEnv(df, params, mode=mode)
        env.reset(seed=seed)
        return Monitor(env)

    return _thunk


def make_vec_env(
    df: pd.DataFrame,
    params: "DeepRLTradingParams",
    *,
    n_envs: int = 8,
    mode: str = "train",
    seed: int = 0,
    subproc: bool = False,
):
    """Build a (Dummy|Subproc)VecEnv of ``n_envs`` decorrelated training envs."""
    sb = _sb3()
    fns = [make_env_fn(df, params, mode, seed + i) for i in range(n_envs)]
    return sb["SubprocVecEnv"](fns) if subproc else sb["DummyVecEnv"](fns)


# ------------------------------------------------------------------- model
def build_model(
    algo: str,
    env,
    params: "DeepRLTradingParams",
    *,
    seed: int = 0,
    device: str = "auto",
    tensorboard_log: str | None = None,
):
    """Instantiate a PPO or DQN model with the doc's default hyperparameters."""
    sb = _sb3()
    arch = list(params.policy_arch)
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=0,
        device=device,
        seed=seed,
        gamma=params.gamma,
        tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=arch),
    )
    if algo == "PPO":
        return sb["PPO"](
            **common,
            learning_rate=params.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        )
    if algo == "DQN":
        return sb["DQN"](
            **common,
            learning_rate=params.learning_rate,
            buffer_size=100_000,
            learning_starts=10_000,
            batch_size=64,
            tau=1.0,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
        )
    raise ValueError(f"Unknown algorithm {algo!r} (expected 'PPO' or 'DQN')")


# -------------------------------------------------------------- checkpoints
def checkpoint_prefix(algo: str, group: str, tf: str, asset: str, seed: int) -> str:
    """Filename stem for one (algo, group, tf, asset, seed) cell."""
    return f"{algo}_{group}_{tf}_{asset}_seed{seed}"


def latest_checkpoint(
    model_dir: str | Path, algo: str, group: str, tf: str, asset: str, seed: int
) -> tuple[Path | None, int]:
    """Return ``(path, steps_trained)`` for the newest checkpoint, or ``(None, 0)``.

    Matches the ``CheckpointCallback`` naming
    ``<prefix>_<steps>_steps.zip``.
    """
    model_dir = Path(model_dir)
    prefix = checkpoint_prefix(algo, group, tf, asset, seed)
    best: tuple[Path | None, int] = (None, 0)
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)_steps\.zip$")
    if model_dir.is_dir():
        for f in model_dir.glob(f"{prefix}_*_steps.zip"):
            m = pat.match(f.name)
            if m and int(m.group(1)) >= best[1]:
                best = (f, int(m.group(1)))
    return best


# ----------------------------------------------------------------- training
def train_one_seed(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    params: "DeepRLTradingParams",
    *,
    algo: str,
    group: str,
    tf: str,
    asset: str,
    seed: int,
    model_dir: str | Path,
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    eval_freq: int = 50_000,
    save_freq: int = 100_000,
    device: str = "auto",
    tensorboard_log: str | None = None,
    subproc: bool = False,
) -> dict:
    """Train (or resume) one seed; return paths + steps-done bookkeeping.

    Resumes from the latest ``<prefix>_<steps>_steps.zip`` if present,
    continuing the global step counter (``reset_num_timesteps=False``). The
    eval-best policy is saved separately under ``<model_dir>/<prefix>_best/``.
    Stops once ``total_timesteps`` (cumulative across resumes) is reached.
    """
    sb = _sb3()
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    prefix = checkpoint_prefix(algo, group, tf, asset, seed)

    train_env = make_vec_env(
        df_train, params, n_envs=n_envs, mode="train", seed=seed, subproc=subproc
    )
    eval_env = make_vec_env(df_val, params, n_envs=1, mode="eval", seed=seed)

    ckpt_path, steps_done = latest_checkpoint(model_dir, algo, group, tf, asset, seed)
    cls = sb[algo]
    if ckpt_path is not None:
        model = cls.load(str(ckpt_path), env=train_env, device=device)
        if algo == "DQN":
            rb = model_dir / f"{prefix}_replay_buffer_{steps_done}_steps.pkl"
            if rb.exists():
                model.load_replay_buffer(str(rb))
        resuming = True
    else:
        model = build_model(
            algo, train_env, params, seed=seed, device=device,
            tensorboard_log=tensorboard_log,
        )
        resuming = False

    remaining = max(total_timesteps - steps_done, 0)
    if remaining == 0:
        train_env.close()
        eval_env.close()
        return dict(prefix=prefix, steps_done=steps_done, remaining=0,
                    resumed=resuming, final_path=ckpt_path,
                    best_path=model_dir / f"{prefix}_best" / "best_model.zip")

    checkpoint_cb = sb["CheckpointCallback"](
        save_freq=max(save_freq // n_envs, 1),
        save_path=str(model_dir),
        name_prefix=prefix,
        save_replay_buffer=(algo == "DQN"),
    )
    eval_cb = sb["EvalCallback"](
        eval_env,
        best_model_save_path=str(model_dir / f"{prefix}_best"),
        log_path=str(model_dir / f"{prefix}_best"),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=1,
        deterministic=True,
    )

    model.learn(
        total_timesteps=remaining,
        callback=[checkpoint_cb, eval_cb],
        reset_num_timesteps=not resuming,
        tb_log_name=prefix,
        progress_bar=False,
    )

    final_path = model_dir / f"{prefix}_{total_timesteps}_steps.zip"
    model.save(str(final_path))
    train_env.close()
    eval_env.close()
    return dict(
        prefix=prefix,
        steps_done=total_timesteps,
        remaining=0,
        resumed=resuming,
        final_path=final_path,
        best_path=model_dir / f"{prefix}_best" / "best_model.zip",
    )


# --------------------------------------------------------------- evaluation
def evaluate_policy_to_signals(
    model, df: pd.DataFrame, params: "DeepRLTradingParams"
) -> np.ndarray:
    """Roll ``model`` deterministically over ``df`` → Backtester ``signal`` array.

    The returned int array is aligned 1:1 with ``df`` rows (zeros before the
    env warm-up). Each entry is the agent's chosen signal at that bar
    (``-1`` / ``0`` hold / ``+1``), exactly what
    :class:`source.strategy.DeepRLTradingStrategy` replays into the Backtester.
    """
    env = TradingEnv(df, params, mode="eval")
    obs, _ = env.reset()
    signals = np.zeros(len(df), dtype=int)
    done = False
    while not done:
        t = env.t
        action, _ = model.predict(obs, deterministic=True)
        signals[t] = ACTION_TO_SIGNAL[int(action)]
        obs, _reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
    return signals
