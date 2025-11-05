#!/usr/bin/env python3
"""
PPO Training Script for SUMO Traffic Light Control (robust)

- Uses embedded TraCI (Python launches SUMO; no remote ports).
- Validates SUMO env + config up front.
- Graceful TensorBoard/protobuf handling (auto-disables TB if incompatible).
- Clean shutdown of training/eval envs in all cases.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# ---- Project import path ----
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# ---- Third-party ----
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# ---- Local ----
from traffic_rl.env import SumoTrafficLightEnv
from traffic_rl.utils import check_sumo_installation, validate_project_structure


def _safe_tensorboard_logdir(requested_logdir: str | None) -> str | None:
    """
    Try to use tensorboard; if protobuf ABI is incompatible, disable it and continue.
    Returns a usable logdir or None.
    """
    if not requested_logdir:
        return None
    try:
        import tensorboard  # noqa
        import google.protobuf  # noqa
        # quick version print for sanity
        # print("tensorboard:", tensorboard.__version__, "protobuf:", google.protobuf.__version__)
        Path(requested_logdir).mkdir(parents=True, exist_ok=True)
        return requested_logdir
    except Exception as e:
        print(f"⚠️  TensorBoard disabled (import/ABI issue): {e}")
        print("    Tip: `python -m pip install 'protobuf==3.20.3' 'tensorboard>=2.14,<2.18'`")
        return None


def _ensure_dirs(*paths: str) -> None:
    for p in paths:
        if p:
            Path(p).mkdir(parents=True, exist_ok=True)


def create_training_env(config_path: str, max_steps: int = 3600, reward_type: str = "waiting_time"):
    env = SumoTrafficLightEnv(
        config_path=config_path,
        max_steps=max_steps,
        reward_type=reward_type,
        gui=False,           # headless by default; set True if you want GUI runs
              # Use mock mode by default
    )
    return Monitor(env, filename=None)


def create_eval_env(config_path: str, max_steps: int = 3600, reward_type: str = "waiting_time"):
    env = SumoTrafficLightEnv(
        config_path=config_path,
        max_steps=max_steps,
        reward_type=reward_type,
        gui=False,
              # Use mock mode by default
    )
    return Monitor(env, filename=None)


def train_ppo_agent(
    config_path: str,
    total_timesteps: int = 100_000,
    max_steps: int = 3600,
    reward_type: str = "waiting_time",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_freq: int = 5000,
    n_eval_episodes: int = 5,
    model_save_path: str | None = None,
    tensorboard_log: str | None = None,
):
    print("🚦 Starting PPO training for SUMO traffic light control...")
    print(f"   Config: {config_path}")
    print(f"   Total timesteps: {total_timesteps:,}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Reward type: {reward_type}")
    print(f"   Learning rate: {learning_rate} | Batch size: {batch_size} | n_steps: {n_steps}\n")

    # Prepare logging/output
    tensorboard_log = _safe_tensorboard_logdir(tensorboard_log)
    _ensure_dirs(model_save_path)

    # Environments
    print("📦 Creating training environment...")
    train_env = create_training_env(config_path, max_steps, reward_type)

    print("📦 Creating evaluation environment...")
    eval_env = create_eval_env(config_path, max_steps, reward_type)

    print("🤖 Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1,
        tensorboard_log=tensorboard_log,
    )

    callbacks = []
    if eval_freq > 0:
        # Use separate dirs for best models and logs (both under model_save_path)
        best_dir = model_save_path if model_save_path else "models/ppo_traffic_light"
        _ensure_dirs(best_dir)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_dir,
            log_path=best_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    print("🏋️ Starting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True,
        )
        print("✅ Training completed successfully!")
        if model_save_path:
            final_model_path = Path(model_save_path) / "final_model"
            model.save(str(final_model_path))
            print(f"💾 Final model saved to: {final_model_path}")

        print("\n📊 Training Summary:")
        print(f"   Total timesteps: {total_timesteps:,}")
        print(f"   Episodes completed (approx): ~{total_timesteps // max_steps}")
        if model_save_path:
            print(f"   Model artifacts: {model_save_path}")
        if tensorboard_log:
            print(f"   TensorBoard logs: {tensorboard_log}")
            print(f"   View with: tensorboard --logdir '{tensorboard_log}'")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
        if model_save_path:
            interrupted_path = Path(model_save_path) / "interrupted_model"
            model.save(str(interrupted_path))
            print(f"💾 Model snapshot saved to: {interrupted_path}")
        raise
    finally:
        # Always clean up
        try:
            train_env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass
        print("🧹 Environments closed")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for SUMO traffic light control")

    # Environment
    parser.add_argument("--config", type=str, default="sumo_configs/intersection.sumo.cfg",
                        help="Path to SUMO configuration file (relative to project root)")
    parser.add_argument("--max-steps", type=int, default=3600, help="Max steps per episode")
    parser.add_argument("--reward-type", type=str, default="waiting_time",
                        choices=["waiting_time", "throughput", "combined"])

    # Training
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Total timesteps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per PPO update")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs per update")

    # PPO params
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Eval & logging
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=5)
    parser.add_argument("--model-save-path", type=str, default="models/ppo_traffic_light")
    parser.add_argument("--tensorboard-log", type=str, default="logs/tensorboard")

    args = parser.parse_args()

    # Resolve absolute paths
    config_path = str((project_root / args.config).resolve())
    model_save_path = str((project_root / args.model_save_path).resolve())
    tensorboard_log = str((project_root / args.tensorboard_log).resolve())

    # Validate SUMO env & project structure
    print("🔍 Validating environment...")
    if not check_sumo_installation():
        print("❌ SUMO installation not found or SUMO_HOME/tools not on PYTHONPATH.")
        print("   Fix your env (export SUMO_HOME, PATH, PYTHONPATH) and try again.")
        sys.exit(1)

    if not validate_project_structure():
        print("❌ Project structure is invalid. Please run your network generation script first.")
        sys.exit(1)

    if not os.path.exists(config_path):
        print(f"❌ SUMO config not found: {config_path}")
        print("   Please run your network generation step to create the cfg/net/route files.")
        sys.exit(1)

    print("✅ Environment validation passed\n")

    # Kick off training
    train_ppo_agent(
        config_path=config_path,
        total_timesteps=args.total_timesteps,
        max_steps=args.max_steps,
        reward_type=args.reward_type,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        model_save_path=model_save_path,
        tensorboard_log=tensorboard_log,
    )


if __name__ == "__main__":
    main()