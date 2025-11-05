#!/usr/bin/env python3
"""
SUMO GUI Launcher Script

- GUI-only mode: launches SUMO GUI with your config and waits until it closes.
- Agent mode: loads PPO model and runs it against the SUMO GUI using your Gym env.

This version forces use of the Eclipse SUMO macOS Framework binaries and avoids
pyenv/pip “sumo” wheel shims (which trigger libparquet dylib errors).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Project root on sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from traffic_rl.utils import check_sumo_installation, get_sumo_command  # keep using your existing helpers

# ---------- SUMO env hardening ----------
# Use framework install by default; overridden by existing env if already set.
DEFAULT_SUMO_HOME = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO"
os.environ.setdefault("SUMO_HOME", DEFAULT_SUMO_HOME)
os.environ.setdefault("SUMO_BINARY", os.path.join(os.environ["SUMO_HOME"], "bin", "sumo"))
os.environ.setdefault("SUMO_GUI_BINARY", os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui"))

def _resolve_gui_binary(gui_mode: str) -> str:
    """
    Prefer explicit SUMO_GUI_BINARY; fallback to get_sumo_command();
    final fallback to $SUMO_HOME/bin/<gui_mode>.
    """
    if gui_mode not in ("sumo-gui", "sumo-gui-simple"):
        raise ValueError(f"Invalid gui_mode: {gui_mode}")

    # 1) Respect explicit env
    if "SUMO_GUI_BINARY" in os.environ and os.path.isfile(os.environ["SUMO_GUI_BINARY"]):
        return os.environ["SUMO_GUI_BINARY"]

    # 2) Your helper (should return a fully-qualified path)
    try:
        cmd = get_sumo_command(gui_mode)
        if cmd and os.path.basename(cmd) in ("sumo-gui", "sumo-gui-simple"):
            return cmd
    except Exception:
        pass

    # 3) Framework fallback
    fallback = os.path.join(os.environ["SUMO_HOME"], "bin", gui_mode)
    return fallback

def _check_cfg(cfg_path: str) -> None:
    """Fail fast with helpful messages if cfg/network/routes are missing."""
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"SUMO config not found: {cfg_path}\n"
            "Run: python scripts/generate_network.py"
        )
    # Lightweight content check: ensure it references network/ and routes/ (no ../)
    text = Path(cfg_path).read_text(encoding="utf-8", errors="ignore")
    if "../network" in text or "../routes" in text:
        raise RuntimeError(
            "Your .sumo.cfg still uses '../network' or '../routes'. "
            "Fix it to 'network/...' and 'routes/...'.\n"
            "Open the file and update:\n"
            "  <net-file value=\"network/intersection.net.xml\"/>\n"
            "  <route-files value=\"routes/traffic.rou.xml\"/>\n"
        )

# ---------- GUI-only launcher ----------
def launch_sumo_gui(
    config_path: str,
    gui_mode: str = "sumo-gui",
    delay: float = 0.0,
    start: bool = True,
    quit_on_end: bool = True
) -> bool:
    """
    Launch SUMO GUI with the specified configuration and wait until it exits.
    """
    print("🚦 Launching SUMO GUI...")
    print(f"   Config: {config_path}")
    print(f"   GUI mode: {gui_mode}")
    print(f"   Delay(step-length): {delay}s")
    print(f"   Auto-start: {start}\n")

    _check_cfg(config_path)

    sumo_gui_cmd = _resolve_gui_binary(gui_mode)
    if not os.path.isfile(sumo_gui_cmd):
        print(f"❌ SUMO GUI binary not found at: {sumo_gui_cmd}")
        print(f"   SUMO_HOME={os.environ.get('SUMO_HOME')}")
        return False

    # Build command (use GUI binary directly; do NOT use pyenv shim)
    cmd = [sumo_gui_cmd, "-c", config_path]

    # SUMO's GUI "delay" is the *step-length* (sim time per UI step)
    # If you want visual slow-down, increase it. For realtime-ish, keep 1.0.
    if delay > 0:
        cmd.extend(["--step-length", str(delay)])

    if start:
        cmd.append("--start")
    if quit_on_end:
        cmd.append("--quit-on-end")

    cmd.extend(["--no-step-log", "--no-warnings", "--random"])

    print(f"Running command: {' '.join(cmd)}\n")

    try:
        subprocess.run(cmd, check=True)
        print("✅ SUMO GUI closed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ SUMO GUI failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ SUMO GUI interrupted by user")
        return True
    except Exception as e:
        print(f"❌ Failed to launch SUMO GUI: {e}")
        return False

# ---------- Agent + GUI ----------
def launch_with_agent(
    config_path: str,
    model_path: str,
    gui_mode: str = "sumo-gui",
    delay: float = 0.1,
    max_steps: int = 3600
) -> bool:
    """
    Launch SUMO GUI and run a trained PPO agent via your Gym env.
    We let the env handle SUMO startup (with GUI enabled) to avoid double-starting.
    """
    print("🤖 Launching SUMO GUI with agent control...")
    print(f"   Config: {config_path}")
    print(f"   Model: {model_path}")
    print(f"   GUI mode: {gui_mode}")
    print(f"   Delay(step-length): {delay}s")
    print(f"   Max steps: {max_steps}\n")

    _check_cfg(config_path)

    # Make sure env + subprocesses see the right binaries
    os.environ.setdefault("LIBSUMO", "0")
    os.environ["SUMO_GUI_BINARY"] = _resolve_gui_binary(gui_mode)
    os.environ["SUMO_BINARY"] = os.environ["SUMO_GUI_BINARY"]  # env will use GUI

    try:
        from stable_baselines3 import PPO
        from traffic_rl.env import SumoTrafficLightEnv
    except ImportError as e:
        print(f"❌ Missing requirements: {e}")
        return False

    # Load model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

    # Create env with GUI=True; let env start SUMO itself
    # NOTE: your env doesn't expose 'delay'—we can pass it via SUMO args by
    # setting an env var your env reads when building the cmd, or keep default 1.0.
    # If you want to honor delay, modify your env to read STEP_LENGTH from env.
    os.environ["SUMO_STEP_LENGTH"] = str(delay)

    try:
        env = SumoTrafficLightEnv(
            config_path=config_path,
            max_steps=max_steps,
            reward_type="waiting_time",
            gui=True  # this should make the env use sumo-gui
        )
        obs, _ = env.reset()
        step = 0
        print("🎮 Agent is now controlling the traffic light (Ctrl+C to stop)")

        while step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if step % 100 == 0:
                print(f"   Step {step}/{max_steps} | Reward {reward:.2f} | Phase {info.get('current_phase', 0)}")
            if terminated or truncated:
                print(f"✅ Simulation ended at step {step}")
                break

        env.close()
        print("✅ Agent control session ended")
        return True

    except KeyboardInterrupt:
        print("\n⏹️ Agent control interrupted by user")
        try:
            env.close()
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"❌ Agent control failed: {e}")
        try:
            env.close()
        except Exception:
            pass
        return False

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Launch SUMO GUI for traffic simulation")
    parser.add_argument("--config", type=str, default="sumo_configs/intersection.sumo.cfg",
                        help="Path to SUMO configuration file")
    parser.add_argument("--gui-mode", type=str, default="sumo-gui",
                        choices=["sumo-gui", "sumo-gui-simple"],
                        help="SUMO GUI mode")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between simulation steps (step-length, in seconds)")
    parser.add_argument("--no-start", action="store_true",
                        help="Don't start simulation automatically")
    parser.add_argument("--no-quit", action="store_true",
                        help="Don't quit when simulation ends")
    parser.add_argument("--with-agent", action="store_true",
                        help="Use trained agent to control traffic light")
    parser.add_argument("--model", type=str, default="models/ppo_traffic_light/final_model.zip",
                        help="Path to trained model (required with --with-agent)")
    parser.add_argument("--max-steps", type=int, default=3600,
                        help="Maximum simulation steps (with agent)")
    args = parser.parse_args()

    cfg = str(project_root / args.config)
    model_path = str(project_root / args.model) if args.with_agent else None

    # Validation
    print("🔍 Validating environment...")
    if not check_sumo_installation():
        print("❌ SUMO not found or misconfigured. Ensure:")
        print(f"   SUMO_HOME={os.environ.get('SUMO_HOME')}")
        print(f"   and that {os.path.join(os.environ.get('SUMO_HOME',''), 'bin')} is valid.")
        sys.exit(1)

    if args.with_agent and not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train first: python scripts/train.py")
        sys.exit(1)

    print("✅ Environment validation passed\n")

    if args.with_agent:
        ok = launch_with_agent(
            config_path=cfg,
            model_path=model_path,
            gui_mode=args.gui_mode,
            delay=args.delay if args.delay > 0 else 0.1,  # nicer default for visual agent runs
            max_steps=args.max_steps,
        )
    else:
        ok = launch_sumo_gui(
            config_path=cfg,
            gui_mode=args.gui_mode,
            delay=args.delay,
            start=not args.no_start,
            quit_on_end=not args.no_quit,
        )

    if not ok:
        print("❌ Failed to launch SUMO GUI")
        sys.exit(1)

    print("\n✅ SUMO GUI session completed!")

if __name__ == "__main__":
    main()