#!/usr/bin/env python3
"""
Custom Gymnasium environment for SUMO traffic light control (embedded TraCI).

- Starts SUMO via traci.start(cmd) in embedded mode (no manual ports).
- One unique TraCI connection per env instance (labelled & switched).
- Auto-detect first traffic light ID and its controlled lanes.
- Observation = [total_veh, total_halts, mean_speed, mean_wait, time_since_switch, current_phase]
- Action space: 0=hold, 1=next phase, 2..=set phase k
"""

from __future__ import annotations

import os
import sys
from typing import Optional, List, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# -------- Locate SUMO tools (traci) with sensible fallbacks --------
def _ensure_traci_tools() -> str:
    sumo_home = os.environ.get("SUMO_HOME", "")
    candidates: List[str] = []

    if sumo_home:
        # If SUMO_HOME is the install root, tools are under share/sumo/tools
        candidates.append(os.path.join(sumo_home, "share", "sumo", "tools"))
        # If SUMO_HOME already points to share/sumo, tools are right below it
        candidates.append(os.path.join(sumo_home, "tools"))

    # macOS .pkg default (Framework) — tools live under share/sumo/tools
    candidates.append("/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO/share/sumo/tools")

    for p in candidates:
        if p and os.path.isdir(p):
            if p not in sys.path:
                sys.path.append(p)
            return p

    raise EnvironmentError(
        "Could not locate SUMO 'tools' (traci).\n"
        "Set SUMO_HOME to the SUMO 'share/sumo' directory. Example:\n"
        "  export SUMO_HOME=/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO/share/sumo"
    )


_tools_path = _ensure_traci_tools()
import traci  # noqa: E402


class SumoTrafficLightEnv(gym.Env):
    """
    Single-intersection traffic light control with SUMO + TraCI.

    Phases are strings of 'G', 'y', 'r' matching the number of signal groups
    in the detected traffic light. We synthesize two green phases (half lanes green each).
    """

    metadata = {"render_modes": []}

    # -------------------- init --------------------
    def __init__(
        self,
        config_path: str,
        max_steps: int = 3600,
        yellow_time: float = 3.0,
        min_green_time: float = 5.0,
        reward_type: str = "waiting_time",
        gui: bool = False,
        mock_mode: bool = False,
    ):
        super().__init__()
        self.config_path = config_path
        self.max_steps = int(max_steps)
        self.yellow_time = float(yellow_time)
        self.min_green_time = float(min_green_time)
        self.reward_type = reward_type
        self.use_gui = bool(gui)
        self.mock_mode = bool(mock_mode)

        # runtime state
        self._connected: bool = False
        self._label: Optional[str] = None  # unique TraCI label per env
        self.current_step: int = 0
        self._time_since_switch: float = 0.0
        self.current_phase: int = 0
        self._n_groups: Optional[int] = None
        self.traffic_light_id: Optional[str] = None
        self.lanes: List[str] = []
        self.phases: List[str] = []
        self.yellow_state: str = ""
        self.allred_state: str = ""

        # action space: 0=hold, 1=next phase, 2..(K+1)=set phase k
        self.action_space = spaces.Discrete(2)  # refreshed after phase discovery

        # observation: [veh_total, halts_total, mean_speed, mean_wait, time_since_switch, current_phase]
        self.observation_space = spaces.Box(low=0, high=1e6, shape=(6,), dtype=np.float32)

        print("✅ SUMO Traffic Light Env initialized")
        print(f"   Config: {self.config_path}")
        print(f"   Max steps: {self.max_steps}")
        print(f"   Reward: {self.reward_type} | GUI: {self.use_gui} | Mock: {self.mock_mode}")

    # -------------------- TraCI helpers --------------------
    def _switch(self) -> None:
        """Ensure further traci.* calls go to this env's connection."""
        if self._connected and self._label:
            try:
                traci.switch(self._label)
            except Exception:
                # If switch fails, connection likely closed; mark disconnected to avoid misuse
                self._connected = False

    # -------------------- SUMO bootstrap --------------------
    def _resolve_sumo_bin(self) -> str:
        env_key = "SUMO_GUI_BINARY" if self.use_gui else "SUMO_BINARY"
        sumo_bin = os.environ.get(env_key, "")
        if sumo_bin:
            return sumo_bin
        base = "/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO/bin"
        return f"{base}/sumo-gui" if self.use_gui else f"{base}/sumo"

    def _start_sumo(self) -> None:
        if self.mock_mode:
            print("⚠️  Running in mock simulation mode (explicit)")
            self._setup_mock_simulation()
            return

        # Close any previous session for this env
        if self._connected:
            self._stop_simulation()

        sumo_bin = self._resolve_sumo_bin()
        cmd = [sumo_bin, "-c", self.config_path, "--no-warnings", "--start"]
        print("CMD:", " ".join(cmd))

        # unique label per env instance
        self._label = f"env-{os.getpid()}-{id(self)}"
        # Start embedded TraCI with a label; raise on failure
        traci.start(cmd, label=self._label)
        self._connected = True
        self._switch()
        print("✅ TraCI connected:", traci.getVersion())

        # Discover TL + lanes
        tls = traci.trafficlight.getIDList()
        if not tls:
            # Clean up this labeled connection
            try:
                self._switch()
                traci.getConnection(self._label).close()
            except Exception:
                pass
            self._connected = False
            self._label = None
            raise RuntimeError(
                "No traffic lights found in the loaded SUMO network.\n"
                "Make sure your net contains a <tlLogic> (e.g., id='A0' or 'B1')."
            )

        self.traffic_light_id = tls[0]
        lanes = traci.trafficlight.getControlledLanes(self.traffic_light_id)
        # dedup while preserving order
        seen, uniq = set(), []
        for ln in lanes:
            if ln not in seen:
                seen.add(ln)
                uniq.append(ln)
        self.lanes = uniq

        state = traci.trafficlight.getRedYellowGreenState(self.traffic_light_id)
        self._n_groups = len(state)

        # Synthesize simple 2-phase plan
        half = max(1, self._n_groups // 2)
        greenA = "G" * half + "r" * (self._n_groups - half)
        greenB = "r" * half + "G" * (self._n_groups - half)
        self.yellow_state = "y" * self._n_groups
        self.allred_state = "r" * self._n_groups
        self.phases = [greenA, greenB]
        self._refresh_action_space()

        print(f"✅ SUMO started, TL={self.traffic_light_id}, groups={self._n_groups}, lanes={len(self.lanes)}")

    def _setup_mock_simulation(self) -> None:
        self._connected = False
        self._label = None
        self.traffic_light_id = "mock_tl"
        self.lanes = [f"mock_lane_{i}" for i in range(8)]
        self._n_groups = 8

        half = self._n_groups // 2
        greenA = "G" * half + "r" * (self._n_groups - half)
        greenB = "r" * half + "G" * (self._n_groups - half)
        self.yellow_state = "y" * self._n_groups
        self.allred_state = "r" * self._n_groups

        self.phases = [greenA, greenB]
        self.current_phase = 0
        self._time_since_switch = 0.0
        self._refresh_action_space()
        print("✅ Mock simulation initialized")

    def _refresh_action_space(self) -> None:
        k = max(0, len(self.phases))
        self.action_space = spaces.Discrete((k + 2) if k > 0 else 1)

    # -------------------- Phase control --------------------
    def _set_phase(self, idx: int, force: bool = False) -> None:
        if len(self.phases) == 0:
            return

        if idx < 0 or idx >= len(self.phases):
            raise ValueError(f"Invalid phase index: {idx} (valid: 0-{len(self.phases)-1})")

        if idx == self.current_phase and not force:
            return

        if self._time_since_switch < self.min_green_time and not force:
            return

        if self.mock_mode or not self._connected:
            self.current_phase = idx
            self._time_since_switch = 0.0
            return

        self._switch()
        if self.yellow_time > 0:
            traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, self.yellow_state)
            self._step_ticks(self.yellow_time)

        traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, self.allred_state)
        self._step_ticks(1.0)

        self.current_phase = idx
        traci.trafficlight.setRedYellowGreenState(self.traffic_light_id, self.phases[self.current_phase])
        self._time_since_switch = 0.0

    # -------------------- Sim control --------------------
    def _stop_simulation(self) -> None:
        if self._connected:
            try:
                if self._label:
                    # Close only this env's connection
                    traci.getConnection(self._label).close()
                else:
                    # Fallback: close current (should not happen if label is set)
                    traci.close()
            except Exception as e:
                print(f"⚠️  Error closing SUMO connection: {e}")
        self._connected = False
        self._label = None
        print("✅ Environment closed")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -------------------- Observation & Reward --------------------
    def _observe(self) -> np.ndarray:
        if self.mock_mode or not self._connected:
            return self._mock_observe()

        self._switch()
        veh = 0
        halts = 0
        speeds: List[float] = []
        waits: List[float] = []
        for l in self.lanes:
            try:
                veh += traci.lane.getLastStepVehicleNumber(l)
                halts += traci.lane.getLastStepHaltingNumber(l)
                speeds.append(traci.lane.getLastStepMeanSpeed(l))
                waits.append(traci.lane.getWaitingTime(l))
            except Exception:
                # lane may be internal or not available; ignore safely
                pass

        mean_speed = float(np.mean(speeds)) if speeds else 0.0
        total_wait = float(sum(waits))
        total_veh = max(1, veh)
        mean_wait = total_wait / total_veh

        return np.array(
            [veh, halts, mean_speed, mean_wait, self._time_since_switch, float(self.current_phase)],
            dtype=np.float32,
        )

    def _mock_observe(self) -> np.ndarray:
        # lightweight synthetic dynamics
        t = self.current_step
        time_factor = np.sin(t / 100.0) * 0.5 + 0.5  # 0..1

        total_veh = int(20 * time_factor) + np.random.randint(0, 5)
        total_halts = int(total_veh * (0.7 * (1 - time_factor)))
        mean_speed = max(0.0, 10.0 * (1 - 0.8 * time_factor) + np.random.normal(0, 1))
        mean_wait = max(0.0, 30.0 * time_factor + np.random.normal(0, 5))

        return np.array(
            [total_veh, total_halts, mean_speed, mean_wait, self._time_since_switch, float(self.current_phase)],
            dtype=np.float32,
        )

    def _reward(self) -> float:
        if self.mock_mode or not self._connected:
            v, h, _ms, mw, _tss, _ph = self._mock_observe()
            if self.reward_type == "waiting_time":
                return float(-(h + mw))
            elif self.reward_type == "throughput":
                return float(v - h)
            return float(v - (h + 0.5 * mw))

        self._switch()
        veh = 0
        halts = 0
        waits_total = 0.0
        for l in self.lanes:
            try:
                veh += traci.lane.getLastStepVehicleNumber(l)
                halts += traci.lane.getLastStepHaltingNumber(l)
                waits_total += traci.lane.getWaitingTime(l)
            except Exception:
                pass
        mean_wait = (waits_total / max(1, veh)) if veh > 0 else 0.0

        if self.reward_type == "waiting_time":
            return float(-(halts + mean_wait))
        elif self.reward_type == "throughput":
            return float(veh - halts)
        return float(veh - (halts + 0.5 * mean_wait))

    def _step_ticks(self, seconds: float) -> None:
        ticks = max(1, int(round(seconds)))
        if self.mock_mode or not self._connected:
            for _ in range(ticks):
                self.current_step += 1
                self._time_since_switch += 1.0
            return

        self._switch()
        for _ in range(ticks):
            traci.simulationStep()
            self.current_step += 1
            self._time_since_switch += 1.0

    # -------------------- Gym API --------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self._time_since_switch = 0.0
        self.current_phase = 0

        self._start_sumo()
        self._refresh_action_space()

        # Force initial phase only if connected to real SUMO with a TL
        if (not self.mock_mode) and self._connected and self.traffic_light_id and len(self.phases) > 0:
            self._set_phase(0, force=True)

        obs = self._observe()
        info = {
            "tl": self.traffic_light_id,
            "n_groups": self._n_groups,
            "lanes": list(self.lanes),
            "phase": int(self.current_phase),
            "mock_mode": bool(self.mock_mode),
        }
        return obs, info

    def step(self, action: int):
        prev_phase = self.current_phase

        if len(self.phases) == 0:
            self._step_ticks(1.0)
            obs = self._observe()
            reward = self._reward()
            terminated = False
            truncated = self.current_step >= self.max_steps
            if self._connected:
                try:
                    self._switch()
                    truncated = truncated or (traci.simulation.getMinExpectedNumber() == 0)
                except Exception:
                    pass
            return obs, float(reward), terminated, truncated, {"current_phase": int(self.current_phase)}

        if action == 0:
            pass
        elif action == 1:
            self._set_phase((self.current_phase + 1) % len(self.phases))
        else:
            self._set_phase(int(action - 2))

        self._step_ticks(1.0)

        obs = self._observe()
        reward = self._reward()
        if self.current_phase != prev_phase:
            reward -= 2.0  # small switch penalty

        terminated = False
        truncated = self.current_step >= self.max_steps
        if self._connected:
            try:
                self._switch()
                truncated = truncated or (traci.simulation.getMinExpectedNumber() == 0)
            except Exception:
                pass

        info = {"current_phase": int(self.current_phase)}
        return obs, float(reward), terminated, truncated, info

    # -------------------- teardown --------------------
    def close(self) -> None:
        self._stop_simulation()