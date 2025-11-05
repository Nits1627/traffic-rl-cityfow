#!/usr/bin/env python3
"""
Utility functions for SUMO environment setup and validation.
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple
import xml.etree.ElementTree as ET


# ---------- small helpers ----------
def _mac_framework_defaults() -> Tuple[Path, Path]:
    """
    Default SUMO Framework paths for macOS .pkg installs.
    Returns (bin_dir, tools_dir).
    """
    base = Path("/Library/Frameworks/EclipseSUMO.framework/Versions/1.24.0/EclipseSUMO")
    return base / "bin", base / "share" / "sumo" / "tools"


def _which(name: str) -> Optional[str]:
    p = shutil.which(name)
    return p if p else None


def _ensure_tools_on_sys_path(tools_dir: Path) -> None:
    if tools_dir.is_dir() and str(tools_dir) not in sys.path:
        sys.path.append(str(tools_dir))


def _parse_cfg_inputs(cfg_path: Path) -> Tuple[Path, Tuple[Path, ...]]:
    """
    Parse <input><net-file>, <route-files> from a .sumo.cfg.
    Returns absolute paths resolved against the cfg directory.
    """
    tree = ET.parse(cfg_path)
    root = tree.getroot()
    inp = root.find("input")
    if inp is None:
        raise FileNotFoundError(f"No <input> section in {cfg_path}")

    def _get_value(tag: str) -> Optional[str]:
        node = inp.find(tag)
        return node.get("value") if node is not None else None

    net_rel = _get_value("net-file")
    routes_rel = _get_value("route-files")
    if not net_rel:
        raise FileNotFoundError(f"No <net-file> entry in {cfg_path}")

    cfg_dir = cfg_path.parent
    net_path = (cfg_dir / net_rel).resolve()

    route_paths: tuple[Path, ...] = tuple()
    if routes_rel:
        # support comma-separated file list
        parts = [p.strip() for p in routes_rel.split(",") if p.strip()]
        route_paths = tuple((cfg_dir / p).resolve() for p in parts)

    return net_path, route_paths


# ---------- public utilities ----------
def setup_sumo_environment() -> None:
    """
    Ensure SUMO tools are importable and SUMO bin is on PATH.
    This DOES NOT hard-fail if something's missing (training will try TraCI and error loudly then).
    """
    sumo_home_env = os.environ.get("SUMO_HOME", "")
    tools_dir: Optional[Path] = None
    bin_dir: Optional[Path] = None

    if sumo_home_env:
        sh = Path(sumo_home_env)
        # If SUMO_HOME points to .../share/sumo
        if (sh / "tools").is_dir():
            tools_dir = sh / "tools"
            # binaries may be alongside (not typical), try anyway
            if (sh / "bin").is_dir():
                bin_dir = sh / "bin"
        # If SUMO_HOME points to a framework root with bin/
        if (sh / "bin").is_dir() and bin_dir is None:
            bin_dir = sh / "bin"

    # Fill from macOS framework defaults if needed
    if tools_dir is None or bin_dir is None:
        mac_bin, mac_tools = _mac_framework_defaults()
        if tools_dir is None and mac_tools.is_dir():
            tools_dir = mac_tools
        if bin_dir is None and mac_bin.is_dir():
            bin_dir = mac_bin

    # Last resort for bin: use PATH discovery
    if bin_dir is None:
        p = _which("sumo")
        if p:
            bin_dir = Path(p).parent

    # Apply
    if tools_dir:
        _ensure_tools_on_sys_path(tools_dir)
        # If SUMO_HOME not set, set it to the parent of tools (…/share/sumo)
        os.environ.setdefault("SUMO_HOME", str(tools_dir.parent))
    else:
        print("⚠️  SUMO tools directory not found; set SUMO_HOME so that $SUMO_HOME/tools exists.")

    if bin_dir and str(bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

    print("✅ SUMO environment configured")
    if tools_dir:
        print(f"   tools: {tools_dir}")
    if bin_dir:
        print(f"   bin:   {bin_dir}")


def check_sumo_installation() -> bool:
    """
    Detect whether SUMO + traci are available enough to run.

    We are tolerant: if the CLI probe times out or returns non-zero, we *still*
    return True, because TraCI startup will fail fast with a clear error later.
    """
    # Make sure environment is shaped before probing
    setup_sumo_environment()

    # Try importing traci (nice to have)
    try:
        import traci  # noqa: F401
        print("✅ SUMO Python package (traci) found")
    except Exception:
        print("⚠️  'traci' import failed now; may still work after setup if SUMO_HOME/tools becomes available.")

    # Resolve the sumo binary
    sumo_bin = (
        _which("sumo")
        or (os.environ.get("SUMO_HOME") and str(Path(os.environ["SUMO_HOME"]) / "bin" / "sumo"))
        or str(_mac_framework_defaults()[0] / "sumo")
    )

    if not sumo_bin or not Path(sumo_bin).exists():
        print("❌ Could not locate a SUMO binary on PATH, SUMO_HOME/bin, or macOS Framework.")
        return False

    # Probe the CLI (tolerant)
    try:
        out = subprocess.run([sumo_bin, "--version"], capture_output=True, text=True, timeout=30)
        if out.returncode == 0:
            first = out.stdout.strip().splitlines()[0] if out.stdout else "SUMO present"
            print(f"✅ SUMO CLI reachable: {first}")
            return True
        else:
            print(f"⚠️  SUMO '--version' returned non-zero: {out.stderr.strip() or out.stdout.strip()}")
            print("⚠️  Proceeding anyway; TraCI will provide a concrete error if there's an issue.")
            return True
    except subprocess.TimeoutExpired:
        print("⚠️  SUMO '--version' timed out; proceeding anyway.")
        return True
    except Exception as e:
        print(f"⚠️  SUMO version probe raised {e}; proceeding anyway.")
        return True


def get_sumo_command(executable: str = "sumo") -> str:
    """
    Resolve the full path to a SUMO executable or raise.
    """
    # 1) PATH
    p = _which(executable)
    if p:
        return p

    # 2) SUMO_HOME/bin
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "bin" / executable
        if candidate.exists():
            return str(candidate)

    # 3) macOS Framework default
    bin_dir, _ = _mac_framework_defaults()
    candidate = bin_dir / executable
    if candidate.exists():
        return str(candidate)

    raise RuntimeError(f"SUMO executable '{executable}' not found on PATH, SUMO_HOME/bin, or macOS Framework.")


def validate_project_structure(config_path: str | None = None) -> bool:
    """
    Validate that the required SUMO artifacts exist.

    Rules:
      - If a config_path is provided, validate the .sumo.cfg and the files it references.
      - Otherwise, accept either:
          a) classic layout: ./network + ./routes at project root
          b) config-driven layout under ./sumo_configs (check the first *.sumo.cfg we find)
    """
    root = Path(__file__).resolve().parents[1]

    # Prefer config-driven validation
    if config_path:
        cfg = Path(config_path)
        if not cfg.is_absolute():
            cfg = (root / cfg).resolve()

        if not cfg.exists():
            print(f"❌ SUMO config file not found: {cfg}")
            return False

        try:
            net_path, route_paths = _parse_cfg_inputs(cfg)
        except Exception as e:
            print(f"❌ Failed to parse inputs from {cfg}: {e}")
            return False

        missing: list[str] = []
        if not net_path.exists():
            missing.append(str(net_path))
        for rp in route_paths:
            if not rp.exists():
                missing.append(str(rp))

        if missing:
            print("❌ Missing required SUMO files referenced by config:")
            for m in missing:
                print(f"   - {m}")
            return False

        print("✅ Project structure is valid (via config)")
        return True

    # Fallback: directory presence or any config under sumo_configs
    classic_ok = (root / "network").is_dir() and (root / "routes").is_dir()

    sc_dir = root / "sumo_configs"
    cfg_candidates = list(sc_dir.glob("*.sumo.cfg")) if sc_dir.is_dir() else []
    cfg_ok = False
    if cfg_candidates:
        try:
            net_path, route_paths = _parse_cfg_inputs(cfg_candidates[0])
            cfg_ok = net_path.exists() and all(rp.exists() for rp in route_paths)
        except Exception:
            cfg_ok = False

    if classic_ok or cfg_ok:
        print("✅ Project structure is valid")
        return True

    # Helpful error
    missing_bits = []
    if not classic_ok:
        missing_bits.append("network/, routes/ at project root")
    if not cfg_ok:
        missing_bits.append("valid .sumo.cfg with existing net-file/route-files under sumo_configs/")
    print(f"❌ Missing required directories/files: {', '.join(missing_bits)}")
    return False