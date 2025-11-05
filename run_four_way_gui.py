#!/usr/bin/env python3
"""
Script to run the 4-way intersection simulation with SUMO GUI.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Path to the SUMO configuration file
    config_path = project_root / "sumo_configs" / "network" / "four_way_intersection.sumo.cfg"
    
    # Check if the configuration file exists
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    print(f"🚦 Running 4-way intersection simulation with SUMO GUI (24-hour simulation)")
    print(f"   Configuration: {config_path}")
    
    # Run SUMO GUI with the configuration
    try:
        # Use sumo-gui command
        cmd = ["sumo-gui", "-c", str(config_path)]
        print(f"   Executing: {' '.join(cmd)}")
        
        # Run the command
        process = subprocess.Popen(cmd)
        print("✅ SUMO GUI started. Close the GUI window to exit.")
        
        # Wait for the process to complete
        process.wait()
        print("✅ SUMO GUI closed.")
        return True
        
    except FileNotFoundError:
        print("❌ Error: sumo-gui command not found. Make sure SUMO is installed correctly.")
        return False
    except Exception as e:
        print(f"❌ Error running SUMO GUI: {e}")
        return False

if __name__ == "__main__":
    main()