#!/usr/bin/env python3

from pathlib import Path
from traffic_rl.env import SumoTrafficLightEnv

def main():
    """Example of using the 4-way intersection SUMO Traffic Light Environment."""
    print("🚦 Example: Using 4-Way Intersection SUMO Traffic Light Environment")
    print("-" * 70)
    
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Create environment with the 4-way intersection configuration
    config_path = project_root / "sumo_configs" / "network" / "four_way_intersection.sumo.cfg"
    env = SumoTrafficLightEnv(
        config_path=str(config_path),
        max_steps=10000,  # Increased from 1000 to 10000
        reward_type="waiting_time"
    )
    
    print("✅ Environment created successfully")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # Run a few random steps
    obs, info = env.reset()
    print(f"\n🎯 Initial observation shape: {obs.shape}")
    
    total_reward = 0
    for step in range(1000):  # Increased from 100 to 1000
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step % 10 == 0:  # Print every 10 steps to avoid too much output
            print(f"   Step {step + 1}: Action={action}, Reward={reward:.2f}, Phase={info.get('current_phase', 0)}")
        
        if terminated or truncated:
            break
    
    print(f"\n📊 Total reward over {step + 1} steps: {total_reward:.2f}")
    
    env.close()
    print("✅ Environment closed")

if __name__ == "__main__":
    main()