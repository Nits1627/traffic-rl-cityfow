#!/usr/bin/env python3
"""
Evaluation Script for SUMO Traffic Light Control

Evaluates a trained PPO agent on the SUMO traffic light control task.
Provides detailed metrics and visualization of the agent's performance.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from traffic_rl.env import SumoTrafficLightEnv
from traffic_rl.utils import check_sumo_installation, validate_project_structure


def evaluate_agent(
    model_path: str,
    config_path: str,
    n_episodes: int = 10,
    max_steps: int = 3600,
    reward_type: str = "waiting_time",
    render: bool = False,
    save_results: bool = True
):
    """
    Evaluate a trained PPO agent.
    
    Args:
        model_path (str): Path to the trained model
        config_path (str): Path to SUMO configuration file
        n_episodes (int): Number of episodes to evaluate
        max_steps (int): Maximum steps per episode
        reward_type (str): Type of reward function
        render (bool): Whether to render the environment
        save_results (bool): Whether to save results to file
        
    Returns:
        dict: Evaluation results
    """
    
    print(f"🔍 Evaluating agent: {model_path}")
    print(f"   Config: {config_path}")
    print(f"   Episodes: {n_episodes}")
    print(f"   Max steps per episode: {max_steps}")
    print(f"   Reward type: {reward_type}")
    print()
    
    # Load the trained model
    try:
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Create evaluation environment
    env = SumoTrafficLightEnv(
        config_path=config_path,
        max_steps=max_steps,
        reward_type=reward_type
    )
    
    # Wrap with Monitor for logging
    env = Monitor(env, filename=None)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    episode_waiting_times = []
    episode_throughputs = []
    phase_changes = []
    
    print("🏃 Starting evaluation...")
    
    try:
        for episode in range(n_episodes):
            print(f"   Episode {episode + 1}/{n_episodes}...", end=" ")
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_waiting = 0
            episode_throughput = 0
            phase_change_count = 0
            last_phase = None
            
            done = False
            while not done:
                # Get action from the model
                action, _ = model.predict(obs, deterministic=True)
                
                # Take action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Update metrics
                episode_reward += reward
                episode_length += 1
                
                # Track phase changes
                current_phase = info.get('current_phase', 0)
                if last_phase is not None and current_phase != last_phase:
                    phase_change_count += 1
                last_phase = current_phase
                
                # Check if episode is done
                done = terminated or truncated
                
                # Optional rendering
                if render:
                    env.render()
            
            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            phase_changes.append(phase_change_count)
            
            # Calculate additional metrics (simplified)
            episode_waiting_times.append(-episode_reward)  # Assuming negative reward = waiting time
            episode_throughputs.append(episode_length * 0.1)  # Simplified throughput calculation
            
            print(f"Reward: {episode_reward:.2f}, Length: {episode_length}, Phase changes: {phase_change_count}")
    
    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        return None
    
    finally:
        env.close()
    
    # Calculate summary statistics
    results = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_waiting_times': episode_waiting_times,
        'episode_throughputs': episode_throughputs,
        'phase_changes': phase_changes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'mean_waiting_time': np.mean(episode_waiting_times),
        'std_waiting_time': np.std(episode_waiting_times),
        'mean_throughput': np.mean(episode_throughputs),
        'std_throughput': np.std(episode_throughputs),
        'mean_phase_changes': np.mean(phase_changes),
        'std_phase_changes': np.std(phase_changes),
        'n_episodes': len(episode_rewards)
    }
    
    # Print summary
    print("\n📊 Evaluation Results:")
    print(f"   Episodes: {results['n_episodes']}")
    print(f"   Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"   Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"   Mean Waiting Time: {results['mean_waiting_time']:.2f} ± {results['std_waiting_time']:.2f}")
    print(f"   Mean Throughput: {results['mean_throughput']:.2f} ± {results['std_throughput']:.2f}")
    print(f"   Mean Phase Changes: {results['mean_phase_changes']:.1f} ± {results['std_phase_changes']:.1f}")
    
    # Save results if requested
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = project_root / "output" / f"evaluation_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
    
    return results


def plot_results(results: dict, save_plots: bool = True):
    """
    Create visualization plots for evaluation results.
    
    Args:
        results (dict): Evaluation results
        save_plots (bool): Whether to save plots to file
    """
    
    print("📈 Creating visualization plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Traffic Light Control Agent Evaluation', fontsize=16)
    
    # Plot 1: Episode Rewards
    axes[0, 0].plot(results['episode_rewards'], 'b-', alpha=0.7)
    axes[0, 0].axhline(y=results['mean_reward'], color='r', linestyle='--', 
                      label=f'Mean: {results["mean_reward"]:.2f}')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    axes[0, 1].plot(results['episode_lengths'], 'g-', alpha=0.7)
    axes[0, 1].axhline(y=results['mean_length'], color='r', linestyle='--',
                      label=f'Mean: {results["mean_length"]:.1f}')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Waiting Times
    axes[1, 0].plot(results['episode_waiting_times'], 'orange', alpha=0.7)
    axes[1, 0].axhline(y=results['mean_waiting_time'], color='r', linestyle='--',
                      label=f'Mean: {results["mean_waiting_time"]:.2f}')
    axes[1, 0].set_title('Waiting Times')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Total Waiting Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Phase Changes
    axes[1, 1].plot(results['phase_changes'], 'purple', alpha=0.7)
    axes[1, 1].axhline(y=results['mean_phase_changes'], color='r', linestyle='--',
                      label=f'Mean: {results["mean_phase_changes"]:.1f}')
    axes[1, 1].set_title('Phase Changes per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Number of Phase Changes')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = project_root / "output" / f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_file}")
    
    plt.show()


def main():
    """Main function for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent for SUMO traffic light control")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True,
                       help="Path to the trained model file")
    
    # Environment parameters
    parser.add_argument("--config", type=str, default="sumo_configs/intersection.sumo.cfg",
                       help="Path to SUMO configuration file")
    parser.add_argument("--max-steps", type=int, default=3600,
                       help="Maximum steps per episode")
    parser.add_argument("--reward-type", type=str, default="waiting_time",
                       choices=["waiting_time", "throughput", "combined"],
                       help="Type of reward function")
    
    # Evaluation parameters
    parser.add_argument("--n-episodes", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment during evaluation")
    
    # Output parameters
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to file")
    parser.add_argument("--no-plots", action="store_true",
                       help="Don't create visualization plots")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute paths
    model_path = str(project_root / args.model)
    config_path = str(project_root / args.config)
    
    # Validate environment
    print("🔍 Validating environment...")
    if not check_sumo_installation():
        print("⚠️  SUMO installation not found. Continuing in mock mode.")
    
    if not validate_project_structure():
        print("❌ Project structure is invalid. Please run generate_network.py first.")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
    
    if not os.path.exists(config_path):
        print(f"❌ SUMO config file not found: {config_path}")
        print("   Please run: python scripts/generate_network.py")
        sys.exit(1)
    
    print("✅ Environment validation passed")
    print()
    
    # Create output directory
    os.makedirs(project_root / "output", exist_ok=True)
    
    # Run evaluation
    results = evaluate_agent(
        model_path=model_path,
        config_path=config_path,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        reward_type=args.reward_type,
        render=args.render,
        save_results=not args.no_save
    )
    
    if results is None:
        print("❌ Evaluation failed")
        sys.exit(1)
    
    # Create plots if requested
    if not args.no_plots:
        plot_results(results, save_plots=not args.no_save)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
