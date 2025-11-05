#!/usr/bin/env python3
"""
Example usage script for Traffic RL SUMO.

This script demonstrates how to use the traffic light control environment
and train a simple agent.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def example_environment_usage():
    """Example of using the SUMO traffic light environment."""
    print("🤖 Example: Using SUMO Traffic Light Environment")
    print("-" * 50)
    
    try:
        from traffic_rl.env import SumoTrafficLightEnv
        
        # Create environment
        config_path = project_root / "sumo_configs" / "intersection.sumo.cfg"
        env = SumoTrafficLightEnv(
            config_path=str(config_path),
            max_steps=100,  # Short example
            reward_type="waiting_time"
        )
        
        print("✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")
        
        # Run a few random steps
        obs, info = env.reset()
        print(f"\n🎯 Initial observation shape: {obs.shape}")
        
        total_reward = 0
        for step in range(10):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {step + 1}: Action={action}, Reward={reward:.2f}, Phase={info.get('current_phase', 0)}")
            
            if terminated or truncated:
                break
        
        print(f"\n📊 Total reward over {step + 1} steps: {total_reward:.2f}")
        
        env.close()
        print("✅ Environment closed")
        
    except Exception as e:
        print(f"❌ Error in environment example: {e}")
        return False
    
    return True

def example_training():
    """Example of training a PPO agent."""
    print("\n🏋️ Example: Training PPO Agent")
    print("-" * 50)
    
    try:
        from stable_baselines3 import PPO
        from traffic_rl.env import SumoTrafficLightEnv
        
        # Create environment
        config_path = project_root / "sumo_configs" / "intersection.sumo.cfg"
        env = SumoTrafficLightEnv(
            config_path=str(config_path),
            max_steps=200,  # Short training
            reward_type="waiting_time"
        )
        
        print("✅ Environment created for training")
        
        # Create PPO model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=64,  # Small for quick example
            batch_size=32,
            verbose=1
        )
        
        print("✅ PPO model created")
        print("   Training for 1000 timesteps...")
        
        # Train the model
        model.learn(total_timesteps=1000, progress_bar=True)
        
        print("✅ Training completed!")
        
        # Test the trained model
        print("\n🧪 Testing trained model...")
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(20):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 5 == 0:
                print(f"   Step {step}: Action={action}, Reward={reward:.2f}")
            
            if terminated or truncated:
                break
        
        print(f"📊 Test reward: {total_reward:.2f}")
        
        # Save model
        model_path = project_root / "models" / "example_model"
        model.save(str(model_path))
        print(f"💾 Model saved to: {model_path}")
        
        env.close()
        print("✅ Training example completed")
        
    except Exception as e:
        print(f"❌ Error in training example: {e}")
        return False
    
    return True

def example_evaluation():
    """Example of evaluating a trained model."""
    print("\n📊 Example: Evaluating Trained Model")
    print("-" * 50)
    
    try:
        from stable_baselines3 import PPO
        from traffic_rl.env import SumoTrafficLightEnv
        
        # Load trained model
        model_path = project_root / "models" / "example_model.zip"
        if not model_path.exists():
            print("❌ No trained model found. Run training example first.")
            return False
        
        model = PPO.load(str(model_path))
        print("✅ Model loaded successfully")
        
        # Create environment
        config_path = project_root / "sumo_configs" / "intersection.sumo.cfg"
        env = SumoTrafficLightEnv(
            config_path=str(config_path),
            max_steps=100,
            reward_type="waiting_time"
        )
        
        # Evaluate model
        print("🔍 Evaluating model...")
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        while step_count < 50:  # Short evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"   Step {step_count}: Action={action}, Reward={reward:.2f}, Phase={info.get('current_phase', 0)}")
            
            if terminated or truncated:
                break
        
        print(f"📊 Evaluation complete:")
        print(f"   Steps: {step_count}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Average reward: {total_reward/step_count:.2f}")
        
        env.close()
        print("✅ Evaluation example completed")
        
    except Exception as e:
        print(f"❌ Error in evaluation example: {e}")
        return False
    
    return True

def main():
    """Run all examples."""
    print("🚦 Traffic RL SUMO - Example Usage")
    print("=" * 60)
    
    # Check if network files exist
    config_path = project_root / "sumo_configs" / "intersection.sumo.cfg"
    if not config_path.exists():
        print("❌ SUMO network files not found.")
        print("   Please run: python scripts/generate_network.py")
        return 1
    
    # Run examples
    examples = [
        ("Environment Usage", example_environment_usage),
        ("Training", example_training),
        ("Evaluation", example_evaluation),
    ]
    
    results = []
    for example_name, example_func in examples:
        try:
            result = example_func()
            results.append((example_name, result))
        except KeyboardInterrupt:
            print(f"\n⏹️ {example_name} interrupted by user")
            results.append((example_name, False))
            break
        except Exception as e:
            print(f"\n❌ {example_name} failed: {e}")
            results.append((example_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Example Summary:")
    
    passed = 0
    for example_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {example_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} examples completed")
    
    if passed == len(results):
        print("\n🎉 All examples completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Run full training: python scripts/train.py")
        print("   2. Evaluate model: python scripts/evaluate.py --model models/ppo_traffic_light/final_model.zip")
        print("   3. Launch GUI: python scripts/launch_gui.py")
    else:
        print(f"\n⚠️  {len(results) - passed} example(s) failed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
