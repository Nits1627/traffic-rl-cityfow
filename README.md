# Traffic RL SUMO

A reinforcement learning project for traffic light control using SUMO simulation and PPO (Proximal Policy Optimization) from Stable-Baselines3.

## 🚦 Overview

This project trains a PPO agent to control a single traffic light at a 1×1 intersection in SUMO (Simulation of Urban Mobility). The agent learns to optimize traffic flow by minimizing waiting times and maximizing throughput.

### Features

- **Simple 1×1 Intersection**: Easy to understand and visualize
- **PPO Agent**: Uses Stable-Baselines3 for robust RL training
- **SUMO Integration**: Direct integration with SUMO simulation
- **Multiple Reward Functions**: Waiting time, throughput, or combined rewards
- **Visualization**: SUMO GUI for real-time visualization
- **Evaluation Tools**: Comprehensive evaluation and plotting
- **macOS Optimized**: Designed for macOS with official SUMO installer

## 📋 Requirements

### System Requirements
- **OS**: macOS (Apple Silicon or Intel)
- **Python**: 3.10 or 3.11
- **SUMO**: Official .pkg installer (not Homebrew)

### Software Installation

#### 1. Install SUMO

Download and install SUMO from the [official website](https://eclipse.org/sumo/):

1. Go to https://eclipse.org/sumo/
2. Download the macOS .pkg installer
3. Install the package
4. Set environment variables:

```bash
# Add to your ~/.zshrc or ~/.bash_profile
export SUMO_HOME="/Library/Frameworks/sumo.framework/Versions/Current"
export PATH="$SUMO_HOME/bin:$PATH"
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

#### 2. Verify SUMO Installation

```bash
# Check SUMO installation
sumo --version
sumo-gui --version
```

#### 3. Install Python Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Generate SUMO Network

```bash
# Generate the 1x1 intersection network
python scripts/generate_network.py
```

This creates:
- `network/intersection.net.xml` - Road network
- `routes/intersection.rou.xml` - Vehicle routes
- `sumo_configs/intersection.sumo.cfg` - SUMO configuration

### 2. Train the Agent

```bash
# Train PPO agent (default: 100k timesteps)
python scripts/train.py

# Custom training parameters
python scripts/train.py \
    --total-timesteps 200000 \
    --learning-rate 1e-4 \
    --batch-size 128 \
    --reward-type waiting_time
```

### 3. Evaluate the Agent

```bash
# Evaluate trained agent
python scripts/evaluate.py --model models/ppo_traffic_light/final_model.zip

# Evaluate with custom parameters
python scripts/evaluate.py \
    --model models/ppo_traffic_light/final_model.zip \
    --n-episodes 20 \
    --render
```

### 4. Visualize with SUMO GUI

```bash
# Launch SUMO GUI (manual control)
python scripts/launch_gui.py

# Launch with agent control
python scripts/launch_gui.py --with-agent --model models/ppo_traffic_light/final_model.zip
```

## 📁 Project Structure

```
traffic-rl-sumo/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.py                 # Package setup
├── .gitignore              # Git ignore rules
├── scripts/                # Executable scripts
│   ├── generate_network.py # Generate SUMO network
│   ├── train.py           # Train PPO agent
│   ├── evaluate.py        # Evaluate trained agent
│   └── launch_gui.py      # Launch SUMO GUI
├── traffic_rl/            # Main package
│   ├── __init__.py
│   ├── env.py            # Custom Gymnasium environment
│   └── utils.py          # Utility functions
├── network/               # Generated network files
├── routes/               # Generated route files
├── sumo_configs/         # SUMO configuration files
├── models/               # Trained models
├── logs/                 # Training logs
└── output/               # Evaluation results and plots
```

## 🔧 Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--total-timesteps` | 100000 | Total training timesteps |
| `--learning-rate` | 3e-4 | Learning rate |
| `--batch-size` | 64 | Batch size |
| `--n-steps` | 2048 | Steps per update |
| `--n-epochs` | 10 | Epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--reward-type` | waiting_time | Reward function type |

### Reward Types

- **`waiting_time`**: Negative reward based on total waiting time
- **`throughput`**: Positive reward based on vehicles passed
- **`combined`**: Combination of throughput and waiting time

### Environment Parameters

- **Max Steps**: 3600 (1 hour at 1 step/second)
- **Yellow Time**: 3 seconds
- **Min Green Time**: 10 seconds
- **Traffic Light Phases**: 4 phases (N-S green, N-S yellow, E-W green, E-W yellow)

## 📊 Monitoring Training

### Tensorboard

```bash
# View training progress
tensorboard --logdir logs/tensorboard

# Open in browser: http://localhost:6006
```

### Training Output

The training script provides real-time feedback:
- Episode rewards and lengths
- Evaluation results
- Model checkpoints
- Progress bars

## 🎯 Evaluation Metrics

The evaluation script provides comprehensive metrics:

- **Episode Rewards**: Total reward per episode
- **Episode Lengths**: Number of steps per episode
- **Waiting Times**: Total waiting time per episode
- **Throughput**: Vehicles processed per episode
- **Phase Changes**: Number of traffic light phase changes

### Visualization

Evaluation results include:
- Line plots for all metrics
- Statistical summaries (mean, std)
- JSON export for further analysis

## 🐛 Troubleshooting

### Common Issues

#### 1. SUMO Not Found
```
❌ SUMO_HOME environment variable not set
```
**Solution**: Set SUMO_HOME environment variable:
```bash
export SUMO_HOME="/Library/Frameworks/sumo.framework/Versions/Current"
```

#### 2. SUMO Installation Issues
```
❌ SUMO bin directory not found
```
**Solution**: Reinstall SUMO using the official .pkg installer, not Homebrew.

#### 3. Python Import Errors
```
ModuleNotFoundError: No module named 'traci'
```
**Solution**: Ensure PYTHONPATH includes SUMO tools:
```bash
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```

#### 4. Network Files Missing
```
❌ SUMO config file not found
```
**Solution**: Generate network files first:
```bash
python scripts/generate_network.py
```

### Debug Mode

Enable verbose output for debugging:
```bash
# Training with debug info
python scripts/train.py --config sumo_configs/intersection.sumo.cfg --verbose

# Evaluation with debug info
python scripts/evaluate.py --model models/ppo_traffic_light/final_model.zip --render
```

## 📈 Performance Tips

### Training Optimization

1. **Increase Training Time**: Use more timesteps for better performance
2. **Tune Hyperparameters**: Experiment with learning rate, batch size
3. **Reward Engineering**: Try different reward functions
4. **Environment Tuning**: Adjust traffic flow rates

### Example High-Performance Training

```bash
python scripts/train.py \
    --total-timesteps 500000 \
    --learning-rate 1e-4 \
    --batch-size 128 \
    --n-steps 4096 \
    --reward-type combined
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [SUMO](https://eclipse.org/sumo/) - Traffic simulation
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - RL algorithms
- [Gymnasium](https://gymnasium.farama.org/) - RL environment interface

## 📚 References

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

**Happy Training! 🚦🤖**
