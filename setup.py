from setuptools import setup, find_packages

setup(
    name="traffic-rl-sumo",
    version="0.1.0",
    description="PPO agent for traffic light control in SUMO simulation",
    author="Traffic RL Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "stable-baselines3>=2.2.1",
        "gymnasium>=0.29.1",
        "numpy>=1.24.3",
        "matplotlib>=3.7.2",
        "tensorboard>=2.14.1",
    ],
    entry_points={
        "console_scripts": [
            "traffic-rl-generate=scripts.generate_network:main",
            "traffic-rl-train=scripts.train:main",
            "traffic-rl-eval=scripts.evaluate:main",
            "traffic-rl-gui=scripts.launch_gui:main",
        ],
    },
)
