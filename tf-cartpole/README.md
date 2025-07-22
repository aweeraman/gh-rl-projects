# TensorFlow CartPole DQN

Deep Q-Network (DQN) implementation for the CartPole environment using TensorFlow and Gymnasium.

## Overview

This project implements a DQN agent that learns to balance a pole on a cart using reinforcement learning. The agent uses experience replay and a target network for stable training.

## Requirements

- Python 3.8+
- TensorFlow
- Gymnasium
- NumPy

## Installation

```bash
uv sync
```

## Usage

Run the training:

```bash
uv run cartpole.py
```

The agent will train for 500 episodes, printing progress every 50 episodes.

## Implementation Details

- **DQN Agent**: Uses a simple neural network with two hidden layers (64 units each)
- **Experience Replay**: 10,000 capacity buffer for storing and sampling experiences
- **Target Network**: Updated every 10 episodes for training stability
- **Epsilon-Greedy**: Exploration rate starts at 1.0 and decays to 0.01

## Files

- `cartpole.py`: Main implementation with DQN agent and training loop