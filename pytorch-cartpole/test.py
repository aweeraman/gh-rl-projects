import torch
import gymnasium as gym
import numpy as np
from models import DQNAgent

def test_agent(agent, episodes=5, render=True):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state, training=False)  # No exploration
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        scores.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    env.close()
    print(f"Average test score: {np.mean(scores):.2f}")
    return scores

def load_trained_agent(model_path='cartpole_dqn.pth'):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()
    
    agent = DQNAgent(state_size, action_size)
    agent.q_network.load_state_dict(torch.load(model_path, map_location=agent.device))
    agent.q_network.eval()
    agent.epsilon = 0  # No exploration during testing
    
    return agent

if __name__ == "__main__":
    print("Loading trained agent...")
    try:
        agent = load_trained_agent()
        print("Testing trained agent with visual rendering...")
        test_scores = test_agent(agent, episodes=5, render=True)
    except FileNotFoundError:
        print("No trained model found. Please run train.py first to create 'cartpole_dqn.pth'")