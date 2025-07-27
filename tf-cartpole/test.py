import gymnasium as gym
import numpy as np
from tensorflow import keras
from train import DQNAgent
import os
os.environ['SDL_VIDEODRIVER'] = 'cocoa'  # For macOS

def test_agent(agent, episodes=10, render=True):
    """Test the trained agent's performance"""
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    total_rewards = []
    
    # Disable exploration for testing
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for step in range(500):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        total_rewards.append(total_reward)
        print(f"Test Episode {episode + 1}: Score = {total_reward}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    env.close()
    
    avg_reward = np.mean(total_rewards)
    print(f"\nAverage score over {episodes} episodes: {avg_reward:.2f}")
    print(f"Best score: {max(total_rewards)}")
    print(f"All scores: {total_rewards}")
    
    # CartPole is considered solved if avg score >= 195 over 100 episodes
    if avg_reward >= 195:
        print("🎉 Agent has solved CartPole! (avg score >= 195)")
    
    return avg_reward

def load_trained_agent(model_path='dqn_cartpole_model.h5'):
    """Load a trained model for testing"""
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()
    
    agent = DQNAgent(state_size, action_size)
    try:
        agent.q_network.load_weights(model_path)
    except:
        # If weights loading fails, try loading the full model
        agent.q_network = keras.models.load_model(model_path, compile=False)
    agent.epsilon = 0.0  # No exploration for testing
    
    print(f"Model loaded from '{model_path}'")
    return agent

if __name__ == "__main__":
    print("Loading trained agent...")
    trained_agent = load_trained_agent()
    
    print("\nTesting with visual rendering...")
    test_agent(trained_agent, episodes=3, render=True)
