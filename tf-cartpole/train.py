import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import numpy as np
import random
from collections import deque

# Disable GPU if you don't have one (optional)
# tf.config.set_visible_devices([], 'GPU')

# Simple neural network for Q-function approximation
def create_dqn_model(state_size, action_size, hidden_size=64):
    model = keras.Sequential([
        keras.layers.Dense(hidden_size, activation='relu', input_shape=(state_size,)),
        keras.layers.Dense(hidden_size, activation='relu'),
        keras.layers.Dense(action_size, activation='linear')  # Linear output for Q-values
    ])
    return model

# Experience replay buffer (same concept as PyTorch version)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), 
                np.array(actions),
                np.array(rewards),
                np.array(next_states),
                np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

# DQN Agent using TensorFlow
class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Create main and target networks
        self.q_network = create_dqn_model(state_size, action_size)
        self.target_network = create_dqn_model(state_size, action_size)
        
        # Compile the main network
        self.q_network.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='mse'
        )
        
        # Experience replay
        self.memory = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        
    def act(self, state):
        # Epsilon-greedy action selection
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Predict Q-values and choose best action
        state_batch = np.expand_dims(state, axis=0)  # Add batch dimension
        q_values = self.q_network.predict(state_batch, verbose=0)
        return np.argmax(q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Current Q-values from main network
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q-values from target network (for stability)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Create target Q-values
        target_q_values = current_q_values.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                # If episode ended, target is just the reward
                target_q_values[i][actions[i]] = rewards[i]
            else:
                # Otherwise, target is reward + discounted max future Q-value
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the main network
        self.q_network.fit(states, target_q_values, batch_size=self.batch_size, 
                          epochs=1, verbose=0)
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        # Copy weights from main network to target network
        self.target_network.set_weights(self.q_network.get_weights())

# Training loop
def train_dqn(render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]  # 4 for CartPole
    action_size = env.action_space.n  # 2 for CartPole
    
    agent = DQNAgent(state_size, action_size)
    episodes = 500
    target_update_freq = 10  # Update target network every 10 episodes
    
    for episode in range(episodes):
        state, _ = env.reset()  # New gym API returns (state, info)
        total_reward = 0
        
        for step in range(500):  # Max steps per episode
            # Agent chooses action
            action = agent.act(state)
            
            # Environment step - new API returns 5 values
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # Episode ends if terminated OR truncated
            
            # Store experience and train
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()
        
        # Print progress
        if episode % 50 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    # Save the trained model
    agent.q_network.save('dqn_cartpole_model.h5')
    print("Model saved as 'dqn_cartpole_model.h5'")
    
    return agent

# Alternative: Using TensorFlow's GradientTape for more control (advanced)
class DQNAgentAdvanced:
    def __init__(self, state_size, action_size, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.q_network = create_dqn_model(state_size, action_size)
        self.target_network = create_dqn_model(state_size, action_size)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        
        self.memory = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99
    
    @tf.function  # Compile for faster execution
    def train_step(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            # Current Q-values
            current_q_values = self.q_network(states)
            current_q_values = tf.gather_nd(current_q_values, 
                                          tf.stack([tf.range(tf.shape(actions)[0]), actions], axis=1))
            
            # Target Q-values
            next_q_values = tf.reduce_max(self.target_network(next_states), axis=1)
            target_q_values = rewards + (1.0 - tf.cast(dones, tf.float32)) * self.gamma * next_q_values
            
            # Loss
            loss = tf.reduce_mean(tf.square(target_q_values - current_q_values))
        
        # Apply gradients
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        return loss

# Run training
if __name__ == "__main__":
    trained_agent = train_dqn()
