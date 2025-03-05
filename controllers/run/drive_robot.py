# drive_robot.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import time
import os
from torch.distributions import Categorical
from datetime import datetime
from maze_env import RobotMazeEnv
import json 

class NavigationPolicy(nn.Module):
    def __init__(self, num_actions=4):
        super().__init__()
        
        # Enhanced network with more capacity
        self.network = nn.Sequential(
            nn.Linear(4, 128),  # Input: GPS (x,y) + sin(yaw), cos(yaw)
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)
        
    def forward(self, gps, yaw):
        # Process GPS and compass data
        yaw_sin = torch.sin(yaw)
        yaw_cos = torch.cos(yaw)
        x = torch.cat([gps, yaw_sin, yaw_cos], dim=-1)
        
        # Forward pass
        features = self.network(x)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        
        return action_logits, value

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class PointGoalNavigator:
    def __init__(self, env, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.env = env
        self.device = device
        self.policy = NavigationPolicy(num_actions=4).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # PPO hyperparameters
        self.n_steps = 128  # steps per update
        self.gamma = 0.99   # discount factor
        self.gae_lambda = 0.95  # GAE parameter
        self.policy_clip = 0.2  # clip parameter for PPO
        self.n_epochs = 4   # number of mini-batch updates
        self.memory = PPOMemory()
        
        # Training metrics
        self.episode_rewards = []
        self.success_rate = deque(maxlen=100)
        self.spl_values = deque(maxlen=100)
        
        # Goal distance for reward shaping
        self.prev_distance = None
        
        # Setup output directory for reward logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("output", timestamp)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Created output directory: {self.output_dir}")
        
    def preprocess_state(self, state):
        # Extract GPS and yaw from state
        gps = torch.tensor(state[:2], dtype=torch.float32).to(self.device)
        yaw = torch.tensor(state[2], dtype=torch.float32).unsqueeze(0).to(self.device)
        return gps, yaw
    
    def choose_action(self, state):
        gps, yaw = self.preprocess_state(state)
        
        with torch.no_grad():
            action_logits, value = self.policy(gps, yaw)
            action_probs = F.softmax(action_logits, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            
        return action.item(), action_log_prob.item(), value.item()
    
    def learn(self):
        for _ in range(self.n_epochs):
            # Convert memory to tensors
            states = self.memory.states
            actions = torch.tensor(self.memory.actions, dtype=torch.long).to(self.device)
            old_probs = torch.tensor(self.memory.probs, dtype=torch.float32).to(self.device)
            vals = torch.tensor(self.memory.vals, dtype=torch.float32).to(self.device)
            
            # Calculate advantages using GAE
            rewards = []
            discounted_reward = 0
            for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
                if done:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward * (1-done))
                rewards.insert(0, discounted_reward)
            
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            
            # Normalize rewards
            if len(rewards) > 1:
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            
            # Compute advantages
            advantages = rewards - vals
            
            # Perform mini-batch optimization
            for i in range(0, len(states), self.n_steps):
                batch_states = states[i:i+self.n_steps]
                batch_actions = actions[i:i+self.n_steps]
                batch_old_probs = old_probs[i:i+self.n_steps]
                batch_advantages = advantages[i:i+self.n_steps]
                
                # Process batch of states
                batch_gps = []
                batch_yaw = []
                for state in batch_states:
                    gps, yaw = self.preprocess_state(state)
                    batch_gps.append(gps)
                    batch_yaw.append(yaw)
                
                batch_gps = torch.stack(batch_gps)
                batch_yaw = torch.stack(batch_yaw)
                
                # Get new action probs and values
                action_logits, values = self.policy(batch_gps, batch_yaw)
                action_probs = F.softmax(action_logits, dim=-1)
                dist = Categorical(action_probs)
                new_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio for PPO
                ratio = torch.exp(new_probs - batch_old_probs)
                
                # PPO loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1-self.policy_clip, 1+self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                critic_loss = F.mse_loss(values.squeeze(-1), rewards[i:i+self.n_steps])
                
                # Total loss
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
                
                # Update policy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
        
        self.memory.clear_memory()
    
    def train_episode(self):
        state = self.env.reset()
        done = False
        episode_reward = 0
        path_length = 0
        
        # Initial distance from GPS
        initial_distance = np.linalg.norm(state[:2] - self.env.goal_position[:2])
        self.prev_distance = initial_distance
        
        steps_since_update = 0
        
        while not done and path_length < 500:  # Reduced max steps
            # Get action from policy
            action, prob, val = self.choose_action(state)
            
            # Take action
            next_state, reward, done, info = self.env.step(action)
            
            # Reward shaping: add progress reward
            current_distance = info["distance"]
            progress_reward = (self.prev_distance - current_distance) * 2.0  # Scaling factor
            shaped_reward = reward + progress_reward
            self.prev_distance = current_distance
            
            # Add collision penalty if agent is stuck
            if path_length > 0 and np.linalg.norm(np.array(state[:2]) - np.array(next_state[:2])) < 0.01:
                shaped_reward -= 0.5  # Small penalty for not moving
            
            # Store in memory
            self.memory.store_memory(state, action, prob, val, shaped_reward, done)
            
            # Update
            state = next_state
            episode_reward += reward  # Use original reward for tracking
            path_length += 1
            steps_since_update += 1
            
            # Update policy every n_steps or at episode end
            if steps_since_update >= self.n_steps or done:
                self.learn()
                steps_since_update = 0
        
        # Calculate metrics
        final_distance = info["distance"]
        success = final_distance < self.env.GOAL_THRESHOLD
        spl = self.compute_spl(success, path_length, initial_distance)
        
        # Update metrics
        self.episode_rewards.append(episode_reward)
        self.success_rate.append(float(success))
        self.spl_values.append(spl)
        
        return episode_reward, success, spl, path_length
    
    def compute_spl(self, success, path_length, shortest_path_length):
        if not success:
            return 0
        return min(1.0, shortest_path_length / max(path_length, shortest_path_length))
    
    def save_rewards(self):
        """Save the reward history to a file"""
        rewards_file = os.path.join(self.output_dir, "rewards.txt")
        with open(rewards_file, 'w') as f:
            json.dump(self.episode_rewards, f)
        print(f"Saved rewards to {rewards_file}")
    
    def train(self, num_episodes=10000):
        best_success_rate = 0
        episode = 0
        
        while True:
            reward, success, spl, path_length = self.train_episode()
            
            # Compute metrics
            avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
            avg_success = np.mean(list(self.success_rate)) if self.success_rate else 0
            avg_spl = np.mean(list(self.spl_values)) if self.spl_values else 0
            
            # Print metrics
            if episode % 10 == 0:
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                      f"Episode {episode}", 
                      f"Reward: {reward:.2f}", 
                      f"Avg Reward: {avg_reward:.2f}", 
                      f"Success Rate: {avg_success:.4f}", 
                      f"SPL: {avg_spl:.2f}",
                      f"Path Length: {path_length}")
            
            # Save model if improved
            if avg_success > best_success_rate and episode > 100:
                best_success_rate = avg_success
                model_path = os.path.join(self.output_dir, 'navigation_policy_best.pt')
                torch.save(self.policy.state_dict(), model_path)
                print(f"Saved best model with success rate: {best_success_rate:.4f}")
            
            # Save rewards periodically
            if (episode + 1) % 1000 == 0:
                self.save_rewards()
                print(f"Saved rewards at episode {episode + 1}")
            
            episode += 1
            
            # Early stopping if mastered
            if avg_success > 0.95 and avg_spl > 0.8:
                print("Environment mastered! Stopping training.")
                self.save_rewards()  # Save final rewards
                break
        
        # Save final rewards if not already saved
        if episode % 1000 != 0:
            self.save_rewards()
    
    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
    
    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))

if __name__ == "__main__":
    # Create environment
    env = RobotMazeEnv(maze=None)
    
    # Create and train navigator
    navigator = PointGoalNavigator(env)
    
    print("Starting training...")
    navigator.train(num_episodes=5000)
    
    # Save final model
    final_model_path = os.path.join(navigator.output_dir, 'navigation_policy_final.pt')
    navigator.save_model(final_model_path)
    print(f"Training completed! Model saved to {final_model_path}")