import torch
import numpy as np
import time
from maze_env import RobotMazeEnv
from drive_robot import NavigationPolicy

def load_and_run_model(model_path, max_steps=1000, verbose=True):
    """
    Load a trained navigation model and run the robot with it.
    
    Args:
        model_path (str): Path to the saved model
        max_steps (int): Maximum number of steps to take in an episode
        verbose (bool): Whether to print detailed information
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create the environment
    env = RobotMazeEnv(maze=None)
    
    # Create the policy network
    policy = NavigationPolicy(num_actions=4).to(device)
    
    # Load the model weights
    try:
        policy.load_state_dict(torch.load(model_path, map_location=device))
        if verbose:
            print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set policy to evaluation mode
    policy.eval()
    
    # Run an episode
    state = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    initial_distance = np.linalg.norm(state[:2] - env.goal_position[:2])
    
    if verbose:
        print(f"Starting position: ({state[0]:.2f}, {state[1]:.2f}), Yaw: {np.degrees(state[2]):.2f}°")
        print(f"Goal position: ({env.goal_position[0]:.2f}, {env.goal_position[1]:.2f})")
        print(f"Initial distance to goal: {initial_distance:.2f} m")
        print("Beginning navigation...")
    
    # Main loop
    while not done and step_count < max_steps:
        # Preprocess state
        gps = torch.tensor(state[:2], dtype=torch.float32).to(device)
        yaw = torch.tensor(state[2], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Get action from policy
        with torch.no_grad():
            action_logits, _ = policy(gps, yaw)
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
            action = torch.argmax(action_probs).item()
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Print status
        if verbose and step_count % 10 == 0:
            distance = info["distance"]
            action_names = ["Forward", "Backward", "Turn Left", "Turn Right"]
            print(f"Step {step_count}: Action={action_names[action]}, " +
                  f"Position=({next_state[0]:.2f}, {next_state[1]:.2f}), " +
                  f"Distance to goal={distance:.2f} m")
        
        # Update for next iteration
        state = next_state
        total_reward += reward
        step_count += 1
        
        # Small delay to better visualize in Webots (can be removed for faster execution)
        time.sleep(0.01)
    
    # Calculate metrics
    final_distance = np.linalg.norm(state[:2] - env.goal_position[:2])
    success = final_distance < env.GOAL_THRESHOLD
    
    # Print results
    if verbose:
        print("\n--- Episode Results ---")
        print(f"Steps taken: {step_count}")
        print(f"Final distance to goal: {final_distance:.2f} m")
        print(f"Total reward: {total_reward:.2f}")
        print(f"Success: {'Yes' if success else 'No'}")
        if success:
            spl = min(1.0, initial_distance / max(step_count, initial_distance))
            print(f"SPL (Success weighted by Path Length): {spl:.4f}")
    
    return success, step_count, total_reward

if __name__ == "__main__":
    # Path to the trained model
    model_path = r"../drive_robot/output/20250302_140715/navigation_policy_best_world_0.pt"
    
    print(f"Starting robot navigation with trained model {model_path}")
    success, steps, reward = load_and_run_model(model_path)
    
    # Run multiple episodes to evaluate performance
    if input("\nRun evaluation over multiple episodes? (y/n): ").lower() == 'y':
        num_episodes = int(input("Enter number of episodes: "))
        
        successes = 0
        total_steps = 0
        total_rewards = 0
        
        print(f"\nRunning {num_episodes} evaluation episodes...")
        for episode in range(num_episodes):
            print(f"\nEpisode {episode+1}/{num_episodes}")
            episode_success, episode_steps, episode_reward = load_and_run_model(model_path, verbose=False)
            
            successes += int(episode_success)
            total_steps += episode_steps
            total_rewards += episode_reward
            
            print(f"Episode {episode+1}: Success={'Yes' if episode_success else 'No'}, Steps={episode_steps}, Reward={episode_reward:.2f}")
        
        # Print evaluation statistics
        print("\n--- Evaluation Results ---")
        print(f"Success rate: {successes/num_episodes:.2f} ({successes}/{num_episodes})")
        print(f"Average steps: {total_steps/num_episodes:.2f}")
        print(f"Average reward: {total_rewards/num_episodes:.2f}")