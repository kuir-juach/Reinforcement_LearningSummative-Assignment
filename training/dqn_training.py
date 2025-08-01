import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import ExamGradingEnv

def train_dqn():
    """Train DQN agent for exam grading task"""
    
    # Create environment
    env = ExamGradingEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = ExamGradingEnv()
    eval_env = Monitor(eval_env)
    
    # DQN hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0005,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log="./tensorboard_logs/dqn/",
        verbose=1
    )
    
    # Callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=150, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=5000,
        best_model_save_path="./models/dqn/",
        log_path="./models/dqn/",
        verbose=1
    )
    
    # Train the model
    print("Starting DQN training...")
    model.learn(
        total_timesteps=100000,
        callback=eval_callback,
        log_interval=100
    )
    
    # Save final model
    model.save("./models/dqn/dqn_final")
    print("DQN training completed!")
    
    return model

def test_dqn(model_path="./models/dqn/dqn_final"):
    """Test trained DQN model"""
    
    # Load model
    model = DQN.load(model_path)
    
    # Create test environment
    env = ExamGradingEnv()
    
    # Test for multiple episodes
    total_rewards = []
    accuracies = []
    
    for episode in range(10):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        print(f"\n=== Episode {episode + 1} ===")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            env.render()
        
        total_rewards.append(episode_reward)
        accuracies.append(env.accuracy_history)
        
        print(f"Episode {episode + 1} - Reward: {episode_reward:.2f}, Accuracy: {env.accuracy_history:.2%}")
    
    print(f"\nDQN Test Results:")
    print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Average Accuracy: {sum(accuracies)/len(accuracies):.2%}")
    
    return total_rewards, accuracies

if __name__ == "__main__":
    # Create directories
    os.makedirs("./models/dqn", exist_ok=True)
    os.makedirs("./tensorboard_logs/dqn", exist_ok=True)
    
    # Train model
    trained_model = train_dqn()
    
    # Test model
    test_dqn()