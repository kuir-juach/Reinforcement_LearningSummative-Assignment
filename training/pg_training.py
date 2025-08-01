import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.custom_env import ExamGradingEnv

def train_ppo():
    """Train PPO agent for exam grading task"""
    
    # Create environment
    env = ExamGradingEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = ExamGradingEnv()
    eval_env = Monitor(eval_env)
    
    # PPO hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/ppo/",
        verbose=1
    )
    
    # Callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=150, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=10000,
        best_model_save_path="./models/pg/ppo/",
        log_path="./models/pg/ppo/",
        verbose=1
    )
    
    # Train the model
    print("Starting PPO training...")
    model.learn(
        total_timesteps=200000,
        callback=eval_callback,
        log_interval=10
    )
    
    # Save final model
    model.save("./models/pg/ppo/ppo_final")
    print("PPO training completed!")
    
    return model

def train_a2c():
    """Train A2C (Actor-Critic) agent for exam grading task"""
    
    # Create environment
    env = ExamGradingEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = ExamGradingEnv()
    eval_env = Monitor(eval_env)
    
    # A2C hyperparameters
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        ent_coef=0.01,
        vf_coef=0.25,
        max_grad_norm=0.5,
        rms_prop_eps=1e-05,
        use_rms_prop=True,
        normalize_advantage=False,
        tensorboard_log="./tensorboard_logs/a2c/",
        verbose=1
    )
    
    # Callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=150, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=10000,
        best_model_save_path="./models/pg/a2c/",
        log_path="./models/pg/a2c/",
        verbose=1
    )
    
    # Train the model
    print("Starting A2C training...")
    model.learn(
        total_timesteps=150000,
        callback=eval_callback,
        log_interval=10
    )
    
    # Save final model
    model.save("./models/pg/a2c/a2c_final")
    print("A2C training completed!")
    
    return model

class REINFORCEAgent:
    """Simple REINFORCE implementation"""
    
    def __init__(self, env, learning_rate=0.01):
        self.env = env
        self.lr = learning_rate
        self.policy_history = []
        self.reward_history = []
        
        # Simple neural network weights (for demonstration)
        self.weights = np.random.randn(env.observation_space.shape[0], env.action_space.n) * 0.1
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def choose_action(self, state):
        """Choose action using policy"""
        logits = np.dot(state, self.weights)
        probs = self.softmax(logits)
        action = np.random.choice(len(probs), p=probs)
        self.policy_history.append((state, action, probs[action]))
        return action
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm"""
        if not self.policy_history or not self.reward_history:
            return
        
        # Calculate discounted rewards
        discounted_rewards = []
        cumulative = 0
        for reward in reversed(self.reward_history):
            cumulative = reward + 0.99 * cumulative
            discounted_rewards.insert(0, cumulative)
        
        # Normalize rewards
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / (np.std(discounted_rewards) + 1e-8)
        
        # Update weights
        for i, (state, action, prob) in enumerate(self.policy_history):
            # Gradient ascent
            gradient = np.zeros_like(self.weights)
            gradient[:, action] = state * (1 - prob) * discounted_rewards[i]
            self.weights += self.lr * gradient
        
        # Clear history
        self.policy_history = []
        self.reward_history = []

def train_reinforce():
    """Train REINFORCE agent"""
    
    env = ExamGradingEnv()
    agent = REINFORCEAgent(env)
    
    episode_rewards = []
    episode_accuracies = []
    
    print("Starting REINFORCE training...")
    
    for episode in range(1000):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.reward_history.append(reward)
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        
        # Update policy after each episode
        agent.update_policy()
        
        episode_rewards.append(episode_reward)
        episode_accuracies.append(env.accuracy_history)
        
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_accuracy = np.mean(episode_accuracies[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Accuracy: {avg_accuracy:.2%}")
    
    print("REINFORCE training completed!")
    
    # Save agent weights
    os.makedirs("./models/pg/reinforce", exist_ok=True)
    np.save("./models/pg/reinforce/weights.npy", agent.weights)
    
    return agent, episode_rewards, episode_accuracies

def test_policy_gradient(algorithm="ppo", model_path=None):
    """Test trained policy gradient model"""
    
    env = ExamGradingEnv()
    
    if algorithm == "reinforce":
        # Load REINFORCE weights
        weights = np.load("./models/pg/reinforce/weights.npy")
        agent = REINFORCEAgent(env)
        agent.weights = weights
        
        total_rewards = []
        accuracies = []
        
        for episode in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            print(f"\n=== REINFORCE Episode {episode + 1} ===")
            
            while not done:
                action = agent.choose_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                env.render()
            
            total_rewards.append(episode_reward)
            accuracies.append(env.accuracy_history)
            
            print(f"Episode {episode + 1} - Reward: {episode_reward:.2f}, Accuracy: {env.accuracy_history:.2%}")
    
    else:
        # Load SB3 model
        if algorithm == "ppo":
            model_path = model_path or "./models/pg/ppo/ppo_final"
            model = PPO.load(model_path)
        elif algorithm == "a2c":
            model_path = model_path or "./models/pg/a2c/a2c_final"
            model = A2C.load(model_path)
        
        total_rewards = []
        accuracies = []
        
        for episode in range(10):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            print(f"\n=== {algorithm.upper()} Episode {episode + 1} ===")
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                
                env.render()
            
            total_rewards.append(episode_reward)
            accuracies.append(env.accuracy_history)
            
            print(f"Episode {episode + 1} - Reward: {episode_reward:.2f}, Accuracy: {env.accuracy_history:.2%}")
    
    print(f"\n{algorithm.upper()} Test Results:")
    print(f"Average Reward: {sum(total_rewards)/len(total_rewards):.2f}")
    print(f"Average Accuracy: {sum(accuracies)/len(accuracies):.2%}")
    
    return total_rewards, accuracies

if __name__ == "__main__":
    # Create directories
    os.makedirs("./models/pg/ppo", exist_ok=True)
    os.makedirs("./models/pg/a2c", exist_ok=True)
    os.makedirs("./models/pg/reinforce", exist_ok=True)
    os.makedirs("./tensorboard_logs/ppo", exist_ok=True)
    os.makedirs("./tensorboard_logs/a2c", exist_ok=True)
    
    # Train all policy gradient methods
    print("Training all Policy Gradient methods...")
    
    # Train PPO
    ppo_model = train_ppo()
    
    # Train A2C
    a2c_model = train_a2c()
    
    # Train REINFORCE
    reinforce_agent, rewards, accuracies = train_reinforce()
    
    # Test all models
    print("\nTesting all models...")
    test_policy_gradient("ppo")
    test_policy_gradient("a2c")
    test_policy_gradient("reinforce")