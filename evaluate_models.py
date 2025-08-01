import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from environment.custom_env import ExamGradingEnv
import os

def evaluate_all_models():
    """Comprehensive evaluation of all trained models"""
    
    results = {}
    algorithms = ['DQN', 'PPO', 'A2C', 'REINFORCE']
    
    print("=== Model Evaluation Report ===")
    print("South Sudan NLP Exam Grading System")
    print("=" * 50)
    
    for algorithm in algorithms:
        print(f"\nEvaluating {algorithm}...")
        
        try:
            # Load and test each model
            if algorithm == 'DQN':
                from stable_baselines3 import DQN
                model = DQN.load("./models/dqn/dqn_final")
                rewards, accuracies, times = test_model(model, algorithm)
            elif algorithm == 'PPO':
                from stable_baselines3 import PPO
                model = PPO.load("./models/pg/ppo/ppo_final")
                rewards, accuracies, times = test_model(model, algorithm)
            elif algorithm == 'A2C':
                from stable_baselines3 import A2C
                model = A2C.load("./models/pg/a2c/a2c_final")
                rewards, accuracies, times = test_model(model, algorithm)
            elif algorithm == 'REINFORCE':
                from training.pg_training import REINFORCEAgent
                env = ExamGradingEnv()
                model = REINFORCEAgent(env)
                model.weights = np.load("./models/pg/reinforce/weights.npy")
                rewards, accuracies, times = test_model(model, algorithm)
            
            results[algorithm] = {
                'rewards': rewards,
                'accuracies': accuracies,
                'times': times,
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'avg_time': np.mean(times),
                'convergence_rate': calculate_convergence_rate(rewards)
            }
            
        except FileNotFoundError:
            print(f"Model {algorithm} not found. Skipping...")
            continue
        except Exception as e:
            print(f"Error evaluating {algorithm}: {e}")
            continue
    
    # Generate comparison report
    generate_comparison_report(results)
    
    # Create visualizations
    create_performance_plots(results)
    
    return results

def test_model(model, algorithm, num_episodes=20):
    """Test a model for multiple episodes"""
    
    env = ExamGradingEnv()
    rewards = []
    accuracies = []
    times = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        start_time = env.time_remaining
        
        while not done:
            if algorithm == 'REINFORCE':
                action = model.choose_action(obs)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rewards.append(episode_reward)
        accuracies.append(env.accuracy_history)
        times.append(start_time - env.time_remaining)
    
    return rewards, accuracies, times

def calculate_convergence_rate(rewards):
    """Calculate how quickly the model converges"""
    if len(rewards) < 10:
        return 0
    
    # Simple convergence metric: stability in last 50% of episodes
    mid_point = len(rewards) // 2
    early_avg = np.mean(rewards[:mid_point])
    late_avg = np.mean(rewards[mid_point:])
    
    improvement = (late_avg - early_avg) / max(abs(early_avg), 1)
    return improvement

def generate_comparison_report(results):
    """Generate detailed comparison report"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ALGORITHM COMPARISON REPORT")
    print("=" * 80)
    
    # Performance table
    print(f"\n{'Algorithm':<12} {'Avg Reward':<12} {'Std Reward':<12} {'Avg Accuracy':<14} {'Avg Time':<10}")
    print("-" * 70)
    
    for algorithm, metrics in results.items():
        print(f"{algorithm:<12} {metrics['avg_reward']:<12.2f} {metrics['std_reward']:<12.2f} "
              f"{metrics['avg_accuracy']:<14.2%} {metrics['avg_time']:<10.1f}")
    
    # Ranking
    print(f"\n{'ALGORITHM RANKINGS'}")
    print("-" * 30)
    
    # Rank by average reward
    reward_ranking = sorted(results.items(), key=lambda x: x[1]['avg_reward'], reverse=True)
    print("By Average Reward:")
    for i, (algorithm, _) in enumerate(reward_ranking, 1):
        print(f"  {i}. {algorithm}")
    
    # Rank by accuracy
    accuracy_ranking = sorted(results.items(), key=lambda x: x[1]['avg_accuracy'], reverse=True)
    print("\nBy Accuracy:")
    for i, (algorithm, _) in enumerate(accuracy_ranking, 1):
        print(f"  {i}. {algorithm}")
    
    # Analysis
    print(f"\n{'ANALYSIS'}")
    print("-" * 20)
    
    best_reward = max(results.items(), key=lambda x: x[1]['avg_reward'])
    best_accuracy = max(results.items(), key=lambda x: x[1]['avg_accuracy'])
    most_stable = min(results.items(), key=lambda x: x[1]['std_reward'])
    
    print(f"Best Overall Performance: {best_reward[0]} (Reward: {best_reward[1]['avg_reward']:.2f})")
    print(f"Highest Accuracy: {best_accuracy[0]} ({best_accuracy[1]['avg_accuracy']:.2%})")
    print(f"Most Stable: {most_stable[0]} (Std: {most_stable[1]['std_reward']:.2f})")
    
    # Recommendations
    print(f"\n{'RECOMMENDATIONS'}")
    print("-" * 20)
    print("For South Sudan's NLP Exam Grading System:")
    
    if best_accuracy[1]['avg_accuracy'] > 0.8:
        print(f"• {best_accuracy[0]} shows excellent accuracy ({best_accuracy[1]['avg_accuracy']:.1%}) suitable for deployment")
    
    if most_stable[1]['std_reward'] < 10:
        print(f"• {most_stable[0]} demonstrates consistent performance, ideal for reliable grading")
    
    if best_reward[1]['avg_reward'] > 100:
        print(f"• {best_reward[0]} achieves optimal reward balance between accuracy and efficiency")

def create_performance_plots(results):
    """Create visualization plots for model comparison"""
    
    if not results:
        print("No results to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('NLP Exam Grading RL Models - Performance Comparison', fontsize=16, fontweight='bold')
    
    algorithms = list(results.keys())
    
    # Average Rewards
    avg_rewards = [results[alg]['avg_reward'] for alg in algorithms]
    std_rewards = [results[alg]['std_reward'] for alg in algorithms]
    
    ax1.bar(algorithms, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    ax1.set_title('Average Reward per Algorithm')
    ax1.set_ylabel('Average Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # Accuracy Comparison
    avg_accuracies = [results[alg]['avg_accuracy'] for alg in algorithms]
    std_accuracies = [results[alg]['std_accuracy'] for alg in algorithms]
    
    ax2.bar(algorithms, avg_accuracies, yerr=std_accuracies, capsize=5, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
    ax2.set_title('Average Accuracy per Algorithm')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    # Reward Distribution (Box Plot)
    reward_data = [results[alg]['rewards'] for alg in algorithms]
    ax3.boxplot(reward_data, labels=algorithms)
    ax3.set_title('Reward Distribution')
    ax3.set_ylabel('Reward')
    ax3.tick_params(axis='x', rotation=45)
    
    # Accuracy vs Reward Scatter
    for i, alg in enumerate(algorithms):
        ax4.scatter(results[alg]['avg_accuracy'], results[alg]['avg_reward'], 
                   s=100, alpha=0.7, label=alg)
    
    ax4.set_xlabel('Average Accuracy')
    ax4.set_ylabel('Average Reward')
    ax4.set_title('Accuracy vs Reward Trade-off')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance plots saved as 'model_comparison.png'")

def create_hyperparameter_analysis():
    """Analyze hyperparameter effects"""
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER ANALYSIS")
    print("=" * 60)
    
    hyperparameters = {
        'DQN': {
            'Learning Rate': 0.0005,
            'Buffer Size': 50000,
            'Batch Size': 32,
            'Target Update': 1000,
            'Exploration Decay': 0.1
        },
        'PPO': {
            'Learning Rate': 0.0003,
            'Steps per Update': 2048,
            'Batch Size': 64,
            'Clip Range': 0.2,
            'Entropy Coefficient': 0.01
        },
        'A2C': {
            'Learning Rate': 0.0007,
            'Steps per Update': 5,
            'Value Function Coef': 0.25,
            'Entropy Coefficient': 0.01
        },
        'REINFORCE': {
            'Learning Rate': 0.01,
            'Discount Factor': 0.99,
            'Baseline': 'None'
        }
    }
    
    for algorithm, params in hyperparameters.items():
        print(f"\n{algorithm} Hyperparameters:")
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    print(f"\nHyperparameter Impact Analysis:")
    print("• Learning Rate: Lower rates (0.0003-0.0007) provide stable learning")
    print("• Batch Size: Larger batches (32-64) improve gradient estimation")
    print("• Exploration: Gradual decay prevents premature convergence")
    print("• Entropy: Small coefficients (0.01) encourage exploration")

if __name__ == "__main__":
    # Run comprehensive evaluation
    results = evaluate_all_models()
    
    # Additional analysis
    create_hyperparameter_analysis()
    
    print(f"\nEvaluation completed! Check 'model_comparison.png' for visualizations.")