import os
import sys
import time
import random
import numpy as np
from environment.custom_env import ExamGradingEnv
from environment.rendering import ExamGradingVisualizer

def demo_random_agent():
    """Demonstrate random agent in the environment with visualization"""
    
    print("=== Random Agent Demo ===")
    print("Demonstrating random actions in the NLP Exam Grading Environment")
    print("This shows the visualization without any trained model")
    
    env = ExamGradingEnv()
    visualizer = ExamGradingVisualizer()
    
    # Run multiple episodes
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        while not done and visualizer.handle_events():
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
            
            # Prepare environment state for visualization
            env_state = {
                'current_response': env.sample_responses[min(env.current_response_idx, len(env.sample_responses)-1)] if env.current_response_idx < len(env.sample_responses) else {},
                'time_remaining': env.time_remaining,
                'graded_count': env.graded_count,
                'total_responses': len(env.sample_responses),
                'accuracy_history': env.accuracy_history
            }
            
            # Render visualization
            visualizer.render_environment(env_state, action, reward)
            
            # Print step info
            if action == 11:
                print(f"Step {step_count}: SKIP for human review (Reward: {reward:.1f})")
            else:
                print(f"Step {step_count}: Grade = {action}/10 (Reward: {reward:.1f})")
            
            time.sleep(0.5)  # Slow down for demonstration
        
        print(f"Episode {episode + 1} completed!")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Final Accuracy: {env.accuracy_history:.2%}")
        print(f"Responses Graded: {env.graded_count}/{len(env.sample_responses)}")
        
        # Pause between episodes
        time.sleep(2)
    
    visualizer.close()
    print("\nRandom agent demonstration completed!")

def run_trained_agent_demo(algorithm="dqn"):
    """Run demonstration with trained agent"""
    
    print(f"=== Trained {algorithm.upper()} Agent Demo ===")
    
    try:
        env = ExamGradingEnv()
        visualizer = ExamGradingVisualizer()
        
        # Load trained model
        if algorithm == "dqn":
            from stable_baselines3 import DQN
            model = DQN.load(f"./models/dqn/dqn_final")
        elif algorithm == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(f"./models/pg/ppo/ppo_final")
        elif algorithm == "a2c":
            from stable_baselines3 import A2C
            model = A2C.load(f"./models/pg/a2c/a2c_final")
        elif algorithm == "reinforce":
            from training.pg_training import REINFORCEAgent
            agent = REINFORCEAgent(env)
            agent.weights = np.load("./models/pg/reinforce/weights.npy")
            model = agent
        
        # Run episodes
        for episode in range(3):
            print(f"\n--- Episode {episode + 1} ---")
            obs, _ = env.reset()
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done and visualizer.handle_events():
                # Get action from trained model
                if algorithm == "reinforce":
                    action = model.choose_action(obs)
                else:
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Prepare environment state for visualization
                env_state = {
                    'current_response': env.sample_responses[min(env.current_response_idx, len(env.sample_responses)-1)] if env.current_response_idx < len(env.sample_responses) else {},
                    'time_remaining': env.time_remaining,
                    'graded_count': env.graded_count,
                    'total_responses': len(env.sample_responses),
                    'accuracy_history': env.accuracy_history
                }
                
                # Render visualization
                visualizer.render_environment(env_state, action, reward)
                
                # Print step info
                if action == 11:
                    print(f"Step {step_count}: SKIP for human review (Reward: {reward:.1f})")
                else:
                    print(f"Step {step_count}: Grade = {action}/10 (Reward: {reward:.1f})")
                
                time.sleep(0.3)  # Slightly faster than random demo
            
            print(f"Episode {episode + 1} completed!")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Final Accuracy: {env.accuracy_history:.2%}")
            print(f"Responses Graded: {env.graded_count}/{len(env.sample_responses)}")
            
            time.sleep(1)
        
        visualizer.close()
        print(f"\n{algorithm.upper()} agent demonstration completed!")
        
    except FileNotFoundError:
        print(f"Trained {algorithm.upper()} model not found. Please train the model first.")
        print("Run: python training/dqn_training.py or python training/pg_training.py")

def compare_all_algorithms():
    """Compare performance of all trained algorithms"""
    
    print("=== Algorithm Comparison ===")
    
    algorithms = ["dqn", "ppo", "a2c", "reinforce"]
    results = {}
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm.upper()}...")
        
        try:
            if algorithm == "dqn":
                from training.dqn_training import test_dqn
                rewards, accuracies = test_dqn()
            else:
                from training.pg_training import test_policy_gradient
                rewards, accuracies = test_policy_gradient(algorithm)
            
            results[algorithm] = {
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }
            
        except FileNotFoundError:
            print(f"Model for {algorithm.upper()} not found. Skipping...")
            continue
    
    # Print comparison table
    print("\n" + "="*80)
    print("ALGORITHM PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Algorithm':<12} {'Avg Reward':<12} {'Std Reward':<12} {'Avg Accuracy':<12} {'Std Accuracy':<12}")
    print("-"*80)
    
    for algorithm, metrics in results.items():
        print(f"{algorithm.upper():<12} {metrics['avg_reward']:<12.2f} {metrics['std_reward']:<12.2f} {metrics['avg_accuracy']:<12.2%} {metrics['std_accuracy']:<12.2%}")
    
    print("="*80)

def main():
    """Main entry point"""
    
    print("NLP Exam Grading RL Environment")
    print("South Sudan Education System Automation")
    print("="*50)
    
    while True:
        print("\nSelect an option:")
        print("1. Demo Random Agent (Visualization)")
        print("2. Demo Trained DQN Agent")
        print("3. Demo Trained PPO Agent")
        print("4. Demo Trained A2C Agent")
        print("5. Demo Trained REINFORCE Agent")
        print("6. Train All Models")
        print("7. Compare All Algorithms")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == "1":
            # Use pygame demo instead of complex visualizer
            import subprocess
            subprocess.run(["python", "pygame_demo.py"])
        elif choice == "2":
            run_trained_agent_demo("dqn")
        elif choice == "3":
            run_trained_agent_demo("ppo")
        elif choice == "4":
            run_trained_agent_demo("a2c")
        elif choice == "5":
            run_trained_agent_demo("reinforce")
        elif choice == "6":
            print("Training all models...")
            print("This will take some time. Please wait...")
            
            # Train DQN
            print("\n1/3 Training DQN...")
            os.system("python training/dqn_training.py")
            
            # Train Policy Gradient methods
            print("\n2/3 Training Policy Gradient methods...")
            os.system("python training/pg_training.py")
            
            print("\nAll models trained successfully!")
            
        elif choice == "7":
            compare_all_algorithms()
        elif choice == "8":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()