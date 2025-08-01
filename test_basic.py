import sys
import os
sys.path.append('.')

# Test basic environment without heavy dependencies
from environment.custom_env import ExamGradingEnv

def test_environment():
    """Test the basic environment functionality"""
    print("Testing NLP Exam Grading Environment...")
    
    env = ExamGradingEnv()
    print("✓ Environment created successfully")
    
    # Test reset
    obs, info = env.reset()
    print(f"✓ Environment reset - Observation shape: {obs.shape}")
    
    # Test random actions
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if action == 11:
            print(f"Step {step+1}: SKIP (Reward: {reward:.1f})")
        else:
            print(f"Step {step+1}: Grade {action}/10 (Reward: {reward:.1f})")
        
        if terminated or truncated:
            break
    
    print(f"\nTest completed!")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Accuracy: {env.accuracy_history:.2%}")
    print(f"Responses Graded: {env.graded_count}")

if __name__ == "__main__":
    test_environment()