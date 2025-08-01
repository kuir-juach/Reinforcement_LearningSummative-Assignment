import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from environment.custom_env import ExamGradingEnv

def create_static_demo_frames():
    """Create static frames showing agent taking random actions"""
    
    env = ExamGradingEnv()
    frames = []
    
    # Create a simple visualization frame
    def create_frame(step, response, action, reward, accuracy, time_left):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('NLP Exam Grading Agent - Random Actions Demo', fontsize=16, fontweight='bold')
        
        # Response details
        ax1.text(0.1, 0.8, f"Step: {step}", fontsize=12, fontweight='bold')
        ax1.text(0.1, 0.7, f"Subject: {response.get('subject', 'N/A')}", fontsize=10)
        ax1.text(0.1, 0.6, f"Complexity: {response.get('complexity', 0)}/10", fontsize=10)
        ax1.text(0.1, 0.5, f"Length: {response.get('length', 0)} words", fontsize=10)
        ax1.text(0.1, 0.4, f"True Score: {response.get('true_score', 0)}/10", fontsize=10)
        ax1.set_title("Current Response")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Action taken
        action_text = "Skip for Human Review" if action == 11 else f"Grade: {action}/10"
        color = 'orange' if action == 11 else 'green'
        ax2.text(0.5, 0.6, "Action Taken:", fontsize=12, fontweight='bold', ha='center')
        ax2.text(0.5, 0.4, action_text, fontsize=14, fontweight='bold', ha='center', color=color)
        ax2.text(0.5, 0.2, f"Reward: {reward:+.1f}", fontsize=12, ha='center')
        ax2.set_title("Agent Decision")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
        
        # Performance metrics
        ax3.text(0.1, 0.8, f"Accuracy: {accuracy:.1%}", fontsize=12)
        ax3.text(0.1, 0.6, f"Time Left: {time_left}s", fontsize=12)
        ax3.text(0.1, 0.4, f"Graded: {step}/10", fontsize=12)
        ax3.set_title("Performance")
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        # Simple accuracy bar
        ax4.bar(['Accuracy'], [accuracy], color='skyblue', alpha=0.7)
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Current Accuracy')
        
        plt.tight_layout()
        
        # Save frame
        frame_path = f'temp_frame_{step}.png'
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return frame_path
    
    # Generate frames
    obs, _ = env.reset()
    step = 0
    
    print("Generating demo frames...")
    
    while step < 10 and env.current_response_idx < len(env.sample_responses):
        current_response = env.sample_responses[env.current_response_idx]
        action = env.action_space.sample()  # Random action
        
        obs, reward, terminated, truncated, _ = env.step(action)
        step += 1
        
        frame_path = create_frame(
            step, 
            current_response, 
            action, 
            reward, 
            env.accuracy_history, 
            env.time_remaining
        )
        frames.append(frame_path)
        
        if terminated or truncated:
            break
    
    # Create GIF
    if frames:
        images = [Image.open(frame) for frame in frames]
        images[0].save(
            'random_agent_demo.gif',
            save_all=True,
            append_images=images[1:],
            duration=2000,  # 2 seconds per frame
            loop=0
        )
        
        # Clean up temporary files
        for frame in frames:
            os.remove(frame)
        
        print("GIF created: random_agent_demo.gif")
    else:
        print("No frames generated")

def create_simple_text_demo():
    """Create a simple text-based demonstration"""
    
    print("=== NLP Exam Grading Agent Demo ===")
    print("Random Agent Taking Actions in South Sudan Education Environment")
    print("=" * 60)
    
    env = ExamGradingEnv()
    obs, _ = env.reset()
    
    episode_reward = 0
    step = 0
    
    while step < 10 and env.current_response_idx < len(env.sample_responses):
        current_response = env.sample_responses[env.current_response_idx]
        action = env.action_space.sample()
        
        print(f"\nStep {step + 1}:")
        print(f"  Subject: {current_response['subject']}")
        print(f"  Complexity: {current_response['complexity']}/10")
        print(f"  Length: {current_response['length']} words")
        print(f"  Expected Score: {current_response['true_score']}/10")
        
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        step += 1
        
        if action == 11:
            print(f"  Action: SKIP for human review")
        else:
            print(f"  Action: Grade = {action}/10")
        
        print(f"  Reward: {reward:+.1f}")
        print(f"  Accuracy: {env.accuracy_history:.1%}")
        print(f"  Time Remaining: {env.time_remaining}s")
        
        if terminated or truncated:
            break
    
    print(f"\nDemo completed!")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"Final Accuracy: {env.accuracy_history:.1%}")
    print(f"Responses Processed: {env.graded_count}")

if __name__ == "__main__":
    print("Creating demonstration materials...")
    
    # Create text demo
    create_simple_text_demo()
    
    # Try to create GIF (requires matplotlib)
    try:
        create_static_demo_frames()
    except ImportError:
        print("Matplotlib not available. Skipping GIF creation.")
        print("Install matplotlib with: pip install matplotlib")
    except Exception as e:
        print(f"Error creating GIF: {e}")
        print("Text demonstration completed successfully.")