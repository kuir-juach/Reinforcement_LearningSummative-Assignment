import sys
import os
import time
sys.path.append('.')

from environment.custom_env import ExamGradingEnv

def simple_random_demo():
    """Simple demonstration without pygame visualization"""
    
    print("=" * 60)
    print("NLP EXAM GRADING AGENT - RANDOM ACTIONS DEMO")
    print("South Sudan Education System Automation")
    print("=" * 60)
    
    env = ExamGradingEnv()
    
    # Run 3 episodes
    for episode in range(3):
        print(f"\n{'='*20} EPISODE {episode + 1} {'='*20}")
        
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        print(f"Starting Episode {episode + 1}")
        print(f"Total Responses to Grade: {len(env.sample_responses)}")
        print(f"Time Limit: {env.time_remaining} seconds")
        print("-" * 50)
        
        while not done and step_count < 15:  # Limit steps for demo
            # Get current response info
            if env.current_response_idx < len(env.sample_responses):
                current = env.sample_responses[env.current_response_idx]
                
                print(f"\nStep {step_count + 1}:")
                print(f"  📝 Subject: {current['subject']}")
                print(f"  🔢 Complexity: {current['complexity']}/10")
                print(f"  📏 Length: {current['length']} words")
                print(f"  ✅ Expected Score: {current['true_score']}/10")
                
                # Random action
                action = env.action_space.sample()
                
                # Take action
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Show action and result
                if action == 11:
                    print(f"  🤖 Agent Action: SKIP for human review")
                    print(f"  💰 Reward: {reward:.1f} (penalty for skipping)")
                else:
                    print(f"  🤖 Agent Action: Grade = {action}/10")
                    accuracy = 1.0 - abs(action - current['true_score']) / 10.0
                    print(f"  🎯 Accuracy: {accuracy:.1%}")
                    print(f"  💰 Reward: {reward:.1f}")
                
                print(f"  ⏰ Time Remaining: {env.time_remaining}s")
                print(f"  📊 Overall Accuracy: {env.accuracy_history:.1%}")
                print(f"  📈 Responses Graded: {env.graded_count}")
                
                time.sleep(0.5)  # Pause for readability
            else:
                break
        
        print(f"\n{'='*15} EPISODE {episode + 1} RESULTS {'='*15}")
        print(f"🏆 Total Reward: {episode_reward:.2f}")
        print(f"🎯 Final Accuracy: {env.accuracy_history:.1%}")
        print(f"📝 Responses Graded: {env.graded_count}/{len(env.sample_responses)}")
        print(f"⏰ Time Used: {300 - env.time_remaining}s")
        
        if episode < 2:  # Pause between episodes
            print("\nPress Enter to continue to next episode...")
            input()
    
    print(f"\n{'='*20} DEMO COMPLETED {'='*20}")
    print("This demonstrates the NLP Exam Grading Environment")
    print("The agent randomly chooses to grade (0-10) or skip responses")
    print("Rewards are based on grading accuracy and efficiency")

def show_environment_details():
    """Show detailed information about the environment"""
    
    print("\n" + "="*60)
    print("ENVIRONMENT SPECIFICATIONS")
    print("="*60)
    
    env = ExamGradingEnv()
    
    print(f"🎯 Action Space: {env.action_space}")
    print("   - Actions 0-10: Grade scores from 0 to 10")
    print("   - Action 11: Skip for human review")
    
    print(f"\n📊 Observation Space: {env.observation_space}")
    print("   - Response complexity (0-10)")
    print("   - Response length (normalized 0-1)")
    print("   - Confidence score (0-1)")
    print("   - Time remaining (0-300 seconds)")
    print("   - Graded count (0-100)")
    print("   - Accuracy history (0-1)")
    
    print(f"\n📚 Sample Subjects:")
    subjects = set(response['subject'] for response in env.sample_responses)
    for subject in subjects:
        print(f"   - {subject}")
    
    print(f"\n🏆 Reward System:")
    print("   - Perfect accuracy: +15 points")
    print("   - High accuracy (≥90%): +10 points")
    print("   - Good accuracy (≥70%): +5 points")
    print("   - Fair accuracy (≥50%): +1 point")
    print("   - Poor accuracy (<50%): -5 points")
    print("   - Skipping: -1 point")
    print("   - Time efficiency bonus: +2 points")
    print("   - Final accuracy bonus: +10-20 points")
    print("   - Time penalty: -20 points")

def create_sample_responses_display():
    """Display sample student responses"""
    
    print("\n" + "="*60)
    print("SAMPLE STUDENT RESPONSES")
    print("="*60)
    
    env = ExamGradingEnv()
    
    for i, response in enumerate(env.sample_responses[:5], 1):
        print(f"\n📝 Response {i}:")
        print(f"   Subject: {response['subject']}")
        print(f"   Complexity: {response['complexity']}/10")
        print(f"   Length: {response['length']} words")
        print(f"   Expected Score: {response['true_score']}/10")
        
        # Generate sample text based on complexity
        if response['subject'] == 'English':
            if response['complexity'] <= 3:
                sample = "The story is about a boy who help his mother. He go to market and buy food."
            elif response['complexity'] <= 7:
                sample = "The narrative describes a young boy who demonstrates responsibility by assisting his mother with household tasks."
            else:
                sample = "The compelling narrative illustrates the profound relationship between a conscientious adolescent and his nurturing mother."
        elif response['subject'] == 'Social Studies':
            if response['complexity'] <= 3:
                sample = "South Sudan is country in Africa. It have many tribes. Juba is capital city."
            elif response['complexity'] <= 7:
                sample = "South Sudan is a landlocked country located in East-Central Africa with diverse ethnic groups."
            else:
                sample = "The Republic of South Sudan represents a nascent nation-state characterized by remarkable ethnic diversity."
        else:  # Religious Studies
            if response['complexity'] <= 3:
                sample = "God love all people. We should pray every day. Bible teach us good things."
            elif response['complexity'] <= 7:
                sample = "Religious teachings emphasize the importance of compassion, prayer, and service to others."
            else:
                sample = "Theological principles underscore the fundamental interconnectedness of humanity through divine love."
        
        print(f"   Sample Text: \"{sample}\"")

def main_menu():
    """Main menu for the demo"""
    
    while True:
        print("\n" + "="*50)
        print("NLP EXAM GRADING RL ENVIRONMENT")
        print("South Sudan Education System")
        print("="*50)
        print("\nSelect an option:")
        print("1. Run Random Agent Demo")
        print("2. Show Environment Details")
        print("3. View Sample Responses")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            simple_random_demo()
        elif choice == "2":
            show_environment_details()
        elif choice == "3":
            create_sample_responses_display()
        elif choice == "4":
            print("\n👋 Thank you for exploring the NLP Exam Grading Environment!")
            print("This system demonstrates RL for South Sudan's education sector.")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()