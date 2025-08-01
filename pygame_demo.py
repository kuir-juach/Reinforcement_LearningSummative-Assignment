import pygame
import sys
import time
import random
from environment.custom_env import ExamGradingEnv

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (70, 130, 180)
GREEN = (34, 139, 34)
RED = (220, 20, 60)
ORANGE = (255, 140, 0)
LIGHT_BLUE = (173, 216, 230)

def run_pygame_demo():
    """Run pygame visualization demo"""
    
    # Create screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NLP Exam Grading Agent - South Sudan Education System")
    
    # Fonts
    font_large = pygame.font.Font(None, 36)
    font_medium = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 18)
    
    # Create environment
    env = ExamGradingEnv()
    clock = pygame.time.Clock()
    
    # Demo variables
    episode = 1
    obs, _ = env.reset()
    episode_reward = 0
    step_count = 0
    done = False
    last_action = None
    last_reward = 0
    
    print("Starting Pygame Demo...")
    print("Close the window to exit")
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Take action every 60 frames (1 second at 60 FPS)
        if pygame.time.get_ticks() % 1000 < 17:  # Approximately every second
            if not done and env.current_response_idx < len(env.sample_responses):
                # Random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                last_action = action
                last_reward = reward
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Print to console
                if action == 11:
                    print(f"Step {step_count}: SKIP for human review (Reward: {reward:.1f})")
                else:
                    print(f"Step {step_count}: Grade = {action}/10 (Reward: {reward:.1f})")
        
        # Reset for next episode if done
        if done:
            print(f"Episode {episode} completed! Reward: {episode_reward:.2f}, Accuracy: {env.accuracy_history:.2%}")
            episode += 1
            if episode <= 3:
                obs, _ = env.reset()
                episode_reward = 0
                step_count = 0
                done = False
                time.sleep(2)
            else:
                running = False  # Stop after 3 episodes
        
        # Clear screen
        screen.fill(WHITE)
        
        # Draw header
        header_rect = pygame.Rect(0, 0, SCREEN_WIDTH, 80)
        pygame.draw.rect(screen, BLUE, header_rect)
        
        title = font_large.render("NLP Exam Grading Agent - South Sudan", True, WHITE)
        subtitle = font_medium.render("Random Agent Demonstration", True, WHITE)
        screen.blit(title, (20, 15))
        screen.blit(subtitle, (20, 50))
        
        # Current response panel
        if env.current_response_idx < len(env.sample_responses):
            current = env.sample_responses[env.current_response_idx]
            
            # Response panel
            response_rect = pygame.Rect(20, 100, 460, 250)
            pygame.draw.rect(screen, LIGHT_BLUE, response_rect)
            pygame.draw.rect(screen, BLACK, response_rect, 2)
            
            y_pos = 120
            screen.blit(font_medium.render("Current Student Response", True, BLACK), (30, y_pos))
            y_pos += 30
            
            screen.blit(font_small.render(f"Subject: {current['subject']}", True, BLACK), (30, y_pos))
            y_pos += 25
            screen.blit(font_small.render(f"Complexity: {current['complexity']}/10", True, BLACK), (30, y_pos))
            y_pos += 25
            screen.blit(font_small.render(f"Length: {current['length']} words", True, BLACK), (30, y_pos))
            y_pos += 25
            screen.blit(font_small.render(f"Expected Score: {current['true_score']}/10", True, BLACK), (30, y_pos))
            y_pos += 35
            
            # Sample text based on complexity
            if current['subject'] == 'English':
                if current['complexity'] <= 3:
                    sample = "The story is about a boy who help his mother..."
                elif current['complexity'] <= 7:
                    sample = "The narrative describes a young boy who demonstrates..."
                else:
                    sample = "The compelling narrative illustrates the profound..."
            elif current['subject'] == 'Social Studies':
                if current['complexity'] <= 3:
                    sample = "South Sudan is country in Africa. It have many tribes..."
                else:
                    sample = "South Sudan is a landlocked country located in..."
            else:
                sample = "Religious teachings emphasize the importance of..."
            
            # Wrap text
            words = sample.split()
            lines = []
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                if font_small.size(test_line)[0] < 420:
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines[:4]:  # Show max 4 lines
                screen.blit(font_small.render(line, True, BLACK), (30, y_pos))
                y_pos += 20
        
        # Agent status panel
        status_rect = pygame.Rect(500, 100, 480, 250)
        pygame.draw.rect(screen, LIGHT_BLUE, status_rect)
        pygame.draw.rect(screen, BLACK, status_rect, 2)
        
        y_pos = 120
        screen.blit(font_medium.render("Agent Status", True, BLACK), (510, y_pos))
        y_pos += 30
        
        screen.blit(font_small.render(f"Episode: {episode}", True, BLACK), (510, y_pos))
        y_pos += 25
        screen.blit(font_small.render(f"Step: {step_count}", True, BLACK), (510, y_pos))
        y_pos += 25
        screen.blit(font_small.render(f"Time Remaining: {env.time_remaining}s", True, BLACK), (510, y_pos))
        y_pos += 25
        screen.blit(font_small.render(f"Accuracy: {env.accuracy_history:.1%}", True, BLACK), (510, y_pos))
        y_pos += 25
        screen.blit(font_small.render(f"Graded: {env.graded_count}/{len(env.sample_responses)}", True, BLACK), (510, y_pos))
        y_pos += 25
        screen.blit(font_small.render(f"Episode Reward: {episode_reward:.1f}", True, BLACK), (510, y_pos))
        
        # Action panel
        action_rect = pygame.Rect(20, 370, 960, 100)
        pygame.draw.rect(screen, LIGHT_BLUE, action_rect)
        pygame.draw.rect(screen, BLACK, action_rect, 2)
        
        screen.blit(font_medium.render("Last Action", True, BLACK), (30, 385))
        
        if last_action is not None:
            if last_action == 11:
                action_text = "SKIP for Human Review"
                action_color = ORANGE
            else:
                action_text = f"Grade: {last_action}/10"
                action_color = GREEN
            
            screen.blit(font_medium.render(action_text, True, action_color), (30, 415))
            screen.blit(font_small.render(f"Reward: {last_reward:+.1f}", True, BLACK), (30, 445))
        
        # Progress bars
        progress_rect = pygame.Rect(20, 490, 960, 150)
        pygame.draw.rect(screen, LIGHT_BLUE, progress_rect)
        pygame.draw.rect(screen, BLACK, progress_rect, 2)
        
        screen.blit(font_medium.render("Progress", True, BLACK), (30, 505))
        
        # Time progress
        time_progress = env.time_remaining / 300
        time_bar_rect = pygame.Rect(30, 535, 400, 20)
        pygame.draw.rect(screen, WHITE, time_bar_rect)
        pygame.draw.rect(screen, GREEN if time_progress > 0.3 else RED, 
                        pygame.Rect(30, 535, int(400 * time_progress), 20))
        pygame.draw.rect(screen, BLACK, time_bar_rect, 1)
        screen.blit(font_small.render("Time Remaining", True, BLACK), (30, 515))
        
        # Accuracy progress
        acc_bar_rect = pygame.Rect(30, 580, 400, 20)
        pygame.draw.rect(screen, WHITE, acc_bar_rect)
        pygame.draw.rect(screen, GREEN, 
                        pygame.Rect(30, 580, int(400 * env.accuracy_history), 20))
        pygame.draw.rect(screen, BLACK, acc_bar_rect, 1)
        screen.blit(font_small.render("Accuracy", True, BLACK), (30, 560))
        
        # Completion progress
        completion = env.graded_count / len(env.sample_responses)
        comp_bar_rect = pygame.Rect(500, 535, 400, 20)
        pygame.draw.rect(screen, WHITE, comp_bar_rect)
        pygame.draw.rect(screen, BLUE, 
                        pygame.Rect(500, 535, int(400 * completion), 20))
        pygame.draw.rect(screen, BLACK, comp_bar_rect, 1)
        screen.blit(font_small.render("Completion", True, BLACK), (500, 515))
        
        # Instructions
        screen.blit(font_small.render("Close window to exit", True, BLACK), (500, 580))
        
        # Update display
        pygame.display.flip()
        clock.tick(60)  # 60 FPS
    
    pygame.quit()
    print("Demo completed!")

if __name__ == "__main__":
    run_pygame_demo()