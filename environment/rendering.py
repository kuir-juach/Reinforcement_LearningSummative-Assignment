import pygame
import numpy as np
import sys
from typing import Dict, List, Tuple

class ExamGradingVisualizer:
    """Advanced visualization for the Exam Grading Environment using Pygame"""
    
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("NLP Exam Grading Agent - South Sudan Education System")
        
        # Colors
        self.colors = {
            'bg': (240, 248, 255),
            'panel': (255, 255, 255),
            'border': (70, 130, 180),
            'text': (25, 25, 112),
            'success': (34, 139, 34),
            'warning': (255, 140, 0),
            'error': (220, 20, 60),
            'progress': (100, 149, 237),
            'accent': (72, 61, 139)
        }
        
        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # Animation variables
        self.animation_time = 0
        self.grade_animation = 0
        self.accuracy_history = []
        
    def render_environment(self, env_state: Dict, action: int = None, reward: float = 0):
        """Render the current state of the grading environment"""
        self.screen.fill(self.colors['bg'])
        
        # Header
        self._draw_header()
        
        # Main panels
        self._draw_current_response_panel(env_state)
        self._draw_agent_status_panel(env_state)
        self._draw_performance_panel(env_state)
        self._draw_action_panel(action, reward)
        
        # Update animation
        self.animation_time += 0.1
        
        pygame.display.flip()
        
    def _draw_header(self):
        """Draw the application header"""
        header_rect = pygame.Rect(0, 0, self.width, 80)
        pygame.draw.rect(self.screen, self.colors['accent'], header_rect)
        
        title = self.font_large.render("NLP Exam Grading Agent - South Sudan", True, (255, 255, 255))
        subtitle = self.font_medium.render("Automated Grading System for Primary Education", True, (200, 200, 255))
        
        self.screen.blit(title, (20, 15))
        self.screen.blit(subtitle, (20, 50))
        
    def _draw_current_response_panel(self, env_state: Dict):
        """Draw panel showing current student response details"""
        panel_rect = pygame.Rect(20, 100, 580, 300)
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['border'], panel_rect, 2)
        
        # Panel title
        title = self.font_medium.render("Current Student Response", True, self.colors['text'])
        self.screen.blit(title, (30, 110))
        
        y_offset = 140
        
        if 'current_response' in env_state:
            response = env_state['current_response']
            
            # Subject
            subject_text = f"Subject: {response.get('subject', 'N/A')}"
            subject_surface = self.font_small.render(subject_text, True, self.colors['text'])
            self.screen.blit(subject_surface, (30, y_offset))
            y_offset += 25
            
            # Complexity bar
            complexity = response.get('complexity', 0)
            self._draw_progress_bar(30, y_offset, 200, 20, complexity / 10, 
                                  f"Complexity: {complexity}/10", self.colors['warning'])
            y_offset += 40
            
            # Length bar
            length = response.get('length', 0)
            normalized_length = min(length / 400, 1.0)  # Normalize to 400 words max
            self._draw_progress_bar(30, y_offset, 200, 20, normalized_length,
                                  f"Length: {length} words", self.colors['progress'])
            y_offset += 40
            
            # True score (for visualization purposes)
            true_score = response.get('true_score', 0)
            score_text = f"Expected Score: {true_score}/10"
            score_surface = self.font_small.render(score_text, True, self.colors['success'])
            self.screen.blit(score_surface, (30, y_offset))
            
            # Sample response preview (simulated)
            preview_rect = pygame.Rect(30, y_offset + 30, 540, 120)
            pygame.draw.rect(self.screen, (248, 248, 255), preview_rect)
            pygame.draw.rect(self.screen, self.colors['border'], preview_rect, 1)
            
            preview_text = self._generate_sample_text(response)
            self._draw_wrapped_text(preview_text, preview_rect, self.font_small, self.colors['text'])
    
    def _draw_agent_status_panel(self, env_state: Dict):
        """Draw panel showing agent's current status"""
        panel_rect = pygame.Rect(620, 100, 560, 200)
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['border'], panel_rect, 2)
        
        title = self.font_medium.render("Agent Status", True, self.colors['text'])
        self.screen.blit(title, (630, 110))
        
        y_offset = 140
        
        # Time remaining
        time_remaining = env_state.get('time_remaining', 300)
        time_color = self.colors['error'] if time_remaining < 60 else self.colors['success']
        time_text = f"Time Remaining: {time_remaining}s"
        time_surface = self.font_small.render(time_text, True, time_color)
        self.screen.blit(time_surface, (630, y_offset))
        
        # Time progress bar
        time_progress = time_remaining / 300
        self._draw_progress_bar(630, y_offset + 20, 300, 15, time_progress, "", time_color)
        y_offset += 50
        
        # Graded count
        graded = env_state.get('graded_count', 0)
        total = env_state.get('total_responses', 10)
        graded_text = f"Responses Graded: {graded}/{total}"
        graded_surface = self.font_small.render(graded_text, True, self.colors['text'])
        self.screen.blit(graded_surface, (630, y_offset))
        
        # Progress bar
        progress = graded / max(total, 1)
        self._draw_progress_bar(630, y_offset + 20, 300, 15, progress, "", self.colors['progress'])
        y_offset += 50
        
        # Current accuracy
        accuracy = env_state.get('accuracy_history', 0)
        accuracy_text = f"Current Accuracy: {accuracy:.1%}"
        accuracy_color = self.colors['success'] if accuracy >= 0.8 else self.colors['warning']
        accuracy_surface = self.font_small.render(accuracy_text, True, accuracy_color)
        self.screen.blit(accuracy_surface, (630, y_offset))
    
    def _draw_performance_panel(self, env_state: Dict):
        """Draw performance metrics and charts"""
        panel_rect = pygame.Rect(620, 320, 560, 200)
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['border'], panel_rect, 2)
        
        title = self.font_medium.render("Performance Metrics", True, self.colors['text'])
        self.screen.blit(title, (630, 330))
        
        # Add current accuracy to history for visualization
        current_accuracy = env_state.get('accuracy_history', 0)
        if len(self.accuracy_history) == 0 or self.accuracy_history[-1] != current_accuracy:
            self.accuracy_history.append(current_accuracy)
            if len(self.accuracy_history) > 50:  # Keep last 50 points
                self.accuracy_history.pop(0)
        
        # Draw accuracy trend line
        if len(self.accuracy_history) > 1:
            chart_rect = pygame.Rect(640, 360, 520, 120)
            self._draw_line_chart(chart_rect, self.accuracy_history, "Accuracy Trend")
    
    def _draw_action_panel(self, action: int, reward: float):
        """Draw current action and reward information"""
        panel_rect = pygame.Rect(20, 420, 580, 100)
        pygame.draw.rect(self.screen, self.colors['panel'], panel_rect)
        pygame.draw.rect(self.screen, self.colors['border'], panel_rect, 2)
        
        title = self.font_medium.render("Current Action & Reward", True, self.colors['text'])
        self.screen.blit(title, (30, 430))
        
        if action is not None:
            if action == 11:
                action_text = "Action: Skip for Human Review"
                action_color = self.colors['warning']
            else:
                action_text = f"Action: Grade Score = {action}/10"
                action_color = self.colors['success']
            
            action_surface = self.font_small.render(action_text, True, action_color)
            self.screen.blit(action_surface, (30, 460))
            
            # Reward
            reward_color = self.colors['success'] if reward > 0 else self.colors['error']
            reward_text = f"Reward: {reward:+.1f}"
            reward_surface = self.font_small.render(reward_text, True, reward_color)
            self.screen.blit(reward_surface, (300, 460))
    
    def _draw_progress_bar(self, x: int, y: int, width: int, height: int, 
                          progress: float, label: str, color: Tuple[int, int, int]):
        """Draw a progress bar with label"""
        # Background
        bg_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (220, 220, 220), bg_rect)
        
        # Progress
        progress_width = int(width * max(0, min(1, progress)))
        if progress_width > 0:
            progress_rect = pygame.Rect(x, y, progress_width, height)
            pygame.draw.rect(self.screen, color, progress_rect)
        
        # Border
        pygame.draw.rect(self.screen, self.colors['border'], bg_rect, 1)
        
        # Label
        if label:
            label_surface = self.font_small.render(label, True, self.colors['text'])
            self.screen.blit(label_surface, (x, y - 18))
    
    def _draw_line_chart(self, rect: pygame.Rect, data: List[float], title: str):
        """Draw a simple line chart"""
        if len(data) < 2:
            return
        
        # Title
        title_surface = self.font_small.render(title, True, self.colors['text'])
        self.screen.blit(title_surface, (rect.x, rect.y - 20))
        
        # Chart background
        pygame.draw.rect(self.screen, (248, 248, 255), rect)
        pygame.draw.rect(self.screen, self.colors['border'], rect, 1)
        
        # Draw data points
        points = []
        for i, value in enumerate(data):
            x = rect.x + (i / (len(data) - 1)) * rect.width
            y = rect.y + rect.height - (value * rect.height)
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.colors['success'], False, points, 2)
    
    def _generate_sample_text(self, response: Dict) -> str:
        """Generate sample response text based on response characteristics"""
        subject = response.get('subject', 'English')
        complexity = response.get('complexity', 5)
        
        if subject == "English":
            if complexity <= 3:
                return "The story is about a boy who help his mother. He go to market and buy food. The boy is good son."
            elif complexity <= 7:
                return "The narrative describes a young boy who demonstrates responsibility by assisting his mother with household tasks. He travels to the local market to purchase necessary provisions for the family."
            else:
                return "The compelling narrative illustrates the profound relationship between a conscientious adolescent and his nurturing mother, emphasizing themes of familial responsibility, community engagement, and moral development through practical life experiences."
        
        elif subject == "Social Studies":
            if complexity <= 3:
                return "South Sudan is country in Africa. It have many tribes. People speak different language. Juba is capital city."
            elif complexity <= 7:
                return "South Sudan is a landlocked country located in East-Central Africa. The nation gained independence in 2011 and consists of diverse ethnic groups with various cultural traditions and languages."
            else:
                return "The Republic of South Sudan represents a nascent nation-state characterized by remarkable ethnic diversity, complex historical narratives, and ongoing challenges in establishing sustainable governance structures while preserving cultural heritage."
        
        else:  # Religious Studies
            if complexity <= 3:
                return "God love all people. We should pray every day. Bible teach us good things. We must help other people."
            elif complexity <= 7:
                return "Religious teachings emphasize the importance of compassion, prayer, and service to others. Sacred texts provide guidance for moral living and spiritual development."
            else:
                return "Theological principles underscore the fundamental interconnectedness of humanity through divine love, emphasizing the transformative power of prayer, scriptural study, and altruistic service in fostering spiritual growth and community cohesion."
    
    def _draw_wrapped_text(self, text: str, rect: pygame.Rect, font: pygame.font.Font, color: Tuple[int, int, int]):
        """Draw text wrapped within a rectangle"""
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            if font.size(test_line)[0] <= rect.width - 10:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        y_offset = rect.y + 5
        for line in lines:
            if y_offset + font.get_height() > rect.y + rect.height:
                break
            line_surface = font.render(line, True, color)
            self.screen.blit(line_surface, (rect.x + 5, y_offset))
            y_offset += font.get_height() + 2
    
    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
    
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()