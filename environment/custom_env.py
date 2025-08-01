import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class ExamGradingEnv(gym.Env):
    """
    Custom Environment for NLP-based Examination Grading System
    
    The agent learns to efficiently grade student responses while maintaining accuracy.
    Actions: Grade assignment (0-10), Skip for human review, Request more context
    Rewards: Based on grading accuracy, efficiency, and fairness
    """
    
    def __init__(self):
        super(ExamGradingEnv, self).__init__()
        
        # Action space: 12 discrete actions
        # 0-10: Grade scores, 11: Skip for human review
        self.action_space = spaces.Discrete(12)
        
        # Observation space: [response_complexity, response_length, confidence_score, 
        #                    time_remaining, graded_count, accuracy_history]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([10, 1000, 1, 300, 100, 1]),
            dtype=np.float32
        )
        
        # Sample student responses with ground truth scores
        self.sample_responses = [
            {"complexity": 2, "length": 50, "true_score": 3, "subject": "English"},
            {"complexity": 5, "length": 150, "true_score": 7, "subject": "Social Studies"},
            {"complexity": 8, "length": 300, "true_score": 9, "subject": "Religious Studies"},
            {"complexity": 3, "length": 80, "true_score": 4, "subject": "English"},
            {"complexity": 7, "length": 200, "true_score": 8, "subject": "Social Studies"},
            {"complexity": 1, "length": 30, "true_score": 2, "subject": "English"},
            {"complexity": 6, "length": 180, "true_score": 6, "subject": "Religious Studies"},
            {"complexity": 4, "length": 120, "true_score": 5, "subject": "Social Studies"},
            {"complexity": 9, "length": 400, "true_score": 10, "subject": "English"},
            {"complexity": 2, "length": 60, "true_score": 3, "subject": "Religious Studies"}
        ]
        
        # Initialize environment state
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_response_idx = 0
        self.graded_count = 0
        self.correct_grades = 0
        self.time_remaining = 300  # 5 minutes per session
        self.accuracy_history = 1.0
        self.total_responses = len(self.sample_responses)
        
        # Shuffle responses for variety
        random.shuffle(self.sample_responses)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        if self.current_response_idx >= len(self.sample_responses):
            return np.array([0, 0, 0, self.time_remaining, self.graded_count, self.accuracy_history], dtype=np.float32)
        
        current = self.sample_responses[self.current_response_idx]
        confidence = max(0.1, 1.0 - (current["complexity"] / 10.0))
        
        return np.array([
            current["complexity"],
            current["length"] / 1000.0,  # Normalize
            confidence,
            self.time_remaining,
            self.graded_count,
            self.accuracy_history
        ], dtype=np.float32)
    
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        
        if self.current_response_idx >= len(self.sample_responses):
            terminated = True
            return self._get_observation(), reward, terminated, truncated, {}
        
        current_response = self.sample_responses[self.current_response_idx]
        true_score = current_response["true_score"]
        
        # Time penalty
        time_cost = 2 if action <= 10 else 5  # Skipping takes more time
        self.time_remaining -= time_cost
        
        if action == 11:  # Skip for human review
            reward = -1  # Small penalty for skipping
            self.graded_count += 1
        else:  # Grading action (0-10)
            predicted_score = action
            accuracy = 1.0 - abs(predicted_score - true_score) / 10.0
            
            # Reward based on accuracy
            if accuracy >= 0.9:
                reward = 10
            elif accuracy >= 0.7:
                reward = 5
            elif accuracy >= 0.5:
                reward = 1
            else:
                reward = -5
            
            # Bonus for perfect accuracy
            if predicted_score == true_score:
                reward += 5
                self.correct_grades += 1
            
            # Update accuracy history
            self.graded_count += 1
            self.accuracy_history = self.correct_grades / max(1, self.graded_count)
            
            # Efficiency bonus
            if self.time_remaining > 200:
                reward += 2
        
        # Move to next response
        self.current_response_idx += 1
        
        # Check termination conditions
        if self.time_remaining <= 0:
            truncated = True
            reward -= 20  # Penalty for running out of time
        elif self.current_response_idx >= len(self.sample_responses):
            terminated = True
            # Final accuracy bonus
            if self.accuracy_history >= 0.8:
                reward += 20
            elif self.accuracy_history >= 0.6:
                reward += 10
        
        return self._get_observation(), reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        if self.current_response_idx < len(self.sample_responses):
            current = self.sample_responses[self.current_response_idx]
            print(f"Response {self.current_response_idx + 1}/{len(self.sample_responses)}")
            print(f"Subject: {current['subject']}")
            print(f"Complexity: {current['complexity']}/10")
            print(f"Length: {current['length']} words")
            print(f"Time Remaining: {self.time_remaining}s")
            print(f"Accuracy: {self.accuracy_history:.2f}")
            print(f"Graded: {self.graded_count}")
            print("-" * 40)