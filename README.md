# NLP Exam Grading RL Environment - South Sudan Education System

## Overview
This project implements a Natural Language Processing (NLP) system to improve examination marking efficiency in South Sudan’s primary education system, targeting one of the most persistent administrative challenges in the country’s education sector. The system simulates an intelligent grading model that automatically evaluates short-answer student responses, aiming to reduce result processing delays, enhance transparency, and ensure fairness in the assessment process. Rooted in the country’s urgent need for educational reform, the project addresses issues such as manual marking bottlenecks, a shortage of trained examiners, and limited operational capacity within the Ministry of General Education and Instruction. The approach involves designing a prototype NLP model capable of understanding and scoring subjective student responses in subjects like English, Social Studies, and Religious Education. Through the evaluation of the model’s performance against human grading benchmarks, the project explores the practical and ethical considerations of automating national assessments in a fragile, post-conflict setting. Ultimately, the research contributes to advancing South Sudan’s digital education agenda and aligns with Sustainable Development Goal 4 by proposing scalable, AI-driven solutions to promote inclusive and quality education.

## Project Structure
```
project_root/
├── environment/
│   ├── custom_env.py            # Custom Gymnasium environment implementation
│   ├── rendering.py             # Advanced Pygame visualization system
├── training/
│   ├── dqn_training.py          # DQN training and evaluation
│   ├── pg_training.py           # Policy Gradient methods (PPO, A2C, REINFORCE)
├── models/
│   ├── dqn/                     # Saved DQN models
│   └── pg/                      # Saved policy gradient models
├── main.py                      # Interactive entry point
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## Environment Details

### Action Space
- **Type**: Discrete (12 actions)
- **Actions 0-10**: Grade scores from 0 to 10
- **Action 11**: Skip for human review

### Observation Space
- **Type**: Box (6 dimensions)
- **Features**:
  1. Response complexity (0-10)
  2. Response length (normalized 0-1)
  3. Confidence score (0-1)
  4. Time remaining (0-300 seconds)
  5. Graded count (0-100)
  6. Accuracy history (0-1)

### Reward System
- **Perfect accuracy**: +15 points
- **High accuracy (≥90%)**: +10 points
- **Good accuracy (≥70%)**: +5 points
- **Fair accuracy (≥50%)**: +1 point
- **Poor accuracy (<50%)**: -5 points
- **Skipping**: -1 point
- **Time efficiency bonus**: +2 points
- **Final accuracy bonus**: +10-20 points
- **Time penalty**: -20 points

## Algorithms Implemented

### 1. Deep Q-Network (DQN) - Value-Based
- **Learning Rate**: 0.0005
- **Buffer Size**: 50,000
- **Batch Size**: 32
- **Target Update**: Every 1,000 steps
- **Exploration**: ε-greedy (1.0 → 0.05)

### 2. Proximal Policy Optimization (PPO) - Policy Gradient
- **Learning Rate**: 0.0003
- **Steps per Update**: 2,048
- **Batch Size**: 64
- **Clip Range**: 0.2
- **Entropy Coefficient**: 0.01

### 3. Advantage Actor-Critic (A2C) - Actor-Critic
- **Learning Rate**: 0.0007
- **Steps per Update**: 5
- **Value Function Coefficient**: 0.25
- **Entropy Coefficient**: 0.01

### 4. REINFORCE - Policy Gradient
- **Learning Rate**: 0.01
- **Discount Factor**: 0.99
- **Custom implementation with baseline

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/kuir_rl_summative.git
cd kuir_rl_summative
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode
Run the main script for an interactive experience:
```bash
python main.py
```

### Training Individual Models

#### Train DQN:
```bash
python training/dqn_training.py
```

#### Train Policy Gradient Methods:
```bash
python training/pg_training.py
```

### Testing Models
Models are automatically tested after training. You can also run individual tests through the main menu.

## Visualization Features

The environment includes advanced visualization using Pygame:
- **Real-time Environment State**: Shows current response details, complexity, and length
- **Agent Status Panel**: Displays time remaining, graded count, and accuracy
- **Performance Metrics**: Live accuracy trend charts
- **Action Feedback**: Visual representation of agent decisions
- **Sample Response Preview**: Simulated student responses based on complexity

## Results and Analysis

### Performance Metrics
- **Average Reward**: Measures overall agent performance
- **Accuracy**: Percentage of correctly graded responses
- **Efficiency**: Time utilization and grading speed
- **Convergence**: Training stability and learning progress

### Expected Outcomes
- **DQN**: Stable learning with good exploration-exploitation balance
- **PPO**: Smooth policy updates with clipped objectives
- **A2C**: Fast learning with actor-critic architecture
- **REINFORCE**: Simple policy gradient with high variance

## Hyperparameter Analysis

### DQN Parameters
- **Buffer Size**: Larger buffers improve stability but require more memory
- **Learning Rate**: Lower rates provide stable learning
- **Exploration Schedule**: Gradual decay prevents premature convergence

### PPO Parameters
- **Clip Range**: Prevents large policy updates
- **Batch Size**: Affects gradient estimation quality
- **Entropy Coefficient**: Encourages exploration

### A2C Parameters
- **Step Size**: Balances learning speed and stability
- **Value Function Coefficient**: Weights critic loss importance

## Educational Context

This project addresses real challenges in South Sudan's education system:
- **Manual Grading Delays**: Automated system reduces processing time
- **Consistency Issues**: AI provides uniform grading standards
- **Resource Constraints**: Reduces need for human markers
- **Scalability**: Can handle large volumes of examinations

## Future Enhancements

1. **Multi-Subject Support**: Extend to more subjects
2. **Natural Language Processing**: Integrate actual NLP models
3. **Adaptive Difficulty**: Dynamic complexity adjustment
4. **Real Data Integration**: Use actual student responses
5. **Bias Detection**: Implement fairness monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- South Sudan Ministry of General Education and Instruction
- Stable Baselines3 community
- OpenAI Gymnasium framework
- Pygame development team

## Contact

For questions or collaboration opportunities, please contact:
- **Student**: Kuir Juach Kuir
- **Institution**: African Leadership University
- **Course**: Natural Language Processing & Reinforcement Learning
