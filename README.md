🚀 Deep Q-Network for Lunar Lander

This project implements a Deep Q-Network (DQN) using PyTorch to train an agent to solve the LunarLander-v3 environment from Gymnasium.

The goal of the agent is to control a spacecraft and land it safely between two flags using reinforcement learning. The agent learns optimal actions through interaction with the environment and improves its policy using the Deep Q-Learning algorithm.

📌 Features

The implementation includes:

Deep Q-Network with multiple fully connected layers
Experience Replay Buffer
Target Network for stable learning
Epsilon-greedy exploration strategy
Training visualization using Matplotlib
Model saving and evaluation
🧠 Algorithm Overview

Deep Q-Learning is a reinforcement learning method where a neural network approximates the Q-function. The Q-function estimates the expected cumulative reward for taking an action in a given state.

Agent Workflow
Observe the current environment state
Choose an action using an epsilon-greedy policy
Receive a reward and next state
Store the transition in a replay buffer
Sample mini-batches from memory to train the neural network

To stabilize training, a target network is periodically updated from the main policy network.

📁 Project Structure
project/
│
├── dqn_lunarlander.py     # Main training and evaluation script
├── lunar_dqn_model.pth    # Saved trained model
├── README.md              # Project documentation
⚙️ Requirements

Install the required Python libraries before running the code.

pip install gymnasium
pip install torch
pip install numpy
pip install matplotlib
▶️ How to Run
1. Clone the repository
git clone https://github.com/yourusername/lunar-lander-dqn
cd lunar-lander-dqn
2. Run the training script
python dqn_lunarlander.py

If a trained model file already exists, the script will load it automatically instead of retraining.

🏋️ Training

During training the agent:

Collects experience from the environment
Stores experiences in a replay buffer
Updates the neural network using mini-batches
Periodically updates the target network
Gradually decreases exploration (epsilon decay)
Example Training Output
Episode: 10  Reward: 120.5  Loss: 0.0231  Epsilon: 0.87
Training Visualizations

The script generates plots showing:

Reward per Episode
Loss per Episode
📊 Evaluation

After training, the agent is evaluated over multiple episodes to measure performance.

Example Output
Eval episode reward: 230.5
Eval episode reward: 215.8
Average reward: 224.3
🎮 Demo

The trained agent can be visualized interacting with the environment using human render mode.

This demonstrates how the learned policy controls the spacecraft to land successfully.

🔬 Method Reference

We use the Lunar Lander implementation from Gymnasium.

The Actor-Critic algorithm loosely follows Ref. [1]
The Deep Q-Learning implementation follows Ref. [2]
The Double Deep Q-Learning implementation follows Ref. [3]
📈 Comparison: Actor-Critic vs Deep Q-Learning

Using the script:

batch_train_and_run.sh

we train 500 agents and run 1000 evaluation episodes for each agent using:

Actor-Critic algorithm
Deep Q-Learning (DQN)
Training Episodes Distribution

A plot shows the distribution of the number of episodes needed for training.

Observation

Actor-Critic distribution is more spread out
Actor-Critic required 28% more episodes on average to complete training compared to DQN
Training Runtime Distribution

Another plot shows training runtime for 500 agents.

Observation

Actor-Critic takes 67% longer to train
Reason: Actor-Critic trains two neural networks
Algorithm	Neural Networks
DQN	1 Network
Actor-Critic	2 Networks (Actor + Critic)
Return Distribution (1000 Episodes)

Both algorithms produce similar return distributions.

Algorithm	Mean Return
DQN	227.4
Actor-Critic	211.6

However, the best Actor-Critic agent slightly outperformed the best DQN agent.

📂 Files and Usage
File	Description
agent_class.py	Implements the agent class used for training and acting
train_and_visualize_agent.ipynb	Trains an agent and generates gameplay videos
train_agent.py	Trains an agent and saves parameters + training statistics
run_agent.py	Runs evaluation episodes using a trained agent
trained_agents/batch_train_and_run.sh	Trains 500 agents and runs evaluation episodes
trained_agents/plot_results.ipynb	Analyzes training statistics and performance
📊 Key Findings
Training Efficiency
DQN required fewer episodes for training
DQN required less training time
Performance
Metric	DQN	Actor-Critic
Mean Return	Higher	Slightly Lower
Training Speed	Faster	Slower
Best Agent	Slightly Lower	Slightly Higher
Summary
DQN trains faster and provides better average performance
Actor-Critic occasionally produces the best individual agents

A larger study with more agents and optimized hyperparameters could provide deeper insights.

📚 References

[1] Sutton, R. S., & Barto, A. G.
Reinforcement Learning: An Introduction
http://incompleteideas.net/book/the-book.html

[2] Mnih, V., Kavukcuoglu, K., Silver, D., et al.
Playing Atari with Deep Reinforcement Learning
https://arxiv.org/abs/1312.5602

[3] van Hasselt, H., Guez, A., Silver, D.
Deep Reinforcement Learning with Double Q-Learning
https://arxiv.org/abs/1509.06461
