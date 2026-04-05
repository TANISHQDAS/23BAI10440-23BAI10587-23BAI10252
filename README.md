Deep Q-Network for Lunar Lander 🚀

This project implements a Deep Q-Network (DQN) using PyTorch to train an agent to solve the LunarLander-v3 environment from Gymnasium.

The goal of the agent is to control a spacecraft and land it safely between two flags using reinforcement learning. The agent learns optimal actions through interaction with the environment and by improving its policy using the Deep Q-Learning algorithm.

The implementation includes:

Deep Q-Network with multiple fully connected layers
Experience Replay Buffer
Target Network for stable learning
Epsilon-greedy exploration strategy
Training visualization using Matplotlib
Model saving and evaluation
Algorithm Overview

Deep Q-Learning is a reinforcement learning method where a neural network approximates the Q-function. The Q-function estimates the expected cumulative reward for taking an action in a given state.

The agent:

Observes the current environment state
Chooses an action using an epsilon-greedy policy
Receives a reward and next state
Stores the transition in a replay buffer
Samples mini-batches from memory to train the neural network

To stabilize training, a target network is periodically updated from the main policy network.

Project Structure
project/
│
├── dqn_lunarlander.py     # Main training and evaluation script
├── lunar_dqn_model.pth    # Saved trained model
├── README.md              # Project documentation
Requirements

Install the required Python libraries before running the code.

pip install gymnasium
pip install torch
pip install numpy
pip install matplotlib
How to Run

Clone the repository:

git clone https://github.com/yourusername/lunar-lander-dqn
cd lunar-lander-dqn

Run the training script:

python dqn_lunarlander.py

If a trained model file already exists, the script will load it automatically instead of retraining.

Training

During training the agent:

Collects experience from the environment
Stores experiences in a replay buffer
Updates the neural network using mini-batches
Periodically updates the target network
Gradually decreases exploration (epsilon decay)

Training progress prints:

Episode: 10 Reward: 120.5 Loss: 0.0231 Epsilon: 0.87

The script also generates plots showing:

Reward per Episode
Loss per Episode
Evaluation

After training, the agent is evaluated over multiple episodes to measure performance.

Example output:

Eval episode reward: 230.5
Eval episode reward: 215.8
Average reward: 224.3
Demo

The trained agent can be visualized interacting with the environment using the human render mode.

This shows how the learned policy controls the spacecraft to land successfully.

Method Reference

We use the lunar lander implementation from gymnasium. For the implementation of the actor-critic algorithm we loosely follow Ref. [1]. While for the implementation of deep Q-learning we follow Ref. [2], for the implementation of double deep Q-learning we follow Ref. [3].

Comparison: Actor-Critic vs Deep Q-Learning

With the script batch_train_and_run.sh we first train 500 agents and then run 1000 episodes for each agent using:

the actor-critic algorithm
the deep q-learning (DQN) algorithm

Here is a plot showing the distribution of the episodes needed for training for each scenario, along with the mean.

We observe that the distribution of episodes needed for training is more spread out for the actor-critic method. Furthermore, the actor-critic algorithm on average needed 28% more episodes to complete the training as compared to the DQN algorithm.

Here is a plot showing the actual runtime distribution of the respective 500 trainings.

On average, the actor-critic algorithm takes 67% longer to train compared to deep Q-learning. This is because the actor-critic algorithm trains two neural networks (actor and critic), while DQN trains only one network.

When evaluating trained agents across 1000 episodes, the distributions of returns are similar. However:

Mean return of DQN agents: 227.4
Mean return of actor-critic agents: 211.6

The best actor-critic agent slightly outperformed the best DQN agent.
