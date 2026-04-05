# 🚀 Deep Q-Network for Lunar Lander

A complete PyTorch implementation of a Deep Q-Network (DQN) agent
trained to solve the `LunarLander-v3` environment from Gymnasium.

The objective of the agent is to control a lunar spacecraft and land
safely between two flags. The model learns entirely through
reinforcement learning. Instead of being programmed with exact
instructions, the agent interacts with the environment, receives rewards
or penalties, and gradually improves its decisions over time.

This project contains:

-   A full DQN implementation in PyTorch
-   Replay memory and target network support
-   Automatic training and evaluation
-   Reward and loss visualization
-   Saved model loading and exporting
-   Optional comparison with Double DQN and Actor-Critic methods

------------------------------------------------------------------------

# 📖 Table of Contents

1.  Project Overview
2.  Features
3.  Deep Q-Learning Explanation
4.  Project Structure
5.  Requirements
6.  Installation
7.  Training
8.  Evaluation
9.  Demo Mode
10. DQN vs Actor-Critic Comparison
11. Files Description
12. Future Improvements
13. References
14. License

------------------------------------------------------------------------

# 🌕 Project Overview

`LunarLander-v3` is a classic reinforcement learning environment from
Gymnasium.

The spacecraft begins high above the landing zone with random speed and
direction. The agent must:

-   Fire the left engine
-   Fire the right engine
-   Fire the main engine
-   Or do nothing

The goal is to:

-   Land between the two flags
-   Keep the spacecraft upright
-   Reduce landing speed
-   Avoid crashing

The environment gives positive rewards for successful movement toward
the landing pad and negative rewards for crashing, moving too fast, or
wasting fuel.

The DQN agent learns the best sequence of actions to maximize total
reward.

------------------------------------------------------------------------

# ✨ Features

## Reinforcement Learning Features

-   Deep Q-Network using fully connected neural layers
-   Experience Replay Buffer
-   Separate Target Network
-   Epsilon-Greedy exploration
-   Epsilon decay during training
-   Mini-batch learning
-   Bellman equation updates
-   Automatic model checkpoint saving
-   Automatic loading of existing trained models

## Visualization Features

-   Reward vs Episode graph
-   Loss vs Episode graph
-   Epsilon decay graph
-   Optional gameplay visualization using Gymnasium render mode

## Project Features

-   Works entirely in Python
-   Built with PyTorch
-   Easy to modify for other Gymnasium environments
-   Can be extended to Double DQN or Dueling DQN
-   Includes training, testing, and comparison scripts

------------------------------------------------------------------------

# 🧠 Deep Q-Learning Explanation

Deep Q-Learning is a reinforcement learning algorithm where a neural
network estimates the Q-value of every possible action.

The Q-value represents:

> How good is this action if taken in the current state?

The neural network receives the current environment state as input and
returns a value for every action.

Example:

``` text
State → Neural Network → [Q(action0), Q(action1), Q(action2), Q(action3)]
```

The agent chooses the action with the highest predicted value.

## Agent Training Workflow

1.  Observe the current state
2.  Choose an action
3.  Execute the action
4.  Receive reward and next state
5.  Store experience in replay memory
6.  Sample random experiences
7.  Train the neural network
8.  Update the target network
9.  Repeat for many episodes

------------------------------------------------------------------------

# 🎯 Why Replay Memory Is Used

Without replay memory, the agent learns only from the most recent
experience. This often causes unstable learning.

Replay memory stores many previous transitions:

``` text
(state, action, reward, next_state, done)
```

Random samples are taken from this memory during training.

Advantages:

-   Reduces correlation between experiences
-   Makes learning more stable
-   Improves convergence speed

------------------------------------------------------------------------

# 🎯 Why a Target Network Is Used

The target network is a second copy of the main network.

Instead of immediately learning from constantly changing Q-values, the
agent uses a slowly updated target network.

Advantages:

-   More stable training
-   Less oscillation
-   Better long-term learning

------------------------------------------------------------------------

# 📁 Project Structure

``` text
project/
│
├── dqn_lunarlander.py
├── requirements.txt
└── README.md
```

## File Descriptions

### `dqn_lunarlander.py`

Main file that trains the DQN agent and optionally evaluates it after
training.

### `agent_class`

Contains:

-   Neural network architecture
-   Replay memory class
-   Action selection logic
-   Training step logic

### `train_agent`

Used only for training and saving the final model.

### `run_agent`

Loads a saved model and runs evaluation episodes.

### `train_and_visualize_agent`

Notebook version of the project for easier experimentation.

------------------------------------------------------------------------

# ⚙️ Requirements

Install the following Python libraries:

``` bash
pip install gymnasium
pip install torch
pip install numpy
pip install matplotlib
```

Or install all at once:

``` bash
pip install gymnasium torch numpy matplotlib
```

Optional packages:

``` bash
pip install tqdm
pip install imageio
```

------------------------------------------------------------------------

# 📥 Installation

Clone the repository:

``` bash
git clone https://github.com/yourusername/lunar-lander-dqn.git
cd lunar-lander-dqn
```

Create a virtual environment if needed:

``` bash
python -m venv venv
```

Activate it:

Windows

``` bash
venv\Scripts\activate
```

Linux / Mac

``` bash
source venv/bin/activate
```

------------------------------------------------------------------------

# 🏋️ Training the Agent

Run:

``` bash
python dqn_lunarlander.py
```

During training:

-   The agent explores the environment
-   Stores transitions into memory
-   Learns from random batches
-   Slowly reduces exploration
-   Updates the target network every few episodes

Example training output:

``` text
Episode: 1     Reward: -180.5    Loss: 0.0000    Epsilon: 1.000
Episode: 10    Reward: 95.4      Loss: 0.0231    Epsilon: 0.870
Episode: 50    Reward: 180.2     Loss: 0.0148    Epsilon: 0.520
Episode: 100   Reward: 240.7     Loss: 0.0093    Epsilon: 0.210
```

The model is automatically saved as:

``` text
lunar_dqn_model.pth
```

If this file already exists, it is loaded automatically.

------------------------------------------------------------------------

# 📊 Training Visualizations

After training, the program creates graphs such as:

-   Reward per Episode
-   Average Reward
-   Loss per Episode
-   Epsilon Value over Time

These graphs help determine whether the agent is improving.

Expected behavior:

-   Reward should increase over time
-   Loss should decrease gradually
-   Epsilon should slowly approach a low value

------------------------------------------------------------------------

# 🧪 Evaluating the Trained Agent

Run:

``` bash
python run_agent.py
```

Example output:

``` text
Evaluation Episode 1  Reward: 230.5
Evaluation Episode 2  Reward: 215.8
Evaluation Episode 3  Reward: 248.7

Average Reward: 231.7
```

A reward above 200 generally indicates that the environment has been
solved successfully.

------------------------------------------------------------------------

# 🎮 Demo Mode

To see the spacecraft landing visually, enable human render mode.

``` python
env = gym.make("LunarLander-v3", render_mode="human")
```

Then run the evaluation file again.

The environment window will open and display the trained spacecraft
landing in real time.

------------------------------------------------------------------------

# 🔬 DQN vs Actor-Critic vs Double DQN

## Deep Q-Network

Advantages

-   Faster training
-   Simple implementation
-   Better average reward
-   Uses only one neural network

Disadvantages

-   Can overestimate Q-values
-   Slightly less stable than Double DQN

## Double DQN

Advantages

-   Reduces Q-value overestimation
-   More stable training
-   Better long-term performance

Disadvantages

-   Slightly more complex
-   Requires extra update logic

## Actor-Critic

Advantages

-   Can produce very strong agents
-   Works well for continuous environments
-   More flexible

Disadvantages

-   Requires two networks
-   Slower training
-   More difficult to tune

------------------------------------------------------------------------

# 📈 Large Scale Comparison

Using:

``` text
trained_agents/batch_train_and_run.sh
```

500 agents were trained and each agent was tested across 1000 episodes.

## Training Episodes Comparison

  Algorithm      Average Episodes Needed
  -------------- -------------------------
  DQN            Lower
  Actor-Critic   28% Higher

## Training Runtime Comparison

  Algorithm      Relative Training Time
  -------------- ------------------------
  DQN            Faster
  Actor-Critic   67% Slower

Reason:

-   DQN trains one network
-   Actor-Critic trains two networks

## Return Comparison

  Algorithm      Mean Return
  -------------- -------------
  DQN            227.4
  Actor-Critic   211.6

Observation:

-   DQN gives better average performance
-   Actor-Critic sometimes produces the single best-performing agent

------------------------------------------------------------------------

# 📂 Important Files

  File                                Description
  ----------------------------------- ---------------------------------
  `agent_class.py`                    DQN agent implementation
  `train_agent.py`                    Trains the model
  `run_agent.py`                      Runs evaluation
  `train_and_visualize_agent.ipynb`   Notebook version
  `batch_train_and_run.sh`            Large scale training
  `plot_results.ipynb`                Graph and statistics generation

------------------------------------------------------------------------

# 🚀 Future Improvements

Possible upgrades:

-   Double DQN
-   Dueling DQN
-   Prioritized Experience Replay
-   TensorBoard support
-   Automatic gameplay recording
-   Hyperparameter tuning
-   GPU optimization
-   Multi-environment support

------------------------------------------------------------------------

# 📚 References

1.  Sutton, R. S., & Barto, A. G.\
    Reinforcement Learning: An Introduction\
    http://incompleteideas.net/book/the-book.html

2.  Mnih, V., Kavukcuoglu, K., Silver, D., et al.\
    Playing Atari with Deep Reinforcement Learning\
    https://arxiv.org/abs/1312.5602

3.  van Hasselt, H., Guez, A., Silver, D.\
    Deep Reinforcement Learning with Double Q-Learning\
    https://arxiv.org/abs/1509.06461

------------------------------------------------------------------------


```
