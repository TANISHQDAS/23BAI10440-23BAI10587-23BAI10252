import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

ENV_NAME="LunarLander-v3"
MODEL_FILE="lunar_dqn_model.pth"
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED=42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    def push(self,s,a,r,ns,d):
        self.buffer.append((s,a,r,ns,d))
    def sample(self,batch):
        batch=random.sample(self.buffer,batch)
        s,a,r,ns,d=map(np.array,zip(*batch))
        return s,a,r,ns,d
    def __len__(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self,state_dim,action_dim):
        super().__init__()
        self.fc1=nn.Linear(state_dim,256)
        self.fc2=nn.Linear(256,256)
        self.fc3=nn.Linear(256,128)
        self.out=nn.Linear(128,action_dim)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        return self.out(x)

class DQNAgent:
    def __init__(self,state_dim,action_dim):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.gamma=0.99
        self.lr=0.0005
        self.batch_size=64
        self.epsilon=1.0
        self.epsilon_min=0.05
        self.epsilon_decay=0.995
        self.memory=ReplayBuffer(100000)
        self.policy=QNetwork(state_dim,action_dim).to(DEVICE)
        self.target=QNetwork(state_dim,action_dim).to(DEVICE)
        self.optimizer=optim.Adam(self.policy.parameters(),lr=self.lr)
        self.update_target()

    def update_target(self):
        self.target.load_state_dict(self.policy.state_dict())

    def act(self,state):
        if random.random()<self.epsilon:
            return random.randrange(self.action_dim)
        state=torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            q=self.policy(state)
        return q.argmax().item()

    def remember(self,s,a,r,ns,d):
        self.memory.push(s,a,r,ns,d)

    def train_step(self):
        if len(self.memory)<self.batch_size:
            return 0
        s,a,r,ns,d=self.memory.sample(self.batch_size)
        s=torch.FloatTensor(s).to(DEVICE)
        ns=torch.FloatTensor(ns).to(DEVICE)
        a=torch.LongTensor(a).unsqueeze(1).to(DEVICE)
        r=torch.FloatTensor(r).to(DEVICE)
        d=torch.FloatTensor(d).to(DEVICE)
        q=self.policy(s).gather(1,a).squeeze()
        next_q=self.target(ns).max(1)[0]
        target=r+self.gamma*next_q*(1-d)
        loss=F.mse_loss(q,target.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

def train(env,agent,episodes=300):
    rewards=[]
    losses=[]
    for ep in range(episodes):
        state,_=env.reset()
        done=False
        total_reward=0
        step_losses=[]
        while not done:
            action=agent.act(state)
            next_state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
            agent.remember(state,action,reward,next_state,done)
            loss=agent.train_step()
            if loss!=0:
                step_losses.append(loss)
            state=next_state
            total_reward+=reward
        if agent.epsilon>agent.epsilon_min:
            agent.epsilon*=agent.epsilon_decay
        if ep%10==0:
            agent.update_target()
        avg_loss=np.mean(step_losses) if step_losses else 0
        rewards.append(total_reward)
        losses.append(avg_loss)
        print("Episode:",ep,"Reward:",round(total_reward,2),"Loss:",round(avg_loss,4),"Epsilon:",round(agent.epsilon,3))
    return rewards,losses

def evaluate(env,agent,episodes=10):
    scores=[]
    for ep in range(episodes):
        state,_=env.reset()
        done=False
        total=0
        while not done:
            s=torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action=agent.policy(s).argmax().item()
            state,reward,terminated,truncated,_=env.step(action)
            done=terminated or truncated
            total+=reward
        scores.append(total)
        print("Eval episode reward:",total)
    print("Average reward:",np.mean(scores))

def plot_training(rewards,losses):
    plt.figure()
    plt.title("Reward per Episode")
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    plt.figure()
    plt.title("Loss per Episode")
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()

def demo(agent):
    env=gym.make(ENV_NAME,render_mode="human")
    state,_=env.reset()
    done=False
    total=0
    while not done:
        s=torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action=agent.policy(s).argmax().item()
        state,reward,terminated,truncated,_=env.step(action)
        done=terminated or truncated
        total+=reward
        time.sleep(0.02)
    print("Demo reward:",total)
    env.close()

def main():
    env=gym.make(ENV_NAME)
    state_dim=env.observation_space.shape[0]
    action_dim=env.action_space.n
    agent=DQNAgent(state_dim,action_dim)
    if os.path.exists(MODEL_FILE):
        print("Loading existing trained model")
        agent.policy.load_state_dict(torch.load(MODEL_FILE,map_location=DEVICE))
        agent.update_target()
    else:
        print("Training model")
        rewards,losses=train(env,agent,episodes=300)
        torch.save(agent.policy.state_dict(),MODEL_FILE)
        plot_training(rewards,losses)
    print("Evaluating agent")
    evaluate(env,agent)
    print("Running live demo")
    demo(agent)

if __name__=="__main__":
    main()