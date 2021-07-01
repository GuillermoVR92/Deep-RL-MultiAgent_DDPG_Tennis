import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_agent import DDPG_Agent
from maddpg_utilities import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 2        # Udpate every
LEARN = 3

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgent_DDPG():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        self.maddpg = []
        for i in range(n_agents):
            self.maddpg.append(DDPG_Agent(state_size, action_size, i, n_agents, random_seed))
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        self.t_step = 0
        self.n_agents = n_agents
        
        
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        for i in range(self.n_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])
            
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            for _ in range(LEARN):
                for agent in self.maddpg:
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, GAMMA)
              
    def learn(self, experiences, agent, gamma):
        
        agent.learn(experiences, gamma)  

    def act(self, states, i_episode=0, add_noise=True):

        actions = []
        for agent, state in zip(self.maddpg, states):
            actions.append(np.squeeze(agent.act(np.expand_dims(state, axis=0), i_episode, add_noise), axis=0))
        return np.stack(actions)        
        
    def reset(self):
        for agent in self.maddpg:
            agent.reset()
 
 