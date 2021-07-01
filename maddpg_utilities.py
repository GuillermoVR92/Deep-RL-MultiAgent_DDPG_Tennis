import numpy as np
import copy
import random
import random
from collections import deque, namedtuple

import torch

#Enable cuda if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """
    Implementation of a fixed size replay buffer as used in DQN algorithms.
    The goal of a replay buffer is to unserialize relationships between sequential experiences, gaining a better temporal understanding.
    """
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initializes the buffer.
        @Param:
        1. action_size: env.action_space.shape[0]
        2. buffer_size: Maximum length of the buffer for extrapolating all experiences into trajectories. default - 1e6 (Source: DeepMind)
        3. batch_size: size of mini-batch to train on. default = 64.
        """
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.replay_memory = deque(maxlen=buffer_size) #Experience replay memory object
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"]) #standard S,A,R,S',done
        
    def add(self, state, action, reward, next_state, done):
        """Adds an experience to existing memory"""
        trajectory = self.experience(state, action, reward, next_state, done)
        self.replay_memory.append(trajectory)
    
    def sample(self):
        """Randomly picks minibatches within the replay_buffer of size mini_batch"""
        experiences = random.sample(self.replay_memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):#override default __len__ operator
        """Return the current size of internal memory."""
        return len(self.replay_memory)

class OUNoise:
    """Ornstein-Uhlenbeck Noise process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class Noise:
    """Random noise process."""
    def __init__(self, size, weight, min_weight, noise_decay,
                 begin_noise_at, seed):
        self.size = size
        self.weight_start = weight
        self.weight = weight
        self.min_weight = min_weight
        self.noise_decay = noise_decay
        self.begin_noise_at = begin_noise_at
        self.seed = random.seed(seed)

    def reset(self):
        self.weight = self.weight_start

    def sample(self, i_episode):
        pwr = max(0, i_episode - self.begin_noise_at)
        if pwr > 0:
            self.weight = max(self.min_weight, self.noise_decay**pwr)
        return self.weight * 0.5 * np.random.standard_normal(self.size)