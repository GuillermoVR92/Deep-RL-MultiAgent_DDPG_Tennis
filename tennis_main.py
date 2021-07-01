from unityagents import UnityEnvironment
import numpy as np
import pandas as pd
from maddpg_utilities import ReplayBuffer
from maddpg import MultiAgent_DDPG
from collections import deque
from ddpg_agent import DDPG_Agent
import torch
import matplotlib.pyplot as plt

NO_VIS = False
TRAIN = False
EVAL = True

def MADDPG(maddpg_agents, num_agents):
    
    number_of_episodes = 10000
    print_every = 100
    
    scores_deque = deque(maxlen=print_every)
    scores = []

    for episode in range(1, number_of_episodes + 1):
        
        maddpg_agents.reset()
        score = np.zeros(num_agents)                          # initialize the score (for each agent)

        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment    
        states = env_info.vector_observations

        while True:
            actions = maddpg_agents.act(states, episode, add_noise=True)          # select an action (for each agent)
                        
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            
            # get next state (for each agent)
            next_states = env_info.vector_observations
            
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            maddpg_agents.step(states, actions, rewards, next_states, dones)

            score += rewards                        # update the score (for each agent)
            states = next_states                              # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        print('\rEpisode {}\tAverage Score: {:.4f}'.format(episode, np.mean(scores_deque)), end="")
        # Save weights every 100 episodes
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(episode, np.mean(scores_deque)), end="")
            for i, agent in enumerate(maddpg_agents.maddpg):   
                torch.save(agent.actor_local.state_dict(), f'./checkpoint/actor_model_{i}.pth')
                torch.save(agent.critic_local.state_dict(), f'./checkpoint/critic_model_{i}.pth')

        #break training if env solved    
        if np.mean(scores_deque) >= 0.5:
            for i, agent in enumerate(maddpg_agents.maddpg):   
                torch.save(agent.actor_local.state_dict(), f'./final/actor_model_{i}.pth')
                torch.save(agent.critic_local.state_dict(), f'./final/critic_model_{i}.pth')
            break
        
    return scores

def MADDPG_Eval(maddpg_agents, num_agents, load_weights=True):
    
    if load_weights:
        for i, agent in enumerate(maddpg_agents.maddpg):   
            agent.actor_local.load_state_dict(torch.load(f'./final/actor_model_{i}.pth', map_location=torch.device('cpu')))
            agent.critic_local.load_state_dict(torch.load(f'./final/critic_model_{i}.pth', map_location=torch.device('cpu')))
    
    number_of_episodes = 100
    print_every = 100
    
    scores_deque = deque(maxlen=print_every)
    scores = []

    for episode in range(1, number_of_episodes + 1):
        
        maddpg_agents.reset()
        score = np.zeros(num_agents)                          # initialize the score (for each agent)

        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
        states = env_info.vector_observations

        while True:
            actions = maddpg_agents.act(states, episode, add_noise=True)          # select an action (for each agent)
                        
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            
            # get next state (for each agent)
            next_states = env_info.vector_observations
            
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished

            maddpg_agents.step(states, actions, rewards, next_states, dones)

            score += rewards                        # update the score (for each agent)
            states = next_states                              # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        print('\rEpisode {}\tAverage Score: {:.4f}'.format(episode, np.mean(scores_deque)), end="")
        
    return scores

def plot_scores(scores, rolling_window=10, save_fig=False):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'scores')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean)

    if save_fig:
        plt.savefig(f'figures_scores.png', bbox_inches='tight', pad_inches=0)

if NO_VIS:
    env = UnityEnvironment(file_name="./Tennis_No_Vis/Tennis.x86_64")
else:
    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
if TRAIN:
    env_info = env.reset(train_mode=True)[brain_name]
else:
    env_info = env.reset(train_mode=False)[brain_name]

# number of agents 
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

maddpg_agents = MultiAgent_DDPG(state_size=state_size, action_size=action_size, n_agents=num_agents, random_seed=42)

if TRAIN:
    scores = MADDPG(maddpg_agents, num_agents)
    
if EVAL:
    scores = MADDPG_Eval(maddpg_agents, num_agents)   

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

plot_scores(scores)

env.close()