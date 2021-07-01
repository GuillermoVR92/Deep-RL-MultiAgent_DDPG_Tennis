# Multi Agent DDPG Tennis
Multi Agent DDPG used for playing Tennis

### Udacity RL-Collaboration and Competition Project

This is a project from the **Deep Reinforcement Learning Nanodegree** @Udacity. 

_The task is to solve the Unity [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) Environment._
You can watch the final result by using the `watch_a_trained_agent.ipynb`.

![](/results/trained.gif)


### Project Overview
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The **observation space consists of 8 variables** corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. **Two continuous actions** are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

2. Place the file in the repository folder, and unzip (or decompress) the file. 

3. Open the `Tennis.ipynb` notebook and follow instructions there.


### Dependencies
For more information on how to run the environment please read the Dependencies section in this [repo](https://github.com/udacity/deep-reinforcement-learning#dependencies).
