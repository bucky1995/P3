#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from agent import Agent
from dqn_model import DQN

from collections import namedtuple
import matplotlib.pyplot as plt
import time
"""
you can import any package and define any extrak function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)

Experience = namedtuple(
        'Experience',
        ('state', 'action', 'next_state', 'reward', 'termination')
    )

class ReplayBuffer():

    def __init__(self, N=10000):
        self.memory = deque(maxlen = N)
        self.capacity = N
        self.len = 0

    def append(self,experience):
        self.memory.append(experience)
        self.len += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_sample(self):
        if self.len >= 5000:
            return True
        else:
            return False

class EpsilonGreedyStrategy():
    def __init__(self, start,end,decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.e_step = 1000000

    def get_exploration_rate(self):
        return (self.start-self.end)/self.e_step



class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            paramters for neural network  
            initialize Q net and target Q net
            parameters for repaly buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN, self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.buffer = ReplayBuffer()
        self.num_action = self.env.get_action_space().n
        self.cur_step = 0
        self.greedyPolicy = EpsilonGreedyStrategy(1, 0.025, 0.01)
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.num_episode = args.num_episode
        self.learning_rate = args.learning_rate
        self.sample_batch_size = args.sample_batch_size
        self.gamma = args.gamma
        self.e = 1
        if args.test_dqn:
            # you can load your model here
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        ###########################
        pass

    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #

        if test:
            with torch.no_grad():
                action = self.policy_net(observation).argmax(dim=1).item()
        else:
            if self.e > random.random():
                action = random.randrange(self.num_action)
            else:
                observation = self.transform(observation)
                with torch.no_grad():
                    action = self.policy_net(observation).argmax(dim=1).item()
            self.e -= self.greedyPolicy.get_exploration_rate()
        ###########################
        return action

    def push(self,experience):
        """ You can add additional arguments as you need. 
        Push new data to buffer and remove the old one if the buffer is full.
        
        Hints:
        -----
            you can consider deque(maxlen = 10000) list
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.buffer.append(experience)
        ###########################

    def replay_buffer(self, batch_size=32):
        """ You can add additional arguments as you need.
        Select batch from buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        experience = self.buffer.sample(batch_size)
        ###########################
        return  experience

    def transform(self, state):
        state = np.asarray(state) / 255.
        state = torch.tensor(state)
        state = state.unsqueeze(0)
        state = state.permute(0, 3, 1, 2)
        state = state.to(device=self.device, dtype=torch.float)
        return state

    def extract_tensors(self, experiences):
        batch = Experience(*zip(*experiences))
        t1 = batch.state
        t2 = batch.action
        t3 = batch.next_state
        t4 = batch.reward
        t5 = batch.termination
        return t1, t2, t3, t4, t5

    def get_current_q(self, states, actions):
        states = np.asarray(states) / 255.
        a = np.count_nonzero(states)
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        states = states.permute(0, 3, 1, 2)
        actions = torch.tensor(np.asarray(actions), device=self.device, dtype=torch.long).unsqueeze(-1)
        QS = self.policy_net(states).gather(1,  actions)#.requires_grad_(True)
        QS = QS.permute(1, 0)
        return QS[0]

    def get_next_q(self, next_states, terminations):
        next_states = np.asarray(next_states) / 255.
        next_states = torch.tensor(next_states,device=self.device, dtype=torch.float)
        next_states = next_states.permute(0, 3, 1, 2)
        QS = self.target_net(next_states).max(1)[0].detach()#.requires_grad_(True)
        QS = QS * torch.tensor(terminations, device=self.device, dtype=torch.float, requires_grad= True)

        return QS

    def train(self):
        """
        Implement your training algorithm here
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        optimizor = optim.Adam(params=self.policy_net.parameters(), lr = self.learning_rate,
                               betas = (0.5, 0.999))
        max_reward = -1
        rewards_sum = 0
        reward_collection = []
        episode_collection = []
        print(self.device)

        for episode in range(self.num_episode):
            done = False
            state = self.env.reset()
            while not done:
                action = self.make_action(state, False)
                next_state, reward, done, info = self.env.step(action)
                self.push(
                    Experience(state,
                               action,
                               next_state,
                               reward,
                               (not done)
                               )
                          )
                rewards_sum += reward
                state = next_state
                if self.buffer.can_sample():
                    experiences = self.buffer.sample(self.sample_batch_size)

                    states, actions, next_states, rewards, terminations = self.extract_tensors(experiences)

                    current_q = self.get_current_q(states, actions)

                    next_q = self.get_next_q(next_states, terminations)

                    target_q = self.gamma * next_q +  torch.tensor(rewards, device=self.device, dtype=torch.float)

                    loss = F.smooth_l1_loss(current_q, target_q)

                    optimizor.zero_grad()
                    loss.backward()
                    for param in self.policy_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                    optimizor.step()
            if episode % 3000 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            if episode % 30 == 0:
                print("episode: ", episode, "\t", "average reward :", rewards_sum/30)
                reward_collection.append(rewards_sum/30)
                episode_collection.append(episode)

                if rewards_sum > max_reward:
                    torch.save(self.policy_net.state_dict(), "model/policy_net_max_reward.pth")
                rewards_sum = 0
            if episode%1000 == 0:
                torch.save(self.policy_net.state_dict(), "model/policy_net.pth")
        torch.save(self.policy_net.state_dict(), "model/policy_net.pth")
        x = episode_collection
        y = reward_collection
        plt.plot(x,y)
        plt.show()
        plt.savefig('episode-reward.png')
        ###########################
