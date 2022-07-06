import os
import random
import numpy as np

# 一个类: ReplayMemory，需要注意的是每一个agent都有一个memory，在main_marl_train.py--class Agent可以看到
class ReplayMemory:
    def __init__(self, entry_size):
        self.entry_size = entry_size
        self.memory_size = 200000
        self.actions = np.empty(self.memory_size, dtype = np.uint8)
        self.rewards = np.empty(self.memory_size, dtype = np.float64)
        self.prestate = np.empty((self.memory_size, self.entry_size), dtype = np.float16)
        self.poststate = np.empty((self.memory_size, self.entry_size), dtype = np.float16)
        self.batch_size = 2000
        self.count = 0
        self.current = 0
        
    def add(self, prestate, poststate, reward, action):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size
        
    def sample(self):

        if self.count < self.batch_size:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0,self.count), self.batch_size)
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        return prestate, poststate, actions, rewards
   
