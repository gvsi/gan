#!/usr/bin/python

import os
import sys
import gym
import numpy as np
import random
from gym import envs
from gym.envs.registration import register, spec
from datetime import datetime
from json_tricks import dumps, loads
from tqdm import tqdm
from math import sqrt
import socket

def verify(env, Q, num_episodes = 10000):
    print("Running validation...")
    # Set learning parameters
    #create lists to contain total rewards and steps per episode
    #jList = []
    rList = []
    successes = 0
    jTot = 0
    for i in tqdm(range(num_episodes)):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0
        #The Q-Table learning algorithm
        while j < 200:
            j+=1
            #Choose an action by greedily (with noise) picking from Q table
            a = np.argmax(Q[s,:])
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a)
            rAll += r
            s = s1
            if d == True and r > 0:
                jTot += j
                successes += 1
            if d == True:
                break
        rList.append(rAll)
    print("Score over time: " +  str(sum(rList)/num_episodes))
    valid_score = sum(rList)/num_episodes
    try:
        avg_steps = jTot / successes
    except:
        avg_steps = 0
    return valid_score, avg_steps

class Experiment(object):
    def __init__(self, env, num_episodes=10000):
        self.env = env
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        self.machine = socket.gethostname()
        self.num_episodes = num_episodes
        self.done = False
        self.score = None
        self.start = None
        self.end = None
        self.train_successes = []
        self.valid_score = None
        self.valid_avg_steps = None

    def print_score(self):
        if not self.done:
            print("Run first.")
            return
        print("Score over time: " +  str(self.score))

    def run(self):
        print("Running experiment...")
        if self.done:
            print("Already done running")
            return

        self.start = datetime.now()
        lr = .8
        e = 0.1
        y = .95
        #create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        for i in tqdm(range(self.num_episodes)):
            #Reset environment and get first new observation
            s = self.env.reset()
            rAll = 0
            d = False
            j = 0
            #The Q-Table learning algorithm
            while j < 200:
                j+=1
                #Choose an action by greedily (with noise) picking from Q table
                a = None
                if random.uniform(0,1) < e:
                    a = self.env.action_space.sample()
                else:
                    a = np.argmax(self.Q[s,:])
                #Get new state and reward from environment
                s1,r,d,_ = self.env.step(a)
                if d == True and r != 1:
                    self.Q[s, a] -= 0.01
                #Update Q-Table with new knowledge
                self.Q[s,a] = self.Q[s,a] + lr*(r + y*np.max(self.Q[s1,:]) - self.Q[s,a])
                rAll += r
                s = s1
                if d == True and r > 0:
                    self.train_successes.append((i, j))
                if d == True:
                    #Reduce chance of random action as we train the model.
        #             e = 1./((i/50) + 10)
                    break
            rList.append(rAll)
        self.done = True
        self.end = datetime.now()
        self.score = sum(rList)/self.num_episodes

    def validate(self):
        valid_score, avg_steps = verify(self.env, self.Q)
        self.valid_score = valid_score
        self.valid_avg_steps = avg_steps

    def dumps(self):
        if not self.done:
            print("Run first.")
            return

        return dumps({'Q': self.Q,
                      'start': self.start,
                      'end': self.end,
                      'train_score': self.score,
                      'num_episodes': self.num_episodes,
                      'train_successes': exp.train_successes,
                      'train_machine' : self.machine,
                      'valid_score' : self.valid_score,
                      'valid_avg_steps' : self.valid_avg_steps
                      })

def new_env(env_map, slippery=True, MY_ENV_NAME='MyFrozenLake-v0'):
    if MY_ENV_NAME in envs.registry.env_specs:
        envs.registry.env_specs.pop(MY_ENV_NAME)

    register(
        id=MY_ENV_NAME,
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'is_slippery': slippery, 'desc': env_map},
        max_episode_steps=100,
        reward_threshold=0.78, # optimum = .8196
    )
    env = gym.make(MY_ENV_NAME)
    return env

lines = []

for line in sys.stdin:
    w = line.strip()
    lines.append(w)

random.shuffle(lines)

while lines:
    w = lines.pop()
    if os.path.isfile("data/data4_valid/res/"+w+".txt"):
        continue
    else:
        os.mknod("data/data4_valid/res/"+w+".txt")
        n = int(sqrt(len(w)))
        # split into lines
        map_str = [w[i:i+n] for i in range(0, len(w), n)]
        env = new_env(map_str, slippery=True)
        exp = Experiment(env, num_episodes=10000)
        exp.run()
        exp.validate()
        f = open("data/data4_valid/res/"+w+".txt","w+")
        f.write(w+"\t" + str(exp.dumps()))
        f.close()
