'''
    Record actions of the gym environment
'''

import gym
from gym.wrappers import Monitor
env = Monitor(gym.make('gym_fault:fault-v0'), '../data/videos', force=True)
i = 0
env.env._randomize = True
env.reset()
while i < 6000:
    i+=1
    action = env.action_space.sample()
    state_next, reward, done, info = env.step(action)
    if done:
        env.reset()
env.close() 