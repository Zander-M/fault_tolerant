import gym
import numpy as np

e = gym.make("gym_fault:fault-v0")
print(e.model.geom_size)
actionSpaceSize = len(e.action_space.sample())
noAction = np.zeros(actionSpaceSize)
while (1):
    # action = e.action_space.sample()
    action = noAction # put actions here
    e.model.geom_size[4][1] += 1e-4 # change model on the fly! 
    e.model.geom_rgba[4] = np.array([1., 0., 0., 1.], dtype='float32') # change fault color
    observation, reward, done, info = e.step(action)
    e.render()
    if done:
        e.reset() 
e.reset()