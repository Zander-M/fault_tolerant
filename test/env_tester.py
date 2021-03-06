import gym
import numpy as np
import mujoco_py
import random

e = gym.make("gym_fault:fault-v0", randomize=True, ankle_error=0.6, leg_error=0.7)
print(e.model.geom_size)
actionSpaceSize = len(e.action_space.sample())
noAction = np.zeros(actionSpaceSize)
# e.model.geom_rgba[4] = np.array(
    # [1., 0., 0., 1.], dtype='float32')  # change fault color
i = 0
total_reward = 0
e.reset()
while (1):
    i += 0
    if i > 500:
        print("qpos: ", e.sim.data.qpos)
        print("xpos: ", e.sim.data.get_body_xpos("torso"))
        print("avg_reward: ", total_reward/500)
        i = 0
        total_reward = 0
    action = e.action_space.sample()
    observation, reward, done, info = e.step(action)
    e.render()
    i += 1
    total_reward += reward
    if done:
        e.reset()
e.reset()
