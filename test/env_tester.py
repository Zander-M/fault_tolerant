import gym
import numpy as np
import mujoco_py
import random

e = gym.make("gym_fault:fault-v0")
print(e.model.geom_size)
actionSpaceSize = len(e.action_space.sample())
noAction = np.zeros(actionSpaceSize)
# e.model.geom_rgba[4] = np.array(
    # [1., 0., 0., 1.], dtype='float32')  # change fault color
i = 0
while (1):
    i += 0
    # if i > 500:
    #     # e._random_model()
    #     print("qpos: ", e.sim.data.qpos)
    #     print("xpos: ", e.sim.data.get_body_xpos())
    #     i = 0
    action = e.action_space.sample()
    # action = noAction  # put actions here
    # e.model.geom_size[4][1] -= 1e-3  # change model on the fly!
    # e.model.geom_pos[4][1] += 1e-4  # change model on the fly!
    # e.model.body_ipos[4][2] += 1e-3  # change model on the fly!
    # ximat = e.sim.data.get_body_ximat("front_right_leg")
    observation, reward, done, info = e.step(action)
    e.render()
    i += 1
    if done:
        e.reset()
e.reset()
