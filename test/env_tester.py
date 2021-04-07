import gym
import mujoco_py

e = gym.make("gym_fault:fault-v0")
while (1):
    action = e.action_space.sample()
    observation, reward, done, info = e.step(action)
    e.render()
    if done:
        e.reset() 
e.reset()