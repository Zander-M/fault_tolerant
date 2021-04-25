'''
    Generate Training Data. We use data format stored in the replay buffer to predict 
'''

from argparse import ArgumentParser
import json
import gym
import numpy as np
import mujoco_py
import random
import pickle


def genData(env, joint_error, num_model, num_action, save_dir):
    e = gym.make(env, joint_error=joint_error)
    actionSpaceSize = len(e.action_space.sample())
    models = random.sample(range(3000), k=num_model)
    log = dict()
    for i, model_idx in enumerate(models):
        # load model
        log[i] = dict()
        xml_file = e.env._xml_path + "front_left_ankle_{}/{}.xml".format(
            e.env._joint_error, model_idx)
        json_file = e.env._xml_path + "front_left_ankle_{}/{}.json".format(
            e.env._joint_error, model_idx)

        with open(json_file) as f:
            data = json.load(f)
            log[i]["idx"] = model_idx
            log[i]["structure"] = (data["legLength"] + data["ankleLength"])
            log[i]["data"] = [] 
        e.model = mujoco_py.load_model_from_path(xml_file)
        e.sim = mujoco_py.MjSim(e.model)
        e.data = e.sim.data
        curr_obs, reward, done = e.reset(), 0, False
        for _ in range(num_action):
            action = e.action_space.sample()
            next_obs, reward, done, _ = e.step(action)
            log[i]["data"].append([curr_obs, action, next_obs, reward, done])
            if done:
                curr_obs, reward, done = e.reset(), 0, False
            curr_obs = next_obs
    print(joint_error, num_action)
    with open(save_dir+"/er{}_ep{}_m{}.pkl".format(joint_error, num_action, num_model), "wb") as f:
        pickle.dump(log,f)
    print("Done")

if __name__ == '__main__':
    import argparse
    parser = ArgumentParser()
    parser.add_argument("--env", type=str, default="gym_fault:fault-v0")
    parser.add_argument("--joint_error", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="../data/train_data")
    parser.add_argument("--num_action", type=int, default=1000)
    parser.add_argument("--num_model", type=int, default=1000)
    args = parser.parse_args()

    genData(args.env,
            joint_error=args.joint_error,
            save_dir=args.save_dir,
            num_model=args.num_model,
            num_action=args.num_action,
            )
