'''
An environment based on OpenAI Gym Ant-v3 environment for testing fault tolerant behaviour.
'''

import os
import numpy as np
from gym import utils
from gym.utils import seeding
from gym.envs.mujoco import mujoco_env
import mujoco_py
import random
import xml.etree.ElementTree as et
from xml.etree.ElementTree import tostring
import re
import pickle
import tempfile

DEFAULT_CAMERA_CONFIG = {
    'distance': 4.0,
}


class FaultEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 data_path=os.path.dirname(__file__)+"/assets/models/",
                 ctrl_cost_weight=0.5,
                 contact_cost_weight=5e-4,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.2, 1.0),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=True,
                 ankle_error=0.1,
                 leg_error=0.1,
                 randomize=False,
                 concat_model=False):
        utils.EzPickle.__init__(**locals())

        self._data_path = data_path
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation)
        self._ankle_error = ankle_error
        self._leg_error = leg_error
        self._randomize = randomize
        self._concat_model = concat_model
        self.xml_file_path = data_path+"ant.xml"
        self.model_struct = [0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4]
        mujoco_env.MujocoEnv.__init__(self, self.xml_file_path, 5)

    @property
    def healthy_reward(self):
        return float(
            self.is_healthy
            or self._terminate_when_unhealthy
        ) * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.sim.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces))
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = (np.isfinite(state).all() and min_z <= state[2] <= max_z)
        return is_healthy

    @property
    def done(self):
        done = (not self.is_healthy
                if self._terminate_when_unhealthy
                else False)
        return done

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._concat_model:
            observations = np.concatenate(
                (position, velocity, contact_force, self.model_struct))
        else:
            observations = np.concatenate(
                (position, velocity, contact_force, np.zeros(8)))

        return observations

    def reset_model(self):

        if self._randomize:
            self._random_model()

        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
            self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        # randomize model
        return observation

    def _random_model(self):
        '''
           Randomly generate a new model based on error margin
        '''
        default_leg_length = 0.2
        default_ankle_length = 0.4

        # changing orientation
        orientation = np.array([[1, 1, 0],
                                [-1, 1, 0],
                                [-1, -1, 0],
                                [1, -1, 0]])

        fault_color = "1. 0. 0. 1"  # mark fault parts as red

        tree = et.ElementTree(file=self.xml_file_path)
        root = tree.getroot()

        geoms = [geom for geom in root.iter("geom")]
        ankles = [ankle for ankle in geoms if ankle.get(
            "name") and re.match(".*ankle_geom$", ankle.get("name"))]
        legs = [leg for leg in geoms if leg.get(
            "name") and re.match(".*leg_geom$", leg.get("name"))]

        legLengths = []
        ankleLengths = []

        for j in range(4):
            legLength = (1-self._leg_error)*default_leg_length + \
                self._leg_error*default_leg_length*random.random()
            legLengths.append(legLength)
            # print(legLength)
            legs[j].set("fromto", "0.0 0.0 0.0 {d[0]} {d[1]} {d[2]}".format(
                d=legLength*orientation[j]))
            legs[j].set("rgba", fault_color)
            ankleLength = (1-self._ankle_error)*default_ankle_length + \
                self._ankle_error * default_ankle_length*random.random()
            ankleLengths.append(ankleLength)
            ankles[j].set("fromto", "0.0 0.0 0.0 {d[0]} {d[1]} {d[2]}".format(
                d=ankleLength*orientation[j]))
            ankles[j].set("rgba", fault_color)

        model_xml = tostring(root, encoding="unicode")
        self.model = mujoco_py.load_model_from_xml(model_xml)
        self.model_struct = ankleLengths + legLengths
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        if self.viewer is not None:
            self.viewer.update_sim(self.sim)

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

    def curr_struct(self):
        '''
            return current model structure
        '''
        return self.model_struct
