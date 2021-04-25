'''
    Fault Tolerant dataset for predicting structure.
'''

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import pickle
import json


class FaultDataset(Dataset):

    def __init__(self, model_path=os.path.dirname(__file__)+"/../../../data/train_data", joint_error=0.1, num_action=100, num_model=100):
        self._joint_error = joint_error
        self._num_action = num_action
        self._num_model = num_model

        # load model
        with open(model_path + "/er{}_ep{}_m{}.pkl".format(self._joint_error, self._num_action, self._num_model), "rb") as f:
            self.data = pickle.load(f)

    # model count times action count
    def __len__(self):
        return self._num_model * self._num_action

    def __getitem__(self, index):
        assert index < self._num_action * self._num_model, "Index out of range"
        model_idx = index // self._num_action
        data_idx = index % self._num_action
        struct = torch.tensor(self.data[model_idx]["structure"]) * 10000
        data = torch.from_numpy(self.flat_data(
            self.data[model_idx]["data"][data_idx])) * 10000
        # sample = {"structure": struct, "data": data}
        sample = (data, struct) 
        return sample

    def flat_data(self, data):
        '''
            Flatten stored data. Convert into numpy array
            [o1,act,o2,rew,done], dim=114+8+114+1+1 = 238

        '''
        np_reward = np.array([data[3]])
        np_done = np.ones(1) if data[4] else np.zeros(1)
        np_data = np.concatenate(
            (data[0], data[1], data[2], np_reward, np_done))
        return np_data
