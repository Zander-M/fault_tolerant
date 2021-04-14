'''
    Predict hardware structure of the robot using Multilayer Perceptron.
'''

import torch
import numpy as np

data = [[1,2], [3,4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
















class MLP:
    '''
        Using a multilayer perceptron to predict the structure of the robot.
    '''
    def __init__(self) -> None:
        pass

    def loadData(self, path):
        '''
            Load Training Data.
        '''
        pass