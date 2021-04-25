'''
    visualize the performance of the nn
'''

import torch
import torch.nn
from matplotlib import pyplot as plt
import numpy as np
from pred_struct import Mlp
from faultDataset import FaultDataset

if __name__ == "__main__":

    # load model
    model = Mlp(238, 8, [512, 512, 512])
    model.load_state_dict(torch.load("predStruct.pth"))
    model.eval()
    real = []
    pred = []

    data = FaultDataset(joint_error=0.5, num_model=1000)

    for i in range(len(data)):
        X, y = data[i]
        pred.append(model(X.float())[4])
        real.append(y[4])
    
    t = np.argsort(np.argsort(np.array(real), axis=None))
    plt.plot(t, real, 'bo')
    plt.plot(t, pred, 'r+')
    plt.show()
