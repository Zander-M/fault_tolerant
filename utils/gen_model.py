'''
    Generate ant models with varied geom. Save the model's parameter
    in an json file. The faulty parts will be marked red.
    Current implementation only changes the front left leg.
'''

import xml.etree.ElementTree as et
import re
import os
import random
import numpy as np
import pickle

ANKLE_ERROR = 1.0  # error range
LEG_ERROR = 1.0  # error range
# modify this when creating new model!
DATASET_NAME = "ankle{}_leg{}".format(ANKLE_ERROR, LEG_ERROR)
SAVE_PATH = "src/gym-fault/gym_fault/envs/assets/models"
MODEL_PATH = "src/gym-fault/gym_fault/envs/assets/ant.xml"
FAULT_COLOR = "1. 0. 0. 1"  # mark fault parts as red

# see ant.xml
DEFAULT_LEG_LENGTH = 0.2
DEFAULT_ANKLE_LENGTH = 0.4

# changing orientation
ORIENTATION = np.array([[1, 1, 0],
                        [-1, 1, 0],
                        [-1, -1, 0],
                        [1, -1, 0]])


def genModel(modelPath, savePath, datasetName,
             randomLeg=False, randomAnkle=False,
             numModel=1000, seed=0):
    random.seed(seed)
    tree = et.ElementTree(file=modelPath)
    root = tree.getroot()
    geoms = [geom for geom in root.iter("geom")]
    ankles = [ankle for ankle in geoms if ankle.get(
        "name") and re.match(".*ankle_geom$", ankle.get("name"))]
    legs = [leg for leg in geoms if leg.get(
        "name") and re.match(".*leg_geom$", leg.get("name"))]
    legLengths = []
    ankleLengths = []
    for i in range(numModel):
        legLength = DEFAULT_LEG_LENGTH
        ankleLength = DEFAULT_ANKLE_LENGTH
        if randomLeg:
            for j in range(4):
                legLength = (1-LEG_ERROR)*DEFAULT_LEG_LENGTH + \
                    LEG_ERROR * DEFAULT_LEG_LENGTH*random.random()
                legLengths.append(legLength)
                # print(legLength)
                legs[j].set("fromto", "0.0 0.0 0.0 {d[0]} {d[1]} {d[2]}".format(
                    d=legLength*ORIENTATION[j]))
                legs[j].set("rgba", FAULT_COLOR)
        if randomAnkle:
            for j in range(4):
                ankleLength = (1-ANKLE_ERROR)*DEFAULT_ANKLE_LENGTH + \
                    ANKLE_ERROR * DEFAULT_ANKLE_LENGTH*random.random()
                ankleLengths.append(ankleLength)
                # print(ankleLength)
                ankles[j].set("fromto", "0.0 0.0 0.0 {d[0]} {d[1]} {d[2]}".format(
                    d=ankleLength*ORIENTATION[j]))
                ankles[j].set("rgba", FAULT_COLOR)

        # save file
        with open("{}/{}/{}.xml".format(savePath, datasetName, i), "wb") as fb:
            tree.write(fb)
            fb.close()

        with open("{}/{}/{}.pkl".format(savePath, datasetName, i), "w") as f:
            pickle.dump(ankleLengths + legLengths)
            f.close()


if __name__ == "__main__":
    os.chdir("../")  # go to root directory
    if not os.path.isdir("{}/{}".format(SAVE_PATH, DATASET_NAME)):
        os.mkdir("{}/{}".format(SAVE_PATH, DATASET_NAME))
    genModel(MODEL_PATH, SAVE_PATH, DATASET_NAME,
             numModel=3000, randomAnkle=True, randomLeg=True)
