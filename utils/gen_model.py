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
import json

ERROR = 0.5 # error range
DATASET_NAME = "front_left_ankle_{}".format(ERROR)  # modify this when creating new model!
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
    for i in range(numModel):
        legLength = DEFAULT_LEG_LENGTH
        ankleLength = DEFAULT_ANKLE_LENGTH
        if randomLeg:
            legLength = (1-ERROR)*DEFAULT_LEG_LENGTH + ERROR * DEFAULT_LEG_LENGTH*random.random()  # left front leg
            # print(legLength)
            legs[0].set("fromto", "0.0 0.0 0.0 {d[0]} {d[1]} {d[2]}".format(
                d=legLength*ORIENTATION[0]))
            legs[0].set("rgba", FAULT_COLOR)
        if randomAnkle:
            ankleLength = (1-ERROR)*DEFAULT_ANKLE_LENGTH + ERROR * DEFAULT_ANKLE_LENGTH*random.random()
            # print(ankleLength)
            ankles[0].set("fromto", "0.0 0.0 0.0 {d[0]} {d[1]} {d[2]}".format(
                d=ankleLength*ORIENTATION[0]))
            ankles[0].set("rgba", FAULT_COLOR)

        # save file
        with open("{}/{}/{}.xml".format(savePath, datasetName, i), "wb") as fb:
            tree.write(fb)
        fb.close()

        d = {
            "legLength": [legLength, DEFAULT_LEG_LENGTH, DEFAULT_LEG_LENGTH, DEFAULT_LEG_LENGTH],
            "ankleLength": [ankleLength, DEFAULT_ANKLE_LENGTH, DEFAULT_ANKLE_LENGTH, DEFAULT_ANKLE_LENGTH]
        }
        with open("{}/{}/{}.json".format(savePath, datasetName, i), "w") as f:
            json.dump(d, f)
        f.close()
    return

if __name__ == "__main__":
    os.chdir("../")  # go to root directory
    if not os.path.isdir("{}/{}".format(SAVE_PATH, DATASET_NAME)):
        os.mkdir("{}/{}".format(SAVE_PATH, DATASET_NAME))
    genModel(MODEL_PATH, SAVE_PATH, DATASET_NAME,
             numModel=3000, randomAnkle=True)
    print("Done. Press Ctrl+C to exit")