'''
    Visualize the generated model by loading MJCF
'''

import mujoco_py
import os

# change model here

XML_PATH = "/home/zdrrrm/Desktop/Capstone/fault_tolerant/data/models/front_left/0.xml"

def visModel(xml):
    model = mujoco_py.load_model_from_path(xml)
    sim = mujoco_py.MjSim(model)
    view = mujoco_py.MjViewerBasic(sim)
    while(1):
        view.render()

if __name__ == "__main__":
    os.chdir("../")
    visModel(XML_PATH)