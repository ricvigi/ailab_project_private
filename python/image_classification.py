import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from resize import resize_grayscale_normalize

# Get the posix paths of elements to train on
def get_posix_paths(path) -> dict:

    data_dir = pathlib.Path(path).with_suffix('')

    folding_knifes = list(data_dir.glob("Folding Knife/*"))
    multi_tool_knifes = list(data_dir.glob("Multi-tool Knife/*"))
    scissors = list(data_dir.glob("Scissor/*"))
    straight_knifes = list(data_dir.glob("Straight Knife/*"))
    utility_knifes = list(data_dir.glob("Utility Knife/*"))

    res = dict()
    res["fknife"] = folding_knifes
    res["mtknife"] = multi_tool_knifes
    res["scissor"] = scissors
    res["sknife"] = straight_knifes
    res["uknife"] = utility_knifes
    return res
