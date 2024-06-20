import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from resize import resize_grayscale_normalize

from image_classification import get_posix_paths

path = '/run/media/rick/MY_DISK/datasets/airport_security_scans/train'

classes = get_posix_paths(path)
normalized = dict()
for key in classes.keys():
    normalized[key] = resize_grayscale_normalize(classes[key])

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

image = tf.cast(tf.expand_dims(normalized[0], 0), tf.float32) # This is necessary for data_augmentation to work
if __name__ == "__main__":
    pass
