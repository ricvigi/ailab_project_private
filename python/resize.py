import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


path = '/run/media/rick/MY_DISK/datasets/airport_security_scans/train/Folding Knife'


files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]

grayscale = [cv2.imread(os.path.join(path, x),cv2.IMREAD_GRAYSCALE) for x in files]

resized = [cv2.resize(x, (400,400)) for x in grayscale]


normalized = [cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F) for x in resized]

def resize_grayscale_normalize(filelist:list) -> list:
    grayscale = [cv2.imread(x,cv2.IMREAD_GRAYSCALE) for x in filelist]
    resized = [cv2.resize(x, (400,400)) for x in grayscale]
    normalized = [cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F) for x in resized]
    return normalized


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

def show_augmented_img(image) -> None:
    if not (len(image.shape) == 3):
        image = tf.cast(tf.expand_dims(image, 0), tf.float32) # This is necessary for data_augmentation to work

    plt.figure(figsize=(10, 10))
    for i in range(9):
        augmented_image = data_augmentation(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0])
        plt.axis("off")

    plt.show()
    return



augmented = [data_augmentation(x) for x in normalized]

model = tf.keras.Sequential([
  # Add the preprocessing layers you created earlier.
  data_augmentation,
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  # Rest of your model.
])
