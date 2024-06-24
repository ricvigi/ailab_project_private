import cv2, os, pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory

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

# return an array of grayscaled, normalized, cropped images
def resize_grayscale_normalize(filelist:list, img_height, img_width) -> list:
    grayscale = [cv2.imread(x,cv2.IMREAD_GRAYSCALE) for x in filelist]
    resized = [cv2.resize(x, (img_height, img_width)) for x in grayscale]
    normalized = [cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F) for x in resized]
    return normalized

# build and train the model
def steps():
    path0 = "/home/rick/Ri/SecondYear/2ndSemester/AI-Lab/project_private/train"
    path1 = "/home/rick/Ri/SecondYear/2ndSemester/AI-Lab/project_private/test"
    data_dir0 = pathlib.Path(path0).with_suffix('')
    data_dir1 = pathlib.Path(path1).with_suffix('')
    
    # count images
    image_count = len(list(data_dir0.glob('*/*.jpg')))
    print(image_count)
    
    batch_size = 32
    img_height = 180
    img_width = 180
    
    # train dataset
    train_ds = image_dataset_from_directory(
      data_dir0,
      validation_split=0.2,
      seed = 123,
      subset="training",
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    # validation dataset
    val_ds = image_dataset_from_directory(
      data_dir0,
      validation_split=.2,
      seed = 123,
      subset="validation",
      image_size=(img_height, img_width),
      batch_size=batch_size)
    
    # test dataset
    test_ds = image_dataset_from_directory(
        data_dir1, 
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    # data augmentation layer
    data_augmentation = tf.keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(img_height,
                                      img_width,
                                      3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )
    
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    
    # model
    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes, name="outputs")
    ])
    
    # build the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    model.summary()
    
    # train the model
    epochs = None
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs
    )

    # save the model
    model_name = "mymodel.keras"
    model.save(model_name)
    return model