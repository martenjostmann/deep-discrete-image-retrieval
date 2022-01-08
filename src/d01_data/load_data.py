# Import libaries
import os
import numpy as np
import tensorflow as tf
import splitfolders
from tensorflow import keras


def load_cifar_10(split=True):
    '''
        Function to load cifar-10 images
        
        ...
        
        Attributes
        ----------
        split : Boolean
            Should the entire data set or a data set divided into training and test data be returned (default = True)
        
        Output
        ------
        X_train : Numpy Array
            Train-images (Only if split is True)
        y_train : Numpy Array
            Train-labels (Only if split is True)
        X_test : Numpy Array
            Test-images (Only if split is True)
        y_test : Numpy Array
            Test-labels (Only if split is True)
        X_all : Numpy Array
            All Images (Only if split is False)
        y_all : Numpy Array
            All Labels (Only if split is False)
        
    '''
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    if split:
        return X_train, y_train, X_test, y_test
    else:
        X_all = np.concatenate((X_train, X_test), axis=0)
        y_all = np.concatenate((y_train, y_test), axis=0)

        return X_all, y_all


def load_satellite_data(path, batch_size=100, split=True, validation_split=0.15, seed=123, img_height=256,
                        img_width=256):
    '''
        Function to load satellite images (NWPU-RESISC45)
        
        ...
        
        Attributes
        ----------
        path : String
            Location of unsplitted data
        batch_size : Integer
            Size of the batches (default 100)
        split : Boolean
            True if data should be splitted into train, val and test to train the model otherwise only one dataset is returned (default True)
        validation_split : Float
            How to split the data set (only necessary when split is True, default 0.15)
        seed : Integer
            Make split reproducible (default 123)
        img_height : Integer
            Height of the image (default 256)
        img_width : Integer
            Width of the image (default 256)
        
            
        Output
        ------
        train: Tensorflow Dataset
            Trainingsdataset (Only if split is True)
        val: Tensorflow Dataset
            Validationdataset (Only if split is True)
        test: Tensorflow Dataset
            Testdataset (Only if split is True)
        data: Tensorflow Dataset
            Dataset with all satellite images (Only if split is False)
        
    '''

    if not os.path.exists(path):
        raise ValueError('Path does not exist. Change path or download the resisc45 dataset first')

    # Check if a splitted Dataset is required for training
    if split:

        # Check if splitted dataset is available or create it
        # Get parent directory
        parent_path = os.path.dirname(path)
        resisc45_split_path = os.path.join(parent_path, 'resisc45_split')

        # Check if folder exists
        if not os.path.isdir(resisc45_split_path):
            print("Split Dataset...")
            # Create folder and split data
            splitfolders.ratio(path, output=resisc45_split_path, seed=1337, ratio=(0.8, 0.2), group_prefix=None)

        # Path for train and validation Date
        path_train = os.path.join(resisc45_split_path, 'train')
        path_test = os.path.join(resisc45_split_path, 'val')

        # Get train data
        train = tf.keras.preprocessing.image_dataset_from_directory(path_train,
                                                                    validation_split=validation_split,
                                                                    subset="training",
                                                                    seed=seed,
                                                                    image_size=(img_height, img_width),
                                                                    batch_size=batch_size)

        # Get validation data
        val = tf.keras.preprocessing.image_dataset_from_directory(path_train,
                                                                  validation_split=validation_split,
                                                                  subset="validation",
                                                                  seed=seed,
                                                                  image_size=(img_height, img_width),
                                                                  batch_size=batch_size)

        # Get test data
        test = tf.keras.preprocessing.image_dataset_from_directory(path_test,
                                                                   image_size=(img_height, img_width),
                                                                   batch_size=batch_size)

        return train, val, test

    else:
        # Get all satellite images
        data = tf.keras.preprocessing.image_dataset_from_directory(path,
                                                                   shuffle=False,
                                                                   image_size=(img_height, img_width),
                                                                   batch_size=batch_size)
        return data
