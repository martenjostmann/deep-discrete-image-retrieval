# Import libaries
import tensorflow as tf
from tensorflow import keras


def to_categorical(x, y):
    y = tf.one_hot(tf.cast(y, tf.int32), 10)
    y = tf.reshape(y, (10,))
    return x, y


def change_type(x, y, tf_type):
    x = tf.cast(x, tf_type)
    return x, y


def preprocess_cifar_10(X_train, X_test):
    '''
        Function to preprocess cifar-10 images
        
        ...
        
        Attributes
        ----------
        X_train: Numpy Array
            Train-images
        X_test: Numpy Array
            Test-images
            
        Output
        ------
        X_train: Numpy Array
            Train-images preprocessed
        X_test: Numpy Array
            Test-images preprocessed
        
    '''

    # Parse numbers as floats
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize data
    X_train = X_train / 255
    X_test = X_test / 255

    return X_train, X_test


def preprocess_satellite_data(train=None, val=None, test=None, all_data=None):
    '''
        Function to preprocess satellite images (NWPU-RESISC45)
        
        ...
        
        Attributes
        ----------
        train: Tensorflow Dataset
            Trainingsdataset
        val: Tensorflow Dataset
            Validationdataset
        test: Tensorflow Dataset
            Testdataset
        data: Tensorflow Dataset
            Dataset with all satellite images
        
            
        Output
        ------
        train: Tensorflow Dataset
            Trainingsdataset preprocessed (only if not None)
        val: Tensorflow Dataset
            Validationdataset preprocessed (only if not None)
        test: Tensorflow Dataset
            Testdataset preprocessed (only if not None)
        data: Tensorflow Dataset
            Dataset with all satellite images preprocessed (only if not None)
        
    '''

    # Define normalization Layer
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    # Check if all_data or train, val and test should be preprocessed
    if all_data is not None:
        # Preprocess
        all_data = all_data.map(lambda x, y: (normalization_layer(x), y))

        # Optimize performance
        AUTOTUNE = tf.data.AUTOTUNE
        all_data = all_data.prefetch(buffer_size=AUTOTUNE).cache()

        return all_data

    else:

        if train is not None:
            train = train.map(lambda x, y: (normalization_layer(x), y))

            # Initialize data augmentation
            data_augmentation = tf.keras.Sequential([
                keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                keras.layers.experimental.preprocessing.RandomRotation(0.2),
            ])

            # Add data augmentation
            train = train.map(lambda x, y: (data_augmentation(x), y))

            if val is not None:
                val = val.map(lambda x, y: (normalization_layer(x), y))

                if test is not None:
                    test = test.map(lambda x, y: (normalization_layer(x), y))

                    # Optimize performance
                    AUTOTUNE = tf.data.AUTOTUNE
                    train = train.prefetch(buffer_size=AUTOTUNE).cache().repeat(2)
                    val = val.prefetch(buffer_size=AUTOTUNE)
                    test = test.prefetch(buffer_size=AUTOTUNE)

                    return train, val, test

                else:
                    raise ValueError('No test dataset found')
            else:
                raise ValueError('No validation dataset found')
        else:
            raise ValueError('No train dataset found')
