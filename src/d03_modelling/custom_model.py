# Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, \
    Input, GlobalAveragePooling2D


@tf.custom_gradient
def hard_scaled_sigmoid(x, t=0.1):
    '''
        Method to create a custom sigmoid gradient with a discrete output
        
        ...
        
        Attributes
        ----------
        x : Float
            Value to be activated by the sigmoid function
        t : Float
            Factor tau, by which the slope of the sigmoid function can be changed (default 0.1)
            t should be greater than 0 and smaller or equals than one
            a small t results in a strong slope
            
        Output
        ------
        y_hard : Boolean
            Discrete output either 0 or 1
        hard_scaled_sigmoid_gradient : Float
            Downstream gradient for backpropagation
        
    '''

    # Define threshold to discretized values
    threshold = 0.5

    # Calculate sigmoid function with updated input
    x = x / t
    y_soft = tf.keras.activations.sigmoid(x)

    # Discretized soft labels
    y_hard = tf.cast((y_soft > threshold), tf.float32)

    # Use soft labels to calculate down stream gradient
    def hard_scaled_sigmoid_gradient(dy):
        return y_soft * (1 - y_soft) * (1 / t) * dy

    return y_hard, hard_scaled_sigmoid_gradient


def create_resnet50(input_shape=(256, 256, 3), n_classes=45, discrete=True, n_features=128):
    '''
        Method to create a ResNet50 with custom layers to train on satellite images
        
        ...
        
        Attributes
        ----------
        input_shape : Triple
            Size of the Input in the form (height, width, channels) (default (256, 256, 3))
        n_classes : Integer
            Define number of classes in the used dataset (default 45 - NWPU-RESISC45 Dataset)
        discrete : Boolean
            Should the model discrete the features
        n_features : Integer
            Define size of feature vector (default 128)
            
        Output
        ------
        model : Keras Model
        
    '''
    # Create base-model
    model_resnet_base = keras.applications.ResNet50(include_top=False, input_shape=input_shape)

    # Create model with base and added layers
    img_input = Input(shape=(256, 256, 3))
    x = model_resnet_base(img_input)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    # Check if output should be discretized
    if discrete:
        x = Dense(n_features, activation=hard_scaled_sigmoid)(x)
    else:
        x = Dense(n_features)(x)

    x = Dropout(0.5)(x)
    img_class_output = Dense(n_classes, activation='softmax')(x)

    # Define model inputs and outputs
    model = Model(inputs=img_input, outputs=img_class_output)

    return model


def create_custom_model(input_shape=(32, 32, 3), n_classes=10, discrete=True, n_features=128):
    '''
        Method to create a ResNet50 with custom layers to train on satellite images
        
        ...
        
        Attributes
        ----------
        input_shape : Triple
            Size of the Input in the form (height, width, channels) (default (32, 32, 3))
        n_classes : Integer
            Define number of classes in the used dataset (default 10 - CIFAR-10 Dataset)
        discrete : Boolean
            Should the model discrete the features
        n_features : Integer
            Define size of feature vector (default 128)
            
        Output
        ------
        model : Keras Model
        
    '''
    if discrete:
        # create normal model
        model = Sequential([
            Conv2D(32, 3, activation="relu", input_shape=input_shape),
            Conv2D(32, 3, activation="relu"),
            MaxPooling2D(2),
            Dropout(0.2),
            Conv2D(64, 3, activation="relu"),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(2),
            Dropout(0.2),
            Conv2D(128, 3, activation="relu"),
            Conv2D(128, 3, activation="relu"),
            Dropout(0.2),
            BatchNormalization(),
            Flatten(),
            Dense(n_features, activation=hard_scaled_sigmoid),
            Dropout(0.2),
            Dense(n_classes, activation="softmax")
        ])
    else:
        # create normal model
        model = Sequential([
            Conv2D(32, 3, activation="relu", input_shape=input_shape),
            Conv2D(32, 3, activation="relu"),
            MaxPooling2D(2),
            Dropout(0.2),
            Conv2D(64, 3, activation="relu"),
            Conv2D(64, 3, activation="relu"),
            MaxPooling2D(2),
            Dropout(0.2),
            Conv2D(128, 3, activation="relu"),
            Conv2D(128, 3, activation="relu"),
            Dropout(0.2),
            BatchNormalization(),
            Flatten(),
            Dense(n_features),
            Dropout(0.2),
            Dense(n_classes, activation="softmax")
        ])

    return model
