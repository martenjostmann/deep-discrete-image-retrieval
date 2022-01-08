# Import Libraries
import sys

# Add path
sys.path.append("..")

from src.d01_data import load_data
from src.d02_processing import preprocess_data
from src.d03_modelling import nearest_neighbor as nn
from src.d04_visualisation import plot_nearest_neighbor
from src.d05_evaluation import nearest_neighbor_performance as nnp

from tensorflow import keras
import numpy as np
import os


class query_lsh(object):

    def __init__(self, model_path=None, layer_index=None, dataset='cifar', num_perm=128):

        '''
            Attributes
            ----------
            model_path: String
                Path of a model that should be used
            layer_index: Integer
                Index of the layer where the features should be extracted (only is model_path is not None)
            dataset: String
                Dataset that should be used for searching nearest neighbors (default 'cifar' other option is 'resisc')

        '''
        self.dataset = dataset
        self.model_path = model_path
        self.layer_index = layer_index
        self.num_perm = num_perm

        print("Loading and preprocessing data...")

        if self.dataset == 'cifar':
            # Load Cifar Data
            X_train, y_train, X_test, y_test = load_data.load_cifar_10()
            # Preprocess Cifar Data
            X_train, X_test = preprocess_data.preprocess_cifar_10(X_train, X_test)

            self.images = np.concatenate((X_train, X_test), axis=0)
            self.y_all = np.concatenate((y_train, y_test), axis=0)

            # Either load features or load model to extract features
            if self.model_path is None:
                feature_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'features', 'discrete_features.npy')
                if os.path.exists(feature_path):
                    print("Load features")
                    with open(feature_path, 'rb') as f:
                        self.y_pred = np.load(f)
                else:
                    self.layer_index = 13
                    self.model_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'experiment',
                                                   'model_exp_discrete_48')


        elif self.dataset == 'resisc':
            # Load Satellite Data
            path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'resisc45')

            if os.path.exists(path) != True:
                raise ValueError('Path does not exist. Change path or download the resisc45 dataset first')

            self.images = load_data.load_satellite_data(path, split=False)

            # Preprocess Satellite Data
            self.images = preprocess_data.preprocess_satellite_data(all_data=self.images)

            self.y_all = np.concatenate([y for x, y in self.images], axis=0)

            # Either load features or load model to extract features
            if self.model_path is None:
                feature_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'features',
                                            'sat_discrete_features.npy')
                if os.path.exists(feature_path):
                    print("Load features")
                    with open(feature_path, 'rb') as f:
                        self.y_pred = np.load(f)
                else:
                    self.layer_index = 5
                    self.model_path = os.path.join(os.path.dirname(os.getcwd()), 'models', 'experiment',
                                                   'model_exp_sat_discrete_512')

        # Load model and extract features if no model_path for a custom model is given
        if self.model_path is not None:
            print("Load model...")
            self.model = keras.models.load_model(self.model_path)
            self.model = keras.Model(self.model.input, self.model.get_layer(index=self.layer_index).output)

            print("Create features... (This will take a while)")
            # Create features
            self.y_pred = self.model.predict(self.images, workers=20)

        # Build LSH
        self.wmg, self.lsh, self.min_hashes = nn.lsh(pred=self.y_pred, num_perm=self.num_perm)

    def query(self, image=None, image_idx=None, k=10, plot=True, verbose=True):
        '''
            Method to search LSH when a query image is given

            ...

            Attributes
            ----------
            image : Numpy Array
                An array with a query image
            image_idx : Integer
                Index of the image to be searched
            k : Integer
                Number of similar images (default 10)
            plot : Boolean
                should the k nearest neighbors be plotted? (default True)
            verbose : Boolean
                Should there be a status output? (default True)

            Output
            ------
            knn : Numpy Array
                An Array with indices of nearest neighbors
            plot :
                If plot is true the k nearest neighbors will be plotted

        '''

        if image is not None:
            knn = nn.image_query_lsh(self.lsh, image, self.model, self.wmg, k=k, verbose=verbose)
        else:
            if image_idx is not None:
                knn = nn.query_lsh(self.lsh, self.y_pred[image_idx], self.wmg, k=k, verbose=verbose)

            else:
                raise ValueError('image or image_idx expected')

        if plot:
            if self.dataset == 'cifar':
                plot_nearest_neighbor.plot_cifar10(self.images, knn)
            else:
                plot_nearest_neighbor.plot_satellite_images(self.images, knn)

        return knn

    def performance(self, k=10):
        '''
            Method to measure the performance

            ...

            Attributes
            ----------
            k : Integer
                Number of similar images (default 10)
                
            Output
            ------
            class_proba : List
                A list of prediction accuracy for each class
            accuracy : Float
                Overall search accuracy
            time : Float
                Time required for the search
                
                
            
        '''
        accuracy, time = nnp.lsh_accuracy(self.lsh, self.min_hashes, self.y_all, k=k)

        return accuracy, time
