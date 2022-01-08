# Import libraries
import matplotlib.pyplot as plt
import numpy as np


def plot_cifar10(images, knn, cols=10):
    '''
        Function to plot the images of the k nearest neighbors of the the cifar-10 images
        
        ...
        
        Attributes
        ----------
        images: Numpy Array
            Array of Images
        knn: Numpy Array
            Indices of the nearest neighbors that should be plotted
        
            
        Output
        ------
        Figure with nearest neighbors
        
    '''
    rows = np.ceil(len(knn) / cols)
    fig = plt.figure(figsize=(16, 16 / (cols / rows)))
    for idx, i in enumerate(knn):
        fig.add_subplot(rows, cols, idx + 1)
        plt.imshow(images[i])
        plt.axis("off")


def plot_satellite_images(images, knn, batch_size=100, cols=10):
    '''
        Function to plot the images of the k nearest neighbors of the satellite images
        
        ...
        
        Attributes
        ----------
        images: Tensorflow Dataset
            Dataset with all images in it
        knn: Numpy Array
            Indices of the nearest neighbors that should be plotted
        batch_size: Integer
            Size of the batches of _data (default 100)
        cols: Integer
            Number of columns of the figure. The number of rows is determined by the number of images
        
            
        Output
        ------
        Figure with nearest neighbors
        
    '''
    rows = np.ceil(len(knn) / cols)
    fig = plt.figure(figsize=(16, 16 / (cols / rows)))
    for idx, i in enumerate(knn):
        sample_idx = i % batch_size
        batch_idx = np.floor(i / batch_size)

        for x, y in images.take(batch_idx + 1).skip(batch_idx):
            fig.add_subplot(rows, cols, idx + 1)
            plt.imshow(x[sample_idx])
            plt.axis("off")
