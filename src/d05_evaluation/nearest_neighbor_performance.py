# Import libraries
import time
import numpy as np
from sklearn.neighbors import KDTree
from src.d03_modelling import nearest_neighbor as nn


def KDTree_accuracy(kdt, pred, y_true, k=10):
    '''
        Measure the accuracy of the KDTree nearest neighbor search
        
        ...
        
        Attributes
        ----------
        kdt : KDTree
            A KDTree from the scikit-learn library
        pred : Numpy Array
            An array with n continuous features that describes an image
        y_true : Numpy Array
            An array with the true classes
        k : int
            Number of similar images (default 10)
            
        Output
        ------
        class_proba : List
            A list of prediction accuracy for each class
        accuracy : Float
            Accuracy over all classes
        time_diff : Float
            Time required for the search        
        
    '''

    start = time.time()
    # Initial list of accuracies
    accuracy_list = []
    # Loop over each instance and search for nearest neighbors
    num_it = pred.shape[0]
    for i in range(num_it):
        # Display Progress every 1000 iterations
        if (i % 1000) == 0:
            print("Iteration {} / {}".format(i, num_it))

        # Select query image
        query = pred[i]

        # Reshape query image
        query = np.reshape(query, (1, -1))

        # Search for nearest neighbors
        index = nn.query_KDTree(query, kdt, k, return_distance=False, verbose=False)

        # Select true class
        true = y_true[i]

        # Select class of knn indices
        knn_pred = y_true[index]

        # Calculate and append accuracy for this class
        accuracy_list.append((np.sum(knn_pred == true)) / (index.shape[1]))

    end = time.time()
    time_diff = end - start
    print("{} searches for the {} nearest neighbors was completed in {} Seconds".format(num_it, k, end - start))

    # Calculate mean
    accuracy = sum(accuracy_list) / len(accuracy_list)

    # Count number of instances per class
    num_classes = calculate_num_class(y_true)

    # Transform List to numpy array
    accuracy_array = np.array(accuracy_list)

    # Combine calculated accuracy with actual class
    accuracy_class = np.array(list(zip(accuracy_array, y_true)))

    # Calculate mean of accuracy for each class
    class_proba = [sum([x[0] for x in accuracy_class if x[1] == y]) / num_classes[y] for y in
                   range(num_classes.shape[0])]

    print("The accuracy is {}".format(accuracy))

    return class_proba, accuracy, time_diff


def lsh_accuracy(lsh, min_hashes, y_true, k=10):
    '''
        Measure the accuracy of the LSH nearest neighbor search
        
        ...
        
        Attributes
        ----------
        lsh : MinHashLSH
            LSH class to search nearest neighbors
        min_hashes : List
            List of min-hashes
        y_true : Numpy Array
            An array with the true classes
        k : int
            Number of similar images (default 10)
            
        Output
        ------
        accuracy : Float
            Accuracy over all classes
        time_diff : Float
            Time required for the search
        
    '''

    start = time.time()
    # Initial list of accuracies
    accuracy_list = []
    # Loop over each instance and search for nearest neighbors
    num_it = len(min_hashes)
    for i in range(num_it):

        # Display Progress every 1000 iterations
        if (i % 1000) == 0:
            print("Iteration {} / {}".format(i, num_it))

        # Select query image
        query_min_hash = min_hashes[i]

        # Search for nearest neighbors
        matches = nn.query_lsh(lsh, min_hash=query_min_hash, k=k, verbose=False)

        # Select true class
        true = y_true[i]

        if len(matches) > k:
            matches = matches[0:k]

        # Select class of knn indices
        pred = y_true[matches]

        # Calculate and append accuracy for this class
        accuracy_list.append((np.sum(pred == true)) / (len(matches)))

    end = time.time()
    time_diff = end - start
    print("{} searches for the {} nearest neighbors was completed in {} Seconds".format(num_it, k, end - start))

    # Calculate mean
    accuracy = sum(accuracy_list) / len(accuracy_list)

    print("The accuracy is {}".format(accuracy))

    return accuracy, time_diff


def calculate_num_class(y_true):
    '''
        Count number of instances per class
        
        ...
        
        Attributes
        ----------
        y_true : Numpy Array
            True labels
            
        Output
        ------
        y_true : Numpy Array
           The number of times each of the unique values comes up in the original array
        
    '''
    unique, counts = np.unique(y_true, return_counts=True)

    return counts
