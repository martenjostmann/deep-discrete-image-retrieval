# Import libraries
import time
from sklearn.neighbors import KDTree
import numpy as np
from datasketch import MinHashLSH, WeightedMinHashGenerator, MinHashLSHForest
from joblib import Parallel, delayed


def build_KDTree(model=None, data=None, pred=None, leaf_size=30, metric='euclidean'):
    '''
        Method to build a KDTree
        
        Either a feature vector is given, or a model and data to determine the feature vector
        
        ...
        
        Attributes
        ----------
        model : Keras Model
            A trained model with n feature output (optional)
        data : Numpy Array
            An array with data points with the same shape as the model input shape (optional)
        pred : Numpy Array
            An array with n continuous features that describes an image (optional)
        leaf_size: Integer
            The number of leaves at the end of the tree (default 30)
        metric : String
            String of the distance metric (default 'euclidean')
            
        Output
        ------
        kdt : KDTree
            A KDTree from the scikit-learn library
        pred : Numpy Array
            Feature vector if it was created during method execution
        
    '''
    if data is not None:
        if model is not None:
            # Make predictions
            print('Create feature vector')
            pred = model.predict(data)
        else:
            raise ValueError('Model expected')
    elif pred is None:
        raise ValueError('Data or feature vector expected')

    start = time.time()
    print('Start building KDTree')

    # Build KDTree
    kdt = KDTree(pred, leaf_size=leaf_size, metric=metric)

    end = time.time()
    print('KDTree finished in {} seconds'.format(end - start))

    # Return statements
    if data is not None:
        return pred, kdt
    else:
        return kdt


def query_KDTree(query, kdt, k=10, return_distance=False, verbose=True):
    '''
        Method to search in a KDTree
        
        ...
        
        Attributes
        ----------
        query : Numpy Array
            An array with features of a query image
        kdt : KDTree
            A KDTree from the scikit-learn library
        k : Integer
            Number of similar images (default 10)
        return_distance : Boolean
            Should a distance be returned (default False)
            
        Output
        ------
        dist : Numpy Array
            An Array with distances
        index : Numpy Array
            An Array with indices of nearest neighbors
        
    '''
    start = time.time()
    if verbose:
        print('Start searching in KDTree')

    # reshape query image
    query = np.reshape(query, (1, -1))

    # search for nearest neighbors
    if return_distance:
        (dist, index) = kdt.query(query, k=k, return_distance=return_distance)
    else:
        index = kdt.query(query, k=k, return_distance=return_distance)

    end = time.time()
    if verbose:
        print('Finish searching in {} seconds'.format(end - start))

    # Return statements
    if return_distance:
        return (dist, index)
    else:
        return index


def image_query_KDTree(image, model, kdt, k=10, return_distance=False, verbose=True):
    '''
        Method to search in a KDTree when a query image is given
        
        ...
        
        Attributes
        ----------
        image : Numpy Array
            An array with a query image
        model : Keras Model
            Model to predict features
        kdt : KDTree
            A KDTree from the scikit-learn library
        k : Integer
            Number of similar images (default 10)
        return_distance : Boolean
            Should a distance be returned (default False)
            
        Output
        ------
        dist : Numpy Array
            An Array with distances
        index : Numpy Array
            An Array with indices of nearest neighbors
        
    '''
    image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

    # Extract features
    pred = model.predict(image, workers=20)
    if return_distance:
        dist, index = query_KDTree(pred, kdt, k, return_distance, verbose)
        return (dist, index)
    else:
        index = query_KDTree(pred, kdt, k, return_distance, verbose)
        return index


def create_min_hash(wmg, feature, index):
    wm = wmg.minhash(feature)
    return (wm)


def lsh(model=None, data=None, pred=None, num_perm=128, n_jobs=20):
    '''
        Method to create LSH
        
        Either a feature vector is given, or a model and data to determine the feature vector
        
        ...
        
        Attributes
        ----------
        model : Keras Model
            A trained model with n feature output (optional)
        data : Numpy Array
            An array with data points witht the same shape as the model input shape (optional)
        pred : Numpy Array
            An array with n discrete features that desribes an image (optional)
        num_perm : Integer
            How many permutations should be created (default 128)
        n_jobs : Integer
            Number of Threads when creating min_hashes in parallel (default 20)
        
        Output
        ------
        wmg : WeightedMinHashGenerator
            Generator for minhashes
        lsh : MinHashLSH
            LSH class to search nearest neighbors
        results : List
            List of min-hashes
        
    '''
    if data is not None:
        if model is not None:
            # Make predictions
            print('Create feature vector')
            pred = model.predict(data)
        else:
            raise ValueError('Model expected')
    elif pred is None:
        raise ValueError('Data or feature vector expected')

    start = time.time()
    print('Start creating min-hashes')

    # Initialize min-hash list
    results = []
    # Create min hash generator
    wmg = WeightedMinHashGenerator(pred.shape[1], sample_size=num_perm)

    # Create min hashes
    results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
        delayed(create_min_hash)(wmg, feature, index) for index, feature in enumerate(pred))

    print('Start building LSH')

    # Build LSH out of min-hashes
    lsh = MinHashLSHForest(num_perm=num_perm)

    for index, wm in enumerate(results):
        lsh.add(index, wm)

    lsh.index()

    end = time.time()
    print('LSH finished in {} seconds'.format(end - start))

    return wmg, lsh, results


def query_lsh(lsh, query=None, wmg=None, min_hash=None, k=15, verbose=True):
    '''
        Method to search LSH
        
        ...
        
        Attributes
        ----------
        lsh : MinHashLSH
            LSH class to search nearest neighbors
        query : Numpy Array
            An array with features of a query image (only necessary if min_hash is None)
        wmg : WeightedMinHashGenerator
            Generator for min-hashes (only necessary if min_hash is None)
        min_hash : WeightedMinHash
            Query minhash signature (Optional)
        k : Integer
            Number of nearest neighbors (default 15)
        verbose : Boolean
            Should time and status information be displayed (default True)
        
            
        Output
        ------
        matches : List
            List of matching indices
        
    '''
    if min_hash is None:
        if wmg is not None:
            if query is not None:
                min_hash = wmg.minhash(query)
            else:
                raise ValueError('Query Feature Vector expected')
        else:
            raise ValueError('WeightedMinHashGenerator or List of MinHashes expected')

    start = time.time()

    matches = lsh.query(min_hash, k)

    end = time.time()
    if verbose:
        print('Search finished in {} seconds'.format(end - start))

    return matches


def image_query_lsh(lsh, image=None, model=None, wmg=None, k=15, verbose=True):
    '''
        Method to search LSH when a query image is given
        
        ...
        
        Attributes
        ----------
        lsh : MinHashLSH
            LSH class to search nearest neighbors
        image : Numpy Array
            An array with a query image
        model : Keras Model
            Model to predict features
        wmg : WeightedMinHashGenerator
            Generator for min-hashes (only necessary if min_hash is None)
        k : Integer
            Number of nearest neighbors (default 15)
        verbose : Boolean
            Should time and status information be displayed (default True)
        
            
        Output
        ------
        matches : List
            List of matching indices
        
    '''

    if wmg is not None:
        if image is not None:
            if model is not None:
                image = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

                # Extract features
                pred = model.predict(image, workers=20)

                matches = query_lsh(lsh, query=pred, wmg=wmg, min_hash=None, k=k, verbose=verbose)
                return matches
            else:
                raise ValueError('Model expected')
        else:
            raise ValueError('Image expected')
    else:
        raise ValueError('WeightedMinHashGenerator expected')
