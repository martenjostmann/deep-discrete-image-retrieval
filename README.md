Reverse image search using deep discrete feature extraction and locality-sensitive hashing 
=

Requirements
---
To run the scripts, a set of additional Python libraries is needed:
- datasketch    (Min-Hashing and LSH)
- matplotlib    (visualization)
- split-folders     (Is required to split records)
- scikit-learn   (Machine Learning)
- tensorflow    (Deep Learning)

The libraries are stored in the `requirements.txt` file and can be installed with the following command:  
`$ pip install -r requirements.txt`

Alternatively, a Docker image can also be created ([More Infos](#Docker)). 

Quick-start
---

```python
from src.d00_utils.query_kdtree import query_kdtree as qkdt
from src.d00_utils.query_lsh import query_lsh as qlsh

'''
K-D Tree
'''

# Create class instance 
kdt = qkdt(dataset="cifar") #resisc or cifar

# Search for 10 nearest neighbors with image 5
matches = kdt.query(image_idx=5, k=10, plot=True)

# Measure performance
class_accr, accuracy, time = kdt.performance(k=10)

'''
Locality-Sensitive Hashing
'''
# Create class instance 
lsh = qlsh(dataset="resisc") #resisc or cifar

# Search for 10 nearest neighbors with image 5
matches = lsh.query(image_idx=5, k=10, plot=True)

# Measure performance
accuracy, time = lsh.performance(k=10)
```

Ordnerstruktur
---
```
BA
│   README.md
│   requirements.txt
│   Dockerfile
│
└───data
│    │
│    └───resisc45
│
└───models
│    │
│    └───experiment
│    │   │
│    │   └───model_exp_normal_32
│    │   │
│    │   └───model_exp_discrete_48
│    │   │
│    │   └───model_exp_sat_normal_32
│    │   │
│    │   └───model_exp_sat_discrete_512
│    │
│    └───features
│    │   │   discrete_features.npy
│    │   │   normal_features.npy
│    │   │   sat_discrete_features.npy
│    │   │   sat_normal_features.npy
│
└───notebooks
│   │   search.ipynb
│
└───src
    │
    └───d00_utils
    │   │   query_kdtree.py
    │   │   query_lsh.py
    │
    └───d01_data
    │   │   load_data.py
    │
    └───d02_processing
    │   │   preprocess_data.py
    │
    └───d03_modelling
    │   │   custom_model.py
    │   │   nearest_neighbor.py
    │   │   train_model.py
    │
    └───d04_visualisation
    │   │   plot_nearest_neighbor.py
    │
    └───d05_evalutation
    │   │   nearest_neighbor_performance.py

```
---
## Datasets

All data can be stored in the `data` folder. The *CIFAR-10* data are downloaded automatically when the function is called. Only the NWPU-RESISC45 data must be loaded into this folder. A description of the installation can be found [here](https://www.tensorflow.org/datasets/catalog/resisc45). The folder must have the name `resisc45`.

---
## Models

The models are located in the 'models' folder. A total of four models are stored here, one for each data set and search method. Furthermore, there is a `features` folder in the `models` folder. Four Numpy arrays with the features of the best models are already stored here. They can be loaded directly if required.

1. model_exp_normal_32 - normal_features.npy (*CIFAR-10* and *k-d Tree*)
2. model_exp_discrete_48 - discrete_features.npy (*CIFAR-10* and *LSH*)
3. model_exp_sat_normal_32 - sat_normal_features.npy (*RESISC45* and *k-d Tree*)
4. model_exp_sat_discrete_512 - sat_discrete_features.npy (*RESISC45* and *LSH*)
---
## Notebooks

In the `notebooks` folder all Jupyter notebooks are stored. In this folder there is only the `search.ipynb`. Through this notebook both search methods can be executed with both data sets.


---
## Code

All the code is located in the `src` folder, which is divided into topic-specific folders:

### d00_utils

All Python scripts related to the whole project are stored in this folder. The classes `query_kdtree.py` and `query_lsh.py` summarize the most important functions. Thus, when the class is called, the specified data is automatically loaded and preprocessed. Furthermore, the features are extracted directly and the k-d tree is created. With the method `query` the most similar images can be displayed by specifying an image and with the method `performance` the accuracy and the time needed is measured.

### d01_data

In this folder are all Python scripts, which are responsible for loading or saving data. The script `load_data.py` contains functions to load the *CIFAR-10* and the *RESISC45* data sets. More detailed information about the methods can be found in the script.

### d02_processing

This folder contains all Python scripts that are responsible for processing data. The script `preprocess_data` preprocesses the respective data. More detailed information about the methods and their use can be found in the script.

### d03_modelling

In this folder are all scripts to create and train models. Among other things, the script `nearest_neighbor.py` is located here to search for similar images.

### d04_visualisation

All scripts responsible for visualizations are placed here. In this folder there is only the `plot_nearest_neighbor.py` script to output the k nearest neighbors.

### d05_evalutation

In this folder there are scripts to evaluate the performance of the applied search methods. The methods for this are in the `nearest_neighbor_performance.py` script.

# Docker

Through the `Dockerfile` a Jupyter notebook can be created with all dependencies inside a Docker container.

## Docker-Image erstellen
```console
$ docker build -t bachelorarbeit .
```

## Docker-Image starten
```console
$ docker run -p 8888:8888 bachelorarbeit
```