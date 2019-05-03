#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np
import csv

from PIL import Image

#### 

##### THIS SCRIPT IS TAKEN FROM HERE
###   https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390
IMAGE_SIZE = 28


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        return labels

train_data_filename = 'data/train-images-idx3-ubyte.gz'
train_labels_filename = 'data/train-labels-idx1-ubyte.gz'
test_data_filename = 'data/t10k-images-idx3-ubyte.gz'
test_labels_filename = 'data/t10k-labels-idx1-ubyte.gz'

# Extract it into np arrays.
tr_data = extract_data(train_data_filename, 60000)
train_labels = extract_labels(train_labels_filename, 60000)
te_data = extract_data(test_data_filename, 10000)
test_labels = extract_labels(test_labels_filename, 10000)


# In[2]:


def flatten_data(data):
    m = np.zeros((data.shape[0], data.shape[1] * data.shape[2]))
    for e in range(data.shape[0]):
        m[e, ] = np.ndarray.flatten(data[e, :, :, 0])
    return m


# In[3]:


train_data = flatten_data(tr_data)
test_data = flatten_data(te_data)

columns_to_use = np.apply_along_axis(lambda x: ~np.all(np.isclose(x, 0)), 0, train_data)

train_data = train_data[:, columns_to_use]
test_data = test_data[:, columns_to_use]


train_data_mean = train_data.mean(axis=0).reshape(1, -1)
train_data = train_data - train_data_mean
test_data = test_data - train_data_mean

# In[4]:


from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[10]:


def run_neighbor_classifier(ncomponents, train_data, test_data, train_labels, test_labels, run = None):
    if run == "pca":
        pca_model = PCA(n_components=ncomponents)
        train_x = pca_model.fit_transform(train_data)
        test_x = pca_model.transform(test_data)
    elif run == "lda":
        lda_model = LinearDiscriminantAnalysis(n_components = ncomponents)
        train_x = lda_model.fit_transform(train_data, train_labels)
        test_x = lda_model.transform(test_data)
    else:
        train_x = train_data
        test_x = test_data
    
    classifier = KNeighborsClassifier()
    
    param_grid = {"n_neighbors" : [1, 3, 5, 7, 11]}
    neighbor_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5)
    neighbor_grid.fit(train_x, train_labels)
    
    model = neighbor_grid.best_estimator_
    parameters = neighbor_grid.best_params_
    train_accuracy = accuracy_score(train_labels, model.predict(train_x))
    
    test_accuracy = accuracy_score(test_labels, model.predict(test_x))
    return {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}, parameters


# In[6]:


results_pca, pca_parameters = run_neighbor_classifier(3, train_data, test_data, train_labels, test_labels, run = "pca")


# In[7]:


import pickle


# In[8]:


with open('data/mnist/results_pca_3_comp.pickle', 'wb') as handle:
    pickle.dump(results_pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('data/mnist/pca_parameters_3_comp.pickle', 'wb') as handle:
    pickle.dump(pca_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)    


# In[11]:


results_lda, lda_parameters = run_neighbor_classifier(10, train_data, test_data, train_labels, test_labels, run = "lda")


# In[12]:


with open('data/mnist/results_lda_3_comp.pickle', 'wb') as handle:
    pickle.dump(results_lda, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('data/mnist/lda_parameters_3_comp.pickle', 'wb') as handle:
    pickle.dump(lda_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)    


# In[14]:


# example pickle load

with open('data/mnist/results_lda_3_comp.pickle', 'rb') as handle:
    results_lda_10_comp_reloaded = pickle.load(handle)


# In[18]:


print("PCA Accuracy")
print(results_pca)


# In[17]:


print("LDA accuracy")
print(results_lda_10_comp_reloaded)


# In[ ]:




