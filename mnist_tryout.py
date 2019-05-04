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



# In[ ]:


from util import *
import pickle


# In[ ]:


run_algorithms_dataset('mnist', train_data, test_data, train_labels, test_labels)


# In[ ]:




