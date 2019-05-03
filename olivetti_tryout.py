
# coding: utf-8

# In[1]:


from util import *


# In[2]:


#dataset = sklearn.datasets.fetch_olivetti_faces(shuffle=True, random_state=272)
dataset = sklearn.datasets.fetch_olivetti_faces(shuffle=False)


# In[3]:


image_shape = dataset.images.shape[1:]
faces = dataset.data
number_samples, number_features = faces.shape
faces_centered = faces - faces.mean(axis=0)
faces_centered = faces_centered - faces_centered.mean(axis=1).reshape(number_samples, -1)


# In[4]:


sklearn.model_selection.train_test_split
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(faces_centered, dataset.target, test_size=0.2, stratify=dataset.target, random_state=272)


# In[6]:


results_pca = run_neighbor_classifier(10, X_train, X_val, y_train, y_val, run = "pca")


# In[7]:


results_lda = run_neighbor_classifier(10, X_train, X_val, y_train, y_val, run = "lda")


# In[8]:


print("PCA Accuracy")
print(results_pca)


# In[9]:


print("LDA accuracy")
print(results_lda)


# In[10]:


lda_dim = []
for j in [1,5,10,15,20,30,40,50,75,100,500,2000]:
    results_lda = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "lda")
    lda_dim.append(results_lda)


# In[11]:


lda_dim


# In[12]:


pca_dim = []
for j in [1,3,10,20,40,50,75,100,200,300]:
    results_pca = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "pca")
    pca_dim.append(results_pca)


# In[13]:


pca_dim


# In[14]:


lfda_dim = []
for j in [40]:
    results_lfda = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "lfda")
    lfda_dim.append(results_lfda)


# In[15]:


lfda_dim


# In[16]:


kpca_dim = []
for j in [1,10,20,40,50,75,100,200,300]:
    results_kpca = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "kpca")
    kpca_dim.append(results_kpca)


# In[17]:


kpca_dim


# In[18]:


lmnn_dim = []
for j in [4096]:
    results_lmnn = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "lmnn")
    lmnn_dim.append(results_lmnn)


# In[19]:


lmnn_dim 


# In[21]:


nca_dim = []
for j in [1,10,20,40,50,75,100,200,300]:
    results_nca = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "nca")
    nca_dim.append(results_nca)


# In[22]:


nca_dim

