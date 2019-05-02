import sklearn
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
from time import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from metric_learn.lfda import LFDA
from metric_learn.lmnn import LMNN
from metric_learn.nca import NCA
from metric_learn.mmc import MMC
from metric_learn import MMC_Supervised
from umap import UMAP

dataset = sklearn.datasets.fetch_olivetti_faces(shuffle=True, random_state=272)

image_shape = dataset.images.shape[1:]
faces = dataset.data
number_samples, number_features = faces.shape
faces_centered = faces - faces.mean(axis=0)
faces_centered = faces_centered - faces_centered.mean(axis=1).reshape(number_samples, -1)

X_train, X_val = faces_centered[:320], faces_centered[320:]
y_train, y_val = dataset.target[:320], dataset.target[320:]

def run_neighbor_classifier(ncomponents, train_data, test_data, train_labels, test_labels, run = None):
    if run == "pca":
        pca_model = PCA(n_components=ncomponents)
        train_x = pca_model.fit_transform(train_data)
        test_x = pca_model.transform(test_data)
    elif run == "lda":
        lda_model = LinearDiscriminantAnalysis(n_components = ncomponents)
        train_x = lda_model.fit_transform(train_data, train_labels)
        test_x = lda_model.transform(test_data) 
    elif run== "lfda":
        lfda_model = LFDA(num_dims = ncomponents, embedding_type='orthonormalized')
        train_x = lfda_model.fit_transform(train_data, train_labels)
        test_x = lfda_model.transform(test_data)  
    elif run == "kpca":
        pca_model = KernelPCA(n_components=ncomponents)
        train_x = pca_model.fit_transform(train_data)
        test_x = pca_model.transform(test_data)
    elif run== "lmnn":
        lmnn_model = LMNN()
        train_x = lmnn_model.fit_transform(train_data, train_labels)
        test_x = lmnn_model.transform(test_data)
    elif run == "nca":
        nca_model = NCA(num_dims = ncomponents)
        train_x = nca_model.fit_transform(train_data, train_labels)
        test_x = nca_model.transform(test_data)
    elif run == "umap":
        umap_model = UMAP(n_components=ncomponents)
        train_x = umap_model.fit_transform(train_data)
        test_x = umap_model.transform(test_data)
    elif run == "mmc":
        mmc_model =  MMC_Supervised()
        train_x = mmc_model.fit_transform(train_data, train_labels)
        test_x = mmc_model.transform(test_data)
    else:
        train_x = train_data
        test_x = test_data
    
    classifier = KNeighborsClassifier()
    
    param_grid = {"n_neighbors" : [1, 3, 5, 7, 11]}
    #neighbor_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5)
    neighbor_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5, scoring=make_scorer(matthews_corrcoef))
    neighbor_grid.fit(train_x, train_labels)
    
    model = neighbor_grid.best_estimator_
    parameters = neighbor_grid.best_params_
    train_accuracy = accuracy_score(train_labels, model.predict(train_x))
    
    test_accuracy = accuracy_score(test_labels, model.predict(test_x))
    return {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy}, parameters

lda_dim = []
for j in [1,5,10,15,20,30,40,50,75,100,500,2000]:
    results_lda, lda_parameters = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "lda")
    lda_dim.append(('dim = '+str(j),results_lda, lda_parameters))

lda_dim

pca_dim = []
for j in [1,10,20,40,50,75,100,1000,2000]:
    results_pca, pca_parameters = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "pca")
    pca_dim.append(('dim = '+str(j),results_pca, pca_parameters))
    
pca_dim

lfda_dim = []
for j in [40]:
    results_lfda, lfda_parameters = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "lfda")
    lfda_dim.append(('dim = '+str(j),results_lfda, lfda_parameters))
    
lfda_dim

kpca_dim = []
for j in [1,10,20,40,50,75,100,1000,2000]:
    results_kpca, kpca_parameters = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "kpca")
    kpca_dim.append(('dim = '+str(j),results_kpca, kpca_parameters))
    
kpca_dim

#lmnn_dim = []
#for j in [2]:
#    results_lmnn, lmnn_parameters = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "lmnn")
#    lmnn_dim.append(('dim = '+str(j),results_lmnn, lmnn_parameters))

#lmnn_dim 

nca_dim = []
for j in [1,10,20,40,50,75,100,1000,2000]:
    results_nca, nca_parameters = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "nca")
    nca_dim.append(('dim = '+str(j),results_nca, nca_parameters))
    
nca_dim

umap_dim = []
for j in [5,10,20,40,50,75,100,200]:
    results_umap, umap_parameters = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "umap")
    umap_dim.append(('dim = '+str(j),results_umap, umap_parameters))
    
umap_dim

#mmc_dim = []
#for j in [1]:
#    results_mmc, mmc_parameters = run_neighbor_classifier(j, X_train, X_val, y_train, y_val, run = "mmc")
#    mmc_dim.append(('dim = '+str(j),results_mmc, mmc_parameters))
