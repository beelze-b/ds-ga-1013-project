from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import sklearn
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import hinge_loss
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import average_precision_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from metric_learn.lfda import LFDA
from metric_learn.lmnn import LMNN
from metric_learn.nca import NCA
from metric_learn.mmc import MMC
from metric_learn import MMC_Supervised

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import scipy

from sklearn.model_selection import train_test_split


import pickle


def closest_k_neighbor(x, X, K = 7 ):
    assert K != 1
    assert x.shape[0] == X.shape[1]
    assert K <= X.shape[0]
    distances = np.apply_along_axis(lambda y: np.sum( (x-y)**2), 1, X)
    distance_arg_sorted = np.argsort(-distances)
    index_of_k_closest = distance_arg_sorted[K-1]
    return X[index_of_k_closest, :]

def run_self(X, y, n_components, beta = 0.5):    
    assert X.shape[0] >= 50
    mask = np.zeros(X.shape[0])
    take_these_label_indices = np.random.choice(X.shape[0], size= 50, replace = False)
    mask[take_these_label_indices] = 1
    X_labeled = X[mask.astype(bool), :]
    y_labeled = y[mask.astype(bool)]
    X_all = np.vstack([X_labeled, X[~mask.astype(bool), :]])
    d = X.shape[1]
    neighbors = np.apply_along_axis(closest_k_neighbor, 1, X_labeled, X = X_all)
    sigmas = np.apply_along_axis(lambda x: np.sqrt(np.sum(x**2)), 1, X_labeled - neighbors)
    
    
    n_labeled = X_labeled.shape[0]
    n_all = X.shape[0]

    
    a_matrix = np.zeros((n_labeled, n_labeled))
    w_lb = np.zeros((n_labeled, n_labeled))
    w_lw = np.zeros((n_labeled, n_labeled))

    for ix in np.arange(n_labeled):
        for jx in np.arange(n_labeled):
            x_i = X_labeled[ix, :]
            x_j = X_labeled[jx, :]

            if (y_labeled[ix] == y_labeled[jx]):
                a_matrix[ix, jx] = np.exp(-np.sum((x_i - x_j)**2) / sigmas[ix] / sigmas[jx])
                n_y = np.sum(y_labeled == y_labeled[ix])
                w_lb[ix, jx] = a_matrix[ix, jx] * (1.0 / n_labeled - 1.0 / n_y)
                w_lw[ix, jx] = a_matrix[ix, jx] * (1.0 / n_y)
            else:
                w_lb[ix, jx] = (1.0 / n_labeled)

    Xprime = np.transpose(X_labeled)
    mu = X_all.mean(axis = 0)
    S_lb = (Xprime.dot(np.diag(w_lb.sum(axis = 0)) - w_lb)).dot(X_labeled)      
    S_lw = (Xprime.dot(np.diag(w_lw.sum(axis = 0)) - w_lw)).dot(X_labeled)  


    S_t = np.transpose(X_all).dot(X_all) - n_all * mu.reshape(-1, 1).dot(mu.reshape(1, -1))

    S_rlb = (1 - beta) * S_lb + beta * S_t
    S_rlw = (1 - beta) * S_lw + beta * np.eye(d)


    eig, vectors = scipy.linalg.eig(a = S_rlb, b = S_rlw)
    eig = np.real(eig)
    vectors = np.real(vectors)
    eig = np.sqrt(eig[:n_components])
    vectors = vectors[:, :n_components]
    T = np.zeros((d, n_components))
    for i in range(n_components):
        T[:, i] = eig[i] * vectors[:, i]
    
    return T


def run_neighbor_classifier(ncomponents, train_data, test_data, train_labels, test_labels, run = None, seed = int(3429)):
    np.random.seed(seed)
    start_time = time.time()
    
    ind, _, _, _ = train_test_split(np.arange(train_data.shape[0]), 
                                    train_labels, 
                                    test_size=train_data.shape[0]-100,
                                    random_state=seed)
    
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
        train_x = lfda_model.fit_transform(train_data[ind, :], train_labels[ind])
        test_x = lfda_model.transform(test_data)  
    elif run == "kpca":
        pca_model = KernelPCA(n_components=ncomponents)
        train_x = pca_model.fit_transform(train_data)
        test_x = pca_model.transform(test_data)
    elif run== "lmnn":
        pca_model = PCA(n_components=ncomponents)
        reduced_train = pca_model.fit_transform(train_data)
        reduced_test = pca_model.transform(test_data)
        lmnn_model = LMNN(k=1,use_pca=False).fit(train_data[ind, :], train_labels[ind])
        train_x = lmnn_model.transform(reduced_train)
        test_x = lmnn_model.transform(reduced_test)
    elif run == "nca":
        nca_model = NCA(num_dims = ncomponents)
        train_x = nca_model.fit_transform(train_data[ind, :], train_labels[ind])
        test_x = nca_model.transform(test_data)
    elif run == "umap":
        umap_model = UMAP(n_components=ncomponents)
        train_x = umap_model.fit_transform(train_data)
        test_x = umap_model.transform(test_data)
    elif run == "mmc":
        pca_model = PCA(n_components=ncomponents)
        reduced_train = pca_model.fit_transform(train_data)
        reduced_test = pca_model.transform(test_data)
        mmc_model =  MMC_Supervised(num_constraints=200).fit(reduced_train, train_labels)
        train_x = mmc_model.transform(reduced_train)
        test_x = mmc_model.transform(reduced_test)
    elif run == "self":
        if (train_data.shape[1] >= 800): 
            pca_model = PCA(n_components=800)
            train_data = pca_model.fit_transform(train_data)
            test_data = pca_model.transform(test_data)
        results = run_self(train_data[ind, :], train_labels[ind], ncomponents)
        train_x = train_data.dot(results)
        test_x = test_data.dot(results)
    else:
        train_x = train_data
        test_x = test_data

    end_time = time.time()

    time_diff = end_time - start_time
    
    
    classifier = KNeighborsClassifier()
    
    param_grid = {"n_neighbors" : [1, 3, 5]}
    neighbor_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv = 5, scoring=make_scorer(matthews_corrcoef))
    neighbor_grid.fit(train_x, train_labels)
    
    model = neighbor_grid.best_estimator_
    parameters = neighbor_grid.best_params_
    train_accuracy = accuracy_score(train_labels, model.predict(train_x))
    
    test_accuracy = accuracy_score(test_labels, model.predict(test_x))
    return {'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy, 'run_time': time_diff, 'n_comp': ncomponents, 'n_neighbors': parameters['n_neighbors']}


