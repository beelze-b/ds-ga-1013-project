from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import scipy


import pickle


def closest_k_neighbor(x, X, K = 7 ):
    assert K != 1
    assert x.shape[0] == X.shape[1]
    assert K <= X.shape[1]
    distances = np.apply_along_axis(lambda y: np.sum( (x-y)**2), 1, X)
    distance_arg_sorted = np.argsort(-distances)
    index_of_k_closest = distance_arg_sorted[K-1]
    return X[index_of_k_closest, :]


def run_self(X, y, n_components, beta = 0.5):    
    mask = np.zeros(X.shape[0])
    take_these_label_indices = np.random.choice(X.shape[0], size= 10, replace = False)
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
            a_matrix[ix, jx] = np.exp(-np.sum((x_i - x_j)**2) / sigmas[ix] / sigmas[jx])

            if (y_labeled[ix] == y_labeled[jx]):
                n_y = np.sum(y_labeled == y_labeled[ix])
                w_lb[ix, jx] = a_matrix[ix, jx] * (1.0 / n_labeled - 1.0 / n_y)
                w_lw[ix, jx] = a_matrix[ix, jx] * (1.0 / n_labeled)
            else:
                w_lb[ix, jx] = (1.0 / n_labeled)

    Xprime = np.transpose(X_labeled)
    mu = X_all.mean(axis = 0)
    S_lb = Xprime @ (np.diag(np.matmul(w_lb, 1.0 * np.ones(n_labeled) / n_labeled)) - w_lb) @ X_labeled      
    S_lw = Xprime @ (np.diag(np.matmul(w_lw, 1.0 * np.ones(n_labeled) / n_labeled)) - w_lw) @ X_labeled  

    S_t = np.transpose(X_all).dot(X_all) - n_all * mu.reshape(-1, 1) @ mu.reshape(1, -1)

    S_rlb = (1 - beta) * S_lb + beta * S_t
    S_rlw = (1 - beta) * S_lw + beta * np.eye(d)

    eig, vectors = scipy.linalg.eig(a = S_rlb, b = S_rlw)
    eig = np.real(eig)
    vectors = np.real(vectors)
    eig = np.sqrt(eig[0:n_components])
    vectors = vectors[:, :n_components]
    T = np.zeros((d, n_components))
    for i in range(n_components):
        T[:, i] = eig[i] * vectors[:, i]
    
    return T



