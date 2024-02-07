from DeepUCSL.clustering_utils.sinkhorn_knopp import cpu_sk
import numpy as np

def predict_proba_from_barycenters(X, barycenters) :
    soft_sim = 1 / np.sum((X[:,:,None] - barycenters.T[None, :, :]) ** 2, axis=1)
    soft_sim = soft_sim / np.sum(soft_sim, axis=1, keepdims=True)
    return soft_sim

def reassign_barycenters(probability_matrix):
    lambda_ = 1.0
    regularized_probability_matrix = np.copy(probability_matrix)
    c = 0
    while len(np.unique(np.argmax(regularized_probability_matrix, axis=1))) < len(probability_matrix):
        regularized_probability_matrix = cpu_sk(probability_matrix, lambda_=lambda_)
        lambda_ = lambda_ * 1.1
        c = c+1
        if c > 25 :
            return np.argmax(cpu_sk(probability_matrix, lambda_=1.0), axis=1)
    return np.argmax(regularized_probability_matrix, axis=1)
