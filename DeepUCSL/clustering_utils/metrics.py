import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics.cluster import contingency_matrix
from DeepUCSL.clustering_utils.sinkhorn_knopp import cpu_sk

def balanced_accuracy_for_clusters(labels, clusters_pred, permutation_indices=None):
    contingency_mat = contingency_matrix(labels_true=labels, labels_pred=clusters_pred).T
    if permutation_indices is None and len(np.unique(clusters_pred)) > 1:
        normalized_contingency_matrix = contingency_mat / np.sum(contingency_mat, axis=1, keepdims=True)
        lambda_ = 1.0
        regularized_contingency_matrix = np.copy(normalized_contingency_matrix)
        while len(np.unique(np.argmax(regularized_contingency_matrix, axis=0))) < contingency_mat.shape[1] and lambda_ < 25:
            regularized_contingency_matrix = cpu_sk(normalized_contingency_matrix, lambda_=lambda_)
            lambda_ = lambda_ * 1.1
        permutation_indices = np.argmax(regularized_contingency_matrix, 0)
    one_hot_clusters_pred = np.identity(contingency_mat.shape[1])[clusters_pred.astype(np.int)]
    permuted_clusters_pred = np.argmax(one_hot_clusters_pred[:, permutation_indices], 1)
    try:
        accuracy_score = balanced_accuracy_score(labels, permuted_clusters_pred)
    except:
        accuracy_score = -1.0
        """print("balanced accuracy for clusters cannot be computed...")
        print("LABELS : ", labels[:10])
        print("PREDICTION : ", clusters_pred.astype(np.int)[:10])"""
    return accuracy_score, permutation_indices


def overall_accuracy_for_clusters_and_classes(labels, labels_pred, clusters_labels, clusters_pred, permutation_indices=None) :
    contingency_mat = contingency_matrix(labels_true=clusters_labels, labels_pred=clusters_pred).T
    if permutation_indices is None:
        normalized_contingency_matrix = contingency_mat / np.sum(contingency_mat, axis=1, keepdims=True)
        lambda_ = 1.0
        regularized_contingency_matrix = np.copy(normalized_contingency_matrix)
        while len(np.unique(np.argmax(regularized_contingency_matrix, axis=0))) < contingency_mat.shape[1] and lambda_ < 25:
            regularized_contingency_matrix = cpu_sk(normalized_contingency_matrix, lambda_=lambda_)
            lambda_ = lambda_ * 1.1
        permutation_indices = np.argmax(regularized_contingency_matrix, 0)
    one_hot_clusters_pred = np.identity(contingency_mat.shape[1])[clusters_pred.astype(np.int)]
    permuted_clusters_pred = np.argmax(one_hot_clusters_pred[:, permutation_indices], 1)

    tn = np.sum(labels_pred[labels == 0] == 0)
    fn = np.sum(labels_pred[labels == 1] == 0)
    fp = np.sum(labels_pred[labels == 0] == 1)
    tp = 0
    y_test_pred_positive = labels_pred[labels == 1]
    for i in range(len(y_test_pred_positive)):
        if y_test_pred_positive[i] == 1 and clusters_labels[i] == permuted_clusters_pred[i]:
            tp += 1
        if y_test_pred_positive[i] == 1 and clusters_labels[i] != permuted_clusters_pred[i]:
            fp += 1
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return (recall + specificity) / 2