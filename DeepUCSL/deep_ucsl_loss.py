import numpy as np
import torch
import torch.nn as nn

INF = 1e8

def conditional_cross_entropy_loss(conditional_probability, labels, pseudo_labels, class_weights=None):
    if class_weights is None:
        class_weights = torch.ones(2, device="cuda")
    else:
        class_weights = torch.tensor(class_weights, device="cuda")

    loss = class_weights[1]*labels.unsqueeze(1) * torch.log(conditional_probability) + \
           class_weights[0]*(1 - labels.unsqueeze(1)) * torch.log(1 - conditional_probability)
    return - (pseudo_labels * loss).mean()

def clustering_regularization_loss(clustering_probability, labels, pseudo_labels, class_weights=None):
    if class_weights is None:
        class_weights = torch.ones(2, device="cuda")
    else:
        class_weights = torch.tensor(class_weights, device="cuda")

    pos_loss = torch.tensor(0.0).cuda()
    neg_loss = torch.tensor(0.0).cuda()
    if torch.sum(labels == 1) > 0:
        pos_loss = - (pseudo_labels[labels == 1] * torch.log(clustering_probability[labels == 1])).mean()
    if torch.sum(labels == 0) > 0:
        neg_loss = - (pseudo_labels[labels == 0] * torch.log(clustering_probability[labels == 0])).mean()

    return class_weights[1]*pos_loss, class_weights[0]*neg_loss


def _get_correlated_mask(batch_size):
    diag = np.eye(2 * batch_size)
    l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
    l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
    mask = torch.from_numpy((diag + l1 + l2))
    mask = (1 - mask).type(torch.bool)
    return mask.to("cuda")

cos = nn.CosineSimilarity(dim=1, eps=1e-8)
cos_2 = nn.CosineSimilarity(dim=2, eps=1e-8)
