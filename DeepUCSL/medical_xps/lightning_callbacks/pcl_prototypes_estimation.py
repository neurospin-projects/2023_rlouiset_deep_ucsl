from math import log

from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters
from sklearn.cluster import KMeans
import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn
from DeepUCSL.clustering_utils.spherical_kmeans import SphericalKMeans

class PCLPrototypesEstimationAndEvaluation(pl.callbacks.Callback):
    def __init__(self):
        self.name = "PCL prototypes estimation and evaluation callback"

    @torch.no_grad()
    def on_fit_start(self, trainer, pl_module):
        self.estimate_and_evaluate_prototypes(pl_module)

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        self.estimate_and_evaluate_prototypes(pl_module)

    def estimate_and_evaluate_prototypes(self, pl_module):
        train_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, train=True).train
        val_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, validation=True).validation

        # train representations
        head_vectors = [pl_module.forward(batch.cuda()).detach() for _, batch, _, _ in train_loader]
        head_vectors = torch.cat(head_vectors, dim=0)

        # validation representations
        head_vectors_val = [pl_module.forward(batch.cuda()).detach() for _, batch, _, _ in val_loader]
        head_vectors_val = torch.cat(head_vectors_val, dim=0)

        y_train = np.array([labels["subtype"].detach().numpy() for _, _, labels, _ in train_loader])
        y_train = np.array([item for sublist in y_train for item in sublist])

        y_val = np.array([labels["subtype"].detach().numpy() for _, _, labels, _ in val_loader])
        y_val = np.array([item for sublist in y_val for item in sublist])

        pl_module.km = SphericalKMeans(n_clusters=pl_module.n_clusters).fit(head_vectors)
        y_train_kmeans = pl_module.km.predict(head_vectors).cpu().numpy()
        Q_train = np.identity(pl_module.n_clusters)[y_train_kmeans]
        prototypes = pl_module.km.centroids
        pl_module.prototypes = nn.functional.normalize(prototypes.float(), dim=1, p=2)

        L2_norms = np.linalg.norm(pl_module.prototypes.cpu().numpy().T[None, :, :] - head_vectors[:, :, None].cpu().numpy(), axis=1)
        Z = [np.sum(y_train_kmeans == cluster) for cluster in range(0, pl_module.n_clusters)]
        pl_module.temperatures = torch.tensor(
            np.array([np.sum(L2_norms[y_train_kmeans == cluster]) / (Z[i] * log(Z[i] + pl_module.alpha))
                      for i, cluster in enumerate(range(0, pl_module.n_clusters))])).float().cuda()
        pl_module.temperatures = pl_module.temperatures - pl_module.temperatures.mean() + pl_module.temperature
        # pl_module.temperatures = torch.tensor([pl_module.temperature, pl_module.temperature]).cuda()

        # compute Q for validation data
        y_val_kmeans = pl_module.km.predict(head_vectors_val).cpu().numpy()
        Q_validation = np.identity(pl_module.n_clusters)[y_val_kmeans]

        balanced_accuracy_Q_train, _ = balanced_accuracy_for_clusters(y_train, y_train_kmeans)
        print("TRAIN balanced accuracy Q: ", balanced_accuracy_Q_train)
        pl_module.logger.experiment.add_scalar("Q Balanced Accuracy/Train",
                                               balanced_accuracy_Q_train,
                                               pl_module.current_epoch)

        balanced_accuracy_Q_val, _ = balanced_accuracy_for_clusters(y_val, y_val_kmeans)
        print("VAL balanced accuracy Q: ", balanced_accuracy_Q_val)
        pl_module.logger.experiment.add_scalar("Q Balanced Accuracy/Val",
                                               balanced_accuracy_Q_val,
                                               pl_module.current_epoch)


        pl_module.Q["train"] = Q_train
        pl_module.Q["validation"] = Q_validation

        pl_module.data_manager.set_pseudo_labels(pl_module.fold_index, Q_train, phase="train")
        pl_module.data_manager.set_pseudo_labels(pl_module.fold_index, Q_validation, phase="validation")