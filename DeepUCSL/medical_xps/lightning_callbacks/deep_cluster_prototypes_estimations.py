import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes
from DeepUCSL.clustering_utils.spherical_kmeans import SphericalKMeans


class DeepClusterPrototypesManager(pl.callbacks.Callback):
    @torch.no_grad()
    def on_fit_start(self, trainer, pl_module):
        self.estimate_and_evaluate_pseudo_labels(pl_module)

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        self.estimate_and_evaluate_pseudo_labels(pl_module)

    def estimate_and_evaluate_pseudo_labels(self, pl_module):
        pl_module.eval()

        train_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, train=True).train
        val_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, validation=True).validation

        # train representations
        head_vectors = [pl_module.forward(batch.cuda())[0].detach() for _, batch, _, _ in train_loader]
        head_vectors = torch.cat(head_vectors, dim=0)

        # validation representations
        head_vectors_val = [pl_module.forward(batch.cuda())[0].detach() for _, batch, _, _ in val_loader]
        head_vectors_val = torch.cat(head_vectors_val, dim=0)

        pl_module.km = SphericalKMeans(n_clusters=pl_module.n_clusters).fit(head_vectors)
        y_train_kmeans = pl_module.km.predict(head_vectors).cpu().numpy()
        y_val_kmeans = pl_module.km.predict(head_vectors_val).cpu().numpy()
        Q_train = np.identity(pl_module.n_clusters)[y_train_kmeans]
        Q_validation = np.identity(pl_module.n_clusters)[y_val_kmeans]
        prototypes = pl_module.km.centroids
        pl_module.prototypes = nn.functional.normalize(prototypes.float(), dim=1, p=2)
        weights = torch.tensor(1 / np.sum(Q_train, 1)[:, None]).cuda()
        pl_module.weights = weights / weights.sum(dim=0, keepdim=True)


        pl_module.Q["train"] = Q_train
        pl_module.Q["validation"] = Q_validation

        pl_module.data_manager.set_pseudo_labels(pl_module.fold_index, Q_train, phase="train", n_clusters=pl_module.n_clusters)
        pl_module.data_manager.set_pseudo_labels(pl_module.fold_index, Q_validation, phase="validation", n_clusters=pl_module.n_clusters)

        pl_module.train()

