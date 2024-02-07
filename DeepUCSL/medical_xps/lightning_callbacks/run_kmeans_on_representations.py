from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI, balanced_accuracy_score
from sklearn.metrics import adjusted_rand_score as ARI
import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn

from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes

class KMeansOnRep(pl.callbacks.Callback):

    def on_validation_end(self, trainer, pl_module):
        train_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, train=True).train
        val_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, validation=True).validation

        # train representations
        head_vectors = [pl_module.forward(batch.cuda())[1].cpu().detach().numpy() for _, batch, _, _ in train_loader]
        head_vectors = np.array([item for sublist in head_vectors for item in sublist])

        # validation representations
        head_vectors_val = [pl_module.forward(batch.cuda())[1].cpu().detach().numpy() for _, batch, _, _ in val_loader]
        head_vectors_val = np.array([item for sublist in head_vectors_val for item in sublist])

        y_train = np.array([labels["subtype"].detach().numpy() for _, _, labels, _ in train_loader])
        y_train = np.array([item for sublist in y_train for item in sublist])

        y_val = np.array([labels["subtype"].detach().numpy() for _, _, labels, _ in val_loader])
        y_val = np.array([item for sublist in y_val for item in sublist])

        head_vectors = pl_module.scaler.fit_transform(head_vectors)
        head_vectors_val = pl_module.scaler.transform(head_vectors_val)

        pl_module.km = KMeans(n_clusters=pl_module.n_clusters).fit(head_vectors)
        y_train_kmeans = pl_module.km.predict(head_vectors)
        y_val_kmeans = pl_module.km.predict(head_vectors_val)

        balanced_accuracy_clusters, permutation_indices = balanced_accuracy_for_clusters(y_train, y_train_kmeans)
        print("TRAIN / balanced accuracy clusters: ", balanced_accuracy_clusters)
        pl_module.logger.experiment.add_scalar("Clustering Balanced Accuracy/Train",
                                          balanced_accuracy_clusters,
                                          pl_module.current_epoch)

        balanced_accuracy_clusters, _ = balanced_accuracy_for_clusters(y_val, y_val_kmeans, permutation_indices)
        print("VAL / balanced accuracy clusters: ", balanced_accuracy_clusters)
        pl_module.logger.experiment.add_scalar("Clustering Balanced Accuracy/Val",
                                          balanced_accuracy_clusters,
                                          pl_module.current_epoch)