from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as AMI, balanced_accuracy_score
from sklearn.metrics import adjusted_rand_score as ARI
import pytorch_lightning as pl
import numpy as np
import torch
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes
from DeepUCSL.neuropsy_xps.utils import create_diagnosis_and_subtype_dict


class BCEClustering(pl.callbacks.Callback):
    def __init__(self):
        self.description = "Perform K-Means every epoch at the end of the validation"

    @torch.no_grad()
    def on_test_start(self, trainer, pl_module):
        train_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, train=True).train
        val_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, test=True).test # , validation=True).validation

        with torch.no_grad():
            # training representations
            representations = []
            preds = []
            for batch, _, labels, _ in train_loader:
                pred, rep = pl_module.forward(batch.cuda())
                representations.append(rep)
                preds.append(pred)
            representations = torch.cat(representations, dim=0)
            preds = np.argmax(torch.cat(preds, dim=0).round().cpu().numpy(), 1)
            y_train = torch.cat([create_diagnosis_and_subtype_dict(labels)["diagnosis"].cpu() for _, _, labels, _ in train_loader], dim=0)
            y_subtype_train = torch.cat([create_diagnosis_and_subtype_dict(labels)["subtype"].cpu() for _, _, labels, _ in train_loader], dim=0)

            # validation representations
            representations_val = []
            preds_val = []
            for batch, _, labels, _ in val_loader:
                pred, rep = pl_module.forward(batch.cuda())
                representations_val.append(rep)
                preds_val.append(pred)
            representations_val = torch.cat(representations_val, dim=0)
            preds_val = np.argmax(torch.cat(preds_val, dim=0).round().cpu().numpy(), 1)
            y_val = torch.cat([labels["diagnosis"].cpu() for _, _, labels, _ in val_loader], dim=0)
            y_subtype_val = torch.cat([labels["subtype"].cpu() for _, _, labels, _ in val_loader], dim=0)

        # get subtypes
        representations_subtypes, y_subtype, mask_subtype = pl_module.get_subtypes(representations, y_subtype_train, y_train)
        representations_subtypes_val, y_subtype_val, _ = pl_module.get_subtypes(representations_val, y_subtype_val, y_val)

        # scale the representations
        representations_subtypes = pl_module.scaler.fit_transform(representations_subtypes).cpu().numpy()
        representations_subtypes_val = pl_module.scaler.transform(representations_subtypes_val).cpu().numpy()

        # compute the clustering
        pl_module.kmeans_representation = KMeans(n_clusters=pl_module.n_clusters)
        y_train_subtype_pred = pl_module.kmeans_representation.fit_predict(representations_subtypes)
        y_val_subtype_pred = pl_module.kmeans_representation.predict(representations_subtypes_val)

        # calculating the metrics
        balanced_accuracy_cluster, permutation_indices = balanced_accuracy_for_clusters(y_subtype, y_train_subtype_pred)
        pl_module.permutation_indices = permutation_indices
        print("TRAIN / B-ACC cluster: ", balanced_accuracy_cluster)
        pl_module.logger.experiment.add_scalar("Balanced Accuracy Cluster/Train",
                                               balanced_accuracy_cluster,
                                               pl_module.current_epoch)

        balanced_accuracy_train = balanced_accuracy_score(y_train, preds)
        print("TRAIN / B-ACC : ", balanced_accuracy_train)
        pl_module.logger.experiment.add_scalar("Balanced Accuracy/Train",
                                               balanced_accuracy_train,
                                               pl_module.current_epoch)

        overall_accuracy_clusters = overall_accuracy_for_clusters_and_classes(y_train, preds, y_subtype, y_train_subtype_pred, permutation_indices)
        print("TRAIN / overall B-ACC clusters", overall_accuracy_clusters)
        pl_module.logger.experiment.add_scalar("Overall Accuracy/Train",
                                               overall_accuracy_clusters,
                                               pl_module.current_epoch)

        balanced_accuracy_cluster, _ = balanced_accuracy_for_clusters(y_subtype_val, y_val_subtype_pred, permutation_indices)
        print("VAL / B-ACC cluster: ", balanced_accuracy_cluster)
        pl_module.logger.experiment.add_scalar("Balanced Accuracy Cluster/Val",
                                               balanced_accuracy_cluster,
                                               pl_module.current_epoch)

        balanced_accuracy_val = balanced_accuracy_score(y_val, preds_val)
        print("VAL / B-ACC : ", balanced_accuracy_val)
        pl_module.logger.experiment.add_scalar("Balanced Accuracy/Val",
                                               balanced_accuracy_val,
                                               pl_module.current_epoch)

        overall_accuracy_clusters = overall_accuracy_for_clusters_and_classes(y_val, preds_val, y_subtype_val, y_val_subtype_pred, permutation_indices)
        print("VAL / overall B-ACC clusters", overall_accuracy_clusters)
        pl_module.logger.experiment.add_scalar("Overall Accuracy/Val",
                                               overall_accuracy_clusters,
                                               pl_module.current_epoch)