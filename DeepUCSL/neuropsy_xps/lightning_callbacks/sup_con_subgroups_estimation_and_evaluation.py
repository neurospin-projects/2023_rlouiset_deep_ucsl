from sklearn.metrics import adjusted_mutual_info_score as AMI, balanced_accuracy_score
from sklearn.metrics import adjusted_rand_score as ARI
import pytorch_lightning as pl
import numpy as np
import torch
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes

class SupConClusteringAndLinearProbing(pl.callbacks.Callback):
    def __init__(self):
        self.description = "Perform K-Means every epoch at the end of the validation"

    @torch.no_grad()
    def on_fit_start(self, trainer, pl_module):
        self.linear_probing_and_subgroups_evaluation(pl_module)

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        self.linear_probing_and_subgroups_evaluation(pl_module)

    @torch.no_grad()
    def linear_probing_and_subgroups_evaluation(self, pl_module):
        train_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, train=True).train
        val_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, validation=True).validation

        with torch.no_grad():
            # training representations
            latent_representations = []
            latent_heads = []
            for _, batch, _, _ in train_loader:
                heads, representations = pl_module.forward(batch.cuda())
                latent_representations.append(representations)
                latent_heads.append(heads)
            latent_representations = torch.cat(latent_representations, dim=0)
            latent_heads = torch.cat(latent_heads, dim=0)
            y_train = torch.cat([labels["diagnosis"].cpu() for _, _, labels, _ in train_loader], dim=0)
            y_subtype_train = torch.cat([labels["subtype"].cpu() for _, _, labels, _ in train_loader], dim=0)

            # validation representations
            latent_representations_val = []
            latent_heads_val = []
            for _, batch, _, _ in val_loader:
                heads, representations = pl_module.forward(batch.cuda())
                latent_representations_val.append(representations)
                latent_heads_val.append(heads)
            latent_representations_val = torch.cat(latent_representations_val, dim=0)
            latent_heads_val = torch.cat(latent_heads_val, dim=0)
            y_val = torch.cat([labels["diagnosis"].cpu() for _, _, labels, _ in val_loader], dim=0)
            y_subtype_val = torch.cat([labels["subtype"].cpu() for _, _, labels, _ in val_loader], dim=0)

        # get subtypes
        representations_subtypes, y_subtype, mask_subtype = pl_module.get_subtypes(latent_representations, y_subtype_train, y_train)
        representations_subtypes_val, y_subtype_val, _ = pl_module.get_subtypes(latent_representations_val, y_subtype_val, y_val)

        # scale the representations and head
        representations_subtypes = pl_module.scaler.fit_transform(representations_subtypes).cpu().numpy()
        latent_heads = latent_heads.cpu().numpy()

        representations_subtypes_val = pl_module.scaler.transform(representations_subtypes_val).cpu().numpy()
        latent_heads_val = latent_heads_val.cpu().numpy()

        # compute the clustering
        y_train_subtype_pred = pl_module.kmeans_representation.fit_predict(representations_subtypes)
        y_val_subtype_pred = pl_module.kmeans_representation.predict(representations_subtypes_val)

        # predict the classification predictions
        pl_module.log_reg = pl_module.log_reg.fit(latent_heads, y_train)
        y_train_pred_head = pl_module.log_reg.predict(latent_heads)
        y_val_pred_head = pl_module.log_reg.predict(latent_heads_val)

        # calculating correct and total predictions
        balanced_accuracy_head = balanced_accuracy_score(y_train, y_train_pred_head)

        # calculating correct and total clustering predictions
        balanced_accuracy_cluster, permutation_indices = balanced_accuracy_for_clusters(y_subtype,  y_train_subtype_pred)
        pl_module.permutation_indices = permutation_indices

        overall_accuracy_clusters = overall_accuracy_for_clusters_and_classes(y_train, y_train_pred_head, y_subtype,
                                                                              y_train_subtype_pred,
                                                                              permutation_indices)

        print("TRAIN / B-ACC cluster: ", balanced_accuracy_cluster)
        pl_module.logger.experiment.add_scalar("Balanced Accuracy Cluster/Train",
                                               balanced_accuracy_cluster,
                                               pl_module.current_epoch)
        print("TRAIN / B-ACC : ", balanced_accuracy_head)
        pl_module.logger.experiment.add_scalar("Balanced Accuracy/Train",
                                               balanced_accuracy_head,
                                               pl_module.current_epoch)
        print("TRAIN / Overall ACC : ", balanced_accuracy_head)
        pl_module.logger.experiment.add_scalar("Overall Balanced Accuracy/Train",
                                               overall_accuracy_clusters,
                                               pl_module.current_epoch)

        balanced_accuracy_head = balanced_accuracy_score(y_val, y_val_pred_head)
        balanced_accuracy_cluster, _ = balanced_accuracy_for_clusters(y_subtype_val, y_val_subtype_pred)
        overall_accuracy_clusters = overall_accuracy_for_clusters_and_classes(y_val, y_val_pred_head, y_subtype_val, y_val_subtype_pred, permutation_indices)

        print("VAL / B-ACC cluster: ", balanced_accuracy_cluster)
        print("VAL / B-ACC: ", balanced_accuracy_head)
        print("VAL / overall B-ACC clusters head", overall_accuracy_clusters)
        pl_module.logger.experiment.add_scalar("Balanced Accuracy Cluster/Val",
                                               balanced_accuracy_cluster,
                                               pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("Balanced Accuracy/Val",
                                               balanced_accuracy_head,
                                               pl_module.current_epoch)
        pl_module.logger.experiment.add_scalar("Overall Balanced Accuracy/Val",
                                               overall_accuracy_clusters,
                                               pl_module.current_epoch)
