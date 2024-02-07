from sklearn.metrics import balanced_accuracy_score
import pytorch_lightning as pl
import numpy as np
import torch

from DeepUCSL.clustering_utils.centroids_reidentification import reassign_barycenters
from DeepUCSL.clustering_utils.centroids_reidentification import predict_proba_from_barycenters
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes
from DeepUCSL.clustering_utils.sk_soft_kmeans import SoftSKKMeans
from DeepUCSL.neuropsy_xps.utils import create_diagnosis_and_subtype_dict


class DeepUCSLPseudoLabeller(pl.callbacks.Callback):
    def __init__(self):
        self.description = "Perform Soft SK K-Means every epoch at the end of the validation"

    @torch.no_grad()
    def on_fit_start(self, trainer, pl_module):
        self.estimate_and_evaluate_pseudo_labels(pl_module)

    @torch.no_grad()
    def on_validation_end(self, trainer, pl_module):
        self.estimate_and_evaluate_pseudo_labels(pl_module)

    def estimate_and_evaluate_pseudo_labels(self, pl_module):
        train_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, train=True).train
        val_loader = pl_module.data_manager.get_dataloader(pl_module.fold_index, shuffle=False, validation=True).validation

        with torch.no_grad():
            latent_representations = torch.cat([pl_module.forward(batch.cuda())[2] for _, batch, _, _ in train_loader], dim=0)
            y_train = torch.cat([create_diagnosis_and_subtype_dict(labels)["diagnosis"].cpu() for _, _, labels, _ in train_loader], dim=0)
            y_subtype_train = torch.cat([create_diagnosis_and_subtype_dict(labels)["subtype"].cpu() for _, _, labels, _ in train_loader], dim=0)

            latent_representations_val, y_clusters_pred, y_cond_clsf_pred = [], [], []
            for _, batch, _, _ in val_loader:
                cond_clsf_pred, clustering_pred, rep_vector = pl_module.forward(batch.cuda())
                latent_representations_val.append(rep_vector)
                y_clusters_pred.append(clustering_pred)
                y_cond_clsf_pred.append(cond_clsf_pred)
            latent_representations_val = torch.cat(latent_representations_val, dim=0)
            y_clusters_pred_val = torch.cat(y_clusters_pred, dim=0)
            y_cond_clsf_pred_val = torch.cat(y_cond_clsf_pred, dim=0)
            y_val = torch.cat([create_diagnosis_and_subtype_dict(labels)["diagnosis"].cpu() for _, _, labels, _ in val_loader], dim=0)
            y_subtype_val = torch.cat([create_diagnosis_and_subtype_dict(labels)["subtype"].cpu() for _, _, labels, _ in val_loader], dim=0)

            # get training subtypes
            latent_representations_subtype, y_subtype_train, train_subtype_mask = pl_module.get_subtypes(latent_representations, y_subtype_train, y_train)

            # predict probability to belong to a cluster
            latent_representations_subtype = pl_module.scaler.fit_transform(latent_representations_subtype)
            latent_representations = pl_module.scaler.transform(latent_representations)
            latent_representations_val = pl_module.scaler.transform(latent_representations_val)

            # get validation subtypes
            latent_representations_val_subtype, y_subtype_val, val_subtype_mask = pl_module.get_subtypes(latent_representations_val, y_subtype_val, y_val)

            # train the Soft SK K-Means with the disease training samples
            pl_module.km = SoftSKKMeans(n_clusters=pl_module.n_clusters).fit(latent_representations_subtype)
            barycenters = pl_module.km.centroids.cpu().numpy()

        if pl_module.current_epoch > 0:
            # re identify the centroids / barycenters
            permutation_index = reassign_barycenters(predict_proba_from_barycenters(pl_module.barycenters, barycenters))
            pl_module.barycenters = barycenters[permutation_index]

            # compute Q for training data
            Q_train = predict_proba_from_barycenters(latent_representations.cpu().numpy(), pl_module.barycenters)
            Q_train[y_train == 0] = (1/pl_module.n_clusters)

            # compute Q for validation data
            Q_validation = predict_proba_from_barycenters(latent_representations_val.cpu().numpy(), pl_module.barycenters)
            print("Q VAL SHAPE IS: ", Q_validation.shape)
            Q_validation[y_val == 0] = (1/pl_module.n_clusters)

            # Training metrics
            Q_subtype_train = Q_train[train_subtype_mask]

            for i in range(pl_module.n_clusters):
                print("TRAIN" + str(i) + " : ", np.sum(np.argmax(Q_subtype_train, 1) == i))

            # save metrics for training Q
            balanced_accuracy_Q, permutation_indices = balanced_accuracy_for_clusters(y_subtype_train, np.argmax(Q_subtype_train, 1))
            pl_module.permutation_indices = permutation_indices
            pl_module.logger.experiment.add_scalar("B-ACC Q/Train", balanced_accuracy_Q, pl_module.current_epoch)
            print("TRAIN / B-ACC Q", balanced_accuracy_Q)
            print("")

            # Validation metrics
            y_subtype_pred_val = y_clusters_pred_val[val_subtype_mask].cpu().numpy()
            y_val_pred = np.sum(Q_validation * y_cond_clsf_pred_val.cpu().numpy(), axis=1)
            y_val_pred = (y_val_pred > 0.5).astype(int)

            balanced_accuracy_clusters, _ = balanced_accuracy_for_clusters(y_subtype_val, np.argmax(y_subtype_pred_val, 1), permutation_indices)
            pl_module.logger.experiment.add_scalar("Clusters Balanced Accuracy/Val", balanced_accuracy_clusters, pl_module.current_epoch)
            print("VAL / B-ACC clusters", balanced_accuracy_clusters)

            balanced_accuracy = balanced_accuracy_score(y_val, np.round(y_val_pred))
            print("VAL / balanced accuracy wrt clusters: ", balanced_accuracy)
            pl_module.logger.experiment.add_scalar("Class Balanced Accuracy/Val", balanced_accuracy, pl_module.current_epoch)

            overall_accuracy_clusters = overall_accuracy_for_clusters_and_classes(y_val, np.round(y_val_pred), y_subtype_val, np.argmax(y_subtype_pred_val, 1), permutation_indices)
            print("VAL / overall B-ACC clusters", overall_accuracy_clusters)
            pl_module.logger.experiment.add_scalar("Clusters overall Balanced Accuracy/Val", overall_accuracy_clusters, pl_module.current_epoch)
            print("")

            # save metrics for validation Q
            Q_subtype_val = Q_validation[val_subtype_mask]

            balanced_accuracy_Q, _ = balanced_accuracy_for_clusters(y_subtype_val, np.argmax(Q_subtype_val, 1), permutation_indices)
            pl_module.logger.experiment.add_scalar("Q Balanced Accuracy/Val", balanced_accuracy_Q, pl_module.current_epoch)
            print("VAL / B-ACC Q", balanced_accuracy_Q)

            overall_accuracy_Q = overall_accuracy_for_clusters_and_classes(y_val, np.round(y_val_pred), y_subtype_val, np.argmax(Q_subtype_val, 1), permutation_indices)
            print("VAL / overall B-ACC Q", overall_accuracy_Q)
            pl_module.logger.experiment.add_scalar("Q overall Balanced Accuracy/Val", overall_accuracy_Q, pl_module.current_epoch)

            balanced_accuracy = balanced_accuracy_score(y_val, np.round(y_val_pred))
            print("VAL / balanced accuracy wrt Q: ", balanced_accuracy)
            pl_module.logger.experiment.add_scalar("Class Balanced Accuracy wrt Q/Val", balanced_accuracy,
                                                   pl_module.current_epoch)

        else:
            pl_module.barycenters = barycenters

            # compute Q for training data
            Q_train = predict_proba_from_barycenters(latent_representations.cpu().numpy(), pl_module.barycenters)
            # set uniform probability for healthy samples !
            Q_train[y_train == 0] = (1/pl_module.n_clusters)

            # compute Q for validation data
            Q_validation = predict_proba_from_barycenters(latent_representations_val.cpu().numpy(), pl_module.barycenters)
            # set uniform probability for healthy samples !
            Q_validation[y_val == 0] = (1/pl_module.n_clusters)

        pl_module.Q["train"] = Q_train
        pl_module.Q["validation"] = Q_validation
