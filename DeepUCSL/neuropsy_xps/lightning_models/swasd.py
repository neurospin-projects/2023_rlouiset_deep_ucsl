import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression

from DeepUCSL.clustering_utils.scalers import PytorchRobustScaler, PytorchStandardScaler
from DeepUCSL.clustering_utils.centroids_reidentification import predict_proba_from_barycenters
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score
from DeepUCSL.neuropsy_xps.architectures.densenet121 import densenet121
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from DeepUCSL.clustering_utils.sinkhorn_knopp import distributed_sinkhorn

INF = 1e8

softmax = torch.nn.Softmax(dim=-1)

# define a Supervised Contrastive PyLightning class
class LitSwASD(LightningModule):
    def __init__(self, model_type, n_clusters, loss, loss_params, lr, fold):
        super().__init__()
        # define models, n_clusters
        self.model = densenet121(num_classes=n_clusters, method_name=model_type).float()
        self.model_type = model_type
        self.n_clusters = n_clusters

        # define loss parameters
        self.loss_params = loss_params
        self.loss = loss

        # define optimizer and scheduler parameters
        # self.step_size_scheduler = step_size_scheduler
        # self.gamma_scheduler = gamma_scheduler
        self.lr = lr

        # define train/val splits
        self.data_manager = None
        self.fold_index = fold

        # define pseudo-labels estimation parameters
        if loss_params["scaler"] == "standard" :
            self.scaler = PytorchStandardScaler()
        elif loss_params["scaler"] == "robust" :
            self.scaler = PytorchRobustScaler()
        else:
            self.scaler = None

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def training_step(self, train_batch, batch_idx):
        x, _, y, _ = train_batch
        x_i, x_j = x[:, 0], x[:, 1]

        z_1, p_1 = self.forward(x_i)
        z_2, p_2 = self.forward(x_j)

        # apply SK
        with torch.no_grad():
            q_1 = torch.clone(p_1)
            q_1[y["diagnosis"] == 1] = distributed_sinkhorn(q_1[y["diagnosis"] == 1]) #.detach()
            q_1[y["diagnosis"] == 0] = 1 / self.n_clusters
            q_2 = torch.clone(p_2)
            q_2[y["diagnosis"] == 1] = distributed_sinkhorn(q_2[y["diagnosis"] == 1]) #.detach()
            q_2[y["diagnosis"] == 0] = 1 / self.n_clusters

        # swap prediction problem
        clustering_loss = - 0.5 * (q_1 * torch.log(p_2) + q_2 * torch.log(p_1)).mean()

        # healthy and disease conditional predictions
        cond_classification_probabilities_1 = torch.exp(z_1 @ self.model.prototypes.weight.data.T / 0.1)
        cond_classification_probabilities_2 = torch.exp(z_2 @ self.model.prototypes.weight.data.T / 0.1)
        healthy_classification_probabilities_1 = torch.exp(z_1 @ self.model.healthy_prototype.weight.data.T / 0.1)
        healthy_classification_probabilities_2 = torch.exp(z_2 @ self.model.healthy_prototype.weight.data.T / 0.1)

        cond_classification_probabilities_1 = cond_classification_probabilities_1 / (cond_classification_probabilities_1 + healthy_classification_probabilities_1)
        cond_classification_probabilities_2 = cond_classification_probabilities_2 / (cond_classification_probabilities_2 + healthy_classification_probabilities_2)

        conditional_classification_loss = - 0.5 * (q_1 * y['diagnosis'][:, None] * torch.log(cond_classification_probabilities_2) +
                                           q_1 * (1 - y['diagnosis'])[:, None] * torch.log(1 - cond_classification_probabilities_2)).mean()
        conditional_classification_loss += - 0.5 * (q_2 * y['diagnosis'][:, None] * torch.log(cond_classification_probabilities_1) +
                                             q_2 * (1 - y['diagnosis'])[:, None] * torch.log(1 - cond_classification_probabilities_1)).mean()

        loss = clustering_loss + conditional_classification_loss

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y["diagnosis"].cpu().detach(),
            "subtypes": y['subtype'].cpu().detach()[y["diagnosis"].cpu().detach()==1]
        }

        return batch_dictionary

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        x, x_, y, _ = val_batch
        x_i, x_j = x[:, 0], x[:, 1]

        self.model.prototypes.weight.data = torch.nn.functional.normalize(self.model.prototypes.weight.data, dim=1, p=2)
        self.model.healthy_prototype.weight.data = torch.nn.functional.normalize(self.model.healthy_prototype.weight.data, dim=1, p=2)

        z_1, p_1 = self.forward(x_i)
        z_2, p_2 = self.forward(x_j)

        # apply SK
        with torch.no_grad():
            q_1 = torch.clone(p_1)
            q_1[y["diagnosis"] == 1] = distributed_sinkhorn(q_1[y["diagnosis"] == 1], epsilon=0.2) #.detach()
            q_1[y["diagnosis"] == 0] = 1 / self.n_clusters
            q_2 = torch.clone(p_2)
            q_2[y["diagnosis"] == 1] = distributed_sinkhorn(q_2[y["diagnosis"] == 1], epsilon=0.2) # .detach()
            q_2[y["diagnosis"] == 0] = 1 / self.n_clusters

        # swap prediction problem
        clustering_loss = - 0.5 * (q_1 * torch.log(p_2) + q_2 * torch.log(p_1)).mean()

        # healthy and disease conditional predictions
        cond_classification_probabilities_1 = torch.exp(z_1 @ self.model.prototypes.weight.data.T / 0.1)
        cond_classification_probabilities_2 = torch.exp(z_2 @ self.model.prototypes.weight.data.T / 0.1)
        healthy_classification_probabilities_1 = torch.exp(z_1 @ self.model.healthy_prototype.weight.data.T / 0.1)
        healthy_classification_probabilities_2 = torch.exp(z_2 @ self.model.healthy_prototype.weight.data.T / 0.1)

        cond_classification_probabilities_1 = cond_classification_probabilities_1 / (
                    cond_classification_probabilities_1 + healthy_classification_probabilities_1)
        cond_classification_probabilities_2 = cond_classification_probabilities_2 / (
                    cond_classification_probabilities_2 + healthy_classification_probabilities_2)

        conditional_classification_loss = - 0.5 * (
                    q_1 * y['diagnosis'][:, None] * torch.log(cond_classification_probabilities_2) +
                    q_1 * (1 - y['diagnosis'])[:, None] * torch.log(1 - cond_classification_probabilities_2)).mean()
        conditional_classification_loss += - 0.5 * (
                    q_2 * y['diagnosis'][:, None] * torch.log(cond_classification_probabilities_1) +
                    q_2 * (1 - y['diagnosis'])[:, None] * torch.log(1 - cond_classification_probabilities_1)).mean()

        loss = clustering_loss + conditional_classification_loss

        x_ = x_.detach()
        z, p = self.forward(x_)

        cond_classification_probabilities = torch.exp(z @ self.model.prototypes.weight.data.T / 0.1)
        healthy_classification_probabilities = torch.exp(z @ self.model.healthy_prototype.weight.data.T / 0.1)
        cond_classification_probabilities = cond_classification_probabilities / (
                cond_classification_probabilities + healthy_classification_probabilities)

        y_pred = (cond_classification_probabilities * p).sum(1)

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y["diagnosis"].cpu().detach(),
            "subtypes": y['subtype'].cpu().detach()[y["diagnosis"].cpu().detach()==1],
            "cluster_pred": p[y["diagnosis"].cpu().detach()==1],
            "class_pred": y_pred
        }

        return batch_dictionary

    @torch.no_grad()
    def test_step(self, test_batch, batch_idx):
        x, x_, y, _ = test_batch
        x_i, x_j = x[:, 0], x[:, 1]

        self.model.prototypes.weight.data = torch.nn.functional.normalize(self.model.prototypes.weight.data, dim=1, p=2)
        self.model.healthy_prototype.weight.data = torch.nn.functional.normalize(self.model.healthy_prototype.weight.data, dim=1, p=2)

        z_1, p_1 = self.forward(x_i)
        z_2, p_2 = self.forward(x_j)

        # apply SK
        with torch.no_grad():
            q_1 = torch.clone(p_1)
            q_1[y["diagnosis"] == 1] = distributed_sinkhorn(q_1[y["diagnosis"] == 1]) #.detach()
            q_1[y["diagnosis"] == 0] = 1 / self.n_clusters
            q_2 = torch.clone(p_2)
            q_2[y["diagnosis"] == 1] = distributed_sinkhorn(q_2[y["diagnosis"] == 1]) # .detach()
            q_2[y["diagnosis"] == 0] = 1 / self.n_clusters

        # swap prediction problem
        clustering_loss = - 0.5 * (q_1 * torch.log(p_2) + q_2 * torch.log(p_1)).mean()

        # healthy and disease conditional predictions
        cond_classification_probabilities_1 = torch.exp(z_1 @ self.model.prototypes.weight.data.T / 0.1)
        cond_classification_probabilities_2 = torch.exp(z_2 @ self.model.prototypes.weight.data.T / 0.1)
        healthy_classification_probabilities_1 = torch.exp(z_1 @ self.model.healthy_prototype.weight.data.T / 0.1)
        healthy_classification_probabilities_2 = torch.exp(z_2 @ self.model.healthy_prototype.weight.data.T / 0.1)

        cond_classification_probabilities_1 = cond_classification_probabilities_1 / (
                    cond_classification_probabilities_1 + healthy_classification_probabilities_1)
        cond_classification_probabilities_2 = cond_classification_probabilities_2 / (
                    cond_classification_probabilities_2 + healthy_classification_probabilities_2)

        conditional_classification_loss = - 0.5 * (
                    q_1 * y['diagnosis'][:, None] * torch.log(cond_classification_probabilities_2) +
                    q_1 * (1 - y['diagnosis'])[:, None] * torch.log(1 - cond_classification_probabilities_2)).mean()
        conditional_classification_loss += - 0.5 * (
                    q_2 * y['diagnosis'][:, None] * torch.log(cond_classification_probabilities_1) +
                    q_2 * (1 - y['diagnosis'])[:, None] * torch.log(1 - cond_classification_probabilities_1)).mean()

        loss = clustering_loss + conditional_classification_loss

        x_ = x_.detach()
        z, p = self.forward(x_)

        cond_classification_probabilities = torch.exp(z @ self.model.prototypes.weight.data.T / 0.1)
        healthy_classification_probabilities = torch.exp(z @ self.model.healthy_prototype.weight.data.T / 0.1)
        cond_classification_probabilities = cond_classification_probabilities / (
                cond_classification_probabilities + healthy_classification_probabilities)

        y_pred = (cond_classification_probabilities * p).sum(1)

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y["diagnosis"].cpu().detach(),
            "subtypes": y['subtype'].cpu().detach()[y["diagnosis"].cpu().detach()==1],
            "cluster_pred": p[y["diagnosis"].cpu().detach()==1],
            "class_pred": y_pred
        }

        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size_scheduler, gamma=self.gamma_scheduler)
        return [optimizer], []

    def validation_epoch_end(self, outputs):
        # average losses
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print("VAL loss : ", val_loss)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val",
                                          val_loss,
                                          self.current_epoch)

        subtypes = torch.cat([x['subtypes'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters_pred = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()
        y_pred = torch.cat([x['class_pred'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        labels = torch.cat([x['labels'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()

        if self.current_epoch > 0:
            # calculating correct and total clustering predictions
            idx = np.random.choice(range(0, len(subtypes)), 10, replace=False)
            print("SUBTYPES : ", subtypes[idx])
            print("PREDICTED SUBTYPES : ", np.argmax(y_clusters_pred, 1)[idx])
            balanced_accuracy_clusters, _ = balanced_accuracy_for_clusters(subtypes, np.argmax(y_clusters_pred, 1))
            print("VAL / balanced accuracy clusters: ", balanced_accuracy_clusters)

            # logging using tensorboard logger
            self.logger.experiment.add_scalar("Clustering Balanced Accuracy/Val",
                                              balanced_accuracy_clusters,
                                              self.current_epoch)

            # calculating correct classif
            idx = np.random.choice(range(0, len(labels)), 10, replace=False)
            print("LABELS : ", labels[idx])
            print("PREDICTED CLASSES : ", y_pred[idx])
            balanced_accuracy = balanced_accuracy_score(labels, np.round(y_pred))
            print("VAL / balanced accuracy: ", balanced_accuracy)

            # logging using tensorboard logger
            self.logger.experiment.add_scalar("Balanced Accuracy/Val",
                                              balanced_accuracy,
                                              self.current_epoch)

    def training_epoch_end(self, outputs):
        # average losses
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print("TRAIN loss : ", train_loss)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          train_loss,
                                          self.current_epoch)

    def test_epoch_end(self, outputs):
        # average losses
        test_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print("TEST loss : ", test_loss)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test",
                                          test_loss,
                                          self.current_epoch)

        subtypes = torch.cat([x['subtypes'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters_pred = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()
        y_pred = np.round(torch.cat([x['class_pred'] for x in outputs], dim=0).view(-1).cpu().detach().numpy())
        labels = torch.cat([x['labels'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()

        # calculating correct and total clustering predictions
        balanced_accuracy_clusters, _ = balanced_accuracy_for_clusters(subtypes, np.argmax(y_clusters_pred, 1))
        print("TEST / balanced accuracy clusters: ", balanced_accuracy_clusters)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Clustering Balanced Accuracy/Test",
                                          balanced_accuracy_clusters,
                                          self.current_epoch)

        # calculating correct classif
        balanced_accuracy = balanced_accuracy_score(labels, y_pred)
        print("TEST / balanced accuracy: ", balanced_accuracy)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Balanced Accuracy/Test",
                                          balanced_accuracy,
                                          self.current_epoch)

    def get_subtypes(self, X, y_subtype, y, label_to_cluster=1):
        subtype_mask = (y == label_to_cluster)
        X_subtype = X[subtype_mask]
        y_subtype = y_subtype[subtype_mask]
        return X_subtype, y_subtype, subtype_mask

    def train_dataloader(self):
        return self.data_manager.get_dataloader(self.fold_index, shuffle=True, train=True).train

    def val_dataloader(self):
        return self.data_manager.get_dataloader(validation=True, fold_index=self.fold_index).validation

    def test_dataloader(self):
        return self.data_manager.get_dataloader(test=True, fold_index=self.fold_index).test
