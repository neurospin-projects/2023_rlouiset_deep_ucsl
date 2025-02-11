from DeepUCSL.clustering_utils.centroids_reidentification import predict_proba_from_barycenters
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes
from DeepUCSL.clustering_utils.scalers import PytorchStandardScaler, PytorchRobustScaler
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score

from DeepUCSL.clustering_utils.sinkhorn_knopp import distributed_sinkhorn
from DeepUCSL.deep_ucsl_loss import *
from DeepUCSL.medical_xps.architectures.resnet_18 import ResNet18

# define a lightning model for SwAV
class LitSwAV(LightningModule):
    def __init__(self, model_type, n_clusters, loss, loss_params, lr, fold):
        super().__init__()
        # define models, n_clusters
        self.model = ResNet18(num_classes=n_clusters, method_name=model_type).float()
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.Q = {"train": None,
                  "validation": None,
                  "test": None}

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

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size_scheduler, gamma=self.gamma_scheduler)
        return [optimizer], []

    def training_step(self, train_batch, batch_idx):
        x, _, y, _ = train_batch
        x_i, x_j = x[:, 0], x[:, 1]

        _, p_1 = self.forward(x_i)
        _, p_2 = self.forward(x_j)
        # apply SK
        with torch.no_grad():
            q_1 = distributed_sinkhorn(p_1)
            q_2 = distributed_sinkhorn(p_2)
        # swap prediction problem
        loss = - 0.5 * (q_1 * torch.log(p_2) + q_2 * torch.log(p_1)).mean()

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "cluster_pred": p_1,
            "labels": y['subtype']
        }
        return batch_dictionary

    def validation_step(self, val_batch, batch_idx):
        x, x_, y, _ = val_batch

        self.model.prototypes.weight.data = torch.nn.functional.normalize(self.model.prototypes.weight.data, dim=1, p=2)

        x_ = x_.detach()
        _, p = self.forward(x_)

        x_i, x_j = x[:, 0], x[:, 1]

        _, p_1 = self.forward(x_i)
        _, p_2 = self.forward(x_j)

        # apply SK
        with torch.no_grad():
            q_1 = distributed_sinkhorn(p_1)
            q_2 = distributed_sinkhorn(p_2)

        # swap prediction problem
        loss = - 0.5 * (q_1 * torch.log(p_2) + q_2 * torch.log(p_1)).mean()

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "cluster_pred": p,
            "labels": y['subtype']
        }
        return batch_dictionary

    def test_step(self, test_batch, batch_idx):
        x, x_, y, _ = test_batch
        x_ = x_.detach()
        _, p = self.forward(x_)

        x_i, x_j = x[:, 0], x[:, 1]

        _, p_1 = self.forward(x_i)
        _, p_2 = self.forward(x_j)

        # apply SK
        with torch.no_grad():
            q_1 = distributed_sinkhorn(p_1)
            q_2 = distributed_sinkhorn(p_2)

        # swap prediction problem
        loss = - 0.5 * (q_1 * torch.log(p_2) + q_2 * torch.log(p_1)).mean()

        # logs training loss in a dictionary
        self.log('test_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "cluster_pred": p,
            "labels": y['subtype']
        }
        return batch_dictionary

    def validation_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters_pred = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()

        if self.current_epoch > 0:
            # calculating correct and total clustering predictions
            balanced_accuracy_clusters, _ = balanced_accuracy_for_clusters(labels, np.argmax(y_clusters_pred, 1))
            print("VAL / balanced accuracy clusters: ", balanced_accuracy_clusters)

            # logging using tensorboard logger
            self.logger.experiment.add_scalar("Clustering Balanced Accuracy/Val",
                                              balanced_accuracy_clusters,
                                              self.current_epoch)

    def training_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters_pred = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()

        # calculating correct and total clustering predictions
        balanced_accuracy_clusters, _ = balanced_accuracy_for_clusters(labels, np.argmax(y_clusters_pred, 1))
        print("TRAIN / balanced accuracy clusters: ", balanced_accuracy_clusters)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Clustering Balanced Accuracy/Train",
                                          balanced_accuracy_clusters,
                                          self.current_epoch)

    def test_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters_pred = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()

        balanced_accuracy_clusters, _ = balanced_accuracy_for_clusters(labels, np.argmax(y_clusters_pred, 1))
        print("TEST / balanced accuracy clusters: ", balanced_accuracy_clusters)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Clustering Balanced Accuracy/Test",
                                          balanced_accuracy_clusters,
                                          self.current_epoch)

    def train_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["train"], phase="train")
        return self.data_manager.get_dataloader(train=True, fold_index=self.fold_index).train

    def val_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["validation"], phase="validation")
        return self.data_manager.get_dataloader(validation=True, fold_index=self.fold_index).validation

    def test_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["test"], phase="test")
        return self.data_manager.get_dataloader(test=True, fold_index=self.fold_index).test
