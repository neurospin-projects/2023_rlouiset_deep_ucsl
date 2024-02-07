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

INF = 1e8

# define a Supervised Contrastive PyLightning class
class LitSupCon(LightningModule):
    def __init__(self, model_type, n_classes, n_clusters, loss, loss_params, lr, fold):
        super().__init__()
        # define models, n_clusters
        self.model = densenet121(num_classes=n_classes, method_name=model_type).float()
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.n_classes = n_classes

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
            return NotImplementedError

        self.kmeans_representation = KMeans(n_clusters)
        self.log_reg = LogisticRegression()
        self.permutation_indices = np.arange(n_clusters)

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def compute_supcon_loss(self, z_i, z_j, labels):
        loss = SupervisedNTXentLoss(z_i, z_j, labels)
        return loss

    def training_step(self, train_batch, batch_idx):
        x, _, y, _ = train_batch
        head_1, _ = self.forward(x[:, 0])
        head_2, _ = self.forward(x[:, 1])
        loss = self.compute_supcon_loss(head_1, head_2, y["diagnosis"])

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y["diagnosis"].cpu().detach(),
            "subtypes": y['subtype'].cpu().detach()
        }

        return batch_dictionary

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx):
        x, x_, y, _ = val_batch
        heads, representations = self.forward(x_)
        heads, representations = heads.detach(), representations.detach()
        head_1, _ = self.forward(x[:, 0])
        head_2, _ = self.forward(x[:, 1])
        loss = self.compute_supcon_loss(head_1, head_2, y["diagnosis"])

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y["diagnosis"].cpu().detach(),
            "subtypes": y['subtype'].cpu().detach(),
            "representations": representations,
            "heads": heads
        }

        return batch_dictionary

    @torch.no_grad()
    def test_step(self, test_batch, batch_idx):
        x, x_, y, _ = test_batch
        heads, representations = self.forward(x_)
        heads, representations = heads.detach(), representations.detach()
        head_1, _ = self.forward(x[:, 0])
        head_2, _ = self.forward(x[:, 1])
        loss = self.compute_supcon_loss(head_1, head_2, y["diagnosis"])

        # logs training loss in a dictionary
        self.log('test_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y["diagnosis"].cpu().detach(),
            "subtypes": y['subtype'].cpu().detach(),
            "representations": representations,
            "heads": heads
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

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test",
                                          test_loss,
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


def discrete_kernel(y1, y2):
    """
    :param y1: matrix shape [N, *]
    :param y2: matrix shape [N, *]
    :return: matrix M shape [N, N] where M[i][j] = 1({y1[i] == y2[j]})
    """
    M = (pairwise_distances(y1, y2, metric="hamming") == 0)  # dist = proportion of components disagreeing
    return M.astype(np.float)


# SupCon loss function
def SupervisedNTXentLoss(z_i, z_j, labels, temperature=0.1):
    N = len(z_i)
    assert N == len(labels), "Unexpected labels length: %i" % len(labels)
    z_i = F.normalize(z_i, p=2, dim=-1)  # dim [N, D]
    z_j = F.normalize(z_j, p=2, dim=-1)  # dim [N, D]
    sim_zii = (z_i @ z_i.T) / temperature  # dim [N, N] => Upper triangle contains incorrect pairs
    sim_zjj = (z_j @ z_j.T) / temperature  # dim [N, N] => Upper triangle contains incorrect pairs
    sim_zij = (z_i @ z_j.T) / temperature  # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
    # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
    sim_zii = sim_zii - INF * torch.eye(N, device=z_i.device)
    sim_zjj = sim_zjj - INF * torch.eye(N, device=z_i.device)

    all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy()  # [2N, *]
    weights = discrete_kernel(all_labels, all_labels)  # [2N, 2N]
    weights = weights * (1 - np.eye(2 * N))  # puts 0 on the diagonal
    weights /= weights.sum(axis=1)
    # compute imbalance weights
    # imbalance_weights = np.array([0.66, 0.33]) # to get rid of
    # imbalance_weights = imbalance_weights[all_labels.astype(np.int)]
    # weights *= imbalance_weights
    # if 'rbf' kernel and sigma->0, we retrieve the classical NTXenLoss (without labels)
    sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1),
                       torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0)  # [2N, 2N]
    log_sim_Z = F.log_softmax(sim_Z, dim=1)

    loss = -1. / N * (torch.from_numpy(weights).to(z_i.device) * log_sim_Z).sum()

    return loss
