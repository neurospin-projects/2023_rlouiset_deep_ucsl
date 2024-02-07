import torch
from pytorch_lightning.core.lightning import LightningModule
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler

from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters
from DeepUCSL.medical_xps.architectures.resnet_18 import ResNet18
import torch.nn.functional as F

# define a SimCLR PyLightning class
class LitSimCLR(LightningModule):
    def __init__(self, model_type, n_clusters, loss, loss_params, lr, fold):
        super().__init__()
        self.model_type = model_type
        self.loss = loss
        self.loss_params = loss_params
        self.n_clusters = n_clusters
        self.lr = lr
        self.model = ResNet18(method_name=model_type).float()
        self.data_manager = None
        self.kmeans_rep = KMeans(n_clusters=n_clusters)

        # define pseudo-labels estimation parameters
        if loss_params["scaler"] == "standard":
            self.scaler = StandardScaler()
        elif loss_params["scaler"] == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def compute_sim_clr_loss(self, z_i, z_j):
        # contrastive loss
        contrastive_loss = NTXentLoss(z_i, z_j, self.loss_params["temperature"])
        return contrastive_loss

    def training_step(self, train_batch, batch_idx):
        x, _, y, pseudo_y = train_batch
        x_i, x_j = x[:, 0], x[:, 1]
        output_vector_i, _ = self.forward(x_i)
        output_vector_j, _ = self.forward(x_j)
        loss = self.compute_sim_clr_loss(output_vector_i, output_vector_j)

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y['subtype'].detach(),
        }
        return batch_dictionary

    def validation_step(self, val_batch, batch_idx):
        x, _, y, pseudo_y = val_batch
        x_i, x_j = x[:, 0], x[:, 1]
        output_vector_i, _ = self.forward(x_i)
        output_vector_j, _ = self.forward(x_j)
        loss = self.compute_sim_clr_loss(output_vector_i, output_vector_j)

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
        }
        return batch_dictionary

    def test_step(self, test_batch, batch_idx):
        x, _, y, pseudo_y = test_batch
        x_i, x_j = x[:, 0], x[:, 1]
        output_vector_i, _ = self.forward(x_i)
        output_vector_j, _ = self.forward(x_j)
        loss = self.compute_sim_clr_loss(output_vector_i, output_vector_j)

        # logs training loss in a dictionary
        self.log('test_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y['subtype'],
        }
        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer], []

    def validation_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        print("VAL loss : ", avg_loss)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val",
                                          avg_loss,
                                          self.current_epoch)

    def training_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)

    def test_epoch_end(self, outputs):
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test",
                                          avg_loss,
                                          self.current_epoch)

    def train_dataloader(self):
        return self.data_manager.get_dataloader(train=True, shuffle=True, fold_index=self.fold_index).train

    def val_dataloader(self):
        return self.data_manager.get_dataloader(validation=True, fold_index=self.fold_index).validation

    def test_dataloader(self):
        return self.data_manager.get_dataloader(test=True, fold_index=self.fold_index).test

# SimCLR loss
def NTXentLoss(z_i, z_j, temperature=0.1, INF=1e8):
    N = len(z_i)
    z_i = F.normalize(z_i, p=2, dim=-1) # dim [N, D]
    z_j = F.normalize(z_j, p=2, dim=-1) # dim [N, D]
    sim_zii = (z_i @ z_i.T) / temperature  # dim [N, N] => Upper triangle contains incorrect pairs
    sim_zjj = (z_j @ z_j.T) / temperature  # dim [N, N] => Upper triangle contains incorrect pairs
    sim_zij = (z_i @ z_j.T) / temperature  # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
    # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
    sim_zii = sim_zii - INF * torch.eye(N, device=z_i.device)
    sim_zjj = sim_zjj - INF * torch.eye(N, device=z_i.device)
    correct_pairs = torch.arange(N, device=z_i.device).long()
    loss_i = F.cross_entropy(torch.cat([sim_zij, sim_zii], dim=1), correct_pairs)
    loss_j = F.cross_entropy(torch.cat([sim_zij.T, sim_zjj], dim=1), correct_pairs)
    return loss_i + loss_j
