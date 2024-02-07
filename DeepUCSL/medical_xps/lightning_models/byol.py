import torch
from pytorch_lightning.core.lightning import LightningModule
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler

from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters
from DeepUCSL.medical_xps.architectures.resnet_18 import ResNet18
import torch.nn.functional as F

# define a SimCLR PyLightning class
class LitBYOL(LightningModule):
    def __init__(self, model_type, n_clusters, loss, loss_params, lr, fold):
        super().__init__()
        self.model_type = model_type
        self.loss = loss
        self.loss_params = loss_params
        self.n_clusters = n_clusters
        self.lr = lr
        # define models, n_clusters
        self.student_backbone = ResNet18(num_classes=n_clusters, method_name=model_type).float()
        self.model = ResNet18(method_name=model_type).float()
        # self.model = self.momentum_step(0, self.student_backbone, self.model)
        for param in self.model.parameters():
            param.requires_grad = False
        self.data_manager = None
        self.kmeans_rep = KMeans(n_clusters=n_clusters)

        # define pseudo-labels estimation parameters
        if loss_params["scaler"] == "standard":
            self.scaler = StandardScaler()
        elif loss_params["scaler"] == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

        self.m = 0.9

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def training_step(self, train_batch, batch_idx):
        x, _, y, pseudo_y = train_batch
        x_i, x_j = x[:, 0], x[:, 1]

        # forward pass: compute predicted outputs by passing inputs to the model
        z1, _, _ = self.model(x_i)
        z2, _, _ = self.model(x_j)

        _, _, p1 = self.student_backbone(x_i)
        _, _, p2 = self.student_backbone(x_j)

        loss = - torch.nn.functional.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss += - torch.nn.functional.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        loss /= 2

        self.model = self.momentum_step(self.m, self.student_backbone, self.model)

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

        # forward pass: compute predicted outputs by passing inputs to the model
        z1, _, _ = self.model(x_i)
        z2, _, _ = self.model(x_j)

        _, _, p1 = self.student_backbone(x_i)
        _, _, p2 = self.student_backbone(x_j)

        loss = - torch.nn.functional.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss += - torch.nn.functional.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        loss /= 2

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

        # forward pass: compute predicted outputs by passing inputs to the model
        z1, _, _ = self.model(x_i)
        z2, _, _ = self.model(x_j)

        _, _, p1 = self.student_backbone(x_i)
        _, _, p2 = self.student_backbone(x_j)

        loss = - torch.nn.functional.cosine_similarity(p1, z2.detach(), dim=-1).mean()
        loss += - torch.nn.functional.cosine_similarity(p2, z1.detach(), dim=-1).mean()
        loss /= 2

        # logs training loss in a dictionary
        self.log('test_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y['subtype'],
        }
        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.student_backbone.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size_scheduler, gamma=self.gamma_scheduler)
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

    @torch.no_grad()
    def momentum_step(self, m, encoder, momentum_encoder):
        '''
        Momentum step (Eq (2)).
        Args:
            - m (float): momentum value. 1) m = 0 -> copy parameter of encoder to key encoder
                                         2) m = 0.999 -> momentum update of key encoder
        '''
        params_q = encoder.state_dict()
        params_k = momentum_encoder.state_dict()

        dict_params_k = dict(params_k)

        for name in params_q:
            theta_k = dict_params_k[name]
            theta_q = params_q[name].data
            dict_params_k[name].data.copy_(m * theta_k + (1 - m) * theta_q)

        momentum_encoder.load_state_dict(dict_params_k)

        return momentum_encoder
