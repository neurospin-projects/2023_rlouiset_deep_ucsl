from sklearn.cluster import KMeans

from DeepUCSL.clustering_utils import *
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score

from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters
from DeepUCSL.deep_ucsl_loss import *
from DeepUCSL.neuropsy_xps.architectures.densenet121 import densenet121
from DeepUCSL.neuropsy_xps.utils import create_diagnosis_and_subtype_dict


class LitPCL(LightningModule):
    def __init__(self, model_type, n_clusters, loss, loss_params, lr, fold):
        super().__init__()
        self.model_type = model_type
        self.loss = loss
        self.loss_params = loss_params
        self.n_clusters = n_clusters
        self.lr = lr

        # define models, n_clusters
        self.student_backbone = densenet121(num_classes=n_clusters, method_name=model_type).float()
        self.model = densenet121(num_classes=n_clusters, method_name=model_type).float()
        self.model = self.momentum_step(0, self.student_backbone, self.model)
        for param in self.model.parameters():
            param.requires_grad = False

        self.data_manager = None
        self.fold_index = fold
        self.barycenters = None
        self.Q = {"train": None,
                  "validation": None,
                  "test": None}
        self.km = None
        self.prototypes = None
        self.temperatures = None
        self.temperature = self.loss_params["temperature"]
        self.alpha = 10
        self.m = 0.9

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def compute_pcl_loss(self, z_i, z_j, pseudo_labels):
        # contrastive loss
        if self.current_epoch < self.loss_params["pre_training_epochs"]:
            contrastive_loss = NTXentLoss(z_i, z_j, self.temperature)
        else:
            contrastive_loss = PCLLoss(z_i, z_j, self.prototypes, pseudo_labels, self.temperatures)
        return contrastive_loss

    def training_step(self, train_batch, batch_idx):
        x, _, y, pseudo_y = train_batch
        y = create_diagnosis_and_subtype_dict(y)
        output_vector_i = self.student_backbone.forward(x[:, 0])
        output_vector_j = self.student_backbone.forward(x[:, 1])
        loss = self.compute_pcl_loss(output_vector_i, output_vector_j, pseudo_y)

        if self.current_epoch <= self.loss_params["pre_training_epochs"]:
            self.model = self.momentum_step(0, self.student_backbone, self.model)
        else:
            self.model = self.momentum_step(self.m, self.student_backbone, self.model)

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y["subtype"],
        }
        return batch_dictionary

    def validation_step(self, val_batch, batch_idx):
        x, _, y, pseudo_y = val_batch
        y = create_diagnosis_and_subtype_dict(y)
        output_vector_i = self.student_backbone.forward(x[:, 0])
        output_vector_j = self.student_backbone.forward(x[:, 1])
        loss = self.compute_pcl_loss(output_vector_i, output_vector_j, pseudo_y)

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "labels": y["subtype"],
        }
        return batch_dictionary

    def test_step(self, test_batch, batch_idx):
        _, x_, y, pseudo_y = test_batch
        y = create_diagnosis_and_subtype_dict(y)
        output_vector_ = self.model.forward(x_)
        y_pred_ = F.softmax(output_vector_ @ self.prototypes.T / self.temperatures[None, :], 1)

        # info that are saved until epoch end
        batch_dictionary = {
            "labels": y["subtype"],
            "clustering_pred": y_pred_,
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
        # concatenate training representations
        y_pred_ = torch.cat([x['clustering_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()
        labels = torch.cat([x['labels'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()

        clustering_bacc, _ = balanced_accuracy_for_clusters(labels, np.argmax(y_pred_, 1))

        self.logger.experiment.add_scalar("Balanced Accuracy Prototypes/Test",
                                          clustering_bacc, self.current_epoch)

    def train_dataloader(self):
        return self.data_manager.get_dataloader(train=True, fold_index=self.fold_index).train

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

# PCL loss
def PCLLoss(z_i, z_j, prototypes, pseudo_labels, temperatures):
    ntx_loss = NTXentLoss(z_i, z_j)
    proto_loss = proto_pcl_loss(z_i, pseudo_labels, prototypes, temperatures)
    proto_loss += proto_pcl_loss(z_j, pseudo_labels, prototypes, temperatures)
    loss = ntx_loss + proto_loss
    return loss

def proto_pcl_loss(v, pseudo_labels, prototypes, temperatures):
    numerator = torch.exp(v @ ((pseudo_labels @ prototypes).T / (temperatures[:, None] * pseudo_labels.T).sum()))
    denominator = torch.exp(v @ (prototypes / temperatures[:, None]).sum(0))
    loss = numerator / denominator[:, None]
    return loss.mean()
