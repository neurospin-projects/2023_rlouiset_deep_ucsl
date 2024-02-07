from DeepUCSL.clustering_utils.centroids_reidentification import predict_proba_from_barycenters
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes
from DeepUCSL.clustering_utils.scalers import PytorchStandardScaler, PytorchRobustScaler
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score
from DeepUCSL.deep_ucsl_loss import *
from DeepUCSL.neuropsy_xps.architectures.densenet121 import densenet121
from DeepUCSL.neuropsy_xps.utils import create_diagnosis_and_subtype_dict


class LitDeepCluster(LightningModule):
    def __init__(self, model_type, n_clusters, loss, loss_params, lr, fold):
        super().__init__()
        # define models, n_clusters
        self.model = densenet121(num_classes=n_clusters, method_name=model_type).float()
        self.model_type = model_type
        self.n_clusters = n_clusters
        self.Q = {"train": None,
                  "validation": None,
                  "test": None}

        # define loss parameters
        self.loss_params = loss_params
        self.loss = loss

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

        self.prototypes = None
        self.km = None
        self.temperature = 0.1
        self.weights = torch.tensor([1/n_clusters]*n_clusters).float().cuda()

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size_scheduler, gamma=self.gamma_scheduler)
        return [optimizer], []

    def compute_deep_cluster_loss(self, clustering_probability, pseudo_labels):
        # clustering regularization loss
        clustering_loss = self.weights[:,None] * F.one_hot(pseudo_labels.argmax(1), num_classes=self.n_clusters) * torch.log(clustering_probability)
        return - clustering_loss.mean()

    def training_step(self, train_batch, batch_idx):
        x, _, y, pseudo_y = train_batch
        y = create_diagnosis_and_subtype_dict(y)

        head_vector, representation_vector = self.forward(x)
        clustering_probability = torch.exp(head_vector @ self.prototypes.T / self.temperature)
        clustering_probability = clustering_probability / clustering_probability.sum(1)[:, None]
        loss = self.compute_deep_cluster_loss(clustering_probability, pseudo_y)

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "cluster_pred": clustering_probability,
            "non_linear_heads": head_vector,
            "labels": y['subtype']
        }
        return batch_dictionary

    def validation_step(self, val_batch, batch_idx):
        x, _, y, pseudo_y = val_batch
        y = create_diagnosis_and_subtype_dict(y)
        head_vector, representation_vector = self.forward(x)
        clustering_probability = torch.exp(head_vector @ self.prototypes.T / self.temperature)
        clustering_probability = clustering_probability / clustering_probability.sum(1)[:, None]
        loss = self.compute_deep_cluster_loss(clustering_probability, pseudo_y)

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "cluster_pred": clustering_probability,
            "non_linear_heads": head_vector,
            "labels": y['subtype']
        }
        return batch_dictionary

    def test_step(self, test_batch, batch_idx):
        _, x_, y, pseudo_y = test_batch
        y = create_diagnosis_and_subtype_dict(y)
        head_vector, representation_vector = self.forward(x_)
        clustering_probability = torch.exp(head_vector @ self.prototypes.T / self.temperature)
        clustering_probability = clustering_probability / clustering_probability.sum(1)[:, None]

        # info that are saved until epoch end
        batch_dictionary = {
            "cluster_pred": clustering_probability,
            "representations": representation_vector,
            "non_linear_heads": head_vector,
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
        self.logger.experiment.add_scalar("clusters Balanced Accuracy/Test",
                                          balanced_accuracy_clusters,
                                          self.current_epoch)

    def train_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["train"], phase="train", n_clusters=self.n_clusters)
        return self.data_manager.get_dataloader(train=True, fold_index=self.fold_index).train

    def val_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["validation"], phase="validation", n_clusters=self.n_clusters)
        return self.data_manager.get_dataloader(validation=True, fold_index=self.fold_index).validation

    def test_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["test"], phase="test", n_clusters=self.n_clusters)
        return self.data_manager.get_dataloader(test=True, fold_index=self.fold_index).test
