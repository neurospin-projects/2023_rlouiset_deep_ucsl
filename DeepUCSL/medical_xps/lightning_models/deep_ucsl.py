from torch.optim.lr_scheduler import LambdaLR

from DeepUCSL.clustering_utils.centroids_reidentification import predict_proba_from_barycenters
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes
from DeepUCSL.clustering_utils.scalers import PytorchStandardScaler, PytorchRobustScaler
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score
from DeepUCSL.deep_ucsl_loss import *
from DeepUCSL.medical_xps.architectures.resnet_18 import ResNet18

# define a classical classifier architecture
class LitDeepUCSL(LightningModule):
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

        # define pseudo-labels estimation parameters
        if loss_params["scaler"] == "standard" :
            self.scaler = PytorchStandardScaler()
        elif loss_params["scaler"] == "robust" :
            self.scaler = PytorchRobustScaler()
        else:
            return NotImplementedError

        self.barycenters = None

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # scheduler = LambdaLR(optimizer, )
        return [optimizer], []

    def train_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["train"], phase="train")
        return self.data_manager.get_dataloader(self.fold_index, shuffle=True, train=True).train

    def val_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["validation"], phase="validation")
        return self.data_manager.get_dataloader(validation=True, fold_index=self.fold_index).validation

    def test_dataloader(self):
        self.data_manager.set_pseudo_labels(self.fold_index, self.Q["test"], phase="test")
        return self.data_manager.get_dataloader(test=True, fold_index=self.fold_index).test

    def compute_ucsl_loss(self, conditional_probability, clustering_probability, labels, pseudo_labels):
        # compute linear interpolation for pseudo-labels
        if self.loss_params["pseudo_labels_confidence"] == "linear" and labels.sum() > 0:
            alpha = max(1 - self.current_epoch / self.loss_params["max_confidence_epoch"], 0)
            pseudo_labels[labels == 1] = (alpha * pseudo_labels[labels == 1] +
                                          (1 - alpha) * torch.nn.functional.one_hot(pseudo_labels[labels == 1].argmax(1),
                                                                                    num_classes=self.n_clusters).float())
        if self.loss_params["pseudo_labels_confidence"] == "hard" and labels.sum() > 0:
            pseudo_labels[labels == 1] = torch.nn.functional.one_hot(pseudo_labels[labels == 1].argmax(1),
                                                                     num_classes=self.n_clusters).float()

        # calculate the conditional classification loss
        conditional_classification_loss = conditional_cross_entropy_loss(conditional_probability, labels, pseudo_labels,
                                                                         class_weights=self.loss_params["class_weights"])

        # clustering regularization loss
        pos_clustering_loss, neg_clustering_loss = clustering_regularization_loss(clustering_probability, labels,
                                                                        pseudo_labels,
                                                                        class_weights=self.loss_params["class_weights"])

        return conditional_classification_loss, pos_clustering_loss + neg_clustering_loss

    def training_step(self, train_batch, batch_idx):
        x, _, y, pseudo_y = train_batch

        conditional_probability, clustering_probability, _ = self.forward(x)

        # identifying number of correct predictions in a given batch
        y_pred = (clustering_probability * conditional_probability).sum(1)

        conditional_classif_loss, clustering_loss = self.compute_ucsl_loss(
            conditional_probability, clustering_probability, y['diagnosis'], pseudo_y)

        loss = self.loss_params["conditional_classification_weight"] * conditional_classif_loss + \
               self.loss_params["clustering_weight"] * clustering_loss

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "conditional_classif_loss": conditional_classif_loss,
            "clustering_loss": clustering_loss,
            "y_pred": y_pred,
            "cluster_pred": clustering_probability,
            "labels": y["diagnosis"],
            "subtypes": y['subtype'],
        }

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx):
        _, x, y, pseudo_y = val_batch

        conditional_probability, clustering_probability, _ = self.forward(x)

        # identifying number of correct predictions in a given batch
        y_pred = (clustering_probability * conditional_probability).sum(1)

        if self.current_epoch > 0:
            conditional_classif_loss, clustering_loss = self.compute_ucsl_loss(conditional_probability,
                                                                               clustering_probability, y['diagnosis'],
                                                                               pseudo_y)

            loss = self.loss_params["conditional_classification_weight"] * conditional_classif_loss + \
                   self.loss_params["clustering_weight"] * clustering_loss
        else:
            loss, conditional_classif_loss, clustering_loss = torch.tensor(0.0),torch.tensor(0.0), torch.tensor(0.0)

        # logs validation loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "conditional_classif_loss": conditional_classif_loss,
            "clustering_loss": clustering_loss,
            "labels": y['diagnosis'],
            "subtypes": y['subtype'],
        }

        return batch_dictionary

    def test_step(self, test_batch, batch_idx):
        x, _, y, pseudo_y = test_batch

        conditional_probability, clustering_probability, head_vector = self.forward(x)

        # identifying number of correct predictions in a given batch
        y_pred = (clustering_probability * conditional_probability).sum(1)

        # info that are saved until epoch end
        batch_dictionary = {
            "y_pred": y_pred,
            "cluster_pred": clustering_probability,
            "labels": y['diagnosis'],
            "subtypes": y['subtype'],
            "representations": head_vector
        }

        return batch_dictionary

    def validation_epoch_end(self, outputs):
        # calculating average loss
        avg_cdntl_clsf_loss = torch.stack([x['conditional_classif_loss'] for x in outputs]).mean()
        avg_cluster_loss = torch.stack([x['clustering_loss'] for x in outputs]).mean()

        print("VAL loss : ", (self.loss_params["conditional_classification_weight"] * avg_cdntl_clsf_loss + self.loss_params["clustering_weight"] * avg_cluster_loss).item())

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Conditional Classification Loss/Val",
                                          avg_cdntl_clsf_loss,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Clustering Regularization Loss/Val",
                                          avg_cluster_loss,
                                          self.current_epoch)

    def training_epoch_end(self, outputs):
        labels = torch.cat([x['labels'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_preds = torch.cat([x['y_pred'] for x in outputs], dim=0).view(-1).cpu().cpu().detach().numpy()
        y_preds = (y_preds>0.5).astype(int)

        # calculating average loss
        avg_cdntl_clsf_loss = torch.stack([x['conditional_classif_loss'] for x in outputs]).mean()
        avg_cluster_loss = torch.stack([x['clustering_loss'] for x in outputs]).mean()

        # calculating correct and total predictions
        balanced_accuracy = balanced_accuracy_score(labels, y_preds)

        print("TRAIN /  loss : ", (self.loss_params["conditional_classification_weight"] * avg_cdntl_clsf_loss +
                                   self.loss_params["clustering_weight"] * avg_cluster_loss).item())

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Conditional Classification Loss/Train",
                                          avg_cdntl_clsf_loss,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Regularization Clustering Loss/Train",
                                          avg_cluster_loss,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Balanced Accuracy/Train",
                                          balanced_accuracy,
                                          self.current_epoch)

    def test_epoch_end(self, outputs):
        # concatenate training representations
        representations = torch.cat([x['representations'] for x in outputs], dim=0).view(-1, 512).detach()
        representations = self.scaler.transform(representations)

        labels = torch.cat([x['labels'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        subtypes = torch.cat([x['subtypes'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_preds = torch.cat([x['y_pred'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()

        # predict probability to belong to a cluster
        Q_test = predict_proba_from_barycenters(representations.cpu().numpy(), self.barycenters)

        # calculating correct and total predictions
        balanced_accuracy = balanced_accuracy_score(labels, np.round(y_preds))

        # calculating correct and total clustering predictions
        y_subtype_pred, y_subtype, mask_subtype = self.get_subtypes(np.argmax(y_clusters, 1), subtypes, labels)
        Q_subtype_test = Q_test[mask_subtype]

        balanced_accuracy_cluster, _ = balanced_accuracy_for_clusters(y_subtype, y_subtype_pred, self.permutation_indices)
        balanced_accuracy_Q, _ = balanced_accuracy_for_clusters(y_subtype, np.argmax(Q_subtype_test, 1), self.permutation_indices)

        print("TEST / balanced accuracy Q: ", balanced_accuracy_Q)
        print("TEST / balanced accuracy cluster: ", balanced_accuracy_cluster)
        print("TEST / balanced accuracy: ", balanced_accuracy)

        overall_accuracy_Q = overall_accuracy_for_clusters_and_classes(labels, np.round(y_preds), y_subtype, np.argmax(Q_subtype_test, 1), self.permutation_indices)
        print("TEST / overall B-ACC Q", overall_accuracy_Q)
        self.logger.experiment.add_scalar("Q overall Balanced Accuracy/Test", overall_accuracy_Q, self.current_epoch)

        overall_accuracy_clusters = overall_accuracy_for_clusters_and_classes(labels, np.round(y_preds), y_subtype, y_subtype_pred, self.permutation_indices)
        print("TEST / overall B-ACC clusters", overall_accuracy_clusters)
        self.logger.experiment.add_scalar("clusters overall Balanced Accuracy/Test", overall_accuracy_clusters, self.current_epoch)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Balanced Accuracy/Test",
                                          balanced_accuracy,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Clustering Balanced Accuracy/Test",
                                          balanced_accuracy_cluster,
                                          self.current_epoch)

        self.logger.experiment.add_scalar("Q Balanced Accuracy/Test",
                                          balanced_accuracy_Q,
                                          self.current_epoch)

    def get_subtypes(self, X, y_subtype, y, label_to_cluster=1):
        subtype_mask = (y == label_to_cluster)
        X_subtype = X[subtype_mask]
        y_subtype = y_subtype[subtype_mask]
        return X_subtype, y_subtype, subtype_mask

