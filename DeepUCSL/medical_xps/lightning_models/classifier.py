import numpy as np
import torch

from DeepUCSL.clustering_utils.scalers import PytorchRobustScaler, PytorchStandardScaler
from DeepUCSL.clustering_utils.centroids_reidentification import predict_proba_from_barycenters
from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters, overall_accuracy_for_clusters_and_classes
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score
from DeepUCSL.medical_xps.architectures.resnet_18 import ResNet18

# define a binary classifier lightning model
class LitClassifier(LightningModule):
    def __init__(self, model_type, n_clusters, n_classes, loss, loss_params, lr, fold):
        super().__init__()

        # define models, n_clusters
        self.model = ResNet18(num_classes=n_classes, method_name=model_type).float()
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


    def forward(self, x):
        return self.model(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def compute_loss(self, preds, labels):
        loss = cross_entropy_loss(preds, labels, class_weights=self.loss_params["class_weights"])
        return loss

    def training_step(self, train_batch, batch_idx):
        x, _, y, _ = train_batch
        preds, _ = self.forward(x)

        if self.model_type == "Classification":
            one_hot_labels = torch.nn.functional.one_hot(y["diagnosis"].long(), num_classes=self.n_classes).float()
        elif self.model_type == "ClassificationOfSubgroups":
            one_hot_labels = torch.nn.functional.one_hot(y["subtype"].long(), num_classes=self.n_classes).float()
        else:
            return NotImplementedError

        loss = self.compute_loss(preds, one_hot_labels)

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "y_pred": preds.detach(),
            "labels": one_hot_labels.detach(),
        }

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx):
        _, x, y, _ = val_batch
        preds, _ = self.forward(x)

        if self.model_type == "Classification":
            one_hot_labels = torch.nn.functional.one_hot(y["diagnosis"].long(), num_classes=self.n_classes).float()
        elif self.model_type == "ClassificationOfSubgroups":
            one_hot_labels = torch.nn.functional.one_hot(y["subtype"].long(), num_classes=self.n_classes).float()
        else:
            return NotImplementedError

        loss = self.compute_loss(preds, one_hot_labels)

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "y_pred": preds,
            "labels": one_hot_labels
        }

        return batch_dictionary

    def test_step(self, batch, batch_idx):
        x, _, y, _ = batch
        preds, _ = self.forward(x)

        if self.model_type == "Classification":
            one_hot_labels = torch.nn.functional.one_hot(y["diagnosis"].long(), num_classes=self.n_classes).float()
        elif self.model_type == "ClassificationOfSubgroups":
            one_hot_labels = torch.nn.functional.one_hot(y["subtype"].long(), num_classes=self.n_classes).float()
        else:
            return NotImplementedError

        loss = self.compute_loss(preds, one_hot_labels)

        # logs training loss in a dictionary
        self.log('test_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "y_pred": preds,
            "labels": one_hot_labels
        }

        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size_scheduler, gamma=self.gamma_scheduler)
        return [optimizer], []

    def validation_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # get labels and predictions
        labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().detach().numpy()
        y_preds = torch.cat([x['y_pred'] for x in outputs], dim=0).cpu().detach().numpy()

        # calculating correct and total predictions
        accuracy = balanced_accuracy_score(np.argmax(labels, 1), np.argmax(y_preds, 1))
        print("VAL accuracy : ", accuracy)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val",
                                          avg_loss,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Val",
                                          accuracy,
                                          self.current_epoch)

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # get labels and predictions
        labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().detach().numpy()
        y_preds = torch.cat([x['y_pred'] for x in outputs], dim=0).cpu().detach().numpy()

        # calculating correct and total predictions
        accuracy = balanced_accuracy_score(np.argmax(labels, 1), np.argmax(y_preds, 1))
        print("TRAIN accuracy : ", accuracy)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Train",
                                          accuracy,
                                          self.current_epoch)

    def test_epoch_end(self, outputs):
        #  the function is called after every epoch is completed
        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # get labels and predictions
        labels = torch.cat([x['labels'] for x in outputs], dim=0).cpu().detach().numpy()
        y_preds = torch.cat([x['y_pred'] for x in outputs], dim=0).cpu().detach().numpy()

        # calculating correct and total predictions
        accuracy = balanced_accuracy_score(np.argmax(labels, 1), np.argmax(y_preds, 1))
        print("TEST accuracy : ", accuracy)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Test",
                                          avg_loss,
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test",
                                          accuracy,
                                          self.current_epoch)

    def get_subtypes(self, X, y_subtype, y, label_to_cluster=1):
        subtype_mask = (y == label_to_cluster)
        X_subtype = X[subtype_mask]
        y_subtype = y_subtype[subtype_mask]
        return X_subtype, y_subtype, subtype_mask

    def train_dataloader(self):
        return self.data_manager.get_dataloader(train=True, shuffle=True, fold_index=self.fold_index).train

    def val_dataloader(self):
        return self.data_manager.get_dataloader(validation=True, fold_index=self.fold_index).validation

    def test_dataloader(self):
        return self.data_manager.get_dataloader(test=True, fold_index=self.fold_index).test

def binary_cross_entropy_loss(logits, labels, class_weights=None):
    if class_weights is None:
        class_weights = [1, 1]
    loss = class_weights[1] * labels * torch.log(logits) + class_weights[0] * (1 - labels) * torch.log(1 - logits)
    return -loss.mean()

def cross_entropy_loss(logits, labels, class_weights=None):
    if class_weights is None:
        loss = labels * torch.log(logits)
    else:
        class_weights = torch.tensor(class_weights).cuda().float()[None, :]
        loss = class_weights * labels * torch.log(logits)
    return - loss.mean()
