from sklearn.cluster import KMeans

from DeepUCSL.clustering_utils import *
from pytorch_lightning.core.lightning import LightningModule
from sklearn.metrics import balanced_accuracy_score

from DeepUCSL.clustering_utils.metrics import balanced_accuracy_for_clusters
from DeepUCSL.deep_ucsl_loss import *
from DeepUCSL.neuropsy_xps.architectures.densenet121 import densenet121
from DeepUCSL.neuropsy_xps.utils import create_diagnosis_and_subtype_dict

cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
def _cosine_simililarity(x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

# define a classical classifier architecture
class LitSCAN(LightningModule):
    def __init__(self, model_type, n_clusters, loss, loss_params, lr, fold):
        super().__init__()
        self.model_type = model_type
        self.loss = loss
        self.n_clusters = n_clusters
        self.lr = lr
        self.backbone = densenet121(method_name=model_type).float()
        self.data_manager = None
        self.fold_index = fold
        self.loss_params = loss_params
        self.stack = None
        self.stack_size = 2048
        self.self_label_loss = ConfidenceBasedCE()
        self.scan_loss = SCANLoss()

    def refresh_stack(self, batch):
        len_batch = len(batch)
        new_stack = torch.cat((self.stack[len_batch:], batch), dim=0)
        return new_stack

    def forward(self, x):
        return self.model.forward(x)

    def set_data_manager(self, data_manager, fold_index=0):
        self.data_manager = data_manager
        self.fold_index = fold_index

    def training_step(self, train_batch, batch_idx):
        x, x_, y, _ = train_batch
        y = create_diagnosis_and_subtype_dict(y)

        if self.current_epoch < 25:
            head_i, rep_i, cluster_pred = self.forward(x[:, 0])
            head_j, _, _ = self.forward(x[:, 1])
            loss = NTXentLoss(head_i, head_j)
            self.stack = self.refresh_stack(rep_i.detach())
        elif self.current_epoch < 50:
            _, rep_i, cluster_pred = self.forward(x[:, 0])
            idx_NN1 = _cosine_simililarity(rep_i, self.stack).topk(5, dim=1).indices
            loss_list = []
            for k in range(5):
                NN1 = self.stack[idx_NN1[:, k]]
                logits1 = self.model.fc_clustering(NN1)
                loss, _, _ = self.scan_loss(cluster_pred, logits1)
                loss_list.append(loss)
            loss = sum(loss_list)
            self.stack = self.refresh_stack(rep_i.detach())
        else:
            _, _, cluster_pred_og = self.forward(x_)
            _, _, cluster_pred = self.forward(x[:, 0])
            loss = self.self_label_loss(cluster_pred_og, cluster_pred)

        # logs training loss in a dictionary
        self.log('train_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "subtypes": y['subtype'],
            "cluster_pred": cluster_pred
        }

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx):
        x, x_, y, _ = val_batch
        y = create_diagnosis_and_subtype_dict(y)

        if self.current_epoch < 25:
            head_i, _, cluster_pred = self.forward(x[:, 0])
            head_j, _, _ = self.forward(x[:, 1])
            loss = NTXentLoss(head_i, head_j)
        elif self.current_epoch < 50:
            _, rep_i, cluster_pred = self.forward(x_)
            idx_NN1 = _cosine_simililarity(rep_i, self.stack).topk(5, dim=1).indices
            loss_list = []
            for k in range(5):
                NN1 = self.stack[idx_NN1[:, k]]
                logits1 = self.model.fc_clustering(NN1)
                loss, _, _ = self.scan_loss(cluster_pred, logits1)
                loss_list.append(loss)
            loss = sum(loss_list)
        else:
            _, _, cluster_pred = self.forward(x_)
            _, _, cluster_pred_strong = self.forward(x[:, 0])
            loss = self.self_label_loss(cluster_pred, cluster_pred_strong)

        # logs training loss in a dictionary
        self.log('val_loss', loss)

        # info that are saved until epoch end
        batch_dictionary = {
            "loss": loss,
            "subtypes": y['subtype'],
            "cluster_pred": cluster_pred
        }

        return batch_dictionary

    def test_step(self, test_batch, batch_idx):
        _, x_, y, _ = test_batch
        y = create_diagnosis_and_subtype_dict(y)

        _, _, cluster_pred = self.forward(x_)

        # info that are saved until epoch end
        batch_dictionary = {
            "subtypes": y['subtype'],
            "cluster_pred": cluster_pred
        }

        return batch_dictionary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_size_scheduler, gamma=self.gamma_scheduler)
        return [optimizer], [scheduler]

    def validation_epoch_end(self, outputs):
        subtypes = torch.cat([x['subtypes'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        if self.current_epoch > 2:
            # calculating correct and total predictions
            balanced_accuracy, _ = balanced_accuracy_for_clusters(subtypes, np.argmax(y_clusters, 1))
            print("VAL / B-ACC : ", balanced_accuracy)

            self.logger.experiment.add_scalar("Balanced Accuracy/Val",
                                              balanced_accuracy,
                                              self.current_epoch)

        print("VAL /  loss : ", avg_loss)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val",
                                          avg_loss,
                                          self.current_epoch)

    def training_epoch_end(self, outputs):
        subtypes = torch.cat([x['subtypes'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correct and total predictions
        if self.current_epoch > 2:
            balanced_accuracy, _ = balanced_accuracy_for_clusters(subtypes, np.argmax(y_clusters, 1))
            print("TRAIN / B-ACC : ", balanced_accuracy)
            self.logger.experiment.add_scalar("Balanced Accuracy/Train",
                                              balanced_accuracy,
                                              self.current_epoch)

        print("TRAIN /  loss : ", avg_loss)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                          avg_loss,
                                          self.current_epoch)


    def test_epoch_end(self, outputs):
        subtypes = torch.cat([x['subtypes'] for x in outputs], dim=0).view(-1).cpu().detach().numpy()
        y_clusters = torch.cat([x['cluster_pred'] for x in outputs], dim=0).view(-1, self.n_clusters).cpu().detach().numpy()

        # calculating correct and total predictions
        balanced_accuracy, _ = balanced_accuracy_for_clusters(subtypes, np.argmax(y_clusters, 1))
        print("TEST / B-ACC : ", balanced_accuracy)

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Balanced Accuracy/Train",
                                          balanced_accuracy,
                                          self.current_epoch)

    def train_dataloader(self):
        return self.data_manager.get_dataloader(self.fold_index, shuffle=True, train=True).train

    def val_dataloader(self):
        return self.data_manager.get_dataloader(validation=True, fold_index=self.fold_index).validation

    def test_dataloader(self):
        return self.data_manager.get_dataloader(test=True, fold_index=self.fold_index).test

EPS = 1e-8
def entropy(x, input_as_probabilities):
    """
    Helper function to compute the entropy over the batch
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight=5.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold=0.99, apply_class_balancing=True):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling
        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            freq = 1 / (counts.float() / n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight=weight, reduction='mean')

        return loss

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
