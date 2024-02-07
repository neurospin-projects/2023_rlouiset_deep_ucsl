import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes=2, method_name=None):
        super(ResNet18, self).__init__()

        num_features = 512
        latent_dim = 128

        self.features = torchvision.models.resnet18(pretrained=True)
        self.features.fc = Identity()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.method_name = method_name

        if method_name in ["Classification", "ClassificationOfSubgroups"]:
            self.hidden_layer = nn.Linear(num_features, latent_dim)
            self.classifier = nn.Linear(latent_dim, num_classes)

        elif method_name in ["PCL", "DeepCluster-v2", "SimCLR", "SupCon"]:
            self.non_linear_head = nn.Sequential(nn.Linear(num_features, latent_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(latent_dim, latent_dim), )

        elif method_name in ["BYOL"]:
            self.non_linear_head = nn.Sequential(nn.Linear(num_features, latent_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(latent_dim, latent_dim), )
            self.non_linear_pred = nn.Sequential(nn.Linear(latent_dim, latent_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(latent_dim, latent_dim), )

        elif method_name in ["Deep UCSL"]:
            self.fc_conditional_classifier = nn.Linear(num_features, num_classes)
            self.fc_clustering = nn.Linear(num_features, num_classes)

        elif method_name in ["SCAN"]:
            self.non_linear_head = nn.Sequential(nn.Linear(num_features, latent_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(latent_dim, latent_dim), )
            # Clustering linear layer
            self.fc_clustering = nn.Linear(num_features, num_classes)

        elif method_name in ["SwAV"]:
            self.non_linear_head = nn.Sequential(nn.Linear(num_features, latent_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(latent_dim, latent_dim), )
            self.prototypes = nn.Linear(latent_dim, num_classes, bias=False)
            self.prototypes.weight.data = torch.nn.functional.normalize(self.prototypes.weight.data, dim=1, p=2)
            self.temperature = 0.1

    def forward(self, x):
        representation_vector = self.features(x)

        if self.method_name == "Deep UCSL":
            representation_vector = torch.flatten(representation_vector, 1)
            out_clusters = torch.softmax(self.fc_clustering(representation_vector), dim=-1)
            out_classes = torch.sigmoid(self.fc_conditional_classifier(representation_vector))
            return out_classes, out_clusters, representation_vector

        if self.method_name == "SCAN":
            representation_vector = torch.flatten(representation_vector, 1)
            head_vector = self.non_linear_head(representation_vector)
            clusters = self.fc_clustering(representation_vector)
            return nn.functional.normalize(head_vector, dim=1, p=2), representation_vector, clusters

        elif self.method_name in ["DeepCluster-v2"]:
            representation_vector = torch.flatten(representation_vector, 1)
            head_features = nn.functional.normalize(self.non_linear_head(representation_vector), dim=1, p=2)
            return head_features, representation_vector

        elif self.method_name in ["PCL"]:
            representation_vector = torch.flatten(representation_vector, 1)
            head_features = nn.functional.normalize(self.non_linear_head(representation_vector), dim=1, p=2)
            return head_features

        elif self.method_name in ["SupCon"]:
            representation_vector = torch.flatten(representation_vector, 1)
            head_features = nn.functional.normalize(self.non_linear_head(representation_vector), dim=1, p=2)
            return head_features, representation_vector

        elif self.method_name in ["SwAV"]:
            representation_vector = torch.flatten(representation_vector, 1)
            head_features = nn.functional.normalize(self.non_linear_head(representation_vector), dim=1, p=2)
            clustering_probabilities = self.softmax(self.prototypes(head_features) / self.temperature)
            return head_features, clustering_probabilities

        elif self.method_name in ["SimCLR"]:
            representation_vector = torch.flatten(representation_vector, 1)
            head_features = nn.functional.normalize(self.non_linear_head(representation_vector), dim=1, p=2)
            return head_features, representation_vector

        elif self.method_name in ["BYOL"]:
            representation_vector = torch.flatten(representation_vector, 1)
            head_features = self.non_linear_head(representation_vector)
            pred_features = nn.functional.normalize(self.non_linear_pred(head_features), dim=1, p=2)
            return nn.functional.normalize(head_features, dim=1, p=2), representation_vector, pred_features

        if self.method_name in ["Classification", "ClassificationOfSubgroups"]:
            h = self.relu(self.hidden_layer(representation_vector))
            pred = self.softmax(self.classifier(h))
            return pred, representation_vector

        else:
            return NotImplementedError


