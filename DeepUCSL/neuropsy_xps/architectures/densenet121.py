import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# from torchvision.models.utils import load_state_dict_from_url


__all__ = ['DenseNet', '_densenet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, bayesian=False,
                 memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),

        self.drop_rate = drop_rate
        self.bayesian = bayesian
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)

        if hasattr(self, 'concrete_dropout'):
            new_features = self.concrete_dropout(self.relu2(self.norm2(bottleneck_output)))
        else:
            new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))

            if self.drop_rate > 0:
                new_features = F.dropout(new_features, p=self.drop_rate,
                                         training=(self.training or self.bayesian))

        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, bayesian=False,
                 memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                bayesian=bayesian,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, num_classes, method_name, growth_rate=32, block_config=(3, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, in_channels=1,
                 bayesian=False, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_channels, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                bayesian=bayesian,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.num_features = num_features
        self.method_name = method_name

        if method_name in ["Classification", "ClassificationOfSubgroups"]:
            self.fc_latent = nn.Linear(num_features, 512)
            self.bn_latent = nn.BatchNorm1d(512)
            self.head_projection = nn.Linear(512, 128)
            self.classifier = nn.Linear(128, num_classes)

        elif method_name in ["PCL", "DeepCluster-v2", "SimCLR", "SupCon"]:
            self.fc_latent = nn.Linear(num_features, 512)
            self.bn_latent = nn.BatchNorm1d(512)
            self.head_projection = nn.Linear(512, 128)
            self.classifier = nn.Linear(128, num_classes)

        elif method_name in ["BYOL"]:
            self.fc_latent = nn.Linear(num_features, 512)
            self.bn_latent = nn.BatchNorm1d(512)
            self.head_projection = nn.Linear(512, 128)
            self.non_linear_pred = nn.Sequential(nn.Linear(128, 128),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(128, 128), )

        elif method_name in ["SwAV"]:
            self.fc_latent = nn.Linear(num_features, 512)
            self.bn_latent = nn.BatchNorm1d(512)
            self.head_projection = nn.Linear(512, 128)
            self.prototypes = nn.Linear(128, num_classes, bias=False)
            self.prototypes.weight.data = torch.nn.functional.normalize(self.prototypes.weight.data, dim=1, p=2)
            self.temperature = 0.1

        if method_name in ["SCAN"]:
            self.fc_latent = nn.Linear(num_features, 512)
            self.bn_latent = nn.BatchNorm1d(512)
            self.head_projection = nn.Linear(512, 128)

        elif method_name in ["Deep UCSL"]:
            self.fc_latent = nn.Linear(num_features, 512)
            self.bn_latent = nn.BatchNorm1d(512)
            self.head_projection = nn.Linear(512, 128)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and method_name not in ["SwAV"]:
                nn.init.constant_(m.bias, 0)

        if method_name in ["SCAN"]:
            self.fc_clustering = nn.Linear(1024, num_classes)
        elif method_name in ["Deep UCSL"]:
            # Classifier and clustering linear layer
            self.fc_conditional_classifier = nn.Linear(128, num_classes)
            self.fc_clustering = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.features(x)
        representation_vector = F.relu(features, inplace=True)
        representation_vector = F.adaptive_avg_pool3d(representation_vector, 1)
        representation_vector = torch.flatten(representation_vector, 1)

        if self.method_name in ["Deep UCSL"]:
            # compute latent vectors
            latent_vector = F.relu(self.fc_latent(representation_vector), inplace=True)
            head_features = self.head_projection(latent_vector)
            # predict p(c|x) and p(y|c, x)
            out_clusters = torch.softmax(self.fc_clustering(head_features), dim=-1)
            out_classes = torch.sigmoid(self.fc_conditional_classifier(head_features))
            return out_classes, out_clusters, head_features

        elif self.method_name in ["DeepCluster-v2"]:
            # compute latent vectors
            latent_vector = F.relu(self.fc_latent(representation_vector), inplace=True)
            head_features = nn.functional.normalize(self.head_projection(latent_vector), dim=1, p=2)
            return head_features, representation_vector

        elif self.method_name in ["PCL"]:
            # compute latent vectors
            latent_vector = F.relu(self.fc_latent(representation_vector), inplace=True)
            head_features = nn.functional.normalize(self.head_projection(latent_vector), dim=1, p=2)
            return head_features

        elif self.method_name in ["SimCLR", "SupCon"]:
            # compute latent vectors
            latent_vector = F.relu(self.fc_latent(representation_vector), inplace=True)
            head_features = nn.functional.normalize(self.head_projection(latent_vector), dim=1, p=2)
            return head_features, representation_vector

        elif self.method_name in ["BYOL"]:
            # compute latent vectors
            latent_vector = F.relu(self.fc_latent(representation_vector), inplace=True)
            head_features = nn.functional.normalize(self.head_projection(latent_vector), dim=1, p=2)
            pred_features = nn.functional.normalize(self.non_linear_pred(head_features), dim=1, p=2)
            return head_features, representation_vector, pred_features

        elif self.method_name in ["SwAV"]:
            # compute latent vectors
            latent_vector = F.relu(self.fc_latent(representation_vector), inplace=True)
            head_features = nn.functional.normalize(self.head_projection(latent_vector), dim=1, p=2)
            # predict p(c|x)
            clustering_probabilities = torch.softmax(self.prototypes(head_features) / self.temperature, dim=-1)
            return head_features, clustering_probabilities

        elif self.method_name in ["SCAN"]:
            # compute latent vectors
            latent_vector = F.relu(self.fc_latent(representation_vector), inplace=True)
            head_features = self.head_projection(latent_vector)
            # predict p(c|x)
            clusters = self.fc_clustering(representation_vector)
            return nn.functional.normalize(head_features, dim=1, p=2), representation_vector, clusters


        elif self.method_name in ["Classification", "ClassificationOfSubgroups"]:
            # compute latent vectors
            latent_vector = F.relu(self.fc_latent(representation_vector), inplace=True)
            head_features = F.relu(self.head_projection(latent_vector), inplace=True)
            out_classes = self.classifier(head_features)
            return out_classes.squeeze(dim=1), head_features

        else:
            return NotImplementedError

def _densenet(num_classes, method_name, growth_rate, block_config, num_init_features):
    model = DenseNet(num_classes, method_name, growth_rate, block_config, num_init_features)
    return model


def densenet121(num_classes, method_name):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return _densenet(num_classes, method_name, 32, (6, 12, 24, 16), 64)


def tiny_densenet121(num_classes, method_name):
    r"""Tiny-Densenet-121 model from
     "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>
    The tiny-version has been specifically designed for neuroimaging data. It is 10X smaller than DenseNet.
    Args:
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return DenseNet(num_classes, method_name, 16, (6, 12, 16), 64)
