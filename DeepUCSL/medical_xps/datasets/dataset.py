# -*- coding: utf-8 -*-

"""
Module that provides core functions to load dataset.
"""

import bisect
# Imports
import json
import os
from collections import OrderedDict

import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from collections import namedtuple

# Global parameters
from torchvision.transforms import transforms, RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation, \
    GaussianBlur, RandomAffine, ToTensor

SetItem = namedtuple("SetItem", ["test", "train", "validation"])
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels", "pseudo_labels"])


class DataManager(object):
    """ Data manager used to split a dataset in train, test and validation
    pytorch datasets.
    """

    def __init__(self, input_path, metadata_path, labels, number_of_folds=5, batch_size=1,
                 data_augmentation=None, self_supervision=None, device='cpu',
                 **dataloader_kwargs):
        """ Splits an input numpy array using memory-mapping into three sets:
        test, train and validation. This function can stratify the data.

        Parameters
        ----------
        input_path: str or dict
            the path to the numpy array containing the input tensor data
            that will be splited/loaded.
        metadata_path: str or dict
            the path to the metadata table in tsv format.
        number_of_folds: int, default 5
            the number of folds that will be used in the cross validation.
        batch_size: int, default 1
            the size of each mini-batch.
        data_augmentation: list of callable, default None
            transforms the training dataset input with pre-defined transformations on the fly during the training.
        self_supervision: a callable, default None
            applies a transformation to each input and generates a label
        """
        assert input_path is None or type(input_path) == type(metadata_path)

        self.train_inputs = np.load(input_path["train_val"][0], mmap_mode='r+')
        self.train_metadata_df = pd.read_csv(metadata_path["train_val"][0], index_col=None)
        self.val_inputs = np.load(input_path["train_val"][1], mmap_mode='r+')
        self.val_metadata_df = pd.read_csv(metadata_path["train_val"][1], index_col=None)

        self.test_inputs = np.load(input_path["test"], mmap_mode='r+')
        self.test_metadata_df = pd.read_csv(metadata_path["test"], index_col=None)
        self.test_metadata = np.array([{key: value for (key, value) in dict_i.items() if key in labels} for dict_i in self.test_metadata_df.to_dict('records')])

        self.labels = labels

        self.train_val_inputs = np.concatenate((self.train_inputs, self.val_inputs), axis=0)
        val_metadata = [{key: value for (key, value) in dict_i.items() if key in labels} for dict_i in self.val_metadata_df.to_dict('records')]
        train_metadata = [{key: value for (key, value) in dict_i.items() if key in labels} for dict_i in self.train_metadata_df.to_dict('records')]
        self.train_val_metadata = np.concatenate((train_metadata, val_metadata), axis=0)

        self.metadata_path = metadata_path
        self.number_of_folds = number_of_folds
        self.batch_size = batch_size
        self.self_supervision = self_supervision
        self.data_loader_kwargs = dataloader_kwargs

        if self.self_supervision in ["SimCLR", "PCL", "SupCon", "SwAV", "BYOL"]:
            self.self_supervision_transforms = get_data_augmentation(data_augmentation, to_tensor=False)
            self.data_augmentation = None
        else:
            self.self_supervision_transforms = None
            self.data_augmentation = get_data_augmentation(data_augmentation, to_tensor=True)

        dataset_cls = ArrayDataset

        self.dataset = dict((key, [])
                            for key in ("train", "test", "validation"))

        self.dataset["test"] = dataset_cls(
            self.test_inputs,
            self.test_metadata,
            self_supervision=self.self_supervision,
            self_supervision_transforms=self.self_supervision_transforms,
            device=device)

        # define train and val set for each fold (in our case, it always remains the same)
        self.gen = [(np.arange(0, len(self.train_inputs)), np.arange(len(self.train_inputs), len(self.train_val_inputs))) for fold_i in range(self.number_of_folds)]

        for fold_train_index, fold_val_index in self.gen:
            assert len(set(fold_val_index) & set(fold_train_index)) == 0, 'Validation set must be independant from val set'

            train_dataset = dataset_cls(
                self.train_val_inputs[fold_train_index],
                self.train_val_metadata[fold_train_index],
                data_augmentation=self.data_augmentation,
                self_supervision=self.self_supervision,
                self_supervision_transforms=self.self_supervision_transforms,
                device=device)
            val_dataset = dataset_cls(
                self.train_val_inputs[fold_val_index],
                self.train_val_metadata[fold_val_index],
                self_supervision=self.self_supervision,
                self_supervision_transforms=self.self_supervision_transforms,
                device=device
            )
            self.dataset["train"].append(train_dataset)
            self.dataset["validation"].append(val_dataset)

    def collate_fn(self, list_samples):
        """ After fetching a list of samples using the indices from sampler,
        the function passed as the collate_fn argument is used to collate lists
        of samples into batches.

        A custom collate_fn is used here to apply the transformations.

        See https://pytorch.org/docs/stable/data.html#dataloader-collate-fn.
        """
        data = OrderedDict()
        for key in ("inputs", "outputs", "labels", "pseudo_labels"):
            if len(list_samples) == 0 or getattr(list_samples[-1], key) is None:
                data[key] = None
            else:
                if key == "labels":
                    data[key] = {label: torch.stack(
                        [torch.as_tensor(getattr(s, key)[label], dtype=torch.float) for s in list_samples], dim=0) for
                        label in self.labels}
                else:
                    data[key] = torch.stack([torch.as_tensor(getattr(s, key), dtype=torch.float) for s in list_samples], dim=0)
        return DataItem(**data)

    def set_pseudo_labels(self, fold_index, pseudo_labels, phase):
        if phase == 'test':
            self.dataset[phase].update_pseudo_labels(pseudo_labels)
        else:
            self.dataset[phase][fold_index].update_pseudo_labels(pseudo_labels)

    def get_dataloader(self, fold_index, shuffle=True, train=False, validation=False, test=False):
        """ Generate a pytorch DataLoader.

        Parameters
        ----------
        train: bool, default False
            return the dataloader over the train set.
        validation: bool, default False
            return the dataloader over the validation set.
        test: bool, default False
            return the dataloader over the test set.

        Returns
        -------
        loaders: list of DataLoader
            the requested data loaders.
        """

        _test, _train, _validation, sampler = (None, None, None, None)
        if test:
            _test = DataLoader(
                self.dataset["test"], batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if train:
            _train = DataLoader(
                self.dataset["train"][fold_index], batch_size=self.batch_size, sampler=sampler, shuffle=shuffle, collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if validation:
            _validation = DataLoader(
                self.dataset["validation"][fold_index], batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, **self.data_loader_kwargs)
        return SetItem(test=_test, train=_train, validation=_validation)

    def save_train_val_indices(self, saving_folder):
        for fold, (fold_train_index, fold_val_index) in enumerate(self.gen):
            # save split indices to records file
            os.mkdir(saving_folder + str(fold))
            fold_indices_dict = {'train': fold_train_index.tolist(), 'val': fold_val_index.tolist()}
            with open(saving_folder + str(fold) + '/fold_train-val_indices.json', 'w') as fp:
                json.dump(fold_indices_dict, fp)


def get_data_augmentation(data_augmentation_dict, to_tensor=True):
    if to_tensor :
        augmentations_list = [ToTensor()]
    else :
        augmentations_list = []

    if data_augmentation_dict is not None :
        for (key, value) in data_augmentation_dict.items():
            if key == "random_resized_crop":
                augmentations_list.append(RandomResizedCrop(**value, antialias=True))
            if key == "horizontal_flip":
                augmentations_list.append(RandomHorizontalFlip(**value))
            if key == "gaussian_blur":
                augmentations_list.append(GaussianBlur(**value))
            if key == "color_jitter":
                augmentations_list.append(ColorJitter(saturation=0.2, contrast=0.2, brightness=0.2, hue=0.1))
            if key == "random_affine":
                augmentations_list.append(RandomAffine(**value))
            if key == "random_rotation":
                augmentations_list.append(RandomRotation(**value))
    return transforms.Compose(augmentations_list)


class ArrayDataset(Dataset):
    """ Initialize the class.
        Parameters
        ----------
        inputs: numpy array or list of numpy array
            the input data.
        self_supervision: callable, default None
            if set, the transformation to apply to each input that will generate a label
    """

    def __init__(self, inputs, metadata, pseudo_labels=None, test=False, device='cuda',
                 data_augmentation=None, self_supervision=None,
                 self_supervision_transforms=None):
        self.inputs = inputs
        self.metadata = metadata
        self.pseudo_labels = pseudo_labels
        self.device = device
        self.test = test
        self.self_supervision = self_supervision
        self.self_supervision_transform = self_supervision_transforms
        self.data_augmentation = data_augmentation

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: namedtuple
            a named tuple containing 'inputs', 'outputs', 'labels' and 'pseudo_labels' data.
        """
        _inputs = self.inputs[item]
        _labels = self.metadata[item]

        _pseudo_labels = None
        if self.pseudo_labels is not None:
            _pseudo_labels = self.pseudo_labels[item]

        # apply a preprocessing if necessary
        if len(_inputs.shape) == 2:
            _inputs = np.repeat(_inputs[:, :, None], 3, axis=2)
        _outputs = np.copy(_inputs)

        _outputs = transforms.Compose([ToTensor()])(_outputs)
        _outputs = _outputs - _outputs.min()
        _outputs = _outputs / _outputs.max()

        # Apply data augmentation
        if self.data_augmentation is None :
            _inputs = torch.tensor(_inputs).permute(2, 0, 1)
        else :
            _inputs = self.data_augmentation(_inputs)

        # Now apply the self supervision twice to have 2 versions of the input
        if self.self_supervision in ["SimCLR", "PCL", "SupCon", "SwAV", "BYOL"]:
            np.random.seed()
            _inputs_i = self.self_supervision_transform(_inputs)
            _inputs_i = _inputs_i - _inputs_i.min()
            _inputs_i = _inputs_i / _inputs_i.max()
            _inputs_j = self.self_supervision_transform(_inputs)
            _inputs_j = _inputs_j - _inputs_j.min()
            _inputs_j = _inputs_j / _inputs_j.max()
            _inputs = np.stack((_inputs_i, _inputs_j), axis=0)

        _inputs = _inputs - _inputs.min()
        _inputs = _inputs / _inputs.max()

        return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels, pseudo_labels=_pseudo_labels)

    def __len__(self):
        """ Return the length of the dataset.
        """
        return len(self.inputs)

    def update_pseudo_labels(self, pseudo_labels):
        self.pseudo_labels = pseudo_labels


def build_data_manager(args, **kwargs):
    labels = args["labels"] or []
    self_supervision = args["self_supervision"]

    input_path = {}
    for key in args["DATA_DIRS"]["input_path"].keys() :
        if key == "train_val" :
            input_path[key] = [args["ROOT"]["input_path"] + set for set in args["DATA_DIRS"]["input_path"][key]]
        else :
            input_path[key] = args["ROOT"]["input_path"] + args["DATA_DIRS"]["input_path"][key]

    metadata_path = {}
    for key in args["DATA_DIRS"]["metadata_path"].keys():
        if key == "train_val":
            metadata_path[key] = [args["ROOT"]["metadata_path"] + set for set in args["DATA_DIRS"]["metadata_path"][key]]
        else:
            metadata_path[key] = args["ROOT"]["metadata_path"] + args["DATA_DIRS"]["metadata_path"][key]

    manager = DataManager(input_path,
                          metadata_path,
                          batch_size=args["batch_size"],
                          number_of_folds=args["nb_folds"],
                          labels=labels,
                          data_augmentation=args["data_augmentation"],
                          self_supervision=self_supervision,
                          pin_memory=args["pin_mem"],
                          drop_last=args["drop_last"],
                          device=('cuda' if args["cuda"] else 'cpu'),
                          num_workers=args["num_cpu_workers"],
                          persistent_workers=args["persistent_workers"])

    return manager

class LabelMapping(object):
    def __init__(self, label_mapping_dict):
        self.label_mapping_dict = label_mapping_dict

    def __call__(self, arr):
        new_arr = np.array([self.instance_mapping(arr_i) for arr_i in arr])
        return new_arr

    def instance_mapping(self, instance):
        return self.label_mapping_dict[instance]

