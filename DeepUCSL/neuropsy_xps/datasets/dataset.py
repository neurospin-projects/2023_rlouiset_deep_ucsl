# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that provides core functions to load and split a dataset.
"""

from collections import OrderedDict

import nibabel
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader

from pynet.augmentation.spatial import cutout
from pynet.configs.general_config import CONFIG
from pynet.transforms import *

SetItem = namedtuple("SetItem", ["test", "train", "validation"])
DataItem = namedtuple("DataItem", ["inputs", "outputs", "labels", "pseudo_labels"])


class ListTensors:
    def __init__(self, *tensor_list):
        self.list_tensors = list(tensor_list)

    def __getitem__(self, item):
        return self.list_tensors[item]

    def to(self, device, **kwargs):
        for i, e in enumerate(self.list_tensors):
            self.list_tensors[i] = e.to(device, **kwargs)
        return self.list_tensors


class DataManager(object):
    """ Data manager used to split a dataset in train, test and validation
    pytorch datasets.
    """

    def __init__(self, input_path, metadata_path, output_path=None, labels=None,
                 stratify_label=None, categorical_strat_label=True, custom_stratification=None,
                 number_of_folds=5, batch_size=1,
                 input_transforms=None, labels_transforms=None,
                 stratify_label_transforms=None, self_supervision=None,
                 train_size=0.9, dataset=None, device='cpu',
                 **dataloader_kwargs):
        """ Splits an input numpy array using memory-mapping into three sets:
        test, train and validation. This function can stratify the data.

        Parameters
        ----------
        input_path: str or list[str]
            the path to the numpy array containing the input tensor data
            that will be splited/loaded.
        metadata_path: str or list[str]
            the path to the metadata table in tsv format.
        labels: list of str, default None
            in case of classification/regression, the name of the column(s)
            in the metadata table to be predicted.
        stratify_label: str, default None
            the name of the column in the metadata table containing the label
            used during the stratification.
        categorical_strat_label: bool, default True
            is the stratification label a categorical or continuous variable ?
        custom_stratification: dict, default None
            same format as projection labels. It will split the dataset into train-val/test according
            to the stratification defined in the dict.
        projection_labels: dict, default None
            selects only the data that match the conditions in the dict
            {<column_name>: <value>}.
        number_of_folds: int, default 10
            the number of folds that will be used in the cross validation.
        batch_size: int, default 1
            the size of each mini-batch.
        input_transforms, output_transforms: list of callable, default None
            transforms a list of samples with pre-defined transformations.
        data_augmentation: list of callable, default None
            transforms the training dataset input with pre-defined transformations on the fly during the training.
        self_supervision: a callable, default None
            applies a transformation to each input and generates a label
        train_size: float, default 0.9
            should be between 0.0 and 1.0 and represent the proportion of the
            dataset to include in the train/val split.
        dataset: Dataset object, default None
            The Dataset used to create the DataLoader. It must be a subclass of <ArrayDataset>
        """
        self.train_size = train_size

        assert input_path is None or type(input_path) == type(metadata_path)
        if output_path is not None:
            assert input_path is None or type(output_path) == type(input_path)

        input_path = [input_path] if type(input_path) == str else input_path
        metadata_path = [metadata_path] if type(metadata_path) == str else metadata_path
        output_path = [output_path] if output_path is not None else None

        assert input_path is None or len(input_path) == len(metadata_path)
        self.logger = logging.getLogger()

        all_df = []
        for p in metadata_path:
            all_df.append(pd.read_csv(p, sep=","))
        df = pd.concat(all_df, ignore_index=True, sort=False)

        try :
            if "site" in labels:
                stratify_label_transforms[labels.index("site")] = LabelMapping(**{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))})
                labels_transforms["site"] = LabelMapping(**{site: indice for (indice, site) in enumerate(sorted(set(df['site'])))})
        except:
            print("site already indexed")

        if input_path is not None:
            for (i, m) in zip(input_path, metadata_path):
                self.logger.info('Correspondance {data} <==> {meta}'.format(data=i, meta=m))

            if input_path[0][-4:] == '.npy':
                self.inputs = [np.load(p, mmap_mode='r+').astype(np.float32) for p in input_path]
                if len(self.inputs) == 1 :
                    self.inputs = self.inputs[0]
                else:
                    # TODO: get rid of
                    for i, input in enumerate(self.inputs):
                        if len(input.shape) == 6 :
                            self.inputs[i] = input[:, 0]
                    self.inputs = np.concatenate(self.inputs, axis=0)
                print("Input images gathered.")
            else:
                self.inputs = [pd.read_csv(p, sep=',') for p in input_path]

        else:
            self.inputs = None

        if output_path is not None:
            self.outputs = [np.load(p, mmap_mode='r') for p in output_path]

        self.metadata_df = df
        self.targets = labels

        mask = DataManager.get_mask(
            df=df,
            check_nan=labels)

        mask_indices = DataManager.get_indices_from_mask(mask)

        # We should only work with masked data but we want to preserve the memory mapping so we are getting the right
        # index at the end (in __getitem__ of ArrayDataset)

        self.outputs, self.labels, self.stratify_label = (None, None, None)

        if labels is not None:
            assert np.all(~df[labels][mask].isna())
            self.labels = np.array([{key: value for (key, value) in dict_i.items() if key in labels} for dict_i in df[labels].to_dict('records')])

        if stratify_label is not None:
            self.stratify_label = df[stratify_label].values.copy()
            # Apply the labels transform here as a mapping to the integer representation of the classes
            for i in mask_indices:
                label_i = self.stratify_label[i]
                for j, tf_j in enumerate(stratify_label_transforms or []):
                    label_i[j] = tf_j(label_i[j])
                self.stratify_label[i] = label_i
            # If necessary, discretize the labels
            for j, categorical in enumerate(categorical_strat_label):
                if self.stratify_label[j] is not None and not categorical:
                    self.stratify_label[mask][:, j] = DataManager.discretize_continous_label(self.stratify_label[mask][:, j], verbose=False)

        self.metadata_path = metadata_path
        self.number_of_folds = number_of_folds
        self.batch_size = batch_size
        self.input_transforms = input_transforms or []
        self.labels_transforms = labels_transforms or []
        self.self_supervision = self_supervision
        self.data_loader_kwargs = dataloader_kwargs

        if self.self_supervision in ["SimCLR", "PCL", "DeepCluster-v2", "BYOL", "SwAV"]:
            self.self_supervision_transforms = Transformer()
            self.self_supervision_transforms.register(flip, probability=0.5)
            self.self_supervision_transforms.register(add_blur, probability=0.5, sigma=(0.1, 1))
            self.self_supervision_transforms.register(add_noise, sigma=(0.1, 1), probability=0.5)
            self.self_supervision_transforms.register(cutout, probability=0.5, patch_size=32, inplace=False)
            self.self_supervision_transforms.register(Crop((96, 96, 96), "random", resize=True), probability=0.5)
        # TODO: check which is better
        elif self.self_supervision in ["Deep UCSL"]:
            self.self_supervision_transforms = Transformer()
            # self.self_supervision_transforms.register(flip, probability=0.5)
            self.self_supervision_transforms.register(add_blur, probability=0.5, sigma=(0.1, 1))
            # self.self_supervision_transforms.register(add_noise, sigma=(0.1, 1), probability=0.5)
            self.self_supervision_transforms.register(cutout, probability=0.5, patch_size=32, inplace=False)
            self.self_supervision_transforms.register(Crop((96, 96, 96), "random", resize=True), probability=0.5)
        else:
            self.self_supervision_transforms = None

        """elif self.self_supervision in ["Deep UCSL"]:
            self.self_supervision_transforms = Transformer()
            self.self_supervision_transforms.register(add_blur, probability=0.5, sigma=(0.1, 1))
            self.self_supervision_transforms.register(cutout, probability=0.5, patch_size=32, inplace=False)
            self.self_supervision_transforms.register(Crop((96, 96, 96), "random", resize=True), probability=0.5)"""

        dataset_cls = ArrayDataset if dataset is None else dataset
        assert issubclass(dataset_cls, ArrayDataset)

        self.dataset = dict((key, [])
                            for key in ("train", "test", "validation"))

        # 1st step: split into train/test (get only indices)
        train_mask, test_mask = (DataManager.get_mask(df, custom_stratification["train"]), DataManager.get_mask(df, custom_stratification["test"]))
        train_mask &= mask
        test_mask &= mask
        train_indices = DataManager.get_indices_from_mask(train_mask)
        test_indices = DataManager.get_indices_from_mask(test_mask)

        if train_indices is None:
            return

        assert len(set(train_indices) & set(test_indices)) == 0, 'Test set must be independent from train set'

        self.dataset["test"] = dataset_cls(
            self.inputs, test_indices,
            labels=self.labels,
            input_transforms=self.input_transforms,
            label_transforms=self.labels_transforms,
            self_supervision=self.self_supervision,
            self_supervision_transforms=self.self_supervision_transforms,
            device=device)

        dummy_like_X_train = np.ones(len(train_indices))

        kfold_splitter = StratifiedShuffleSplit(n_splits=self.number_of_folds,
                            train_size=float(self.train_size * len(train_indices) / len(train_indices)), random_state=0)
        strat_indices = np.array(self.stratify_label[train_indices], dtype=np.int32) if stratify_label is not None else None
        gen = kfold_splitter.split(dummy_like_X_train, strat_indices)
        gen = [(train_indices[tr], train_indices[val]) for (tr, val) in gen]

        for fold_train_index, fold_val_index in gen:
            assert len(set(fold_val_index) & set(fold_train_index)) == 0, \
                'Validation set must be independant from test set'

            train_dataset = dataset_cls(
                self.inputs, fold_train_index,
                labels=self.labels,
                outputs=self.outputs,
                input_transforms=self.input_transforms,
                label_transforms=self.labels_transforms,
                self_supervision=self.self_supervision,
                self_supervision_transforms=self.self_supervision_transforms,
                device=device)
            val_dataset = dataset_cls(
                self.inputs, fold_val_index,
                labels=self.labels,
                outputs=self.outputs,
                input_transforms=self.input_transforms,
                label_transforms=self.labels_transforms,
                self_supervision=self.self_supervision,
                self_supervision_transforms=self.self_supervision_transforms,
                device=device
            )
            self.dataset["train"].append(train_dataset)
            self.dataset["validation"].append(val_dataset)

    @staticmethod
    def get_indices_from_mask(mask):
        return np.arange(len(mask))[mask]

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: Dataset or list of Dataset
            the requested set of data: test, train or validation.
        """
        if item not in ("train", "test", "validation"):
            raise ValueError("Unknown set! Must be 'train', 'test' or "
                             "'validation'.")
        return self.dataset[item]

    @staticmethod
    def get_labels(df, labels, train_indices):
        train_labels = []
        for idx in train_indices:
            train_labels.append(df[labels[0]].iloc[idx])
        return np.array(train_labels)

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
                    data[key] = {}
                    for label in self.targets:
                        data[key][label] = torch.stack([torch.as_tensor(getattr(s, key)[label], dtype=torch.float) for s in list_samples], dim=0)

                else:
                    data[key] = torch.stack([torch.as_tensor(getattr(s, key), dtype=torch.float) for s in list_samples], dim=0)

        return DataItem(**data)

    def set_pseudo_labels(self, fold_index, pseudo_labels, phase, n_clusters):
        if phase == 'test':
            self.dataset[phase].update_pseudo_labels(pseudo_labels, n_clusters)
        else:
            self.dataset[phase][fold_index].update_pseudo_labels(pseudo_labels, n_clusters)
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
                self.dataset["test"], batch_size=self.batch_size,
                collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if train:
            _train = DataLoader(
                self.dataset["train"][fold_index], batch_size=self.batch_size, sampler=sampler, shuffle=shuffle,
                collate_fn=self.collate_fn, **self.data_loader_kwargs)
        if validation:
            _validation = DataLoader(
                self.dataset["validation"][fold_index],
                batch_size=self.batch_size, collate_fn=self.collate_fn,
                **self.data_loader_kwargs)
        return SetItem(test=_test, train=_train, validation=_validation)

    @staticmethod
    def get_mask(df, projection_labels=None, check_nan=None):
        """ Filter a table.

        Parameters
        ----------
        df: a pandas DataFrame
            a table data.
        projection_labels: dict, default None
            selects only the data that match the conditions in the dict
            {<column_name>: <value>}.
        check_nan: list of str, default None
            check if there is nan in the selected columns. Select only the rows without nan
        Returns
        -------
        mask: a list of boolean values
        """

        mask = np.ones(len(df), dtype=np.bool)
        if projection_labels is not None:
            for (col, val) in projection_labels.items():
                if isinstance(val, list):
                    mask &= getattr(df, col).isin(val)
                elif val is not None:
                    mask &= getattr(df, col).eq(val)
        if check_nan is not None:
            for col in check_nan:
                mask &= ~getattr(df, col).isna()
        return mask

class ArrayDataset(Dataset):
    """ A dataset based on numpy array.
    """

    def __init__(self, inputs, indices, labels=None, pseudo_labels=None, outputs=None,
                 in_features_transforms=None,
                 input_transforms=None,
                 label_transforms=None, self_supervision=None, self_supervision_transforms=None,
                 input_size=None, device='cpu'):
        """ Initialize the class.

        Parameters
        ----------
        inputs: numpy array or list of numpy array
            the input data.
        indices: iterable of int
            the list of indices that is considered in this dataset.
        labels: DataFrame or numpy array
        outputs: numpy array or list of numpy array
            the output data.
        self_supervision: callable, default None
            if set, the transformation to apply to each input that will generate a label
        concat_datasets: bool, default False
            whether to consider a list of inputs/outputs as a list of multiple datasets or a unique dataset
        """
        self.inputs = inputs
        self.labels = labels
        self.pseudo_labels = pseudo_labels
        self.outputs = outputs
        self.device = device
        self.indices = indices
        # self.concat_datasets = concat_datasets
        self.input_size = input_size
        self.in_features_transforms = in_features_transforms or []
        self.input_transforms = input_transforms or []
        self.labels_transforms = label_transforms or []
        self.self_supervision = self_supervision
        self.self_supervision_transform = self_supervision_transforms

        if self.outputs is not None and self.inputs is not None:
            assert len(self.inputs) == len(self.outputs)

    def __getitem__(self, item):
        """ Return the requested item.

        Returns
        -------
        item: namedtuple
            a named tuple containing 'inputs', 'outputs', 'labels' and 'pseudo_labels' data.
        """
        idx = self.indices[item]
        _outputs = None
        _inputs = self.inputs[idx]
        if self.outputs is not None:
            _outputs = self.outputs[idx]

        _labels = None
        if self.labels is not None:  # Particular case in which we can deal with strings before transformations...
            _labels = self.labels[idx]

        _pseudo_labels = None
        if self.pseudo_labels is not None:
            _pseudo_labels = self.pseudo_labels[idx]

        # Apply the transformations to the data
        for tf in self.input_transforms:
            _inputs = tf(_inputs)
        if _labels is not None:
            for (label_name, label_mapping) in self.labels_transforms.items():
                _labels[label_name] = label_mapping(_labels[label_name])

        # Now apply the self supervision twice to have 2 versions of the input
        if self.self_supervision in ["SimCLR", "PCL", "BYOL", "SwAV"]:
            _outputs = np.copy(_inputs)
            _inputs_i = self.self_supervision_transform(_inputs)
            _inputs_j = self.self_supervision_transform(_inputs)
            _inputs = np.stack((_inputs_i, _inputs_j), axis=0)
        elif self.self_supervision in ["DeepCluster-v2", "Deep UCSL"]:
            _outputs = np.copy(_inputs)
            _inputs = self.self_supervision_transform(_inputs)
        else:
            _outputs = np.copy(_inputs)

        return DataItem(inputs=_inputs, outputs=_outputs, labels=_labels, pseudo_labels=_pseudo_labels)

    def __len__(self):
        """ Return the length of the dataset.
        """
        return len(self.indices)

    def update_pseudo_labels(self, pseudo_labels, n_clusters):
        self.pseudo_labels = (1 / n_clusters) * np.ones((self.labels.shape[0], n_clusters)) # set uniform probability
        self.pseudo_labels[np.array(self.indices)] = pseudo_labels

def build_data_manager(args, **kwargs):
    labels = args["labels"] or []
    self_supervision = args["self_supervision"]
    input_transforms = kwargs.get('input_transforms')
    labels_mapping = args["labels_mapping"]

    if args["db"] is None:
        stratif = None
    else:
        stratif = CONFIG['db'][args["db"]]

    # Set the preprocessing step with an exception for GAN
    if input_transforms is None:
        if args["preproc"] == 'cat12':  # Size [121 x 145 x 121], 1.5mm3
            input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), Normalize()]
        elif args["preproc"] == 'cat12_no_normalization':  # Size [121 x 145 x 121], 1.5mm3
            input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant')]
        elif args["preproc"] == 'quasi_raw':  # Size [122 x 146 x 122], 1.5mm
            input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), Normalize()]
        elif args["preproc"] == 'cat12_undersample':  # Size [122 x 146 x 122], 1.5mm
            input_transforms = [Crop((1, 121, 128, 121)), Padding([1, 128, 128, 128], mode='constant'), Resize([1, 64, 64, 64]), Normalize()]


    # <label>: [LabelMapping(), IsCategorical]
    known_labels = {'age': [LabelMapping(), False],
                    'sex': [LabelMapping(**labels_mapping['sex']) if (labels_mapping is not None and 'sex' in labels_mapping) else LabelMapping(), True], # [LabelMapping(**{"female":0, "male":1}), True]
                    'site': [LabelMapping(), True],
                    'diagnosis': [LabelMapping(**labels_mapping['diagnosis']) if (labels_mapping is not None and 'diagnosis' in labels_mapping) else LabelMapping(), True],
                    }

    assert set(labels) <= set(known_labels.keys()), \
        "Unknown label(s), chose from {}".format(set(known_labels.keys()))

    # assert (args["stratify_label"] is None), "Unknown stratification label, chose from {}".format(set(known_labels.keys()))
    stratif_labels = args["stratify_label"]
    valid_stratify_label = (stratif_labels is None) or all(item in set(known_labels.keys()) for item in stratif_labels)
    assert valid_stratify_label, "Unknown stratification label, chose from {}".format(set(known_labels.keys()))

    # categorical_strat_label = known_labels[args["stratify_label"]][1] if args["stratify_label"] is not None else None
    if len(stratif_labels) == 0:
        strat_label_transforms = None
        categorical_strat_label = None
    elif len(stratif_labels) == 1:
        strat_label_transforms = [known_labels[stratif_labels[0]][0]]
        categorical_strat_label = [known_labels[stratif_labels[0]][1]]
    else:
        strat_label_transforms = [known_labels[stratif_labels[i]][0] for i, l in enumerate(stratif_labels)]
        categorical_strat_label = [known_labels[stratif_labels[i]][1] for i, l in enumerate(stratif_labels)]

    if len(labels) == 0:
        labels_transforms = None
    elif len(labels) == 1:
        labels_transforms = {l: known_labels[l][0] for i, l in enumerate(labels)} # [known_labels[labels[0]][0]]
    else:
        labels_transforms = {l: known_labels[l][0] for i, l in enumerate(labels)}

    dataset_cls = None

    manager = DataManager([args["ROOT"]["input_path"] + p for p in args["DATA_DIRS"]["input_path"]],
                          [args["ROOT"]["metadata_path"] + p for p in args["DATA_DIRS"]["metadata_path"]],
                          batch_size=args["batch_size"],
                          number_of_folds=args["nb_folds"],
                          labels=labels or None,
                          custom_stratification=stratif,
                          categorical_strat_label=categorical_strat_label,
                          stratify_label=args["stratify_label"],
                          train_size=args["train_size"],
                          input_transforms=input_transforms,
                          stratify_label_transforms=strat_label_transforms,
                          labels_transforms=labels_transforms,
                          self_supervision=self_supervision,
                          pin_memory=args["pin_mem"],
                          drop_last=args["drop_last"],
                          dataset=dataset_cls,
                          device=('cuda' if args["cuda"] else 'cpu'),
                          num_workers=args["num_cpu_workers"],
                          persistent_workers=args["persistent_workers"])

    return manager
