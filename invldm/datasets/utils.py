import logging
import importlib

import torch
import numpy as np

from torch.utils.data import DataLoader, Subset, TensorDataset

from .base_dataset import BaseDataset


def _instance_dataset(args):
    # mod = __import__(f'datasets.{args.dataset.lower()}_dataset',
    #                  fromlist=[args.dataset])
    
    mod = importlib.import_module(f'.{args.dataset.lower()}_dataset', "invldm.datasets")
    cls_name = args.dataset + "Dataset"
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        logging.warn(f'Expecting {args.dataset.lower()}_dataset.py to'
                     f'contain a dataset class named {cls_name} derived'
                     f'from the BaseDataset class')
        raise (AttributeError, e)
    assert (issubclass(cls, BaseDataset))
    dataset = cls(args)
    return dataset


def _instance_dataloader(args, dataset, shuffle=True):
    if dataset and len(dataset) > 0:
        dataloader = DataLoader(dataset,
                                batch_size=args.worker_batch_size,
                                shuffle=shuffle,
                                # num_workers=args.num_workers
                    )
        return dataloader
    else:
        return None


def _wrap_tensor_dataset(dataset):
    return TensorDataset(dataset, torch.zeros(dataset.shape[0]))


def _split_valid_dataset(args, dataset):
    if not args.validation.split:
        args.validation.split = 0.
    n_items = len(dataset)
    r_split = max(0., min(args.validation.split, 0.9))  # cap
    indices = list(range(n_items))
    random_state = np.random.get_state()
    np.random.seed(2023)
    np.random.shuffle(indices)
    np.random.set_state(random_state)
    train_indices, valid_indices = (
        indices[int(n_items * r_split):],
        indices[:int(n_items * r_split)],
    )
    train_dataset = Subset(dataset, train_indices)
    valid_dataset = Subset(dataset, valid_indices)
    return train_dataset, valid_dataset
