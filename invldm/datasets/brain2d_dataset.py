import os
import gzip
import numpy as np
import torch
from . import BaseDataset


class Brain2DDataset(BaseDataset):
    def __init__(self, args):
        BaseDataset.__init__(self, args)

        # Save args in object
        self.args = args

        # Checking mode
        try:
            mode = self.args.mode
            if mode.lower() in ["mri", "vp"]:
                self.mode = mode
            else:
                raise ValueError("mode must be 'mri' or 'vp', \
                                but got '{}'".format(mode))
        except KeyError:
            self.mode = "mri"

        # Check maxsamples
        try:
            maxsamples = self.args.maxsamples
        except KeyError:
            maxsamples = None

        # Get image paths
        self.data_paths = []
        self._get_image_paths(self.args.data_path, maxsamples)
        return None

    def _get_image_paths(self, path, maxsamples=None):
        prefix = "m" if self.mode == "mri" else "vp"
        suffix = (".npy", ".npy.gz")

        # Loop through folder
        for filename in os.listdir(path):
            if filename.lower().startswith(prefix) and \
                filename.lower().endswith(suffix):
                self.data_paths.append(os.path.join(path, filename))
        self.data_paths = self.data_paths[:maxsamples]
        assert len(self.data_paths) > 0, f" Found no data samples to load in {path} with prefix {prefix} and suffixes {suffix}"
        return None
    
    def _get_condition_path(self, data_path):
        """
        Finds inside self.args.condition.path a .npy or .npy.gz file
        containing the name of the data item of data_path with suffix '-self.args.condition.mode'
        """

        name = os.path.splitext(os.path.split(data_path)[-1])[0]

        cond_path = os.path.join(self.args.condition.path, name + f"-{self.args.condition.mode}")

        if os.path.isfile(cond_path + ".npy"):
            return cond_path + ".npy"
        elif os.path.isfile(cond_path + ".npy.gz"):
            return cond_path + ".npy.gz"
        else:
            raise FileNotFoundError(f"Condition file {cond_path} does not exist in neither .npy or .npy.gz format")

    
    def _read_npy_data(self, data_path):
        # Load .npy or .npy.gz data into torch tensor
        if data_path.endswith(".npy.gz"):
            with gzip.open(data_path, 'rb') as f:
                y = torch.from_numpy(np.load(f))
        elif data_path.endswith(".npy"):
            y = torch.from_numpy(np.load(data_path))
        else:
            raise TypeError

        # Check channel dimension exists
        if len(y.shape) < 3:
            y = y.unsqueeze(0)
        return y

    def __getitem__(self, index):
        # Data item path
        y_path = self.data_paths[index]

        # Read data
        y = self._read_npy_data(y_path)

        # slowness
        if self.args.slowness:
            y = 1./ y
        
        # Apply transform
        if self.transform:
            y = self.transform(y)

        # Get condition, apply steps above
        if self.args.condition.mode is not None and self.args.condition.path is not None:
            cond_path = self._get_condition_path(y_path)
            cond = self._read_npy_data(cond_path)
            cond = cond / torch.abs(torch.max(cond))
            if self.cond_transform:
                cond = self.cond_transform(cond)
            return y, cond
        return y

    def __len__(self):
        return len(self.data_paths)

    def __str__(self):
        dic = {}
        dic["name"] = self.__class__.__name__
        dic.update(self.__dict__)
        dic.pop("data_paths")
        dic["len"] = self.__len__()
        return "{}".format(dic)

    def info(self, nsamples=None):
        nsamples = self.__len__() if nsamples is None else nsamples
        sample = self.__getitem__(0)[0]
        idx = torch.randint(0, self.__len__(), [nsamples])
        arr = torch.empty_like(sample).unsqueeze(0).repeat(nsamples, 1, 1, 1)
        for i in range(nsamples):
            arr[i] = self.__getitem__(idx[i])[0]
        stats = {"max": arr.max().item(), "min": arr.min().item(),
                 "mean": arr.mean().item(), "std": arr.std().item(),
                 "shape": sample.shape}
        return stats
