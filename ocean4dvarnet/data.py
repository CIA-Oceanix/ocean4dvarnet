"""
This module provides data handling utilities for machine learning tasks using PyTorch and xarray.
It includes classes for creating datasets, augmenting data, and managing data loading pipelines.
"""

import itertools
import functools as ft
from collections import namedtuple
import pytorch_lightning as pl
import numpy as np
import torch.utils.data
import xarray as xr

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])


class IncompleteScanConfiguration(Exception):
    """
    Exception raised when the scan configuration does not cover the entire domain.
    """
    # pass


class DangerousDimOrdering(Exception):
    """
    Exception raised when the dimension ordering of the input data is incorrect.
    """
    # pass


class XrDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset based on an xarray.DataArray with on-the-fly slicing.

    This class allows efficient extraction of patches from an xarray.DataArray
    for training machine learning models.

    ### Usage: ####
    If you want to be able to reconstruct the input

    the input xr.DataArray should:
        - have coordinates
        - have the last dims correspond to the patch dims in same order
        - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)

    the batches passed to self.reconstruct should:
        - have the last dims correspond to the patch dims in same order

    Attributes:
        da (xarray.DataArray): The input data array.
        patch_dims (dict): Dimensions and sizes of patches to extract.
        domain_limits (dict): Limits for selecting a subset of the domain.
        strides (dict): Strides for patch extraction.
        check_full_scan (bool): Whether to check if the entire domain is scanned.
        check_dim_order (bool): Whether to check the dimension ordering.
        postpro_fn (callable): A function for post-processing extracted patches.
    """

    def __init__(
            self, da, patch_dims, domain_limits=None, strides=None,
            check_full_scan=False, check_dim_order=False,
            postpro_fn=None
    ):
        """
        Initialize the XrDataset.

        Args:
            da (xarray.DataArray): Input data, with patch dims at the end in the dim orders
            patch_dims (dict):  da dimension and sizes of patches to extract.
            domain_limits (dict, optional): da dimension slices of domain, to Limits for selecting
                                            a subset of the domain. for patch extractions
            strides (dict, optional): dims to strides size for patch extraction.(default to one)
            check_full_scan (bool, optional): if True raise an error if the whole domain is not scanned by the patch.
            check_dim_order (bool, optional): Whether to check the dimension ordering.
            postpro_fn (callable, optional): A function for post-processing extracted patches.
        """
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.da = da.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        da_dims = dict(zip(self.da.dims, self.da.shape))
        self.ds_size = {
            dim: max((da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in patch_dims
        }

        if check_full_scan:
            for dim in patch_dims:
                if (da_dims[dim] - self.patch_dims[dim]) % self.strides.get(dim, 1) != 0:
                    raise IncompleteScanConfiguration(
                        f"""
                        Incomplete scan in dimension dim {dim}:
                        dataarray shape on this dim {da_dims[dim]}
                        patch_size along this dim {self.patch_dims[dim]}
                        stride along this dim {self.strides.get(dim, 1)}
                        [shape - patch_size] should be divisible by stride
                        """
                    )

        if check_dim_order:
            for dim in patch_dims:
                if not '#'.join(da.dims).endswith('#'.join(list(patch_dims))):
                    raise DangerousDimOrdering(
                        f"""
                        input dataarray's dims should end with patch_dims
                        dataarray's dim {da.dims}:
                        patch_dims {list(patch_dims)}
                        """
                    )

    def __len__(self):
        """
        Return the total number of patches in the dataset.

        Returns:
            int: Number of patches.
        """
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        """
        Iterate over the dataset.

        Yields:
            Patch data for each index.
        """
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        """
        Get the coordinates of all patches in the dataset.

        Returns:
            list: List of coordinates for each patch.
        """
        self.return_coords = True
        coords = []
        try:
            for _, item in enumerate(self):
                coords.append(item)
        finally:
            self.return_coords = False
        return coords

    def __getitem__(self, item):
        """
        Get a specific patch by index.

        Args:
            item (int): Index of the patch.

        Returns:
            Patch data or coordinates, depending on the mode.
        """
        sl = {
            dim: slice(self.strides.get(dim, 1) * idx,
                       self.strides.get(dim, 1) * idx + self.patch_dims[dim])
            for dim, idx in zip(self.ds_size.keys(),
                                np.unravel_index(item, tuple(self.ds_size.values())))
        }
        item = self.da.isel(**sl)

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item

    def reconstruct(self, batches, weight=None):
        """
        Reconstruct the original data array from patches.
        Takes as input a list of np.ndarray of dimensions (b, \\*, \\*patch_dims)

        Args:
            batches (list): List of patches (torch tensor) corresponding to batches without shuffle.
            weight (np.ndarray, optional): tensor of size patch_dims corresponding to the weight of a prediction
                depending on the position on the patch (default to ones everywhere) overlapping patches will
                be averaged with weighting

        Returns:
            xarray.DataArray: Reconstructed data array. A stitched xarray.DataArray with the coords of patch_dims
        """
        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)

    def reconstruct_from_items(self, items, weight=None):
        """
        Reconstruct the original data array from individual items.

        Args:
            items (list): List of individual patches.
            weight (np.ndarray, optional): Weighting for overlapping patches.

        Returns:
            xarray.DataArray: Reconstructed data array.
        """
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
        w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))

        coords = self.get_coords()

        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + list(coords[0].dims)

        das = [xr.DataArray(it.numpy(), dims=dims, coords=co.coords)
               for it, co in zip(items, coords)]

        da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        rec_da = xr.DataArray(
            np.zeros([*new_shape.values(), *da_shape.values()]),
            dims=dims,
            coords={d: self.da[d] for d in self.patch_dims}
        )
        count_da = xr.zeros_like(rec_da)

        for da in das:
            rec_da.loc[da.coords] = rec_da.sel(da.coords) + da * w
            count_da.loc[da.coords] = count_da.sel(da.coords) + w

        return rec_da / count_da


class XrConcatDataset(torch.utils.data.ConcatDataset):
    """
    A concatenation of multiple XrDatasets.

    This class allows combining multiple datasets into one for training or evaluation.
    """

    def reconstruct(self, batches, weight=None):
        """
        Reconstruct the original data arrays from batches.

        Args:
            batches (list): List of batches.
            weight (np.ndarray, optional): Weighting for overlapping patches.

        Returns:
            list: List of reconstructed xarray.DataArray objects.
        """
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.reconstruct_from_items(ds_items, weight))

        return rec_das


class AugmentedDataset(torch.utils.data.Dataset):
    """
    A dataset that applies data augmentation to an input dataset.

    Attributes:
        inp_ds (torch.utils.data.Dataset): The input dataset.
        aug_factor (int): The number of augmented copies to generate.
        aug_only (bool): Whether to include only augmented data.
        noise_sigma (float): Standard deviation of noise to add to augmented data.
    """

    def __init__(self, inp_ds, aug_factor, aug_only=False, noise_sigma=None):
        """
        Initialize the AugmentedDataset.

        Args:
            inp_ds (torch.utils.data.Dataset): The input dataset.
            aug_factor (int): The number of augmented copies to generate.
            aug_only (bool, optional): Whether to include only augmented data.
            noise_sigma (float, optional): Standard deviation of noise to add to augmented data.
        """
        self.aug_factor = aug_factor
        self.aug_only = aug_only
        self.inp_ds = inp_ds
        self.perm = np.random.permutation(len(self.inp_ds))
        self.noise_sigma = noise_sigma

    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns:
            int: Total number of items.
        """
        return len(self.inp_ds) * (1 + self.aug_factor - int(self.aug_only))

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            TrainingItem: The requested item.
        """
        if self.aug_only:
            idx = idx + len(self.inp_ds)

        if idx < len(self.inp_ds):
            return self.inp_ds[idx]

        tgt_idx = idx % len(self.inp_ds)
        perm_idx = tgt_idx
        for _ in range(idx // len(self.inp_ds)):
            perm_idx = self.perm[perm_idx]

        item = self.inp_ds[tgt_idx]
        perm_item = self.inp_ds[perm_idx]

        noise = np.zeros_like(item.input, dtype=np.float32)
        if self.noise_sigma is not None:
            noise = np.random.randn(*item.input.shape).astype(np.float32) * self.noise_sigma

        return item._replace(input=noise + np.where(np.isfinite(perm_item.input),
                             item.tgt, np.full_like(item.tgt, np.nan)))


class BaseDataModule(pl.LightningDataModule):
    """
    A base data module for managing datasets and data loaders in PyTorch Lightning.

    Attributes:
        input_da (xarray.DataArray): The input data array.
        domains (dict): Dictionary of domain splits (train, val, test).
        xrds_kw (dict): Keyword arguments for XrDataset.
        dl_kw (dict): Keyword arguments for DataLoader.
        aug_kw (dict): Keyword arguments for AugmentedDataset.
        norm_stats (tuple): Normalization statistics (mean, std).
    """

    def __init__(self, input_da, domains, xrds_kw, dl_kw, aug_kw=None, norm_stats=None, **kwargs):
        """
        Initialize the BaseDataModule.

        Args:
            input_da (xarray.DataArray): The input data array.
            domains (dict): Dictionary of domain splits (train, val, test).
            xrds_kw (dict): Keyword arguments for XrDataset.
            dl_kw (dict): Keyword arguments for DataLoader.
            aug_kw (dict, optional): Keyword arguments for AugmentedDataset.
            norm_stats (tuple, optional): Normalization statistics (mean, std).
        """
        super().__init__()
        self.input_da = input_da
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_kw = aug_kw if aug_kw is not None else {}
        self._norm_stats = norm_stats

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self._post_fn = None

    def norm_stats(self):
        """
        Compute or retrieve normalization statistics (mean, std).

        Returns:
            tuple: Normalization statistics (mean, std).
        """
        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
            print("Norm stats", self._norm_stats)
        return self._norm_stats

    def train_mean_std(self, variable='tgt'):
        """
        Compute the mean and standard deviation of the training data.

        Args:
            variable (str, optional): Variable to compute statistics for.

        Returns:
            tuple: Mean and standard deviation.
        """
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {})).sel(self.domains['train'])
        return train_data.sel(variable=variable).pipe(lambda da: (da.mean().values.item(), da.std().values.item()))

    def post_fn(self):
        """
        Create a post-processing function for normalizing data.

        Returns:
            callable: Post-processing function.
        """
        m, s = self.norm_stats()
        def normalize(item): return (item - m) / s
        return ft.partial(ft.reduce, lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(tgt=normalize(item.tgt)),
            lambda item: item._replace(input=normalize(item.input)),
        ])

    def setup(self, stage='test'):
        """
        Set up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the setup ('train', 'val', 'test').
        """
        train_data = self.input_da.sel(self.domains['train'])
        post_fn = self.post_fn()
        self.train_ds = XrDataset(
            train_data, **self.xrds_kw, postpro_fn=post_fn,
        )
        if self.aug_kw:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrDataset(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn,
        )
        self.test_ds = XrDataset(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,
        )

    def train_dataloader(self):
        """
        Create a DataLoader for the training dataset.

        Returns:
            DataLoader: Training DataLoader.
        """
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        """
        Create a DataLoader for the validation dataset.

        Returns:
            DataLoader: Validation DataLoader.
        """
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        """
        Create a DataLoader for the testing dataset.

        Returns:
            DataLoader: Testing DataLoader.
        """
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)


class ConcatDataModule(BaseDataModule):
    """
    A data module for concatenating datasets from multiple domains.
    """

    def train_mean_std(self):
        """
        Compute the mean and standard deviation of the training data across domains.

        Returns:
            tuple: Mean and standard deviation.
        """
        sum, count = 0, 0
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {}))
        for domain in self.domains['train']:
            _sum, _count = train_data.sel(domain).sel(variable='tgt').pipe(
                lambda da: (da.sum(), da.pipe(np.isfinite).sum())
            )
            sum += _sum
            count += _count

        mean = sum / count
        sum = 0
        for domain in self.domains['train']:
            _sum = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: da - mean).pipe(np.square).sum()
            sum += _sum
        std = (sum / count)**0.5
        return mean.values.item(), std.values.item()

    def setup(self, stage='test'):
        """
        Set up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the setup ('train', 'val', 'test').
        """
        post_fn = self.post_fn()
        self.train_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['train']
        ])
        if self.aug_factor >= 1:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.val_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['val']
        ])
        self.test_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['test']
        ])


class RandValDataModule(BaseDataModule):
    """
    A data module that randomly splits the training data into training and validation sets.

    Attributes:
        val_prop (float): Proportion of data to use for validation.
    """

    def __init__(self, val_prop, *args, **kwargs):
        """
        Initialize the RandValDataModule.

        Args:
            val_prop (float): Proportion of data to use for validation.
        """
        super().__init__(*args, **kwargs)
        self.val_prop = val_prop

    def setup(self, stage='test'):
        """
        Set up the datasets for training, validation, and testing.

        Args:
            stage (str, optional): Stage of the setup ('train', 'val', 'test').
        """
        post_fn = self.post_fn()
        train_ds = XrDataset(self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn,)
        n_val = int(self.val_prop * len(train_ds))
        n_train = len(train_ds) - n_val
        self.train_ds, self.val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])

        if self.aug_factor > 1:
            self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

        self.test_ds = XrDataset(self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,)
