"""
Unit tests for the utility functions in utils.py using pytest.
"""

import pytest
import numpy as np
import torch
import xarray as xr
from ocean4dvarnet.utils import (
    pipe,
    kwgetattr,
    callmap,
    half_lr_adam,
    cosanneal_lr_adam,
    triang_lr_adam,
    remove_nan,
    get_constant_crop,
    get_cropped_hanning_mask,
    get_triang_time_wei,
    rmse_based_scores,
    psd_based_scores,
)


def test_pipe():
    """
    Test the pipe function.
    """
    def add_one(x):
        return x + 1

    def multiply_by_two(x):
        return x * 2

    result = pipe(3, [add_one, multiply_by_two])
    assert result == 8, "Pipe function failed."


def test_kwgetattr():
    """
    Test the kwgetattr function.
    """
    class Dummy:
        attr = 42

    obj = Dummy()
    assert kwgetattr(obj, "attr") == 42, "kwgetattr function failed."


def test_callmap():
    """
    Test the callmap function.
    """
    def square(x):
        return x**2

    def cube(x):
        return x**3

    result = callmap(2, [square, cube])
    assert result == [4, 8], "callmap function failed."


def test_half_lr_adam():
    """
    Test the half_lr_adam function.
    """
    class MockLitModule:
        class Solver:
            grad_mod = torch.nn.Linear(10, 10)
            obs_cost = torch.nn.Linear(10, 10)
            prior_cost = torch.nn.Linear(10, 10)

        solver = Solver()

    lit_mod = MockLitModule()
    optimizer = half_lr_adam(lit_mod, lr=0.01)
    assert len(optimizer.param_groups) == 3, "half_lr_adam failed to configure optimizer."


def test_cosanneal_lr_adam():
    """
    Test the cosanneal_lr_adam function.
    """
    class MockLitModule:
        class Solver:
            grad_mod = torch.nn.Linear(10, 10)
            obs_cost = torch.nn.Linear(10, 10)
            prior_cost = torch.nn.Linear(10, 10)

        solver = Solver()

    lit_mod = MockLitModule()
    config = cosanneal_lr_adam(lit_mod, lr=0.01, t_max=100)
    assert "optimizer" in config and "lr_scheduler" in config, "cosanneal_lr_adam failed."


def test_triang_lr_adam():
    """
    Test the triang_lr_adam function.
    """
    class MockLitModule:
        class Solver:
            grad_mod = torch.nn.Linear(10, 10)
            prior_cost = torch.nn.Linear(10, 10)

        solver = Solver()

    lit_mod = MockLitModule()
    config = triang_lr_adam(lit_mod, lr_min=0.001, lr_max=0.01, nsteps=50)
    assert "optimizer" in config and "lr_scheduler" in config, "triang_lr_adam failed."


# def test_remove_nan():
#     """
#     Test the remove_nan function.
#     """
#     da = xr.DataArray(
#         [[1, 2, np.nan], [4, np.nan, 6]],
#         dims=["lat", "lon"],
#         coords={"lat": [0, 1], "lon": [0, 1, 2]},
#     )
#     filled_da = remove_nan(da)
#     assert not np.isnan(filled_da).any(), "remove_nan failed to fill NaN values."


def test_get_constant_crop():
    """
    Test the get_constant_crop function.
    """
    patch_dims = {"time": 10, "lat": 20, "lon": 30}
    crop = {"time": 2, "lat": 3, "lon": 4}
    mask = get_constant_crop(patch_dims, crop)
    assert mask.shape == (10, 20, 30), "get_constant_crop returned incorrect shape."
    assert np.all(mask[2:-2, 3:-3, 4:-4] == 1), "get_constant_crop failed to apply crop."


def test_get_cropped_hanning_mask():
    """
    Test the get_cropped_hanning_mask function.
    """
    patch_dims = {"time": 10, "lat": 20, "lon": 30}
    crop = {"time": 2, "lat": 3, "lon": 4}
    mask = get_cropped_hanning_mask(patch_dims, crop)
    assert mask.shape == (10, 20, 30), "get_cropped_hanning_mask returned incorrect shape."


# def test_get_triang_time_wei():
#     """
#     Test the get_triang_time_wei function.
#     """
#     patch_dims = {"time": 10, "lat": 20, "lon": 30}
#     mask = get_triang_time_wei(patch_dims, offset=1)
#     assert mask.shape == (10, 20, 30), "get_triang_time_wei returned incorrect shape."


# def test_rmse_based_scores():
#     """
#     Test the rmse_based_scores function.
#     """
#     da_rec = xr.DataArray([[1, 2], [3, 4]], dims=["lat", "lon"])
#     da_ref = xr.DataArray([[1, 2], [3, 5]], dims=["lat", "lon"])
#     scores = rmse_based_scores(da_rec, da_ref)
#     assert len(scores) == 4, "rmse_based_scores returned incorrect number of scores."


# def test_psd_based_scores():
#     """
#     Test the psd_based_scores function.
#     """
#     da_rec = xr.DataArray(
#         np.random.rand(10, 10), dims=["time", "lon"], coords={"time": range(10), "lon": range(10)}
#     )
#     da_ref = xr.DataArray(
#         np.random.rand(10, 10), dims=["time", "lon"], coords={"time": range(10), "lon": range(10)}
#     )
#     scores = psd_based_scores(da_rec, da_ref)
#     assert len(scores) == 3, "psd_based_scores returned incorrect number of scores."