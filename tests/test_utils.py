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


# Création d'un DataArray avec Nan pour les tests
@pytest.fixture
def dataarray_with_nans():
    # Création d'un DataArray avec des NaN dans certaines positions
    lon = np.linspace(0, 10, 11)
    lat = np.linspace(-5, 5, 11)
    time = np.linspace(0, 2, 3)
    data = np.random.rand(3, 11, 11)

    # Introduire des NaN dans des positions aléatoires
    data[0, 5, 5] = np.nan  # NaN à une position
    data[1, 4, 4] = np.nan  # NaN à une autre position

    da = xr.DataArray(data, coords=[("time", time), ("lat", lat), ("lon", lon)])
    return da

# Création d'un patch_dims et d'un crop de test
@pytest.fixture
def patch_and_crop():
    patch_dims = {"time": 5, "lat": 10, "lon": 10}  # Exemple de dimensions du patch
    crop = {"time": 3, "lat": 8, "lon": 8}  # Exemple de crop
    return patch_dims, crop

# -------------------------------

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



# Test pour la fonction remove_nan
def test_remove_nan(dataarray_with_nans):
    da = dataarray_with_nans

    # Avant de remplir les NaN
    assert np.isnan(da.sel(time=0).isel(lat=5, lon=5))  # Il y a un NaN à cette position
    assert np.isnan(da.sel(time=1).isel(lat=4, lon=4))  # NaN à une autre position

    # Appeler la fonction remove_nan
    da_filled = remove_nan(da)

    # Vérifier que les NaN ont été remplis
    assert not np.isnan(da_filled.sel(time=0).isel(lat=5, lon=5))  # Vérifier que ce n'est plus un NaN
    assert not np.isnan(da_filled.sel(time=1).isel(lat=4, lon=4))  # Vérifier l'autre NaN

    # Vérifier que les dimensions sont correctes
    assert da_filled.dims == da.dims  # Les dimensions doivent être les mêmes

    # Vérifier les attributs des coordonnées
    assert da_filled.lon.attrs["units"] == "degrees_east"
    assert da_filled.lat.attrs["units"] == "degrees_north"

    # Vérifier que la méthode de remplissage n'a pas modifié des données non-NaN
    original_value = da.sel(time=0).isel(lat=0, lon=0).values
    assert da_filled.sel(time=0).isel(lat=0, lon=0).values == original_value  # La valeur ne doit pas avoir changé

# def test_get_constant_crop():
#     """
#     Test the get_constant_crop function.
#     """
#     patch_dims = {"time": 10, "lat": 20, "lon": 30}
#     crop = {"time": 2, "lat": 3, "lon": 4}
#     mask = get_constant_crop(patch_dims, crop)
#     assert mask.shape == (10, 20, 30), "get_constant_crop returned incorrect shape."
#     assert np.all(mask[2:-2, 3:-3, 4:-4] == 1), "get_constant_crop failed to apply crop."


# # Test de la fonction get_cropped_hanning_mask
# def test_get_cropped_hanning_mask(patch_and_crop):
#     patch_dims, crop = patch_and_crop

#     # Appeler la fonction
#     mask = get_cropped_hanning_mask(patch_dims, crop)

#     # Vérifier que la forme du masque est correcte
#     expected_shape = (patch_dims["time"], crop["lat"], crop["lon"])
#     assert mask.shape == expected_shape, f"Expected shape {expected_shape}, but got {mask.shape}"

#     # Vérifier que le masque contient bien des valeurs entre 0 et 1 (puisque c'est un masque de Hanning)
#     assert np.all(mask >= 0) and np.all(mask <= 1), "The mask contains values outside the range [0, 1]"

#     # Vérifier que la dimension 'time' du masque est bien conforme à la taille donnée par patch_dims["time"]
#     time_mask = mask[0, :, :]  # Prenons la première tranche du temps
#     hanning_kernel = kornia.filters.get_hanning_kernel1d(patch_dims["time"]).cpu().numpy()  # Générer un kernel Hanning pour vérifier
#     assert np.allclose(time_mask, hanning_kernel[:, None], atol=1e-6), "Time dimension Hanning kernel mismatch"

#     # Vérifier que les autres dimensions (lat, lon) sont correctement modifiées par le crop
#     # Les autres dimensions devraient être une multiplication du poids de Hanning et de l'autre facteur `pw`
#     expected_patch_weight = time_mask[:, None, None] * get_constant_crop(patch_dims, crop).cpu().numpy()
#     assert np.allclose(mask, expected_patch_weight, atol=1e-6), "The mask does not match the expected values"

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