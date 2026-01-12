"""
This module provides utility functions for 4D-VarNet.

Utility functions include data preprocessing, optimization configuration,
diagnostics, and evaluation metrics.

Functions:
    pipe: Apply a sequence of functions to an input.
    kwgetattr: Get an attribute of an object by name.
    callmap: Apply a list of functions to an input and return the results.
    half_lr_adam: Configure an Adam optimizer with specific learning rates for model components.
    cosanneal_lr_adam: Configure an Adam optimizer with cosine annealing learning rate scheduling.
    cosanneal_lr_lion: Configure a Lion optimizer with cosine annealing learning rate scheduling.
    triang_lr_adam: Configure an Adam optimizer with triangular cyclic learning rate scheduling.
    remove_nan: Fill NaN values in a DataArray using Gauss-Seidel interpolation.
    get_constant_crop: Generate a constant cropping mask for patches.
    get_cropped_hanning_mask: Generate a cropped Hanning mask for patches.
    get_triang_time_wei: Generate a triangular time weighting mask for patches.
    load_enatl: Load ENATL dataset and preprocess it.
    load_altimetry_data: Load altimetry data and preprocess it.
    load_dc_data: Load DC data (currently a placeholder function).
    load_full_natl_data: Load full NATL dataset and preprocess it.
    rmse_based_scores_from_ds: Compute RMSE-based scores from a dataset.
    psd_based_scores_from_ds: Compute PSD-based scores from a dataset.
    rmse_based_scores: Compute RMSE-based scores for reconstruction evaluation.
    psd_based_scores: Compute PSD-based scores for reconstruction evaluation.
    diagnostics: Compute diagnostics for a given test domain.
    diagnostics_from_ds: Compute diagnostics from a dataset.
    test_osse: Perform OSSE testing and compute metrics.
    ensemble_metrics: Compute ensemble metrics for multiple checkpoints.
    add_geo_attrs: Add geographic attributes to a DataArray.
    vort: Compute vorticity from a DataArray.
    geo_energy: Compute geostrophic energy from a DataArray.
    best_ckpt: Retrieve the best checkpoint from an experiment directory.
    load_cfg: Load configuration files for an experiment.
"""



from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import metpy.calc as mpcalc
import kornia
import pandas as pd
import xrft
import torch
import xarray as xr
import matplotlib.pyplot as plt
from . import data

def pipe(inp, fns):
    """
    Apply a sequence of functions to an input.

    Args:
        inp: The input to process.
        fns (list): A list of functions to apply.

    Returns:
        The processed input after applying all functions.
    """
    for f in fns:
        inp = f(inp)
    return inp


def kwgetattr(obj, name):
    """
    Get an attribute of an object by name.

    Args:
        obj: The object to query.
        name (str): The name of the attribute.

    Returns:
        The value of the attribute.
    """
    return getattr(obj, name)


def callmap(inp, fns):
    """
    Apply a list of functions to an input and return the results.

    Args:
        inp: The input to process.
        fns (list): A list of functions to apply.

    Returns:
        list: A list of results from applying each function.
    """
    return [fn(inp) for fn in fns]


def half_lr_adam(lit_mod, lr):
    """
    Configure an Adam optimizer with specific learning rates for model components.

    Args:
        lit_mod: The Lightning module containing the model.
        lr (float): The base learning rate.

    Returns:
        torch.optim.Adam: The configured optimizer.
    """
    return torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ],
    )


def cosanneal_lr_adam(lit_mod, lr, T_max=100, weight_decay=0.):
    """
    Configure an Adam optimizer with cosine annealing learning rate scheduling.

    Args:
        lit_mod: The Lightning module containing the model.
        lr (float): The base learning rate.
        T_max (int): Maximum number of iterations for the scheduler.
        weight_decay (float): Weight decay for the optimizer.

    Returns:
        dict: A dictionary containing the optimizer and scheduler.
    """
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.obs_cost.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=weight_decay
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=T_max
        ),
    }


def cosanneal_lr_lion(lit_mod, lr, T_max=100):
    """
    Configure a Lion optimizer with cosine annealing learning rate scheduling.

    Args:
        lit_mod: The Lightning module containing the model.
        lr (float): The base learning rate.
        T_max (int): Maximum number of iterations for the scheduler.

    Returns:
        dict: A dictionary containing the optimizer and scheduler.
    """
    import lion_pytorch
    opt = lion_pytorch.Lion(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr / 2},
        ], weight_decay=1e-3
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max),
    }


def triang_lr_adam(lit_mod, lr_min=5e-5, lr_max=3e-3, nsteps=200):
    """
    Configure an Adam optimizer with triangular cyclic learning rate scheduling.

    Args:
        lit_mod: The Lightning module containing the model.
        lr_min (float): Minimum learning rate.
        lr_max (float): Maximum learning rate.
        nsteps (int): Number of steps for the triangular cycle.

    Returns:
        dict: A dictionary containing the optimizer and scheduler.
    """
    opt = torch.optim.Adam(
        [
            {"params": lit_mod.solver.grad_mod.parameters(), "lr": lr_max},
            {"params": lit_mod.solver.prior_cost.parameters(), "lr": lr_max / 2},
        ],
    )
    return {
        "optimizer": opt,
        "lr_scheduler": torch.optim.lr_scheduler.CyclicLR(
            opt,
            base_lr=lr_min,
            max_lr=lr_max,
            step_size_up=nsteps,
            step_size_down=nsteps,
            gamma=0.95,
            cycle_momentum=False,
            mode="exp_range",
        ),
    }


def remove_nan(da):
    """
    Fill NaN values in a DataArray using Gauss-Seidel interpolation.
    Version proche de pyinterp.fill.gauss_seidel.

    Args:
        da (xarray.DataArray): The input DataArray.

    Returns:
        xarray.DataArray: The DataArray with NaN values filled.
    """
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")
    
    da_filled = da.copy()
    
    # Transposer pour avoir (lon, lat, time)
    data = da.transpose("lon", "lat", "time").values.copy()
    
    # Appliquer Gauss-Seidel sur chaque tranche temporelle
    for t in range(data.shape[2]):
        data[:, :, t] = gauss_seidel_fill(data[:, :, t])
    
    # Réassigner les données
    da_filled.transpose("lon", "lat", "time")[:, :] = data
    
    return da_filled


def gauss_seidel_fill(grid, max_iterations=1000, tolerance=1e-5):
    """
    Fill NaN values using Gauss-Seidel iterative method.
    
    Args:
        grid (np.ndarray): 2D array with NaN values to fill
        max_iterations (int): Maximum number of iterations
        tolerance (float): Convergence threshold
    
    Returns:
        np.ndarray: Grid with NaN values filled
    """
    grid = grid.copy()
    nan_mask = np.isnan(grid)
    
    if not np.any(nan_mask):
        return grid
    
    # Initialiser les NaN avec la moyenne des valeurs valides
    initial_value = np.nanmean(grid)
    grid[nan_mask] = initial_value
    
    nx, ny = grid.shape
    
    for iteration in range(max_iterations):
        max_change = 0.0
        
        # Balayage Gauss-Seidel
        for i in range(nx):
            for j in range(ny):
                if nan_mask[i, j]:
                    old_value = grid[i, j]
                    
                    # Collecter les voisins valides
                    neighbors = []
                    weights = []
                    
                    if i > 0:
                        neighbors.append(grid[i-1, j])
                        weights.append(1.0)
                    if i < nx - 1:
                        neighbors.append(grid[i+1, j])
                        weights.append(1.0)
                    if j > 0:
                        neighbors.append(grid[i, j-1])
                        weights.append(1.0)
                    if j < ny - 1:
                        neighbors.append(grid[i, j+1])
                        weights.append(1.0)
                    
                    if neighbors:
                        # Moyenne pondérée des voisins
                        new_value = np.average(neighbors, weights=weights)
                        grid[i, j] = new_value
                        max_change = max(max_change, abs(new_value - old_value))
        
        # Vérifier la convergence
        if max_change < tolerance:
            break
    
    return grid


def get_constant_crop(patch_dims, crop, dim_order=["time", "lat", "lon"]):
    """
    Generate a constant cropping mask for patches.

    Args:
        patch_dims (dict): Dimensions of the patch.
        crop (dict): Crop sizes for each dimension.
        dim_order (list): Order of dimensions.

    Returns:
        numpy.ndarray: A mask with cropped regions set to 0 and others to 1.
    """
    patch_weight = np.zeros([patch_dims[d] for d in dim_order], dtype="float32")
    mask = tuple(
        slice(crop[d], -crop[d]) if crop.get(d, 0) > 0 else slice(None, None)
        for d in dim_order
    )
    patch_weight[mask] = 1.0
    return patch_weight


def get_cropped_hanning_mask(patch_dims, crop, **kwargs):
    """
    Generate a cropped Hanning mask for patches.

    Args:
        patch_dims (dict): Dimensions of the patch.
        crop (dict): Crop sizes for each dimension.

    Returns:
        numpy.ndarray: The cropped Hanning mask.
    """
    pw = get_constant_crop(patch_dims, crop)
    t_msk = kornia.filters.get_hanning_kernel1d(patch_dims["time"])
    patch_weight = t_msk[:, None, None] * pw
    return patch_weight.cpu().numpy()


def get_triang_time_wei(patch_dims, offset=0, **crop_kw):
    """
    Generate a triangular time weighting mask for patches.

    Args:
        patch_dims (dict): Dimensions of the patch.
        offset (int): Offset for the triangular weighting.
        crop_kw (dict): Additional cropping parameters.

    Returns:
        numpy.ndarray: The triangular time weighting mask.
    """
    pw = get_constant_crop(patch_dims, **crop_kw)
    return np.fromfunction(
        lambda t, *a: (
            (1 - np.abs(offset + 2 * t - patch_dims["time"]) / patch_dims["time"]) * pw
        ),
        patch_dims.values(),
    )

# ============================================================================
# Dataloader Functions
# ============================================================================
def _create_training_dataset(input_data, target_data, coords=None, 
                            input_field='input', target_field='tgt'):
    """
    Create a standardized dataset with customizable fields.
    
    Args:
        input_data (xr.DataArray): Input data (observations)
        target_data (xr.DataArray): Target data (ground truth)
        coords (dict, optional): Coordinates to use (default: coords from input_data)
        input_field (str): Name of the field for input data
        target_field (str): Name of the field for target data
        
    Returns:
        xr.Dataset: Dataset with the specified fields
    """
    if coords is None:
        coords = input_data.coords
    
    return xr.Dataset(
        {
            input_field: input_data,
            target_field: (target_data.dims, target_data.values)
        },
        coords
    )


def _finalize_dataset(ds, obs_from_tgt=False, transpose_dims=('time', 'lat', 'lon'), 
                     load=False, transpose_tgt_for_obs=True,
                     fields=None):
    """
    Finalize a dataset by applying common transformations.
    
    Args:
        ds (xr.Dataset): Dataset with the specified variables
        obs_from_tgt (bool): If True, replaces input with target masked by input
        transpose_dims (tuple): Dimensions for transposition
        load (bool): If True, loads data into memory
        transpose_tgt_for_obs (bool): If True, transposes target before masking
        fields (list[str], optional): List of fields to select 
                                     (default: TrainingItem._fields)
        
    Returns:
        xr.DataArray: Finalized and sorted DataArray by variable
    """
    if fields is None:
        fields = list(TrainingItem._fields)
    
    input_field = fields[0]
    target_field = fields[1]
    
    # Apply obs_from_tgt if requested
    if obs_from_tgt:
        tgt = ds[target_field]
        if transpose_tgt_for_obs:
            tgt = tgt.transpose(*ds[input_field].dims)
        ds = ds.assign(
            **{input_field: tgt.where(np.isfinite(ds[input_field]), np.nan)}
        )
    
    # Select fields, transpose and convert
    result = (
        ds[fields]
        .transpose(*transpose_dims)
        .to_array()
        .sortby('variable')
    )
    
    return result.load() if load else result

def load_altimetry_data(path, obs_from_tgt=False):
    """
    Load and preprocess altimetry data.

    Args:
        path (str): Path to the altimetry dataset.
        obs_from_tgt (bool): Whether to use target data as observations.

    Returns:
        xarray.DataArray: The preprocessed altimetry dataset.
    """
    # Load raw data
    raw_ds = xr.open_dataset(path).load()
    
    # Create standardized dataset with TrainingItem
    ds = _create_training_dataset(
        input_data=raw_ds.nadir_obs,
        target_data=remove_nan(raw_ds.ssh),
        coords=raw_ds.nadir_obs.coords,
        input_field=TrainingItem._fields[0],
        target_field=TrainingItem._fields[1]
    )
    
    # Finalize with common transformations
    return _finalize_dataset(
        ds, 
        obs_from_tgt=obs_from_tgt,
        fields=list(TrainingItem._fields)
    )



def load_enatl(*args, obs_from_tgt=True, **kwargs):
    """
    Load and preprocess the ENATL dataset.

    Args:
        obs_from_tgt (bool): Whether to use target data as observations.

    Returns:
        xarray.DataArray: The preprocessed ENATL dataset.
    """
    # ds = xr.open_dataset('../sla-data-registry/qdata/enatl_wo_tide.nc')
    # print(ds)
    # return ds.rename(nadir_obs='input', ssh='tgt')\
    #     .to_array()\
    #     .transpose('variable', 'time', 'lat', 'lon')\
    #     .sortby('variable')
    ssh = xr.open_zarr('../sla-data-registry/enatl_preproc/truth_SLA_SSH_NATL60.zarr/').ssh
    nadirs = xr.open_zarr('../sla-data-registry/enatl_preproc/SLA_SSH_5nadirs.zarr/').ssh
    ssh = ssh.interp(
        lon=np.arange(ssh.lon.min(), ssh.lon.max(), 1/20),
        lat=np.arange(ssh.lat.min(), ssh.lat.max(), 1/20)
    )
    nadirs = nadirs.interp(time=ssh.time, method='nearest')\
        .interp(lat=ssh.lat, lon=ssh.lon, method='zero')
    ds = xr.Dataset(dict(input=nadirs, tgt=(ssh.dims, ssh.values)), nadirs.coords)
    if obs_from_tgt:
        ds = ds.assign(input=ds.tgt.transpose(*ds.input.dims).where(np.isfinite(ds.input), np.nan))
    return ds.transpose('time', 'lat', 'lon').to_array().load().sortby('variable')


def load_dc_data(**kwargs):
    """
    Load DC data.

    This is currently a placeholder function for loading DC data.

    Args:
        kwargs

    Returns:
        None
    """
    path_gt = "../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
    path_obs = "NATL60/NATL/data_new/dataset_nadir_0d.nc"


def load_full_natl_data(
    path_obs="../sla-data-registry/CalData/cal_data_new_errs.nc",
    path_gt="../sla-data-registry/NATL60/NATL/ref_new/NATL60-CJM165_NATL_ssh_y2013.1y.nc",
    obs_var='five_nadirs',
    gt_var='ssh',
    **kwargs
):
    """
    Load and preprocess the full NATL dataset.

    Args:
        path_obs (str): Path to the observation dataset.
        path_gt (str): Path to the ground truth dataset.
        obs_var (str): Observation variable name.
        gt_var (str): Ground truth variable name.

    Returns:
        xarray.DataArray: The preprocessed NATL dataset.
    """
    inp = xr.open_dataset(path_obs)[obs_var]
    gt = (
        xr.open_dataset(path_gt)[gt_var]
        # .isel(time=slice(0, -1))
        .sel(lat=inp.lat, lon=inp.lon, method="nearest")
    )

    return xr.Dataset(dict(input=inp, tgt=(gt.dims, gt.values)), inp.coords).to_array().sortby('variable')


def rmse_based_scores_from_ds(ds, ref_variable='tgt', study_variable='out'):
    """
    Compute RMSE-based scores from a dataset.

    Args:
        ds (xarray.Dataset): The dataset containing the reference and study variables.
        ref_variable (str): The name of the reference variable.
        study_variable (str): The name of the study variable.

    Returns:
        list: A list containing RMSE-based scores.
    """
    try:
        return rmse_based_scores(ds[study_variable], ds[ref_variable])[2:]
    except Exception:
        return [np.nan, np.nan]


def psd_based_scores_from_ds(ds, ref_variable='tgt', study_variable='out'):
    """
    Compute PSD-based scores from a dataset.

    Args:
        ds (xarray.Dataset): The dataset containing the reference and study variables.
        ref_variable (str): The name of the reference variable.
        study_variable (str): The name of the study variable.

    Returns:
        list: A list containing PSD-based scores.
    """
    try:
        return psd_based_scores(ds[study_variable], ds[ref_variable])[1:]
    except Exception:
        return [np.nan, np.nan]


def rmse_based_scores(da_rec, da_ref):
    """
    Compute RMSE-based scores for reconstruction evaluation.

    Args:
        da_rec (xarray.DataArray): The reconstructed data.
        da_ref (xarray.DataArray): The reference data.

    Returns:
        tuple: A tuple containing RMSE-based scores.
    """
    rmse_t = (
        1.0
        - (((da_rec - da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
        / (((da_ref) ** 2).mean(dim=("lon", "lat"))) ** 0.5
    )
    rmse_xy = (((da_rec - da_ref) ** 2).mean(dim=("time"))) ** 0.5
    rmse_t = rmse_t.rename("rmse_t")
    rmse_xy = rmse_xy.rename("rmse_xy")
    reconstruction_error_stability_metric = rmse_t.std().values
    leaderboard_rmse = (
        1.0 - (((da_rec - da_ref) ** 2).mean()) ** 0.5 / (((da_ref) ** 2).mean()) ** 0.5
    )
    return (
        rmse_t,
        rmse_xy,
        np.round(leaderboard_rmse.values, 5).item(),
        np.round(reconstruction_error_stability_metric, 5).item(),
    )


def psd_based_scores(da_rec, da_ref):
    """
    Compute PSD-based scores for reconstruction evaluation.

    Args:
        da_rec (xarray.DataArray): The reconstructed data.
        da_ref (xarray.DataArray): The reference data.

    Returns:
        tuple: A tuple containing PSD-based scores and resolved wavelengths.
    """
    err = da_rec - da_ref
    err["time"] = (err.time - err.time[0]) / np.timedelta64(1, "D")
    signal = da_ref
    signal["time"] = (signal.time - signal.time[0]) / np.timedelta64(1, "D")
    psd_err = xrft.power_spectrum(
        err, dim=["time", "lon"], detrend="constant", window="hann"
    ).compute()
    psd_signal = xrft.power_spectrum(
        signal, dim=["time", "lon"], detrend="constant", window="hann"
    ).compute()
    mean_psd_signal = psd_signal.mean(dim="lat").where(
        (psd_signal.freq_lon > 0.0) & (psd_signal.freq_time > 0), drop=True
    )
    mean_psd_err = psd_err.mean(dim="lat").where(
        (psd_err.freq_lon > 0.0) & (psd_err.freq_time > 0), drop=True
    )
    psd_based_score = 1.0 - mean_psd_err / mean_psd_signal
    level = [0.5]
    cs = plt.contour(
        1.0 / psd_based_score.freq_lon.values,
        1.0 / psd_based_score.freq_time.values,
        psd_based_score,
        level,
    )
    x05, y05 = cs.collections[0].get_paths()[0].vertices.T
    plt.close()

    shortest_spatial_wavelength_resolved = np.min(x05)
    shortest_temporal_wavelength_resolved = np.min(y05)
    psd_da = 1.0 - mean_psd_err / mean_psd_signal
    psd_da.name = "psd_score"
    return (
        psd_da.to_dataset(),
        np.round(shortest_spatial_wavelength_resolved, 3).item(),
        np.round(shortest_temporal_wavelength_resolved, 3).item(),
    )


def diagnostics(lit_mod, test_domain):
    """
    Compute diagnostics for a given test domain.

    Args:
        lit_mod: The Lightning module containing the model.
        test_domain (dict): The test domain to evaluate.

    Returns:
        pandas.Series: A series containing diagnostic metrics.
    """
    test_data = lit_mod.test_data.sel(test_domain)
    return diagnostics_from_ds(test_data, test_domain)


def diagnostics_from_ds(test_data, test_domain):
    """
    Compute diagnostics from a dataset.

    Args:
        test_data (xarray.Dataset): The test data.
        test_domain (dict): The test domain to evaluate.

    Returns:
        pandas.Series: A series containing diagnostic metrics.
    """
    test_data = test_data.sel(test_domain)
    metrics = {
        "RMSE (m)": test_data.pipe(lambda ds: (ds.out - ds.tgt))
        .pipe(lambda da: da**2)
        .mean()
        .pipe(np.sqrt)
        .item(),
        **dict(
            zip(
                ["λx", "λt"],
                test_data.pipe(lambda ds: psd_based_scores(ds.out, ds.tgt)[1:]),
            )
        ),
        **dict(
            zip(
                ["μ", "σ"],
                test_data.pipe(lambda ds: rmse_based_scores(ds.out, ds.tgt)[2:]),
            )
        ),
    }
    return pd.Series(metrics, name="osse_metrics")


def test_osse(trainer, lit_mod, osse_dm, osse_test_domain, ckpt, diag_data_dir=None):
    """
    Perform OSSE (Observing System Simulation Experiment) testing and compute metrics.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer instance.
        lit_mod (pl.LightningModule): The Lightning module to test.
        osse_dm (pl.LightningDataModule): The datamodule for OSSE testing.
        osse_test_domain (dict): The test domain for evaluation.
        ckpt (str): Path to the checkpoint to load.
        diag_data_dir (Path, optional): Directory to save diagnostic data.

    Returns:
        pandas.Series: A series containing OSSE metrics.
    """
    lit_mod.norm_stats = osse_dm.norm_stats()
    trainer.test(lit_mod, datamodule=osse_dm, ckpt_path=ckpt)
    osse_tdat = lit_mod.test_data[['out', 'ssh']]
    osse_metrics = diagnostics_from_ds(
        osse_tdat, test_domain=osse_test_domain
    )

    print(osse_metrics.to_markdown())

    if diag_data_dir is not None:
        osse_metrics.to_csv(diag_data_dir / "osse_metrics.csv")
        if (diag_data_dir / "osse_test_data.nc").exists():
            xr.open_dataset(diag_data_dir / "osse_test_data.nc").close()
        osse_tdat.to_netcdf(diag_data_dir / "osse_test_data.nc")

    return osse_metrics


def ensemble_metrics(trainer, lit_mod, ckpt_list, dm, save_path):
    """
    Compute ensemble metrics for multiple checkpoints.

    Args:
        trainer (pl.Trainer): The PyTorch Lightning trainer instance.
        lit_mod (pl.LightningModule): The Lightning module to test.
        ckpt_list (list): List of checkpoint paths to evaluate.
        dm (pl.LightningDataModule): The datamodule for testing.
        save_path (str): Path to save the metrics and ensemble outputs.

    Returns:
        None
    """
    metrics = []
    test_data = xr.Dataset()
    for i, ckpt in enumerate(ckpt_list):
        trainer.test(lit_mod, ckpt_path=ckpt, datamodule=dm)
        rmse = (
            lit_mod.test_data.pipe(lambda ds: (ds.out - ds.ssh))
            .pipe(lambda da: da**2)
            .mean()
            .pipe(np.sqrt)
            .item()
        )
        lx, lt = psd_based_scores(lit_mod.test_data.out, lit_mod.test_data.ssh)[1:]
        mu, sig = rmse_based_scores(lit_mod.test_data.out, lit_mod.test_data.ssh)[2:]

        metrics.append(dict(ckpt=ckpt, rmse=rmse, lx=lx, lt=lt, mu=mu, sig=sig))

        if i == 0:
            test_data = lit_mod.test_data
            test_data = test_data.rename(out=f"out_{i}")
        else:
            test_data = test_data.assign(**{f"out_{i}": lit_mod.test_data.out})
        test_data[f"out_{i}"] = test_data[f"out_{i}"].assign_attrs(
            ckpt=str(ckpt)
        )

    metric_df = pd.DataFrame(metrics)
    print(metric_df.to_markdown())
    print(metric_df.describe().to_markdown())
    metric_df.to_csv(save_path + "/metrics.csv")
    test_data.to_netcdf(save_path + "ens_out.nc")


def add_geo_attrs(da):
    """
    Add geographic attributes (longitude and latitude units) to a DataArray.

    Args:
        da (xarray.DataArray): The input DataArray.

    Returns:
        xarray.DataArray: The DataArray with geographic attributes added.
    """
    da["lon"] = da.lon.assign_attrs(units="degrees_east")
    da["lat"] = da.lat.assign_attrs(units="degrees_north")
    return da


def vort(da):
    """
    Compute the vorticity from a DataArray.

    Args:
        da (xarray.DataArray): The input DataArray.

    Returns:
        xarray.DataArray: The vorticity computed from the input data.
    """
    return mpcalc.vorticity(
        *mpcalc.geostrophic_wind(
            da.pipe(add_geo_attrs).assign_attrs(units="m").metpy.quantify()
        )
    ).metpy.dequantify()


def geo_energy(da):
    """
    Compute the geostrophic energy from a DataArray.

    Args:
        da (xarray.DataArray): The input DataArray.

    Returns:
        xarray.DataArray: The geostrophic energy computed from the input data.
    """
    return np.hypot(*mpcalc.geostrophic_wind(da.pipe(add_geo_attrs))).metpy.dequantify()


def best_ckpt(xp_dir):
    """
    Retrieve the best checkpoint from an experiment directory.

    Args:
        xp_dir (str): Path to the experiment directory.

    Returns:
        str: Path to the best checkpoint file.
    """
    _, xpn = load_cfg(xp_dir)
    if xpn is None:
        return None
    print(Path(xp_dir) / xpn / 'checkpoints')
    ckpt_last = max(
        (Path(xp_dir) / xpn / 'checkpoints').glob("*.ckpt"), key=lambda p: p.stat().st_mtime
    )
    cbs = torch.load(ckpt_last)["callbacks"]
    ckpt_cb = cbs[next(k for k in cbs.keys() if "ModelCheckpoint" in k)]
    return ckpt_cb["best_model_path"]


def load_cfg(xp_dir):
    """
    Load configuration files for an experiment.

    Args:
        xp_dir (str): Path to the experiment directory.

    Returns:
        tuple: A tuple containing the configuration and the experiment name.
    """
    hydra_cfg = OmegaConf.load(Path(xp_dir) / ".hydra/hydra.yaml").hydra
    cfg = OmegaConf.load(Path(xp_dir) / ".hydra/config.yaml")
    OmegaConf.register_new_resolver(
        "hydra", lambda k: OmegaConf.select(hydra_cfg, k), replace=True
    )
    try:
        OmegaConf.resolve(cfg)
        OmegaConf.resolve(cfg)
    except Exception:
        return None, None

    return cfg, OmegaConf.select(hydra_cfg, "runtime.choices.xp")
