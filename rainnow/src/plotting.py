"""Plotting module for generic plotting functionality used throughout the repo / rainnow package."""

import re
from typing import Any, Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.feature import NaturalEarthFeature


def plot_raw_imerg_xarray_tile(
    imerg_tile: xr.DataArray,
    plot_params: dict,
    cbar_params: dict,
    global_params: dict,
    figsize: Tuple[int, int] = (10, 10),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot an xarray DataArray representing IMERG data.

    Parameters
    ----------
    imerg_tile : xr.DataArray
        An xarray DataArray containing the IMERG data to be plotted.
    plot_params : dict
        Dictionary containing parameters to pass to the `imshow` plotting function.
    cbar_params : dict
        Dictionary containing parameters to configure the colorbar.
    figsize : Tuple[int, int], optional
        Size of the figure (width, height) in inches. Default is (10, 10).

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object created.
    ax : plt.Axes
        The matplotlib axes object created.
    """
    plt.rcParams.update(global_params)  # Update global parameters.

    fig, ax = plt.subplots(figsize=figsize)
    plot = ax.imshow(imerg_tile, **plot_params)

    if cbar_params:
        # colour bar.
        cbar = fig.colorbar(plot, ax=ax, **cbar_params)
        # harcoded cbar ticks and label.
        cbar.set_label(r"$mm.hr^{-1}$")
        cbar.set_ticks(np.arange(1, 9, 1))
        cbar.set_ticklabels(["0.1", "0.5", "1", "2", "4", "8", "16", "32"])

    return fig, ax


def geoplot_raw_imerg_xarray_tile(
    imerg_tile: xr.DataArray,
    plot_params: dict,
    cbar_params: dict,
    geo_params: dict,
    global_params: dict,
    projection=ccrs.PlateCarree(),
    feature_resolution: str = "110m",
    figsize: Tuple[int, int] = (10, 10),
    remove_axes_ticks: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot raw IMERG data from an xarray DataArray on a geographical map using Cartopy.

    Parameters
    ----------
    imerg_tile : xr.DataArray
        The xarray DataArray containing the IMERG data.
    plot_params : dict
        Dictionary containing parameters for the `pcolormesh` plotting function.
    cbar_params : dict
        Dictionary containing parameters for the colorbar.
    geo_params : dict
        Dictionary containing parameters for the geo features.
    projection : cartopy.crs.Projection, optional
        The cartopy projection to use for plotting the data.
    feature_resolution : str, optional
        Resolution of the natural earth features ('10m', '50m', '110m').
    figsize : tuple of int, optional
        Size of the figure to create.
    remove_axes_ticks: bool
        Bool flag to either keep (False) or remove axes ticks (True). Defaults to False (keep).
    Returns
    -------
    tuple
        A tuple containing the matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection})
    plt.rcParams.update(global_params)  # Update global parameters.

    # create grid.
    xx, yy = np.float32(np.meshgrid(imerg_tile.longitude, imerg_tile.latitude))
    plot = ax.pcolormesh(xx, yy, imerg_tile.data, transform=projection, **plot_params)

    if not remove_axes_ticks:
        ax.gridlines(draw_labels=True)

    ax.set_extent(
        [
            imerg_tile.longitude.min(),
            imerg_tile.longitude.max(),
            imerg_tile.latitude.min(),
            imerg_tile.latitude.max(),
        ],
        projection,
    )
    # add coastlines and lakes.
    lakes = NaturalEarthFeature("physical", "lakes", feature_resolution)
    ax.add_feature(lakes, edgecolor="grey", facecolor="none", **geo_params)
    ax.coastlines(resolution=feature_resolution, color="grey", **geo_params)

    if cbar_params:
        # colour bar.
        cbar = fig.colorbar(plot, ax=ax, **cbar_params)
        # harcoded cbar ticks and label.
        cbar.ax.text(
            1.05,
            -5,
            r"$\text{mm}\cdot\text{h}^{-1}$",
            transform=cbar.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        cbar.set_ticks(np.arange(1, 9, 1))
        cbar.set_ticklabels(["0.1", "0.5", "1", "2", "4", "8", "16", "32"])

    return fig, ax

def add_bounding_box_to_plot(
    ax: plt.Axes,
    xs: List[float],
    ys: List[float],
    box_params: Dict[str, Any],
) -> None:
    """
    Adds a bounding box to a plot using specified x and y coordinates.

    Need to give same number of x and y coords.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib Axes object where the bounding box will be added.
    xs : list of float
        List of x-coordinates for the vertices of the bounding box.
    ys : list of float
        List of y-coordinates for the vertices of the bounding box.
    line_color : str, optional
        Color of the bounding box line. Default is 'red'.
    line_width : int, optional
        Width of the bounding box line. Default is 1.

    Raises
    ------
    AssertionError
        If the lengths of xs and ys do not match.
    """
    assert len(xs) == len(ys)
    ax.plot(xs, ys, **box_params)


def add_patch_to_geoplot(
    ax: plt.Axes,
    patch_size: int,
    lat_pixel_resolution: float,
    lon_pixel_resolution: float,
    top_left_lon_lat: tuple,
    nx: int,
    ny: int,
    box_params,
) -> None:
    """
    Adds a rectangular patch on a geographic plot to highlight a specific area.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to which the patch will be added.
    patch_size : int
        Size of the patch in pixels.
    top_left_lon_lat : tuple
        Top left corner longitude and latitude of the first patch.
    nx : int
        Number of patches along the x-axis (longitude).
    ny : int
        Number of patches along the y-axis (latitude).
    box_params : Dict[str, Any]
        Parameters to be passed to the `plot` function to define the box properties.
    Returns
    -------
    None
    """
    patch_lat_deg = patch_size * lat_pixel_resolution
    patch_lon_deg = patch_size * lon_pixel_resolution

    # patch starting coordinates.
    x_start = top_left_lon_lat[0]
    y_start = top_left_lon_lat[1]

    # draw patch.
    ax.plot(
        [
            x_start + (nx * patch_lon_deg),
            x_start + ((nx + 1) * patch_lon_deg),
            x_start + ((nx + 1) * patch_lon_deg),
            x_start + (nx * patch_lon_deg),
            x_start + (nx * patch_lon_deg),
        ],
        [
            y_start - (ny * patch_lat_deg),
            y_start - (ny * patch_lat_deg),
            y_start - ((ny + 1) * patch_lat_deg),
            y_start - ((ny + 1) * patch_lat_deg),
            y_start - (ny * patch_lat_deg),
        ],
        **box_params,
    )


def add_patch_to_plot(
    ax: plt.Axes, patch_size: int, nx: int, ny: int, box_params: Dict[str, Any]
) -> None:
    """
    Adds a rectangular patch on a plot to highlight a specific area, assumes (0, 0) top-left.

    Parameters
    ----------
    ax : plt.Axes
        The matplotlib axes to which the patch will be added.
    patch_size : int
        Size of the patch along one side, assuming a square patch.
    nx : int
        x-coordinate multiplier to determine the starting x-position.
    ny : int
        y-coordinate multiplier to determine the starting y-position.
    box_params : Dict[str, Any]
        Parameters to be passed to the `plot` function to define the box properties.

    Returns
    -------
    None
    """
    x0 = nx * patch_size
    y0 = ny * patch_size

    ax.plot(
        [x0, x0, x0 + patch_size, x0 + patch_size, x0],
        [y0, y0 + patch_size, y0 + patch_size, y0, y0],
        **box_params,
    )


def plot_single_t_patch_from_stacked_patched_imerg_grid(
    patched_imerg_tile: Any,
    timeslice: int,
    plot_params: Dict[str, Any],
    figsize: Tuple[int, int] = (6, 6),
) -> None:
    """
    Plots a grid of patched IMERG tiles for a given timeslice.

    Parameters
    ----------
    patched_imerg_tile : Any
        The patched IMERG tile data. This can be a h5py._hl.dataset.Dataset, numpy array, xarray, or any similar data object.
    timeslice : int
        The timeslice to plot. If the given timeslice is too large, the last timeslice is plotted.
    plot_params : Dict[str, Any]
        A dictionary of parameters to pass to `imshow` for customizing the plot.

    Returns
    -------
    None
    """
    ts, rows, cols, _, _ = patched_imerg_tile.shape

    # Plot last timeslice if given timeslice is too large.
    timeslice = min(timeslice, ts - 1)

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            axs[r, c].imshow(patched_imerg_tile[timeslice, r, c, :, :], **plot_params)
            axs[r, c].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.025, hspace=0.025)
    plt.show()


def plot_single_patched_imerg_grid(
    patched_imerg_tile: Any,
    plot_params: Dict[str, Any],
    layout_params: Dict[str, Any] = {"wspace": 0.025, "hspace": 0.025},
    figsize: Tuple[int, int] = (6, 6),
) -> None:
    """
    Plots a grid of patched IMERG tiles.

    Parameters
    ----------
    patched_imerg_tile : Any
        The patched IMERG tile data. This can be a h5py._hl.dataset.Dataset, numpy array, xarray, or any similar data object.
    plot_params : Dict[str, Any]
        A dictionary of parameters to pass to `imshow` for customizing the plot.
    plot_params : Dict[str, Any]
        A dictionary of parameters to pass plt.subplots_adjust().

    Returns
    -------
    None
    """
    rows, cols, _, _ = patched_imerg_tile.shape
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            axs[r, c].imshow(patched_imerg_tile[r, c, :, :], **plot_params)
            axs[r, c].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(**layout_params)
    plt.show()


def plot_sequence(
    idx: Optional[int],
    data: np.ndarray,
    sequence_length: int,
    horizon: int,
    global_params: Dict[str, Any] = {},
    plot_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {"wspace": 0.1, "hspace": 0.1},
    figsize: Tuple[float, float] = (12, 3),
) -> None:
    """
    Plots a sequence of images in a grid layout with specified parameters.

    Parameters
    ----------
    idx : int or None, optional
        Index of the sequence to plot. If `None`, a random index is chosen.
    data : np.ndarray
        3D array of shape (N, H, W), where `N` is the number of sequences,
        `H` is the height, and `W` is the width of each image.
    sequence_length : int
        The length of the input sequence to plot.
    horizon : int
        The horizon length for the sequence. Used to determine the number of plots.
    global_params : dict, optional
        Dictionary of parameters to update `plt.rcParams` with, affecting global plot settings.
    plot_params : dict, optional
        Dictionary of parameters passed to `axs.imshow()` for each subplot.
    layout_params : dict, optional
        Dictionary of layout parameters to pass to `plt.subplots_adjust()`. Defaults to
        `{"wspace": 0.1, "hspace": 0.1}`.
    figsize : tuple of float, optional
        Size of the entire figure in inches. Defaults to (12, 3).

    Returns
    -------
    None
    """
    if idx is None:
        idx_to_plot = np.random.randint(0, data.shape[0])
    else:
        idx_to_plot = min(idx, data.shape[0] - 1)

    fig, axs = plt.subplots(nrows=2, ncols=sequence_length, figsize=figsize)
    plt.rcParams.update(global_params)
    for _row in range(2):  # inputs and target.
        for _col in range(sequence_length):
            idx = _col + (_row * sequence_length)
            if idx < (sequence_length + horizon):
                axs[_row, _col].imshow(data[idx_to_plot, idx, :, :], **plot_params)
                axs[_row, _col].set_title(f"t{idx - sequence_length}")
                axs[_row, _col].axis("off")
            else:
                axs[_row, _col].axis("off")

        plt.tight_layout()
        plt.subplots_adjust(**layout_params)


def plot_experiment_train_val_logs(
    config, val_metrics: List[str], figsizes: List[Dict[str, Any]]
) -> None:
    """
    Wrapper function to plot and save the model training/validation log information.

    Parameters
    ----------
    config : object
        Configuration object containing paths and settings for the experiment.
    val_metrics : list of str
        List of validation metrics to plot.
    figsizes : list of dict
        List of dictionaries specifying the figure sizes for different plots.

    Returns
    -------
    None
    """
    df = pd.read_csv(f"{config.log_dir}/metrics.csv")
    lr_col = [i for i in df.columns if "lr" in i][0]
    training_info = df[(df["train/loss"].notnull()) & (df[lr_col].isnull())]
    validation_info = df[(df["train/loss"].isnull()) & (df[lr_col].isnull())]
    # get name of training loss used.
    pattern = r".*\/([a-zA-Z0-9]+loss)"
    training_loss_name = [
        re.match(pattern, item).group(1) for item in df.columns.to_list() if re.match(pattern, item)
    ][0]

    # global plot params.
    plt.rcParams.update(
        {
            # "font.family": "Times New Roman",
            "font.size": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 7,
        }
    )

    # training + val loss per epoch.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsizes[0])
    ax.plot(
        training_info.epoch,
        training_info["train/loss"],
        color="C0",
        label=f"{training_loss_name}_train",
    )
    pattern = re.compile(rf"val/[^/]*ens_mems/{config.module.experiment_type}/avg/[^/]*loss")
    val_loss = [col for col in validation_info.columns if pattern.search(col)][0]
    ax.plot(
        training_info.epoch,
        validation_info[val_loss],
        color="C1",
        label=f"{training_loss_name}_val_ens_avg",
    )
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_xlabel("epochs")
    ax.set_ylabel("loss")
    ax.legend(loc="best")
    plt.savefig(f"{config.work_dir}/train_val_loss.png")

    # ensemble metrics.
    pattern = re.compile(
        f"val/\\d+ens_mems/{config.module.experiment_type}/(t\\d+|avg)/({'|'.join(val_metrics)})"
    )
    metrics_to_plot = {
        metric: [
            col
            for col in validation_info.columns
            if pattern.search(col) and metric == col.split("/")[-1]
        ]
        for metric in val_metrics
    }
    fig, axs = plt.subplots(nrows=len(metrics_to_plot), ncols=1, figsize=figsizes[1])
    for i, (k, v) in enumerate(metrics_to_plot.items()):
        for col in v:
            axs[i].plot(validation_info["epoch"], validation_info[col], label=col.split("/")[-2])
            axs[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            axs[i].set_xlabel("epochs")
            axs[i].set_ylabel(col.split("/")[-1])
            axs[i].legend(loc="best")
    axs[0].set_title("_".join(col.split("/")[:3]))
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    plt.savefig(f"{config.work_dir}/val_ensemble_metrics.png")

    # experiment type intermediates metrics.
    pattern = re.compile(rf"val/(t\d+|\d+h_avg)/{config.module.experiment_type}/")
    exp_cols = [col for col in validation_info.columns if pattern.search(col)]
    default_exp_metric = [i for i in exp_cols if i[-3:] == "mse"]
    custom_exp_metric = [i for i in exp_cols if i not in default_exp_metric]
    for e, cols in enumerate([default_exp_metric, custom_exp_metric]):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsizes[0])
        for col in cols:
            ax.plot(validation_info["epoch"], validation_info[col], label=col.split("/")[1])
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
            ax.set_xlabel("epochs")
            ax.set_ylabel(col.split("/")[-1])
            ax.legend(loc="best")
        ax.set_title("val_" + "_".join(col.split("/")[-2:]))
        plt.savefig(
            f"{config.work_dir}/val_{config.module.experiment_type}_{'default' if e == 0 else 'custom'}_metrics.png"
        )

    # save the results to a .csv.
    df_results = pd.merge(
        left=training_info[["epoch", "train/loss"]],
        right=validation_info.loc[:, ~validation_info.columns.isin(["train/loss", "step", lr_col])],
        on=["epoch"],
    )
    df_results[lr_col] = df.loc[:, df.columns.isin([lr_col])].dropna().values  # add learning rate to df.
    df_results.to_csv(f"{config.work_dir}/results.csv")


def plot_a_sequence(
    X,
    b,
    c: int = 0,
    global_params: Dict[str, Any] = {},
    plot_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {"wspace": 0.1, "hspace": 0.1},
    figsize: Tuple[float, float] = (12, 3),
) -> None:
    """
    Plot a sequence from X with dimensions (B, S, C, H, W).

    Parameters
    ----------
    X : torch.Tensor or np.ndarray
        The input data to plot, with shape (B, S, C, H, W).
    b : int
        Batch index to plot from X.
    c : int, optional
        Channel index to plot, by default 0.
    global_params : dict, optional
        Dictionary of parameters to update `plt.rcParams` with, affecting global plot settings.
    plot_params : dict, optional
        Dictionary of parameters passed to `imshow()` for each subplot.
    layout_params : dict, optional
        Dictionary of layout parameters to pass to `plt.subplots_adjust()`. Defaults to
        `{"wspace": 0.1, "hspace": 0.1}`.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(1, X.size(1), figsize=figsize)
    plt.rcParams.update(**global_params)
    for i in range(X.size(1)):
        axs[i].imshow(X[b, i, c, :, :], **plot_params)
        axs[i].set_title(f"$x_{i}$")
        axs[i].axis("off")
    plt.tight_layout()
    plt.subplots_adjust(**layout_params)


def plot_an_interpolated_sequence(
    x0,
    x_interp,
    xh,
    b,
    c: int = 0,
    plot_start_end: bool = True,
    global_params: Dict[str, Any] = {},
    plot_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {"wspace": 0.1, "hspace": 0.1},
    figsize: Tuple[float, float] = (12, 3),
) -> None:
    """
    Plot a sequence from x0, x_interp, and xh.

    Parameters
    ----------
    x0 : torch.Tensor or np.ndarray
        The initial frame, with shape (B, C, H, W).
    x_interp : torch.Tensor or np.ndarray
        The interpolated sequence, with shape (B, S, C, H, W).
    xh : torch.Tensor or np.ndarray
        The final frame, with shape (B, C, H, W).
    b : int
        Batch index to plot from x0, x_interp, and xh.
    plot_start_end : bool, optional
        Whether to plot the start (x0) and end (xh) frames, by default True.
    global_params : dict, optional
        Dictionary of parameters to update `plt.rcParams` with, affecting global plot settings.
    plot_params : dict, optional
        Dictionary of parameters passed to `imshow()` for each subplot.
    layout_params : dict, optional
        Dictionary of layout parameters to pass to `plt.subplots_adjust()`. Defaults to
        `{"wspace": 0.1, "hspace": 0.1}`.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(1, x_interp.size(1) + 2, figsize=figsize)
    plt.rcParams.update(**global_params)

    if plot_start_end:
        # plot x0 and xh.
        axs[0].imshow(x0[b, c, :, :], **plot_params)
        axs[0].set_title("$x_0$")
        axs[-1].imshow(xh[b, c, :, :], **plot_params)
        axs[-1].set_title("$x_h$")

    for i in range(x_interp.size(1)):
        axs[i + 1].imshow(x_interp[b, i, c, :, :], **plot_params)
        axs[i + 1].set_title(f"$x_{i+1}$")

    for ax in axs.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(**layout_params)


def plot_dyffusion_predictions(
    targets,
    predictions,
    c: int = 0,
    plot_params: Dict[str, Any] = {},
    global_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {"wspace": 0.1, "hspace": 0.1},
    figsize: Tuple[int, int] = (14, 6),
) -> None:
    """Wrapper plotting function to plot DYffusion predictions.

    Parameters
    ----------
    targets : torch.Tensor or np.ndarray
        The targets with shape (B, S, C, H, W).
    predictions : torch.Tensor or np.ndarray
        The predictions with shape (B, S, C, H, W).
    c : int
        The channel to plot.
    global_params : dict, optional
        Dictionary of parameters to update `plt.rcParams` with, affecting global plot settings.
    plot_params : dict, optional
        Dictionary of parameters passed to `imshow()` for each subplot.
    layout_params : dict, optional
        Dictionary of layout parameters to pass to `plt.subplots_adjust()`. Defaults to
        `{"wspace": 0.1, "hspace": 0.1}`.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(2, targets.size(0), figsize=figsize)
    plt.rcParams.update(global_params)
    # plot ground truth in the 1st row.
    for j in range(targets.size(0)):
        axs[0, j].imshow(targets[j, c, ...], **plot_params)
        axs[0, j].set_title(f"Ground Truth: x_t+{j}")

    # plot predictions.
    for e, (k, pred) in enumerate(predictions.items()):
        axs[1, e + 1].imshow(pred[0, c, :, :], **plot_params)
        axs[1, e + 1].set_title(f"{k}")

    for ax in axs.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(**layout_params)
