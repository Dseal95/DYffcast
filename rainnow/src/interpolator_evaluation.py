"""Module that contains functions specifically for evaluating DYffusion interpolator models."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from rainnow.src.dyffusion.utilities.evaluation import (
    evaluate_ensemble_crps,
    evaluate_ensemble_spread_skill_ratio,
)
from rainnow.src.normalise import PreProcess
from rainnow.src.utilities.utils import enable_inference_dropout


def crps_metrics(
    ensemble_predictions: List[torch.Tensor], targets: torch.Tensor, mean_crps: bool = True
) -> List[float]:
    """
    Wrapper function to get CRPS metrics for the ensemble interpolations.

    Parameters
    ----------
    ensemble_predictions : List[torch.Tensor]
        List of ensemble prediction tensors.
    targets : torch.Tensor
        Target tensor.
    mean_crps : bool, optional
        Whether to compute mean CRPS over samples. Default is True.

    Returns
    -------
    List[float]
        List of CRPS scores for each time step.
    """
    crps = []
    for ens_preds, _targets in zip(ensemble_predictions, targets):
        _crps_score = evaluate_ensemble_crps(
            ensemble_predictions=ens_preds, targets=_targets, mean_over_samples=mean_crps
        )
        crps.append(_crps_score)
    return crps


def ssr_metrics(ensemble_predictions: List[torch.Tensor], targets: torch.Tensor) -> List[float]:
    """
    Wrapper function to get SSR metrics for the ensemble interpolations.

    Parameters
    ----------
    ensemble_predictions : List[torch.Tensor]
        List of ensemble prediction tensors.
    targets : torch.Tensor
        Target tensor.

    Returns
    -------
    List[float]
        List of SSR scores for each time step.
    """
    ssr = []
    for ens_preds, _targets in zip(ensemble_predictions, targets):
        _ssr_score = evaluate_ensemble_spread_skill_ratio(
            ensemble_predictions=ens_preds, targets=_targets
        )
        ssr.append(float(_ssr_score))
    return ssr


def eval_trained_interpolators(
    X: torch.Tensor,
    loaded_models: Dict[str, Any],
    ckpt_dict: Dict[str, str],
    pprocessor_obj: PreProcess,
    batch_num: int,
    channel_to_plot: int,
    plot_params: Dict[str, Any] = {},
    global_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {},
    ylabel_params: Dict[str, Any] = {},
    figsize: Tuple[int, int] = (14, 2),
) -> None:
    """
    Wrapper function to compare trained interpolators.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor containing the data sequence.
    loaded_models : Dict[str, Any]
        Dictionary of loaded models to evaluate.
    ckpt_dict : Dict[str, str]
        Dictionary mapping model IDs to checkpoint descriptions.
    pprocessor_obj : PreProcess
        Preprocessor object for reversing data processing.
    batch_num : int
        Batch number to evaluate.
    channel_to_plot : int
        Channel index to plot.
    plot_params : Dict[str, Any]
        Dictionary of plot parameters.
    global_params : Dict[str, Any]
        Dictionary of global matplotlib parameters.
    layout_params : Dict[str, Any]
        Dictionary of plotting parameters for plt.subplots_adjust.
    ylabel_params : Dict[str, Any]
        Dictionary of parameters for y-axis labels.
    figsize : Tuple[int, int], optional
        Figure size. Default is (14, 2).

    Returns
    -------
    None
    """
    # get x0, interpolated values and xh from X.
    x0 = X[batch_num, 0, :, :, :]
    xh = X[batch_num, -1, :, :, :]
    x_interp = X[batch_num, 1:-1, :, :, :]
    seq_len = X.size(1)

    # get raw data back from pre-processed X.
    X_reversed = pprocessor_obj.reverse_processing(X)

    # modify figsize by +3*model_ckpt_ids-1. 3 seems like a good height per plot.
    figsize = (figsize[0], figsize[1] + (3 * (len(loaded_models) - 1)))
    fig, axs = plt.subplots(nrows=len(loaded_models) + 1, ncols=seq_len, figsize=figsize)
    plt.rcParams.update(**global_params)

    # plot ground truth sequence in 1st row.
    for i in range(seq_len):
        axs[0, i].imshow(X_reversed[batch_num, i, channel_to_plot, :, :], **plot_params)
        # axs[0, i].set_title(f"$t_{i}$" if i != seq_len - 1 else "$t_{h}$")
        axs[0, i].set_title(f"$t+{i*30*2}min$" if i != 0 else "t")

    for k in range(seq_len - 2):
        # add interpolated to 2nd row.
        axs[1, k + 1].set_title(f"Interpolated $t+{(k+1) * 30 * 2}min$")

    for j, (_id, _model) in enumerate(loaded_models.items()):
        print(f"** running inference for model ID: {_id}: {ckpt_dict[_id]} **")
        interpolated_ts = []
        for i in range(x_interp.size(0)):
            # interpolate ts.
            t = i + 1
            t_interp = _model(
                torch.concat([x0, xh]).unsqueeze(0), time=torch.tensor([t], dtype=torch.long)
            )

            # make this more robust.
            if isinstance(_model.final_conv[1], nn.Tanh):
                # print(f"[-1, 1] --> [0, 1] for model {_id}: {ckpt_dict[_id]}.")
                # need to reverse normed ouput from [-1, 1] to [0, 1] prior to preprocessing.
                t_interp = (t_interp + 1) / 2

            # reverse the data pre-processing to get RAW data.
            t_interp_reversed = pprocessor_obj.reverse_processing(t_interp)
            interpolated_ts.append(t_interp_reversed.squeeze(0).detach().cpu())

        # plot interpolated values.
        for e, img in enumerate(interpolated_ts):
            axs[j + 1, e + 1].imshow(img[channel_to_plot, :, :], **plot_params)

    # add labels to the left hand axes.
    axs[0, 0].set_ylabel("ground truth".upper(), rotation=90, **ylabel_params)
    axs[0, 0].yaxis.set_label_coords(-0.05, 1)
    for j, (k, _) in enumerate(loaded_models.items()):
        title = f"{ckpt_dict[k]}"
        axs[j + 1, 1].set_ylabel(title, rotation=90, **ylabel_params)
        axs[j + 1, 1].yaxis.set_label_coords(-0.05, 1)

    # remove the box around each subplot.
    for j in range(len(loaded_models) + 1):
        for i in range(seq_len):
            axs[j, i].spines["top"].set_visible(False)
            axs[j, i].spines["right"].set_visible(False)
            axs[j, i].spines["bottom"].set_visible(False)
            axs[j, i].spines["left"].set_visible(False)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(**layout_params)


def eval_single_trained_interpolator(
    X: torch.Tensor,
    model: Any,
    pprocessor_obj: PreProcess,
    channel_to_plot: int,
    cmap: str,
    plot_params: Dict[str, Any] = {},
    global_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {},
    ylabel_params: Dict[str, Any] = {},
    figsize: Tuple[int, int] = (13, 2),
) -> None:
    """
    Wrapper function to visually evaluate an interpolator on a batch of data.

    Parameters
    ----------
    X : torch.Tensor
        Input tensor containing the data sequence.
    model : Any
        The interpolator model to evaluate.
    pprocessor_obj : PreProcess
        Preprocessor object for reversing data processing.
    channel_to_plot : int
        Channel index to plot.
    cmap : str
        Colormap to use for plotting.
    plot_params : Dict[str, Any]
        Dictionary of plot parameters.
    global_params : Dict[str, Any]
        Dictionary of global matplotlib parameters.
    layout_params : Dict[str, Any]
        Dictionary of plotting parameters for plt.subplots_adjust.
    ylabel_params : Dict[str, Any]
        Dictionary of parameters for y-axis labels.
    figsize : Tuple[int, int], optional
        Figure size. Default is (13, 2).

    Returns
    -------
    None
    """
    # get x0, interpolated values and xh from X.
    x0 = X[:, 0, :, :, :]
    xh = X[:, -1, :, :, :]
    seq_len = X.size(1)
    bs = X.size(0)
    # get raw data back from pre-processed X.
    X_reversed = pprocessor_obj.reverse_processing(X)

    # modify figsize by +3*batchsize. 3 seems like a good height per plot.
    figsize = (figsize[0], figsize[1] + (3 * bs))
    fig, axs = plt.subplots(nrows=2 * (bs), ncols=seq_len, figsize=figsize)
    plt.rcParams.update(**global_params)

    # ground truth titles in 1st row.
    for i in range(seq_len):
        axs[0, i].set_title(f"$t_{i}$" if i != seq_len - 1 else "$t_{h}$")

    # plot all the rela sequences at 1 axes apart.
    for m in [1, 2]:
        for b in range(bs):
            for j in range(seq_len):
                if m % 2 == 0:
                    # plot the full ground truth seq.
                    xax = 0 if b == 0 else (b * m)
                    axs[xax, j].imshow(X_reversed[b, j, channel_to_plot, :, :], **plot_params)
                else:
                    xax = (2 * b) + m
                    if j == 0:
                        axs[xax, j].imshow(
                            X_reversed[b, j, channel_to_plot, :, :], **plot_params
                        )  # plot t_0
                    elif j == (seq_len - 1):
                        axs[xax, j].imshow(
                            X_reversed[b, j, channel_to_plot, :, :], **plot_params
                        )  # plot t_h
                    else:
                        # interpolation at time, t.
                        t_interp = model(
                            torch.concat([x0[b, :, :, :], xh[b, :, :, :]], axis=0).unsqueeze(0),
                            time=torch.tensor([j], dtype=torch.long),
                        )

                        # TODO: make this more robust.
                        if str(model.final_conv[1]) == "Tanh()":
                            # print(f"[-1, 1] --> [0, 1] for model.")
                            # need to reverse normed ouput from [-1, 1] to [0, 1] prior to preprocessing.
                            t_interp = (t_interp + 1) / 2

                        t_interp_reversed = pprocessor_obj.reverse_processing(t_interp)
                        axs[xax, j].imshow(
                            t_interp_reversed[0, channel_to_plot, :, :].detach().cpu(), **plot_params
                        )

    # add labels to the left hand axes.
    ts = [f"$t_{i+1}$" for i in range(seq_len - 2)]
    for j in range(2 * bs):
        if j % 2 == 0:
            title = "ground truth".upper()
        else:
            title = f"{'Interpolate'.upper()} {' '.join(ts)}"
        axs[j, 0].set_ylabel(title, rotation=90, **ylabel_params)
        axs[j, 0].yaxis.set_label_coords(-0.1, 0.85)

    # remove the box around each subplot
    for j in range(2 * bs):
        for i in range(seq_len):
            axs[j, i].spines["top"].set_visible(False)
            axs[j, i].spines["right"].set_visible(False)
            axs[j, i].spines["bottom"].set_visible(False)
            axs[j, i].spines["left"].set_visible(False)
            axs[j, i].set_xticks([])
            axs[j, i].set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(**layout_params)


def get_an_interpolated_ensemble_batch(
    inputs: torch.Tensor,
    pprocessor_obj: PreProcess,
    model: nn.Module,
    batch_num: int,
    num_ens_preds: int,
):
    """Wrapper function to get an ensemble of interpolations from an interpolator model."""
    # get x0, interpolated values and xh from X.
    x0 = inputs[batch_num, 0, :, :, :]
    xh = inputs[batch_num, -1, :, :, :]
    x_interp = inputs[batch_num, 1:-1, :, :, :]
    # get raw data back from pre-processed X for given batch.
    X_raw_input = inputs[batch_num, ...]
    X_reversed = pprocessor_obj.reverse_processing(X_raw_input)

    # enable monte-carlo randomness in dropout layers during inference.
    enable_inference_dropout(model)
    assert all(
        layer.training
        for layer in model.modules()
        if isinstance(layer, (torch.nn.Dropout, torch.nn.Dropout2d))
    ), f"Not all dropout layers are in training mode for model."

    # interpolate ts.
    interpolated_ens_ts, interpolated_ens_ts_reversed = [], []
    for i in range(x_interp.size(0)):
        t = i + 1
        t_interp = model(
            torch.concat([x0, xh]).unsqueeze(0).expand(num_ens_preds, -1, -1, -1),
            time=torch.tensor([t], dtype=torch.long),
        )
        if isinstance(model.final_conv[1], nn.Tanh):
            # need to reverse normed ouput from [-1, 1] to [0, 1] prior to preprocessing.
            t_interp = (t_interp + 1) / 2

        # reverse the data pre-processing to get RAW data.
        t_interp_reversed = pprocessor_obj.reverse_processing(t_interp)

        interpolated_ens_ts.append(t_interp.detach().cpu())
        interpolated_ens_ts_reversed.append(t_interp_reversed.detach().cpu())

    return X_raw_input, X_reversed, interpolated_ens_ts, interpolated_ens_ts_reversed


def plot_ens_eval_single_t_interpolation(
    inputs: torch.Tensor,
    interpolated_ens_ts: List[torch.Tensor],
    t_to_plot: int,
    ckpt_desc: str = "",
    non_zero_thd: float = 0.1,
    channel_to_plot: int = 0,
    plot_params: Dict[str, Any] = {},
    global_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {},
    figsize: Tuple = (16, 6),
    kde_linewidth: float = 1,
    plt_ens_dist: bool = True,
    crps_scores: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Plot the ground truth interpolated value, along with the ensemble of interpolations and their distributions (raw and non-zero).

    Parameters
    ----------
    inputs : torch.Tensor
        Input tensor containing ground truth data.
    interpolated_ens_ts : List[torch.Tensor]
        List of interpolated ensemble tensors.
    t_to_plot : int
        Time step to plot.
    ckpt_desc : str, optional
        Description of the checkpoint. Default is an empty string.
    non_zero_thd : float, optional
        Threshold for non-zero values. Default is 0.1.
    channel_to_plot : int, optional
        Channel index to plot. Default is 0.
    plot_params : Dict[str, Any]
        Dictionary of plot parameters.
    global_params : Dict, optional
        Dictionary of global matplotlib parameters.
    layout_params : Dict, optional
        Dictionary of plotting parameters for plt.subplots_adjust.
    figsize : Tuple, optional
        Figure size. Default is (16, 6).
    kde_linewidth : float, optional
        Line width for KDE plots. Default is 1.
    plt_ens_dist : bool, optional
        Whether to plot ensemble distributions. Default is True.
    crps_scores : Optional[Tuple[float, float]], optional
        CRPS scores (scaled, raw). Default is None.

    Returns
    -------
    None
    """
    num_ens_preds = interpolated_ens_ts[0].size(0)  # get it from 1st elem. Assume all same.
    assert (t_to_plot > 0) and (
        t_to_plot < len(interpolated_ens_ts) + 1
    )  # interp t needs to be > 0 and < seq_len
    # get ensemble interpolated values. Assert allows t_to_plot-1.
    ens_interp_ts = interpolated_ens_ts[t_to_plot - 1]

    nrows = 2  # images, dist raw, dist non-zero
    fig, axs = plt.subplots(nrows=nrows, ncols=num_ens_preds + 1 + 1, figsize=figsize)
    fig.suptitle(
        f"Interpolated t={t_to_plot} (out of {len(ens_interp_ts)}) for model: {ckpt_desc} w/ num ens = {num_ens_preds} | ** CRPS Scores (scaled, raw, mu(!=0)) = {crps_scores if crps_scores is not None else 'N/A'}",
        fontsize=10,
        fontweight="bold",
    )
    plt.rcParams.update(**global_params)

    # plot ground truth sequence + distributions in 1st row, 1st col.
    gt = inputs[t_to_plot, channel_to_plot, :, :]
    axs[0, 0].imshow(gt, **plot_params)
    axs[0, 0].set_title(f"$t_{t_to_plot}$")
    axs[0, 0].axis("off")
    sns.kdeplot(gt.flatten(), ax=axs[1, 0], label=f"t_{t_to_plot}", alpha=0.6, linewidth=kde_linewidth)
    sns.kdeplot(
        gt[gt > non_zero_thd].flatten(),
        ax=axs[1, 0],
        label=f"t_{t_to_plot} non-zero",
        alpha=0.8,
        linewidth=kde_linewidth,
    )

    for j in range(1, num_ens_preds + 1):
        _interp_t = ens_interp_ts[j - 1, channel_to_plot, :, :]
        axs[0, j].imshow(_interp_t, **plot_params)
        axs[0, j].set_title(f"$t_{t_to_plot} ens_{j}$")
        axs[0, j].axis("off")
        # plot distributions.
        sns.kdeplot(
            _interp_t.flatten(), ax=axs[1, j], label=f"ens_{j}", alpha=0.6, linewidth=kde_linewidth
        )
        sns.kdeplot(
            _interp_t[_interp_t > non_zero_thd].flatten(),
            ax=axs[1, j],
            label=f"ens_{j} non-zero",
            alpha=0.8,
            linewidth=kde_linewidth,
        )
        # Remove y-label for non-leftmost axes
        axs[1, j].set_ylabel("")

    # plot mean pred.
    mu_ens = torch.mean(ens_interp_ts, dim=0)
    axs[0, num_ens_preds + 1].imshow(mu_ens[0, :, :], **plot_params)
    axs[0, num_ens_preds + 1].set_title(f"$t_{t_to_plot} mu$")
    axs[0, num_ens_preds + 1].axis("off")
    sns.kdeplot(
        mu_ens.flatten(), ax=axs[1, num_ens_preds + 1], label="mu", alpha=0.5, linewidth=kde_linewidth
    )
    sns.kdeplot(
        gt.flatten(), ax=axs[1, num_ens_preds + 1], label="gt", alpha=0.6, linewidth=kde_linewidth
    )
    sns.kdeplot(
        mu_ens[mu_ens > non_zero_thd].flatten(),
        ax=axs[1, num_ens_preds + 1],
        label="mu, non-zero",
        linewidth=kde_linewidth,
    )
    sns.kdeplot(
        gt[gt > non_zero_thd].flatten(),
        ax=axs[1, num_ens_preds + 1],
        label="gt, non-zero",
        linewidth=kde_linewidth,
    )

    # axes formatting.
    axs[1, num_ens_preds + 1].set_ylabel("")
    for n, ax in enumerate(axs.flatten()):
        if n >= num_ens_preds + 2:
            ax.legend(loc="best")

    plt.tight_layout()
    plt.subplots_adjust(**layout_params)

    if plt_ens_dist:
        # plot showing all ens distributions together.
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        plt.rcParams.update(**global_params)
        for j in range(num_ens_preds):
            _interp_t = ens_interp_ts[j, channel_to_plot, :, :]
            sns.kdeplot(_interp_t.flatten(), ax=axs[0], label=f"ens_{j}", linewidth=kde_linewidth)
            sns.kdeplot(
                _interp_t[_interp_t > non_zero_thd].flatten(),
                ax=axs[1],
                label=f"ens_{j} non-zero",
                linewidth=kde_linewidth,
            )

        axs[0].legend(loc="best")
        axs[1].legend(loc="best")
        axs[1].set_ylabel("")

        plt.tight_layout()
        plt.subplots_adjust(**layout_params)


def plot_interpolated_batch(
    targets: torch.Tensor,
    preds: torch.Tensor,
    c: int = 0,
    figsize: Tuple[int, int] = (16, 4),
    plot_title: str = "",
    plot_params: Dict[str, Any] = {},
    global_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {},
) -> None:
    """
    Wrapper function for plotting an interpolated batch against targets.

    Parameters
    ----------
    targets : torch.Tensor
        Target tensor containing ground truth data.
    preds : torch.Tensor
        Predictions tensor containing interpolated values.
    c : int, optional
        Channel index to plot. Default is 0.
    figsize : Tuple[int, int], optional
        Figure size. Default is (16, 4).
    plot_title : str, optional
        Title for the plot. Default is an empty string.
    plot_params : Dict, optional
        Dictionary of plotting parameters for imshow.
    global_params : Dict, optional
        Dictionary of global matplotlib parameters.
    layout_params : Dict, optional
        Dictionary of layout parameters for plt.subplots_adjust.

    Returns
    -------
    None
    """
    fig, axs = plt.subplots(2, targets.size(0), figsize=figsize)
    plt.rcParams.update(global_params)
    # plot ground truth in the 1st row.
    for j in range(targets.size(0)):
        axs[0, j].imshow(targets[j, c, ...], **plot_params)
        if plot_title:
            axs[0, j].set_title(f"Ground Truth: x_t+{j}", fontsize=8)
        axs[0, j].axis("off")
    # plot interpolated values between x0 and xh.
    for k in range(targets.size(0)):
        if k != 0 and k != targets.size(0) - 1:
            axs[1, k].imshow(preds[k - 1, c, ...], **plot_params)
            if plot_title:
                axs[1, k].set_title(f"Interpolated x_t+{k}", fontsize=8)
        axs[1, k].axis("off")

    plt.tight_layout()
    plt.subplots_adjust(**layout_params)


def plot_interpolation_ensemble(
    ensemble_interpolations: List[torch.Tensor],
    targets: torch.Tensor,
    t_to_plot: int,
    model_desc: str,
    plot_ensemble: bool = False,
    plot_ens_zero: bool = False,
    crps: List[float] = [],
    ssr: List[float] = [],
    non_zero_thd: float = 0.1,
    channel_to_plot: int = 0,
    plot_params: Dict[str, Any] = {},
    global_params: Dict[str, Any] = {},
    layout_params: Dict[str, Any] = {},
    kde_linewidth: float = 1,
    figsize: Tuple[int, int] = (16, 4),
) -> None:
    """
    Plot interpolation ensemble results.

    Parameters
    ----------
    ensemble_interpolations : List[torch.Tensor]
        List of ensemble interpolation tensors.
    targets : torch.Tensor
        Target tensor.
    t_to_plot : int
        Time step to plot.
    model_desc : str
        Description of the model.
    plot_ensemble : bool, optional
        Whether to plot the full ensemble. Default is False.
    plot_ens_zero : bool, optional
        Whether to plot ensemble distributions including zeros. Default is False.
    crps : List[float], optional
        List of Continuous Ranked Probability Score values.
    ssr : List[float], optional
        List of Spread-Skill Ratio values.
    non_zero_thd : float, optional
        Threshold for non-zero values. Default is 0.1.
    channel_to_plot : int, optional
        Channel index to plot. Default is 0.
    plot_params : Dict, optional
        Dictionary of plotting parameters.
    global_params : Dict, optional
        Dictionary of global matplotlib parameters.
    layout_params : Dict, optional
        Dictionary of layout parameters for plt.subplots_adjust.
    kde_linewidth : float, optional
        Line width for KDE plots. Default is 1.
    figsize : Tuple[int, int], optional
        Figure size. Default is (16, 4).

    Returns
    -------
    None
    """
    num_ens_preds = ensemble_interpolations[0].size(0)  # get it from 1st elem. Assume all same.
    ens_interp = ensemble_interpolations[t_to_plot]
    ncols = 4  # images, dist raw, dist non-zero
    fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=figsize)
    plt.rcParams.update(**global_params)
    fig.suptitle(
        f"** {num_ens_preds} ensemble ** Interpolated t={t_to_plot+1} (of {len(ensemble_interpolations)}) | model: {model_desc} | CRPS / SSR (processed, raw, non-zero):  {[round(i,2) for i in crps]} / {[round(i,2) for i in ssr]}",
        fontsize=10,
        fontweight="bold",
    )

    # plot ground truth sequence + distributions in 1st row, 1st col.
    gt = targets[t_to_plot, channel_to_plot, :, :]
    axs[0].imshow(gt, **plot_params)
    axs[0].set_title(f"Ground Truth $t_{t_to_plot}$")
    axs[0].axis("off")

    # plot mean pred.
    mu_ens = torch.mean(ens_interp, dim=0)
    axs[1].imshow(mu_ens[0, :, :], **plot_params)
    axs[1].set_title(f"$t_{t_to_plot} mu$")
    axs[1].axis("off")

    # meam versus pred prediction.
    sns.kdeplot(mu_ens.flatten(), ax=axs[2], label="mu", alpha=0.5, linewidth=kde_linewidth)
    sns.kdeplot(gt.flatten(), ax=axs[2], label="gt", alpha=0.6, linewidth=kde_linewidth)
    sns.kdeplot(
        mu_ens[mu_ens > non_zero_thd].flatten(), ax=axs[2], label="mu, non-zero", linewidth=kde_linewidth
    )
    sns.kdeplot(
        gt[gt > non_zero_thd].flatten(), ax=axs[2], label="gt, non-zero", linewidth=kde_linewidth
    )
    axs[2].legend(loc="best")

    # plot ens distributions
    plt.rcParams.update(**global_params)
    for j in range(num_ens_preds):
        _interp_t = ens_interp[j, channel_to_plot, :, :]
        sns.kdeplot(
            _interp_t[_interp_t > non_zero_thd].flatten(),
            ax=axs[3],
            label=f"ens_{j} non-zero",
            linewidth=kde_linewidth,
        )
        if plot_ens_zero:
            sns.kdeplot(
                _interp_t.flatten(), ax=axs[3], label=f"ens_{j}", linewidth=kde_linewidth, alpha=0.6
            )
        axs[3].legend(loc="best")
        axs[3].legend(loc="best")
        axs[3].set_ylabel("")
    plt.tight_layout()
    plt.subplots_adjust(**layout_params)

    if plot_ensemble:
        # get sufficient rows, columns. Use num_cols = 6 as a base.
        base_col = 6
        total_rows = (ensemble_interpolations[0].size(0) // base_col) + 1  # keep +1 to access axs[i, j].
        fig, axs = plt.subplots(total_rows, base_col, figsize=figsize)
        plt.rcParams.update(**global_params)
        k = 0
        for i in range(total_rows):
            for j in range(base_col):
                if k < ensemble_interpolations[0].size(0):
                    _interp_t = ens_interp[k, channel_to_plot, :, :]
                    axs[i, j].imshow(_interp_t, **plot_params)
                    axs[i, j].set_title(f"$t_{t_to_plot} ens_{k}$")
                axs[i, j].axis("off")
                k += 1
        plt.tight_layout()
        plt.subplots_adjust(**layout_params)
