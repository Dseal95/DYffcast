"""Code adapted from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/utilities/evaluation.py."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import xarray as xr
import xskillscore as xs


def evaluate_ensemble_prediction(
    predictions: np.ndarray,
    targets: np.ndarray,
    training_criterion: torch.nn.Module,
    ensemble_dim: int = 0,
    also_per_member_metrics: bool = False,
    mean_over_samples: bool = True,
    device: str = "cpu",
):
    """
    Evaluate the predictions of an ensemble of models.

    Parameters
    ----------
    predictions : np.ndarray
        The predictions of the ensemble, with shape `(n_models, n_samples, ...)`.
    targets : np.ndarray
        The ground truth targets, with shape `(n_samples, ...)`.
    training_criterion : torch.nn.Module
        The loss function used during training.
    ensemble_dim : int, optional
        The dimension of the ensemble within the predictions array. Default is `0`.
    also_per_member_metrics : bool, optional
        If `True`, compute metrics for each individual model in the ensemble. Default is `False`.
    mean_over_samples : bool, optional
        If `True`, compute the metrics by averaging over the samples dimension. Default is `True`.
    device : str, optional
        The device to perform computation on, e.g., `"cpu"` or `"cuda"`. Default is `"cpu"`.

    Returns
    -------
    dict
        A dictionary containing the evaluation metrics, possibly including per-member metrics if
        `also_per_member_metrics` is `True`.
    """
    assert (
        predictions.shape[1] == targets.shape[ensemble_dim]
    ), f"predictions.shape[1] ({predictions.shape[1]}) != targets.shape[0] ({targets.shape[ensemble_dim]})"
    # shape could be: preds: (10, 730, 3, 60, 60), targets: (730, 3, 60, 60), or (5, 64, 12), (64, 12)

    # if channel dimension is missing, add it
    if len(predictions.shape) == 3:
        predictions = predictions[:, :, np.newaxis]
    if len(targets.shape) == 2:
        targets = targets[:, np.newaxis]

    # get mean pred ('ensemble').
    mean_preds = predictions.mean(axis=ensemble_dim)
    # compute default mse for the mean pred.
    mean_dims = tuple(range(mean_preds.ndim)) if mean_over_samples else tuple(range(1, mean_preds.ndim))
    mse_ensemble = np.mean((mean_preds - targets) ** 2, axis=mean_dims)  # shape: () or (n_samples,)
    rmse_ensemble = np.sqrt(mse_ensemble)
    mses = {"mse": mse_ensemble}

    # compute the 'training loss' metric for mean pred.
    _train_loss_name = training_criterion.__class__.__name__.lower()
    val_tloss = training_criterion(
        torch.from_numpy(mean_preds).to(device), torch.from_numpy(targets).to(device)
    ).item()

    # compute CRPS for ensemble.
    crps = evaluate_ensemble_crps(predictions, targets, mean_over_samples=mean_over_samples)

    # compute the ssr score (spread of the ensemble) for ensemble.
    spread_skill_ratio = evaluate_ensemble_spread_skill_ratio(
        predictions, targets, skill_metric=rmse_ensemble, mean_dims=mean_dims
    )

    # get mse per ensemble member.
    if also_per_member_metrics:
        diff = predictions - targets
        assert np.allclose(diff[0], predictions[ensemble_dim] - targets)
        mses["mse_per_mem"] = np.mean(diff**2, axis=tuple(range(1, predictions.ndim)))
        mses["mse_per_mem_mean"] = np.mean(mses["mse_per_mem"])

    ens_metrics = {
        "ssr": spread_skill_ratio,
        "crps": crps,
        _train_loss_name: val_tloss,
        **mses,
    }

    return ens_metrics


def evaluate_ensemble_crps(
    ensemble_predictions: np.ndarray,
    targets: np.ndarray,
    member_dim: str = "member",
    mean_over_samples: bool = True,
) -> float | np.ndarray:
    """
    Compute the Continuous Ranked Probability Score (CRPS) for an ensemble of predictions.

    This function is a wrapper around the xarray implementation of CRPS.

    Parameters
    ----------
    ensemble_predictions : np.ndarray
        The predictions of the ensemble, with shape `(n_models, n_samples, ...)`.
    targets : np.ndarray
        The ground truth targets, with shape `(n_samples, ...)`.
    member_dim : str, optional
        The name of the dimension representing the ensemble members in the predictions array. Default is `"member"`.
    mean_over_samples : bool, optional
        If `True`, the CRPS is averaged over the samples dimension. If `False`, the CRPS is computed for each sample
        individually. Default is `True`.

    Returns
    -------
    float or np.ndarray
        The CRPS value, averaged over samples if `mean_over_samples` is `True`. If `mean_over_samples` is `False`,
        returns an array of CRPS values for each sample.
    """
    dummy_dims = [f"dummy_dim_{i}" for i in range(targets.ndim - 1)]
    preds_da = xr.DataArray(ensemble_predictions, dims=[member_dim, "sample"] + dummy_dims)
    targets_da = xr.DataArray(targets, dims=["sample"] + dummy_dims)
    mean_dims = ["sample"] + dummy_dims if mean_over_samples else dummy_dims
    crps = xs.crps_ensemble(
        observations=targets_da,
        forecasts=preds_da,
        member_dim=member_dim,
        dim=mean_dims,
    ).values  # shape: ()
    return float(crps) if mean_over_samples else crps


def evaluate_ensemble_spread_skill_ratio(
    ensemble_predictions: np.ndarray,
    targets: np.ndarray,
    skill_metric: float = None,
    mean_dims=None,
) -> float:
    """
    Compute the spread-skill ratio (SSR) of an ensemble of predictions.

    The SSR is defined as the ratio of the ensemble spread to the ensemble skill.
    The ensemble spread is calculated as the standard deviation across the ensemble members.

    Parameters
    ----------
    ensemble_predictions : np.ndarray
        The predictions of the ensemble, with shape `(n_models, n_samples, ...)`.
    targets : np.ndarray
        The ground truth targets, with shape `(n_samples, ...)`.
    skill_metric : float, optional
        The skill metric to use for calculating the ensemble skill. If `None`, the Root Mean Square Error (RMSE) is used.
        Default is `None`.
    mean_dims : tuple, optional
        The dimensions over which to compute the mean for both the spread and skill calculations. If `None`,
        the mean is computed over all dimensions. Default is `None`.

    Returns
    -------
    float
        The spread-skill ratio (SSR) of the ensemble predictions.
    """
    variance = np.var(ensemble_predictions, axis=0).mean(axis=mean_dims)
    spread = np.sqrt(variance)

    if skill_metric is None:
        mse = evaluate_ensemble_mse(ensemble_predictions, targets, mean_dims=mean_dims)
        skill_metric = np.sqrt(mse)

    spread_skill_ratio = spread / skill_metric
    return spread_skill_ratio


def evaluate_ensemble_nll(
    mean_predictions: np.ndarray,
    var_predictions: np.ndarray,
    targets: np.ndarray,
    mean_dims=None,
) -> float:
    """
    Compute the negative log-likelihood of an ensemble of predictions.
    """
    nll = 0.5 * np.log(2 * np.pi * var_predictions) + (targets - mean_predictions) ** 2 / (
        2 * var_predictions
    )
    return nll.mean(axis=mean_dims)


def evaluate_ensemble_mse(
    ensemble_predictions: np.ndarray, targets: np.ndarray, mean_dims=None
) -> float:
    mean_preds = ensemble_predictions.mean(axis=0)
    mse = np.mean((mean_preds - targets) ** 2, axis=mean_dims)
    return mse


def evaluate_ensemble_corr(ensemble_predictions: np.ndarray, targets: np.ndarray) -> float:
    mean_preds = ensemble_predictions.mean(axis=0)
    corr = np.corrcoef(mean_preds.reshape(1, -1), targets.reshape(1, -1), rowvar=False)[0, 1]
    return float(corr)


def evaluate_ensemble_prediction_for_varying_members(predictions: np.ndarray, targets: np.ndarray):
    n_members, n_samples = predictions.shape[:2]
    results = defaultdict(list)
    for n in range(1, n_members + 1):
        results_n = evaluate_ensemble_prediction(predictions[:n], targets)
        # for each result, only keep the values if they are scalars and add them to the list
        for k, v in results_n.items():
            if np.isscalar(v):
                results[k] += [v]
            elif n == n_members:
                results[k] = v
    return results
