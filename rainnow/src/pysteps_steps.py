"""Module to implement STEPS probabilistic nowcasts using the PySteps implementation.

Followed tutorial: https://pysteps.readthedocs.io/en/stable/auto_examples/plot_steps_nowcast.html.
"""

from typing import List, Tuple, Union

import numpy as np
import torch
from pysteps import motion, nowcasts
from pysteps.utils import transformation


class PyStepsNowcastModel:
    """STEPS Probabilistic Nowcast Model wrapper for precipitation forecasting.

    Handles precipitation data in (S, C, H, W) format and produces ensemble nowcasts
    using the PySTEPS library implementation of STEPS.

    Note: following https://pysteps.readthedocs.io/en/stable/auto_examples/plot_steps_nowcast.html.
    """

    def __init__(
        self,
        input_precip_sequence: Union[np.ndarray, torch.Tensor, List],
        input_dims: Tuple[int, int, int] = (1, 128, 128),  # (C, H, W).
        horizon: int = 30,  # minutes.
        data_time_interval: int = 30,  # minutes.
        data_km_per_pixel_resolution: float = 10,
        num_ensemble: int = 1,
        transform_mm_h_to_dBR: bool = True,
    ):
        """Initialisation."""

        self.data = input_precip_sequence
        self.C, self.H, self.W = input_dims
        self.horizon = horizon
        self.time_interval = data_time_interval
        self.km_per_pixel = data_km_per_pixel_resolution
        self.n_ens_members = num_ensemble
        self.transform_mm_h_to_dBR = transform_mm_h_to_dBR
        self.channel_dim = None

        # checks.
        self.convert_inputs_to_numpy()
        self.check_input_dims()
        self._validate_data()
        self._validate_params()

        # motion field using LK (lucas-kanade) method.
        self.motion_field = motion.get_method("lucaskanade")

        self.V = None
        # nowcast method.
        self.nowcast_method = nowcasts.get_method("steps")

    @staticmethod
    def num_timesteps_to_nowcast(horizon, time_interval):
        """"""
        assert (
            horizon % time_interval == 0
        ), f"Prediction horizon is not a multiple of the data freq (min) {self.time_interval}."
        return int(horizon / time_interval)

    @property
    def input_dims(self) -> Tuple[int, int, int]:
        """Return input dimensions (C, H, W)."""
        return (self.C, self.H, self.W)

    @property
    def num_pred_timesteps(self) -> int:
        """Calculate number of prediction timesteps."""
        return self.num_timesteps_to_nowcast(self.horizon, self.time_interval)

    def convert_inputs_to_numpy(self):
        """Convert input data to numpy array format."""
        if isinstance(self.data, np.ndarray):
            return self

        if isinstance(self.data, torch.Tensor):
            self.data = self.data.detach().cpu().numpy()
        elif isinstance(self.data, list):
            self.data = np.asarray(self.data)
        else:
            raise TypeError(
                f"Input data must be numpy.ndarray, torch.Tensor, or list. Got {type(self.data)}"
            )
        return self

    def check_input_dims(self):
        """Validate input data dimensions match expected format."""
        expected_shape = (self.data.shape[0], *self.input_dims)
        if self.data.shape != expected_shape:
            raise ValueError(
                f"Data shape {self.data.shape} does not match expected shape {expected_shape}. "
                f"Input should be (Samples, Channels, Height, Width)"
            )
        self.channel_dim = 1

        return self

    def zero_prediction(self, R, zero_value):
        """Create zero-filled prediction array with appropriate shape."""
        out_shape = (self.n_ens_members, self.num_pred_timesteps, self.H, self.W)

        return np.full(out_shape, zero_value, dtype=R.dtype)

    def _validate_params(self):
        """Validate initialization parameters."""
        if self.horizon <= 0:
            raise ValueError(f"Horizon must be positive, got {self.horizon}")
        if self.time_interval <= 0:
            raise ValueError(f"Time interval must be positive, got {self.time_interval}")
        if self.km_per_pixel <= 0:
            raise ValueError(f"km_per_pixel must be positive, got {self.km_per_pixel}")
        if self.n_ens_members <= 0:
            raise ValueError(f"num_ensemble must be positive, got {self.n_ens_members}")

    def _validate_data(self):
        """Validate input data."""
        if np.any(~np.isfinite(self.data)):
            raise ValueError("Input data contains non-finite values")

    def nowcast(
        self,
        mm_h_precip_threshold: float = 0.1,
        dBR_precip_threshold: float = -15,
        n_cascade_levels: int = 6,
        noise_method: str = "nonparametric",
        vel_pert_method: str = "bps",
        mask_method: str = "incremental",
        extrap_method: str = "semilagrangian",
        domain: str = "spatial",
        return_output: bool = True,
        seed: int = 42,
    ):
        """Generate precipitation nowcast using PySteps STEPS implementation.

        Args:
            mm_h_precip_threshold: Threshold for precipitation in mm/h. Defaults to 0.1 mm/h which is very little rain.
            dBR_precip_threshold: Threshold for precipitation in dBR units.
            zero_value: Value to use for zero precipitation in dBR.
            n_cascade_levels: Number of cascade levels for the STEPS method.
            noise_method: Method for generating noise.
            vel_pert_method: Method for perturbing velocity field
            mask_method: Method for masking
            seed: Random seed for reproducibility

        Returns:
            np.ndarray: Nowcast predictions


        If noise_method and vel_pert_method are None, then STEPS ~ a deterministic forecast.
        """
        if self.transform_mm_h_to_dBR:
            # log-transform the data from mm/h to dBR.
            # The threshold of 0.1 mm/h sets the fill value to -15 dBR.
            R, _ = transformation.dB_transform(
                R=self.data.copy(),  # (t, h, w).
                metadata=None,
                threshold=mm_h_precip_threshold,  # in mm/h. Sets < threshold to 0.0.
                zerovalue=dBR_precip_threshold,  # what zeros are set to in the new units: R[zeros] = zerovalue.
            )

        else:
            R = self.data.copy()

        # index out the channel to only have HxW.
        R_channel = np.take(R, indices=0, axis=self.channel_dim).copy()

        # get precip motion field.
        # input needs to only be 1 channel: (t, h, w).
        self.V = self.motion_field(R_channel)
        # self.V = dense_lucaskanade(R_channel)

        try:
            R_f = self.nowcast_method(
                precip=R_channel,
                velocity=self.V,
                timesteps=self.num_pred_timesteps,
                n_ens_members=self.n_ens_members,
                n_cascade_levels=n_cascade_levels,  # using default: https://pysteps.readthedocs.io/en/latest/generated/pysteps.nowcasts.steps.forecast.html.
                precip_thr=dBR_precip_threshold,  # remember to be in dBR units if transformed.
                kmperpixel=self.km_per_pixel,
                timestep=self.time_interval,  # time step of the motion vectors (minutes).
                noise_method=noise_method,
                vel_pert_method=vel_pert_method,
                mask_method=mask_method,
                extrap_method=extrap_method,
                domain=domain,
                return_output=return_output,
                seed=seed,
            )

            # post-process to clean any NaNs.
            if R_f is not None:
                if self.transform_mm_h_to_dBR:
                    R_f = np.nan_to_num(R_f, nan=dBR_precip_threshold)  # give NaNs zero equivalent.
                    # transform back from dBr to mm/h.
                    R_f = transformation.dB_transform(
                        R=R_f,
                        metadata=None,
                        threshold=(
                            10 * np.log10(mm_h_precip_threshold)
                        ),  # convert the mm/h threshold to dBR threshold.
                        zerovalue=0.0,
                        inverse=True,
                    )[0]
                else:
                    # stay in mm/h.
                    R_f = np.nan_to_num(R_f, nan=0.0)

        except (ValueError, RuntimeError) as e:

            zero_error = (
                str(e).endswith("contains non-finite values")
                or str(e).startswith("zero-size array to reduction operation")
                or str(e).endswith("nonstationary AR(p) process")
            )
            if zero_error:
                # occasional PySTEPS errors that happen with little/no precip.
                # therefore returning all zeros makes sense.
                print(f"\n** Error **: {str(e)}")
                zero_value = 0.0  # current setup is to always return mm/h so zero_value is 0.0 mm/h.
                R_f = self.zero_prediction(R=R, zero_value=zero_value)
            else:
                raise

        # store the nowcast.
        return R_f
