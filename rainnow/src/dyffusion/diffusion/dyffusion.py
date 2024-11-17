"""Code adapted from: https://github.com/Rose-STL-Lab/dyffusion/blob/main/src/diffusion/dyffusion.py."""

import math
from abc import abstractmethod
from contextlib import ExitStack
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from rainnow.src.dyffusion.diffusion._base_diffusion import BaseDiffusion
from rainnow.src.utilities.loading import (
    instantiate_interpolator_experiment,
    load_model_checkpoint_config,
    load_model_checkpoint_for_eval,
)
from rainnow.src.utilities.utils import (
    freeze_model,
    generate_data_driven_gaussian_noise,
    get_logger,
    raise_error_if_invalid_value,
    transform_0_1_to_minus1_1,
    transform_minus1_1_to_0_1,
)

log = get_logger(name=__name__)


class BaseDYffusion(BaseDiffusion):
    def __init__(
        self,
        forward_conditioning: str = "data",
        schedule: str = "before_t1_only",
        additional_interpolation_steps: int = 0,
        additional_interpolation_steps_factor: int = 0,
        interpolate_before_t1: bool = False,
        sampling_type: str = "cold",  # 'cold' or 'naive'
        sampling_schedule: Union[List[float], str] = None,
        return_sampled_values: bool = False,
        time_encoding: str = "dynamics",
        refine_intermediate_predictions: bool = True,
        prediction_timesteps: Optional[Sequence[float]] = None,
        enable_interpolator_dropout: Union[bool, str] = True,
        log_every_t: Union[str, int] = None,
        *args,
        **kwargs,
    ):
        """initialisation."""
        super().__init__(*args, **kwargs, sampling_schedule=sampling_schedule)
        sampling_schedule = None if sampling_schedule == "None" else sampling_schedule
        self.save_hyperparameters(ignore=["model"])
        self.num_timesteps = self.hparams.timesteps

        fcond_options = ["data", "none", "data+noise"]
        raise_error_if_invalid_value(forward_conditioning, fcond_options, "forward_conditioning")

        # Add additional interpolation steps to the diffusion steps.
        # we substract 2 because we don't want to use the interpolator in timesteps outside [1, num_timesteps-1].
        horizon = self.num_timesteps
        assert (
            horizon > 1
        ), f"horizon must be > 1, but got {horizon}. Please use datamodule.horizon with > 1"
        if schedule == "linear":
            assert (
                additional_interpolation_steps == 0
            ), "additional_interpolation_steps must be 0 when using linear schedule"
            self.additional_interpolation_steps_fac = additional_interpolation_steps_factor
            if interpolate_before_t1:
                interpolated_steps = horizon - 1
                self.di_to_ti_add = 0
            else:
                interpolated_steps = horizon - 2
                self.di_to_ti_add = additional_interpolation_steps_factor

            self.additional_diffusion_steps = additional_interpolation_steps_factor * interpolated_steps
        elif schedule == "before_t1_only":
            assert (
                additional_interpolation_steps_factor == 0
            ), "additional_interpolation_steps_factor must be 0 when using before_t1_only schedule"
            assert (
                interpolate_before_t1
            ), "interpolate_before_t1 must be True when using before_t1_only schedule"
            self.additional_diffusion_steps = additional_interpolation_steps
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

        self.num_timesteps += self.additional_diffusion_steps
        d_to_i_step = {
            d: self.diffusion_step_to_interpolation_step(d) for d in range(1, self.num_timesteps)
        }
        self.dynamical_steps = {d: i_n for d, i_n in d_to_i_step.items() if float(i_n).is_integer()}
        self.i_to_diffusion_step = {i_n: d for d, i_n in d_to_i_step.items()}
        self.artificial_interpolation_steps = {
            d: i_n for d, i_n in d_to_i_step.items() if not float(i_n).is_integer()
        }

        # whether or not to return the predictions with or without the sampling update applied.
        self.return_sampled_values = return_sampled_values

        # check that float tensors and floats return the same value.
        for d, i_n in d_to_i_step.items():
            i_n2 = float(self.diffusion_step_to_interpolation_step(torch.tensor(d, dtype=torch.float)))
            assert math.isclose(
                i_n, i_n2, abs_tol=4e-6
            ), f"float and tensor return different values for diffusion_step_to_interpolation_step({d}): {i_n} != {i_n2}"
        # note that self.dynamical_steps does not include t=0, which is always dynamical (but not an output!).
        if additional_interpolation_steps_factor > 0 or additional_interpolation_steps > 0:
            self.log_text.info(
                f"Added {self.additional_diffusion_steps} steps.. total diffusion num_timesteps={self.num_timesteps}. \n"
                # f'Mapping diffusion -> interpolation steps: {d_to_i_step}. \n'
                f"Diffusion -> Dynamical timesteps: {self.dynamical_steps}."
            )
        self.enable_interpolator_dropout = enable_interpolator_dropout
        raise_error_if_invalid_value(
            enable_interpolator_dropout, [True, False], "enable_interpolator_dropout"
        )
        if refine_intermediate_predictions:
            self.log_text.info("Enabling refinement of intermediate predictions.")

        # which diffusion steps to take during sampling.
        self.full_sampling_schedule = list(range(0, self.num_timesteps))
        self.sampling_schedule = sampling_schedule or self.full_sampling_schedule

    @property
    def fcast_model_final_layer_act(self) -> nn.Module:
        """Return the final convolution activation function in the Forecastor."""
        return self.model.final_conv[-1]

    @property
    def interp_model_final_layer_act(self) -> nn.Module:
        """Return the final convolution activation function in the Interpolator."""
        return self.interpolator.model.final_conv[-1]

    @property
    def diffusion_steps(self) -> List[int]:
        """Return the number of 'diffusion' steps."""
        return list(range(0, self.num_timesteps))

    @property
    def sampling_schedule(self) -> List[Union[int, float]]:
        """Return the sampling schedule."""
        return self._sampling_schedule

    def diffusion_step_to_interpolation_step(
        self, diffusion_step: Union[int, Tensor]
    ) -> Union[float, Tensor]:
        """
        Convert a diffusion step to an interpolation step
        Args:
            diffusion_step: the diffusion step  (in [1, num_timesteps-1])
        Returns:
            the interpolation step
        """
        # assert correct range.
        if torch.is_tensor(diffusion_step):
            # checks that the diffusion step is not the target.
            assert (0 <= diffusion_step).all() and (
                diffusion_step <= self.num_timesteps - 1
            ).all(), f"diffusion_step must be in [1, num_timesteps-1]=[{1}, {self.num_timesteps - 1}], but got {diffusion_step}"
        else:
            assert (
                0 <= diffusion_step <= self.num_timesteps - 1
            ), f"diffusion_step must be in [1, num_timesteps-1]=[1, {self.num_timesteps - 1}], but got {diffusion_step}"
        if self.hparams.schedule == "linear":
            i_n = (diffusion_step + self.di_to_ti_add) / (self.additional_interpolation_steps_fac + 1)
        elif self.hparams.schedule == "before_t1_only":
            # map d_N to h-1, d_N-1 to h-2, ..., d_n to 1, and d_n-1..d_1 uniformly to [0, 1)
            # e.g. if h=5, then d_5 -> 4, d_4 -> 3, d_3 -> 2, d_2 -> 1, d_1 -> 0.5
            # or                d_6 -> 4, d_5 -> 3, d_4 -> 2, d_3 -> 1, d_2 -> 0.66, d_1 -> 0.33
            # or                d_7 -> 4, d_6 -> 3, d_5 -> 2, d_4 -> 1, d_3 -> 0.75, d_2 -> 0.5, d_1 -> 0.25
            if torch.is_tensor(diffusion_step):
                i_n = torch.where(
                    diffusion_step >= self.additional_diffusion_steps + 1,
                    (diffusion_step - self.additional_diffusion_steps).float(),
                    diffusion_step / (self.additional_diffusion_steps + 1),
                )
            elif diffusion_step >= self.additional_diffusion_steps + 1:
                i_n = diffusion_step - self.additional_diffusion_steps
            else:
                i_n = diffusion_step / (self.additional_diffusion_steps + 1)
        else:
            raise ValueError(f"schedule=``{self.hparams.schedule}`` not supported.")

        return i_n

    def q_sample(
        self,
        x0,
        x_end,
        t: Optional[Tensor],
        interpolation_time: Optional[Tensor] = None,
        is_artificial_step: bool = True,
        **kwargs,
    ) -> Tensor:
        """
        Interpolate using the trained stochastic Interpolator (monte carlo dropout enabled).

        q_sample is a wrapper for interpolation sampling (forward pass through trained interpolator).

        Parameters
        ----------
        x0 : Tensor
            The starting point for interpolation (t=T, last timestep in diffusion model terminology).
        x_end : Tensor
            The endpoint for interpolation (t=0, initial conditions in diffusion model terminology).
        t : Optional[Tensor]
            The timestep for diffusion. Either t or interpolation_time must be None.
        interpolation_time : Optional[Tensor], optional
            The time for interpolation. Either t or interpolation_time must be None.
        is_artificial_step : bool, optional
            Flag indicating whether this is an artificial step. Default is True.

        Returns
        -------
        Tensor
            The interpolated tensor at time t or interpolation_time.
        """
        # just remember that x_end here refers to t=0 (the initial conditions).
        # and x_0 (terminology of diffusion models) refers to t=T, i.e. the last timestep.
        assert t is None or interpolation_time is None, "Either t or interpolation_time must be None."
        t = interpolation_time if t is None else self.diffusion_step_to_interpolation_step(t)
        do_enable = self.training or self.enable_interpolator_dropout
        ipol_handles = [self.interpolator] if hasattr(self, "interpolator") else [self]
        with ExitStack() as stack:
            # inference_dropout_scope of all handles (enable and disable) is managed by the ExitStack.
            for ipol in ipol_handles:
                stack.enter_context(ipol.inference_dropout_scope(condition=do_enable))
            x_ti = self._interpolate(initial_condition=x_end, x_last=x0, t=t, **kwargs)
        return x_ti

    @abstractmethod
    def _interpolate(
        self,
        initial_condition: Tensor,
        x_last: Tensor,
        t: Tensor,
        static_condition: Optional[Tensor] = None,
        num_predictions: int = 1,
    ):
        """This is an internal method. Please use q_sample to access it."""
        raise NotImplementedError(f"``_interpolate`` must be implemented in {self.__class__.__name__}")

    def get_condition(
        self,
        condition,
        x_last: Optional[Tensor],
        prediction_type: str,
        static_condition: Optional[Tensor] = None,
        shape: Sequence[int] = None,
    ) -> Tensor:
        """
        Get the condition, either static, condition or both ([static, condition]).

        Parameters
        ----------
        condition : Tensor or None
            The main condition tensor.
        x_last : Optional[Tensor]
            The last state tensor.
        prediction_type : str
            The type of prediction being made.
        static_condition : Optional[Tensor], optional
            The static condition tensor, if any.
        shape : Sequence[int], optional
            The shape of the output tensor, if needed.

        Returns
        -------
        Tensor
            The combined condition tensor.
        """
        if static_condition is None:
            return condition
        elif condition is None:
            return static_condition
        else:
            return torch.cat([condition, static_condition], dim=1)

    def _predict_last_dynamics(self, forward_condition: Tensor, x_t: Tensor, t: Tensor):
        """
        Predict the last dynamics based on the current state and time: x_last = F(x0 (forward condition), x_t).

        Parameters
        ----------
        forward_condition : Tensor
            The forward condition tensor used for prediction.
        x_t : Tensor
            The current state tensor at time step t.
        t : Tensor
            The time step tensor.

        Returns
        -------
        Tensor
            The predicted last dynamics (x_last_pred).
        """
        if self.hparams.time_encoding == "discrete":
            time = t
        elif self.hparams.time_encoding == "normalized":
            time = t / self.num_timesteps
        elif self.hparams.time_encoding == "dynamics":
            time = self.diffusion_step_to_interpolation_step(t)
        else:
            raise ValueError(f"Invalid time_encoding: {self.hparams.time_encoding}")

        x_last_pred = self.model.predict_forward(x_t, time=time, condition=forward_condition)
        return x_last_pred

    def predict_x_last(
        self,
        condition: Tensor,
        x_t: Tensor,
        t: Tensor,
        is_sampling: bool = False,
        static_condition: Optional[Tensor] = None,
    ):
        """
        Predict x_last using the model in forward mode.
        The forward condition options are: 'none', 'data' and 'data+noise'. If noise is added,
        it is done using a linear time schedule determined from the num_timesteps.

        Parameters
        ----------
        condition : Tensor
            The conditioning tensor.
        x_t : Tensor
            The input tensor at time step t.
        t : Tensor
            The time step tensor. Must be in the range [0, num_timesteps - 1].
        is_sampling : bool, optional
            Flag indicating whether the method is being called during sampling. Default is False.
        static_condition : Optional[Tensor], optional
            Static condition tensor, if any.

        Returns
        -------
        Tensor
            The predicted x_last tensor.
        """
        assert (0 <= t).all() and (t <= self.num_timesteps - 1).all(), f"Invalid timestep: {t}"
        cond_type = self.hparams.forward_conditioning
        if cond_type == "data":
            forward_cond = condition
        elif cond_type == "none":
            forward_cond = None
        elif "data+noise" in cond_type:
            # simply use factor t/T to scale the condition and factor (1-t/T) to scale the noise.
            # this is the same as using a linear combination of the condition and noise.
            tfactor = t / (self.num_timesteps - 1)  # shape: (b,)
            tfactor = tfactor.view(condition.shape[0], *[1] * (condition.ndim - 1))  # (b, 1, 1, 1).

            # TODO: have a way to choose what noise to use from the config.
            # ** noise (linear combination) **
            # The noise is more important at the beginning (t=0) and less important at the end (t=T).
            # if t=0, forward_cond is pure noise. If t=1, forward_cond is just the real data w/o noise.
            # data-driven / guided (adapted) gaussian noise (~ right hand skewed noise).
            noise = generate_data_driven_gaussian_noise(
                batch=condition, lam=0.5, std_factor=0.1, data_lims=[0, 1]
            )
            # # uniform noise.
            # noise = torch.rand_like(condition)
            forward_cond = tfactor * condition + (1 - tfactor) * noise
        else:
            raise ValueError(f"Invalid forward conditioning type: {cond_type}")

        forward_cond = self.get_condition(
            condition=forward_cond,
            x_last=None,
            prediction_type="forward",
            static_condition=static_condition,
            shape=condition.shape,
        )
        x_last_pred = self._predict_last_dynamics(x_t=x_t, forward_condition=forward_cond, t=t)
        return x_last_pred

    @sampling_schedule.setter
    def sampling_schedule(self, schedule: Union[str, List[Union[int, float]]]):
        """
        Set the sampling schedule. At the very minimum, the sampling schedule will go through all dynamical steps.

        Notation:
        - N: number of diffusion steps
        - h: number of dynamical steps
        - h_0: first dynamical step

        Options for diffusion sampling schedule trajectories:
        - 'only_dynamics': the diffusion steps corresponding to dynamical steps (this is the minimum)
        - 'only_dynamics_plus_discreteINT': add INT discrete non-dynamical steps, uniformly drawn between 0 and h_0
        - 'only_dynamics_plusINT': add INT non-dynamical steps (possibly continuous), uniformly drawn between 0 and h_0
        - 'everyINT': only use every INT-th diffusion step (e.g. 'every2' for every second diffusion step)
        - 'firstINT': only use the first INT diffusion steps
        - 'firstFLOAT': only use the first FLOAT*N diffusion steps

        Parameters
        ----------
        schedule_type : str
            The type of sampling schedule to use. Must be one of the options described above.
        value : int or float, optional
            The value to use for schedules that require an additional parameter (INT or FLOAT).
        """
        schedule_name = schedule
        if isinstance(schedule_name, str):
            base_schedule = [0] + list(
                self.dynamical_steps.keys()
            )  # already included: + [self.num_timesteps - 1].
            artificial_interpolation_steps = list(self.artificial_interpolation_steps.keys())
            if "only_dynamics" in schedule_name:
                schedule = []  # only sample from base_schedule (added below).

                if "only_dynamics_plus" in schedule_name:
                    # parse schedule 'only_dynamics_plusN' to get N.
                    plus_n = int(
                        schedule_name.replace("only_dynamics_plus", "").replace("_discrete", "")
                    )
                    # Add N additional steps to the front of the schedule
                    schedule = list(np.linspace(0, base_schedule[1], plus_n + 1, endpoint=False))
                    if "_discrete" in schedule_name:  # floor the values.
                        schedule = [int(np.floor(s)) for s in schedule]
                else:
                    assert "only_dynamics" == schedule_name, f"Invalid sampling schedule: {schedule}"

            elif schedule_name.startswith("every"):
                # parse schedule 'everyNth' to get N.
                every_nth = (
                    schedule.replace("every", "").replace("th", "").replace("nd", "").replace("rd", "")
                )
                every_nth = int(every_nth)
                assert 1 <= every_nth <= self.num_timesteps, f"Invalid sampling schedule: {schedule}"
                schedule = artificial_interpolation_steps[::every_nth]

            elif schedule.startswith("first"):
                # parse schedule 'firstN' to get N.
                first_n = float(schedule.replace("first", "").replace("v2", ""))
                if first_n < 1:
                    assert (
                        0 < first_n < 1
                    ), f"Invalid sampling schedule: {schedule}, must end with number/float > 0"
                    first_n = int(np.ceil(first_n * len(artificial_interpolation_steps)))
                    schedule = artificial_interpolation_steps[:first_n]
                    self.log_text.info(
                        f"Using sampling schedule: {schedule_name} -> (first {first_n} steps)"
                    )
                else:
                    assert (
                        first_n.is_integer()
                    ), f"If first_n >= 1, it must be an integer, but got {first_n}"
                    assert 1 <= first_n <= self.num_timesteps, f"Invalid sampling schedule: {schedule}"
                    first_n = int(first_n)
                    # Simple schedule: sample using first N steps.
                    schedule = artificial_interpolation_steps[:first_n]
            else:
                raise ValueError(f"Invalid sampling schedule: ``{schedule}``. ")

            # Add dynamic steps to the schedule.
            schedule += base_schedule
            # need to sort in ascending order and remove duplicates.
            schedule = list(sorted(set(schedule)))

        assert (
            1 <= schedule[-1] <= self.num_timesteps
        ), f"Invalid sampling schedule: {schedule}, must end with number/float <= {self.num_timesteps}"
        if schedule[0] != 0:
            self.log_text.warning(
                f"Sampling schedule {schedule_name} must start at 0. Adding 0 to the beginning of it."
            )
            schedule = [0] + schedule

        last = schedule[-1]
        if last != self.num_timesteps - 1:
            self.log_text.warning("------" * 20)
            self.log_text.warning(
                f"Are you sure you don't want to sample at the last timestep? (current last timestep: {last})"
            )
            self.log_text.warning("------" * 20)

        # check that schedule is monotonically increasing.
        for i in range(1, len(schedule)):
            assert (
                schedule[i] > schedule[i - 1]
            ), f"Invalid sampling schedule not monotonically increasing: {schedule}"

        if all(float(s).is_integer() for s in schedule):
            schedule = [int(s) for s in schedule]
        else:
            self.log_text.info(
                f"Sampling schedule {schedule_name} uses diffusion steps it has not been trained on!"
            )
        self._sampling_schedule = schedule

    def sample_loop(
        self,
        initial_condition,
        static_condition: Optional[Tensor] = None,
        log_every_t: Optional[Union[str, int]] = None,
        num_predictions: int = None,
    ):
        """
        Emulates the validation / sampling step.
        Returns the x_t+1, .. x_t+N-1, x_t+N.

        The method performs the following steps:
        1. At t=0, forecast x+t+h: F(x0, x0) = x_t+h^(1).
        2. Interpolate, x_t_i+1 using: I(x0, x_t+h^(1)) (x_interpolated_s_next)
        3. Apply 'cold' sampling update (suggested). Interpolate x_t_i using I(I(x0, x_t+h^(1)))
        and calculate an updated x_t_i+1 = x_interpolated_s_next - x_interpolated_s + x_s.
        where x_s at t=0 is the initial condition.
        4. At t=1, forecast x+t+h: F(x0, x_t_i+1) = x_t+h^(2).
        5. Interpolate, x_t_i+2 using: I(x0, x_t+h^(2)) (x_interpolated_s_next).
        6. Repeat 'cold' sampling.
        7. Repeat until final timestep. At which t=N, and x_t_i+N = x_t+h^(N).
        8. The final 'cold' sampling is: x_t+h = and x_t_i+N-1 + I(x0, x_t+h^(N-1)) + x_t+h^(N).

        Parameters
        ----------
        initial_condition : Tensor
            The initial condition for the sampling process.
        static_condition : Optional[Tensor], optional
            Static condition data, if any.
        log_every_t : Optional[Union[str, int]], optional
            Specifies how often to log during the sampling process.
        num_predictions : int, optional
            The number of predictions to make.

        Returns
        -------
        Tensor
            The predicted values x_t+1, .. x_t+N-1, x_t+N.
        """
        batch_size = initial_condition.shape[0]
        log_every_t = log_every_t or self.hparams.log_every_t
        log_every_t = log_every_t if log_every_t != "auto" else 1
        sc_kw = dict(static_condition=static_condition)
        assert (
            len(initial_condition.shape) == 4
        ), f"condition.shape: {initial_condition.shape} (should be 4D)."
        intermediates, x0_hat, dynamics_pred_step = dict(), None, 0
        last_i_n_plus_one = self.sampling_schedule[-1] + 1

        # set up sampling schedule.
        s_and_snext = zip(
            self.sampling_schedule,
            self.sampling_schedule[1:] + [last_i_n_plus_one],
            self.sampling_schedule[2:] + [last_i_n_plus_one, last_i_n_plus_one],
        )

        # always start with x_s = x0.
        x_s = initial_condition.clone()
        for s, s_next, s_nnext in tqdm(
            s_and_snext,
            desc="Sampling time step",
            total=len(self.sampling_schedule),
            leave=False,
        ):
            is_last_step = s == self.num_timesteps - 1

            # handle data range of interpolated x_s.
            # after each sampling step, x_s is 'reused' in the forecastor.
            # need to ensure that it has data range [0, 1].
            # at s=0, x_s = initial_condition which is in raw scale [0, 1].
            if (s > 0) and (isinstance(self.interp_model_final_layer_act, nn.Tanh)):
                x_s = transform_minus1_1_to_0_1(x_s)

            # ** forecasting **
            # F(x0, x_s) = predict target data, xh where x0 is initial state and x_s is interpolated sample.
            # x0 is used when either "data" or "data+noise" is chosen for condition in dyffusion.yaml.
            step_s = torch.full((batch_size,), s, dtype=torch.float32, device=self.device)
            x0_hat = self.predict_x_last(
                condition=initial_condition, x_t=x_s, t=step_s, is_sampling=True, **sc_kw
            )

            if isinstance(self.fcast_model_final_layer_act, nn.Tanh):
                # x_t+h is used in the interpolator so needs to be [0, 1].
                x0_hat = transform_minus1_1_to_0_1(x0_hat)

            # get interpolator inputs:
            # determine what step to interpolate: dynamic or artificial.
            # I(x0, x_t+h(n)) where n is the number of times forecasting x_t+h.
            time_i_n = self.diffusion_step_to_interpolation_step(s_next) if not is_last_step else np.inf
            is_dynamics_pred = float(time_i_n).is_integer() or is_last_step
            q_sample_kwargs = dict(
                x0=x0_hat.clone(),
                x_end=initial_condition,
                is_artificial_step=not is_dynamics_pred,
                reshape_ensemble_dim=not is_last_step,
                num_predictions=1 if is_last_step else num_predictions,
            )
            if s_next <= self.num_timesteps - 1:
                # ** interpolating ** (interpolate the next step, s+1).
                step_s_next = torch.full((batch_size,), s_next, dtype=torch.float32, device=self.device)
                x_interpolated_s_next = self.q_sample(**q_sample_kwargs, t=step_s_next, **sc_kw)
            else:
                # in the last step, x_s+1 = last forecasted x_t+h.
                # if the interpolated values are [-1, 1], need to get x0_hat to same scale prior to cold sampling update.
                if isinstance(self.interp_model_final_layer_act, nn.Tanh):
                    x_interpolated_s_next = transform_0_1_to_minus1_1(x0_hat)
                else:
                    x_interpolated_s_next = x0_hat

            # ** sampling (update x_s) **
            if self.hparams.sampling_type in ["cold"]:
                # ** cold sampling update: Algorithm 2, step 4 (paper) **
                # the step below needs to be before the interpolation step to ensure correct data range for all interpolated, x_s.
                if isinstance(self.interp_model_final_layer_act, nn.Tanh):
                    x_s = transform_0_1_to_minus1_1(x_s)

                # ** interpolating ** (interpolate at the current step, s).
                x_interpolated_s = self.q_sample(**q_sample_kwargs, t=step_s, **sc_kw) if s > 0 else x_s

                # apply x̂_t+i_n+1 = I(x_t, x_t+h, i_n+1) - I(x_t, x_t+h, i_n) +  x̂_t+i_n (Algorithm 2, step 4).
                # for s = 0, we have x_s_degraded = x_s, so we just directly return x_s_degraded_next.
                x_s = x_interpolated_s_next - x_interpolated_s + x_s  # update x_s.

                # clamp the updates.
                # the cold sampling can cause the data to go out-of-bounds.
                if isinstance(self.interp_model_final_layer_act, nn.Tanh):
                    x_s = torch.clamp(x_s, -1, 1)  # [-1, 1].
                elif isinstance(self.interp_model_final_layer_act, nn.ReLU):
                    x_s = torch.clamp(x_s, 0, None)  # > 0.
                else:
                    # assumes not constraint on output.
                    pass

            elif self.hparams.sampling_type == "naive":
                x_s = x_interpolated_s_next
            else:
                raise ValueError(f"unknown sampling type {self.hparams.sampling_type}")

            dynamics_pred_step = int(time_i_n) if s < self.num_timesteps - 1 else dynamics_pred_step + 1
            if is_dynamics_pred:
                if self.return_sampled_values:
                    intermediates[f"t{dynamics_pred_step}_preds"] = x_s
                else:
                    # return the predictions without (cold) sampling applied.
                    intermediates[f"t{dynamics_pred_step}_preds"] = x_interpolated_s_next

                if log_every_t is not None:
                    intermediates[f"t{dynamics_pred_step}_preds2"] = x_interpolated_s_next

            s1, s2 = s, s  # s + 1, next_step  # s, next_step.
            if log_every_t is not None:
                intermediates[f"intermediate_{s1}_x0hat"] = x0_hat
                intermediates[f"xipol_{s2}_dmodel"] = x_interpolated_s_next
                if self.hparams.sampling_type == "cold":
                    intermediates[f"xipol_{s1}_dmodel2"] = x_interpolated_s

        # refine the interpolated values, t<N.
        # Algorithm 2, step 6 (DYffusion paper).
        if self.hparams.refine_intermediate_predictions:
            # use last prediction of x0 for final prediction of intermediate steps (not the last timestep!).
            # make sure x0 is in the desired data range and that the range is consistent.
            q_sample_kwargs["x0"] = x0_hat
            q_sample_kwargs["is_artificial_step"] = False
            dynamical_steps = self.hparams.prediction_timesteps or list(self.dynamical_steps.values())
            dynamical_steps = [i for i in dynamical_steps if i < self.num_timesteps]
            for i_n in dynamical_steps:
                i_n_time_tensor = torch.full((batch_size,), i_n, dtype=torch.float32, device=self.device)
                i_n_for_str = int(i_n) if float(i_n).is_integer() else i_n
                assert (
                    not float(i_n).is_integer() or f"t{i_n_for_str}_preds" in intermediates
                ), f"t{i_n_for_str}_preds not in intermediates"

                # ** interpolating ** (re-interpolate each step using the final predicted x_h).
                intermediates[f"t{i_n_for_str}_preds"] = self.q_sample(
                    **q_sample_kwargs,
                    t=None,
                    interpolation_time=i_n_time_tensor,
                    **sc_kw,
                )

        # TODO: handle this better.
        if isinstance(self.fcast_model_final_layer_act, nn.ReLU) and isinstance(
            self.interp_model_final_layer_act, nn.Tanh
        ):
            # convert all outputs to [0, 1] if model ouptuts expected to be [0, 1] but interpolator outputs [-1,1].
            x_s = transform_minus1_1_to_0_1(x_s)
            intermediates = {k: transform_minus1_1_to_0_1(v) for k, v in intermediates.items()}
            x_interpolated_s_next = transform_minus1_1_to_0_1(x_interpolated_s_next)

        if last_i_n_plus_one < self.num_timesteps:
            return x_s, intermediates, x_interpolated_s_next
        return x0_hat, intermediates, x_s

    @torch.no_grad()
    def sample(self, initial_condition, num_samples=1, **kwargs):
        """
        Sampling process for DYffusion. Wrapper function for .sample_loop

        Parameters
        ----------
        initial_condition : Tensor
            The initial condition for the sampling process.
        num_samples : int, optional
            The number of samples to generate. Default is 1.

        Returns
        -------
        Tensor
            The intermediate results from the sampling process.
        """
        x_0, intermediates, x_s = self.sample_loop(initial_condition, **kwargs)
        return intermediates


class DYffusion(BaseDYffusion):
    """
    DYffusion model with a pretrained interpolator.

    Parameters
    ----------
    interpolator_checkpoint_id : str
        The id of the interpolator checkpoint.
    interpolator_checkpoint_base_path : str
        The base path that contains the interpolator checkpoint.
    lambda_reconstruction : float
        The weight of the reconstruction loss.
    lambda_reconstruction2 : float
        The weight of the one-step-ahead reconstruction loss (using the predicted xt_last as feedback).
    """

    def __init__(
        self,
        interpolator_checkpoint_id: Optional[str],
        interpolator_checkpoint_base_path: Optional[str],
        lambda_reconstruction: float = 1.0,
        lambda_reconstruction2: float = 0.0,
        initial_forecast_linear_schedule: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """initialisation."""
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=["interpolator", "model"])

        # need to provide sufficient information to load in an interpolator.
        assert all([interpolator_checkpoint_id, interpolator_checkpoint_base_path]), (
            "Both interpolator_checkpoint_id and interpolator_checkpoint_base_path must be provided."
            "Please provide str in dyffusion.yaml config file."
        )

        # training loss params.
        self.lam1 = lambda_reconstruction
        self.lam2 = lambda_reconstruction2
        self.init_fcast_linear_schedule = initial_forecast_linear_schedule

        # extract the schedule infomation to class attrs.
        if self.init_fcast_linear_schedule["apply_schedule"]:
            self.apply_schedule = True
            self.schedule_start = self.init_fcast_linear_schedule["schedule_start"]
            self.schedule_end = self.init_fcast_linear_schedule["schedule_end"]
            self.num_epochs_to_zero = self.init_fcast_linear_schedule["num_epochs_decay"]
        else:
            self.apply_schedule = False

        # load in a trained interpolator.
        self.interpolator_ckpt_id = interpolator_checkpoint_id
        self.interpolator_ckpt_base_path = interpolator_checkpoint_base_path
        interpolator_model_ckpt_cfg = load_model_checkpoint_config(
            ckpt_id=self.interpolator_ckpt_id, model_ckpt_cfg_base_path=self.interpolator_ckpt_base_path
        )
        interpolator = instantiate_interpolator_experiment(
            model_ckpt_cfg=interpolator_model_ckpt_cfg.model_config,
            datamodule_cfg=self.hparams.datamodule_config,
        )
        # load in ckpt interpolator (wights and biases) to the interpolator.model.
        interpolator.model = load_model_checkpoint_for_eval(
            model=interpolator.model,
            ckpt_id=self.interpolator_ckpt_id,
            ckpt_base_path=self.interpolator_ckpt_base_path,
        )
        # set the trained interpolator as an attr and freeze it.
        self.interpolator = interpolator
        freeze_model(self.interpolator.model)
        log.info(f"Successfully loaded (and frozen) interpolator (id: {self.interpolator_ckpt_id}).")
        self.interpolator_window = self.interpolator.window
        self.interpolator_horizon = self.interpolator.true_horizon
        last_d_to_i_tstep = self.diffusion_step_to_interpolation_step(
            diffusion_step=(self.num_timesteps - 1)
        )
        if self.interpolator_horizon != last_d_to_i_tstep + 1:
            raise ValueError(
                f"interpolator horizon {self.interpolator_horizon} must be equal to the "
                f"last interpolation step+1=i_N=i_{self.num_timesteps - 1}={last_d_to_i_tstep + 1}"
            )

    def _interpolate(
        self,
        initial_condition: Tensor,
        x_last: Tensor,
        t: Tensor,
        static_condition: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Interpolate x_t using x0 (initial condition) and xt+h (x_last) where 0 < t < h.

        Parameters
        ----------
        initial_condition : Tensor
            The initial condition data (x0).
        x_last : Tensor
            The final state data (xt+h).
        t : Tensor
            The time steps at which to interpolate. Must be in the range
            (0, self.interpolator_horizon).
        static_condition : Optional[Tensor], optional
            Static condition data, if any.

        Returns
        -------
        Tensor
            The interpolated outputs at the specified time steps.
        """
        # interpolator networks uses time in [1, horizon-1].
        assert (0 < t).all() and (
            t < self.interpolator_horizon
        ).all(), f"interpolate time must be in (0, {self.interpolator_horizon}), got {t}"
        # select condition data to be consistent with the interpolator training data.
        interpolator_inputs = torch.cat([initial_condition, x_last], dim=1)
        kwargs["reshape_ensemble_dim"] = False
        interpolator_outputs = self.interpolator.predict(
            interpolator_inputs, condition=static_condition, time=t, **kwargs
        )
        interpolator_outputs = interpolator_outputs["preds"]

        return interpolator_outputs

    def p_losses(
        self,
        xt_last: Tensor,
        condition: Tensor,
        t: Tensor,
        criterion: nn.Module,
        static_condition: Tensor = None,
    ):
        """
        Perform a training step analogous to the standard approach, returning the loss.
        This function executes the following steps:
        1. Make an initial forecast of x_t+h using only x0: F(x0, x0) = x_t+h^(init).
        2. Calculate the initial loss between target and x_t+h^(init).
        3. Interpolate x_t+i values using x_t+h^(init) and initial condition at x0:
        I(x0, x_t+h^(init)) = x_t+i, for 0 < t < h.
        4. Forecast x_t+h using F(x_0, x_t+i) = x_t+h.
        5. Calculate the loss between F(x_t, x0) and x_t+h.
        6. Emulate 1 more step and calculate the additional one-step-ahead loss term.
        7. Compute the total loss: alpha*initial_loss + (1-alpha)*(lam1*loss + lam2*one-step-ahead-loss).


        Parameters
        ----------
        xt_last : Tensor
            The start/target data (time = horizon) with dimensions [batch, c, h, w].
        condition : Tensor
            The initial condition data (time = 0).
        t : Tensor
            The time step of the diffusion process.
        criterion : nn.Module
            The loss criterion used for the loss calculation.
        static_condition : Tensor, optional
            The static condition data, if any.

        Returns
        -------
        float
            The computed total loss.
        """
        # initial forecast loss weighting.
        if self.apply_schedule:
            # simple linear schedule to warm up the realstic predictions.
            # after n_to_zero epochs, we optimise the entire loss. The initial forcast is implicit in the total loss.
            if self.current_epoch <= self.num_epochs_to_zero:
                alpha = self.schedule_start - (self.current_epoch * (1 / self.num_epochs_to_zero))
            else:
                alpha = self.schedule_end
        else:
            # default to equally weighted initial loss, exisitng forecast loss.
            alpha = 0.5

        # ensure targets are correct data range.
        # don't want to modify xt_last as is it in raw data range and is used in interpolator.
        if isinstance(self.fcast_model_final_layer_act, nn.Tanh):
            xt_last_target = transform_0_1_to_minus1_1(xt_last.clone())  # [-1, 1]
        else:
            xt_last_target = xt_last.clone()

        # initial (t zero) forecast, F(x0, x0) = x_h:
        #  ** forecasting ** predict x_h using F(x0, x0) to emulate the sampling process.
        t_zero = torch.zeros(t.size(), dtype=torch.long).to(self.device)
        xt_last_pred_realistic = self.predict_x_last(
            condition=condition, x_t=condition, t=t_zero, static_condition=static_condition
        )
        loss_forward_initial = criterion(xt_last_pred_realistic, xt_last_target)

        # handle forecast outputs != [0, 1].
        if isinstance(self.fcast_model_final_layer_act, nn.Tanh):
            xt_last_pred_realistic = transform_minus1_1_to_0_1(xt_last_pred_realistic)

        # t non-zero forecasts, F(x0, x_t) = x_h:
        # ** interpolating **  interpolate x_t for t > 0 (at t=0, x_t = initial condition).
        x_t = condition.clone()
        t_nonzero = t > 0
        if t_nonzero.any():
            x_interpolated = self.q_sample(
                x_end=condition[t_nonzero],
                # x0=xt_last[t_nonzero],
                x0=xt_last_pred_realistic[t_nonzero],
                t=t[t_nonzero],
                static_condition=(None if static_condition is None else static_condition[t_nonzero]),
                num_predictions=1,  # sample one interpolation prediction
            )
            # update x_t with the interpolated values.
            if isinstance(self.interp_model_final_layer_act, nn.Tanh):
                x_t[t_nonzero] = transform_minus1_1_to_0_1(x_interpolated.to(x_t.dtype))
            else:
                x_t[t_nonzero] = x_interpolated.to(x_t.dtype)

        # ** forecasting ** F(x0, x_t) = x_h.
        xt_last_pred = self.predict_x_last(
            condition=condition, x_t=x_t, t=t, static_condition=static_condition
        )
        loss_forward = criterion(xt_last_pred, xt_last_target)

        # train the forward predictions II by emulating one more step of the dyffusion process.
        tnot_last = t <= self.num_timesteps - 2
        t2 = t[tnot_last] + 1  # t2 is the next time step, between 1 and T-1.
        calc_t2 = tnot_last.any()
        if self.lam2 > 0 and calc_t2:
            # handle data range of preds prior to interpolator.
            if isinstance(self.fcast_model_final_layer_act, nn.Tanh):
                xt_last_pred = transform_minus1_1_to_0_1(xt_last_pred)

            # ** interpolating ** train the predictions using x0 = xlast = forward_pred(condition, t=0).
            cond_notlast = condition[tnot_last]
            x0not_last = xt_last_pred[tnot_last]  # pred needs to be [0, 1].
            sc_notlast = None if static_condition is None else static_condition[tnot_last]
            # use the predictions of xt_last = x0_not_last to interpolate the next step.
            x_interpolated2 = self.q_sample(
                x_end=cond_notlast,
                x0=x0not_last,
                t=t2,
                static_condition=sc_notlast,
                num_predictions=1,
            )
            if isinstance(self.interp_model_final_layer_act, nn.Tanh):
                x_interpolated2 = transform_minus1_1_to_0_1(x_interpolated2)

            # ** forecasting ** F(x0, x_t+1) = x_h+1.
            xt_last_pred2 = self.predict_x_last(
                condition=cond_notlast,
                x_t=x_interpolated2,
                t=t2,
                static_condition=sc_notlast,
            )
            loss_forward2 = criterion(xt_last_pred2, xt_last_target[tnot_last])
        else:
            loss_forward2 = 0.0

        # lam weighted loss: initial loss + (loss + one-step-ahead loss).
        # See Section 'Forecasting as a reverse process' in DYffusion.
        loss = (alpha * loss_forward_initial) + (
            (1 - alpha) * (self.lam1 * loss_forward + self.lam2 * loss_forward2)
        )

        # logs.
        log_prefix = "train" if self.training else "val"
        loss_dict = {
            "loss": loss,
            f"{log_prefix}/loss_forward_initial": loss_forward_initial,
            f"{log_prefix}/loss_forward": loss_forward,
            f"{log_prefix}/loss_forward2": loss_forward2,
        }

        return loss_dict
