"""Module to train a DYffusion Interpolator or Forecastor."""

import argparse
import os
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate

from rainnow.src.plotting import plot_experiment_train_val_logs
from rainnow.src.utilities.instantiators import (
    instantiate_callbacks,
    instantiate_loggers,
    update_callback_monitors,
)
from rainnow.src.utilities.loading import load_model_state_dict
from rainnow.src.utilities.utils import (
    generate_alphanumeric_id,
    get_datamodule,
    get_device,
    get_logger,
    get_module,
    print_config,
)

MAIN_CONFIG_FILENAME = "main_config.yaml"
CONFIGS_BASE_FILEPATH = "src/dyffusion/configs/"


def main(ckpt_path: str = None, use_train_loss_for_ckpt_monitor: bool = False):
    """
    example: python train.py | python train.py --ckpt_path <run_id>/checkpoints/last.ckpt

    """
    # instantiate logger and get config.
    log = get_logger(log_file_name=None, name=__name__)
    with hydra.initialize(config_path=CONFIGS_BASE_FILEPATH, version_base=None):
        config = hydra.compose(config_name=MAIN_CONFIG_FILENAME)

    # seed for reproducibility.
    if config.get("seed"):
        pl.seed_everything(config.seed, workers=True)

    # check device availability.
    device = get_device()
    if device == "cpu":
        # swtich config to 'cpu' accelerator.
        log.info("** No GPU available. Switching trainer to CPU. **")
        config.trainer.accelerator = "cpu"

    # need to keep this before overwriting the config.work_dir.
    # if a ckpt_path is passed, check exists and load it. If it doesn't end train.py.
    to_load = False
    if ckpt_path:
        _ckpt_path = Path(os.path.join(config.work_dir, ckpt_path))
        if _ckpt_path.exists():
            log.info(f"Loading in checkpoint from {_ckpt_path}.")
            loaded_model_state_dict = load_model_state_dict(_ckpt_path, device)
            to_load = True
        else:
            log.warning(f"Cannot find {_ckpt_path}. Please check the input ckpt_path ({ckpt_path}).")
            return

    # generate a new 8-alphanumeric id for this run. All results will then be saved in work_dir/<id>/.
    run_id = f"{config.module.experiment_type}-{str.lower(generate_alphanumeric_id(8))}"
    log.info(f"*** (training & val) Run ID: {run_id} ***\n")
    # config modifications need to be before instantiating the pl modules.
    config.work_dir = f"{config.work_dir}/{run_id}"
    config.ckpt_dir = f"{config.work_dir}/checkpoints/"
    config.log_dir = f"{config.work_dir}/logs/"

    # debug info. print out config nicely.
    print_config(config)  # override <fields> kwarg to choose print.

    # instantiate pl.Lightning object required for training.
    datamodule: pl.LightningDataModule = get_datamodule(config)
    model: pl.LightningModule = get_module(config)
    callbacks: List[pl.Callback] = instantiate_callbacks(config.get("callbacks"))
    # update callback monitors.
    if use_train_loss_for_ckpt_monitor:
        update_callback_monitors(callbacks, new_monitor=model.default_monitor_metric)
    logger: List[pl.Logger] = instantiate_loggers(config.get("logger"))
    trainer: pl.Trainer = instantiate(config=config.trainer, callbacks=callbacks, logger=logger)

    # load in checkpoint model.
    if to_load:
        model._model.load_state_dict(state_dict=loaded_model_state_dict, strict=True)

    # debug model info.
    log.info(f"** Running {model.experiment_type} experiment **")
    log.info("** Model Info: **")
    log.info(f"Loss function: {getattr(model.criterion, '_init_kwargs', model.criterion)}")
    log.info(f"Ouput layer activation = ** {model.model_final_layer_activation} **")

    # train the model.
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=None)
    log.info("*** Training successfully finished ***")
    log.info(f"Plotting (and saving) log information ({config.work_dir}).")

    plot_experiment_train_val_logs(
        config=config,
        val_metrics=["mse", "crps", "ssr", model.criterion_name],
        figsizes=[(8, 4), (18, 10)],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint file (optional). Only include <run_id>/checkpoints/<ckpt_file_name>.ckpt",
    )
    parser.add_argument(
        "--use_train_loss_for_ckpt_monitor",
        type=str,
        default=True,
        help="If True, the ckpt monitor metric will be updated with the loss used for training. If False, will default to MSE.",
    )
    args = parser.parse_args()

    main(ckpt_path=args.ckpt_path, use_train_loss_for_ckpt_monitor=args.use_train_loss_for_ckpt_monitor)
