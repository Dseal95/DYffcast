"""A train script for ConvLSTM."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from rainnow.src.configs.convlstm_train_config import trainer as config
from rainnow.src.convlstm_trainer import save_checkpoint, train, validate
from rainnow.src.datasets import IMERGDataset
from rainnow.src.loss import LPIPSMSELoss
from rainnow.src.models.conv_lstm import ConvLSTMModel
from rainnow.src.normalise import PreProcess
from rainnow.src.plotting import plot_training_val_loss
from rainnow.src.utilities.loading import load_imerg_datamodule_from_config
from rainnow.src.utilities.utils import (generate_alphanumeric_id, get_device,
                                         get_logger)

# ** DIRs **
BASE_PATH = "/teamspace/studios/this_studio/DYffcast/"  # /rds/general/user/ds423/home
CKPT_BASE_PATH = f"{BASE_PATH}/DYffcast/rainnow/results/"
CONFIGS_BASE_PATH = f"{BASE_PATH}/DYffcast/rainnow/src/dyffusion/configs/"


def main(ckpt_path: str = None):
    """Wrapper function to train a ConvLSTM model."""
    # instantiate logger and get device.
    log = get_logger(log_file_name=None, name=__name__)
    device = get_device()

    # generate a new 8-alphanumeric id for this run. All results will then be saved in work_dir/<id>/.
    run_id = f"convlstm-{str.lower(generate_alphanumeric_id(8))}"
    # make a dir for the new model.
    save_dir = Path(os.path.join(CKPT_BASE_PATH, f"{run_id}"))
    os.mkdir(save_dir)
    log.info(f"*** (training & val) Run ID: {run_id} ***")

    # get training and validation data.
    datamodule = load_imerg_datamodule_from_config(
        cfg_base_path=CONFIGS_BASE_PATH,
        cfg_name="imerg_precipitation.yaml",
        overrides={
            "boxes": config["data"]["boxes"],
            "window": config["data"]["window"],
            "horizon": config["data"]["horizon"],
            "prediction_horizon": config["data"]["horizon"],
            "sequence_dt": config["data"]["dt"],
        },
    )
    datamodule.setup("fit")

    # debug info.
    log.info("** datamodule params **")
    log.info(
        f"boxes={config['data']['boxes']} | window={config['data']['window']} | horizon={config['data']['horizon']}."
    )

    # create datasets.
    train_dataset = IMERGDataset(
        datamodule,
        "train",
        sequence_length=config["model"]["input_sequence_length"],
        target_length=config["model"]["output_sequence_length"],
    )
    val_dataset = IMERGDataset(
        datamodule,
        "validate",
        sequence_length=config["model"]["input_sequence_length"],
        target_length=config["model"]["output_sequence_length"],
    )

    # create dataloaders.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=datamodule.hparams.batch_size,
        num_workers=config["data"]["num_workers"],
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=datamodule.hparams.batch_size,
        num_workers=config["data"]["num_workers"],
        shuffle=False,
    )

    # instantiate model.
    model = ConvLSTMModel(
        input_sequence_length=config["model"]["input_sequence_length"],
        output_sequence_length=config["model"]["output_sequence_length"],
        input_dims=config["model"]["input_dims"],
        hidden_channels=config["model"]["hidden_channels"],
        output_channels=config["model"]["output_channels"],
        num_layers=config["model"]["num_layers"],
        kernel_size=config["model"]["kernel_size"],
        output_activation=config["model"]["output_activation"],
        apply_batchnorm=config["model"]["apply_batchnorm"],
        cell_dropout=config["model"]["cell_dropout"],
        device=device,
    )
    model = model.to(device)

    # load in a checkpoint.
    if ckpt_path:
        _ckpt_path = Path(os.path.join(CKPT_BASE_PATH, ckpt_path))
        if _ckpt_path.exists():
            log.info(f"Loading in checkpoint from {_ckpt_path}.")
            model.load_state_dict(
                torch.load(_ckpt_path, map_location=torch.device(device))["model_state_dict"]
            )
        else:
            log.warning(f"Cannot find {_ckpt_path}. Please check the input ckpt_path ({ckpt_path}).")
            return

    # set up optimizers and scheduler.
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=config["training"]["scheduler_factor"],
        patience=config["training"]["scheduler_patience"],
    )

    # ** instantiate the preprocesser obj **
    pprocessor = PreProcess(
        percentiles=datamodule.normalization_hparams["percentiles"],
        minmax=datamodule.normalization_hparams["min_max"],
    )
    # setup loss.
    # CBLoss need to instantiate it here as requires pprocessor to scale the nodes.
    CRITERION = nn.BCELoss(reduction="mean")
    CRITERION = LPIPSMSELoss(
        alpha=0.6,
        model_name="alex",  # trains better with 'alex' - https://github.com/richzhang/PerceptualSimilarity.
        reduction="mean",
        gamma=1.0,
        mse_type="cb",
        **{"beta": 1, "data_preprocessor_obj": pprocessor},
    ).to(device)

    # debug info.
    # fmt: off
    log.info("** model params **")
    log.info(f"input={config['model']['input_sequence_length']} | target={config['model']['output_sequence_length']}.")
    log.info(f"inputs dims={config['model']['input_dims']} | output channels={config['model']['output_channels']}.")
    log.info(f"hidden channels={config['model']['hidden_channels']} | num layers={config['model']['num_layers']}.")
    log.info(f"kernel size={config['model']['kernel_size']} | output activation={config['model']['output_activation']}.")
    log.info(f"batchnorm = {config['model']['apply_batchnorm']} | cell dropout={config['model']['cell_dropout']}.")
    log.info("** training params **")
    log.info(f"learning rate = {config['training']['lr']} | \
             weight decay = {config['training']['weight_decay']} | num epochs = {config['training']['num_epochs']}.")
    log.info(f"loss function = {CRITERION.__class__.__name__}.\n")
    # fmt: on
    
    # training and validation.
    model_save_path = Path(os.path.join(save_dir, f"{run_id}.pt"))
    train_losses, val_losses = [], []
    with tqdm(
        total=len(train_loader) * config["training"]["num_epochs"], desc="Training", unit="batch"
    ) as tepoch:
        for i in range(config["training"]["num_epochs"]):
            logs = {}
            train_loss = train(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                criterion=CRITERION,
                tepoch=tepoch,
                curr_epoch=i,
                n_epochs=config["training"]["num_epochs"],
                device=device,
                log_training=True,
            )
            train_losses.append(train_loss)

            if val_loader:
                val_loss = validate(
                    model=model, data_loader=val_loader, criterion=CRITERION, device=device
                )
                val_losses.append(val_loss)

            # save a model checkpoint.
            if model_save_path:
                # save 'best' val loss checkpoint.
                if val_loss <= min(val_losses, default=float("inf")):
                    save_checkpoint(
                        model_weights=model.state_dict(),
                        optimizer_info=optimizer.state_dict(),
                        model_save_path=model_save_path,
                        epoch=i,
                        train_loss=train_loss,
                        val_loss=val_loss,
                    )
            # save the last epoch.
            if i == (config["training"]["num_epochs"] - 1):
                save_checkpoint(
                    model_weights=model.state_dict(),
                    optimizer_info=optimizer.state_dict(),
                    model_save_path=f"{str(model_save_path)[:-3]}_last.pt",
                    epoch=i,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
            # add scheduler.
            if scheduler:
                scheduler.step(train_loss)

    log.info(f"*** Training successfully finished (id: {run_id}) ***")
    log.info(f"Plotting (and saving) log information.")
    fig = plot_training_val_loss(
        train_losses=train_losses, val_losses=val_losses, criterion_name=CRITERION.__class__.__name__
    )
    df_result = pd.DataFrame(
        data=list(zip(range(len(train_losses)), train_losses, val_losses)),
        columns=["epoch", "train_loss", "val_loss"],
    )
    df_result.to_csv(Path(os.path.join(save_dir, "results.csv")))
    fig.savefig(Path(os.path.join(save_dir, "results.png")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to the checkpoint file (optional). Only include <run_id>/checkpoints/<ckpt_file_name>.ckpt",
    )
    args = parser.parse_args()

    main(ckpt_path=args.ckpt_path)
