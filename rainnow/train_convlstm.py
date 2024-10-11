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

from rainnow.src.conv_lstm_utils import (
    IMERGDataset,
    plot_training_val_loss,
    save_checkpoint,
    train,
    validate,
)
from rainnow.src.models.conv_lstm import ConvLSTMModel
from rainnow.src.utilities.loading import load_imerg_datamodule_from_config
from rainnow.src.utilities.utils import generate_alphanumeric_id, get_device, get_logger

# ** data params **
BOXES = ["1,0"]
WINDOW = 1
HORIZON = 8
DT = 1
BATCH_SIZE = 8
NUM_WORKERS = 0

CKPT_BASE_PATH = "/Users/ds423/git_uni/irp-ds423/rainnow/results/"
# CKPT_BASE_PATH = "/teamspace/studios/this_studio/irp-ds423/rainnow/results/"
# CKPT_BASE_PATH = "/rds/general/user/ds423/home/rainnow/results/"

CONFIGS_BASE_PATH = "/Users/ds423/git_uni/irp-ds423/rainnow/src/dyffusion/configs/"
# CONFIGS_BASE_PATH = "/teamspace/studios/this_studio/irp-ds423/rainnow/src/dyffusion/configs/"
# CONFIGS_BASE_PATH = "/rds/general/user/ds423/home/rainnow/src/dyffusion/configs/"

# ** modelling params **
KERNEL_SIZE = (5, 5)
INPUT_DIMS = (1, 128, 128)  # C, H, W
HIDDEN_CHANNELS = [64, 64]
OUTPUT_CHANNELS = 1
NUM_LAYERS = 2
OUTPUT_SEQUENCE_LENGTH = 4
INPUT_SEQUENCE_LENGTH = 5
APPLY_BATCHNORM = True
CELL_DROPOUT = 0.3
OUTPUT_ACTIVATION = nn.Sigmoid()
# ** training params **
LR = 3e-4
WEIGHT_DECAY = 1e-5
SCHEDULER_FACTOR = 0.1
SCHEDULER_PATIENCE = 10
CRITERION = nn.BCELoss(reduction="mean")
NUM_EPOCHS = 2


def main(ckpt_path: str = None):
    """"""
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
            "boxes": BOXES,
            "window": WINDOW,
            "horizon": HORIZON,
            "prediction_horizon": HORIZON,
            "sequence_dt": DT,
        },
    )
    datamodule.setup("fit")

    # debug info.
    log.info("** datamodule params **")
    log.info(f"boxes={BOXES} | window={WINDOW} | horizon={HORIZON}.")

    # create datasets.
    train_dataset = IMERGDataset(
        datamodule, "train", sequence_length=INPUT_SEQUENCE_LENGTH, target_length=OUTPUT_SEQUENCE_LENGTH
    )
    val_dataset = IMERGDataset(
        datamodule,
        "validate",
        sequence_length=INPUT_SEQUENCE_LENGTH,
        target_length=OUTPUT_SEQUENCE_LENGTH,
    )

    # create dataloaders.
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=datamodule.hparams.batch_size,
        num_workers=NUM_WORKERS,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=datamodule.hparams.batch_size,
        num_workers=NUM_WORKERS,
        shuffle=False,
    )

    # instantiate model.
    model = ConvLSTMModel(
        input_sequence_length=INPUT_SEQUENCE_LENGTH,
        output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
        input_dims=INPUT_DIMS,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        num_layers=NUM_LAYERS,
        kernel_size=KERNEL_SIZE,
        output_activation=OUTPUT_ACTIVATION,
        apply_batchnorm=APPLY_BATCHNORM,
        cell_dropout=CELL_DROPOUT,
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
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )
    # debug info.
    log.info("** model params **")
    log.info(f"input={INPUT_SEQUENCE_LENGTH} | target={OUTPUT_SEQUENCE_LENGTH}.")
    log.info(f"inputs dims={INPUT_DIMS} | output channels={OUTPUT_CHANNELS}.")
    log.info(f"hidden channels={HIDDEN_CHANNELS} | num layers={NUM_LAYERS}.")
    log.info(f"kernel size={KERNEL_SIZE} | output activation={OUTPUT_ACTIVATION}.")
    log.info(f"batchnorm = {APPLY_BATCHNORM} | cell dropout={CELL_DROPOUT}.")
    log.info("** training params **")
    log.info(f"learning rate = {LR} | weight decay = {WEIGHT_DECAY} | num epochs = {NUM_EPOCHS}.")
    log.info(f"loss function = {CRITERION.__class__.__name__}.\n")

    # training and validation.
    model_save_path = Path(os.path.join(save_dir, f"{run_id}.pt"))
    train_losses, val_losses = [], []
    with tqdm(total=len(train_loader) * NUM_EPOCHS, desc="Training", unit="batch") as tepoch:
        for i in range(NUM_EPOCHS):
            logs = {}
            train_loss = train(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                criterion=CRITERION,
                tepoch=tepoch,
                curr_epoch=i,
                n_epochs=NUM_EPOCHS,
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
            if i == (NUM_EPOCHS - 1):
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
