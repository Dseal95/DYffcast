"""Configuration file for ConvLSTM trainer."""

from torch import nn

trainer = {
    # data params
    "data": {
        "boxes": ["0,0", "1,0", "2,0", "2,1"],
        "window": 1,
        "horizon": 8,
        "dt": 1,
        "batch_size": 12,
        "num_workers": 0,
    },
    # modelling params
    "model": {
        "kernel_size": (5, 5),
        "input_dims": (1, 128, 128),  # C, H, W
        "hidden_channels": [128, 128],
        "output_channels": 1,
        "num_layers": 2,
        "input_sequence_length": 4,
        "output_sequence_length": 1,
        "apply_batchnorm": True,
        "cell_dropout": 0.3,
        "output_activation": nn.Tanh(),  # nn.Sigmoid()
    },
    # training params
    "training": {
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "scheduler_factor": 0.1,
        "scheduler_patience": 5,
        "num_epochs": 30,
    },
}
