"""Convolutional LSTM module. Recreated from https://arxiv.org/pdf/1506.04214."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTMCell module.

    Parameters
    ----------
    input_channels : Tuple[int, int, int]
        The number of channels (C, _, _) of each image in the input image sequence.
    input_height : Tuple[int, int, int]
        The height (_, H, _) of each image in the input image sequence.
    input_width : Tuple[int, int, int]
        The width (_, _, W) of each image in the input image sequence.
    hidden_channels:  int
        The number of hidden channels. This is the number of output channels in the convolution(i, h).
    kernel_size: Tuple[int]
        The size of the kernel to perform the convolutional operations.
    apply_batchnorm : bool, optional
        If `True`, a batchnorm will be applied after each convolution(i, h) in each ConvLSTM cell. Default is `True`.
    cell_dropout : float
        Dropout applied in the convolutional layer into the cell. Default is no dropout, 0.0.
    bias : bool, optional
        If `True`, the LSTM layers will use bias weights. Default is `True`.
    device : str
        The device (CPU or GPU) on which the model will be run. Should be specified as a string ('cpu' or 'cuda'). Default is 'cpu'.
    """

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        hidden_channels: int,
        kernel_size: int,
        apply_batchnorm: bool = True,
        cell_dropout: float = 0.0,
        bias: bool = True,
        device: str = "cpu",
        _print: bool = False,
    ):
        """Initialisation."""
        super(ConvLSTMCell, self).__init__()

        # inputs.
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.apply_batchnorm = apply_batchnorm
        self.device = device
        self.print = _print
        self.cell_dropout = cell_dropout

        # conv layer into cell.
        self.conv_input_and_hidden_to_cell = nn.Sequential(
            nn.Conv2d(
                in_channels=(self.input_channels + self.hidden_channels),
                out_channels=(4 * self.hidden_channels),
                kernel_size=self.kernel_size,
                padding="same",  # (1, 1),
                bias=self.bias,
            ),
            (
                nn.BatchNorm2d(num_features=(4 * self.hidden_channels))
                if self.apply_batchnorm
                else nn.Identity()
            ),
            nn.Dropout(p=self.cell_dropout),
        )

        # initialise the weights (and biases) of conv layers and for Hadamard Products.
        self.apply(self.__init_weights)
        self.Wci, self.Wcf, self.Wco = self._init_hadamard_products_weights()

    @staticmethod  # taken from: https://github.com/pytorch/examples/blob/main/dcgan/main.py.
    def __init_weights(module):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.weight.data.normal_(0.0, 0.02)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _init_hadamard_products_weights(self):
        """Initialise the weights used in the Hadamard Product details in - https://arxiv.org/abs/1506.04214.

        These are initialised w/ dims: (1, hidden_channels, H, W). The weights are initialised so that each cell
        within each layer shares the same Wci, Wcf, and Wco weights but these are distinct from those in other layers.
        There will be only 1 set of trainable Wci, Wcf and Wco per layer.
        """
        Wci = torch.zeros(
            1,
            self.hidden_channels,
            self.input_height,
            self.input_width,
            requires_grad=True,
        ).to(self.device)
        Wcf = torch.zeros(
            1,
            self.hidden_channels,
            self.input_height,
            self.input_width,
            requires_grad=True,
        ).to(self.device)
        Wco = torch.zeros(
            1,
            self.hidden_channels,
            self.input_height,
            self.input_width,
            requires_grad=True,
        ).to(self.device)

        return Wci, Wcf, Wco

    def forward(self, i: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        """Forward pass through the ConvLSTM Cell.

        Parameters
        ----------
        i : torch.Tensor
            input image into ConvLSTM cell w/ dims: (batch_size, channels, height, width)
        h : torch.Tensor
            input hidden state into ConvLSTM cell w/ dims: (batch_size, channels, height, width)
        c : torch.Tensor
            input cell state into ConvLSTM cell w/ dims: (batch_size, channels, height, width)

        Returns
        -------
        h_t: torch.Tensor
            output hidden state w/ dims: (batch_size, channels, height, width)
        c_t: torch.Tensor
            output cell state w/ dims: (batch_size, channels, height, width)
        """
        # bottom left of the LSTM cell (see docstrings diagram). This is performing a convolution on the combined input i(t) with h(t-1).
        gates = self.conv_input_and_hidden_to_cell(torch.cat([i, h], dim=1))

        # separate the output into each of cell gate operations.
        input_gate, forget_gate, candidate_update, output_gate = gates.chunk(4, dim=1)

        # apply the corresponding activations [details in - https://arxiv.org/abs/1506.04214]
        # it = σ((Wxi * Xt + Whi * Ht−1) + (Wci o Ct−1) + bi)
        i_t = torch.sigmoid(input_gate + (self.Wci * c))

        # ft = σ((Wxf * Xt + Whf * Ht−1) + (Wcf o Ct−1) + bf)
        f_t = torch.sigmoid(forget_gate + (self.Wcf * c))

        # Ct = (ft o Ct−1) + (it o tanh((Wxc * Xt + Whc * Ht−1) + bc))
        cd_t = (c * f_t) + (i_t * torch.tanh(candidate_update))

        # ot = σ((Wxo * Xt + Who * Ht−1) + (Wco o Ct) + bo)
        o_t = torch.sigmoid(output_gate + (self.Wco * c))

        # calculate the next cell state and hidden state.
        c_t = (f_t * c) + (i_t * cd_t)
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class ConvLSTM(nn.Module):
    """ConvLSTM module.

    Parameters
    ----------
    input_sequence_length : int
        The length of the input image sequence. Determines how many ConvLSTM cells are in each layes - 1 cell per element in sequence.
    input_dims : Tuple[int, int, int]
        The (C, H, W) [number of channels, height, width] of each image in the input image sequence.
    hidden_channels:  List[int]
        A list of the number of channels for the convolutional output for each layer. There should be a hidden_channel values for each layer.
    output_channels:  int
        The desired number of channels in the convoled hidden outputs from the cells in the last layer.
    num_layers : int
        Number of stacked ConvLSTM layers.Each ConvLSTM layer, m will consist of n ConvLSTMCells where n = input seuqnce length.
    kernel_size: Tuple[int]
        The size of the kernel to perform the convolutional operations.
    apply_batchnorm : bool, optional
        If `True`, a batchnorm will be applied after each convolution(i, h) in each ConvLSTM cell. Default is `True`.
    cell_dropout : float
        Dropout applied in the convolutional layer into the cell. Default is no dropout, 0.0.
    bias : bool, optional
        If `True`, the LSTM layers will use bias weights. Default is `True`.
    device : str
        The device (CPU or GPU) on which the model will be run. Should be specified as a string ('cpu' or 'cuda'). Default is 'cpu'.
    _print : bool, optional
        If `False`, enables printing of model-related information for debugging or logging. Default is `False`.
    """

    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        input_dims: Tuple[int, int, int],
        hidden_channels: List[int],
        output_channels: int,
        num_layers: int,
        kernel_size: Tuple[int],
        output_activation: Optional[nn.Module] = None,
        apply_batchnorm: bool = True,
        cell_dropout: float = 0.0,
        bias: bool = True,
        device: str = "cpu",
    ):
        """Initialisation."""
        super(ConvLSTM, self).__init__()

        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.input_dims = input_dims
        self.input_channels, self.input_height, self.input_width = input_dims
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_activation = output_activation
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.apply_batchnorm = apply_batchnorm
        self.cell_dropout = cell_dropout
        self.bias = bias
        self.device = device

        assert len(self.hidden_channels) == self.num_layers

        # create a list of lstm cell modules
        self.conv_lstm_cell_list = nn.ModuleList()
        for l in range(self.num_layers):
            _cell = ConvLSTMCell(
                input_channels=(self.input_channels if l == 0 else self.hidden_channels[l - 1]),
                input_height=self.input_height,
                input_width=self.input_width,
                hidden_channels=self.hidden_channels[l],
                kernel_size=self.kernel_size,
                apply_batchnorm=self.apply_batchnorm,
                cell_dropout=self.cell_dropout,
                bias=self.bias,
                device=self.device,
            )
            self.conv_lstm_cell_list.append(_cell)

        # create a final convolutional layer from hidden state to network output
        self.conv_hidden_to_output = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_channels[-1],  # take the hidden channels from the last layer.
                out_channels=self.output_channels,
                kernel_size=self.kernel_size,
                padding="same",  # (1, 1),
                bias=self.bias,
            ),
            self.output_activation if self.output_activation else nn.Identity(),
        )

        # # initilaise weights of conv layers.
        self.apply(self.__init_weights)

    @staticmethod  # taken from: https://github.com/pytorch/examples/blob/main/dcgan/main.py.
    def __init_weights(module):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.weight.data.normal_(0.0, 0.02)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _init_hidden_and_cell_states(self, batch_size):
        """Initialise the _hidden_states and _cell_states that will go into the 1st ConvLSTM cells in each layer.

        _states are initialised to zero w/ dims: (batch_size, hidden_channels, input_image_height, input_image_width).
        The position of the _state in the list corresponds to the layer.
        """
        hs, cs = [], []
        for _layer in range(self.num_layers):
            hs.append(
                torch.zeros(
                    batch_size,
                    self.hidden_channels[_layer],
                    self.input_height,
                    self.input_width,
                    requires_grad=False,
                ).to(self.device)
            )
            cs.append(
                torch.zeros(
                    batch_size,
                    self.hidden_channels[_layer],
                    self.input_height,
                    self.input_width,
                    requires_grad=False,
                ).to(self.device)
            )

        return hs, cs

    def forward(self, x: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor):
        """Forward pass through the ConvLSTM.

        Parameters:
        ----------
        x : torch.Tensor
            The input image sequence to each cell in the 1st layer w/ dims: (batch_size, sequence_length, num_channels, height, width)
        h0 : torch.Tensor
            The input hidden states (for each layer) w/ dims: (batch_size, sequence_length, num_channels, height, width)
        c0 : torch.Tensor
            The input cell states (for each layer) w/ dims: (batch_size, sequence_length, num_channels, height, width)

        Returns
        -------
        torch.Tensor
            The output of each ConvLSTMCell in the last layer.

        Workflow:
        ---------
                                                output1                            output2                                outputN
                                                   ^                                  ^                                      ^
                                                   | (hN,1)                           | (hN,2)                               | (hN,n)
        LayerN:    (hN,0, cn,0)--> [(C(h2,1, hN,0)) ] -> (hn,1, cn,1) [(C(h2,2, hN,1)) ] -> (hn,2, cn,2) ... [(C(h2,n, hN,n)) ]
                                                   ^                                   ^                                      ^
                                                   | (h2,1)                            | (h2,2)                               | (h2,n)
        Layer2:   (h2,0, c2,0)--> [(C(h1,1, h2,0)).. ] -> (h2,1, c2,1) [(C(h1,2, h2,1)).. ] -> (h2,2, c2,2) ... [(C(h1,n, h2,2)).. ]
                                                 ^                                  ^                                     ^
                                                 | (h1,1)                           | (h1,2)                              | (h1,n)
        Layer1: (h1,0, c1,0) --> [(C(i1, h1,0))..  ] -> (h1,1, c1,1) [(C(i1, h1,0)..  ] -> (h1,2, c1,2) ... [C(iN, h1,2)..  ]
                                        ^                                   ^                                     ^
                                        |                                   |                                     |
                                      input1                               input2                               inputN
        """
        _hidden_states = []
        _cell_states = []
        for layer in range(self.num_layers):
            _hidden_states.append(h0[layer])
            _cell_states.append(c0[layer])

        _outputs = []
        for t in range(x.size(1)):  # loop over each element in the input sequence.
            for layer in range(self.num_layers):  # loop over each layer in the network.
                if layer == 0:
                    # each cell in the 1st layer takes in the raw element from the input sequence + hidden state and a cell state.
                    hidden_l, cell_l = self.conv_lstm_cell_list[layer](
                        i=x[:, t, :, :, :],
                        h=_hidden_states[layer],
                        c=_cell_states[layer],
                    )
                else:
                    # each cell in > 1st layer takes in 2 hidden states as input and hidden and a cell state.
                    hidden_l, cell_l = self.conv_lstm_cell_list[layer](
                        i=_hidden_states[layer - 1],
                        h=_hidden_states[layer],
                        c=_cell_states[layer],
                    )

                # update the hidden and cell states.
                _hidden_states[layer] = hidden_l
                _cell_states[layer] = cell_l

            # the output from each cell in the last layer.
            _outputs.append(hidden_l)

        assert len(_outputs) == self.input_sequence_length

        # take only desired number of outputs.
        _output_seq = _outputs[-self.output_sequence_length :]

        # apply a convolution from each output in the last layer to the desired number of channels.
        output = torch.stack([self.conv_hidden_to_output(_out) for _out in _output_seq], dim=1)

        return output


class ConvLSTMModel(nn.Module):
    """Wrapper module for the ConvLSTM.

    Parameters
    ----------
    input_sequence_length : int
        The length of the input image sequence. Determines how many ConvLSTM cells are in each layes - 1 cell per element in sequence.
    output_sequence_length : int
        The desired length of the output image sequence. Return last output_sequence_length number of convolved hidden outputs from the last layer.
    input_dims : Tuple[int, int, int]
        The (C, H, W) [number of channels, height, width] of each image in the input image sequence.
    hidden_channels:  List[int]
        A list of the number of channels for the convolutional output for each layer. There should be a hidden_channel values for each layer.
    output_channels:  int
        The desired number of channels in the convoled hidden outputs from the cells in the last layer.
    num_layers : int
        Number of stacked ConvLSTM layers.Each ConvLSTM layer, m will consist of n ConvLSTMCells where n = input seuqnce length.
    kernel_size: Tuple[int]
        The size of the kernel to perform the convolutional operations.
    apply_batchnorm : bool, optional
        If `True`, a batchnorm will be applied after each convolution(i, h) in each ConvLSTM cell. Default is `True`.
    cell_dropout : float
        Dropout applied in the convolutional layer into the cell. Default is no dropout, 0.0.
    bias : bool, optional
        If `True`, the LSTM layers will use bias weights. Default is `True`.
    device : str
        The device (CPU or GPU) on which the model will be run. Should be specified as a string ('cpu' or 'cuda'). Default is 'cpu'.
    _print : bool, optional
        If `True`, enables printing of model-related information for debugging or logging. Default is `False`.
    """

    def __init__(
        self,
        input_sequence_length: int,
        output_sequence_length: int,
        input_dims: Tuple[int, int, int],
        hidden_channels: List[int],
        output_channels: int,
        num_layers: int,
        kernel_size: Tuple[int],
        output_activation: nn.Module,
        apply_batchnorm: bool = True,
        cell_dropout: float = 0.0,
        bias: bool = True,
        device: str = "cpu",
        _print: bool = False,
    ):
        """Initialisation."""
        super(ConvLSTMModel, self).__init__()

        # inputs.
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        assert self.output_sequence_length <= self.input_sequence_length

        self.input_dims = input_dims
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.output_activation = output_activation
        self.apply_batchnorm = apply_batchnorm
        self.cell_dropout = cell_dropout
        self.bias = bias
        self.device = device

        # output sequence can't be longer than the input.
        assert self.output_sequence_length <= self.input_sequence_length

        # create the ConvLSTM network.
        self.convlstm = ConvLSTM(
            input_sequence_length=self.input_sequence_length,
            output_sequence_length=self.output_sequence_length,
            input_dims=self.input_dims,
            hidden_channels=self.hidden_channels,
            output_channels=self.output_channels,
            output_activation=self.output_activation,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            apply_batchnorm=self.apply_batchnorm,
            cell_dropout=self.cell_dropout,
            bias=self.bias,
            device=self.device,
        )

    def forward(self, x: torch.Tensor):
        """Forward pass through the Convolution LSTM Model.

        Parameters:
        ----------
        x : torch.Tensor
            The input image sequence with dims: (batch_size, sequence_length, num_channels, height, width).

        Returns
        -------
        torch.Tensor
            Return the last <self.output_sequence_length> convolved hidden outputs after applying nn.Sigmoid() to make sure pixels are [0,1].

            output w/ dims: (batch_size, output_sequence_length, num_channels, height, width).
        """
        # create the initial h0 and c0 for input into each of the ConvLSTM layers
        h0, c0 = self.convlstm._init_hidden_and_cell_states(batch_size=x.size(0))
        output = self.convlstm(x=x, h0=h0, c0=c0)

        return output
