"""
Temporal Convolutional Network for binary aggression prediction from wearable biosignal windows.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ============= Constants =============

N_INPUT_CHANNELS = 10       # BVP, EDA, ACC_X, ACC_Y, ACC_Z, Magnitude, HR, RMSSD, PHASIC, TONIC
DEFAULT_CHANNEL_LIST = [64, 64, 128, 128, 256, 256, 256, 256]
DEFAULT_KERNEL_SIZE = 7
DEFAULT_DROPOUT = 0.2
DEFAULT_READOUT = 'mean'    # 'last' | 'mean' | 'adaptive_max'

# Receptive field with defaults:
# RF = 1 + 2*(7-1)*(1+2+4+8+16+32+64+128) = 3061 samples
# Full window = 12*16*15 = 2880 samples; RF exceeds window at all readout positions.
#
# Readout default is 'mean' rather than 'last' because pre-aggression physiological
# buildup (EDA arousal, HR elevation) can begin minutes before onset and is
# distributed across the observation window rather than concentrated at the final
# timestep. Mean pooling weights all timesteps equally and is more robust to
# annotation timing variability in staff-coded aggression labels.


# ============= Helpers =============

class Chomp1d(nn.Module):
    """
    Removes trailing time steps introduced by causal left-padding.

    Parameters
    ----------
    chomp_size : int
        Number of trailing time steps to remove.
    """

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size]


def _make_causal_conv(
    n_in: int,
    n_out: int,
    kernel_size: int,
    padding: int,
    dilation: int,
) -> nn.Conv1d:
    """
    Constructs a weight-normalized Conv1d with causal padding.

    Parameters
    ----------
    n_in : int
    n_out : int
    kernel_size : int
    padding : int
        Symmetric padding; right side is removed by Chomp1d.
    dilation : int

    Returns
    -------
    conv : nn.Conv1d
    """
    conv = nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)
    conv.weight.data.normal_(0, 0.01)
    return weight_norm(conv)


# ============= Residual Block =============

class TemporalBlock(nn.Module):
    """
    Single TCN residual block: two dilated causal convolutions with weight normalization.

    Parameters
    ----------
    n_inputs : int
    n_outputs : int
    kernel_size : int
    dilation : int
    dropout : float
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        causal_pad = (kernel_size - 1) * dilation

        self.causal = nn.Sequential(
            _make_causal_conv(n_inputs, n_outputs, kernel_size, causal_pad, dilation),
            Chomp1d(causal_pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            _make_causal_conv(n_outputs, n_outputs, kernel_size, causal_pad, dilation),
            Chomp1d(causal_pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs else None
        )
        self.act_out = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_inputs, seq_len)

        Returns
        -------
        out : torch.Tensor, shape (batch, n_outputs, seq_len)
        """
        out = self.causal(x)
        res = self.downsample(x) if self.downsample is not None else x
        return self.act_out(out + res)


# ============= TCN Backbone =============

class TCN(nn.Module):
    """
    Stack of TemporalBlocks with exponentially increasing dilation.

    Dilation at block i = 2^i. Receptive field:
    RF = 1 + 2 * (kernel_size - 1) * sum(2^i for i in range(n_blocks)).

    Parameters
    ----------
    n_inputs : int
    channel_list : list of int
    kernel_size : int
    dropout : float
    """

    def __init__(
        self,
        n_inputs: int,
        channel_list: list,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dropout: float = DEFAULT_DROPOUT,
    ):
        super().__init__()
        layers = []
        for i, n_out in enumerate(channel_list):
            n_in = n_inputs if i == 0 else channel_list[i - 1]
            layers.append(
                TemporalBlock(n_in, n_out, kernel_size, dilation=2 ** i, dropout=dropout)
            )
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_inputs, seq_len)

        Returns
        -------
        out : torch.Tensor, shape (batch, channel_list[-1], seq_len)
        """
        return self.blocks(x)


# ============= Full Model =============

class AggressionTCN(nn.Module):
    """
    Full TCN model for binary aggression prediction.

    Architecture:
        1. Per-channel input projection (1x1 Conv): maps heterogeneous input
           channels to a uniform hidden width before temporal processing.
           Decouples amplitude-scale differences from temporal pattern learning.
        2. TCN backbone: stacked dilated causal residual blocks.
        3. Readout: collapses the temporal axis to a single vector.
        4. Linear classifier: single logit for BCEWithLogitsLoss.

    Input:  (batch, n_input_channels, seq_len)
            Default seq_len = 12 * 16 * 15 = 2880 (3-min window at 16Hz).
    Output: (batch, 1) -- raw logit, no sigmoid applied.

    Parameters
    ----------
    n_input_channels : int
        Number of biosignal channels. Must equal len(selected_feat) = 10.
    channel_list : list of int
        Per-block output channel sizes. Controls depth and width.
    kernel_size : int
    dropout : float
    readout : str
        Temporal aggregation strategy after the TCN backbone.
        'mean'         -- mean over all time steps. Default: pre-aggression
                          buildup is distributed across the window so uniform
                          temporal weighting is more appropriate than last-step.
        'last'         -- output at the final time step only.
        'adaptive_max' -- AdaptiveMaxPool1d(1), captures peak activation.
    """

    def __init__(
        self,
        n_input_channels: int = N_INPUT_CHANNELS,
        channel_list: list = None,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dropout: float = DEFAULT_DROPOUT,
        readout: str = DEFAULT_READOUT,
    ):
        super().__init__()
        channel_list = channel_list if channel_list is not None else DEFAULT_CHANNEL_LIST

        if readout not in ('last', 'mean', 'adaptive_max'):
            raise ValueError(
                f"readout must be 'last', 'mean', or 'adaptive_max', got '{readout}'"
            )

        self.input_proj = nn.Conv1d(n_input_channels, channel_list[0], kernel_size=1)
        self.tcn = TCN(channel_list[0], channel_list, kernel_size, dropout)

        self.readout = readout
        self._pool = nn.AdaptiveMaxPool1d(1) if readout == 'adaptive_max' else None

        self.classifier = nn.Linear(channel_list[-1], 1)

    @property
    def receptive_field(self) -> int:
        """Receptive field in input samples."""
        rf = 1
        for block in self.tcn.blocks:
            causal_pad = block.causal[1].chomp_size   # Chomp1d is index 1
            rf += 2 * causal_pad
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_input_channels, seq_len)

        Returns
        -------
        logits : torch.Tensor, shape (batch, 1)
            Raw logit; apply sigmoid for probability.
        """
        x = self.input_proj(x)              # (batch, channel_list[0], seq_len)
        x = self.tcn(x)                     # (batch, channel_list[-1], seq_len)

        if self.readout == 'last':
            x = x[:, :, -1]
        elif self.readout == 'mean':
            x = x.mean(dim=2)
        elif self.readout == 'adaptive_max':
            x = self._pool(x).squeeze(2)

        return self.classifier(x)           # (batch, 1)