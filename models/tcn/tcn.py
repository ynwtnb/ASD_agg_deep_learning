"""
Temporal Convolutional Network for binary aggression prediction from wearable biosignal windows.
"""

import torch
import torch.nn as nn


# ============= Constants =============

N_INPUT_CHANNELS = 10       # BVP, EDA, ACC_X, ACC_Y, ACC_Z, Magnitude, HR, RMSSD, PHASIC, TONIC
DEFAULT_CHANNEL_LIST = [64, 64, 128, 128, 256, 256, 256, 256]
DEFAULT_KERNEL_SIZE = 7
DEFAULT_DROPOUT = 0.2
DEFAULT_READOUT = 'mean'    # 'last' | 'mean' | 'adaptive_max'
DEFAULT_N_GROUPS = 8


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


def _n_groups_for(n_channels, target_groups=DEFAULT_N_GROUPS):
    """
    Returns largest divisor of n_channels that is <= target_groups.

    GroupNorm requires n_channels % n_groups == 0. This helper finds a
    valid group count close to the target without requiring channel widths
    to be exact multiples of DEFAULT_N_GROUPS.

    Parameters
    ----------
    n_channels : int
    target_groups : int

    Returns
    -------
    n_groups : int
    """
    for g in range(target_groups, 0, -1):
        if n_channels % g == 0:
            return g
    return 1


# ============= Residual Block =============

class TemporalBlock(nn.Module):
    """
    Single TCN residual block with GroupNorm for activation normalization.

    Architecture per block:
        Conv1d -> Chomp1d -> GroupNorm -> ReLU -> Dropout ->
        Conv1d -> Chomp1d -> GroupNorm -> ReLU -> Dropout

    GroupNorm normalizes per-instance across channel groups, making it
    independent of batch composition. BatchNorm was unstable on this dataset
    because positive instances (pre-aggression arousal) and negative instances
    (baseline physiology) have different signal distributions, causing volatile
    per-batch statistics.

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
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=causal_pad, dilation=dilation),
            Chomp1d(causal_pad),
            nn.GroupNorm(_n_groups_for(n_outputs), n_outputs),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=causal_pad, dilation=dilation),
            Chomp1d(causal_pad),
            nn.GroupNorm(_n_groups_for(n_outputs), n_outputs),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1)
            if n_inputs != n_outputs else None
        )
        self.act_out = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        """Kaiming initialization for Conv1d layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
        1. Per-channel input projection (1x1 Conv + GroupNorm + ReLU): maps
           heterogeneous input channels to a uniform hidden width.
        2. TCN backbone: stacked dilated causal residual blocks with GroupNorm.
        3. Readout: collapses the temporal axis to a single vector.
        4. Linear classifier: single logit for BCEWithLogitsLoss.

    Input:  (batch, n_input_channels, seq_len)
            Default seq_len = 12 * 16 * 15 = 2880 (3-min window at 16Hz).
    Output: (batch, 1) -- raw logit, no sigmoid applied.

    Parameters
    ----------
    n_input_channels : int
        Number of biosignal channels.
    channel_list : list of int
        Per-block output channel sizes. Controls depth and width.
    kernel_size : int
    dropout : float
    readout : str
        'mean'         -- mean over all time steps (default).
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

        self.input_proj = nn.Sequential(
            nn.Conv1d(n_input_channels, channel_list[0], kernel_size=1),
            nn.GroupNorm(_n_groups_for(channel_list[0]), channel_list[0]),
            nn.ReLU(),
        )
        nn.init.kaiming_normal_(self.input_proj[0].weight, nonlinearity='relu')
        nn.init.zeros_(self.input_proj[0].bias)

        self.tcn = TCN(channel_list[0], channel_list, kernel_size, dropout)

        self.readout = readout
        self._pool = nn.AdaptiveMaxPool1d(1) if readout == 'adaptive_max' else None

        self.classifier = nn.Linear(channel_list[-1], 1)

    @property
    def receptive_field(self) -> int:
        """Receptive field in input samples."""
        rf = 1
        for block in self.tcn.blocks:
            for m in block.causal.modules():
                if isinstance(m, Chomp1d):
                    rf += m.chomp_size
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
        x = self.input_proj(x)
        x = self.tcn(x)

        if self.readout == 'last':
            x = x[:, :, -1]
        elif self.readout == 'mean':
            x = x.mean(dim=2)
        elif self.readout == 'adaptive_max':
            x = self._pool(x).squeeze(2)

        return self.classifier(x)