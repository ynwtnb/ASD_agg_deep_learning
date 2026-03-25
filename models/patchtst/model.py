"""
Wraps HuggingFace PatchTSTForClassification for aggression prediction.
#transformers.PatchTSTForClassification
Source: https://huggingface.co/docs/transformers/model_doc/patchtst

Parameters:
- num_input_channels = 10 (BVP, EDA, ACC_X/Y/Z, Magnitude, HR, RMSSD, PHASIC, TONIC)
- context_length = 2880 (180s * 16Hz)
- patch_length = 64 (4s per patch at 16Hz)
- patch_stride = 32 (50% overlap)
- num_targets = 1 (BCEWithLogicsLoss)
- use_cls_token = True

Input: past_values shape [batch, seq_len, n_channels] = [batch, 2880, 10]
Output: prediction_logits  shape [batch, 1]  (Raw logits)
"""

import torch
import torch.nn as nn
from transformers import PatchTSTConfig, PatchTSTForClassification

N_CHANNELS = 10
SEQ_LEN = 2880
PATCH_LEN = 64
PATCH_STRIDE = 32


def build_patchtst_config(
        n_channels=N_CHANNELS,
        seq_len=SEQ_LEN,
        patch_len=PATCH_LEN,
        patch_stride=PATCH_STRIDE,
        d_model=128,
        n_heads=8,
        n_layers=3,
        ffn_dim=256,
        dropout=0.1,
        head_dropout=0.1,
):
    return PatchTSTConfig(
        num_input_channels=n_channels,
        context_length=seq_len,
        patch_length=patch_len,
        patch_stride=patch_stride,
        d_model=d_model,
        num_attention_heads=n_heads,
        num_hidden_layers=n_layers,
        ffn_dim=ffn_dim,

        attention_dropout=dropout,
        positional_dropout=dropout,
        ff_dropout=dropout,
        head_dropout=head_dropout,

        num_targets=1,
        use_cls_token=True,

        norm_type='batchnorm',
        scaling='std',
    )


class AggPatchTST(nn.Module):
	"""
	Wraps PatchTSTForClassification model with:
	- task-specific forward interface
	- label passing via target_values
	- predict_proba() convenience method for evaluation
	"""

	def __init__(self, config: PatchTSTConfig = None, **kwargs):
			super().__init__()
			if config is None:
					config = build_patchtst_config(**kwargs) 
			self.config = config
			self.model = PatchTSTForClassification(config)

	def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
		"""
		Args:
				x (torch.Tensor): _description_
				labels (torch.Tensor, optional): _description_. Defaults to None.

		Returns:
				_type_: _description_
		"""
		return self.model(
				past_values=x,       
				target_values=labels, 
		)

	def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Args:
				x (torch.Tensor): _description_

		Returns:
				torch.Tensor: _description_
		"""
		self.eval()
		with torch.no_grad():
				logits = self.forward(x).prediction_logits
		return torch.sigmoid(logits)  

	def count_parameters(self) -> int:
		return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_loss_fn(pos_weight: torch.Tensor = None) -> nn.BCEWithLogitsLoss:
    """
    PatchTSTForClassification model use MSE loss by default. 
    Override with BCEWithLogitsLoss for binary classification task.

    Usage:
        criterion = get_loss_fn(pos_weight)
        loss = criterion(logits.squeeze(), labels)  
    """
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  
  
# sanity check:
if __name__ == '__main__':
    print("=== PatchTST Sanity Check ===\n")

    model = AggPatchTST()
    print(f"Trainable parameters: {model.count_parameters():,}\n")

    x = torch.randn(4, SEQ_LEN, N_CHANNELS)   # [4, 2880, 10]
    y = torch.randint(0, 2, (4,)).float()      # [4]

    print(f"Input shape:  {x.shape}")

    out    = model(x)
    logits = out.prediction_logits         
    print(f"Output shape: {logits.shape}")   

    criterion = get_loss_fn(pos_weight=torch.tensor([5.0]))
    loss      = criterion(logits.squeeze(), y)
    print(f"Loss:         {loss.item():.4f}")

    probs = model.predict_proba(x)
    print(f"Probs:        {probs.squeeze().tolist()}")
    print("\n✓ All checks passed.")
