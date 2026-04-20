"""
PatchTSTClassifier 
    classifier = wrappers.PatchTSTClassifier()
    classifier.set_params(**params)
    classifier.fit(train_X, train_y, test_X, test_y)
    classifier.save(prefix)
    classifier.load(prefix)
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt

import utils
from losses.bce import get_loss_fn
from networks.patchtst import AggPatchTST, build_patchtst_config


class PatchTSTClassifier:
    """
    Scikit-learn-style wrapper around AggPatchTST.
    Matches the save/load/fit/predict interface of CausalCNNEncoderClassifier.
    """

    def __init__(
        self,
        epochs=30,
        batch_size=32,
        lr=1e-4,
        weight_decay=1e-5,
        pos_weight=5.0,
        d_model=128,
        n_heads=8,
        n_layers=3,
        ffn_dim=256,
        dropout=0.1,
        head_dropout=0.1,
        cuda=False,
        gpu=0,
        use_focal=False,
        focal_alpha=0.25,
        focal_gamma=2.0,
        patch_len=64,
        patch_stride=32,
        use_onecycle=False,
        patience=5,
        channel_attention=False,
        seq_len=None,  # inferred from data if not provided

    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.head_dropout = head_dropout
        self.cuda = cuda
        self.gpu = gpu

        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.use_onecycle = use_onecycle
        self.patience = patience
        self.channel_attention = channel_attention
        self.seq_len = seq_len


        self.model = None
        self.device = torch.device(
            f'cuda:{gpu}' if cuda and torch.cuda.is_available() else 'cpu'
        )

    # sklearn-style API
    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        # update device in case cuda/gpu changed
        self.device = torch.device(
            f'cuda:{self.gpu}' if self.cuda and torch.cuda.is_available() else 'cpu'
        )
        return self

    def get_params(self):
        return {
            'epochs':       self.epochs,
            'batch_size':   self.batch_size,
            'lr':           self.lr,
            'weight_decay': self.weight_decay,
            'pos_weight':   self.pos_weight,
            'd_model':      self.d_model,
            'n_heads':      self.n_heads,
            'n_layers':     self.n_layers,
            'ffn_dim':      self.ffn_dim,
            'dropout':      self.dropout,
            'head_dropout': self.head_dropout,
            'cuda':         self.cuda,
            'gpu':          self.gpu,
            'use_focal':    self.use_focal,
            'focal_alpha':  self.focal_alpha,
            'focal_gamma':  self.focal_gamma,
            'patch_len':         self.patch_len,
            'patch_stride':      self.patch_stride,
            'use_onecycle':      self.use_onecycle,
            'patience':          self.patience,
            'channel_attention': self.channel_attention,
            'seq_len':           self.seq_len,
        }

    # Data helpers
    def _to_loader(self, X: np.ndarray, y: np.ndarray = None, shuffle=False):
        """
        X : [N, 10, 2880]  numpy float array
        y : [N]            numpy int/float array, or None
        """
        X_t = torch.from_numpy(X).float()
        if y is not None:
            y_t = torch.from_numpy(y).float()
            ds = TensorDataset(X_t, y_t)
        else:
            ds = TensorDataset(X_t)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

    # Training

    def fit(self, X, y, test, test_labels, prefix_file,
            save_memory=False, verbose=True):
        """
        Args:
            X            : np.ndarray [N_train, 10, 2880]
            y            : np.ndarray [N_train]
            test         : np.ndarray [N_test,  10, 2880]
            test_labels  : np.ndarray [N_test]
            prefix_file  : str  — path prefix for checkpoint saving
            save_memory  : ignored (kept for interface compatibility)
            verbose      : print epoch logs if True
        """
        cfg = build_patchtst_config(
            d_model=self.d_model, n_heads=self.n_heads,
            n_layers=self.n_layers, ffn_dim=self.ffn_dim,
            dropout=self.dropout, head_dropout=self.head_dropout,
            patch_len=self.patch_len, patch_stride=self.patch_stride,
            channel_attention=self.channel_attention,
            seq_len=self.seq_len if self.seq_len is not None else X.shape[2],
        )
        self.model = AggPatchTST(config=cfg).to(self.device)

        train_loader = self._to_loader(X, y, shuffle=True)
        val_loader = self._to_loader(test, test_labels)

        # pos_w = torch.tensor([self.pos_weight], device=self.device)
        # criterion = get_loss_fn(pos_weight=pos_w)
        if self.use_focal:
            from losses.focal import get_focal_loss
            criterion = get_focal_loss(alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            pos_w = torch.tensor([self.pos_weight], device=self.device)
            criterion = get_loss_fn(pos_weight=pos_w)

        optimizer = AdamW(self.model.parameters(),
                          lr=self.lr, weight_decay=self.weight_decay)
        # scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        if self.use_onecycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr,
                steps_per_epoch=len(train_loader), epochs=self.epochs
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)

        best_val_loss = float('inf')
        best_score = -1.0  # 0.5 * auroc + 0.5 * auprc
        best_metrics = {}
        epochs_no_improve = 0
        history = []

        for epoch in range(self.epochs):
            # train
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x_batch).prediction_logits.squeeze(1)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y_batch)
                if self.use_onecycle:
                    scheduler.step()
            train_loss /= len(train_loader.dataset)

            # validate
            val_loss, metrics = self._eval(val_loader, criterion)
            if not self.use_onecycle:
                scheduler.step()

            score = 0.5 * metrics['auroc'] + 0.5 * metrics['auprc']

            history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'auroc': metrics['auroc'],
                'f1': metrics['f1'],
                'auprc': metrics['auprc'],
                'score': score,
            })

            if verbose:
                print(
                    f"Epoch {epoch+1:03d}/{self.epochs} | "
                    f"train={train_loss:.4f} | val={val_loss:.4f} | "
                    f"AUROC={metrics['auroc']:.4f} | F1={metrics['f1']:.4f} | AUPRC={metrics['auprc']:.4f} | Score={score:.4f}"
                )

            if score > best_score:
                best_score = score
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_metrics = {
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'auroc': metrics['auroc'],
                    'f1': metrics['f1'],
                    'auprc': metrics['auprc'],
                    'score': score,
                }
                utils.save_checkpoint(
                    prefix_file + '_best.pt', self.model, optimizer,
                    epoch + 1, best_val_loss
                )
            else:
                epochs_no_improve += 1
                if self.patience > 0 and epochs_no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                    break

        with open(prefix_file + '_val_results.json', 'w') as fp:
            json.dump(best_metrics, fp, indent=2)

        with open(prefix_file + '_history.json', 'w') as fp:
            json.dump(history, fp, indent=2)

        self._plot_history(history, prefix_file)

        return self

    def _plot_history(self, history, prefix_file):
        epochs = [h['epoch'] for h in history]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epochs, [h['train_loss'] for h in history], label='train')
        axes[0].plot(epochs, [h['val_loss'] for h in history], label='val')
        axes[0].set_title('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()

        axes[1].plot(epochs, [h['auroc'] for h in history], color='tab:blue')
        axes[1].set_title('AUROC')
        axes[1].set_xlabel('Epoch')

        axes[2].plot(epochs, [h['auprc'] for h in history], color='tab:orange')
        axes[2].set_title('AUPRC')
        axes[2].set_xlabel('Epoch')

        fig.tight_layout()
        fig.savefig(prefix_file + '_curves.png', dpi=150)
        plt.close(fig)

    # Evaluation

    @torch.no_grad()
    def _eval(self, loader, criterion):
        self.model.eval()
        total_loss, all_logits, all_labels = 0.0, [], []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            logits = self.model(x_batch).prediction_logits.squeeze(1)
            total_loss += criterion(logits, y_batch).item() * len(y_batch)
            all_logits.append(logits.cpu())
            all_labels.append(y_batch.cpu())
        self.model.train()
        loss = total_loss / sum(len(b) for b in all_labels)
        metrics = utils.compute_metrics(
            torch.cat(all_logits), torch.cat(all_labels)
        )
        return loss, metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions [N]."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns sigmoid probabilities [N]."""
        loader = self._to_loader(X, shuffle=False)
        all_probs = []
        self.model.eval()
        with torch.no_grad():
            for (x_batch,) in loader:
                probs = self.model.predict_proba(x_batch.to(self.device))
                all_probs.append(probs.cpu().numpy())
        return np.concatenate(all_probs).squeeze()

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y.astype(int)))

    def save(self, prefix_file: str):
        torch.save(
            self.model.state_dict(),
            prefix_file + '_patchtst_encoder.pth'
        )

    def load(self, prefix_file: str):
        if self.model is None:
            cfg = build_patchtst_config(
                d_model=self.d_model, n_heads=self.n_heads,
                n_layers=self.n_layers, ffn_dim=self.ffn_dim,
                dropout=self.dropout, head_dropout=self.head_dropout,
                patch_len=self.patch_len, patch_stride=self.patch_stride,
            )
            self.model = AggPatchTST(config=cfg).to(self.device)

        map_loc = (lambda s, l: s.cuda(self.gpu)) if self.cuda else 'cpu'
        self.model.load_state_dict(
            torch.load(prefix_file + '_patchtst_encoder.pth',
                       map_location=map_loc)
        )
