"""
methodology.py
==============
Complete implementation of the IoT Robotic Handwriting Arm methodology.

Covers all 5 stages + Dataset Training:
  0. IAM Dataset Loading, Parsing & Training Loop
  1. AI-Based Handwriting Stroke Generation  (Graves LSTM + MDN)
  2. Stroke Representation & Pen Control Logic
  3. Coordinate Normalisation & Workspace Mapping
  4. Mathematical Modelling of the Robotic Arm (Forward Kinematics)
  5. Inverse Kinematics Formulation

Pair this file with:
  - coap_server.py    (IoT communication layer)
  - coap_client_ui.py (Tkinter GUI)

IAM Dataset:
  Download from: https://fki.iam.unibe.ch/databases/iam-on-line-handwriting-database
  Register for free, then download:
    - lineStrokes-all.tar.gz  → extract to  ./data/lineStrokes/
    - ascii-all.tar.gz        → extract to  ./data/ascii/

  Folder structure expected:
    ./data/
      lineStrokes/
        a01/
          a01-000u/
            a01-000u-01.xml
            ...
      ascii/
        a01/
          a01-000u.txt
          ...

Quick start:
    # 1. Train the model
    python methodology.py --train --epochs 50 --save model.pt

    # 2. Generate strokes with trained weights
    python methodology.py --generate --load model.pt --text "HELLO"

Dependencies:
    pip install numpy torch tqdm
"""

import os
import math
import glob
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):          # fallback no-op wrapper
        return x

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — IAM DATASET LOADING, PARSING & TRAINING
#
# The IAM On-Line Handwriting Database contains handwriting captured as
# sequences of pen strokes in XML format. Each stroke is a series of (x, y)
# points. We convert these into (Δx, Δy, pen_state) offset sequences.
#
# Dataset structure:
#   lineStrokes/<writer>/<form>/<line>.xml
#     <StrokeSet>
#       <Stroke>
#         <Point x="..." y="..." time="..."/>
#         ...
#       </Stroke>
#     </StrokeSet>
#
# Training recipe:
#   1. Parse XML → list of absolute (x, y) + pen states
#   2. Convert to offset sequences (Δx, Δy, pen)
#   3. Normalise offsets to zero-mean unit-variance
#   4. Batch + pad → feed into GravesHandwritingModel
#   5. Minimise mdn_loss, checkpoint best model
# ═══════════════════════════════════════════════════════════════════════════════

def parse_iam_xml(xml_path: str) -> list:
    """
    Parse a single IAM lineStrokes XML file into a list of
    absolute (x, y, pen_state) tuples.

    pen_state = 1  → pen is on paper (within a stroke)
    pen_state = 0  → pen is lifted   (transition between strokes)

    Parameters
    ----------
    xml_path : path to .xml file

    Returns
    -------
    points : list of (x, y, pen_state)  — empty list on parse error
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    points = []
    stroke_set = root.find('StrokeSet')
    if stroke_set is None:
        return []

    strokes = stroke_set.findall('Stroke')
    for s_idx, stroke in enumerate(strokes):
        pts = stroke.findall('Point')
        for p_idx, pt in enumerate(pts):
            x = float(pt.get('x', 0))
            y = float(pt.get('y', 0))
            # Pen is down for all points inside a stroke
            # Pen lifts at the very last point of each stroke
            is_last_in_stroke = (p_idx == len(pts) - 1)
            pen = 0 if is_last_in_stroke else 1
            points.append((x, y, pen))

    return points


def absolute_to_offsets(points: list) -> np.ndarray:
    """
    Convert absolute (x, y, pen) coordinates to offset sequences (Δx, Δy, pen).

    This is the representation used by the Graves model — the network learns
    to predict relative movements rather than absolute positions, making it
    scale- and position-invariant.

    Parameters
    ----------
    points : list of (x, y, pen_state)

    Returns
    -------
    offsets : np.ndarray shape (N-1, 3) — columns: [Δx, Δy, pen]
    """
    if len(points) < 2:
        return np.zeros((0, 3), dtype=np.float32)

    arr = np.array(points, dtype=np.float32)           # (N, 3)
    dx  = np.diff(arr[:, 0])                           # (N-1,)
    dy  = np.diff(arr[:, 1])                           # (N-1,)
    pen = arr[:-1, 2]                                  # pen state of source point

    return np.stack([dx, dy, pen], axis=1)             # (N-1, 3)


def compute_dataset_stats(data_dir: str) -> tuple:
    """
    Compute mean and standard deviation of (Δx, Δy) offsets across
    the entire IAM dataset. Used for standardisation before training.

    Parameters
    ----------
    data_dir : path to lineStrokes directory

    Returns
    -------
    (mean_dx, std_dx, mean_dy, std_dy) : floats
    """
    all_dx, all_dy = [], []

    xml_files = glob.glob(os.path.join(data_dir, '**', '*.xml'), recursive=True)
    print(f"  Computing stats over {len(xml_files)} XML files...")

    for fpath in xml_files:
        pts = parse_iam_xml(fpath)
        if len(pts) < 2:
            continue
        offsets = absolute_to_offsets(pts)
        all_dx.append(offsets[:, 0])
        all_dy.append(offsets[:, 1])

    if not all_dx:
        return 0.0, 1.0, 0.0, 1.0

    all_dx = np.concatenate(all_dx)
    all_dy = np.concatenate(all_dy)

    return (float(all_dx.mean()), float(all_dx.std() + 1e-8),
            float(all_dy.mean()), float(all_dy.std() + 1e-8))


class IAMStrokeDataset(Dataset):
    """
    PyTorch Dataset for the IAM On-Line Handwriting Database.

    Each sample is a tensor of shape (seq_len, 3) representing:
      column 0 : Δx  (normalised x offset)
      column 1 : Δy  (normalised y offset)
      column 2 : pen state (0 = up, 1 = down)

    Parameters
    ----------
    data_dir    : path to lineStrokes/ directory
    max_seq_len : truncate sequences longer than this (default 700)
    min_seq_len : discard sequences shorter than this (default 10)
    stats       : (mean_dx, std_dx, mean_dy, std_dy) for normalisation.
                  If None, computed automatically from the dataset.
    """

    def __init__(
        self,
        data_dir:    str,
        max_seq_len: int   = 700,
        min_seq_len: int   = 10,
        stats:       tuple = None,
    ):
        self.max_seq_len = max_seq_len
        self.sequences   = []

        xml_files = glob.glob(
            os.path.join(data_dir, '**', '*.xml'), recursive=True
        )
        print(f"  Found {len(xml_files)} XML stroke files in '{data_dir}'")

        # Compute normalisation statistics if not provided
        if stats is None:
            self.mean_dx, self.std_dx, self.mean_dy, self.std_dy = \
                compute_dataset_stats(data_dir)
        else:
            self.mean_dx, self.std_dx, self.mean_dy, self.std_dy = stats

        print(f"  Normalisation stats → "
              f"Δx: μ={self.mean_dx:.3f} σ={self.std_dx:.3f} | "
              f"Δy: μ={self.mean_dy:.3f} σ={self.std_dy:.3f}")

        # Parse and build sequence list
        skipped = 0
        for fpath in tqdm(xml_files, desc='  Loading IAM strokes'):
            pts = parse_iam_xml(fpath)
            if len(pts) < min_seq_len + 1:
                skipped += 1
                continue

            offsets = absolute_to_offsets(pts)     # (N-1, 3)

            # Normalise Δx, Δy to zero-mean unit-variance
            offsets[:, 0] = (offsets[:, 0] - self.mean_dx) / self.std_dx
            offsets[:, 1] = (offsets[:, 1] - self.mean_dy) / self.std_dy

            # Truncate to max length
            offsets = offsets[:max_seq_len]

            self.sequences.append(
                torch.tensor(offsets, dtype=torch.float32)
            )

        print(f"  Loaded {len(self.sequences)} sequences "
              f"({skipped} skipped — too short).\n")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]

    def get_stats(self) -> tuple:
        """Return (mean_dx, std_dx, mean_dy, std_dy) for use at inference time."""
        return self.mean_dx, self.std_dx, self.mean_dy, self.std_dy


def collate_strokes(batch: list) -> tuple:
    """
    Custom collate function for variable-length stroke sequences.

    Pads all sequences in the batch to the length of the longest one,
    and creates a boolean padding mask.

    Parameters
    ----------
    batch : list of Tensors, each shape (seq_len_i, 3)

    Returns
    -------
    padded  : Tensor [batch_size, max_seq_len, 3]   — padded sequences
    targets : Tensor [batch_size, max_seq_len, 3]   — next-step targets
    lengths : list of int                           — original lengths
    """
    # Input  = sequence[:-1],  Target = sequence[1:]  (next-step prediction)
    inputs  = [seq[:-1] for seq in batch]
    targets = [seq[1:]  for seq in batch]
    lengths = [len(s) for s in inputs]

    padded_inputs  = pad_sequence(inputs,  batch_first=True, padding_value=0.0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)

    return padded_inputs, padded_targets, lengths


def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimiser:  torch.optim.Optimizer,
    device:     torch.device,
    clip_grad:  float = 10.0,
) -> float:
    """
    Run one full training epoch.

    Parameters
    ----------
    model     : GravesHandwritingModel
    loader    : DataLoader yielding (inputs, targets, lengths)
    optimiser : e.g. Adam
    device    : 'cpu' or 'cuda'
    clip_grad : gradient clipping max norm

    Returns
    -------
    avg_loss : mean batch loss for this epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, lengths in tqdm(loader, desc='    batches', leave=False):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        optimiser.zero_grad()

        # Forward pass — no teacher forcing (free-running during training)
        (pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen), _ = model(inputs)

        # Only compute loss on non-padded positions
        max_len = inputs.size(1)
        mask = torch.zeros(inputs.size(0), max_len, dtype=torch.bool, device=device)
        for i, l in enumerate(lengths):
            mask[i, :l] = True

        # Apply mask — zero out padded positions
        pi_m      = pi[mask]
        mu_x_m    = mu_x[mask]
        mu_y_m    = mu_y[mask]
        sigma_x_m = sigma_x[mask]
        sigma_y_m = sigma_y[mask]
        rho_m     = rho[mask]
        pen_m     = pen[mask]
        tgt_m     = targets[mask]

        # Add batch & seq dims back for mdn_loss (expects [..., K])
        loss = mdn_loss(
            pi_m.unsqueeze(0).unsqueeze(0),
            mu_x_m.unsqueeze(0).unsqueeze(0),
            mu_y_m.unsqueeze(0).unsqueeze(0),
            sigma_x_m.unsqueeze(0).unsqueeze(0),
            sigma_y_m.unsqueeze(0).unsqueeze(0),
            rho_m.unsqueeze(0).unsqueeze(0),
            pen_m.unsqueeze(0).unsqueeze(0),
            tgt_m.unsqueeze(0).unsqueeze(0),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()

        total_loss  += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate model on a validation split.

    Returns
    -------
    avg_val_loss : float
    """
    model.eval()
    total_loss  = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, targets, lengths in loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            (pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen), _ = model(inputs)

            max_len = inputs.size(1)
            mask = torch.zeros(inputs.size(0), max_len, dtype=torch.bool,
                               device=device)
            for i, l in enumerate(lengths):
                mask[i, :l] = True

            pi_m      = pi[mask]
            mu_x_m    = mu_x[mask]
            mu_y_m    = mu_y[mask]
            sigma_x_m = sigma_x[mask]
            sigma_y_m = sigma_y[mask]
            rho_m     = rho[mask]
            pen_m     = pen[mask]
            tgt_m     = targets[mask]

            loss = mdn_loss(
                pi_m.unsqueeze(0).unsqueeze(0),
                mu_x_m.unsqueeze(0).unsqueeze(0),
                mu_y_m.unsqueeze(0).unsqueeze(0),
                sigma_x_m.unsqueeze(0).unsqueeze(0),
                sigma_y_m.unsqueeze(0).unsqueeze(0),
                rho_m.unsqueeze(0).unsqueeze(0),
                pen_m.unsqueeze(0).unsqueeze(0),
                tgt_m.unsqueeze(0).unsqueeze(0),
            )
            total_loss  += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train_model(
    data_dir:    str,
    save_path:   str  = 'model.pt',
    epochs:      int  = 50,
    batch_size:  int  = 32,
    lr:          float = 1e-3,
    hidden_size: int  = 400,
    num_layers:  int  = 2,
    num_mixtures:int  = 20,
    max_seq_len: int  = 700,
    val_split:   float= 0.1,
    patience:    int  = 5,
) -> 'GravesHandwritingModel':
    """
    Full training pipeline for the Graves Handwriting Model on IAM data.

    Parameters
    ----------
    data_dir    : path to lineStrokes/ directory
    save_path   : where to save the best model checkpoint (.pt)
    epochs      : number of training epochs
    batch_size  : mini-batch size
    lr          : Adam learning rate
    hidden_size : LSTM hidden units
    num_layers  : number of stacked LSTM layers
    num_mixtures: number of MDN Gaussian components
    max_seq_len : maximum sequence length (longer sequences are truncated)
    val_split   : fraction of data used for validation
    patience    : early-stopping patience (epochs without improvement)

    Returns
    -------
    model : trained GravesHandwritingModel (also saved to save_path)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"  STAGE 0 — IAM Dataset Training")
    print(f"  Device      : {device}")
    print(f"  Data dir    : {data_dir}")
    print(f"  Epochs      : {epochs}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Hidden size : {hidden_size}  Layers: {num_layers}  Mixtures: {num_mixtures}")
    print(f"{'='*60}\n")

    # ── Load dataset ──────────────────────────────────────────────────────────
    full_dataset = IAMStrokeDataset(
        data_dir    = data_dir,
        max_seq_len = max_seq_len,
    )

    if len(full_dataset) == 0:
        raise RuntimeError(
            f"No sequences loaded from '{data_dir}'. "
            "Check that lineStrokes XML files are present."
        )

    # Train / validation split
    n_val   = max(1, int(len(full_dataset) * val_split))
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size  = batch_size,
        shuffle     = True,
        collate_fn  = collate_strokes,
        num_workers = 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_strokes,
        num_workers = 0,
    )

    print(f"  Train samples : {n_train}")
    print(f"  Val   samples : {n_val}\n")

    # ── Build model ───────────────────────────────────────────────────────────
    model = GravesHandwritingModel(
        input_size   = 3,
        hidden_size  = hidden_size,
        num_layers   = num_layers,
        num_mixtures = num_mixtures,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}\n")

    # ── Optimiser: Adam with learning-rate schedule ───────────────────────────
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=3, verbose=True
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss  = float('inf')
    patience_count = 0
    history        = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        print(f"  Epoch {epoch:3d}/{epochs}")

        train_loss = train_one_epoch(model, train_loader, optimiser, device)
        val_loss   = evaluate(model, val_loader, device)

        scheduler.step(val_loss)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f"    train loss: {train_loss:.4f}   val loss: {val_loss:.4f}   "
              f"lr: {optimiser.param_groups[0]['lr']:.2e}")

        # ── Checkpoint best model ─────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            torch.save({
                'epoch':        epoch,
                'model_state':  model.state_dict(),
                'optim_state':  optimiser.state_dict(),
                'val_loss':     best_val_loss,
                'train_loss':   train_loss,
                'hidden_size':  hidden_size,
                'num_layers':   num_layers,
                'num_mixtures': num_mixtures,
                'dataset_stats': full_dataset.get_stats(),
            }, save_path)
            print(f"    ✔ Saved best model → {save_path}  (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n  Early stopping triggered after {epoch} epochs "
                      f"(no improvement for {patience} epochs).")
                break

    print(f"\n  Training complete. Best val loss: {best_val_loss:.4f}")
    print(f"  Model saved to: {save_path}\n")

    return model


def load_trained_model(checkpoint_path: str) -> tuple:
    """
    Load a trained GravesHandwritingModel from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : path to .pt file saved by train_model()

    Returns
    -------
    (model, dataset_stats) :
        model        — GravesHandwritingModel with loaded weights
        dataset_stats — (mean_dx, std_dx, mean_dy, std_dy) for inference normalisation
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    model = GravesHandwritingModel(
        input_size   = 3,
        hidden_size  = ckpt.get('hidden_size',  400),
        num_layers   = ckpt.get('num_layers',   2),
        num_mixtures = ckpt.get('num_mixtures', 20),
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    stats = ckpt.get('dataset_stats', (0.0, 1.0, 0.0, 1.0))
    print(f"  Loaded model from '{checkpoint_path}' "
          f"(epoch {ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.4f})")

    return model, stats


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — AI-BASED HANDWRITING STROKE GENERATION
# Graves Handwriting Synthesis Model: LSTM + Mixture Density Network (MDN)
# Trained on: IAM Handwriting Dataset
# Input : Text sequence
# Output: Continuous (x, y) stroke coordinates + pen-up/pen-down states
# ═══════════════════════════════════════════════════════════════════════════════

class MixtureDensityNetwork(nn.Module):
    """
    Mixture Density Network (MDN) output head.

    Given LSTM hidden states, predicts a mixture of K bivariate Gaussians
    over (Δx, Δy) offsets, plus a Bernoulli probability for the pen state.

    Parameters
    ----------
    input_size : int   — size of LSTM output
    num_mixtures : int — number of Gaussian mixture components (K)
    """

    def __init__(self, input_size: int, num_mixtures: int = 20):
        super().__init__()
        self.K = num_mixtures

        # Each mixture component needs: pi, mu_x, mu_y, sigma_x, sigma_y, rho
        # Total params per component = 6; plus 1 for pen state = 6K + 1
        self.output_layer = nn.Linear(input_size, 6 * num_mixtures + 1)

    def forward(self, h):
        """
        h : Tensor [batch, seq_len, input_size]

        Returns
        -------
        pi     : mixture weights          [batch, seq, K]
        mu_x   : x means                  [batch, seq, K]
        mu_y   : y means                  [batch, seq, K]
        sigma_x: x standard deviations   [batch, seq, K]
        sigma_y: y standard deviations   [batch, seq, K]
        rho    : correlations             [batch, seq, K]
        pen    : pen-down probability     [batch, seq, 1]
        """
        out = self.output_layer(h)                          # [..., 6K+1]

        # Split into components
        pi_raw      = out[..., : self.K]
        mu_x        = out[..., self.K     : 2 * self.K]
        mu_y        = out[..., 2 * self.K : 3 * self.K]
        sigma_x_raw = out[..., 3 * self.K : 4 * self.K]
        sigma_y_raw = out[..., 4 * self.K : 5 * self.K]
        rho_raw     = out[..., 5 * self.K : 6 * self.K]
        pen_raw     = out[..., 6 * self.K :]

        # Apply activations
        pi      = torch.softmax(pi_raw, dim=-1)             # sum to 1
        sigma_x = torch.exp(sigma_x_raw)                   # > 0
        sigma_y = torch.exp(sigma_y_raw)                   # > 0
        rho     = torch.tanh(rho_raw)                      # in (-1, 1)
        pen     = torch.sigmoid(pen_raw)                   # in (0, 1)

        return pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen


class GravesHandwritingModel(nn.Module):
    """
    Graves Handwriting Synthesis Model.
    Architecture: 2-layer stacked LSTM + MDN output head.

    Input  : sequence of stroke tuples (Δx, Δy, pen_state)
    Output : predicted distribution over next (Δx, Δy) + pen state
    """

    def __init__(
        self,
        input_size: int   = 3,        # (Δx, Δy, pen)
        hidden_size: int  = 400,
        num_layers: int   = 2,
        num_mixtures: int = 20,
        dropout: float    = 0.2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Stacked LSTM
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # MDN head
        self.mdn = MixtureDensityNetwork(hidden_size, num_mixtures)

    def forward(self, x, hidden=None):
        """
        x      : Tensor [batch, seq_len, 3]  — input strokes
        hidden : optional LSTM hidden state

        Returns MDN parameters + updated hidden state.
        """
        lstm_out, hidden = self.lstm(x, hidden)
        pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen = self.mdn(lstm_out)
        return (pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen), hidden

    def init_hidden(self, batch_size: int = 1):
        """Initialise LSTM hidden state to zeros."""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return (h0, c0)


def mdn_loss(pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen, target):
    """
    Negative log-likelihood loss for the MDN.

    target : Tensor [batch, seq_len, 3]  — (Δx, Δy, pen_state)
    """
    dx  = target[..., 0:1]   # [B, T, 1]
    dy  = target[..., 1:2]
    p   = target[..., 2:3]

    # Bivariate Gaussian log-probability for each component
    z = (
        ((dx - mu_x) / sigma_x) ** 2
        + ((dy - mu_y) / sigma_y) ** 2
        - 2 * rho * (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
    )
    norm = (
        2 * math.pi * sigma_x * sigma_y * torch.sqrt(1 - rho ** 2)
    )
    gaussian = torch.exp(-z / (2 * (1 - rho ** 2))) / norm

    # Mixture log-likelihood
    mixture = (pi * gaussian).sum(dim=-1, keepdim=True)
    nll_xy  = -torch.log(mixture + 1e-8)

    # Bernoulli pen-state loss
    nll_pen = -(
        p * torch.log(pen + 1e-8)
        + (1 - p) * torch.log(1 - pen + 1e-8)
    )

    return (nll_xy + nll_pen).mean()


def sample_from_mdn(pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen,
                    temperature: float = 1.0):
    """
    Sample (Δx, Δy, pen_state) from the MDN output at the last time step.

    temperature : controls randomness (higher = more varied handwriting)
    """
    # Pick mixture component
    pi_t = pi[0, -1].detach().numpy()
    k    = np.random.choice(len(pi_t), p=pi_t)

    # Sample from chosen bivariate Gaussian
    mx  = mu_x[0, -1, k].item()
    my  = mu_y[0, -1, k].item()
    sx  = sigma_x[0, -1, k].item() * temperature
    sy  = sigma_y[0, -1, k].item() * temperature
    r   = rho[0, -1, k].item()

    cov = [[sx**2, r*sx*sy], [r*sx*sy, sy**2]]
    dx, dy = np.random.multivariate_normal([mx, my], cov)

    # Sample pen state
    p_down = pen[0, -1, 0].item()
    pen_state = 1 if np.random.random() < p_down else 0

    return float(dx), float(dy), pen_state


def generate_strokes(
    model: GravesHandwritingModel,
    seed_stroke: np.ndarray,
    num_steps: int      = 700,
    temperature: float  = 0.8,
) -> list:
    """
    Auto-regressively generate a handwriting stroke sequence.

    Parameters
    ----------
    model       : trained GravesHandwritingModel
    seed_stroke : np.ndarray shape (1, 3) — initial (Δx, Δy, pen)
    num_steps   : number of stroke points to generate
    temperature : sampling temperature

    Returns
    -------
    strokes : list of (x, y, pen_state) in absolute coordinates
    """
    model.eval()
    strokes   = []
    hidden    = model.init_hidden(batch_size=1)
    x, y      = 0.0, 0.0

    # Prime with seed stroke
    inp = torch.tensor(seed_stroke, dtype=torch.float32).unsqueeze(0)  # [1,1,3]

    with torch.no_grad():
        for _ in range(num_steps):
            (pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen), hidden = model(inp, hidden)

            dx, dy, p = sample_from_mdn(
                pi, mu_x, mu_y, sigma_x, sigma_y, rho, pen, temperature
            )

            x += dx
            y += dy
            strokes.append((x, y, p))

            # Next input is current output
            inp = torch.tensor([[[dx, dy, float(p)]]], dtype=torch.float32)

    return strokes


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — STROKE REPRESENTATION & PEN CONTROL LOGIC
#
# Each handwriting sample is represented as:
#   (x_t, y_t, p_t)
# where:
#   x_t, y_t  → Cartesian coordinates of the pen tip
#   p_t = 1   → Pen Down  (actively writing on paper)
#   p_t = 0   → Pen Up    (moving between strokes, no mark)
#
# Pen lifts occur between strokes, between letters, and between words.
# ═══════════════════════════════════════════════════════════════════════════════

def build_stroke_sequence(raw_strokes: list) -> list:
    """
    Convert raw (x, y, pen) tuples into a clean stroke sequence
    with explicit pen-lift transitions.

    Parameters
    ----------
    raw_strokes : list of (x, y, pen_state)

    Returns
    -------
    sequence : list of dicts with keys 'x', 'y', 'pen'
               pen=1 → pen down (draws on paper)
               pen=0 → pen up   (no mark)
    """
    sequence = []
    prev_pen = 1  # start pen-down

    for (x, y, pen) in raw_strokes:
        if prev_pen == 1 and pen == 0:
            # Pen lift: record the lift-off point first
            sequence.append({'x': x, 'y': y, 'pen': 1})  # last contact
            sequence.append({'x': x, 'y': y, 'pen': 0})  # lift off

        elif prev_pen == 0 and pen == 1:
            # Pen touch-down: move without mark, then start writing
            sequence.append({'x': x, 'y': y, 'pen': 0})  # travel
            sequence.append({'x': x, 'y': y, 'pen': 1})  # touch-down
        else:
            sequence.append({'x': x, 'y': y, 'pen': pen})

        prev_pen = pen

    return sequence


def insert_word_gaps(sequences: list, gap_x: float = 0.3) -> list:
    """
    Merge stroke sequences for individual words, inserting a pen-up gap
    between words to ensure legibility.

    Parameters
    ----------
    sequences : list of stroke sequences (one per word)
    gap_x     : horizontal gap (in normalised units) between words

    Returns
    -------
    merged : single flat stroke sequence
    """
    merged  = []
    x_offset = 0.0

    for i, seq in enumerate(sequences):
        if not seq:
            continue

        # Shift this word's strokes by cumulative x offset
        shifted = [{'x': s['x'] + x_offset, 'y': s['y'], 'pen': s['pen']}
                   for s in seq]
        merged.extend(shifted)

        # After each word (except the last), insert pen-up gap
        if i < len(sequences) - 1:
            last_x = shifted[-1]['x']
            merged.append({'x': last_x,          'y': 0.0, 'pen': 0})  # lift
            merged.append({'x': last_x + gap_x,  'y': 0.0, 'pen': 0})  # travel

            x_offset += gap_x

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — COORDINATE NORMALISATION & WORKSPACE MAPPING
#
# Raw stroke coordinates are normalised to [0, 1] then scaled and offset
# to fit within the robot's physical reachable workspace.
#
# Normalisation:
#   x_norm = (x - x_min) / (x_max - x_min)
#   y_norm = (y - y_min) / (y_max - y_min)
#
# Scaling & Offset:
#   x' = Sx · x_norm + X_offset
#   y' = Sy · y_norm + Y_offset
#
# Ensures:
#   - Collision-free motion within the reachable workspace
#   - Workspace safety (no joint-limit violations)
#   - Consistent handwriting size
# ═══════════════════════════════════════════════════════════════════════════════

# Physical workspace of the robotic arm (metres)
WORKSPACE = {
    'x_min': 0.28,    # closest point to robot base
    'x_max': 0.58,    # furthest point from robot base
    'y_min': -0.62,   # rightmost (from robot's perspective)
    'y_max':  0.62,   # leftmost
    'z_pen_down': 0.012,   # pen touching paper (metres above ground)
    'z_pen_up':   0.100,   # pen lifted (metres above ground)
}


def normalise_strokes(strokes: list) -> list:
    """
    Normalise stroke (x, y) coordinates to [0, 1].

    Parameters
    ----------
    strokes : list of dicts with 'x', 'y', 'pen'

    Returns
    -------
    normalised : same structure with x, y in [0, 1]
    """
    xs = [s['x'] for s in strokes]
    ys = [s['y'] for s in strokes]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Avoid division by zero for degenerate strokes
    dx = x_max - x_min if x_max != x_min else 1.0
    dy = y_max - y_min if y_max != y_min else 1.0

    normalised = []
    for s in strokes:
        normalised.append({
            'x':   (s['x'] - x_min) / dx,   # x_norm
            'y':   (s['y'] - y_min) / dy,   # y_norm
            'pen': s['pen'],
        })

    return normalised


def map_to_workspace(normalised_strokes: list, ws: dict = WORKSPACE) -> list:
    """
    Scale and offset normalised [0,1] stroke coordinates into the robot's
    physical workspace (metres), and assign pen height.

    Mapping:
      x' = Sx · x_norm + X_offset   (forward axis)
      y' = Sy · y_norm + Y_offset   (lateral axis, centred on Y=0)

    Parameters
    ----------
    normalised_strokes : list of dicts with 'x', 'y' in [0,1] and 'pen'
    ws                 : workspace bounds dict

    Returns
    -------
    waypoints : list of [x, y, z, pen] in metres — ready for IK
    """
    Sx = ws['x_max'] - ws['x_min']          # x scaling factor
    Sy = ws['y_max'] - ws['y_min']          # y scaling factor
    X_offset = ws['x_min']
    Y_offset = ws['y_min']

    waypoints = []
    for s in normalised_strokes:
        wx = Sx * s['x'] + X_offset          # world X (forward)
        wy = Sy * s['y'] + Y_offset          # world Y (lateral)
        wz = ws['z_pen_down'] if s['pen'] == 1 else ws['z_pen_up']

        waypoints.append([
            round(wx, 4),
            round(wy, 4),
            round(wz, 4),
            int(s['pen']),
        ])

    return waypoints


def strokes_to_waypoints(strokes: list) -> list:
    """
    Full pipeline: raw strokes → normalised → workspace-mapped waypoints.

    Parameters
    ----------
    strokes : list of (x, y, pen_state) tuples from the LSTM model

    Returns
    -------
    waypoints : list of [x, y, z, pen] ready for the IK solver
    """
    # Convert tuples to dicts
    stroke_dicts = [{'x': x, 'y': y, 'pen': p} for (x, y, p) in strokes]

    # Stage 2: build clean pen-state sequence
    clean = build_stroke_sequence(stroke_dicts)

    # Stage 3a: normalise to [0,1]
    normalised = normalise_strokes(clean)

    # Stage 3b: map to physical workspace
    waypoints = map_to_workspace(normalised)

    return waypoints


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — MATHEMATICAL MODELLING OF THE ROBOTIC ARM
#           Forward Kinematics (FK)
#
# Robot Configuration:
#   - 3-DOF planar robotic arm
#   - Three revolute joints (base, shoulder, elbow)
#   - End-effector holds the pen
#
# Assumptions:
#   - Rigid links
#   - Fixed base
#   - Planar motion (XY writing plane)
#
# DH Parameters & FK equations:
#   x = L1·cos(θ1) + L2·cos(θ1+θ2) + L3·cos(θ1+θ2+θ3)
#   y = L1·sin(θ1) + L2·sin(θ1+θ2) + L3·sin(θ1+θ2+θ3)
# ═══════════════════════════════════════════════════════════════════════════════

# Physical link lengths (metres) — adjust to match your actual arm
L1 = 0.400   # upper arm  (shoulder → elbow)
L2 = 0.400   # forearm    (elbow → wrist)
L3 = 0.080   # pen holder (wrist → pen tip)

SHOULDER_HEIGHT = 0.36   # shoulder joint height above ground (metres)


def forward_kinematics(theta1: float, theta2: float, theta3: float,
                       l1: float = L1, l2: float = L2, l3: float = L3
                       ) -> tuple:
    """
    Compute end-effector (pen tip) position from joint angles.

    FK equations for 3-DOF planar arm:
      x = L1·cos(θ1) + L2·cos(θ1+θ2) + L3·cos(θ1+θ2+θ3)
      y = L1·sin(θ1) + L2·sin(θ1+θ2) + L3·sin(θ1+θ2+θ3)

    Parameters
    ----------
    theta1 : base/shoulder joint angle (radians)
    theta2 : elbow joint angle (radians)
    theta3 : wrist joint angle (radians)
    l1, l2, l3 : link lengths (metres)

    Returns
    -------
    (x, y) : pen-tip position in the arm's plane (metres)
    """
    a12  = theta1 + theta2
    a123 = theta1 + theta2 + theta3

    x = l1 * math.cos(theta1) + l2 * math.cos(a12) + l3 * math.cos(a123)
    y = l1 * math.sin(theta1) + l2 * math.sin(a12) + l3 * math.sin(a123)

    return x, y


def dh_transform(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Denavit–Hartenberg homogeneous transformation matrix.

    T = Rz(θ) · Tz(d) · Tx(a) · Rx(α)

    Parameters
    ----------
    theta : joint angle (radians)
    d     : link offset along previous z-axis
    a     : link length along rotated x-axis
    alpha : link twist angle (radians)

    Returns
    -------
    T : 4×4 homogeneous transformation matrix (numpy array)
    """
    ct, st   = math.cos(theta), math.sin(theta)
    ca, sa   = math.cos(alpha), math.sin(alpha)

    return np.array([
        [ct,  -st * ca,   st * sa,   a * ct],
        [st,   ct * ca,  -ct * sa,   a * st],
        [0,        sa,       ca,         d],
        [0,         0,        0,         1],
    ])


def forward_kinematics_full(theta1: float, theta2: float, theta3: float,
                            l1: float = L1, l2: float = L2, l3: float = L3
                            ) -> dict:
    """
    Full DH-based FK returning positions of all joints + end-effector.

    DH parameters for a planar arm (d=0, alpha=0 for all joints):
      Joint 1: a=L1
      Joint 2: a=L2
      Joint 3: a=L3

    Returns
    -------
    dict with keys 'base', 'elbow', 'wrist', 'pen'
    each containing (x, y, z) position in metres.
    """
    T1 = dh_transform(theta1, 0, l1, 0)
    T2 = dh_transform(theta2, 0, l2, 0)
    T3 = dh_transform(theta3, 0, l3, 0)

    T01  = T1
    T02  = T01 @ T2
    T03  = T02 @ T3

    origin = np.array([0, 0, 0, 1])

    return {
        'base':  (0.0, 0.0, SHOULDER_HEIGHT),
        'elbow': tuple(T01 @ origin)[:3],
        'wrist': tuple(T02 @ origin)[:3],
        'pen':   tuple(T03 @ origin)[:3],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — INVERSE KINEMATICS FORMULATION
#
# Objective: Compute joint angles (θ1, θ2, θ3) given desired end-effector
#            position (x, y, z).
#
# Decomposed into:
#   θ0 = atan2(y, x)          — base yaw rotation
#   r  = √(x² + y²)           — radial reach distance
#   cos(θ2) = (r² + dz² − L1² − L2²) / (2·L1·L2)
#   θ1 = atan2(dz, r) − atan2(L2·sinθ2, L1 + L2·cosθ2)
#
# Configuration: Elbow-down (θ2 negated)
# Reach clamped to [min_reach, max_reach] to avoid singularities.
# ═══════════════════════════════════════════════════════════════════════════════

def solve_ik(x_target: float, y_target: float, z_target: float,
             l1: float = L1, l2: float = L2,
             shoulder_height: float = SHOULDER_HEIGHT) -> tuple:
    """
    Analytical 2-link planar IK for the robotic writing arm.

    Returns (theta0, theta1, theta2) in radians.

    Parameters
    ----------
    x_target, y_target, z_target : desired pen-tip position (metres)
    l1, l2          : upper-arm and forearm link lengths (metres)
    shoulder_height : height of shoulder joint above ground (metres)

    Returns
    -------
    (theta0, theta1, theta2) : joint angles in radians
                               theta0 = base yaw
                               theta1 = shoulder pitch
                               theta2 = elbow (negative = elbow-down)
    """
    # ── θ0: rotate base toward target in XY plane ────────────────────────────
    theta0 = math.atan2(y_target, x_target)

    # ── Radial reach in XY plane ─────────────────────────────────────────────
    r = math.sqrt(x_target ** 2 + y_target ** 2)

    # ── Vertical offset from shoulder ────────────────────────────────────────
    # Negative dz means the arm is reaching DOWN to the writing surface
    dz = z_target - shoulder_height

    dist = math.sqrt(r ** 2 + dz ** 2)

    # ── Clamp to reachable envelope (avoids singularities) ───────────────────
    max_reach = l1 + l2 - 0.01
    min_reach = abs(l1 - l2) + 0.01

    if dist > max_reach:
        scale = max_reach / dist
        r    *= scale
        dz   *= scale
        dist  = max_reach
    elif dist < min_reach:
        # Scale outward to minimum reachable distance
        if dist > 1e-6:
            scale = min_reach / dist
            r    *= scale
            dz   *= scale
        dist = min_reach

    # ── Elbow angle (law of cosines) ──────────────────────────────────────────
    #   cos(θ2) = (r² + dz² − L1² − L2²) / (2·L1·L2)
    cos_theta2 = (r ** 2 + dz ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))   # numerical clamp

    theta2_raw = math.acos(cos_theta2)

    # ── Shoulder angle ────────────────────────────────────────────────────────
    #   θ1 = atan2(dz, r) − atan2(L2·sinθ2, L1 + L2·cosθ2)
    theta1 = (
        math.atan2(dz, r)
        - math.atan2(l2 * math.sin(theta2_raw),
                     l1 + l2 * math.cos(theta2_raw))
    )

    # ── Elbow-down configuration ──────────────────────────────────────────────
    # Negate theta2 so the elbow bends downward toward the writing surface
    theta2 = -theta2_raw

    return theta0, theta1, theta2


def ik_home() -> tuple:
    """
    Safe home position — arm pointing up and slightly forward.
    Returns (theta0, theta1, theta2) in radians.
    """
    return (0.0, -0.3, 0.8)


def angles_to_servo_pwm(angle_rad: float,
                         pwm_min: int = 500,
                         pwm_max: int = 2500,
                         angle_min: float = -math.pi / 2,
                         angle_max: float =  math.pi / 2) -> int:
    """
    Convert a joint angle (radians) to a servo PWM pulse width (microseconds).

    Linear mapping:
      PWM = pwm_min + (angle − angle_min) / (angle_max − angle_min)
                    × (pwm_max − pwm_min)

    Parameters
    ----------
    angle_rad : joint angle in radians
    pwm_min   : minimum pulse width μs (typically 500 or 1000)
    pwm_max   : maximum pulse width μs (typically 2500 or 2000)
    angle_min : minimum joint angle supported (radians)
    angle_max : maximum joint angle supported (radians)

    Returns
    -------
    pwm : pulse width in microseconds (clamped to [pwm_min, pwm_max])
    """
    t = (angle_rad - angle_min) / (angle_max - angle_min)
    t = max(0.0, min(1.0, t))                                 # clamp [0,1]
    return int(pwm_min + t * (pwm_max - pwm_min))


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE — text → waypoints → joint angles
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    text: str,
    model: 'GravesHandwritingModel' = None,
    checkpoint_path: str = None,
    num_steps: int    = 700,
    temperature: float = 0.8,
) -> list:
    """
    End-to-end pipeline:
      text (str)
        → LSTM stroke generation          [Stage 1]
        → pen control logic               [Stage 2]
        → normalisation + workspace map   [Stage 3]
        → IK solve per waypoint           [Stage 5]
        → list of (theta0, theta1, theta2, pen_state)

    Parameters
    ----------
    text             : input text string (e.g. "HELLO")
    model            : trained GravesHandwritingModel (or None)
    checkpoint_path  : path to .pt checkpoint — loaded if model is None
    num_steps        : stroke sequence length
    temperature      : LSTM sampling temperature

    Returns
    -------
    joint_commands : list of (theta0, theta1, theta2, pen_state)
    """
    print(f"\n{'='*60}")
    print(f"  IoT Robotic Handwriting — Full Pipeline")
    print(f"  Input text: '{text}'")
    print(f"{'='*60}\n")

    # ── Stage 1: AI stroke generation ────────────────────────────────────────
    print("[1/4] Generating handwriting strokes via LSTM+MDN...")

    dataset_stats = None
    if model is None and checkpoint_path and os.path.exists(checkpoint_path):
        model, dataset_stats = load_trained_model(checkpoint_path)
        print(f"      Loaded trained weights from: {checkpoint_path}")
    elif model is None:
        model = GravesHandwritingModel()
        print("      (Using randomly initialised model — train first for real handwriting)")

    seed = np.array([[[0.0, 0.0, 1.0]]])         # seed: pen-down at origin
    raw_strokes = generate_strokes(model, seed, num_steps=num_steps,
                                   temperature=temperature)
    print(f"      Generated {len(raw_strokes)} raw stroke points.\n")

    # ── Stage 2 + 3: pen control + workspace mapping ─────────────────────────
    print("[2/4] Applying pen control logic & workspace mapping...")
    waypoints = strokes_to_waypoints(raw_strokes)
    print(f"      Produced {len(waypoints)} waypoints.\n")

    # ── Stage 5: IK solve for each waypoint ──────────────────────────────────
    print("[3/4] Solving inverse kinematics per waypoint...")
    joint_commands = []

    for i, (wx, wy, wz, pen) in enumerate(waypoints):
        try:
            theta0, theta1, theta2 = solve_ik(wx, wy, wz)
        except Exception as e:
            print(f"      Warning: IK failed at waypoint {i}: {e}. Skipping.")
            continue

        joint_commands.append((theta0, theta1, theta2, pen))

    print(f"      Computed {len(joint_commands)} joint-angle commands.\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("[4/4] Pipeline complete. Sample joint commands:")
    for i, (t0, t1, t2, p) in enumerate(joint_commands[:5]):
        print(f"      [{i:03d}] θ0={math.degrees(t0):+6.1f}°  "
              f"θ1={math.degrees(t1):+6.1f}°  "
              f"θ2={math.degrees(t2):+6.1f}°  "
              f"pen={'DOWN' if p else 'UP  '}")

    if len(joint_commands) > 5:
        print(f"      ... ({len(joint_commands) - 5} more commands)\n")

    return joint_commands


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="IoT Robotic Handwriting — Methodology Pipeline"
    )
    parser.add_argument('--train',    action='store_true',
                        help='Train the LSTM+MDN model on the IAM dataset')
    parser.add_argument('--generate', action='store_true',
                        help='Run the full pipeline (stroke gen → IK)')
    parser.add_argument('--demo',     action='store_true',
                        help='Run FK + IK demos without a dataset')
    parser.add_argument('--data_dir', type=str, default='./data/lineStrokes',
                        help='Path to IAM lineStrokes/ directory')
    parser.add_argument('--save',     type=str, default='model.pt',
                        help='Path to save/load model checkpoint')
    parser.add_argument('--load',     type=str, default=None,
                        help='Path to load a trained checkpoint for generation')
    parser.add_argument('--epochs',   type=int, default=50)
    parser.add_argument('--batch',    type=int, default=32)
    parser.add_argument('--lr',       type=float, default=1e-3)
    parser.add_argument('--text',     type=str, default='HELLO',
                        help='Text to generate when using --generate')
    parser.add_argument('--steps',    type=int, default=700,
                        help='Number of stroke steps to generate')
    parser.add_argument('--temp',     type=float, default=0.8,
                        help='Sampling temperature (higher = more varied)')
    args = parser.parse_args()

    # ── No flags → run everything as a demo ──────────────────────────────────
    run_all = not (args.train or args.generate or args.demo)

    # ── Training ──────────────────────────────────────────────────────────────
    if args.train or run_all:
        if not os.path.isdir(args.data_dir):
            print(f"\n  [SKIP] Training: data directory not found: '{args.data_dir}'")
            print(  "         Download IAM lineStrokes and set --data_dir accordingly.\n")
        else:
            train_model(
                data_dir   = args.data_dir,
                save_path  = args.save,
                epochs     = args.epochs,
                batch_size = args.batch,
                lr         = args.lr,
            )

    # ── FK demo ───────────────────────────────────────────────────────────────
    if args.demo or run_all:
        print("=" * 60)
        print("  STAGE 4 — Forward Kinematics Demo")
        print("=" * 60)
        theta1_demo = math.radians(45)
        theta2_demo = math.radians(-30)
        theta3_demo = math.radians(10)
        fx, fy = forward_kinematics(theta1_demo, theta2_demo, theta3_demo)
        print(f"  θ1={math.degrees(theta1_demo):.1f}°  "
              f"θ2={math.degrees(theta2_demo):.1f}°  "
              f"θ3={math.degrees(theta3_demo):.1f}°")
        print(f"  → Pen tip position: x={fx:.4f} m,  y={fy:.4f} m\n")

        all_pos = forward_kinematics_full(theta1_demo, theta2_demo, theta3_demo)
        for joint, pos in all_pos.items():
            print(f"  {joint:6s}: ({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}) m")

        # ── IK demo ───────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  STAGE 5 — Inverse Kinematics Demo")
        print("=" * 60)

        targets = [
            (0.40,  0.00, 0.012),   # centre of writing surface
            (0.45,  0.15, 0.012),   # right of centre
            (0.32, -0.20, 0.012),   # left of centre
        ]

        for (xt, yt, zt) in targets:
            t0, t1, t2 = solve_ik(xt, yt, zt)
            print(f"\n  Target : ({xt:.3f}, {yt:.3f}, {zt:.3f}) m")
            print(f"  θ0={math.degrees(t0):+6.1f}°  "
                  f"θ1={math.degrees(t1):+6.1f}°  "
                  f"θ2={math.degrees(t2):+6.1f}°")
            pwm0 = angles_to_servo_pwm(t0)
            pwm1 = angles_to_servo_pwm(t1)
            pwm2 = angles_to_servo_pwm(t2)
            print(f"  PWM  : servo0={pwm0} μs  servo1={pwm1} μs  servo2={pwm2} μs")

    # ── Generation pipeline ───────────────────────────────────────────────────
    if args.generate or run_all:
        checkpoint = args.load or (args.save if os.path.exists(args.save) else None)
        run_pipeline(
            text             = args.text,
            model            = None,
            checkpoint_path  = checkpoint,
            num_steps        = args.steps,
            temperature      = args.temp,
        )

