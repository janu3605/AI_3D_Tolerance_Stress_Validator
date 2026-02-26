"""
GNN Stress Predictor Model (V2)
================================
A Graph Neural Network with message-passing layers for physics-informed
structural stress prediction.

Architecture:
  1. Encoder  : Linear projection of [x,y,z,nx,ny,nz] → 128-dim latent
  2. GNN Body : 4× GCNConv layers with BatchNorm, ReLU, residual skip connections
  3. Dual Heads:
       - Regression Head  (global): mean pooling → MLP → max stress (MPa)
       - Localization Head (per-node): MLP → per-node risk score [0,1]

Input : torch_geometric.data.Data with x(N,6), edge_index(2,E), edge_attr(E,1)
Output: dict with "stress" (B,1), "node_risk" (N,1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# ---------------------------------------------------------------------------
#  Edge-aware message weight module
# ---------------------------------------------------------------------------

class EdgeMLP(nn.Module):
    """
    Lightweight MLP that transforms edge attributes (distances) into
    scalar weights for modulating GCN messages.
    """

    def __init__(self, edge_dim: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        edge_attr : (E, edge_dim)

        Returns
        -------
        (E,) scalar weights in [0, 1]
        """
        return self.mlp(edge_attr).squeeze(-1)


# ---------------------------------------------------------------------------
#  GNN Model
# ---------------------------------------------------------------------------

class StressGNN(nn.Module):
    """
    Physics-informed Graph Neural Network for stress prediction.

    Parameters
    ----------
    in_channels : int
        Number of input node features (default: 6 for [x,y,z,nx,ny,nz]).
    hidden_channels : int
        Hidden dimension for GCN layers (default: 128).
    num_gnn_layers : int
        Number of message-passing layers (default: 4).
    edge_dim : int
        Dimension of edge attributes (default: 1 for distance).
    dropout : float
        Dropout rate for output heads (default: 0.3).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 128,
        num_gnn_layers: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_gnn_layers = num_gnn_layers

        # ---- Encoder: project input features to hidden dim ----
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, hidden_channels),
            nn.ReLU(inplace=True),
        )

        # ---- Message Passing: GCN layers with BatchNorm ----
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # ---- Edge MLP for distance-aware messaging ----
        self.edge_mlp = EdgeMLP(edge_dim=edge_dim, hidden_dim=16)

        # ---- Regression Head (Global): predict max stress ----
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.ReLU(inplace=True),  # stress is non-negative
        )

        # ---- Localization Head (Per-Node): predict risk score ----
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, data) -> dict:
        """
        Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data or Batch
            Must have: x (N, in_channels), edge_index (2, E),
                       edge_attr (E, edge_dim), batch (N,) [if batched]

        Returns
        -------
        dict with:
            "stress"    : (B, 1) predicted max stress values
            "node_risk" : (N, 1) per-node risk scores (sigmoid)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else None

        # 1. Encode input features
        x = self.encoder(x)  # (N, hidden)

        # 2. Compute edge weights from edge attributes
        if edge_attr is not None:
            edge_weight = self.edge_mlp(edge_attr)  # (E,)
        else:
            edge_weight = None

        # 3. Message passing with residual connections
        for i in range(self.num_gnn_layers):
            identity = x
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.bns[i](x)
            x = F.relu(x)
            x = x + identity  # skip connection

        # 4. Per-node risk scores (before pooling)
        node_risk = torch.sigmoid(self.risk_head(x))  # (N, 1)

        # 5. Global pooling for stress regression
        if batch is not None:
            pooled = global_mean_pool(x, batch)  # (B, hidden)
        else:
            pooled = x.mean(dim=0, keepdim=True)  # (1, hidden)

        stress = self.stress_head(pooled)  # (B, 1)

        return {
            "stress": stress,
            "node_risk": node_risk,
        }


# ---------------------------------------------------------------------------
#  Convenience functions
# ---------------------------------------------------------------------------

def load_gnn_model(
    weights_path: str,
    device: str = "cpu",
    **model_kwargs,
) -> StressGNN:
    """
    Load a trained StressGNN model from a state dict file.

    Parameters
    ----------
    weights_path : str
        Path to saved `.pth` state dict.
    device : str
        Target device ('cpu' or 'cuda').
    **model_kwargs
        Override model hyperparameters (in_channels, hidden_channels, etc.).

    Returns
    -------
    StressGNN
        Model in eval mode with loaded weights.
    """
    model = StressGNN(**model_kwargs)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_stress(
    model: StressGNN,
    data,
    device: str = "cpu",
):
    """
    Run inference on a single graph and return results.

    Parameters
    ----------
    model : StressGNN
        Trained model in eval mode.
    data : torch_geometric.data.Data
        Single graph data object.
    device : str
        Device to run inference on.

    Returns
    -------
    dict with:
        "predicted_stress" : float — predicted max stress in MPa
        "max_risk_score"   : float — highest per-node risk score
        "max_risk_node_idx": int   — index of the highest-risk node
        "max_risk_position": np.ndarray — (3,) world coordinates of risk zone
    """
    import numpy as np

    model.eval()
    data = data.to(device)

    with torch.no_grad():
        out = model(data)

    stress_val = out["stress"].item()
    node_risk = out["node_risk"].squeeze(-1).cpu().numpy()

    max_risk_idx = int(np.argmax(node_risk))
    max_risk_score = float(node_risk[max_risk_idx])

    # Get world-space position of highest-risk node
    if hasattr(data, "pos") and data.pos is not None:
        max_risk_pos = data.pos[max_risk_idx].cpu().numpy()
    else:
        max_risk_pos = data.x[max_risk_idx, :3].cpu().numpy()

    return {
        "predicted_stress": stress_val,
        "max_risk_score": max_risk_score,
        "max_risk_node_idx": max_risk_idx,
        "max_risk_position": max_risk_pos,
    }
