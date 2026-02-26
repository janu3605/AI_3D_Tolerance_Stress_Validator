"""
Graph Dataset for Bracket Stress Prediction (V2)
==================================================
PyTorch Geometric InMemoryDataset that loads STL files, converts them
to graph representations, and pairs them with FEA stress labels from
bracket_labels.csv.
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from .mesh_to_graph import mesh_to_graph


class BracketGraphDataset(InMemoryDataset):
    """
    Graph dataset for DeepJEB bracket stress prediction.

    Each sample is a graph built from a decimated STL mesh, with:
      - Node features: [x, y, z, nx, ny, nz]
      - Edge attributes: [euclidean_distance]
      - Label: binary pass/fail (1 = fail)
      - Stress value: continuous max stress in MPa

    Parameters
    ----------
    root : str
        Root directory for dataset storage (processed files cached here).
    stl_dir : str
        Directory containing STL files (searched recursively).
    csv_path : str
        Path to bracket_labels.csv with FEA stress data.
    target_faces : int
        Target face count for QEM decimation.
    stress_fail_quantile : float
        Quantile threshold for pass/fail labeling (default: 0.80).
    num_samples : int or None
        Max number of samples to use (None = all).
    transform, pre_transform, pre_filter :
        Standard PyG dataset hooks.
    """

    def __init__(
        self,
        root: str,
        stl_dir: str,
        csv_path: str,
        target_faces: int = 2000,
        stress_fail_quantile: float = 0.80,
        num_samples: int = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.stl_dir = stl_dir
        self.csv_path = csv_path
        self.target_faces = target_faces
        self.stress_fail_quantile = stress_fail_quantile
        self.num_samples = num_samples

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # We handle raw files manually

    @property
    def processed_file_names(self):
        return [f"bracket_graphs_f{self.target_faces}.pt"]

    def process(self):
        """Convert all STL files to graphs and save as a single .pt file."""
        # --- Discover STL files ---
        stl_files = sorted(
            glob.glob(os.path.join(self.stl_dir, "**", "*.stl"), recursive=True)
        )
        if not stl_files:
            raise FileNotFoundError(
                f"No STL files found in {self.stl_dir}"
            )
        if self.num_samples is not None:
            stl_files = stl_files[: self.num_samples]
        print(f"📁 Found {len(stl_files)} STL files to process.")

        # --- Load stress labels ---
        df = pd.read_csv(self.csv_path)

        stress_cols = [
            "max_ver_stress(MPa)",
            "max_hor_stress(MPa)",
            "max_dia_stress(MPa)",
            "max_tor_stress(MPa)",
        ]
        # Fallback: find any stress columns
        missing = [c for c in stress_cols if c not in df.columns]
        if missing:
            stress_cols = [c for c in df.columns if "stress" in c.lower()]

        df["max_stress_all"] = df[stress_cols].max(axis=1)

        # Compute threshold
        threshold = df["max_stress_all"].quantile(self.stress_fail_quantile)
        df["label"] = (df["max_stress_all"] >= threshold).astype(float)

        # Build lookup dicts
        stress_lookup = {}
        label_lookup = {}
        for _, row in df.iterrows():
            name = str(row["item_name"])
            stress_lookup[name] = float(row["max_stress_all"])
            label_lookup[name] = float(row["label"])

        print(f"📊 Stress threshold (q={self.stress_fail_quantile}): {threshold:.2f} MPa")
        n_fail = int(df["label"].sum())
        print(f"🏷️  Labels: {len(df) - n_fail} PASS, {n_fail} FAIL")

        # --- Convert each STL to a graph ---
        data_list = []
        skipped = 0

        for i, stl_path in enumerate(stl_files):
            try:
                # Extract item name from filename (remove extension)
                basename = os.path.splitext(os.path.basename(stl_path))[0]

                # Convert to graph
                graph = mesh_to_graph(stl_path, target_faces=self.target_faces)

                # Attach labels
                stress_val = stress_lookup.get(basename, 0.0)
                label = label_lookup.get(basename, 0.0)

                graph.y = torch.tensor([label], dtype=torch.float32)
                graph.stress_value = torch.tensor([stress_val], dtype=torch.float32)
                graph.item_name = basename

                data_list.append(graph)

                if (i + 1) % 25 == 0:
                    print(f"  ✅ Processed {i + 1}/{len(stl_files)} meshes...")

            except Exception as e:
                skipped += 1
                print(f"  ⚠️ Skipped {stl_path}: {e}")

        print(f"\n✅ Converted {len(data_list)} graphs ({skipped} skipped).")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        self.save(data_list, self.processed_paths[0])


def create_dataset_from_lists(
    graph_data_list: list,
    labels: list,
    stress_values: list = None,
    item_names: list = None,
) -> list:
    """
    Utility to create a list of Data objects from pre-converted graphs.

    Useful when preprocessing is done separately (e.g., in a Colab cell)
    and you want to attach labels after the fact.

    Parameters
    ----------
    graph_data_list : list of Data
        Pre-converted graph Data objects.
    labels : list of float
        Binary labels (0.0 = pass, 1.0 = fail).
    stress_values : list of float, optional
        Continuous stress values in MPa.
    item_names : list of str, optional
        Item name identifiers.

    Returns
    -------
    list of Data
        Updated Data objects with labels attached.
    """
    for i, data in enumerate(graph_data_list):
        data.y = torch.tensor([labels[i]], dtype=torch.float32)
        if stress_values is not None:
            data.stress_value = torch.tensor(
                [stress_values[i]], dtype=torch.float32
            )
        if item_names is not None:
            data.item_name = item_names[i]
    return graph_data_list
