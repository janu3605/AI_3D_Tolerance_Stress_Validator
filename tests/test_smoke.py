"""
Smoke Tests for V2 GNN Modules
================================
Lightweight tests to verify the core V2 pipeline components
work correctly with synthetic data.

Run with:
    python -m pytest tests/test_smoke.py -v
"""

import numpy as np
import torch
import trimesh
import pytest


# ---------------------------------------------------------------------------
#  Test 1: Mesh-to-Graph Conversion
# ---------------------------------------------------------------------------

class TestMeshToGraph:
    """Tests for utils/mesh_to_graph.py"""

    def _make_box_stl(self, tmp_path):
        """Create a temporary box STL file for testing."""
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        path = str(tmp_path / "test_box.stl")
        mesh.export(path)
        return path, mesh

    def test_output_data_type(self, tmp_path):
        """mesh_to_graph should return a PyG Data object."""
        from torch_geometric.data import Data
        from utils.mesh_to_graph import mesh_to_graph

        stl_path, _ = self._make_box_stl(tmp_path)
        data = mesh_to_graph(stl_path, target_faces=100)

        assert isinstance(data, Data)

    def test_node_features_shape(self, tmp_path):
        """Node features should be (N, 6) = [x, y, z, nx, ny, nz]."""
        from utils.mesh_to_graph import mesh_to_graph

        stl_path, _ = self._make_box_stl(tmp_path)
        data = mesh_to_graph(stl_path, target_faces=100)

        assert data.x.dim() == 2
        assert data.x.shape[1] == 6  # [x, y, z, nx, ny, nz]
        assert data.x.shape[0] > 0   # at least some nodes

    def test_edge_index_shape(self, tmp_path):
        """edge_index should be (2, E) with bidirectional edges."""
        from utils.mesh_to_graph import mesh_to_graph

        stl_path, _ = self._make_box_stl(tmp_path)
        data = mesh_to_graph(stl_path, target_faces=100)

        assert data.edge_index.dim() == 2
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0
        # Bidirectional: edge count should be even
        assert data.edge_index.shape[1] % 2 == 0

    def test_edge_attr_shape(self, tmp_path):
        """edge_attr should be (E, 1) = [distance]."""
        from utils.mesh_to_graph import mesh_to_graph

        stl_path, _ = self._make_box_stl(tmp_path)
        data = mesh_to_graph(stl_path, target_faces=100)

        assert data.edge_attr.dim() == 2
        assert data.edge_attr.shape[1] == 1
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]

    def test_edge_lengths_positive(self, tmp_path):
        """All edge lengths should be non-negative."""
        from utils.mesh_to_graph import mesh_to_graph

        stl_path, _ = self._make_box_stl(tmp_path)
        data = mesh_to_graph(stl_path, target_faces=100)

        assert (data.edge_attr >= 0).all()

    def test_pos_preserved(self, tmp_path):
        """Raw vertex positions should be stored in data.pos."""
        from utils.mesh_to_graph import mesh_to_graph

        stl_path, _ = self._make_box_stl(tmp_path)
        data = mesh_to_graph(stl_path, target_faces=100)

        assert data.pos is not None
        assert data.pos.shape == (data.num_nodes, 3)


# ---------------------------------------------------------------------------
#  Test 2: Decimation
# ---------------------------------------------------------------------------

class TestDecimation:
    """Tests for mesh decimation."""

    def test_decimation_reduces_faces(self):
        """Decimation should reduce face count to approximately the target."""
        from utils.mesh_to_graph import decimate_mesh

        # Create a sphere with many faces
        mesh = trimesh.creation.icosphere(subdivisions=4)
        original_faces = len(mesh.faces)
        target = original_faces // 4

        decimated = decimate_mesh(mesh, target_faces=target)

        assert len(decimated.faces) <= target * 1.2  # allow 20% tolerance
        assert len(decimated.faces) < original_faces

    def test_no_decimation_needed(self):
        """If mesh already has fewer faces than target, return unchanged."""
        from utils.mesh_to_graph import decimate_mesh

        mesh = trimesh.creation.box()
        original_faces = len(mesh.faces)

        decimated = decimate_mesh(mesh, target_faces=original_faces + 100)

        assert len(decimated.faces) == original_faces


# ---------------------------------------------------------------------------
#  Test 3: GNN Model Forward Pass
# ---------------------------------------------------------------------------

class TestStressGNN:
    """Tests for utils/gnn_model.py"""

    def _make_synthetic_graph(self, num_nodes=50, num_edges=150):
        """Create a synthetic graph Data object for testing."""
        from torch_geometric.data import Data

        x = torch.randn(num_nodes, 6)  # [x,y,z,nx,ny,nz]
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_attr = torch.rand(num_edges, 1)
        pos = torch.randn(num_nodes, 3)

        return Data(
            x=x, edge_index=edge_index,
            edge_attr=edge_attr, pos=pos,
            num_nodes=num_nodes,
        )

    def test_model_instantiation(self):
        """StressGNN should instantiate without errors."""
        from utils.gnn_model import StressGNN

        model = StressGNN()
        assert model is not None

    def test_forward_pass_output_keys(self):
        """Forward pass should return dict with 'stress' and 'node_risk'."""
        from utils.gnn_model import StressGNN

        model = StressGNN()
        model.eval()

        data = self._make_synthetic_graph()

        with torch.no_grad():
            out = model(data)

        assert "stress" in out
        assert "node_risk" in out

    def test_forward_pass_shapes(self):
        """Output tensors should have correct shapes."""
        from utils.gnn_model import StressGNN

        model = StressGNN()
        model.eval()

        num_nodes = 50
        data = self._make_synthetic_graph(num_nodes=num_nodes)

        with torch.no_grad():
            out = model(data)

        # Stress: (1, 1) for single graph
        assert out["stress"].shape == (1, 1)
        # Node risk: (N, 1)
        assert out["node_risk"].shape == (num_nodes, 1)

    def test_stress_non_negative(self):
        """Predicted stress should be non-negative (ReLU output)."""
        from utils.gnn_model import StressGNN

        model = StressGNN()
        model.eval()
        data = self._make_synthetic_graph()

        with torch.no_grad():
            out = model(data)

        assert (out["stress"] >= 0).all()

    def test_risk_scores_bounded(self):
        """Node risk scores should be in [0, 1] (sigmoid output)."""
        from utils.gnn_model import StressGNN

        model = StressGNN()
        model.eval()
        data = self._make_synthetic_graph()

        with torch.no_grad():
            out = model(data)

        assert (out["node_risk"] >= 0).all()
        assert (out["node_risk"] <= 1).all()

    def test_custom_hyperparameters(self):
        """Model should work with custom hyperparameters."""
        from utils.gnn_model import StressGNN

        model = StressGNN(
            in_channels=6,
            hidden_channels=64,
            num_gnn_layers=2,
            edge_dim=1,
            dropout=0.1,
        )
        model.eval()

        data = self._make_synthetic_graph()

        with torch.no_grad():
            out = model(data)

        assert out["stress"].shape == (1, 1)


# ---------------------------------------------------------------------------
#  Test 4: predict_stress helper
# ---------------------------------------------------------------------------

class TestPredictStress:
    """Tests for the predict_stress convenience function."""

    def test_predict_stress_output(self):
        """predict_stress should return a dict with expected keys."""
        from utils.gnn_model import StressGNN, predict_stress
        from torch_geometric.data import Data

        model = StressGNN()
        model.eval()

        data = Data(
            x=torch.randn(30, 6),
            edge_index=torch.randint(0, 30, (2, 80)),
            edge_attr=torch.rand(80, 1),
            pos=torch.randn(30, 3),
            num_nodes=30,
        )

        result = predict_stress(model, data, device="cpu")

        assert "predicted_stress" in result
        assert "max_risk_score" in result
        assert "max_risk_node_idx" in result
        assert "max_risk_position" in result
        assert isinstance(result["predicted_stress"], float)
        assert isinstance(result["max_risk_score"], float)
        assert isinstance(result["max_risk_node_idx"], int)
        assert result["max_risk_position"].shape == (3,)
