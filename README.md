# 🧠 AI 3D Tolerance & Stress Validator

> Instantly detect structural weak points in 3D CAD models — no heavy FEA simulations required.

| Version | Approach | Notebook |
|---------|----------|----------|
| **V1** — 3D CNN | Voxel grid + Conv3D | [![V1 Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janu3605/AI_3D_Tolerance_Stress_Validator/blob/main/AI_3D_Stress_Validator.ipynb) |
| **V2** — GNN ⭐ | Mesh graph + GCN | [![V2 Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janu3605/AI_3D_Tolerance_Stress_Validator/blob/main/AI_3D_Stress_Validator_V2_GNN.ipynb) |

---

## What It Does

This tool takes a standard 3D CAD file (STL) and predicts whether a part has a **critical stress concentration** — specifically, when a hole is drilled too close to an edge, creating fracture risk. If a danger zone is detected, it outputs the exact **3D coordinates** and renders a risk heatmap over the part.

---

## V2 Architecture (GNN) ⭐

V2 replaces the voxel grid with a **mesh-graph** representation and uses a **Graph Convolutional Network** to simulate stress flow through the actual geometry.

```
STL File  →  QEM Decimate  →  Mesh Graph  →  GNN (4× GCN)  →  Stress + Risk Map
```

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────────────┐
│   STL Mesh      │     │  Mesh Graph           │     │   StressGNN (PyTorch Geo)    │
│   (CAD file)    │────▶│  Nodes: vertices      │────▶│                              │
│                 │     │  Edges: connectivity   │     │  ┌────────────────────────┐  │
└─────────────────┘     │  Features:             │     │  │ Encoder: Linear(6→128) │  │
     trimesh            │   • pos  (x,y,z)       │     │  └──────────┬─────────────┘  │
     + Open3D           │   • normals (nx,ny,nz) │     │             │                │
     (QEM decimate)     │   • edge len (dist)    │     │  ┌──────────▼─────────────┐  │
                        └──────────────────────┘     │  │ 4× GCNConv + BatchNorm │  │
                                                      │  │ + Residual Skip Conn.  │  │
                                                      │  │ + Edge Distance MLP    │  │
                                                      │  └──────────┬─────────────┘  │
                                                      │     ┌───────┴───────┐        │
                                                      │     │               │        │
                                                      │  ┌──▼───┐     ┌────▼────┐   │
                                                      │  │Stress│     │ Node    │   │
                                                      │  │ Head │     │ Risk    │   │
                                                      │  │(glob)│     │ Head    │   │
                                                      │  └──┬───┘     └────┬────┘   │
                                                      └─────┼──────────────┼────────┘
                                                            │              │
                                                       Max Stress    Per-Node Risk
                                                        (MPa)        Score [0,1]
```

### V1 vs V2 Comparison

| Feature | V1 (Voxel/CNN) | V2 (Graph/GNN) |
|---------|---------------|----------------|
| **Data Structure** | 3D Voxel Grid (64³) | Mesh Graph (~1K nodes) |
| **Physics Intuition** | Geometric proximity only | Structural connectivity |
| **Resolution** | Limited by grid (blocky) | High-fidelity (mesh-based) |
| **Input Features** | Occupancy (0 or 1) | Position + Surface Normals |
| **Edge Logic** | None | Euclidean Distance Injection |
| **Output** | Pass/Fail + BBox | Stress (MPa) + Per-Node Risk |
| **Preprocessing** | Voxelization | QEM Decimation (~80% reduction) |

---

## V1 Architecture (3D CNN)

```
STL File  →  Voxelize (64³ grid)  →  3D CNN  →  Pass/Fail + Danger Zone BBox
```

<details>
<summary>V1 Architecture Diagram</summary>

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────────┐
│   STL Mesh      │     │  Voxel Grid      │     │   3D CNN (PyTorch)         │
│   (CAD file)    │────▶│  64 × 64 × 64    │────▶│                            │
│                 │     │  binary (0/1)     │     │  ┌──────────────────────┐  │
└─────────────────┘     └──────────────────┘     │  │ 4× Conv3D + BN+Pool │  │
     trimesh                                      │  └──────────┬───────────┘  │
                                                  │             │              │
                                                  │     ┌───────┴───────┐      │
                                                  │     │               │      │
                                                  │  ┌──▼──┐       ┌───▼──┐   │
                                                  │  │ CLS  │       │ BBOX │   │
                                                  │  │ head │       │ head │   │
                                                  │  └──┬──┘       └───┬──┘   │
                                                  │     │               │      │
                                                  └─────┼───────────────┼──────┘
                                                        │               │
                                                   Pass/Fail     (x,y,z,w,h,d)
```

</details>

---

## Project Structure

```
AI_3D_Tolerance_Stress_Validator/
├── AI_3D_Stress_Validator.ipynb         ← V1 Colab notebook (Voxel/CNN)
├── AI_3D_Stress_Validator_V2_GNN.ipynb  ← V2 Colab notebook (Graph/GNN) ⭐
├── updated_cells.py                     ← V1 cell patches
├── README.md
├── tests/
│   └── test_smoke.py                    ← Smoke tests for V2 modules
└── utils/
    ├── __init__.py
    ├── voxelizer.py                     ← V1: Mesh → voxel grid
    ├── model.py                         ← V1: 3D CNN model
    ├── mesh_to_graph.py                 ← V2: Mesh → graph (QEM + normals)
    ├── graph_dataset.py                 ← V2: PyG dataset class
    └── gnn_model.py                     ← V2: GNN model (StressGNN)
```

## Quick Start (Google Colab)

### V2 — GNN (Recommended)

1. Click the **V2 Colab** badge above
2. Set runtime to **GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run all cells — the notebook handles:
   - **Phase 0**: Installs PyTorch Geometric, Open3D, trimesh
   - **Phase 1**: Loads STL files, decimates via QEM, builds mesh-graphs
   - **Phase 2**: Trains the GNN (~15–20 min on T4)
   - **Phase 3**: Runs inference with 3D risk heatmap visualization
4. Upload your own STL in the final cell

### V1 — 3D CNN

1. Click the **V1 Colab** badge above
2. Follow the same GPU setup and run all cells

## Dataset

Uses the [DeepJEB](https://www.narnia.ai/dataset) dataset — 2,138 synthetic jet engine bracket designs with FEA stress data. We use 250 samples for the MVP.

## Tech Stack

| Component | V1 | V2 |
|-----------|----|----|
| Data Prep | `trimesh`, `numpy` | `trimesh`, `Open3D`, `numpy` |
| AI Model  | PyTorch 3D CNN | PyTorch Geometric GCN |
| Training  | Google Colab (T4 GPU) | Google Colab (T4 GPU) |
| Viz       | `matplotlib` | `matplotlib` (3D risk heatmap) |

## License

MIT
