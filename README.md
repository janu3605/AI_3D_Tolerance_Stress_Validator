# 🧠 AI 3D Tolerance & Stress Validator

> Instantly detect structural weak points in 3D CAD models using a 3D CNN — no heavy FEA simulations required.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janu3605/AI_3D_Tolerance_Stress_Validator/blob/main/AI_3D_Stress_Validator.ipynb)


---

## What It Does

This tool takes a standard 3D CAD file (STL) and predicts whether a part has a **critical stress concentration** — specifically, when a hole is drilled too close to an edge, creating fracture risk. If a danger zone is detected, it outputs the exact **3D bounding box** coordinates and renders a translucent red alert over the part.

```
STL File  →  Voxelize (64³ grid)  →  3D CNN  →  Pass/Fail + Danger Zone BBox
```

## Architecture

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

## Project Structure

```
AI_3D_Tolerance_Stress_Validator/
├── AI_3D_Stress_Validator.ipynb    ← Main Colab notebook (all 3 phases)
├── README.md
└── utils/
    ├── voxelizer.py                ← Mesh → voxel grid conversion
    └── model.py                    ← 3D CNN model definition
```

## Quick Start (Google Colab)

1. Click the **Open in Colab** badge above (or upload `AI_3D_Stress_Validator.ipynb` to Colab)
2. Set runtime to **GPU**: `Runtime → Change runtime type → T4 GPU`
3. Run all cells sequentially — the notebook handles:
   - **Phase 1**: Downloads 250 DeepJEB samples, voxelizes them
   - **Phase 2**: Trains the 3D CNN (~15–20 min on T4)
   - **Phase 3**: Runs inference and visualizes danger zones
4. Upload your own STL file in the final cell to test

## Dataset

Uses the [DeepJEB](https://www.narnia.ai/dataset) dataset — 2,138 synthetic jet engine bracket designs with FEA stress data. We use 250 samples for the MVP.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Data Prep | `trimesh`, `numpy` |
| AI Model  | PyTorch 3D CNN |
| Training  | Google Colab (free T4 GPU) |
| Viz       | `matplotlib`, `pyvista` |

## License

MIT
