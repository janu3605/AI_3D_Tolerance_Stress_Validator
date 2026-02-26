"""
Mesh-to-Graph Converter (V2)
=============================
Converts 3D mesh files (STL) into PyTorch Geometric graph objects
suitable for GNN-based stress prediction.

Pipeline:
  STL → trimesh.load → QEM Decimation → Extract Vertices/Edges
      → Compute Surface Normals → Compute Edge Lengths
      → torch_geometric.data.Data

Node features  : [x, y, z, nx, ny, nz]  (position + surface normal)
Edge attributes: [euclidean_distance]     (length of each edge)
"""

import numpy as np
import trimesh
import torch
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
#  Mesh Decimation (QEM)
# ---------------------------------------------------------------------------

def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """
    Reduce triangle count using Quadric Error Metrics (QEM) decimation.

    Preserves critical geometric features (holes, fillets) while cutting
    up to 80% of the mesh complexity for efficient GNN processing.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input high-resolution mesh.
    target_faces : int
        Desired number of output faces.

    Returns
    -------
    trimesh.Trimesh
        Decimated mesh with approximately `target_faces` triangles.
    """
    if len(mesh.faces) <= target_faces:
        return mesh

    # Try Open3D first (better QEM implementation)
    try:
        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        o3d_mesh.compute_vertex_normals()

        decimated = o3d_mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )

        result = trimesh.Trimesh(
            vertices=np.asarray(decimated.vertices),
            faces=np.asarray(decimated.triangles),
            process=True,
        )
        return result

    except ImportError:
        # Fallback to trimesh's built-in simplification
        result = mesh.simplify_quadric_decimation(target_faces)
        return result


# ---------------------------------------------------------------------------
#  Surface Normal Computation
# ---------------------------------------------------------------------------

def compute_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Compute per-vertex surface normals via area-weighted face normal averaging.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh (must have valid faces).

    Returns
    -------
    np.ndarray
        Shape (N, 3) array of unit normal vectors per vertex.
    """
    vertex_normals = mesh.vertex_normals.copy()

    # Normalize (trimesh usually does this, but ensure it)
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)  # avoid division by zero
    vertex_normals = vertex_normals / norms

    return vertex_normals.astype(np.float32)


# ---------------------------------------------------------------------------
#  Edge Extraction
# ---------------------------------------------------------------------------

def extract_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """
    Extract unique undirected edges from a triangle face array.

    Parameters
    ----------
    faces : np.ndarray
        Shape (F, 3) array of face vertex indices.

    Returns
    -------
    np.ndarray
        Shape (E, 2) array of unique edge vertex index pairs.
    """
    # Each triangle (a, b, c) has edges: (a,b), (b,c), (a,c)
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [0, 2]],
    ])

    # Sort each edge so (min, max) for deduplication
    edges = np.sort(edges, axis=1)

    # Remove duplicates
    edges = np.unique(edges, axis=0)

    return edges


# ---------------------------------------------------------------------------
#  Full Conversion Pipeline
# ---------------------------------------------------------------------------

def mesh_to_graph(
    stl_path: str,
    target_faces: int = 2000,
    normalize_pos: bool = True,
) -> Data:
    """
    Convert an STL file to a PyTorch Geometric Data object.

    Pipeline:
      1. Load STL with trimesh
      2. Decimate via QEM to `target_faces`
      3. Extract unique edges from faces
      4. Compute vertex normals
      5. Build node features = [x, y, z, nx, ny, nz]
      6. Compute edge attributes = [euclidean_distance]
      7. Return as torch_geometric.data.Data

    Parameters
    ----------
    stl_path : str
        Path to the STL file.
    target_faces : int
        Target number of faces after decimation (default: 2000).
    normalize_pos : bool
        If True, center and scale vertex positions to unit sphere.

    Returns
    -------
    torch_geometric.data.Data
        Graph with:
          - x        : (N, 6)  node features [pos + normals]
          - edge_index: (2, 2E) bidirectional edge indices
          - edge_attr : (2E, 1) edge lengths
          - pos      : (N, 3)  raw vertex positions (for visualization)
          - num_nodes : N
    """
    # 1. Load mesh
    mesh = trimesh.load(stl_path, force="mesh")

    # 2. Decimate
    mesh = decimate_mesh(mesh, target_faces)

    # 3. Extract vertices and edges
    vertices = mesh.vertices.astype(np.float32)  # (N, 3)
    edges = extract_edges_from_faces(mesh.faces)   # (E, 2) undirected

    # 4. Compute normals
    normals = compute_vertex_normals(mesh)  # (N, 3)

    # 5. Normalize positions (center + scale to unit sphere)
    raw_pos = vertices.copy()
    if normalize_pos:
        centroid = vertices.mean(axis=0)
        vertices = vertices - centroid
        max_dist = np.max(np.linalg.norm(vertices, axis=1))
        if max_dist > 1e-8:
            vertices = vertices / max_dist

    # 6. Build node features: [x, y, z, nx, ny, nz]
    node_features = np.hstack([vertices, normals])  # (N, 6)

    # 7. Build bidirectional edge_index for PyG (COO format)
    # PyG expects (2, num_edges) with both directions
    edge_index = np.vstack([
        np.hstack([edges[:, 0], edges[:, 1]]),
        np.hstack([edges[:, 1], edges[:, 0]]),
    ])  # (2, 2E)

    # 8. Compute edge lengths (Euclidean distance)
    src_pos = raw_pos[edge_index[0]]
    dst_pos = raw_pos[edge_index[1]]
    edge_lengths = np.linalg.norm(src_pos - dst_pos, axis=1, keepdims=True)
    edge_lengths = edge_lengths.astype(np.float32)

    # Normalize edge lengths
    max_len = edge_lengths.max()
    if max_len > 1e-8:
        edge_lengths = edge_lengths / max_len

    # 9. Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_lengths, dtype=torch.float32)
    pos = torch.tensor(raw_pos, dtype=torch.float32)

    data = Data(
        x=x,
        edge_index=edge_index_t,
        edge_attr=edge_attr,
        pos=pos,
        num_nodes=x.shape[0],
    )

    return data


# ---------------------------------------------------------------------------
#  Batch Conversion Helper
# ---------------------------------------------------------------------------

def batch_convert_stls(
    stl_paths: list,
    target_faces: int = 2000,
    normalize_pos: bool = True,
    verbose: bool = True,
) -> list:
    """
    Convert a list of STL files to graph Data objects.

    Parameters
    ----------
    stl_paths : list of str
        Paths to STL files.
    target_faces : int
        Target decimation face count.
    normalize_pos : bool
        Whether to normalize vertex positions.
    verbose : bool
        Print progress messages.

    Returns
    -------
    list of (Data, str)
        List of (graph_data, file_path) tuples. Failed conversions are skipped.
    """
    results = []
    for i, path in enumerate(stl_paths):
        try:
            data = mesh_to_graph(path, target_faces, normalize_pos)
            results.append((data, path))
            if verbose and (i + 1) % 25 == 0:
                print(f"  Converted {i + 1}/{len(stl_paths)} meshes...")
        except Exception as e:
            if verbose:
                print(f"  ⚠️ Failed to convert {path}: {e}")
    if verbose:
        print(f"  ✅ Successfully converted {len(results)}/{len(stl_paths)} meshes.")
    return results
