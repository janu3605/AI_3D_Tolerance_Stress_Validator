"""
Voxelizer Utility
=================
Converts 3D mesh files (STL, OBJ, etc.) into binary voxel grids
suitable for input to the 3D CNN stress predictor.
"""

import numpy as np
import trimesh


def voxelize_mesh(file_path: str, resolution: int = 64) -> np.ndarray:
    """
    Load a mesh from disk and convert it into a binary voxel grid.

    Parameters
    ----------
    file_path : str
        Path to the mesh file (STL, OBJ, PLY, etc.).
    resolution : int, optional
        Number of voxels along the longest axis (default 64).

    Returns
    -------
    np.ndarray
        A (resolution, resolution, resolution) binary float32 array
        where 1.0 = solid material, 0.0 = empty space.
    """
    mesh = trimesh.load(file_path, force="mesh")

    # Compute a pitch that maps the longest bbox axis to `resolution` voxels
    extents = mesh.bounding_box.extents  # (dx, dy, dz)
    pitch = max(extents) / resolution

    voxel_grid = mesh.voxelized(pitch=pitch)
    matrix = voxel_grid.matrix.astype(np.float32)  # bool -> float32

    # Pad or crop to exact (resolution, resolution, resolution)
    padded = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    slices_src = tuple(slice(0, min(s, resolution)) for s in matrix.shape)
    slices_dst = tuple(slice(0, min(s, resolution)) for s in matrix.shape)
    padded[slices_dst] = matrix[slices_src]

    return padded


def voxel_to_world(
    voxel_coords: np.ndarray,
    mesh_bounds: np.ndarray,
    resolution: int = 64,
) -> np.ndarray:
    """
    Convert voxel-space coordinates back to world-space (mesh) coordinates.

    Parameters
    ----------
    voxel_coords : np.ndarray
        Array of shape (N, 3) with voxel indices or (6,) bbox [x,y,z,w,h,d].
    mesh_bounds : np.ndarray
        Shape (2, 3) array: [min_corner, max_corner] of the original mesh.
    resolution : int
        The voxel grid resolution used during voxelization.

    Returns
    -------
    np.ndarray
        Corresponding world-space coordinates, same shape as input.
    """
    min_corner = mesh_bounds[0]
    extents = mesh_bounds[1] - mesh_bounds[0]
    scale = extents / resolution

    voxel_coords = np.asarray(voxel_coords, dtype=np.float64)

    if voxel_coords.shape[-1] == 6:
        # Bounding box format: [x, y, z, w, h, d]
        world = voxel_coords.copy()
        world[..., :3] = voxel_coords[..., :3] * scale + min_corner
        world[..., 3:] = voxel_coords[..., 3:] * scale
        return world
    else:
        # Point coordinates: (N, 3)
        return voxel_coords * scale + min_corner


def get_mesh_bounds(file_path: str) -> np.ndarray:
    """
    Return the axis-aligned bounding box of a mesh.

    Returns
    -------
    np.ndarray
        Shape (2, 3): [min_corner, max_corner].
    """
    mesh = trimesh.load(file_path, force="mesh")
    return mesh.bounds.copy()
