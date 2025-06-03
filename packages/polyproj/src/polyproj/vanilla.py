import numpy as np

from polyproj.utils import ProjectionResult

def project_on_polyline_naive(
    points: np.ndarray,
    segments: np.ndarray,
    return_distance: bool = True,
    return_projection: bool = False,
    return_param: bool = False,
    return_index: bool = False,
) -> ProjectionResult:
    """Baseline loop implementation for projection (no blocking)."""
    N = points.shape[0]
    best_dist_sq = np.full(N, np.inf)
    best_proj = np.empty((N, 2)) if return_projection else None
    best_t = np.empty(N) if return_param else None
    best_idx = np.empty(N, dtype=int) if return_index else None

    for seg_idx, seg in enumerate(segments):
        a, b = seg
        v = b - a
        v_norm_sq = float(np.dot(v, v))

        # Treat as point if length is zero
        if v_norm_sq == 0.0:
            t_raw = np.zeros(N)
            proj = np.broadcast_to(a, (N, 2))
        else:
            w = points - a
            t_raw = np.sum(w * v, axis=1) / v_norm_sq
            t_raw = np.clip(t_raw, 0.0, 1.0)  # clamp to segment
            proj = a + t_raw[:, None] * v

        dist_sq = np.sum((points - proj) ** 2, axis=1)
        mask = dist_sq < best_dist_sq

        best_dist_sq[mask] = dist_sq[mask]
        if best_proj is not None:
            best_proj[mask] = proj[mask]
        if best_t is not None:
            best_t[mask] = t_raw[mask]
        if best_idx is not None:
            best_idx[mask] = seg_idx

    return ProjectionResult(
        distances=np.sqrt(best_dist_sq) if return_distance else None,
        projections=best_proj if return_projection else None,
        params=best_t if return_param else None,
        indices=best_idx if return_index else None,
    )


def project_on_polyline(
    points: np.ndarray,
    segments: np.ndarray,
    return_distance: bool = True,
    return_projection: bool = False,
    return_param: bool = False,
    return_index: bool = False,
) -> ProjectionResult:
    n_points = points.shape[0]

    best_dists_sq = np.full(n_points, np.inf)
    best_projs = np.empty((n_points, 2)) if return_projection else None
    best_params = np.empty(n_points) if return_param else None
    best_indices = np.empty(n_points, dtype=int) if return_index else None

    for seg_idx, seg in enumerate(segments):
        a, b = seg
        v = b - a
        v_norm_sq = float(np.dot(v, v))

        # Treat as point if length is zero
        if v_norm_sq == 0.0:
            t_raw = np.zeros(n_points)
            proj = np.broadcast_to(a, (n_points, 2))
        else:
            w = points - a
            t_raw = np.sum(w * v, axis=1) / v_norm_sq
            t_raw = np.clip(t_raw, 0.0, 1.0)  # clamp to segment
            proj = a + t_raw[:, None] * v

        dist_sq = np.sum((points - proj) ** 2, axis=1)
        mask = dist_sq < best_dists_sq

        best_dists_sq[mask] = dist_sq[mask]
        if best_projs is not None:
            best_projs[mask] = proj[mask]
        if best_params is not None:
            best_params[mask] = t_raw[mask]
        if best_indices is not None:
            best_indices[mask] = seg_idx

    return ProjectionResult(
        distances=np.sqrt(best_dists_sq) if return_distance else None,
        projections=best_projs if return_projection else None,
        params=best_params if return_param else None,
        indices=best_indices if return_index else None,
    )


def project_on_polyline_blockwise(
    points: np.ndarray,  # (N, 2)
    segments: np.ndarray,  # (M, 2, 2)
    block_size: int = 64,
    return_distance: bool = True,
    return_projection: bool = False,
    return_param: bool = False,
    return_index: bool = False,
) -> ProjectionResult:
    """Project points onto a polyline in blocks to reduce peak memory usage."""
    N = points.shape[0]
    best_dist_sq = np.full(N, np.inf)
    best_projs = np.empty((N, 2)) if return_projection else None
    best_params = np.empty(N) if return_param else None
    best_indices = np.empty(N, dtype=int) if return_index else None

    M = segments.shape[0]
    for start in range(0, M, block_size):
        block = segments[start : start + block_size]
        a = block[:, 0, :]
        b = block[:, 1, :]
        v = b - a
        v_norm_sq = np.einsum("ij,ij->i", v, v)
        zero_length = v_norm_sq == 0
        v_norm_sq_safe = v_norm_sq.copy()
        v_norm_sq_safe[zero_length] = 1.0

        w = points[None, :, :] - a[:, None, :]
        t_raw = np.einsum("mnd,md->mn", w, v) / v_norm_sq_safe[:, None]
        t_clipped = np.clip(t_raw, 0.0, 1.0)
        t_clipped[zero_length, :] = 0.0

        proj = a[:, None, :] + t_clipped[:, :, None] * v[:, None, :]
        diff = points[None, :, :] - proj
        dist_sq = np.einsum("mnd,mnd->mn", diff, diff)

        best_seg_loc = np.argmin(dist_sq, axis=0)
        block_best_dist = dist_sq[best_seg_loc, np.arange(N)]
        mask = block_best_dist < best_dist_sq
        best_dist_sq[mask] = block_best_dist[mask]
        if best_projs is not None:
            best_projs[mask] = proj[best_seg_loc[mask], np.arange(N)[mask], :]
        if best_params is not None:
            best_params[mask] = t_clipped[best_seg_loc[mask], np.arange(N)[mask]]
        if best_indices is not None:
            best_indices[mask] = start + best_seg_loc[mask]

    return ProjectionResult(
        distances=np.sqrt(best_dist_sq) if return_distance else None,
        projections=best_projs if return_projection else None,
        params=best_params if return_param else None,
        indices=best_indices if return_index else None,
    )
