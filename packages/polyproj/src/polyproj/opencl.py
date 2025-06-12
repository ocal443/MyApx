from typing import Optional

import numpy as np
import pyopencl as cl

from polyproj.utils import ProjectionResult

# Module-level OpenCL context and program
_ctx: Optional[cl.Context] = None
_queue: Optional[cl.CommandQueue] = None
_program: Optional[cl.Program] = None


def _get_context_and_queue():
    """Initialize OpenCL context and command queue if needed"""
    global _ctx, _queue

    if _ctx is None:
        # Use PyOpenCL's built-in context creation which respects PYOPENCL_CTX
        _ctx = cl.create_some_context()

        # Get device info for logging
        device = _ctx.devices[0]
        device_name = device.name
        device_type = cl.device_type.to_string(device.type)

        _queue = cl.CommandQueue(_ctx)
        print(f"Using OpenCL device: {device_name} ({device_type})")

    return _ctx, _queue


def _get_program() -> cl.Program:
    """Get the compiled program, building it once if needed"""
    global _program
    ctx, _ = _get_context_and_queue()

    if _program is None:
        # Build with optimization flags for better performance
        build_options = [
            "-cl-fast-relaxed-math",  # Allow optimizations that may reduce precision
            "-cl-mad-enable",         # Enable multiply-add instructions
            "-Werror"                 # Treat warnings as errors
        ]

        _program = cl.Program(ctx, _KERNEL_SOURCE).build(options=build_options)

    return _program


# OpenCL kernel for polyline projection
_KERNEL_SOURCE = """
__kernel void project_points(
    __global const float2 *points,
    __global const float4 *segments,  // (ax, ay, bx, by)
    __global float *best_dist_sq,
    __global float2 *best_proj,
    __global float *best_t,
    __global int *best_idx,
    const int n_points,
    const int n_segments,
    const int store_proj,
    const int store_param,
    const int store_idx
) {
    int point_idx = get_global_id(0);
    if (point_idx >= n_points) return;

    float2 point = points[point_idx];
    float min_dist_sq = INFINITY;
    float2 min_proj = (float2)(0.0f, 0.0f);
    float min_t = 0.0f;
    int min_idx = 0;

    for (int seg_idx = 0; seg_idx < n_segments; seg_idx++) {
        float4 segment = segments[seg_idx];
        float2 a = (float2)(segment.x, segment.y);
        float2 b = (float2)(segment.z, segment.w);
        float2 v = b - a;

        float v_norm_sq = dot(v, v);
        float t;
        float2 proj;

        float v_norm_sq_safe = (v_norm_sq < 1e-10f) ? 1.0f : v_norm_sq;
        float2 w = point - a;
        t = dot(w, v) / v_norm_sq_safe;
        t = clamp(t, 0.0f, 1.0f);
        t = (v_norm_sq < 1e-10f) ? 0.0f : t;
        proj = a + t * v;

        float2 diff = point - proj;
        float dist_sq = dot(diff, diff);

        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            min_proj = proj;
            min_t = t;
            min_idx = seg_idx;
        }
    }

    best_dist_sq[point_idx] = min_dist_sq;
    if (store_proj) best_proj[point_idx] = min_proj;
    if (store_param) best_t[point_idx] = min_t;
    if (store_idx) best_idx[point_idx] = min_idx;
}
"""


def project_on_polyline_opencl(
    points: np.ndarray,  # (N, 2)
    segments: np.ndarray,  # (M, 2, 2)
    return_distance: bool = True,
    return_projection: bool = False,
    return_param: bool = False,
    return_index: bool = False,
) -> ProjectionResult:
    ctx, queue = _get_context_and_queue()

    program = _get_program()

    n_points = points.shape[0]
    n_segments = segments.shape[0]

    points_cl = np.ascontiguousarray(points.astype(np.float32))

    segments_cl = np.empty((n_segments, 4), dtype=np.float32)
    segments_cl[:, 0:2] = segments[:, 0, :]  # start points
    segments_cl[:, 2:4] = segments[:, 1, :]  # end points
    segments_cl = np.ascontiguousarray(segments_cl)

    points_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=points_cl
    )
    segments_buf = cl.Buffer(
        ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=segments_cl
    )

    best_dist_sq_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, size=n_points * np.dtype(np.float32).itemsize
    )

    best_proj_buf = None
    dummy_buf = cl.Buffer(
        ctx, cl.mem_flags.WRITE_ONLY, size=4
    )  # Dummy buffer for unused outputs

    if return_projection:
        best_proj_buf = cl.Buffer(
            ctx,
            cl.mem_flags.WRITE_ONLY,
            size=n_points * 2 * np.dtype(np.float32).itemsize,
        )

    best_t_buf = None
    if return_param:
        best_t_buf = cl.Buffer(
            ctx, cl.mem_flags.WRITE_ONLY, size=n_points * np.dtype(np.float32).itemsize
        )

    best_idx_buf = None
    if return_index:
        best_idx_buf = cl.Buffer(
            ctx, cl.mem_flags.WRITE_ONLY, size=n_points * np.dtype(np.int32).itemsize
        )

    kernel = program.project_points
    kernel(
        queue,
        (n_points,),
        None,
        points_buf,
        segments_buf,
        best_dist_sq_buf,
        best_proj_buf if return_projection else dummy_buf,
        best_t_buf if return_param else dummy_buf,
        best_idx_buf if return_index else dummy_buf,
        np.int32(n_points),
        np.int32(n_segments),
        np.int32(1 if return_projection else 0),
        np.int32(1 if return_param else 0),
        np.int32(1 if return_index else 0),
    )

    # Read results
    best_dist_sq = np.empty(n_points, dtype=np.float32)
    cl.enqueue_copy(queue, best_dist_sq, best_dist_sq_buf)

    best_proj = None
    if return_projection:
        best_proj = np.empty((n_points, 2), dtype=np.float32)
        cl.enqueue_copy(queue, best_proj, best_proj_buf)

    best_t = None
    if return_param:
        best_t = np.empty(n_points, dtype=np.float32)
        cl.enqueue_copy(queue, best_t, best_t_buf)

    best_idx = None
    if return_index:
        best_idx = np.empty(n_points, dtype=np.int32)
        cl.enqueue_copy(queue, best_idx, best_idx_buf)

    return ProjectionResult(
        distances=np.sqrt(best_dist_sq) if return_distance else None,
        projections=best_proj if return_projection else None,
        params=best_t if return_param else None,
        indices=best_idx if return_index else None,
    )
