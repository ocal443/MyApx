
# polyproj

A high-performance Python package for projecting points onto a collection of polyline segments.

---

## Overview

`polyproj` provides efficient, interchangeable implementations for computing:

✅ the minimal distance from points to a set of segments  
✅ the exact projected (closest) point on the nearest segment  
✅ the parametric position \( t \) on the segment (0 = start, 1 = end)  
✅ the index of the nearest segment  

It supports multiple backends (pure Python, NumPy, Numba, AVX2, OpenCL, etc.) with consistent interfaces, correctness tests, and performance benchmarks.

---

## Core Function

```python
def project_on_polyline(
    points: np.ndarray,           # shape: (N, 2)
    segments: np.ndarray,         # shape: (M, 2, 2), each segment = [start, end]
    return_distance: bool = True,
    return_projection: bool = False,
    return_param: bool = False,
    return_index: bool = False
) -> ProjectionResult:
    """
    For each point in `points` and each segment in `segments`,
    compute—and optionally return—any of:
      • distances        : minimal Euclidean distance to a segment
      • projections      : (x, y) coordinates of the projection onto that segment
      • param_t          : parametric t ∈ [0,1] along the segment (0=start, 1=end)
      • segment_indices  : integer index (0 … M–1) of the nearest segment

    Only the requested outputs are computed.
    Returns a single `ProjectionResult` object.
    """
