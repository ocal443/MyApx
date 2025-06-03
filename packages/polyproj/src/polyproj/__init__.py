import numpy as np

from ._core import project_on_polyline as _project_on_polyline_cpp
from .utils import ProjectionResult


def project_on_polyline_cpp(
    points: np.ndarray,  # shape (N,2)
    segments: np.ndarray,  # shape (M,2,2)
    return_distance: bool = True,
    return_projection: bool = False,
    return_param: bool = False,
    return_index: bool = False,
) -> ProjectionResult:
    """
    C++/pybind11 implementation of project_on_polyline.

    Arguments:
        points             -- array of shape (N,2)
        segments           -- array of shape (M,2,2)
        return_distance    -- if True, include distances
        return_projection  -- if True, include projections
        return_param       -- if True, include t‐parameters
        return_index       -- if True, include segment indices

    Returns:
        ProjectionResult
    """
    # call into the binding, which returns a dict
    result_dict = _project_on_polyline_cpp(
        points,
        segments,
        return_distance=return_distance,
        return_projection=return_projection,
        return_param=return_param,
        return_index=return_index,
    )

    # wrap into the same ProjectionResult signature as the other implementations
    return ProjectionResult(
        distances=result_dict["distances"],
        projections=result_dict["projections"],
        params=result_dict["params"],
        indices=result_dict["indices"],
    )
