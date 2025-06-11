from dataclasses import dataclass
from itertools import product

import numpy as np
import pytest

from polyproj import project_on_polyline_cpp
from polyproj.opencl import project_on_polyline_opencl
from polyproj.vanilla import project_on_polyline_naive
from polyproj.utils import ProjectionResult
from polyproj.vanilla import project_on_polyline_naive as project_on_polyline_base

# List of all implementations to test
implementations = [
    project_on_polyline_naive,
    project_on_polyline_cpp,
    project_on_polyline_opencl,
]

# Create IDs for pytest parametrization
implementation_ids = [
    "naive",
    "cpp",
    "opencl",
]


@dataclass
class Example:
    name: str
    pts: np.ndarray
    seg: np.ndarray
    expected: ProjectionResult


examples = [
    Example(
        name="zero_length_on",
        pts=np.array([[0.0, 0.0]]),
        seg=np.array([[[0.0, 0.0], [0.0, 0.0]]]),
        expected=ProjectionResult(
            distances=np.array([0.0]),
            projections=np.array([[0.0, 0.0]]),
            params=np.array([0.0]),
            indices=np.array([0]),
        ),
    ),
    Example(
        name="zero_length_off",
        pts=np.array([[1.0, 0.0]]),
        seg=np.array([[[0.0, 0.0], [0.0, 0.0]]]),
        expected=ProjectionResult(
            distances=np.array([1.0]),
            projections=np.array([[0.0, 0.0]]),
            params=np.array([0.0]),
            indices=np.array([0]),
        ),
    ),
    Example(
        name="single_segment_start_end_mid",
        pts=np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]),
        seg=np.array([[[0.0, 0.0], [1.0, 1.0]]]),
        expected=ProjectionResult(
            distances=np.array([0.0, 0.0, 0.0]),
            projections=np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]),
            params=np.array([0.0, 1.0, 0.5]),
            indices=np.array([0, 0, 0]),
        ),
    ),
    Example(
        name="single_segment_projected_on",
        pts=np.array([[0.0, 1.0], [1.0, 1.0], [0.5, 1.0]]),
        seg=np.array([[[0.0, 0.0], [1.0, 0.0]]]),
        expected=ProjectionResult(
            distances=np.array([1.0, 1.0, 1.0]),
            projections=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]]),
            params=np.array([0.0, 1.0, 0.5]),
            indices=np.array([0, 0, 0]),
        ),
    ),
    Example(
        name="single_segment_clipped",
        pts=np.array([[-1.0, 0.0], [2.0, 0.0]]),
        seg=np.array([[[0.0, 0.0], [1.0, 0.0]]]),
        expected=ProjectionResult(
            distances=np.array([1.0, 1.0]),
            projections=np.array([[0.0, 0.0], [1.0, 0.0]]),
            params=np.array([0.0, 1.0]),
            indices=np.array([0, 0]),
        ),
    ),
]


@pytest.mark.parametrize("project_func", implementations, ids=implementation_ids)
@pytest.mark.parametrize("ex", examples, ids=lambda ex: ex.name)
def test_full_returns(project_func, ex: Example):
    """When all return_* flags are True, we get correct arrays."""
    res = project_func(
        ex.pts,
        ex.seg,
        return_distance=True,
        return_projection=True,
        return_param=True,
        return_index=True,
    )
    np.testing.assert_allclose(res.distances, ex.expected.distances)
    np.testing.assert_allclose(res.projections, ex.expected.projections)
    np.testing.assert_allclose(res.params, ex.expected.params)
    np.testing.assert_array_equal(res.indices, ex.expected.indices)


@pytest.mark.parametrize(
    "return_distance, return_projection, return_param, return_index",
    list(product([True, False], repeat=4)),
)
@pytest.mark.parametrize("project_func", implementations, ids=implementation_ids)
def test_return_combinations(
    project_func, return_distance, return_projection, return_param, return_index
):
    pts = np.array([[1.0, 2.0]])
    seg = np.array([[[0.0, 0.0], [2.0, 0.0]]])
    full_res = project_func(
        pts,
        seg,
        return_distance=True,
        return_projection=True,
        return_param=True,
        return_index=True,
    )
    res = project_func(
        pts,
        seg,
        return_distance=return_distance,
        return_projection=return_projection,
        return_param=return_param,
        return_index=return_index,
    )
    assert (res.distances is not None) == return_distance
    assert (res.projections is not None) == return_projection
    assert (res.params is not None) == return_param
    assert (res.indices is not None) == return_index
    # partial results should match the full-output where requested
    if return_distance:
        np.testing.assert_allclose(res.distances, full_res.distances)
    if return_projection:
        np.testing.assert_allclose(res.projections, full_res.projections)
    if return_param:
        np.testing.assert_allclose(res.params, full_res.params)
    if return_index:
        np.testing.assert_array_equal(res.indices, full_res.indices)


@pytest.mark.parametrize("project_func", implementations, ids=implementation_ids)
def test_random_equivalence(project_func):
    # Skip comparing naive to naive
    if project_func == project_on_polyline_base and "naive" in implementation_ids and implementation_ids.index("naive") == implementations.index(project_func) :
        pytest.skip("Skipping comparison of naive implementation with itself.")

    np.random.seed(0)
    N = 10_000
    M_segments = 1022
    pts = np.random.rand(N, 2)
    seg = np.random.rand(M_segments, 2, 2)
    base_res = project_on_polyline_base(
        pts,
        seg,
        return_distance=True,
        return_projection=True,
        return_param=True,
        return_index=True,
    )
    test_res = project_func( # Use the parametrized project_func
        pts,
        seg,
        return_distance=True,
        return_projection=True,
        return_param=True,
        return_index=True,
    )
    np.testing.assert_allclose(test_res.distances, base_res.distances, atol=1e-6)
    np.testing.assert_allclose(test_res.projections, base_res.projections, atol=1e-6)
    np.testing.assert_allclose(test_res.params, base_res.params, atol=1e-6)
    np.testing.assert_array_equal(test_res.indices, base_res.indices)
