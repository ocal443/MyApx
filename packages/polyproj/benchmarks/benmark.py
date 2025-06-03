"""
Simple benchmark for polyline projection implementations.
"""

import time
from typing import Dict

import numpy as np

from polyproj.vanilla import project_on_polyline_blockwise, project_on_polyline_naive

from polyproj.opencl import project_on_polyline_opencl


def run_benchmark(n_points: int, n_segments: int, repeat: int = 3) -> Dict:
    """Run benchmark with specified dimensions."""
    print(f"Benchmarking with N={n_points}, M={n_segments}")

    # Generate random test data
    points = np.random.rand(n_points, 2) * 100  # Random points
    segments = np.zeros((n_segments, 2, 2))
    for i in range(n_segments):
        segments[i, 0, :] = np.random.rand(2) * 100  # Start point
        segments[i, 1, :] = segments[i, 0, :] + np.random.rand(2) * 10  # End point

    # Dictionary to store results
    results = {}

    # Test each implementation
    implementations = {
        "Naive": project_on_polyline_naive,
        "Blockwise": lambda p, s: project_on_polyline_blockwise(
            p, s, block_size=min(64, n_segments)
        ),
    }

    implementations["OpenCL"] = project_on_polyline_opencl

    for name, func in implementations.items():
        print(f"  Testing {name}...")

        # Warmup
        try:
            func(points[: min(100, n_points)], segments[: min(10, n_segments)])
        except Exception as e:
            print(f"  Error in warmup: {e}")
            continue

        # Benchmark
        times = []
        for _ in range(repeat):
            try:
                start = time.time()
                func(points, segments)
                times.append(time.time() - start)
            except Exception as e:
                print(f"  Error: {e}")
                break

        if times:
            best_time = min(times)
            results[name] = best_time
            print(f"  {name}: {best_time:.4f}s")
        else:
            print(f"  {name}: Failed")

    # Print speedups
    if "Naive" in results:
        naive_time = results["Naive"]
        for name, duration in results.items():
            if name != "Naive":
                speedup = naive_time / duration
                print(f"  {name} speedup: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    # Test cases from the requirements
    test_cases = [
        {"n_points": 8192, "n_segments": 7, "name": "Small case (N=8K, M=7)"},
        {"n_points": 100_000, "n_segments": 512, "name": "Large case (N=1M, M=1K)"},
    ]

    all_results = {}
    for case in test_cases:
        print(f"\nRunning {case['name']}")
        results = run_benchmark(case["n_points"], case["n_segments"], repeat=3)
        all_results[case["name"]] = results

    # Print summary
    print("\nSummary:")
    for case, results in all_results.items():
        print(f"{case}:")
        for impl, time in results.items():
            print(f"  {impl}: {time:.4f}s")
