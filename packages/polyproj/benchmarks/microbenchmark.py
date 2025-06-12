"""
Comprehensive microbenchmarking suite for polyline projection implementations.
Similar to Google Benchmark but for Python.
"""

import gc
import time
import statistics
import sys
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np

# Import all available implementations
from polyproj.vanilla import (
    project_on_polyline_naive,
    project_on_polyline,
    project_on_polyline_blockwise
)

try:
    from polyproj import project_on_polyline_cpp
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    project_on_polyline_cpp = None

try:
    from polyproj.opencl import project_on_polyline_opencl
    HAS_OPENCL = True
except ImportError:
    HAS_OPENCL = False
    project_on_polyline_opencl = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    n_points: int
    n_segments: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    points_per_second: float
    bytes_per_second: float
    time_per_point_per_segment_ns: float
    point_segments_per_second: float
    memory_peak_mb: Optional[float] = None

    def __str__(self) -> str:
        return (f"{self.name:30} "
                f"N={self.n_points:>7} M={self.n_segments:>4} "
                f"Time={self.mean_time*1000:>8.2f}ms "
                f"Rate={self.points_per_second/1e6:>6.1f}Mpts/s "
                f"PtSeg={self.point_segments_per_second/1e9:>6.1f}Gops/s "
                f"({self.time_per_point_per_segment_ns:>5.1f}ns/op) "
                f"±{self.std_time/self.mean_time*100:>4.1f}%")


class MemoryTracker:
    """Simple memory usage tracker."""

    def __init__(self):
        self.peak_mb = 0.0
        self.initial_mb = 0.0

    @contextmanager
    def track(self):
        """Context manager to track peak memory usage."""
        gc.collect()
        self.initial_mb = self._get_memory_mb()
        self.peak_mb = self.initial_mb

        try:
            yield self
        finally:
            gc.collect()
            final_mb = self._get_memory_mb()
            self.peak_mb = max(self.peak_mb, final_mb)

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def update_peak(self):
        """Update peak memory if current usage is higher."""
        current = self._get_memory_mb()
        self.peak_mb = max(self.peak_mb, current)


def generate_test_data(n_points: int, n_segments: int, seed: int = 42, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """Generate reproducible test data."""
    rng = np.random.RandomState(seed)

    # Generate points
    points = rng.uniform(0.0, 100.0, size=(n_points, 2)).astype(dtype)

    # Generate segments as polyline corners
    corners = rng.uniform(0.0, 100.0, size=(n_segments + 1, 2)).astype(dtype)
    segments = np.zeros((n_segments, 2, 2), dtype=dtype)
    for i in range(n_segments):
        segments[i, 0] = corners[i]
        segments[i, 1] = corners[i + 1]

    return points, segments


class MicroBenchmark:
    """Main benchmarking class."""

    def __init__(self, min_time: float = 0.1, max_iterations: int = 1000):
        self.min_time = min_time  # Minimum time to run each benchmark
        self.max_iterations = max_iterations
        self.results: List[BenchmarkResult] = []

    def _estimate_iterations(self, func: Callable, args: tuple) -> int:
        """Estimate how many iterations we need for reliable timing."""
        # Quick test run
        start = time.perf_counter()
        func(*args)
        single_time = time.perf_counter() - start

        if single_time == 0:
            return self.max_iterations

        # Calculate iterations needed for min_time
        estimated = max(1, int(self.min_time / single_time))
        return min(estimated, self.max_iterations)

    def _run_timed(self, func: Callable, args: tuple, iterations: int) -> Tuple[List[float], Optional[float]]:
        """Run function multiple times and collect timing data."""
        times = []
        memory_tracker = MemoryTracker()

        with memory_tracker.track():
            for _ in range(iterations):
                # Force garbage collection before each run
                if len(times) % 10 == 0:
                    gc.collect()
                    memory_tracker.update_peak()

                start = time.perf_counter()
                result = func(*args)
                end = time.perf_counter()

                times.append(end - start)

                # Prevent compiler optimizations
                if hasattr(result, 'distances') and result.distances is not None:
                    _ = result.distances.sum()

        peak_memory = memory_tracker.peak_mb - memory_tracker.initial_mb
        return times, peak_memory if peak_memory > 0 else None

    def benchmark_function(self, func: Callable, name: str, points: np.ndarray,
                          segments: np.ndarray, **kwargs) -> BenchmarkResult:
        """Benchmark a single function."""
        n_points, n_segments = points.shape[0], segments.shape[0]
        
        # Add precision info to name
        precision = "f64" if points.dtype == np.float64 else "f32"
        name_with_precision = f"{name}_{precision}"

        # Prepare arguments
        args = (points, segments)
        if kwargs:
            # For functions that take additional keyword arguments
            func_with_kwargs = lambda p, s: func(p, s, **kwargs)
            args = (points, segments)
            test_func = func_with_kwargs
        else:
            test_func = func

        try:
            # Warmup run
            test_func(*args)

            # Estimate iterations
            iterations = self._estimate_iterations(test_func, args)

            # Run benchmark
            times, peak_memory = self._run_timed(test_func, args, iterations)

            # Calculate statistics
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0
            min_time = min(times)
            max_time = max(times)

            # Calculate rates
            points_per_second = n_points / mean_time
            bytes_per_second = n_points * 8 / mean_time  # Assuming 8 bytes per point (2 float32)
            time_per_point_per_segment_ns = (mean_time * 1e9) / (n_points * n_segments)
            point_segments_per_second = (n_points * n_segments) / mean_time

            result = BenchmarkResult(
                name=name_with_precision,
                n_points=n_points,
                n_segments=n_segments,
                mean_time=mean_time,
                std_time=std_time,
                min_time=min_time,
                max_time=max_time,
                iterations=iterations,
                points_per_second=points_per_second,
                bytes_per_second=bytes_per_second,
                time_per_point_per_segment_ns=time_per_point_per_segment_ns,
                point_segments_per_second=point_segments_per_second,
                memory_peak_mb=peak_memory
            )

            self.results.append(result)
            return result

        except Exception as e:
            print(f"Error benchmarking {name}: {e}")
            # Return dummy result
            precision = "f64" if points.dtype == np.float64 else "f32"
            return BenchmarkResult(
                name=f"{name}_{precision} (FAILED)",
                n_points=n_points,
                n_segments=n_segments,
                mean_time=float('inf'),
                std_time=0.0,
                min_time=float('inf'),
                max_time=float('inf'),
                iterations=0,
                points_per_second=0.0,
                bytes_per_second=0.0,
                time_per_point_per_segment_ns=float('inf'),
                point_segments_per_second=0.0
            )

    def run_suite(self, point_counts: List[int], segment_counts: List[int]):
        """Run complete benchmark suite."""
        print("=" * 80)
        print("Polyline Projection Microbenchmarks")
        print("=" * 80)

        # Define implementations to test
        # Note: OpenCL always uses f32, others adapt to input dtype
        implementations = [
            ("Vanilla", project_on_polyline),
            ("Blockwise_64", lambda p, s: project_on_polyline_blockwise(p, s, block_size=64)),
            ("Blockwise_32", lambda p, s: project_on_polyline_blockwise(p, s, block_size=32)),
        ]

        if HAS_CPP:
            implementations.append(("C++", project_on_polyline_cpp))

        # OpenCL is always f32, so we'll handle it separately
        opencl_implementations = []
        if HAS_OPENCL:
            opencl_implementations.append(("OpenCL", project_on_polyline_opencl))

        # Only include naive for small datasets
        naive_max_points = 50000  # Avoid hanging on large datasets

        # Test both f32 and f64 precisions
        precisions = [
            (np.float32, "f32"),
            (np.float64, "f64")
        ]

        for n_points in point_counts:
            for n_segments in segment_counts:
                print(f"\nBenchmarking N={n_points:,} points, M={n_segments} segments")
                print("-" * 80)

                baseline_times = {}  # Track baseline for each precision
                
                for dtype, precision_name in precisions:
                    print(f"\n  === {precision_name.upper()} Precision ===")
                    
                    # Generate test data with specified precision
                    points, segments = generate_test_data(n_points, n_segments, dtype=dtype)

                    # Test implementations that support this precision
                    current_implementations = implementations.copy()
                    if n_points <= naive_max_points:
                        current_implementations.insert(0, ("Naive", project_on_polyline_naive))

                    baseline_time = None
                    for name, func in current_implementations:
                        result = self.benchmark_function(func, name, points, segments)
                        print(result)

                        # Track baseline for speedup calculations
                        if name == "Naive" or (baseline_time is None and name == "Vanilla"):
                            baseline_time = result.mean_time
                            baseline_times[precision_name] = baseline_time
                        elif baseline_time is not None and result.mean_time > 0:
                            speedup = baseline_time / result.mean_time
                            print(f"                               Speedup: {speedup:.2f}x")

                # Test OpenCL separately (always f32, but test with both input precisions)
                if opencl_implementations:
                    print(f"\n  === OpenCL (always f32 internally) ===")
                    for dtype, precision_name in precisions:
                        points, segments = generate_test_data(n_points, n_segments, dtype=dtype)
                        
                        for name, func in opencl_implementations:
                            # Add note about input precision for OpenCL
                            opencl_name = f"{name}_input{precision_name}"
                            result = self.benchmark_function(func, opencl_name, points, segments)
                            print(result)
                            
                            # Compare to baseline if available
                            if precision_name in baseline_times and result.mean_time > 0:
                                speedup = baseline_times[precision_name] / result.mean_time
                                print(f"                               Speedup vs {precision_name}: {speedup:.2f}x")

        self._print_summary()

    def _print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            return

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        # Group by problem size
        by_size = {}
        for result in self.results:
            key = (result.n_points, result.n_segments)
            if key not in by_size:
                by_size[key] = []
            by_size[key].append(result)

        for (n_points, n_segments), results in by_size.items():
            print(f"\nN={n_points:,} points, M={n_segments} segments:")

            # Sort by performance
            results.sort(key=lambda r: r.mean_time)

            best_time = results[0].mean_time if results[0].mean_time != float('inf') else None

            for result in results:
                if result.mean_time == float('inf'):
                    print(f"  {result.name:15}: FAILED")
                else:
                    speedup = ""
                    if best_time and result.mean_time > best_time:
                        speedup = f" ({best_time/result.mean_time:.2f}x slower)"
                    elif best_time and result.mean_time == best_time:
                        speedup = " (fastest)"

                    memory_info = ""
                    if result.memory_peak_mb:
                        memory_info = f" [{result.memory_peak_mb:.1f}MB]"

                    print(f"  {result.name:20}: {result.mean_time*1000:8.2f}ms "
                          f"({result.points_per_second/1e6:.1f}Mpts/s) "
                          f"({result.point_segments_per_second/1e9:.1f}Gops/s) "
                          f"[{result.time_per_point_per_segment_ns:.1f}ns/op]{speedup}{memory_info}")


def run_output_config_benchmarks():
    """Benchmark different output configurations like the C++ version."""
    print("\n" + "=" * 80)
    print("Output Configuration Benchmarks")
    print("=" * 80)

    n_points = 1_000_000
    n_segments = 7

    benchmark = MicroBenchmark(min_time=0.5)  # Longer runs for accuracy

    configs = [
        ("distance_only", {"return_distance": True}),
        ("full_output", {
            "return_distance": True,
            "return_projection": True,
            "return_param": True,
            "return_index": True
        }),
    ]

    # Test both precisions
    precisions = [(np.float32, "f32"), (np.float64, "f64")]
    
    for dtype, precision_name in precisions:
        print(f"\n=== {precision_name.upper()} Precision ===")
        points, segments = generate_test_data(n_points, n_segments, dtype=dtype)

        if HAS_CPP:
            for config_name, kwargs in configs:
                name = f"C++_{config_name}"
                result = benchmark.benchmark_function(
                    project_on_polyline_cpp, name, points, segments, **kwargs
                )
                print(result)

        # Test vanilla implementation too
        for config_name, kwargs in configs:
            name = f"Vanilla_{config_name}"
            result = benchmark.benchmark_function(
                project_on_polyline, name, points, segments, **kwargs
            )
            print(result)

    # Test OpenCL separately (always f32 internally)
    if HAS_OPENCL:
        print(f"\n=== OpenCL (always f32 internally) ===")
        for dtype, precision_name in precisions:
            points, segments = generate_test_data(n_points, n_segments, dtype=dtype)
            for config_name, kwargs in configs:
                name = f"OpenCL_{config_name}_input{precision_name}"
                result = benchmark.benchmark_function(
                    project_on_polyline_opencl, name, points, segments, **kwargs
                )
                print(result)


def main():
    """Run all benchmarks."""
    # Test different problem sizes
    point_counts = [
        1_000,      # Small
        10_000,     # Medium
        100_000,    # Large
        1_000_000,  # Very large
    ]

    segment_counts = [7, 64, 512]  # Different polyline complexities

    benchmark = MicroBenchmark(min_time=0.1)
    benchmark.run_suite(point_counts, segment_counts)

    # Additional output configuration benchmarks
    run_output_config_benchmarks()


if __name__ == "__main__":
    main()
