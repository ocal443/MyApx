#!/usr/bin/env python3
"""
Benchmark runner script for polyline projection implementations.
Provides a simple interface similar to the C++ Google Benchmark setup.

Usage:
    python run_benchmarks.py                    # Run default benchmarks
    python run_benchmarks.py --quick           # Quick benchmarks only
    python run_benchmarks.py --comprehensive   # Full comprehensive suite
    python run_benchmarks.py --output-configs  # Test output configurations
    python run_benchmarks.py --memory          # Include memory profiling
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path so we can import polyproj
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from microbenchmark import MicroBenchmark, run_output_config_benchmarks, generate_test_data


def run_quick_benchmarks():
    """Run quick benchmarks for development."""
    print("Running quick benchmarks...")
    
    point_counts = [10_000, 100_000]
    segment_counts = [7, 64]
    
    benchmark = MicroBenchmark(min_time=0.05)  # Faster for development
    benchmark.run_suite(point_counts, segment_counts)


def run_default_benchmarks():
    """Run default benchmark suite."""
    print("Running default benchmark suite...")
    
    point_counts = [
        10_000,      # Small - includes naive
        100_000,     # Medium  
        1_000_000,   # Large
    ]
    
    segment_counts = [7, 64, 512]
    
    benchmark = MicroBenchmark(min_time=0.1)
    benchmark.run_suite(point_counts, segment_counts)


def run_comprehensive_benchmarks():
    """Run comprehensive benchmark suite."""
    print("Running comprehensive benchmark suite...")
    
    point_counts = [
        1_000,       # Very small - includes naive
        10_000,      # Small - includes naive  
        50_000,      # Medium-small - includes naive
        100_000,     # Medium
        500_000,     # Medium-large
        1_000_000,   # Large
        5_000_000,   # Very large
    ]
    
    segment_counts = [7, 32, 64, 128, 512, 1024]
    
    benchmark = MicroBenchmark(min_time=0.2)  # More accurate timing
    benchmark.run_suite(point_counts, segment_counts)


def run_scaling_benchmarks():
    """Run benchmarks to test scaling behavior."""
    print("Running scaling benchmarks...")
    
    # Test point count scaling (fixed segments)
    print("\n" + "="*60)
    print("POINT COUNT SCALING (M=7 segments)")
    print("="*60)
    
    point_counts = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    segment_counts = [7]
    
    benchmark = MicroBenchmark(min_time=0.1)
    benchmark.run_suite(point_counts, segment_counts)
    
    # Test segment count scaling (fixed points)
    print("\n" + "="*60)
    print("SEGMENT COUNT SCALING (N=100K points)")
    print("="*60)
    
    point_counts = [100_000]
    segment_counts = [1, 7, 16, 32, 64, 128, 256, 512, 1024]
    
    benchmark = MicroBenchmark(min_time=0.1)
    benchmark.run_suite(point_counts, segment_counts)


def run_block_size_optimization():
    """Run benchmarks to find optimal block sizes."""
    print("Running block size optimization benchmarks...")
    
    n_points = 500_000
    n_segments = 256
    
    points, segments = generate_test_data(n_points, n_segments)
    
    print(f"\nBlock Size Optimization (N={n_points:,}, M={n_segments})")
    print("-" * 60)
    
    from polyproj.vanilla import project_on_polyline_blockwise
    
    benchmark = MicroBenchmark(min_time=0.2)
    
    block_sizes = [8, 16, 32, 64, 128, 256, 512]
    
    results = []
    for block_size in block_sizes:
        result = benchmark.benchmark_function(
            lambda p, s, bs=block_size: project_on_polyline_blockwise(p, s, block_size=bs),
            f"Blockwise_{block_size}",
            points,
            segments
        )
        results.append((block_size, result.mean_time, result.points_per_second))
        print(f"Block size {block_size:3d}: {result.mean_time*1000:8.2f}ms "
              f"({result.points_per_second/1e6:6.1f}Mpts/s)")
    
    # Find optimal block size
    best_block_size, best_time, best_rate = min(results, key=lambda x: x[1])
    print(f"\nOptimal block size: {best_block_size} "
          f"({best_time*1000:.2f}ms, {best_rate/1e6:.1f}Mpts/s)")


def run_memory_benchmarks():
    """Run memory profiling benchmarks."""
    print("Running memory profiling benchmarks...")
    
    try:
        import psutil
    except ImportError:
        print("psutil not available, skipping memory benchmarks")
        return
    
    n_points = 1_000_000
    n_segments = 64
    
    points, segments = generate_test_data(n_points, n_segments)
    
    print(f"\nMemory Usage (N={n_points:,}, M={n_segments})")
    print("-" * 50)
    
    from polyproj.vanilla import project_on_polyline, project_on_polyline_blockwise
    
    try:
        from polyproj import project_on_polyline_cpp
        has_cpp = True
    except ImportError:
        has_cpp = False
    
    implementations = [
        ("Vanilla", project_on_polyline),
        ("Blockwise_64", lambda p, s: project_on_polyline_blockwise(p, s, block_size=64)),
        ("Blockwise_32", lambda p, s: project_on_polyline_blockwise(p, s, block_size=32)),
    ]
    
    if has_cpp:
        implementations.append(("C++", project_on_polyline_cpp))
    
    benchmark = MicroBenchmark(min_time=0.1)
    
    for name, func in implementations:
        result = benchmark.benchmark_function(func, name, points, segments)
        memory_str = f"{result.memory_peak_mb:.1f}MB" if result.memory_peak_mb else "N/A"
        print(f"{name:15}: {result.mean_time*1000:8.2f}ms, Memory: {memory_str}")


def run_pytest_benchmarks():
    """Run pytest-benchmark suite."""
    print("Running pytest-benchmark suite...")
    
    try:
        import pytest
    except ImportError:
        print("pytest not available, install with: pip install pytest pytest-benchmark")
        return
    
    # Run pytest benchmarks
    benchmark_file = Path(__file__).parent / "test_benchmarks.py"
    if not benchmark_file.exists():
        print(f"Benchmark file {benchmark_file} not found")
        return
    
    cmd = [
        "python", "-m", "pytest", 
        str(benchmark_file),
        "--benchmark-only",
        "--benchmark-sort=mean",
        "--benchmark-columns=min,max,mean,stddev,ops,rounds",
    ]
    
    import subprocess
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for polyline projection implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_benchmarks.py                    # Default benchmarks
    python run_benchmarks.py --quick           # Quick development benchmarks  
    python run_benchmarks.py --comprehensive   # Full benchmark suite
    python run_benchmarks.py --scaling         # Scaling analysis
    python run_benchmarks.py --block-size      # Block size optimization
    python run_benchmarks.py --output-configs  # Output configuration tests
    python run_benchmarks.py --memory          # Memory profiling
    python run_benchmarks.py --pytest          # Run pytest-benchmark suite
        """
    )
    
    parser.add_argument("--quick", action="store_true",
                        help="Run quick benchmarks for development")
    parser.add_argument("--comprehensive", action="store_true", 
                        help="Run comprehensive benchmark suite")
    parser.add_argument("--scaling", action="store_true",
                        help="Run scaling analysis benchmarks")
    parser.add_argument("--block-size", action="store_true",
                        help="Run block size optimization benchmarks")
    parser.add_argument("--output-configs", action="store_true",
                        help="Run output configuration benchmarks")
    parser.add_argument("--memory", action="store_true",
                        help="Run memory profiling benchmarks")
    parser.add_argument("--pytest", action="store_true",
                        help="Run pytest-benchmark suite")
    parser.add_argument("--all", action="store_true",
                        help="Run all benchmark types")
    
    args = parser.parse_args()
    
    # Check if any specific benchmark was requested
    specific_requested = any([
        args.quick, args.comprehensive, args.scaling, args.block_size,
        args.output_configs, args.memory, args.pytest, args.all
    ])
    
    # Print system info
    print("Polyline Projection Benchmarks")
    print("=" * 50)
    print(f"Python: {sys.version}")
    
    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: Not available")
    
    try:
        from polyproj import project_on_polyline_cpp
        print("C++ implementation: Available")
    except ImportError:
        print("C++ implementation: Not available")
    
    try:
        from polyproj.opencl import project_on_polyline_opencl
        print("OpenCL implementation: Available")
    except ImportError:
        print("OpenCL implementation: Not available")
    
    print()
    
    # Run requested benchmarks
    if args.all or not specific_requested:
        # Default: run standard benchmark suite
        run_default_benchmarks()
        run_output_config_benchmarks()
    
    if args.quick:
        run_quick_benchmarks()
    
    if args.comprehensive or args.all:
        run_comprehensive_benchmarks()
    
    if args.scaling or args.all:
        run_scaling_benchmarks()
    
    if args.block_size or args.all:
        run_block_size_optimization()
    
    if args.output_configs:
        run_output_config_benchmarks()
    
    if args.memory or args.all:
        run_memory_benchmarks()
    
    if args.pytest or args.all:
        success = run_pytest_benchmarks()
        if not success:
            print("pytest-benchmark failed")
            return 1
    
    print("\nBenchmarking complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())