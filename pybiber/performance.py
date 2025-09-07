"""
Performance optimization utilities for pybiber.

This module provides caching, memory optimization, and performance monitoring
utilities for corpus analysis operations. These tools help improve performance
for large corpora and repeated analyses while providing insights into
processing bottlenecks.

Classes:
    PerformanceCache: File-based caching for expensive computations
    MemoryOptimizer: Memory usage optimization for large datasets
    ProgressTracker: Progress tracking for long-running operations
    PerformanceMonitor: Performance monitoring and timing utilities

Functions:
    cached_result: Decorator for automatic result caching
    memory_efficient_join: Memory-optimized DataFrame joins
    optimize_polars_settings: Optimize Polars configuration

Example:
    Using automatic caching::

        from pybiber.performance import cached_result

        @cached_result
        def expensive_analysis(data):
            # This result will be cached automatically
            return complex_computation(data)

    Memory optimization for large corpora::

        from pybiber.performance import MemoryOptimizer

        optimizer = MemoryOptimizer()

        if optimizer.is_large_corpus(tokens):
            print("Using memory optimizations...")
            with optimizer.batch_processing(tokens) as batches:
                for batch in batches:
                    process_batch(batch)

    Performance monitoring::

        from pybiber.performance import PerformanceMonitor

        with PerformanceMonitor("My analysis") as monitor:
            result = expensive_operation()

        print(f"Operation took {monitor.elapsed_time:.2f} seconds")

.. codeauthor:: David Brown <dwb2@andrew.cmu.edu>
"""

import time
import gc
from functools import wraps
from typing import Dict, Any, Optional, Callable, Union
import hashlib
import pickle
from pathlib import Path

import polars as pl

from .config import CONFIG


class PerformanceCache:
    """
    File-based cache for expensive computations.

    This class provides a simple but effective caching mechanism that stores
    computation results to disk, allowing for persistence across Python sessions.
    The cache uses content-based hashing to ensure cache validity.

    Attributes:
        cache_dir: Path to the cache directory
        max_size: Maximum number of cached items (not strictly enforced)

    Example:
        Basic caching usage::

            cache = PerformanceCache()

            # Store a result
            cache.set("my_key", expensive_computation_result)

            # Retrieve a result
            if cache.has("my_key"):
                result = cache.get("my_key")
            else:
                result = expensive_computation()
                cache.set("my_key", result)

        Custom cache directory::

            cache = PerformanceCache("/tmp/my_cache", max_size=50)
    """  # noqa: E501

    def __init__(self, cache_dir: Optional[str] = None, max_size: int = 100):
        default_cache = str(Path.home() / ".pybiber_cache")
        cache_dir_path = cache_dir if cache_dir else default_cache
        self.cache_dir = Path(cache_dir_path)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self._access_times: Dict[str, float] = {}

    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a cache key from function name and arguments."""
        # Create a hash of the function name and arguments
        content = f"{func_name}_{str(args)}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(content.encode()).hexdigest()

    def _cleanup_old_files(self):
        """Remove oldest cache files if we exceed max_size."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        if len(cache_files) <= self.max_size:
            return

        # Sort by access time and remove oldest
        files_with_times = [
            (f, self._access_times.get(f.stem, f.stat().st_mtime)) for f in cache_files  # noqa: E501
        ]
        files_with_times.sort(key=lambda x: x[1])

        # Remove oldest files
        for file_path, _ in files_with_times[: -self.max_size]:
            file_path.unlink(missing_ok=True)
            self._access_times.pop(file_path.stem, None)

    def get(self, func_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Get cached result if available."""
        if not CONFIG.ENABLE_CACHING:
            return None

        cache_key = self._get_cache_key(func_name, args, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    result = pickle.load(f)
                self._access_times[cache_key] = time.time()
                return result
            except (pickle.PickleError, EOFError):
                cache_file.unlink(missing_ok=True)

        return None

    def set(self, func_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache a result."""
        if not CONFIG.ENABLE_CACHING:
            return

        cache_key = self._get_cache_key(func_name, args, kwargs)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            self._access_times[cache_key] = time.time()
            self._cleanup_old_files()
        except (pickle.PickleError, OSError):
            # If caching fails, just continue without caching
            pass


# Global cache instance
_cache = PerformanceCache(max_size=CONFIG.CACHE_MAX_SIZE)


def cached_result(func: Callable) -> Callable:
    """Decorator to cache function results."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to get from cache
        cached = _cache.get(func.__name__, args, kwargs)
        if cached is not None:
            return cached

        # Compute result
        result = func(*args, **kwargs)

        # Cache result
        _cache.set(func.__name__, args, kwargs, result)

        return result

    return wrapper


class MemoryOptimizer:
    """Utilities for memory-efficient processing."""

    @staticmethod
    def is_large_corpus(tokens_table: pl.DataFrame) -> bool:
        """Check if corpus is large and needs memory optimization."""
        doc_count = tokens_table.select(pl.col("doc_id").n_unique()).item()
        return doc_count > CONFIG.LARGE_CORPUS_THRESHOLD

    @staticmethod
    def is_large_corpus_size(size: int) -> bool:
        """Check if corpus size is large and needs memory optimization."""
        return size > CONFIG.LARGE_CORPUS_THRESHOLD

    @staticmethod
    def optimize_dataframe(df: pl.DataFrame) -> pl.DataFrame:
        """Optimize DataFrame memory usage."""
        if not CONFIG.MEMORY_EFFICIENT_MODE:
            return df

        # Use lazy evaluation where possible
        if hasattr(df, "lazy"):
            return df.lazy().collect(streaming=True)

        return df

    @staticmethod
    def batch_process(
        data: pl.DataFrame,
        process_func: Callable,
        batch_size: Optional[int] = None,
        **kwargs,
    ) -> pl.DataFrame:
        """Process data in batches to reduce memory usage."""
        if batch_size is None:
            batch_size = CONFIG.DEFAULT_BATCH_SIZE * 100

        if len(data) <= batch_size:
            return process_func(data, **kwargs)

        results = []

        for i in range(0, len(data), batch_size):
            batch = data[i: i + batch_size]
            result = process_func(batch, **kwargs)
            results.append(result)

            # Force garbage collection after each batch
            if CONFIG.MEMORY_EFFICIENT_MODE:
                gc.collect()

        return pl.concat(results)


class ProgressTracker:
    """Lightweight progress tracking for long-running operations."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self.show_progress = total > CONFIG.PROGRESS_THRESHOLD

        if self.show_progress:
            print(f"Starting {description} ({total:,} items)...")

    def update(self, increment: int = 1):
        """Update progress counter."""
        self.current += increment

        if self.show_progress and self.current % max(1, self.total // 10) == 0:
            elapsed = time.time() - self.start_time
            percent = (self.current / self.total) * 100
            rate = self.current / elapsed if elapsed > 0 else 0

            print(
                f"{self.description}: {percent:.1f}% ({self.current:,}/{self.total:,}) "  # noqa: E501
                f"[{rate:.1f} items/sec]"
            )

    def finish(self):
        """Mark progress as complete."""
        if self.show_progress:
            elapsed = time.time() - self.start_time
            print(
                f"{self.description} completed in {elapsed:.2f}s "
                f"({self.current:,} items processed)"
            )


class PerformanceMonitor:
    """Monitor performance metrics for optimization."""

    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None
        self.end_time = None
        self.peak_memory = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time

        if elapsed > 5.0:  # Only log operations taking more than 5 seconds
            print(f"Performance: {self.operation} completed in {elapsed:.2f}s")


def memory_efficient_join(
    left: pl.DataFrame, right: pl.DataFrame, on: Union[str, list], how: str = "inner"  # noqa: E501
) -> pl.DataFrame:
    """Memory-efficient join for large DataFrames."""
    if not CONFIG.MEMORY_EFFICIENT_MODE:
        return left.join(right, on=on, how=how)

    # Use lazy evaluation for large joins
    return left.lazy().join(right.lazy(), on=on, how=how).collect(streaming=True)  # noqa: E501


def optimize_polars_settings():
    """Set optimal Polars configuration for performance."""
    # These settings can be adjusted based on system resources
    pl.Config.set_tbl_rows(20)  # Limit display rows
    pl.Config.set_tbl_cols(10)  # Limit display columns

    # Use streaming for large datasets if available
    if hasattr(pl.Config, "set_streaming_chunk_size"):
        pl.Config.set_streaming_chunk_size(10000)


# Initialize optimal settings on import
optimize_polars_settings()
