"""Functions that allow caching of results."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import urllib.request

from explainaboard.utils.logging import get_logger


def sanitize_path(path: str) -> str:
    """Take in a path and remove any dangerous relative references.

    Args:
        path: The path to sanitize.

    Returns:
        The sanitized path.
    """
    return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")


def get_cache_dir() -> str:
    """Get the directory used for caching things.

    Returns:
        The cache directory.
    """
    if "EXPLAINABOARD_CACHE" in os.environ:
        cache_dir = os.environ["EXPLAINABOARD_CACHE"]
    elif "HOME" in os.environ:
        cache_dir = os.path.join(os.environ["HOME"], ".cache", "explainaboard")
    else:
        raise FileNotFoundError(
            "Could not find cache directory for explainaboard."
            "Please set EXPLAINABOARD_CACHE environment variable."
        )
    return cache_dir


def get_statistics_path(dataset_name: str, subset_name: str | None = None) -> str:
    """Get the path to statistics for a particular dataset.

    Args:
        dataset_name: The dataset.
        subset_name: The sub dataset.

    Returns:
        The path to the statistics.
    """
    # Sanitize file path
    if "/" in dataset_name or (subset_name is not None and "/" in subset_name):
        raise ValueError(
            "dataset names cannot contain slashes:"
            f"dataset_name={dataset_name}, subset_name={subset_name}"
        )
    file_name = "stats.json" if subset_name is None else f"stats-{subset_name}.json"
    return os.path.join(get_cache_dir(), "stats", dataset_name, file_name)


def read_statistics_from_cache(
    dataset_name: str,
    subset_name: str | None = None,
) -> dict | None:
    """Return statistics from the cache.

    Args:
        dataset_name: The name of the dataset.
        subset_name: The name of the sub-dataset.

    Returns:
        A dictionary of statistics or None if they don't exist
    """
    stats_path = get_statistics_path(dataset_name, subset_name)
    if os.path.exists(stats_path):
        with open(stats_path, "r") as stats_in:
            return json.load(stats_in)
    else:
        return None


def write_statistics_to_cache(
    content: dict,
    dataset_name: str,
    subset_name: str | None = None,
) -> None:
    """Write statistics to the cache.

    Args:
        content: The content to be written.
        dataset_name: The name of the dataset.
        subset_name: The name of the sub-dataset.
    """
    stats_path = get_statistics_path(dataset_name, subset_name)
    path_dir = Path(stats_path).parent.absolute()
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(stats_path, "w") as stats_out:
        json.dump(content, stats_out)


def cache_online_file(
    online_path: str, local_path: str, lifetime: datetime.timedelta | None = None
) -> str:
    """Caches an online file locally and returns the path to the local file.

    Args:
        online_path: The path online
        local_path: The relative path to the file locally
        lifetime: How long this file should be cached before reloading

    Returns:
        The absolute file to the cached path locally
    """
    sanitized_path = sanitize_path(local_path)
    file_path = os.path.join(get_cache_dir(), sanitized_path)
    # Use cached file if it exists and is young enough
    if os.path.exists(file_path):
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        age = datetime.datetime.now() - mod_time
        if lifetime is None or age <= lifetime:
            return file_path
    # Else download from online
    get_logger().info(f"Caching {online_path} to {file_path}")
    path_dir = Path(file_path).parent.absolute()
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    urllib.request.urlretrieve(online_path, file_path)
    return file_path


def open_cached_file(relative_path: str, lifetime: datetime.timedelta) -> str | None:
    """Get the path of a file that has been cached.

    Args:
        relative_path: The relative path in the cache directory
        lifetime: The amount of time that should pass before re-caching

    Returns:
        The path, or None if the file is not cached or stale.
    """
    sanitized_path = sanitize_path(relative_path)
    file_path = os.path.join(get_cache_dir(), sanitized_path)
    if os.path.exists(file_path):
        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
        age = datetime.datetime.now() - mod_time
        if lifetime is None or age < lifetime:
            return file_path
    return None
