from __future__ import annotations

import json
import os
from pathlib import Path


def get_cache_dir() -> str:
    if 'EXPLAINABOARD_CACHE' in os.environ:
        cache_dir = os.environ['EXPLAINABOARD_CACHE']
    elif 'HOME' in os.environ:
        cache_dir = os.path.join(os.environ['HOME'], '.cache', 'explainaboard', 'stats')
    else:
        raise FileNotFoundError(
            'Could not find cache directory for explainaboard.'
            'Please set EXPLAINABOARD_CACHE environment variable.'
        )
    return cache_dir


def get_statistics_path(dataset_name: str, subset_name: str | None = None) -> str:
    # Sanitize file path
    if '/' in dataset_name or (subset_name is not None and '/' in subset_name):
        raise ValueError(
            'dataset names cannot contain slashes:'
            f'dataset_name={dataset_name}, subset_name={subset_name}'
        )
    file_name = 'stats.json' if subset_name is None else f'stats-{subset_name}.json'
    return os.path.join(get_cache_dir(), dataset_name, file_name)


def read_statistics_from_cache(
    dataset_name: str,
    subset_name: str | None = None,
) -> dict | None:

    stats_path = get_statistics_path(dataset_name, subset_name)
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as stats_in:
            return json.load(stats_in)
    else:
        return None


def write_statistics_to_cache(
    content,
    dataset_name: str,
    subset_name: str | None = None,
):
    stats_path = get_statistics_path(dataset_name, subset_name)
    path_dir = Path(stats_path).parent.absolute()
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
    with open(stats_path, 'w') as stats_out:
        return json.dump(content, stats_out)
