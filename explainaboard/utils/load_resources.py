"""Code for loading resources."""

from __future__ import annotations

import json
import os
from typing import Optional

from explainaboard.config import CUSTOMIZED_FEATURES_CONFIG_FILE


def get_customized_features(path_file: Optional[str] = None) -> dict:
    """Get customized features from configuration file.

    Returns:
        A dictionary of customized features.
    """
    customized_feature_config_file = (
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../resources/" + CUSTOMIZED_FEATURES_CONFIG_FILE,
        )
        if path_file is None
        else path_file
    )
    with open(customized_feature_config_file, "r") as file:
        customized_features = json.loads(file.read())
    return customized_features
