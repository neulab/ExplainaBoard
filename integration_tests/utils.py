from __future__ import annotations

import os
import pathlib
from typing import Final

# OPTIONAL_TEST_SUITES = ['cli_all']
OPTIONAL_TEST_SUITES: list[str] = []


def load_file_as_str(path) -> str:
    content = ""
    with open(path, "r") as f:
        content = f.read()
    return content


# TODO(odashi):
# Remove following variables and use environment variables or temporary directories
# instead.


test_artifacts_path: Final = os.path.join(
    os.path.dirname(pathlib.Path(__file__)), "artifacts"
)

top_path: Final = os.path.join(
    pathlib.Path(__file__).parent.parent.absolute(),
)
