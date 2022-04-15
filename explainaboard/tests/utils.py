import os
import pathlib
from typing import Final

OPTIONAL_TEST_SUITES = ['cli_all']


def load_file_as_str(path) -> str:
    content = ""
    with open(path, 'r') as f:
        content = f.read()
    return content


test_artifacts_path: Final = os.path.join(
    os.path.dirname(pathlib.Path(__file__)), "artifacts"
)
