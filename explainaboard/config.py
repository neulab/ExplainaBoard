from dataclasses import dataclass
from typing import Optional


@dataclass
class BuilderConfig:
    path_output_file: Optional[str] = None
    path_report: Optional[str] = None
    data_file: Optional[str] = None
    description: Optional[str] = None
    path_test_set: Optional[str] = None  # for question answering task


TORCH_VERSION = "N/A"
TORCH_AVAILABLE = False


TF_VERSION = "N/A"
TF_AVAILABLE = False


JAX_VERSION = "N/A"
JAX_AVAILABLE = False


# File Names
SYS_OUTPUT_INFO_FILENAME = "system_analysis.json"
