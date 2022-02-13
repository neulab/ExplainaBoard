from dataclasses import dataclass


@dataclass
class BuilderConfig:
    path_output_file: str = None
    path_report: str = None
    data_file: str = None
    description: str = None
    path_test_set: str = None  # for question answering task


TORCH_VERSION = "N/A"
TORCH_AVAILABLE = False


TF_VERSION = "N/A"
TF_AVAILABLE = False


JAX_VERSION = "N/A"
JAX_AVAILABLE = False


# File Names
SYS_OUTPUT_INFO_FILENAME = "system_analysis.json"
