#!/bin/bash
# This script must be run from the root directory of the repository.

EXPLAINABOARD_HIDE_PROGRESS=1 python -m unittest discover \
    -p '*_test.py' \
    -s explainaboard \
    -t . \
    -v
