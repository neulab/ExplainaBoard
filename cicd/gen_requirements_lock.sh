#!/bin/bash
# This script must be run from the root directory of the repository.

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "# Do not edit by hand. Automatically generated from ${BASH_SOURCE}." \
  > requirements.lock
python3 -m pip freeze >> requirements.lock
rm -fr .venv
