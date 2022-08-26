#!/bin/bash -x
# Generates lock files of requirements for multiple Python versions to cache
# dependencies in CI.
#
# This script must be run from the root directory of the repository. This script
# requires Docker to be installed.
#
# Usage:
# ./cicd/gen_requirements_lock.sh

run_docker() {
    local python_version=$1
    local image_name=$2
    local container_name="explainaboard-deps"

    docker pull ${image_name}
    docker run --rm -it --name ${container_name} -v `pwd`:/work -w /work \
        -t -d ${image_name} /bin/bash

    docker exec ${container_name} /bin/bash \
        -c "./cicd/gen_requirements_lock.sh ${python_version}"

    docker stop ${container_name}
}

gen_lockfile() {
    local python_version=$1
    local lock_file=.github/workflows/requirements.lock.${python_version}

    python3 -m venv /tmp/venv
    source /tmp/venv/bin/activate
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    echo "# Do not edit by hand. Automatically generated from ${BASH_SOURCE}." \
        > ${lock_file}
    python3 -m pip freeze >> ${lock_file}
}

if [ $# -eq 0 ]; then
    for python_version in 3.8 3.9 3.10; do
        run_docker ${python_version} python:${python_version}-slim
    done
else
    gen_lockfile $1
fi
