#!/usr/bin/env bash
# Activate the baked conda environment, then run the given command.
#
# CFTree's main.py spawns each stage as a bare "python -m scripts.<stage>"
# subprocess, so the cftree environment must be the active python on PATH for
# the child stages to resolve the same interpreter. Activating here puts the
# environment first on PATH and sets LD_LIBRARY_PATH, so both the geospatial
# stack and the baked C++ binaries find their shared libraries.
# Note: no `set -u` here. conda's activate.d scripts (e.g. libpdal-core)
# reference variables like PDAL_DRIVER_PATH that may be unset, which nounset
# turns into a fatal error that aborts activation. -e and pipefail are fine.
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate cftree
exec "$@"
