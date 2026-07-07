#!/usr/bin/env bash
# Put the baked runtime environment (a conda env at /opt/env, no conda
# installation alongside it) on PATH, run its activation scripts, then run
# the given command.
#
# CFTree's main.py spawns each stage as a bare "python -m scripts.<stage>"
# subprocess, so /opt/env/bin must be first on PATH for the child stages to
# resolve the same interpreter. The activation scripts under
# etc/conda/activate.d set the variables the geospatial stack needs
# (GDAL_DATA, PDAL_DRIVER_PATH, PROJ settings). LD_LIBRARY_PATH covers the
# two compiled C++ binaries, whose build-time rpath points into the build
# stage's environment rather than /opt/env.
# Note: no `set -u` here. Activation scripts (e.g. libpdal-core's) reference
# variables that may be unset, which nounset turns into a fatal error that
# aborts activation. -e and pipefail are fine.
set -eo pipefail

export CONDA_PREFIX=/opt/env
export PATH="/opt/env/bin:$PATH"
export LD_LIBRARY_PATH="/opt/env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ -d /opt/env/etc/conda/activate.d ]; then
  for script in /opt/env/etc/conda/activate.d/*.sh; do
    [ -e "$script" ] && . "$script"
  done
fi

exec "$@"
