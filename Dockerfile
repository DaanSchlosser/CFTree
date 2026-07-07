# syntax=docker/dockerfile:1
#
# CFTree pipeline image, built in stages so the published image carries only
# what the pipeline runs with:
#
#   build-binaries : compiles awrap_points and segmentation with the full
#                    C++ toolchain (docker/environment.build.yml)
#   runtime-env    : solves the pruned runtime conda environment at /opt/env
#                    (docker/environment.runtime.yml) and strips caches,
#                    headers, and docs
#   test           : the final image plus pytest, so CI runs the suite
#                    against the exact environment consumers get
#   (final)        : debian-slim + /opt/env + the source + the two binaries
#
# The toolchain, linters, type checkers, and plotting stack of the
# development environment (environment.yml) never enter the final image.
#
# Build:
#   docker build -t cftree:local .
#
# Run (CPU): bind-mount a CFTree checkout at /work so cases/ and data/ stay on
# the host where the consumer's merge step reads them:
#   docker run --rm -v "$PWD":/work cftree:local \
#       python main.py --case wippolder --ahn-version 6 --n-cores 8 --buffer 20 --overwrite
#
# GPU morphometrics (CFTREE_GPU_METRICS=1) are an opt-in extra and the
# warp-lang backend is not baked in; the CPU path takes over automatically.
# To run them in a container, derive an image:
#   FROM cftree:local
#   RUN python -m pip install --no-cache-dir warp-lang
# and run it with --gpus all (NVIDIA driver plus nvidia-container-toolkit on
# the host, which Docker Desktop provides).
#
# The binaries are baked at /opt/cftree/bin (outside /work) and reached through
# CFTREE_BIN, so the bind-mounted checkout, which carries no build/ outputs,
# still resolves them.
#
# License: this image embeds CGAL alpha-wrap (GPL-3.0) and the TreeSeparation
# binary, so the image is a GPL-3.0 distribution, like the repository.

# ---------------------------------------------------------------------------
# Stage 1: compile the two C++ binaries with a toolchain-only environment.
# ---------------------------------------------------------------------------
FROM condaforge/miniforge3:latest AS build-binaries

SHELL ["/bin/bash", "-c"]
WORKDIR /opt/cftree

COPY docker/environment.build.yml ./
RUN mamba env create -p /opt/build-env -f environment.build.yml && mamba clean -afy

COPY src/reconstruction/AlphaWrap src/reconstruction/AlphaWrap
COPY src/segmentation/TreeSeparation src/segmentation/TreeSeparation

# CGAL 5.6.1 from conda-forge has three this->base() call sites in
# CGAL/boost/graph/iterator.h that GCC 15 rejects; replace them with the
# equivalent this->g test. Idempotent and a no-op on a CGAL version without the
# bug, so it survives a future env solve that ships a fixed CGAL.
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate /opt/build-env \
    && header="$CONDA_PREFIX/include/CGAL/boost/graph/iterator.h" \
    && if [ -f "$header" ]; then \
         sed -i 's/this->base() == nullptr/this->g == nullptr/g' "$header"; \
       fi \
    && cmake -S src/reconstruction/AlphaWrap -B src/reconstruction/AlphaWrap/build \
         -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    && cmake --build src/reconstruction/AlphaWrap/build -j "$(nproc)" \
    && cmake -S src/segmentation/TreeSeparation -B src/segmentation/TreeSeparation/build \
         -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    && cmake --build src/segmentation/TreeSeparation/build -j "$(nproc)" \
    && mkdir -p /out \
    && cp src/reconstruction/AlphaWrap/build/awrap_points /out/ \
    && cp src/segmentation/TreeSeparation/build/segmentation /out/

# ---------------------------------------------------------------------------
# Stage 2: solve the runtime environment at its final absolute path, so no
# path relocation is needed when it is copied into the final stage.
# ---------------------------------------------------------------------------
FROM condaforge/miniforge3:latest AS runtime-env

SHELL ["/bin/bash", "-c"]

COPY docker/environment.runtime.yml /tmp/environment.runtime.yml
RUN mamba env create -p /opt/env -f /tmp/environment.runtime.yml \
    && mamba clean -afy \
    # Strip what the pipeline cannot need at runtime: static libraries,
    # C/C++ headers, bytecode caches, docs, and the test suites that ship
    # inside numpy/scipy/pandas. The pytest run in the test stage guards
    # against stripping too much.
    && find /opt/env -name '*.a' -delete \
    && find /opt/env -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /opt/env/lib/python3.11/site-packages -type d \( -name tests -o -name test \) -prune -exec rm -rf {} + \
    && rm -rf /opt/env/include \
    && rm -rf /opt/env/share/man /opt/env/share/doc /opt/env/share/info \
              /opt/env/share/locale /opt/env/share/gtk-doc /opt/env/share/terminfo

# ---------------------------------------------------------------------------
# Final stage: a slim base, the runtime environment, the source, the binaries.
# ---------------------------------------------------------------------------
FROM debian:12-slim AS runtime

LABEL org.opencontainers.image.title="CFTree" \
      org.opencontainers.image.description="CFD-ready urban tree reconstruction pipeline" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/NoahAlting/CFTree"

# ca-certificates backs the HTTPS AHN downloads PDAL makes through libcurl at
# runtime (python requests carries its own certifi bundle).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=runtime-env /opt/env /opt/env

# On PATH at build time too, so a derived image's RUN steps (for example the
# GPU derivation in the README) reach python without going through the
# entrypoint, which docker build never invokes. The entrypoint remains
# responsible for LD_LIBRARY_PATH and the conda activation scripts.
ENV PATH="/opt/env/bin:$PATH"

WORKDIR /opt/cftree
COPY . .

# The binaries live at a stable path outside /work and CFTREE_BIN points at
# it, so a bind-mounted checkout (no build/ dirs) still resolves them.
COPY --from=build-binaries /out/awrap_points /out/segmentation /opt/cftree/bin/
ENV CFTREE_BIN=/opt/cftree/bin

RUN install -m 0755 docker/entrypoint.sh /usr/local/bin/cftree-entrypoint
ENTRYPOINT ["/usr/local/bin/cftree-entrypoint"]

# The consumer overrides this with the full main.py invocation and mounts the
# checkout at /work. With no mount the baked source still answers --help.
WORKDIR /work
CMD ["python", "/opt/cftree/main.py", "--help"]

# ---------------------------------------------------------------------------
# Test stage: the final image plus pytest, so the CI suite runs against the
# exact dependency set a consumer gets. Built with --target test; not
# published.
# ---------------------------------------------------------------------------
FROM runtime AS test
RUN /opt/env/bin/python -m pip install --no-cache-dir pytest

# Default build target: the runtime image, not the test derivation.
FROM runtime
