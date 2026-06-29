# syntax=docker/dockerfile:1
#
# CFTree pipeline image. Bakes the conda environment and the two compiled C++
# binaries (awrap_points, segmentation) so a colleague runs the pipeline with no
# manual WSL or conda setup and no C++ toolchain. The optional GPU morphometrics
# (warp-lang + cupy) are installed too; they are used only when the container is
# run with a GPU (see below) and CFTREE_GPU_METRICS=1.
#
# Build:
#   docker build -t cftree:local .
#
# Run (CPU): bind-mount a CFTree checkout at /work so cases/ and data/ stay on
# the host where the consumer's merge step reads them:
#   docker run --rm -v "$PWD":/work cftree:local \
#       python main.py --case wippolder --ahn-version 6 --n-cores 8 --buffer 20 --overwrite
#
# Run (GPU): add --gpus all (needs an NVIDIA driver on the host plus the
# nvidia-container-toolkit, which Docker Desktop provides) and enable the GPU
# morphometrics:
#   docker run --rm --gpus all -e CFTREE_GPU_METRICS=1 -v "$PWD":/work cftree:local \
#       python main.py --case wippolder --ahn-version 6 --n-cores 8 --buffer 20 --overwrite
#
# The binaries are baked at /opt/cftree/bin (outside /work) and reached through
# CFTREE_BIN, so the bind-mounted checkout, which carries no build/ outputs,
# still resolves them.
#
# License: this image embeds CGAL alpha-wrap (GPL-3.0) and the TreeSeparation
# binary, so the image is a GPL-3.0 distribution, like the repository.

FROM condaforge/miniforge3:latest

LABEL org.opencontainers.image.title="CFTree" \
      org.opencontainers.image.description="CFD-ready urban tree reconstruction pipeline" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.source="https://github.com/NoahAlting/CFTree"

SHELL ["/bin/bash", "-c"]

# ca-certificates backs the HTTPS AHN downloads done at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/cftree

# Create the conda env first, in its own layer, so editing source does not
# rebuild the slow (about 20 min, several GB) environment solve.
COPY environment.yml ./
RUN mamba env create -f environment.yml && mamba clean -afy

# Copy the source and build the two C++ binaries with the env's toolchain.
COPY . .

# CGAL 5.6.1 from conda-forge has three this->base() call sites in
# CGAL/boost/graph/iterator.h that GCC 15 rejects; replace them with the
# equivalent this->g test. Idempotent and a no-op on a CGAL version without the
# bug, so it survives a future env solve that ships a fixed CGAL.
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate cftree \
    && header="$CONDA_PREFIX/include/CGAL/boost/graph/iterator.h" \
    && if [ -f "$header" ]; then \
         sed -i 's/this->base() == nullptr/this->g == nullptr/g' "$header"; \
       fi \
    && cmake -S src/reconstruction/AlphaWrap -B src/reconstruction/AlphaWrap/build \
         -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    && cmake --build src/reconstruction/AlphaWrap/build -j "$(nproc)" \
    && cmake -S src/segmentation/TreeSeparation -B src/segmentation/TreeSeparation/build \
         -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    && cmake --build src/segmentation/TreeSeparation/build -j "$(nproc)"

# Bake the binaries at a stable path outside /work and point CFTREE_BIN at it,
# so a bind-mounted checkout (no build/ dirs) still resolves them.
RUN mkdir -p /opt/cftree/bin \
    && cp src/reconstruction/AlphaWrap/build/awrap_points /opt/cftree/bin/ \
    && cp src/segmentation/TreeSeparation/build/segmentation /opt/cftree/bin/
ENV CFTREE_BIN=/opt/cftree/bin

RUN install -m 0755 docker/entrypoint.sh /usr/local/bin/cftree-entrypoint
ENTRYPOINT ["/usr/local/bin/cftree-entrypoint"]

# The consumer overrides this with the full main.py invocation and mounts the
# checkout at /work. With no mount the baked source still answers --help.
WORKDIR /work
CMD ["python", "/opt/cftree/main.py", "--help"]
