# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

# src/get_data/tiles_clipper_robust.sh

#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Clip one or more LAZ files to a region polygon using PDAL, writing one LAZ.
#
# Usage:
#   bash src/get_data/tiles_clipper_robust.sh <region_geojson> <output_laz> <input_laz> [<input_laz> ...]
#
# Multiple inputs (a tile plus its neighbours' overlap) are merged before the
# crop so a tree straddling a tile boundary is clipped from the combined cloud
# (the "halo"). With a single input this is the plain per-tile clip. The region
# is the owning tile's core cell expanded by the halo margin, intersected with
# the buffered AOI; it is converted to WKT automatically.
# ---------------------------------------------------------------------------

set -euo pipefail

REGION_GEOJSON="${1:-}"
OUTPUT_LAZ="${2:-}"
shift 2 2>/dev/null || true
INPUTS=("$@")

if [[ -z "$REGION_GEOJSON" || -z "$OUTPUT_LAZ" || ${#INPUTS[@]} -eq 0 ]]; then
    echo "Usage: $0 <region_geojson> <output_laz> <input_laz> [<input_laz> ...]"
    exit 1
fi

if [[ ! -f "$REGION_GEOJSON" ]]; then
    echo "ERROR: region GeoJSON not found: $REGION_GEOJSON"
    exit 3
fi

for f in "${INPUTS[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: input LAZ not found: $f"
        exit 2
    fi
done

TILE_ID="$(basename "$(dirname "$OUTPUT_LAZ")")"
TMP_DIR="$(mktemp -d "/tmp/tmp_pdal_${TILE_ID}_XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

# Convert region polygon to WKT
WKT_FILE="$TMP_DIR/region.wkt"
ogrinfo -geom=YES -al "$REGION_GEOJSON" | grep POLYGON | head -n 1 | sed 's/^[ \t]*//' > "$WKT_FILE"

if [[ ! -s "$WKT_FILE" ]]; then
    echo "ERROR: Failed to extract WKT from $REGION_GEOJSON"
    exit 4
fi

# Build the PDAL pipeline: N readers -> merge -> crop -> writer.
PIPELINE_FILE="$TMP_DIR/${TILE_ID}_clip.json"
{
    echo '{'
    echo '  "pipeline": ['
    for f in "${INPUTS[@]}"; do
        echo "    \"$f\","
    done
    echo '    {"type": "filters.merge"},'
    echo '    {'
    echo '      "type": "filters.crop",'
    echo "      \"polygon\": \"$(cat "$WKT_FILE")\""
    echo '    },'
    echo '    {'
    echo '      "type": "writers.las",'
    echo "      \"filename\": \"$OUTPUT_LAZ\","
    echo '      "compression": true,'
    echo '      "minor_version": 4,'
    echo '      "dataformat_id": 8'
    echo '    }'
    echo '  ]'
    echo '}'
} > "$PIPELINE_FILE"

# Run PDAL clipping
echo "[$TILE_ID] Clipping ${#INPUTS[@]} input(s)..."
if ! pdal pipeline "$PIPELINE_FILE" > "$TMP_DIR/pdal.log" 2>&1; then
    echo "[$TILE_ID] ERROR: PDAL clipping failed:"
    cat "$TMP_DIR/pdal.log"
    exit 5
fi

echo "[$TILE_ID] Done: clipped to $OUTPUT_LAZ"
