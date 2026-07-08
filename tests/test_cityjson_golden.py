# Copyright (C) 2025 Noah Alting
# Licensed under the GNU General Public License v3.0
# See the LICENSE file for more details.

"""Golden-file test for the CityJSON writer's byte-identical output contract.

finalize_tile documents that the CityJSON bytes are identical no matter how the
batches were scheduled, and the downstream consumer's reuse manifest relies on
reruns reproducing the same file. The writer is pure (dict in, dict out), so
one fixed tree pinned byte-for-byte turns any drift — quantization, key order,
lod formatting, a numpy upgrade that changes rounding — into a red test instead
of a silently different published output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reconstruction.write_cityjson import add_tree, finalize_cityjson, init_cityjson  # noqa: E402

EXPECTED = (
    '{"type":"CityJSON","version":"2.0","CityObjects":{"T_7":{"type":"SolitaryVegetationObject",'
    '"geometry":[{"type":"Solid","lod":3.0,"boundaries":[[[[0,1,2]],[[0,1,3]],[[0,2,3]],[[1,2,3]]]]},'
    '{"type":"Solid","lod":3.0,"boundaries":[[[[4,5,6]],[[4,5,7]],[[4,6,7]],[[5,6,7]]]]}],'
    '"attributes":{"height_m":7.5,"crown_width_m":2.0,"r50_m":null,"porosity":null}}},'
    '"vertices":[[0,0,4000],[2000,0,4000],[0,2000,4000],[0,0,7500],[0,0,0],[500,0,0],[0,500,0],[0,0,4000]],'
    '"transform":{"scale":[0.001,0.001,0.001],"translate":[85000.25,446000.75,1.5]},'
    '"metadata":{"referenceSystem":"https://www.opengis.net/def/crs/EPSG/0/28992",'
    '"geographicalExtent":[85000.25,446000.75,1.5,85002.25,446002.75,9.0],"presentLoDs":[3.0]}}'
)


def _one_tree_city() -> dict:
    city = init_cityjson()
    components = [
        {
            "role": "crown",
            "lod": 3.0,
            "vertices_local": [[0.0, 0.0, 4.0], [2.0, 0.0, 4.0], [0.0, 2.0, 4.0], [0.0, 0.0, 7.5]],
            "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        },
        {
            "role": "trunk",
            "lod": 3.0,
            "vertices_local": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 4.0]],
            "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        },
    ]
    attributes = {"height_m": 7.5, "crown_width_m": 2.0, "r50_m": None, "porosity": None}
    add_tree(city, 7, components, [85000.25, 446000.75, 1.5], attributes)
    return finalize_cityjson(city, crs="EPSG:28992")


def test_cityjson_bytes_are_pinned() -> None:
    # Serialized exactly as finalize_tile writes it (compact separators).
    assert json.dumps(_one_tree_city(), separators=(",", ":")) == EXPECTED


def test_cityjson_quantization_is_millimetre_offset_from_bbox_min() -> None:
    # Redundant with the golden string, but names the two properties that
    # matter if the fixture is ever regenerated: millimetre scale and a
    # translate equal to the real-world bbox minimum.
    city = _one_tree_city()
    assert city["transform"]["scale"] == [0.001, 0.001, 0.001]
    assert city["transform"]["translate"] == city["metadata"]["geographicalExtent"][:3]
