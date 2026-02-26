#!/usr/bin/env python3
"""
glTF to AAMF Converter for UnnamedHB1 PlayStation 1 Animated Model Format

This script converts glTF 2.0 files (as produced by aamf2gltf.py) back to
AAMF (Animated AMF) binary format used by the UnnamedHB1 PS1 homebrew game.

AAMF format specification derived from:
https://github.com/achrostel/UnnamedHB1

Usage:
    python gltf2aamf.py input.gltf [output.aamf]

Dependencies:
    None (uses only Python standard library)

Binary Layout Reference (from anim_model.h / anim_model.c / model.h / model.c):
============================================================================

AAMF File Layout:
  [0..1]   uint16  boneamount
  [2..3]   uint16  animamount
  [4..]    bone parent table: boneamount * (uint16 bone_index, uint16 parent_index)
  Then for each bone:
    uint32  block_size (includes this 4-byte field)
    <AMF data of (block_size - 4) bytes>
  Then for each animation:
    uint32  block_size (includes this 4-byte field)
    <Animation data>

AMF Data Layout (per bone, embedded in AAMF):
  AMF_HEADER (8 + 16 = 24 bytes):
    uint32  used_textures
    uint16  x (chunk count X)
    uint16  z (chunk count Z)
    WorldBounds (16 bytes): int32 minX, minZ, maxX, maxZ
  Texture names: used_textures * 8 bytes (char[8] each)
  Chunk table: (x * z) * 4 bytes (uint32 entries, reinterpreted at runtime)
  Chunk data for each chunk:
    AMF_CHUNK header: 8 * uint16 (poly counts) + 8 * uint32 (pointers, zeroed in file)
    Polygon data in order: PF4, PG4, PFT4, PGT4, PF3, PG3, PFT3, PGT3

PG4 struct (100 bytes, from model.h):
    SVECTOR v0;           // 8 bytes
    SVECTOR v1;           // 8 bytes
    SVECTOR v2;           // 8 bytes
    SVECTOR v3;           // 8 bytes  (total vertices: 32)
    SVECTOR n0;           // 8 bytes
    SVECTOR n1;           // 8 bytes
    SVECTOR n2;           // 8 bytes
    SVECTOR n3;           // 8 bytes  (total normals: 32)
    POLY_G4 pol;          // 36 bytes (4 tag + 32 body)

POLY_G4 struct (from PSn00bSDK psxgpu.h):
    uint32_t tag;         // 4 bytes (display list link, zeroed in file)
    uint8_t  r0, g0, b0, code;  // 4 bytes (code = 0x38)
    int16_t  x0, y0;     // 4 bytes (screen coords, zeroed in file)
    uint8_t  r1, g1, b1, pad0;  // 4 bytes
    int16_t  x1, y1;     // 4 bytes
    uint8_t  r2, g2, b2, pad1;  // 4 bytes
    int16_t  x2, y2;     // 4 bytes
    uint8_t  r3, g3, b3, pad2;  // 4 bytes
    int16_t  x3, y3;     // 4 bytes (total: 36 bytes)

PS1 Quad Winding (Z-pattern):
    v0 ---- v1
    |  \\    |       v0-v1-v2-v3 stored in struct
    |   \\   |       Triangulated by aamf2gltf as:
    |    \\  |         tri0: v0, v2, v1  (CCW)
    v2 ---- v3         tri1: v1, v2, v3  (CCW)

    Reverse: given tri0=[A,B,C] tri1=[D,E,F]:
      v0=A, v2=B, v1=C, v3=F  (since tri0.v0=v0, tri0.v1=v2, tri0.v2=v1
                                       tri1.v0=v1, tri1.v1=v2, tri1.v2=v3)

Animation Layout:
  char    name[8]
  uint32  keyframeamount
  uint32  keyframe_ptr (unused in file, set at runtime = 0)
  Keyframe data: boneamount * keyframeamount * 32 bytes (MATRIX per keyframe)
    Organized as: all keyframes for bone 0, then bone 1, etc.

PS1 GTE MATRIX (32 bytes):
  int16 m[3][3]  (9 * 2 = 18 bytes)
  int16 pad      (2 bytes)
  int32 t[3]     (3 * 4 = 12 bytes)
"""

import struct
import sys
import os
import json
import base64
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from amf_types import SVECTOR_SIZE, CVECTOR_SIZE, FIXED_POINT_SCALE, PG4_SIZE


# ============================================================================
# Constants
# ============================================================================

# Struct sizes matching the C code exactly
MATRIX_SIZE = 32   # 3x3 int16 rotation + pad + 3 int32 translation

# Polygon struct sizes (including the POLY_* with tag)
PF3_SIZE = 52
PF4_SIZE = 64
PFT3_SIZE = 68
PFT4_SIZE = 84
PG3_SIZE = 76
PGT3_SIZE = 92
PGT4_SIZE = 136


# ============================================================================
# Helper Functions
# ============================================================================

def float_to_fixed(val: float, scale: float = FIXED_POINT_SCALE) -> int:
    """Convert floating point to PS1 4.12 fixed-point int16"""
    result = int(round(val * scale))
    return max(-32768, min(32767, result))


def float_to_fixed32(val: float, scale: float = FIXED_POINT_SCALE) -> int:
    """Convert floating point to PS1 fixed-point int32 (for translation)"""
    result = int(round(val * scale))
    return max(-2147483648, min(2147483647, result))


def pack_svector(vx: int, vy: int, vz: int, pad: int = 0) -> bytes:
    """Pack an SVECTOR (4 * int16 = 8 bytes)"""
    return struct.pack('<hhhh', vx, vy, vz, pad)


def pack_svector_from_float(x: float, y: float, z: float, pad: int = 0) -> bytes:
    """Pack an SVECTOR from float values"""
    return pack_svector(float_to_fixed(x), float_to_fixed(y), float_to_fixed(z), pad)


def pack_matrix(rotation: List[List[float]], translation: Tuple[float, float, float]) -> bytes:
    """
    Pack a PS1 GTE MATRIX (32 bytes).
    rotation: 3x3 float matrix
    translation: (tx, ty, tz) in float
    """
    data = bytearray()
    # 3x3 rotation matrix as int16 (4.12 fixed-point)
    for row in range(3):
        for col in range(3):
            val = float_to_fixed(rotation[row][col])
            data += struct.pack('<h', val)
    # 2 bytes padding after the 9 int16s (18 bytes -> pad to 20 before translation)
    data += struct.pack('<h', 0)
    # Translation as 3 int32
    for i in range(3):
        val = float_to_fixed32(translation[i])
        data += struct.pack('<i', val)
    assert len(data) == 32
    return bytes(data)


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> List[List[float]]:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix"""
    length = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if length > 0:
        qx /= length
        qy /= length
        qz /= length
        qw /= length

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    m = [
        [1.0 - 2.0*(yy + zz), 2.0*(xy - wz),       2.0*(xz + wy)],
        [2.0*(xy + wz),       1.0 - 2.0*(xx + zz),  2.0*(yz - wx)],
        [2.0*(xz - wy),       2.0*(yz + wx),         1.0 - 2.0*(xx + yy)]
    ]
    return m


# ============================================================================
# glTF Reader
# ============================================================================

class GltfReader:
    """Reads and parses a glTF 2.0 file"""

    def __init__(self, filename: str):
        self.filename = filename
        self.gltf: Dict[str, Any] = {}
        self.buffer_data: bytes = b''

    def load(self):
        """Load and parse the glTF file"""
        with open(self.filename, 'r') as f:
            self.gltf = json.load(f)

        # Load the binary buffer (base64 data URI or external file)
        if self.gltf.get('buffers'):
            uri = self.gltf['buffers'][0]['uri']
            if uri.startswith('data:'):
                _, encoded = uri.split(',', 1)
                self.buffer_data = base64.b64decode(encoded)
            else:
                buf_path = os.path.join(os.path.dirname(self.filename), uri)
                with open(buf_path, 'rb') as f:
                    self.buffer_data = f.read()

        print(f"Loaded glTF: {self.filename}")
        print(f"  Buffer size: {len(self.buffer_data)} bytes")
        print(f"  Nodes: {len(self.gltf.get('nodes', []))}")
        print(f"  Meshes: {len(self.gltf.get('meshes', []))}")
        print(f"  Skins: {len(self.gltf.get('skins', []))}")
        print(f"  Animations: {len(self.gltf.get('animations', []))}")

    def read_accessor(self, accessor_idx: int) -> List:
        """Read data from an accessor, returning a flat list of values"""
        accessor = self.gltf['accessors'][accessor_idx]
        buffer_view = self.gltf['bufferViews'][accessor['bufferView']]

        byte_offset = buffer_view['byteOffset'] + accessor.get('byteOffset', 0)
        count = accessor['count']
        comp_type = accessor['componentType']
        acc_type = accessor['type']

        comp_sizes = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
        comp_formats = {5120: 'b', 5121: 'B', 5122: 'h', 5123: 'H', 5125: 'I', 5126: 'f'}
        type_counts = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}

        comp_count = type_counts[acc_type]
        fmt_char = comp_formats[comp_type]
        comp_size = comp_sizes[comp_type]

        stride = buffer_view.get('byteStride', comp_count * comp_size)

        values = []
        for i in range(count):
            offset = byte_offset + i * stride
            for j in range(comp_count):
                val = struct.unpack_from(f'<{fmt_char}', self.buffer_data, offset + j * comp_size)[0]
                values.append(val)

        return values

    def get_bone_hierarchy(self) -> List[Tuple[int, int]]:
        """
        Extract bone hierarchy from the glTF skin/nodes.
        Returns list of (bone_index, parent_index) tuples.
        """
        if not self.gltf.get('skins'):
            return []

        skin = self.gltf['skins'][0]
        joints = skin['joints']
        nodes = self.gltf['nodes']

        node_to_bone = {}
        for bone_idx, node_idx in enumerate(joints):
            node_to_bone[node_idx] = bone_idx

        hierarchy = []
        for bone_idx, node_idx in enumerate(joints):
            parent_bone = bone_idx  # Default: self-parent (root)
            for potential_parent_node_idx, potential_parent_node in enumerate(nodes):
                if 'children' in potential_parent_node:
                    if node_idx in potential_parent_node['children']:
                        if potential_parent_node_idx in node_to_bone:
                            parent_bone = node_to_bone[potential_parent_node_idx]
                        break
            hierarchy.append((bone_idx, parent_bone))

        return hierarchy

    def get_triangles_per_bone(self) -> Tuple[Dict, Dict]:
        """
        Extract triangles grouped by bone assignment.
        Returns two dicts:
          bone_idx -> list of (v0, v1, v2) position tuples per triangle
          bone_idx -> list of (n0, n1, n2) normal tuples per triangle
        Each vertex/normal is a (x, y, z) float tuple.
        """
        if not self.gltf.get('meshes'):
            return {}, {}

        mesh = self.gltf['meshes'][0]
        prim = mesh['primitives'][0]

        positions = self.read_accessor(prim['attributes']['POSITION'])
        joints = self.read_accessor(prim['attributes']['JOINTS_0'])
        indices = self.read_accessor(prim['indices'])

        normals = []
        if 'NORMAL' in prim['attributes']:
            normals = self.read_accessor(prim['attributes']['NORMAL'])

        bone_tris: Dict[int, List] = {}
        bone_tri_normals: Dict[int, List] = {}

        for i in range(0, len(indices), 3):
            idx0 = indices[i]
            idx1 = indices[i + 1]
            idx2 = indices[i + 2]

            bone_idx = joints[idx0 * 4]

            if bone_idx not in bone_tris:
                bone_tris[bone_idx] = []
                bone_tri_normals[bone_idx] = []

            tri_verts = []
            tri_norms = []
            for idx in [idx0, idx1, idx2]:
                px = positions[idx * 3]
                py = positions[idx * 3 + 1]
                pz = positions[idx * 3 + 2]
                tri_verts.append((px, py, pz))

                if normals:
                    nx = normals[idx * 3]
                    ny = normals[idx * 3 + 1]
                    nz = normals[idx * 3 + 2]
                    tri_norms.append((nx, ny, nz))
                else:
                    tri_norms.append((0.0, 1.0, 0.0))

            bone_tris[bone_idx].append(tri_verts)
            bone_tri_normals[bone_idx].append(tri_norms)

        return bone_tris, bone_tri_normals

    def get_animations(self) -> List[Dict]:
        """
        Extract animation data.
        Returns list of animation dicts with name, keyframe_count, and per-bone transforms.
        """
        if not self.gltf.get('animations'):
            return []

        skin = self.gltf['skins'][0]
        joints = skin['joints']
        bone_count = len(joints)

        node_to_bone = {}
        for bone_idx, node_idx in enumerate(joints):
            node_to_bone[node_idx] = bone_idx

        animations = []
        for anim in self.gltf['animations']:
            name = anim.get('name', 'anim')

            bone_translations = {}
            bone_rotations = {}
            keyframe_count = 0

            for channel in anim['channels']:
                target_node = channel['target']['node']
                path = channel['target']['path']
                sampler = anim['samplers'][channel['sampler']]

                if target_node not in node_to_bone:
                    continue

                bone_idx = node_to_bone[target_node]
                output_data = self.read_accessor(sampler['output'])

                if path == 'translation':
                    translations = []
                    for j in range(0, len(output_data), 3):
                        translations.append((output_data[j], output_data[j+1], output_data[j+2]))
                    bone_translations[bone_idx] = translations
                    keyframe_count = max(keyframe_count, len(translations))

                elif path == 'rotation':
                    rotations = []
                    for j in range(0, len(output_data), 4):
                        rotations.append((output_data[j], output_data[j+1],
                                         output_data[j+2], output_data[j+3]))
                    bone_rotations[bone_idx] = rotations
                    keyframe_count = max(keyframe_count, len(rotations))

            animations.append({
                'name': name,
                'keyframe_count': keyframe_count,
                'translations': bone_translations,
                'rotations': bone_rotations,
                'bone_count': bone_count
            })

        return animations


# ============================================================================
# Triangle-pair to Quad merging
# ============================================================================

def try_merge_triangle_pair(tri0_verts, tri0_norms, tri1_verts, tri1_norms):
    """
    Try to merge two triangles back into a PS1 quad (PG4).

    The aamf2gltf.py converter splits PS1 quads as:
        PS1 Z-pattern: v0--v1
                        |    |
                        v2--v3

        tri0: [v0, v2, v1]   (indices idx0, idx2, idx1)
        tri1: [v1, v2, v3]   (indices idx1, idx2, idx3)

    So the shared edge is v1-v2 (tri0's vertices [2] and [1],
    tri1's vertices [0] and [1]).

    We check if tri0[2]==tri1[0] and tri0[1]==tri1[1] (the shared edge).
    If so, reconstruct:
        v0 = tri0[0], v1 = tri0[2] (=tri1[0]), v2 = tri0[1] (=tri1[1]), v3 = tri1[2]

    Returns (quad_verts, quad_norms) or None if can't merge.
    """
    # Check shared edge: tri0[2] should equal tri1[0], tri0[1] should equal tri1[1]
    eps = 1e-7

    def verts_equal(a, b):
        return all(abs(a[i] - b[i]) < eps for i in range(3))

    if verts_equal(tri0_verts[2], tri1_verts[0]) and verts_equal(tri0_verts[1], tri1_verts[1]):
        # Perfect match with the expected aamf2gltf winding
        quad_verts = (tri0_verts[0], tri0_verts[2], tri0_verts[1], tri1_verts[2])
        quad_norms = (tri0_norms[0], tri0_norms[2], tri0_norms[1], tri1_norms[2])
        return quad_verts, quad_norms

    return None


def merge_triangles_to_quads(tris, tri_normals):
    """
    Given a list of triangles (from glTF), merge consecutive pairs back into
    PS1 quads where possible. The aamf2gltf converter always emits pairs of
    triangles from each quad, so consecutive pairs should merge cleanly.

    Returns:
        quads: list of (v0, v1, v2, v3) vertex tuples
        quad_normals: list of (n0, n1, n2, n3) normal tuples
        remaining_tris: list of triangles that couldn't be merged
        remaining_tri_normals: corresponding normals
    """
    quads = []
    quad_normals = []
    remaining_tris = []
    remaining_tri_normals = []

    i = 0
    while i < len(tris):
        merged = False
        if i + 1 < len(tris):
            result = try_merge_triangle_pair(
                tris[i], tri_normals[i],
                tris[i + 1], tri_normals[i + 1]
            )
            if result is not None:
                q_verts, q_norms = result
                quads.append(q_verts)
                quad_normals.append(q_norms)
                i += 2
                merged = True

        if not merged:
            remaining_tris.append(tris[i])
            remaining_tri_normals.append(tri_normals[i])
            i += 1

    return quads, quad_normals, remaining_tris, remaining_tri_normals


# ============================================================================
# AMF Binary Writer (per-bone mesh data)
# ============================================================================

class AMFWriter:
    """Writes AMF binary data for a single bone's mesh"""

    def write_bone_amf(self, tris: List, tri_normals: List) -> bytes:
        """
        Write AMF data for a single bone.

        Merges triangle pairs back into PG4 quads (gouraud-shaded quads with
        per-vertex normals), matching the original AAMF format.
        Any leftover odd triangles are written as PG3.

        AMF layout:
          AMF_HEADER (24 bytes)
          Texture names (0 bytes if no textures)
          Chunk table (chunk_count * 4 bytes)
          Chunk data (header + polygon arrays)
        """
        # Merge triangle pairs back into quads
        quads, quad_normals, leftover_tris, leftover_tri_normals = \
            merge_triangles_to_quads(tris, tri_normals)

        data = bytearray()

        # Collect all vertices for bounds calculation
        all_verts = []
        for q in quads:
            all_verts.extend(q)
        for t in leftover_tris:
            all_verts.extend(t)

        # AMF_HEADER
        used_textures = 0
        chunk_x = 1
        chunk_z = 1

        if all_verts:
            min_x = min(v[0] for v in all_verts)
            min_z = min(v[2] for v in all_verts)
            max_x = max(v[0] for v in all_verts)
            max_z = max(v[2] for v in all_verts)
        else:
            min_x = min_z = max_x = max_z = 0.0

        data += struct.pack('<I', used_textures)
        data += struct.pack('<HH', chunk_x, chunk_z)
        data += struct.pack('<iiii',
                            float_to_fixed32(min_x),
                            float_to_fixed32(min_z),
                            float_to_fixed32(max_x),
                            float_to_fixed32(max_z))

        # No texture names

        # Chunk table: 1 entry
        chunk_count = chunk_x * chunk_z
        for _ in range(chunk_count):
            data += struct.pack('<I', 0)

        # AMF_CHUNK header
        # Order in chunk: F4, G4, FT4, GT4, F3, G3, FT3, GT3
        f4_amount = 0
        g4_amount = len(quads)
        ft4_amount = 0
        gt4_amount = 0
        f3_amount = 0
        g3_amount = len(leftover_tris)
        ft3_amount = 0
        gt3_amount = 0

        # Pack counts (8 * uint16 = 16 bytes)
        data += struct.pack('<HHHHHHHH',
                            f4_amount, g4_amount, ft4_amount, gt4_amount,
                            f3_amount, g3_amount, ft3_amount, gt3_amount)
        # Pack 8 pointers (zeroed, set at runtime) (8 * uint32 = 32 bytes)
        for _ in range(8):
            data += struct.pack('<I', 0)

        # --- Write PG4 polygon data ---
        # PG4 struct (100 bytes):
        #   SVECTOR v0, v1, v2, v3   (4 * 8 = 32 bytes)
        #   SVECTOR n0, n1, n2, n3   (4 * 8 = 32 bytes)
        #   POLY_G4 pol              (4 + 32 = 36 bytes)
        for quad_idx in range(len(quads)):
            v0, v1, v2, v3 = quads[quad_idx]
            n0, n1, n2, n3 = quad_normals[quad_idx]

            # 4 vertices as SVECTORs
            data += pack_svector_from_float(v0[0], v0[1], v0[2])
            data += pack_svector_from_float(v1[0], v1[1], v1[2])
            data += pack_svector_from_float(v2[0], v2[1], v2[2])
            data += pack_svector_from_float(v3[0], v3[1], v3[2])

            # 4 normals as SVECTORs
            data += pack_svector_from_float(n0[0], n0[1], n0[2])
            data += pack_svector_from_float(n1[0], n1[1], n1[2])
            data += pack_svector_from_float(n2[0], n2[1], n2[2])
            data += pack_svector_from_float(n3[0], n3[1], n3[2])

            # POLY_G4 (36 bytes)
            # Tag (4 bytes, zeroed in file)
            data += struct.pack('<I', 0)
            # Body (32 bytes):
            #   r0,g0,b0,code  x0,y0  r1,g1,b1,pad0  x1,y1
            #   r2,g2,b2,pad1  x2,y2  r3,g3,b3,pad2  x3,y3
            r, g, b = 128, 128, 128
            code = 0x38  # POLY_G4 code from setPolyG4 macro
            data += struct.pack('<BBBB', r, g, b, code)   # r0,g0,b0,code
            data += struct.pack('<hh', 0, 0)               # x0, y0
            data += struct.pack('<BBBB', r, g, b, 0)       # r1,g1,b1,pad0
            data += struct.pack('<hh', 0, 0)               # x1, y1
            data += struct.pack('<BBBB', r, g, b, 0)       # r2,g2,b2,pad1
            data += struct.pack('<hh', 0, 0)               # x2, y2
            data += struct.pack('<BBBB', r, g, b, 0)       # r3,g3,b3,pad2
            data += struct.pack('<hh', 0, 0)               # x3, y3

        # --- Write PG3 polygon data (leftover odd triangles) ---
        # PG3 struct (76 bytes):
        #   SVECTOR v0, v1, v2       (3 * 8 = 24 bytes)
        #   SVECTOR n0, n1, n2       (3 * 8 = 24 bytes)
        #   POLY_G3 pol              (4 + 24 = 28 bytes)
        for tri_idx in range(len(leftover_tris)):
            t0, t1, t2 = leftover_tris[tri_idx]
            tn0, tn1, tn2 = leftover_tri_normals[tri_idx]

            # 3 vertices
            data += pack_svector_from_float(t0[0], t0[1], t0[2])
            data += pack_svector_from_float(t1[0], t1[1], t1[2])
            data += pack_svector_from_float(t2[0], t2[1], t2[2])

            # 3 normals
            data += pack_svector_from_float(tn0[0], tn0[1], tn0[2])
            data += pack_svector_from_float(tn1[0], tn1[1], tn1[2])
            data += pack_svector_from_float(tn2[0], tn2[1], tn2[2])

            # POLY_G3 (28 bytes)
            # Tag (4 bytes)
            data += struct.pack('<I', 0)
            # Body (24 bytes):
            #   r0,g0,b0,code  x0,y0  r1,g1,b1,pad0  x1,y1  r2,g2,b2,pad1  x2,y2
            r, g, b = 128, 128, 128
            code = 0x30  # POLY_G3 code from setPolyG3 macro
            data += struct.pack('<BBBB', r, g, b, code)
            data += struct.pack('<hh', 0, 0)
            data += struct.pack('<BBBB', r, g, b, 0)
            data += struct.pack('<hh', 0, 0)
            data += struct.pack('<BBBB', r, g, b, 0)
            data += struct.pack('<hh', 0, 0)

        return bytes(data)

    def write_empty_bone_amf(self) -> bytes:
        """Write minimal AMF data for a bone with no geometry"""
        data = bytearray()

        # AMF_HEADER
        data += struct.pack('<I', 0)       # used_textures
        data += struct.pack('<HH', 1, 1)   # x=1, z=1
        data += struct.pack('<iiii', 0, 0, 0, 0)  # WorldBounds

        # Chunk table (1 entry)
        data += struct.pack('<I', 0)

        # Chunk header (all zeros)
        data += struct.pack('<HHHHHHHH', 0, 0, 0, 0, 0, 0, 0, 0)
        # 8 pointers
        for _ in range(8):
            data += struct.pack('<I', 0)

        return bytes(data)


# ============================================================================
# AAMF Binary Writer
# ============================================================================

class AAMFWriter:
    """Writes AAMF binary files"""

    def write(self, bone_hierarchy: List[Tuple[int, int]],
              bone_amf_blocks: List[bytes],
              animations: List[Dict],
              bone_count: int) -> bytes:
        """
        Write complete AAMF binary data.

        Layout:
          uint16 boneamount
          uint16 animamount
          bone_parent_table: boneamount * (uint16, uint16)
          For each bone: uint32 block_size + AMF data
          For each animation: uint32 block_size + animation data
        """
        data = bytearray()

        anim_count = len(animations)

        # Header
        data += struct.pack('<H', bone_count)
        data += struct.pack('<H', anim_count)

        # Bone parent table
        for bone_idx, parent_idx in bone_hierarchy:
            data += struct.pack('<HH', bone_idx, parent_idx)

        # Bone AMF data blocks
        for amf_data in bone_amf_blocks:
            block_size = len(amf_data) + 4  # +4 for the block_size field itself
            data += struct.pack('<I', block_size)
            data += amf_data

        # Animation blocks
        for anim_info in animations:
            anim_data = self._write_animation(anim_info, bone_count)
            block_size = len(anim_data) + 4
            data += struct.pack('<I', block_size)
            data += anim_data

        return bytes(data)

    def _write_animation(self, anim_info: Dict, bone_count: int) -> bytes:
        """
        Write a single animation block.

        Layout:
          char name[8]
          uint32 keyframeamount
          uint32 keyframe_ptr (0 in file, set at runtime)
          Keyframe data: bone_count * keyframeamount * MATRIX(32 bytes)
            Organized as: all keyframes for bone 0, then bone 1, etc.
        """
        data = bytearray()

        name = anim_info['name']
        keyframe_count = anim_info['keyframe_count']
        translations = anim_info['translations']
        rotations = anim_info['rotations']

        # Name (8 bytes, padded with nulls)
        name_bytes = name.encode('ascii', errors='replace')[:8]
        name_bytes = name_bytes.ljust(8, b'\x00')
        data += name_bytes

        # Keyframe count
        data += struct.pack('<I', keyframe_count)

        # Keyframe pointer (0, set at runtime)
        data += struct.pack('<I', 0)

        # Keyframe matrices: [bone][keyframe]
        for bone_idx in range(bone_count):
            for kf_idx in range(keyframe_count):
                if bone_idx in translations and kf_idx < len(translations[bone_idx]):
                    tx, ty, tz = translations[bone_idx][kf_idx]
                else:
                    tx, ty, tz = 0.0, 0.0, 0.0

                if bone_idx in rotations and kf_idx < len(rotations[bone_idx]):
                    qx, qy, qz, qw = rotations[bone_idx][kf_idx]
                else:
                    qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

                rot_matrix = quaternion_to_rotation_matrix(qx, qy, qz, qw)
                data += pack_matrix(rot_matrix, (tx, ty, tz))

        return bytes(data)


# ============================================================================
# Main Conversion Logic
# ============================================================================

def convert_gltf_to_aamf(input_file: str, output_file: str):
    """Convert a glTF file to AAMF format"""

    reader = GltfReader(input_file)
    reader.load()

    # Extract bone hierarchy
    hierarchy = reader.get_bone_hierarchy()
    if not hierarchy:
        print("Error: No skeleton/skin found in glTF file!")
        sys.exit(1)

    bone_count = len(hierarchy)
    print(f"\nBone hierarchy ({bone_count} bones):")
    for bone_idx, parent_idx in hierarchy:
        print(f"  Bone {bone_idx}: parent = {parent_idx}")

    # Extract per-bone triangles
    bone_tris, bone_tri_normals = reader.get_triangles_per_bone()

    # Build AMF data for each bone
    amf_writer = AMFWriter()
    bone_amf_blocks = []

    total_quads = 0
    total_leftover_tris = 0

    for bone_idx in range(bone_count):
        if bone_idx in bone_tris and bone_tris[bone_idx]:
            tris = bone_tris[bone_idx]
            norms = bone_tri_normals.get(bone_idx, [])

            amf_data = amf_writer.write_bone_amf(tris, norms)

            # Count for reporting
            quads, _, leftovers, _ = merge_triangles_to_quads(tris, norms)
            n_quads = len(quads)
            n_leftovers = len(leftovers)
            total_quads += n_quads
            total_leftover_tris += n_leftovers

            print(f"  Bone {bone_idx}: {len(tris)} triangles -> "
                  f"{n_quads} PG4 quads + {n_leftovers} PG3 tris, "
                  f"{len(amf_data)} bytes")
        else:
            amf_data = amf_writer.write_empty_bone_amf()
            print(f"  Bone {bone_idx}: empty (no geometry)")

        bone_amf_blocks.append(amf_data)

    # Extract animations
    animations = reader.get_animations()
    print(f"\nAnimations ({len(animations)}):")
    for anim in animations:
        print(f"  '{anim['name']}': {anim['keyframe_count']} keyframes, "
              f"{len(anim['translations'])} bone translations, "
              f"{len(anim['rotations'])} bone rotations")

    # Write AAMF
    aamf_writer = AAMFWriter()
    aamf_data = aamf_writer.write(hierarchy, bone_amf_blocks, animations, bone_count)

    with open(output_file, 'wb') as f:
        f.write(aamf_data)

    print(f"\nExported to {output_file}")
    print(f"  Total size: {len(aamf_data)} bytes")
    print(f"  Bones: {bone_count}")
    print(f"  PG4 quads: {total_quads}")
    print(f"  PG3 leftover triangles: {total_leftover_tris}")
    print(f"  Animations: {len(animations)}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.gltf [output.aamf]")
        print("\nConverts glTF 2.0 files to AAMF format for UnnamedHB1 PS1 homebrew.")
        print("\nThe input glTF should contain:")
        print("  - A skinned mesh with bone (joint) assignments")
        print("  - A skeleton hierarchy")
        print("  - Optionally, skeletal animations with translation/rotation channels")
        print("\nExpects glTF files as produced by aamf2gltf.py.")
        print("\nTriangle pairs are merged back into PG4 quads (gouraud-shaded quads)")
        print("matching the PS1 Z-pattern winding order.")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = os.path.splitext(input_file)[0] + ".aamf"

    print(f"Converting {input_file} -> {output_file}")
    convert_gltf_to_aamf(input_file, output_file)


if __name__ == "__main__":
    main()