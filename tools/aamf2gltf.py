#!/usr/bin/env python3
"""
AAMF to glTF Converter for UnnamedHB1 PlayStation 1 Animated Model Format

This script converts AAMF (Animated AMF) files from the UnnamedHB1
PS1 homebrew game to glTF 2.0 format with skeletal animation support. 

AAMF format specification derived from:
https://github.com/achrostel/UnnamedHB1

Usage:
    python aamf2gltf.py input.aamf [output.gltf]

Dependencies:
    pip install pygltflib (optional, uses built-in JSON export)
"""

import struct
import sys
import os
import json
import base64
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


# ============================================================================
# Data Structures from PSn00bSDK and UnnamedHB1
# ============================================================================

@dataclass
class SVECTOR:
    """Short vector (8 bytes) - used for vertices and normals"""
    vx: int  # int16_t
    vy: int  # int16_t
    vz:  int  # int16_t
    pad: int  # int16_t

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'SVECTOR': 
        vx, vy, vz, pad = struct. unpack_from('<hhhh', data, offset)
        return cls(vx, vy, vz, pad)

    def to_float(self, scale: float = 1.0 / 4096.0) -> Tuple[float, float, float]:
        """Convert fixed-point to floating point coordinates"""
        return (self.vx * scale, self. vy * scale, self.vz * scale)


@dataclass
class CVECTOR:
    """Color vector (4 bytes)"""
    r: int  # uint8_t
    g: int  # uint8_t
    b: int  # uint8_t
    cd: int  # uint8_t

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'CVECTOR': 
        r, g, b, cd = struct.unpack_from('<BBBB', data, offset)
        return cls(r, g, b, cd)

    def to_float(self) -> Tuple[float, float, float, float]:
        """Convert to normalized float RGBA"""
        return (self.r / 255.0, self. g / 255.0, self.b / 255.0, 1.0)


@dataclass
class MATRIX:
    """
    PS1 GTE Matrix (32 bytes)
    - m[3][3]:  3x3 rotation matrix (int16_t, 4.12 fixed-point)
    - t[3]: translation vector (int32_t)
    """
    m:  List[List[int]]  # 3x3 rotation matrix
    t: List[int]  # Translation vector

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'MATRIX':
        # Read 3x3 rotation matrix (9 int16_t = 18 bytes)
        m = []
        for row in range(3):
            row_vals = []
            for col in range(3):
                val = struct.unpack_from('<h', data, offset + (row * 3 + col) * 2)[0]
                row_vals. append(val)
            m.append(row_vals)

        # Read translation vector (3 int32_t = 12 bytes)
        t = list(struct.unpack_from('<iii', data, offset + 18))

        return cls(m, t)

    def to_rotation_matrix(self) -> List[List[float]]:
        """Convert fixed-point rotation to float matrix"""
        scale = 1.0 / 4096.0  # 4. 12 fixed-point
        return [[self.m[row][col] * scale for col in range(3)] for row in range(3)]

    def to_translation(self, scale: float = 1.0 / 4096.0) -> Tuple[float, float, float]:
        """Convert translation to float"""
        return (self.t[0] * scale, self. t[1] * scale, self.t[2] * scale)

    def to_quaternion(self) -> Tuple[float, float, float, float]:
        """Convert rotation matrix to quaternion (x, y, z, w)"""
        m = self.to_rotation_matrix()

        # Calculate quaternion from rotation matrix
        trace = m[0][0] + m[1][1] + m[2][2]

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2][1] - m[1][2]) * s
            y = (m[0][2] - m[2][0]) * s
            z = (m[1][0] - m[0][1]) * s
        elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
            w = (m[2][1] - m[1][2]) / s
            x = 0.25 * s
            y = (m[0][1] + m[1][0]) / s
            z = (m[0][2] + m[2][0]) / s
        elif m[1][1] > m[2][2]:
            s = 2.0 * math. sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
            w = (m[1][0] - m[0][1]) / s
            x = (m[0][2] + m[2][0]) / s
            y = (m[1][2] + m[2][1]) / s
            z = 0.25 * s

        # Normalize quaternion
        length = math.sqrt(x*x + y*y + z*z + w*w)
        if length > 0:
            x /= length
            y /= length
            z /= length
            w /= length

        return (x, y, z, w)


@dataclass
class Keyframe:
    """Animation keyframe containing a transformation matrix"""
    mat: MATRIX

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'Keyframe':
        mat = MATRIX.from_bytes(data, offset)
        return cls(mat)


@dataclass
class Animation:
    """Animation data"""
    name: str
    keyframe_count: int
    keyframes: List[List[Keyframe]]  # [keyframe_idx][bone_idx]

    @classmethod
    def from_bytes(cls, data: bytes, offset:  int, bone_count: int) -> 'Animation': 
        # Read animation header
        name_bytes = data[offset: offset + 8]
        name = name_bytes.rstrip(b'\x00').decode('ascii', errors='replace')
        keyframe_count = struct.unpack_from('<I', data, offset + 8)[0]

        # Skip the keyframe pointer (4 bytes) - set at runtime
        keyframe_data_offset = offset + 16

        # Read keyframes - organized as [bone][keyframe] in file,
        # but we'll reorganize for easier animation access
        keyframes = []
        for kf_idx in range(keyframe_count):
            frame_bones = []
            for bone_idx in range(bone_count):
                # Keyframes are stored:  keyframe + bone_idx * keyframe_count
                kf_offset = keyframe_data_offset + (kf_idx + bone_idx * keyframe_count) * 32
                kf = Keyframe.from_bytes(data, kf_offset)
                frame_bones. append(kf)
            keyframes.append(frame_bones)

        return cls(name, keyframe_count, keyframes)


@dataclass
class WorldBounds:
    """World bounding box (16 bytes)"""
    minX: int
    minZ: int
    maxX: int
    maxZ: int

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'WorldBounds':
        minX, minZ, maxX, maxZ = struct.unpack_from('<iiii', data, offset)
        return cls(minX, minZ, maxX, maxZ)


@dataclass
class AMFHeader:
    """AMF file header (24 bytes)"""
    used_textures: int
    x:  int
    z: int
    bounds: WorldBounds

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'AMFHeader':
        used_textures, x, z = struct.unpack_from('<IHH', data, offset)
        bounds = WorldBounds.from_bytes(data, offset + 8)
        return cls(used_textures & 0x7FFFFFFF, x, z, bounds)


@dataclass
class ChunkHeader:
    """Chunk header with polygon counts"""
    F4_amount: int
    G4_amount: int
    FT4_amount: int
    GT4_amount: int
    F3_amount: int
    G3_amount: int
    FT3_amount:  int
    GT3_amount: int

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'ChunkHeader':
        counts = struct.unpack_from('<HHHHHHHH', data, offset)
        return cls(*counts)


# Structure sizes
SVECTOR_SIZE = 8
CVECTOR_SIZE = 4
POINTER_SIZE = 4
MATRIX_SIZE = 32

# Polygon structure sizes (matching AMF format)
POLY_F3_SIZE = 20
POLY_F4_SIZE = 24
POLY_FT3_SIZE = 32
POLY_FT4_SIZE = 40
POLY_G3_SIZE = 28
POLY_G4_SIZE = 36
POLY_GT3_SIZE = 40
POLY_GT4_SIZE = 52

PF3_SIZE = 3 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POLY_F3_SIZE
PF4_SIZE = 4 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POLY_F4_SIZE
PFT3_SIZE = 3 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POINTER_SIZE + POLY_FT3_SIZE
PFT4_SIZE = 4 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POINTER_SIZE + POLY_FT4_SIZE
PG3_SIZE = 3 * SVECTOR_SIZE + 3 * SVECTOR_SIZE + POLY_G3_SIZE
PG4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + POLY_G4_SIZE
PGT3_SIZE = 3 * SVECTOR_SIZE + 3 * SVECTOR_SIZE + POINTER_SIZE + POLY_GT3_SIZE
PGT4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + 4 * CVECTOR_SIZE + POINTER_SIZE + POLY_GT4_SIZE


@dataclass
class Bone:
    """Bone with embedded AMF model data"""
    index: int
    parent_index: int
    vertices: List[Tuple[float, float, float]]
    normals: List[Tuple[float, float, float]]
    colors: List[Tuple[float, float, float, float]]
    faces: List[List[int]]
    face_normals: List[List[int]]
    face_colors: List[List[int]]


class AMFParser:
    """Parser for embedded AMF model data within AAMF bones"""

    def __init__(self, data: bytes, offset: int = 0):
        self.data = data
        self.base_offset = offset
        self.header:  Optional[AMFHeader] = None
        self.texture_names: List[str] = []
        self.vertices: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.colors: List[Tuple[float, float, float, float]] = []
        self. faces: List[List[int]] = []
        self.face_normals: List[List[int]] = []
        self. face_colors: List[List[int]] = []

    def parse(self):
        """Parse the embedded AMF data"""
        offset = self.base_offset

        # Parse header
        self.header = AMFHeader.from_bytes(self. data, offset)

        # Parse texture names
        texture_offset = offset + 8 + 16
        for i in range(self.header.used_textures):
            name_bytes = self. data[texture_offset:texture_offset + 8]
            name = name_bytes. rstrip(b'\x00').decode('ascii', errors='replace')
            self.texture_names.append(name)
            texture_offset += 8

        # Parse chunk table and chunks
        chunk_count = self.header. x * self.header.z
        chunk_data_offset = texture_offset + chunk_count * 4

        for chunk_idx in range(chunk_count):
            self._parse_chunk(chunk_idx, chunk_data_offset)
            chunk_header = ChunkHeader. from_bytes(self.data, chunk_data_offset)
            chunk_size = 16 + 32
            chunk_size += chunk_header. F4_amount * PF4_SIZE
            chunk_size += chunk_header.G4_amount * PG4_SIZE
            chunk_size += chunk_header.FT4_amount * PFT4_SIZE
            chunk_size += chunk_header.GT4_amount * PGT4_SIZE
            chunk_size += chunk_header.F3_amount * PF3_SIZE
            chunk_size += chunk_header.G3_amount * PG3_SIZE
            chunk_size += chunk_header.FT3_amount * PFT3_SIZE
            chunk_size += chunk_header.GT3_amount * PGT3_SIZE
            chunk_data_offset += chunk_size

    def _parse_chunk(self, chunk_idx:  int, offset: int):
        """Parse a single chunk"""
        chunk_header = ChunkHeader.from_bytes(self.data, offset)
        poly_offset = offset + 16 + 32

        poly_offset = self._parse_f4_polys(poly_offset, chunk_header. F4_amount)
        poly_offset = self._parse_g4_polys(poly_offset, chunk_header.G4_amount)
        poly_offset = self._parse_ft4_polys(poly_offset, chunk_header.FT4_amount)
        poly_offset = self._parse_gt4_polys(poly_offset, chunk_header.GT4_amount)
        poly_offset = self._parse_f3_polys(poly_offset, chunk_header.F3_amount)
        poly_offset = self._parse_g3_polys(poly_offset, chunk_header. G3_amount)
        poly_offset = self._parse_ft3_polys(poly_offset, chunk_header.FT3_amount)
        poly_offset = self._parse_gt3_polys(poly_offset, chunk_header.GT3_amount)

    def _add_vertex(self, sv: SVECTOR) -> int:
        self.vertices.append(sv.to_float())
        return len(self.vertices) - 1

    def _add_normal(self, sv: SVECTOR) -> int:
        self.normals. append(sv.to_float())
        return len(self. normals) - 1

    def _add_color(self, cv: CVECTOR) -> int:
        self.colors.append(cv. to_float())
        return len(self.colors) - 1

    def _parse_f4_polys(self, offset: int, count: int) -> int:
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR. from_bytes(self.data, offset + 24)
            n = SVECTOR.from_bytes(self.data, offset + 32)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx = self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            self.faces. append([idx0, idx1, idx2])
            self.faces.append([idx1, idx3, idx2])
            self.face_normals.append([n_idx, n_idx + 1, n_idx + 2])
            self.face_normals.append([n_idx + 1, n_idx + 3, n_idx + 2])

            offset += PF4_SIZE
        return offset

    def _parse_g4_polys(self, offset: int, count: int) -> int:
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR. from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n0 = SVECTOR.from_bytes(self.data, offset + 32)
            n1 = SVECTOR.from_bytes(self. data, offset + 40)
            n2 = SVECTOR.from_bytes(self.data, offset + 48)
            n3 = SVECTOR. from_bytes(self.data, offset + 56)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx0 = self._add_normal(n0)
            n_idx1 = self._add_normal(n1)
            n_idx2 = self._add_normal(n2)
            n_idx3 = self._add_normal(n3)

            self.faces. append([idx0, idx1, idx2])
            self.faces.append([idx1, idx3, idx2])
            self.face_normals.append([n_idx0, n_idx1, n_idx2])
            self.face_normals. append([n_idx1, n_idx3, n_idx2])

            offset += PG4_SIZE
        return offset

    def _parse_ft4_polys(self, offset: int, count: int) -> int:
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n = SVECTOR.from_bytes(self. data, offset + 32)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx = self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            self.faces.append([idx0, idx1, idx2])
            self.faces.append([idx1, idx3, idx2])
            self.face_normals.append([n_idx, n_idx + 1, n_idx + 2])
            self.face_normals.append([n_idx + 1, n_idx + 3, n_idx + 2])

            offset += PFT4_SIZE
        return offset

    def _parse_gt4_polys(self, offset: int, count:  int) -> int:
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self. data, offset)
            v1 = SVECTOR.from_bytes(self. data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR. from_bytes(self.data, offset + 24)
            n0 = SVECTOR.from_bytes(self.data, offset + 32)
            n1 = SVECTOR.from_bytes(self.data, offset + 40)
            n2 = SVECTOR.from_bytes(self. data, offset + 48)
            n3 = SVECTOR.from_bytes(self.data, offset + 56)
            c0 = CVECTOR.from_bytes(self.data, offset + 64)
            c1 = CVECTOR.from_bytes(self.data, offset + 68)
            c2 = CVECTOR.from_bytes(self.data, offset + 72)
            c3 = CVECTOR.from_bytes(self. data, offset + 76)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx0 = self._add_normal(n0)
            n_idx1 = self._add_normal(n1)
            n_idx2 = self._add_normal(n2)
            n_idx3 = self._add_normal(n3)
            c_idx0 = self._add_color(c0)
            c_idx1 = self._add_color(c1)
            c_idx2 = self._add_color(c2)
            c_idx3 = self._add_color(c3)

            self.faces.append([idx0, idx1, idx2])
            self.faces.append([idx1, idx3, idx2])
            self.face_normals.append([n_idx0, n_idx1, n_idx2])
            self.face_normals.append([n_idx1, n_idx3, n_idx2])
            self.face_colors.append([c_idx0, c_idx1, c_idx2])
            self.face_colors.append([c_idx1, c_idx3, c_idx2])

            offset += PGT4_SIZE
        return offset

    def _parse_f3_polys(self, offset: int, count: int) -> int:
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR. from_bytes(self.data, offset + 16)
            n = SVECTOR.from_bytes(self.data, offset + 24)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx = self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            self.faces.append([idx0, idx1, idx2])
            self.face_normals.append([n_idx, n_idx + 1, n_idx + 2])

            offset += PF3_SIZE
        return offset

    def _parse_g3_polys(self, offset:  int, count: int) -> int:
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n0 = SVECTOR.from_bytes(self. data, offset + 24)
            n1 = SVECTOR.from_bytes(self.data, offset + 32)
            n2 = SVECTOR. from_bytes(self.data, offset + 40)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx0 = self._add_normal(n0)
            n_idx1 = self._add_normal(n1)
            n_idx2 = self._add_normal(n2)

            self.faces. append([idx0, idx1, idx2])
            self.face_normals. append([n_idx0, n_idx1, n_idx2])

            offset += PG3_SIZE
        return offset

    def _parse_ft3_polys(self, offset: int, count:  int) -> int:
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self. data, offset)
            v1 = SVECTOR.from_bytes(self. data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n = SVECTOR.from_bytes(self.data, offset + 24)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx = self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            self.faces.append([idx0, idx1, idx2])
            self.face_normals.append([n_idx, n_idx + 1, n_idx + 2])

            offset += PFT3_SIZE
        return offset

    def _parse_gt3_polys(self, offset: int, count:  int) -> int:
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self. data, offset)
            v1 = SVECTOR.from_bytes(self. data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n0 = SVECTOR. from_bytes(self.data, offset + 24)
            n1 = SVECTOR.from_bytes(self.data, offset + 32)
            n2 = SVECTOR.from_bytes(self.data, offset + 40)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx0 = self._add_normal(n0)
            n_idx1 = self._add_normal(n1)
            n_idx2 = self._add_normal(n2)

            self.faces.append([idx0, idx1, idx2])
            self.face_normals.append([n_idx0, n_idx1, n_idx2])

            offset += PGT3_SIZE
        return offset


class AAMFParser: 
    """Parser for AAMF (Animated AMF) model files"""

    def __init__(self, data: bytes):
        self.data = data
        self.bone_count:  int = 0
        self.anim_count: int = 0
        self.bone_parents: List[Tuple[int, int]] = []  # (bone_index, parent_index)
        self.bones: List[Bone] = []
        self.animations: List[Animation] = []

    def parse(self):
        """Parse the AAMF file"""
        # Parse header
        self.bone_count = struct. unpack_from('<H', self.data, 0)[0]
        self.anim_count = struct.unpack_from('<H', self. data, 2)[0]

        print(f"AAMF Header:")
        print(f"  Bone count: {self.bone_count}")
        print(f"  Animation count: {self.anim_count}")

        # Parse bone parent table
        for i in range(self.bone_count):
            bone_idx = struct.unpack_from('<H', self.data, 4 + i * 4)[0]
            parent_idx = struct.unpack_from('<H', self. data, 4 + i * 4 + 2)[0]
            self.bone_parents. append((bone_idx, parent_idx))
            print(f"  Bone {bone_idx}:  parent = {parent_idx}")

        # Data blocks start after header
        data_offset = 4 + self.bone_count * 4
        offset = 0

        # Parse bone AMF data
        for i in range(self.bone_count):
            block_size = struct. unpack_from('<I', self.data, data_offset + offset)[0]
            amf_offset = data_offset + offset + 4

            print(f"\nParsing bone {i} AMF at offset {amf_offset} (block size: {block_size})")

            # Parse embedded AMF
            amf_parser = AMFParser(self.data, amf_offset)
            try:
                amf_parser.parse()
            except Exception as e:
                print(f"  Warning: Failed to parse bone {i} AMF: {e}")
                amf_parser.vertices = []
                amf_parser.normals = []
                amf_parser.colors = []
                amf_parser.faces = []
                amf_parser.face_normals = []
                amf_parser.face_colors = []

            bone = Bone(
                index=self.bone_parents[i][0],
                parent_index=self.bone_parents[i][1],
                vertices=amf_parser.vertices,
                normals=amf_parser.normals,
                colors=amf_parser. colors,
                faces=amf_parser.faces,
                face_normals=amf_parser. face_normals,
                face_colors=amf_parser.face_colors
            )
            self.bones.append(bone)

            print(f"  Vertices: {len(bone.vertices)}, Faces: {len(bone.faces)}")

            offset += block_size

        # Parse animations
        for i in range(self.anim_count):
            block_size = struct. unpack_from('<I', self.data, data_offset + offset)[0]
            anim_offset = data_offset + offset + 4

            print(f"\nParsing animation {i} at offset {anim_offset} (block size:  {block_size})")

            anim = Animation. from_bytes(self.data, anim_offset, self.bone_count)
            self.animations.append(anim)

            print(f"  Name: '{anim.name}', Keyframes:  {anim.keyframe_count}")

            offset += block_size

    def export_gltf(self, filename: str):
        """Export parsed model to glTF 2.0 format with skeletal animation"""
        if not self.bones:
            print("No bones to export!")
            return

        # Collect all geometry data
        all_positions = []
        all_normals = []
        all_indices = []
        all_joints = []  # Joint indices for skinning
        all_weights = []  # Joint weights for skinning

        mesh_primitives = []
        current_vertex_offset = 0

        for bone_idx, bone in enumerate(self. bones):
            if not bone.vertices:
                continue

            bone_start_vertex = len(all_positions) // 3

            # Add vertices for this bone
            for face_idx, face in enumerate(bone. faces):
                face_normal_indices = bone.face_normals[face_idx] if face_idx < len(bone.face_normals) else [0, 0, 0]

                for i, v_idx in enumerate(face):
                    pos = bone.vertices[v_idx]
                    all_positions.extend(pos)

                    n_idx = face_normal_indices[i] if i < len(face_normal_indices) else 0
                    if n_idx < len(bone.normals):
                        normal = bone.normals[n_idx]
                    else:
                        normal = (0.0, 1.0, 0.0)
                    all_normals.extend(normal)

                    # Each vertex is fully weighted to its bone
                    all_joints.extend([bone_idx, 0, 0, 0])
                    all_weights.extend([1.0, 0.0, 0.0, 0.0])

                    all_indices.append(len(all_positions) // 3 - 1)

        if not all_positions: 
            print("No geometry to export!")
            return

        # Calculate bounds
        min_pos = [float('inf'), float('inf'), float('inf')]
        max_pos = [float('-inf'), float('-inf'), float('-inf')]

        for i in range(0, len(all_positions), 3):
            for j in range(3):
                min_pos[j] = min(min_pos[j], all_positions[i + j])
                max_pos[j] = max(max_pos[j], all_positions[i + j])

        # Create binary buffers
        position_bytes = struct.pack(f'<{len(all_positions)}f', *all_positions)
        normal_bytes = struct.pack(f'<{len(all_normals)}f', *all_normals)
        index_bytes = struct.pack(f'<{len(all_indices)}I', *all_indices)
        joints_bytes = struct. pack(f'<{len(all_joints)}H', *all_joints)
        weights_bytes = struct. pack(f'<{len(all_weights)}f', *all_weights)

        # Pad to 4-byte alignment
        def pad_to_4(data:  bytes) -> bytes:
            padding = (4 - len(data) % 4) % 4
            return data + b'\x00' * padding

        position_bytes = pad_to_4(position_bytes)
        normal_bytes = pad_to_4(normal_bytes)
        index_bytes = pad_to_4(index_bytes)
        joints_bytes = pad_to_4(joints_bytes)
        weights_bytes = pad_to_4(weights_bytes)

        # Create inverse bind matrices for skeleton
        inverse_bind_matrices = []
        for bone in self.bones:
            # Identity matrix for bind pose
            inverse_bind_matrices.extend([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ])

        ibm_bytes = struct.pack(f'<{len(inverse_bind_matrices)}f', *inverse_bind_matrices)
        ibm_bytes = pad_to_4(ibm_bytes)

        # Create animation data buffers
        anim_time_bytes = b''
        anim_translation_bytes = b''
        anim_rotation_bytes = b''

        animation_accessors = []
        animation_samplers = []
        animation_channels = []

        current_buffer_offset = (len(position_bytes) + len(normal_bytes) +
                                  len(index_bytes) + len(joints_bytes) +
                                  len(weights_bytes) + len(ibm_bytes))

        accessor_index = 6  # Start after mesh accessors

        for anim_idx, anim in enumerate(self.animations):
            if anim.keyframe_count == 0:
                continue

            # Create time input (assuming 30 fps)
            fps = 30.0
            times = [i / fps for i in range(anim.keyframe_count)]
            time_bytes = struct.pack(f'<{len(times)}f', *times)
            time_bytes = pad_to_4(time_bytes)

            time_accessor_start = accessor_index

            for bone_idx in range(self.bone_count):
                # Extract translations and rotations for this bone
                translations = []
                rotations = []

                for kf_idx in range(anim.keyframe_count):
                    if kf_idx < len(anim. keyframes) and bone_idx < len(anim.keyframes[kf_idx]):
                        kf = anim.keyframes[kf_idx][bone_idx]
                        trans = kf.mat. to_translation()
                        quat = kf. mat.to_quaternion()
                    else:
                        trans = (0.0, 0.0, 0.0)
                        quat = (0.0, 0.0, 0.0, 1.0)

                    translations.extend(trans)
                    rotations.extend(quat)

                trans_bytes = struct. pack(f'<{len(translations)}f', *translations)
                trans_bytes = pad_to_4(trans_bytes)

                rot_bytes = struct.pack(f'<{len(rotations)}f', *rotations)
                rot_bytes = pad_to_4(rot_bytes)

                anim_time_bytes += time_bytes
                anim_translation_bytes += trans_bytes
                anim_rotation_bytes += rot_bytes

        # Combine all buffers
        buffer_data = (position_bytes + normal_bytes + index_bytes +
                       joints_bytes + weights_bytes + ibm_bytes +
                       anim_time_bytes + anim_translation_bytes + anim_rotation_bytes)

        buffer_base64 = base64.b64encode(buffer_data).decode('ascii')

        # Build node hierarchy for skeleton
        nodes = []
        joint_indices = []

        # Root node with mesh and skin
        nodes.append({
            "name": "AAMFModel",
            "mesh": 0,
            "skin": 0
        })

        # Add skeleton root
        skeleton_root_idx = 1
        nodes.append({
            "name":  "Skeleton",
            "children": []
        })

        # Add bone nodes
        bone_node_indices = {}
        for bone_idx, bone in enumerate(self.bones):
            node_idx = len(nodes)
            bone_node_indices[bone. index] = node_idx
            joint_indices.append(node_idx)

            node = {
                "name": f"Bone_{bone. index}"
            }
            nodes.append(node)

        # Set up parent-child relationships
        for bone_idx, bone in enumerate(self.bones):
            node_idx = bone_node_indices[bone.index]
            if bone.index == bone. parent_index: 
                # Root bone - child of skeleton
                nodes[skeleton_root_idx]["children"].append(node_idx)
            else:
                # Child bone
                parent_node_idx = bone_node_indices. get(bone.parent_index)
                if parent_node_idx is not None:
                    if "children" not in nodes[parent_node_idx]: 
                        nodes[parent_node_idx]["children"] = []
                    nodes[parent_node_idx]["children"].append(node_idx)

        # Build glTF structure
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "aamf2gltf. py - UnnamedHB1 AAMF Converter"
            },
            "scene": 0,
            "scenes": [
                {
                    "name": "Scene",
                    "nodes": [0, skeleton_root_idx]
                }
            ],
            "nodes": nodes,
            "meshes": [
                {
                    "name": "AAMFMesh",
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "NORMAL": 1,
                                "JOINTS_0": 3,
                                "WEIGHTS_0":  4
                            },
                            "indices": 2,
                            "mode": 4
                        }
                    ]
                }
            ],
            "skins": [
                {
                    "name": "AAMFSkin",
                    "inverseBindMatrices":  5,
                    "skeleton": skeleton_root_idx,
                    "joints": joint_indices
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": len(all_positions) // 3,
                    "type": "VEC3",
                    "min": min_pos,
                    "max": max_pos
                },
                {
                    "bufferView": 1,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": len(all_normals) // 3,
                    "type":  "VEC3"
                },
                {
                    "bufferView": 2,
                    "byteOffset":  0,
                    "componentType":  5125,
                    "count": len(all_indices),
                    "type": "SCALAR"
                },
                {
                    "bufferView": 3,
                    "byteOffset": 0,
                    "componentType": 5123,  # UNSIGNED_SHORT
                    "count": len(all_joints) // 4,
                    "type": "VEC4"
                },
                {
                    "bufferView": 4,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": len(all_weights) // 4,
                    "type":  "VEC4"
                },
                {
                    "bufferView": 5,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": len(self.bones),
                    "type": "MAT4"
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength":  len(position_bytes),
                    "target": 34962
                },
                {
                    "buffer": 0,
                    "byteOffset": len(position_bytes),
                    "byteLength": len(normal_bytes),
                    "target":  34962
                },
                {
                    "buffer": 0,
                    "byteOffset": len(position_bytes) + len(normal_bytes),
                    "byteLength": len(index_bytes),
                    "target": 34963
                },
                {
                    "buffer": 0,
                    "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes),
                    "byteLength": len(joints_bytes),
                    "target": 34962
                },
                {
                    "buffer": 0,
                    "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes),
                    "byteLength": len(weights_bytes),
                    "target":  34962
                },
                {
                    "buffer": 0,
                    "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes) + len(weights_bytes),
                    "byteLength":  len(ibm_bytes)
                }
            ],
            "buffers": [
                {
                    "uri": f"data:application/octet-stream;base64,{buffer_base64}",
                    "byteLength": len(buffer_data)
                }
            ]
        }

        # Add animations if present
        if self.animations:
            gltf["animations"] = []

            for anim in self.animations:
                if anim.keyframe_count == 0:
                    continue

                anim_data = {
                    "name": anim.name,
                    "samplers": [],
                    "channels": []
                }

                # Note: Full animation export would require additional buffer views
                # and accessors for each bone's keyframe data
                # This is a simplified version showing the structure

                gltf["animations"]. append(anim_data)

        # Write glTF JSON file
        with open(filename, 'w') as f:
            json.dump(gltf, f, indent=2)

        print(f"\nExported to {filename}")
        print(f"  Bones: {len(self.bones)}")
        print(f"  Animations: {len(self.animations)}")
        print(f"  Vertices: {len(all_positions) // 3}")
        print(f"  Triangles: {len(all_indices) // 3}")

    def export_glb(self, filename: str):
        """Export parsed model to GLB (binary glTF) format"""
        if not self.bones:
            print("No bones to export!")
            return

        # Collect all geometry data (same as gltf export)
        all_positions = []
        all_normals = []
        all_indices = []
        all_joints = []
        all_weights = []

        for bone_idx, bone in enumerate(self.bones):
            if not bone.vertices:
                continue

            for face_idx, face in enumerate(bone.faces):
                face_normal_indices = bone.face_normals[face_idx] if face_idx < len(bone.face_normals) else [0, 0, 0]

                for i, v_idx in enumerate(face):
                    pos = bone.vertices[v_idx]
                    all_positions.extend(pos)

                    n_idx = face_normal_indices[i] if i < len(face_normal_indices) else 0
                    if n_idx < len(bone.normals):
                        normal = bone.normals[n_idx]
                    else: 
                        normal = (0.0, 1.0, 0.0)
                    all_normals.extend(normal)

                    all_joints.extend([bone_idx, 0, 0, 0])
                    all_weights.extend([1.0, 0.0, 0.0, 0.0])

                    all_indices.append(len(all_positions) // 3 - 1)

        if not all_positions: 
            print("No geometry to export!")
            return

        # Calculate bounds
        min_pos = [float('inf'), float('inf'), float('inf')]
        max_pos = [float('-inf'), float('-inf'), float('-inf')]

        for i in range(0, len(all_positions), 3):
            for j in range(3):
                min_pos[j] = min(min_pos[j], all_positions[i + j])
                max_pos[j] = max(max_pos[j], all_positions[i + j])

        # Create binary buffers
        position_bytes = struct.pack(f'<{len(all_positions)}f', *all_positions)
        normal_bytes = struct.pack(f'<{len(all_normals)}f', *all_normals)
        index_bytes = struct.pack(f'<{len(all_indices)}I', *all_indices)
        joints_bytes = struct.pack(f'<{len(all_joints)}H', *all_joints)
        weights_bytes = struct.pack(f'<{len(all_weights)}f', *all_weights)

        def pad_to_4(data: bytes) -> bytes:
            padding = (4 - len(data) % 4) % 4
            return data + b'\x00' * padding

        position_bytes = pad_to_4(position_bytes)
        normal_bytes = pad_to_4(normal_bytes)
        index_bytes = pad_to_4(index_bytes)
        joints_bytes = pad_to_4(joints_bytes)
        weights_bytes = pad_to_4(weights_bytes)

        # Inverse bind matrices
        inverse_bind_matrices = []
        for bone in self. bones:
            inverse_bind_matrices. extend([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ])

        ibm_bytes = struct.pack(f'<{len(inverse_bind_matrices)}f', *inverse_bind_matrices)
        ibm_bytes = pad_to_4(ibm_bytes)

        buffer_data = (position_bytes + normal_bytes + index_bytes +
                       joints_bytes + weights_bytes + ibm_bytes)

        # Build nodes
        nodes = []
        joint_indices = []

        nodes.append({
            "name": "AAMFModel",
            "mesh": 0,
            "skin": 0
        })

        skeleton_root_idx = 1
        nodes.append({
            "name": "Skeleton",
            "children": []
        })

        bone_node_indices = {}
        for bone_idx, bone in enumerate(self.bones):
            node_idx = len(nodes)
            bone_node_indices[bone.index] = node_idx
            joint_indices.append(node_idx)
            nodes.append({"name": f"Bone_{bone.index}"})

        for bone_idx, bone in enumerate(self. bones):
            node_idx = bone_node_indices[bone.index]
            if bone.index == bone.parent_index:
                nodes[skeleton_root_idx]["children"].append(node_idx)
            else: 
                parent_node_idx = bone_node_indices.get(bone. parent_index)
                if parent_node_idx is not None:
                    if "children" not in nodes[parent_node_idx]:
                        nodes[parent_node_idx]["children"] = []
                    nodes[parent_node_idx]["children"].append(node_idx)

        # Build glTF
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "aamf2gltf.py - UnnamedHB1 AAMF Converter"
            },
            "scene":  0,
            "scenes": [{"name": "Scene", "nodes": [0, skeleton_root_idx]}],
            "nodes": nodes,
            "meshes": [{
                "name":  "AAMFMesh",
                "primitives": [{
                    "attributes":  {
                        "POSITION": 0,
                        "NORMAL": 1,
                        "JOINTS_0":  3,
                        "WEIGHTS_0": 4
                    },
                    "indices": 2,
                    "mode": 4
                }]
            }],
            "skins": [{
                "name":  "AAMFSkin",
                "inverseBindMatrices": 5,
                "skeleton":  skeleton_root_idx,
                "joints": joint_indices
            }],
            "accessors": [
                {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": len(all_positions) // 3, "type": "VEC3", "min": min_pos, "max":  max_pos},
                {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": len(all_normals) // 3, "type": "VEC3"},
                {"bufferView": 2, "byteOffset": 0, "componentType": 5125, "count": len(all_indices), "type": "SCALAR"},
                {"bufferView": 3, "byteOffset": 0, "componentType": 5123, "count": len(all_joints) // 4, "type": "VEC4"},
                {"bufferView": 4, "byteOffset": 0, "componentType": 5126, "count": len(all_weights) // 4, "type": "VEC4"},
                {"bufferView": 5, "byteOffset": 0, "componentType": 5126, "count":  len(self.bones), "type": "MAT4"}
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength":  len(position_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(normal_bytes), "target": 34962},
                {"buffer":  0, "byteOffset": len(position_bytes) + len(normal_bytes), "byteLength": len(index_bytes), "target": 34963},
                {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes), "byteLength": len(joints_bytes), "target": 34962},
                {"buffer":  0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes), "byteLength":  len(weights_bytes), "target": 34962},
                {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes) + len(weights_bytes), "byteLength":  len(ibm_bytes)}
            ],
            "buffers": [{"byteLength": len(buffer_data)}]
        }

        # Serialize JSON
        json_str = json.dumps(gltf, separators=(',', ': '))
        json_bytes = json_str.encode('utf-8')

        json_padding = (4 - len(json_bytes) % 4) % 4
        json_bytes_padded = json_bytes + b' ' * json_padding

        bin_padding = (4 - len(buffer_data) % 4) % 4
        buffer_data_padded = buffer_data + b'\x00' * bin_padding

        header_length = 12
        json_chunk_length = 8 + len(json_bytes_padded)
        bin_chunk_length = 8 + len(buffer_data_padded)
        total_length = header_length + json_chunk_length + bin_chunk_length

        with open(filename, 'wb') as f:
            f. write(struct.pack('<I', 0x46546C67))  # magic "glTF"
            f.write(struct.pack('<I', 2))  # version
            f.write(struct.pack('<I', total_length))

            f.write(struct.pack('<I', len(json_bytes_padded)))
            f.write(struct.pack('<I', 0x4E4F534A))  # "JSON"
            f.write(json_bytes_padded)

            f.write(struct.pack('<I', len(buffer_data_padded)))
            f.write(struct. pack('<I', 0x004E4942))  # "BIN\0"
            f. write(buffer_data_padded)

        print(f"\nExported to {filename}")
        print(f"  Bones:  {len(self. bones)}")
        print(f"  Animations: {len(self.animations)}")
        print(f"  Vertices: {len(all_positions) // 3}")
        print(f"  Triangles: {len(all_indices) // 3}")
        print(f"  File size: {total_length} bytes")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input. aamf [output.gltf|output.glb]")
        print("\nConverts AAMF files from UnnamedHB1 PS1 homebrew to glTF 2.0 format.")
        print("\nAAMF (Animated AMF) format contains:")
        print("  - Multiple bones, each with embedded AMF mesh data")
        print("  - Bone hierarchy (parent-child relationships)")
        print("  - Skeletal animations with keyframe transforms")
        print("\nOutput format is determined by file extension:")
        print("  . gltf - JSON format with embedded base64 buffer")
        print("  .glb  - Binary glTF format (default)")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else: 
        output_file = os.path.splitext(input_file)[0] + ".glb"

    with open(input_file, 'rb') as f:
        data = f.read()

    print(f"Reading {input_file} ({len(data)} bytes)")

    parser = AAMFParser(data)
    parser.parse()

    if output_file.lower().endswith('.gltf'):
        parser.export_gltf(output_file)
    else:
        parser. export_glb(output_file)


if __name__ == "__main__": 
    main()