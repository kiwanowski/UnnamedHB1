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
    None (uses only Python standard library)
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
    vz: int  # int16_t
    pad: int  # int16_t

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'SVECTOR': 
        vx, vy, vz, pad = struct.unpack_from('<hhhh', data, offset)
        return cls(vx, vy, vz, pad)

    def to_float(self, scale: float = 1.0 / 4096.0) -> Tuple[float, float, float]:
        """Convert fixed-point to floating point coordinates"""
        return (self.vx * scale, self. vy * scale, self.vz * scale)


@dataclass
class CVECTOR:
    """Color vector (4 bytes)"""
    r:  int  # uint8_t
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
    - m[3][3]:  3x3 rotation matrix (int16_t, 4. 12 fixed-point)
    - t[3]: translation vector (int32_t)
    """
    m: List[List[int]]  # 3x3 rotation matrix
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

        # Read translation vector (3 int32_t = 12 bytes) - starts at offset 20 (with 2 bytes padding after matrix)
        t = list(struct.unpack_from('<iii', data, offset + 20))

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
            s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
            w = (m[0][2] - m[2][0]) / s
            x = (m[0][1] + m[1][0]) / s
            y = 0.25 * s
            z = (m[1][2] + m[2][1]) / s
        else:
            s = 2.0 * math. sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
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
    # keyframes[bone_idx][keyframe_idx] - matches the C code layout
    keyframes:  List[List[Keyframe]]

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int, bone_count: int) -> 'Animation': 
        # Read animation header
        name_bytes = data[offset: offset + 8]
        name = name_bytes.rstrip(b'\x00').decode('ascii', errors='replace')
        keyframe_count = struct.unpack_from('<I', data, offset + 8)[0]

        # Skip the keyframe pointer (4 bytes) - set at runtime
        keyframe_data_offset = offset + 16

        # Read keyframes - organized as [bone][keyframe] in memory
        # From C code: keyframe + bone_idx * keyframeamount
        # So it's stored as: all keyframes for bone 0, then all keyframes for bone 1, etc.
        keyframes = []
        for bone_idx in range(bone_count):
            bone_keyframes = []
            for kf_idx in range(keyframe_count):
                kf_offset = keyframe_data_offset + (bone_idx * keyframe_count + kf_idx) * 32
                kf = Keyframe.from_bytes(data, kf_offset)
                bone_keyframes.append(kf)
            keyframes.append(bone_keyframes)

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
    """AMF file header"""
    used_textures: int
    x:  int  # chunk count X
    z: int  # chunk count Z
    bounds: WorldBounds

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'AMFHeader':
        used_textures, x, z = struct.unpack_from('<IHH', data, offset)
        bounds = WorldBounds.from_bytes(data, offset + 8)
        return cls(used_textures & 0x7FFFFFFF, x, z, bounds)


@dataclass
class ChunkHeader:
    """Chunk header with polygon counts (16 bytes for counts + 32 bytes for 8 pointers)"""
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


# Sizes of tagless GPU primitives (POLY_*_T versions without the 4-byte tag header)
# From psxgpu.h in PSn00bSDK
POLY_F3_T_SIZE = 16   # r,g,b,code + 3*(x,y)
POLY_F4_T_SIZE = 20   # r,g,b,code + 4*(x,y)
POLY_FT3_T_SIZE = 28  # r,g,b,code + (x,y,u,v,clut) + (x,y,u,v,tpage) + (x,y,u,v,pad)
POLY_FT4_T_SIZE = 36  # r,g,b,code + 4*(x,y,u,v) + clut,tpage,pad0,pad1
POLY_G3_T_SIZE = 24   # 3*(r,g,b,pad/code + x,y)
POLY_G4_T_SIZE = 32   # 4*(r,g,b,pad/code + x,y)
POLY_GT3_T_SIZE = 36  # (r,g,b,code,x,y,u,v,clut) + 2*(r,g,b,pad,x,y,u,v,tpage/pad)
POLY_GT4_T_SIZE = 48  # 4*(r,g,b,pad/code,x,y,u,v) + clut,tpage,pad2,pad3,pad4

SVECTOR_SIZE = 8
CVECTOR_SIZE = 4
POINTER_SIZE = 4

# Polygon structure sizes from model.h
# PF3: 3 vertices + 1 normal + POLY_F3
PF3_SIZE = 3 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POLY_F3_T_SIZE  # 24 + 8 + 16 = 48?  
# Actually looking at struct:  v0,v1,v2 (3*8=24) + n (8) + POLY_F3 (with tag=4+16=20) = 52
# But file stores tagless primitives - let me recalculate based on C struct sizes

# From model.h, these structs contain the POLY_* (with tag), not POLY_*_T
# sizeof(PF3) = 3*8 + 1*8 + sizeof(POLY_F3) = 32 + (4+16) = 52
# sizeof(PF4) = 4*8 + 1*8 + sizeof(POLY_F4) = 40 + (4+20) = 64
# sizeof(PFT3) = 3*8 + 1*8 + 4 + sizeof(POLY_FT3) = 36 + (4+28) = 68
# sizeof(PFT4) = 4*8 + 1*8 + 4 + sizeof(POLY_FT4) = 44 + (4+36) = 84
# sizeof(PG3) = 3*8 + 3*8 + sizeof(POLY_G3) = 48 + (4+24) = 76
# sizeof(PG4) = 4*8 + 4*8 + sizeof(POLY_G4) = 64 + (4+32) = 100
# sizeof(PGT3) = 3*8 + 3*8 + 4 + sizeof(POLY_GT3) = 52 + (4+36) = 92
# sizeof(PGT4) = 4*8 + 4*8 + 4*4 + 4 + sizeof(POLY_GT4) = 84 + (4+48) = 136

PF3_SIZE = 52
PF4_SIZE = 64
PFT3_SIZE = 68
PFT4_SIZE = 84
PG3_SIZE = 76
PG4_SIZE = 100
PGT3_SIZE = 92
PGT4_SIZE = 136


@dataclass
class SkinWeight:
    """Skin weight data for a single vertex (max 4 bone influences)"""
    bone_indices: List[int]  # up to 4 bone indices
    weights: List[float]  # normalized weights (0.0-1.0)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'SkinWeight':
        """Parse skin weight from 8 bytes: 4 bone indices + 4 weights (uint8_t each)"""
        bone_indices = list(struct.unpack_from('<BBBB', data, offset))
        weight_bytes = list(struct.unpack_from('<BBBB', data, offset + 4))
        # Convert 0-255 to 0.0-1.0
        weights = [w / 255.0 for w in weight_bytes]
        return cls(bone_indices, weights)


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

    def __init__(self, data: bytes, offset: int = 0, max_size: int = 0):
        self.data = data
        self.base_offset = offset
        self.max_size = max_size
        self.header:  Optional[AMFHeader] = None
        self.texture_names: List[str] = []
        self.vertices: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.colors: List[Tuple[float, float, float, float]] = []
        self. faces: List[List[int]] = []
        self.face_normals: List[List[int]] = []
        self.face_colors: List[List[int]] = []

    def parse(self):
        """Parse the embedded AMF data"""
        offset = self.base_offset

        # Parse header (24 bytes:  4 + 2 + 2 + 16)
        self.header = AMFHeader. from_bytes(self.data, offset)

        print(f"    AMF:  textures={self.header.used_textures}, chunks={self.header.x}x{self.header.z}")

        # Parse texture names (8 bytes each)
        texture_offset = offset + 8 + 16  # Header = 8 + WorldBounds = 16
        for i in range(self.header.used_textures):
            name_bytes = self.data[texture_offset:texture_offset + 8]
            name = name_bytes. rstrip(b'\x00').decode('ascii', errors='replace')
            self.texture_names.append(name)
            texture_offset += 8

        # Parse chunks
        chunk_count = self.header. x * self.header.z
        
        # Chunk table:  chunk_count * 4 bytes (offsets/info)
        chunk_table_start = texture_offset
        
        # Polygon data follows the chunk table
        # Looking at amfInitData:  chunks[i] = (AMF_CHUNK*)(temp+ltemp) where ltemp starts at chunk_count
        # The chunk table entries are used differently - they're reinterpreted as AMF_CHUNK pointers
        
        # Based on model.c line 68:  amf->chunks[i] = (AMF_CHUNK*)(temp+ltemp)
        # where temp is at chunk table start and ltemp = chunk_count initially
        # So the actual chunk data starts after chunk_count * 4 bytes from the chunk table
        
        poly_data_offset = chunk_table_start + chunk_count * 4
        
        for chunk_idx in range(chunk_count):
            chunk_offset = poly_data_offset
            self._parse_chunk(chunk_idx, chunk_offset)
            
            # Read chunk header to calculate size
            chunk_header = ChunkHeader. from_bytes(self.data, chunk_offset)
            
            # Chunk size = 16 (header counts) + 32 (8 pointers) + polygon data
            chunk_size = 16 + 32  # Header (8 uint16) + 8 pointers
            chunk_size += chunk_header. F4_amount * PF4_SIZE
            chunk_size += chunk_header.G4_amount * PG4_SIZE
            chunk_size += chunk_header.FT4_amount * PFT4_SIZE
            chunk_size += chunk_header.GT4_amount * PGT4_SIZE
            chunk_size += chunk_header.F3_amount * PF3_SIZE
            chunk_size += chunk_header.G3_amount * PG3_SIZE
            chunk_size += chunk_header.FT3_amount * PFT3_SIZE
            chunk_size += chunk_header.GT3_amount * PGT3_SIZE
            
            poly_data_offset += chunk_size

    def _parse_chunk(self, chunk_idx: int, offset: int):
        """Parse a single chunk"""
        chunk_header = ChunkHeader. from_bytes(self.data, offset)
        
        # Polygon data starts after header (16 bytes) and pointers (32 bytes)
        # From model.c line 82: f4_polies = (PF4*)(((uint32_t*)amf->chunks[i])+12)
        # 12 * 4 = 48 bytes = 16 (header) + 32 (pointers)
        poly_offset = offset + 48

        poly_offset = self._parse_f4_polys(poly_offset, chunk_header. F4_amount)
        poly_offset = self._parse_g4_polys(poly_offset, chunk_header.G4_amount)
        poly_offset = self._parse_ft4_polys(poly_offset, chunk_header. FT4_amount)
        poly_offset = self._parse_gt4_polys(poly_offset, chunk_header. GT4_amount)
        poly_offset = self._parse_f3_polys(poly_offset, chunk_header.F3_amount)
        poly_offset = self._parse_g3_polys(poly_offset, chunk_header.G3_amount)
        poly_offset = self._parse_ft3_polys(poly_offset, chunk_header.FT3_amount)
        poly_offset = self._parse_gt3_polys(poly_offset, chunk_header.GT3_amount)

    def _add_vertex(self, sv: SVECTOR) -> int:
        self.vertices.append(sv. to_float())
        return len(self.vertices) - 1

    def _add_normal(self, sv: SVECTOR) -> int:
        self.normals. append(sv.to_float())
        return len(self. normals) - 1

    def _add_color(self, cv: CVECTOR) -> int:
        self.colors.append(cv.to_float())
        return len(self.colors) - 1

    def _parse_f4_polys(self, offset: int, count: int) -> int:
        """Parse flat-shaded quads (PF4)
        struct:  v0,v1,v2,v3 (4*8=32), n (8), POLY_F4 (24)
        """
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n = SVECTOR.from_bytes(self.data, offset + 32)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx = self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            # PS1 quad winding:  v0-v1-v2-v3 forms a Z pattern
            # Split into two triangles:  v0-v1-v2 and v1-v3-v2
            self.faces.append([idx0, idx2, idx1])
            self.faces.append([idx1, idx2, idx3])
            self.face_normals.append([n_idx, n_idx + 2, n_idx + 1])
            self.face_normals.append([n_idx + 1, n_idx + 2, n_idx + 3])

            offset += PF4_SIZE
        return offset

    def _parse_g4_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded quads (PG4)
        struct: v0,v1,v2,v3 (32), n0,n1,n2,n3 (32), POLY_G4 (36)
        """
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n0 = SVECTOR.from_bytes(self.data, offset + 32)
            n1 = SVECTOR. from_bytes(self.data, offset + 40)
            n2 = SVECTOR.from_bytes(self.data, offset + 48)
            n3 = SVECTOR.from_bytes(self.data, offset + 56)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx0 = self._add_normal(n0)
            n_idx1 = self._add_normal(n1)
            n_idx2 = self._add_normal(n2)
            n_idx3 = self._add_normal(n3)

            self. faces.append([idx0, idx2, idx1])
            self.faces.append([idx1, idx2, idx3])
            self.face_normals.append([n_idx0, n_idx2, n_idx1])
            self.face_normals. append([n_idx1, n_idx2, n_idx3])

            offset += PG4_SIZE
        return offset

    def _parse_ft4_polys(self, offset: int, count:  int) -> int:
        """Parse flat-shaded textured quads (PFT4)
        struct: v0,v1,v2,v3 (32), n (8), tex* (4), POLY_FT4 (40)
        """
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n = SVECTOR. from_bytes(self.data, offset + 32)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx = self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            self.faces.append([idx0, idx2, idx1])
            self.faces. append([idx1, idx2, idx3])
            self.face_normals. append([n_idx, n_idx + 2, n_idx + 1])
            self.face_normals.append([n_idx + 1, n_idx + 2, n_idx + 3])

            offset += PFT4_SIZE
        return offset

    def _parse_gt4_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded textured quads (PGT4)
        struct: v0,v1,v2,v3 (32), n0,n1,n2,n3 (32), c0,c1,c2,c3 (16), tex* (4), POLY_GT4 (52)
        """
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n0 = SVECTOR.from_bytes(self.data, offset + 32)
            n1 = SVECTOR. from_bytes(self.data, offset + 40)
            n2 = SVECTOR.from_bytes(self.data, offset + 48)
            n3 = SVECTOR.from_bytes(self.data, offset + 56)
            c0 = CVECTOR.from_bytes(self.data, offset + 64)
            c1 = CVECTOR.from_bytes(self.data, offset + 68)
            c2 = CVECTOR.from_bytes(self.data, offset + 72)
            c3 = CVECTOR.from_bytes(self.data, offset + 76)

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

            self.faces.append([idx0, idx2, idx1])
            self.faces.append([idx1, idx2, idx3])
            self.face_normals.append([n_idx0, n_idx2, n_idx1])
            self.face_normals.append([n_idx1, n_idx2, n_idx3])
            self.face_colors.append([c_idx0, c_idx2, c_idx1])
            self.face_colors.append([c_idx1, c_idx2, c_idx3])

            offset += PGT4_SIZE
        return offset

    def _parse_f3_polys(self, offset:  int, count: int) -> int:
        """Parse flat-shaded triangles (PF3)
        struct: v0,v1,v2 (24), n (8), POLY_F3 (20)
        """
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n = SVECTOR.from_bytes(self.data, offset + 24)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx = self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            self.faces. append([idx0, idx1, idx2])
            self.face_normals. append([n_idx, n_idx + 1, n_idx + 2])

            offset += PF3_SIZE
        return offset

    def _parse_g3_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded triangles (PG3)
        struct: v0,v1,v2 (24), n0,n1,n2 (24), POLY_G3 (28)
        """
        for _ in range(count):
            v0 = SVECTOR. from_bytes(self.data, offset)
            v1 = SVECTOR. from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n0 = SVECTOR.from_bytes(self.data, offset + 24)
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

            offset += PG3_SIZE
        return offset

    def _parse_ft3_polys(self, offset: int, count: int) -> int:
        """Parse flat-shaded textured triangles (PFT3)
        struct: v0,v1,v2 (24), n (8), tex* (4), POLY_FT3 (32)
        """
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n = SVECTOR.from_bytes(self.data, offset + 24)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx = self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            self.faces. append([idx0, idx1, idx2])
            self.face_normals. append([n_idx, n_idx + 1, n_idx + 2])

            offset += PFT3_SIZE
        return offset

    def _parse_gt3_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded textured triangles (PGT3)
        struct: v0,v1,v2 (24), n0,n1,n2 (24), tex* (4), POLY_GT3 (40)
        """
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n0 = SVECTOR.from_bytes(self.data, offset + 24)
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
        self.is_skinned: bool = False
        self.format_version: int = 1
        self.bind_poses: List[MATRIX] = []
        # Skinned mesh data (unified mesh for all bones)
        self.vertices: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.skin_weights: List[SkinWeight] = []
        self.faces: List[List[int]] = []

    def parse(self):
        """Parse the AAMF file"""
        # Check for skinned format magic ('SAMF')
        magic = self.data[0:4]
        if magic == b'SAMF':
            self.is_skinned = True
            self._parse_skinned()
        elif magic == b'AAMF':
            self.is_skinned = True
            self._parse_skinned()
        else:
            # Legacy format (no magic, starts with bone count)
            self.is_skinned = False
            self._parse_legacy()

    def _parse_legacy(self):
        """Parse the legacy AAMF file format"""
        # Parse header
        self.bone_count = struct.unpack_from('<H', self.data, 0)[0]
        self.anim_count = struct.unpack_from('<H', self.data, 2)[0]

        print(f"Legacy AAMF Header:")
        print(f"  Bone count: {self.bone_count}")
        print(f"  Animation count: {self.anim_count}")

        # Parse bone parent table (pairs of uint16_t for each bone)
        for i in range(self.bone_count):
            bone_idx = struct.unpack_from('<H', self.data, 4 + i * 4)[0]
            parent_idx = struct.unpack_from('<H', self.data, 4 + i * 4 + 2)[0]
            self.bone_parents.append((bone_idx, parent_idx))
            print(f"  Bone {bone_idx}:  parent = {parent_idx}")

        # Data blocks start after header:  4 bytes + boneamount * 4 bytes
        data_start = 4 + self.bone_count * 4
        offset = 0

        # Parse bone AMF data
        for i in range(self.bone_count):
            # Each block starts with a 4-byte size
            block_size = struct.unpack_from('<I', self.data, data_start + offset)[0]
            amf_offset = data_start + offset + 4

            print(f"\nParsing bone {i} AMF at offset {amf_offset} (block size: {block_size})")

            # Parse embedded AMF
            amf_parser = AMFParser(self.data, amf_offset, block_size - 4)
            try:
                amf_parser.parse()
            except Exception as e:
                print(f"  Warning: Failed to parse bone {i} AMF: {e}")
                import traceback
                traceback.print_exc()
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
                colors=amf_parser.colors,
                faces=amf_parser.faces,
                face_normals=amf_parser.face_normals,
                face_colors=amf_parser.face_colors
            )
            self.bones.append(bone)

            print(f"  Vertices: {len(bone.vertices)}, Faces: {len(bone.faces)}")

            offset += block_size

        # Parse animations
        for i in range(self. anim_count):
            block_size = struct.unpack_from('<I', self.data, data_start + offset)[0]
            anim_offset = data_start + offset + 4

            print(f"\nParsing animation {i} at offset {anim_offset} (block size: {block_size})")

            anim = Animation.from_bytes(self.data, anim_offset, self.bone_count)
            self.animations.append(anim)

            print(f"  Name: '{anim.name}', Keyframes: {anim.keyframe_count}")

            offset += block_size

    def _parse_skinned(self):
        """Parse the new skinned AAMF file format"""
        # Header (16 bytes):
        #   - uint32_t magic ('AAMF' or 'SAMF')
        #   - uint16_t version (2 for skinned format)
        #   - uint16_t bone_count
        #   - uint16_t anim_count
        #   - uint32_t vertex_count
        #   - uint16_t face_count
        
        magic = struct.unpack_from('<4s', self.data, 0)[0]
        self.format_version = struct.unpack_from('<H', self.data, 4)[0]
        self.bone_count = struct.unpack_from('<H', self.data, 6)[0]
        self.anim_count = struct.unpack_from('<H', self.data, 8)[0]
        vertex_count = struct.unpack_from('<I', self.data, 10)[0]
        face_count = struct.unpack_from('<H', self.data, 14)[0]
        
        print(f"Skinned AAMF Header:")
        print(f"  Magic: {magic}")
        print(f"  Version: {self.format_version}")
        print(f"  Bone count: {self.bone_count}")
        print(f"  Animation count: {self.anim_count}")
        print(f"  Vertex count: {vertex_count}")
        print(f"  Face count: {face_count}")
        
        offset = 16
        
        # Parse bone parent table
        for i in range(self.bone_count):
            bone_idx = struct.unpack_from('<H', self.data, offset)[0]
            parent_idx = struct.unpack_from('<H', self.data, offset + 2)[0]
            self.bone_parents.append((bone_idx, parent_idx))
            print(f"  Bone {bone_idx}: parent = {parent_idx}")
            offset += 4
        
        # Parse bind pose transforms
        for i in range(self.bone_count):
            bind_pose = MATRIX.from_bytes(self.data, offset)
            self.bind_poses.append(bind_pose)
            offset += 32  # MATRIX is 32 bytes
        
        # Parse unified mesh data
        # Vertices
        for i in range(vertex_count):
            v = SVECTOR.from_bytes(self.data, offset)
            self.vertices.append(v.to_float())
            offset += 8
        
        # Normals
        for i in range(vertex_count):
            n = SVECTOR.from_bytes(self.data, offset)
            self.normals.append(n.to_float())
            offset += 8
        
        # Skin weights
        for i in range(vertex_count):
            sw = SkinWeight.from_bytes(self.data, offset)
            self.skin_weights.append(sw)
            offset += 8  # 4 bytes bone indices + 4 bytes weights
        
        # Faces
        for i in range(face_count):
            v0 = struct.unpack_from('<H', self.data, offset)[0]
            v1 = struct.unpack_from('<H', self.data, offset + 2)[0]
            v2 = struct.unpack_from('<H', self.data, offset + 4)[0]
            self.faces.append([v0, v1, v2])
            offset += 6
        
        print(f"  Parsed {len(self.vertices)} vertices, {len(self.faces)} faces")
        
        # Parse animations (same format as legacy)
        for i in range(self.anim_count):
            block_size = struct.unpack_from('<I', self.data, offset)[0]
            anim_offset = offset + 4
            
            print(f"\nParsing animation {i} at offset {anim_offset} (block size: {block_size})")
            
            anim = Animation.from_bytes(self.data, anim_offset, self.bone_count)
            self.animations.append(anim)
            
            print(f"  Name: '{anim.name}', Keyframes: {anim.keyframe_count}")
            
            offset += block_size

    def export_gltf(self, filename: str):
        """Export parsed model to glTF 2.0 format with skeletal animation"""
        if self.is_skinned:
            self._export_skinned_gltf(filename)
        else:
            self._export_legacy_gltf(filename)

    def _export_legacy_gltf(self, filename: str):
        """Export legacy AAMF format to glTF"""
        if not self.bones:
            print("No bones to export!")
            return

        # Collect all geometry data per bone (for separate meshes)
        all_positions = []
        all_normals = []
        all_indices = []
        all_joints = []
        all_weights = []

        for bone_idx, bone in enumerate(self. bones):
            if not bone.vertices:
                continue

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

        def pad_to_4(data:  bytes) -> bytes:
            padding = (4 - len(data) % 4) % 4
            return data + b'\x00' * padding

        position_bytes = pad_to_4(position_bytes)
        normal_bytes = pad_to_4(normal_bytes)
        index_bytes = pad_to_4(index_bytes)
        joints_bytes = pad_to_4(joints_bytes)
        weights_bytes = pad_to_4(weights_bytes)

        # Create inverse bind matrices for skeleton (identity for bind pose)
        inverse_bind_matrices = []
        for bone in self.bones:
            inverse_bind_matrices.extend([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ])

        ibm_bytes = struct.pack(f'<{len(inverse_bind_matrices)}f', *inverse_bind_matrices)
        ibm_bytes = pad_to_4(ibm_bytes)

        # Create animation data
        animation_buffer = b''
        animations_gltf = []
        
        current_accessor = 6  # Start after mesh accessors (pos, normal, indices, joints, weights, IBM)
        current_buffer_offset = len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes) + len(weights_bytes) + len(ibm_bytes)
        
        buffer_views = []
        accessors = []
        
        # Add mesh buffer views
        buffer_views.extend([
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(normal_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes), "byteLength": len(index_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes), "byteLength": len(joints_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes), "byteLength": len(weights_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes) + len(weights_bytes), "byteLength": len(ibm_bytes)},
        ])
        
        # Add mesh accessors
        accessors.extend([
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": len(all_positions) // 3, "type": "VEC3", "min": min_pos, "max": max_pos},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": len(all_normals) // 3, "type": "VEC3"},
            {"bufferView": 2, "byteOffset":  0, "componentType": 5125, "count": len(all_indices), "type": "SCALAR"},
            {"bufferView": 3, "byteOffset":  0, "componentType": 5123, "count": len(all_joints) // 4, "type": "VEC4"},
            {"bufferView":  4, "byteOffset": 0, "componentType":  5126, "count": len(all_weights) // 4, "type": "VEC4"},
            {"bufferView":  5, "byteOffset": 0, "componentType":  5126, "count": len(self.bones), "type": "MAT4"},
        ])

        # Process animations
        for anim in self.animations:
            if anim.keyframe_count == 0:
                continue

            # Create time values (assuming 30 fps, but PS1 typically uses 60 fps for smooth animation)
            fps = 30.0
            times = [i / fps for i in range(anim.keyframe_count)]
            time_bytes = struct.pack(f'<{len(times)}f', *times)
            time_bytes = pad_to_4(time_bytes)

            # Add time buffer view and accessor
            time_buffer_view_idx = len(buffer_views)
            buffer_views.append({
                "buffer":  0,
                "byteOffset":  current_buffer_offset,
                "byteLength": len(time_bytes)
            })
            
            time_accessor_idx = len(accessors)
            accessors. append({
                "bufferView": time_buffer_view_idx,
                "byteOffset": 0,
                "componentType": 5126,
                "count": len(times),
                "type": "SCALAR",
                "min": [times[0]],
                "max": [times[-1]]
            })
            
            animation_buffer += time_bytes
            current_buffer_offset += len(time_bytes)

            samplers = []
            channels = []

            for bone_idx in range(self.bone_count):
                # Extract translations for this bone
                translations = []
                rotations = []

                for kf_idx in range(anim.keyframe_count):
                    kf = anim.keyframes[bone_idx][kf_idx]
                    trans = kf.mat. to_translation()
                    quat = kf. mat.to_quaternion()
                    
                    translations.extend(trans)
                    rotations.extend(quat)

                # Add translation data
                trans_bytes = struct. pack(f'<{len(translations)}f', *translations)
                trans_bytes = pad_to_4(trans_bytes)

                trans_buffer_view_idx = len(buffer_views)
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": current_buffer_offset,
                    "byteLength": len(trans_bytes)
                })
                
                trans_accessor_idx = len(accessors)
                accessors.append({
                    "bufferView": trans_buffer_view_idx,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": anim.keyframe_count,
                    "type": "VEC3"
                })
                
                animation_buffer += trans_bytes
                current_buffer_offset += len(trans_bytes)

                # Add rotation data
                rot_bytes = struct.pack(f'<{len(rotations)}f', *rotations)
                rot_bytes = pad_to_4(rot_bytes)

                rot_buffer_view_idx = len(buffer_views)
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": current_buffer_offset,
                    "byteLength": len(rot_bytes)
                })
                
                rot_accessor_idx = len(accessors)
                accessors.append({
                    "bufferView": rot_buffer_view_idx,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": anim. keyframe_count,
                    "type": "VEC4"
                })
                
                animation_buffer += rot_bytes
                current_buffer_offset += len(rot_bytes)

                # Add samplers for translation and rotation
                trans_sampler_idx = len(samplers)
                samplers.append({
                    "input": time_accessor_idx,
                    "output": trans_accessor_idx,
                    "interpolation": "LINEAR"
                })
                
                rot_sampler_idx = len(samplers)
                samplers. append({
                    "input": time_accessor_idx,
                    "output":  rot_accessor_idx,
                    "interpolation": "LINEAR"
                })

                # Add channels (target node is bone_idx + 2 because 0 is mesh, 1 is skeleton root)
                target_node = bone_idx + 2
                channels. append({
                    "sampler": trans_sampler_idx,
                    "target":  {"node": target_node, "path": "translation"}
                })
                channels.append({
                    "sampler": rot_sampler_idx,
                    "target": {"node": target_node, "path": "rotation"}
                })

            animations_gltf. append({
                "name": anim. name,
                "samplers": samplers,
                "channels":  channels
            })

        # Combine all buffers
        buffer_data = position_bytes + normal_bytes + index_bytes + joints_bytes + weights_bytes + ibm_bytes + animation_buffer
        buffer_base64 = base64.b64encode(buffer_data).decode('ascii')

        # Build node hierarchy
        nodes = []
        joint_indices = []

        # Root node with mesh and skin
        nodes.append({
            "name": "AAMFModel",
            "mesh": 0,
            "skin": 0
        })

        # Skeleton root node
        skeleton_root_idx = 1
        nodes. append({
            "name": "Skeleton",
            "children": []
        })

        # Add bone nodes
        bone_node_indices = {}
        for bone_idx, bone in enumerate(self. bones):
            node_idx = len(nodes)
            bone_node_indices[bone. index] = node_idx
            joint_indices.append(node_idx)

            node = {"name": f"Bone_{bone.index}"}
            nodes.append(node)

        # Set up parent-child relationships
        for bone_idx, bone in enumerate(self. bones):
            node_idx = bone_node_indices[bone.index]
            if bone.index == bone. parent_index: 
                # Root bone
                nodes[skeleton_root_idx]["children"].append(node_idx)
            else:
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
            "scenes": [{"name": "Scene", "nodes": [0, skeleton_root_idx]}],
            "nodes": nodes,
            "meshes": [{
                "name":  "AAMFMesh",
                "primitives": [{
                    "attributes":  {
                        "POSITION": 0,
                        "NORMAL": 1,
                        "JOINTS_0": 3,
                        "WEIGHTS_0":  4
                    },
                    "indices": 2,
                    "mode": 4
                }]
            }],
            "skins": [{
                "name": "AAMFSkin",
                "inverseBindMatrices": 5,
                "skeleton": skeleton_root_idx,
                "joints":  joint_indices
            }],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{
                "uri": f"data:application/octet-stream;base64,{buffer_base64}",
                "byteLength": len(buffer_data)
            }]
        }

        if animations_gltf: 
            gltf["animations"] = animations_gltf

        # Write glTF JSON file
        with open(filename, 'w') as f:
            json.dump(gltf, f, indent=2)

        print(f"\nExported to {filename}")
        print(f"  Bones: {len(self.bones)}")
        print(f"  Animations:  {len(self. animations)}")
        print(f"  Vertices: {len(all_positions) // 3}")
        print(f"  Triangles: {len(all_indices) // 3}")

    def _export_skinned_gltf(self, filename: str):
        """Export skinned AAMF format to glTF with proper vertex weights"""
        if not self.vertices:
            print("No vertices to export!")
            return
        
        # Build vertex data with proper skinning
        all_positions = []
        all_normals = []
        all_indices = []
        all_joints = []
        all_weights = []
        
        # Add all vertices with their skin weights
        for i, face in enumerate(self.faces):
            for v_idx in face:
                pos = self.vertices[v_idx]
                all_positions.extend(pos)
                
                normal = self.normals[v_idx] if v_idx < len(self.normals) else (0.0, 1.0, 0.0)
                all_normals.extend(normal)
                
                # Get skin weight for this vertex
                sw = self.skin_weights[v_idx] if v_idx < len(self.skin_weights) else SkinWeight([0, 0, 0, 0], [1.0, 0.0, 0.0, 0.0])
                all_joints.extend(sw.bone_indices)
                all_weights.extend(sw.weights)
                
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
        
        # Create inverse bind matrices from bind poses
        inverse_bind_matrices = []
        for bind_pose in self.bind_poses:
            # Convert bind pose MATRIX to inverse bind matrix
            # For simplicity, we'll use identity matrices for now
            # In production, you'd invert the bind pose matrices
            inverse_bind_matrices.extend([
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0
            ])
        
        ibm_bytes = struct.pack(f'<{len(inverse_bind_matrices)}f', *inverse_bind_matrices)
        ibm_bytes = pad_to_4(ibm_bytes)
        
        # Create animation data (same as legacy)
        animation_buffer = b''
        animations_gltf = []
        
        current_accessor = 6
        current_buffer_offset = len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes) + len(weights_bytes) + len(ibm_bytes)
        
        buffer_views = []
        accessors = []
        
        # Add mesh buffer views
        buffer_views.extend([
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(normal_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes), "byteLength": len(index_bytes), "target": 34963},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes), "byteLength": len(joints_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes), "byteLength": len(weights_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(index_bytes) + len(joints_bytes) + len(weights_bytes), "byteLength": len(ibm_bytes)},
        ])
        
        # Add mesh accessors
        accessors.extend([
            {"bufferView": 0, "byteOffset": 0, "componentType": 5126, "count": len(all_positions) // 3, "type": "VEC3", "min": min_pos, "max": max_pos},
            {"bufferView": 1, "byteOffset": 0, "componentType": 5126, "count": len(all_normals) // 3, "type": "VEC3"},
            {"bufferView": 2, "byteOffset": 0, "componentType": 5125, "count": len(all_indices), "type": "SCALAR"},
            {"bufferView": 3, "byteOffset": 0, "componentType": 5123, "count": len(all_joints) // 4, "type": "VEC4"},
            {"bufferView": 4, "byteOffset": 0, "componentType": 5126, "count": len(all_weights) // 4, "type": "VEC4"},
            {"bufferView": 5, "byteOffset": 0, "componentType": 5126, "count": self.bone_count, "type": "MAT4"},
        ])
        
        # Process animations (same as legacy)
        for anim in self.animations:
            if anim.keyframe_count == 0:
                continue
            
            fps = 30.0
            times = [i / fps for i in range(anim.keyframe_count)]
            time_bytes = struct.pack(f'<{len(times)}f', *times)
            time_bytes = pad_to_4(time_bytes)
            
            time_buffer_view_idx = len(buffer_views)
            buffer_views.append({
                "buffer": 0,
                "byteOffset": current_buffer_offset,
                "byteLength": len(time_bytes)
            })
            
            time_accessor_idx = len(accessors)
            accessors.append({
                "bufferView": time_buffer_view_idx,
                "byteOffset": 0,
                "componentType": 5126,
                "count": len(times),
                "type": "SCALAR",
                "min": [times[0]],
                "max": [times[-1]]
            })
            
            animation_buffer += time_bytes
            current_buffer_offset += len(time_bytes)
            
            samplers = []
            channels = []
            
            for bone_idx in range(self.bone_count):
                translations = []
                rotations = []
                
                for kf_idx in range(anim.keyframe_count):
                    kf = anim.keyframes[bone_idx][kf_idx]
                    trans = kf.mat.to_translation()
                    quat = kf.mat.to_quaternion()
                    
                    translations.extend(trans)
                    rotations.extend(quat)
                
                # Add translation data
                trans_bytes = struct.pack(f'<{len(translations)}f', *translations)
                trans_bytes = pad_to_4(trans_bytes)
                
                trans_buffer_view_idx = len(buffer_views)
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": current_buffer_offset,
                    "byteLength": len(trans_bytes)
                })
                
                trans_accessor_idx = len(accessors)
                accessors.append({
                    "bufferView": trans_buffer_view_idx,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": anim.keyframe_count,
                    "type": "VEC3"
                })
                
                animation_buffer += trans_bytes
                current_buffer_offset += len(trans_bytes)
                
                # Add rotation data
                rot_bytes = struct.pack(f'<{len(rotations)}f', *rotations)
                rot_bytes = pad_to_4(rot_bytes)
                
                rot_buffer_view_idx = len(buffer_views)
                buffer_views.append({
                    "buffer": 0,
                    "byteOffset": current_buffer_offset,
                    "byteLength": len(rot_bytes)
                })
                
                rot_accessor_idx = len(accessors)
                accessors.append({
                    "bufferView": rot_buffer_view_idx,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": anim.keyframe_count,
                    "type": "VEC4"
                })
                
                animation_buffer += rot_bytes
                current_buffer_offset += len(rot_bytes)
                
                # Add samplers
                trans_sampler_idx = len(samplers)
                samplers.append({
                    "input": time_accessor_idx,
                    "output": trans_accessor_idx,
                    "interpolation": "LINEAR"
                })
                
                rot_sampler_idx = len(samplers)
                samplers.append({
                    "input": time_accessor_idx,
                    "output": rot_accessor_idx,
                    "interpolation": "LINEAR"
                })
                
                # Add channels
                target_node = bone_idx + 2
                channels.append({
                    "sampler": trans_sampler_idx,
                    "target": {"node": target_node, "path": "translation"}
                })
                channels.append({
                    "sampler": rot_sampler_idx,
                    "target": {"node": target_node, "path": "rotation"}
                })
            
            animations_gltf.append({
                "name": anim.name,
                "samplers": samplers,
                "channels": channels
            })
        
        # Combine all buffers
        buffer_data = position_bytes + normal_bytes + index_bytes + joints_bytes + weights_bytes + ibm_bytes + animation_buffer
        buffer_base64 = base64.b64encode(buffer_data).decode('ascii')
        
        # Build node hierarchy
        nodes = []
        joint_indices = []
        
        # Root node with mesh and skin
        nodes.append({
            "name": "SkinnedAAMFModel",
            "mesh": 0,
            "skin": 0
        })
        
        # Skeleton root node
        skeleton_root_idx = 1
        nodes.append({
            "name": "Skeleton",
            "children": []
        })
        
        # Add bone nodes
        bone_node_indices = {}
        for bone_idx, (bone_index, parent_index) in enumerate(self.bone_parents):
            node_idx = len(nodes)
            bone_node_indices[bone_index] = node_idx
            joint_indices.append(node_idx)
            
            node = {"name": f"Bone_{bone_index}"}
            nodes.append(node)
        
        # Set up parent-child relationships
        for bone_idx, (bone_index, parent_index) in enumerate(self.bone_parents):
            node_idx = bone_node_indices[bone_index]
            if bone_index == parent_index:
                # Root bone
                nodes[skeleton_root_idx]["children"].append(node_idx)
            else:
                parent_node_idx = bone_node_indices.get(parent_index)
                if parent_node_idx is not None:
                    if "children" not in nodes[parent_node_idx]:
                        nodes[parent_node_idx]["children"] = []
                    nodes[parent_node_idx]["children"].append(node_idx)
        
        # Build glTF structure
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "aamf2gltf.py - UnnamedHB1 Skinned AAMF Converter"
            },
            "scene": 0,
            "scenes": [{"name": "Scene", "nodes": [0, skeleton_root_idx]}],
            "nodes": nodes,
            "meshes": [{
                "name": "SkinnedAAMFMesh",
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "NORMAL": 1,
                        "JOINTS_0": 3,
                        "WEIGHTS_0": 4
                    },
                    "indices": 2,
                    "mode": 4
                }]
            }],
            "skins": [{
                "name": "AAMFSkin",
                "inverseBindMatrices": 5,
                "skeleton": skeleton_root_idx,
                "joints": joint_indices
            }],
            "accessors": accessors,
            "bufferViews": buffer_views,
            "buffers": [{
                "uri": f"data:application/octet-stream;base64,{buffer_base64}",
                "byteLength": len(buffer_data)
            }]
        }
        
        if animations_gltf:
            gltf["animations"] = animations_gltf
        
        # Write glTF JSON file
        with open(filename, 'w') as f:
            json.dump(gltf, f, indent=2)
        
        print(f"\nExported skinned AAMF to {filename}")
        print(f"  Format version: {self.format_version}")
        print(f"  Bones: {self.bone_count}")
        print(f"  Animations: {len(self.animations)}")
        print(f"  Vertices: {len(all_positions) // 3}")
        print(f"  Triangles: {len(all_indices) // 3}")

    def export_glb(self, filename: str):
        """Export parsed model to GLB (binary glTF) format"""
        # For brevity, we'll create a simplified GLB export
        # First generate the gltf JSON, then convert to GLB
        
        if not self.bones:
            print("No bones to export!")
            return

        # Use the same logic as gltf but output as binary
        # ...  (similar to export_gltf but with binary output)
        
        # For now, just call gltf export with . gltf extension replaced
        gltf_filename = filename.replace('.glb', '.gltf')
        self.export_gltf(gltf_filename)
        print(f"Note: GLB export not fully implemented, saved as glTF instead:  {gltf_filename}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.aamf [output.gltf|output.glb]")
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
        output_file = os.path.splitext(input_file)[0] + ". gltf"

    with open(input_file, 'rb') as f:
        data = f.read()

    print(f"Reading {input_file} ({len(data)} bytes)")

    parser = AAMFParser(data)
    parser.parse()

    parser.export_gltf(output_file)


if __name__ == "__main__":
    main()