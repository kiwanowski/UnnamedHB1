#!/usr/bin/env python3
"""
AMF to glTF Converter for UnnamedHB1 PlayStation 1 Model Format

This script converts AMF (Animated Model Format) files from the UnnamedHB1
PS1 homebrew game to glTF 2.0 format. 

AMF format specification derived from:
https://github.com/achrostel/UnnamedHB1

Usage:
    python amf2gltf.py input.amf [output.gltf]

Dependencies:
    pip install pygltflib
"""

import struct
import sys
import os
import json
import base64
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class SVECTOR: 
    """Short vector (8 bytes) - used for vertices and normals"""
    vx: int  # int16_t
    vy: int  # int16_t
    vz:  int  # int16_t
    pad: int  # int16_t (flags: depth check type, subdiv depth, culling mode)

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'SVECTOR': 
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
    cd: int  # uint8_t (code/alpha)

    @classmethod
    def from_bytes(cls, data: bytes, offset:  int = 0) -> 'CVECTOR':
        r, g, b, cd = struct.unpack_from('<BBBB', data, offset)
        return cls(r, g, b, cd)

    def to_float(self) -> Tuple[float, float, float, float]:
        """Convert to normalized float RGBA"""
        return (self.r / 255.0, self. g / 255.0, self.b / 255.0, 1.0)


@dataclass
class WorldBounds:
    """World bounding box (16 bytes)"""
    minX: int  # int32_t
    minZ:  int  # int32_t
    maxX: int  # int32_t
    maxZ: int  # int32_t

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'WorldBounds': 
        minX, minZ, maxX, maxZ = struct. unpack_from('<iiii', data, offset)
        return cls(minX, minZ, maxX, maxZ)


@dataclass
class AMFHeader:
    """AMF file header (24 bytes)"""
    used_textures: int  # uint32_t (highest bit for Texture* already set)
    x: int  # uint16_t (chunk count in X)
    z: int  # uint16_t (chunk count in Z)
    bounds: WorldBounds

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'AMFHeader':
        used_textures, x, z = struct.unpack_from('<IHH', data, offset)
        bounds = WorldBounds.from_bytes(data, offset + 8)
        return cls(used_textures & 0x7FFFFFFF, x, z, bounds)


@dataclass
class ChunkHeader:
    """Chunk header with polygon counts (16 bytes for counts)"""
    F4_amount: int   # uint16_t
    G4_amount: int   # uint16_t
    FT4_amount: int  # uint16_t
    GT4_amount:  int  # uint16_t
    F3_amount: int   # uint16_t
    G3_amount: int   # uint16_t
    FT3_amount: int  # uint16_t
    GT3_amount: int  # uint16_t

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'ChunkHeader':
        counts = struct.unpack_from('<HHHHHHHH', data, offset)
        return cls(*counts)


# Structure sizes based on PSn00bSDK and UnnamedHB1 definitions
SVECTOR_SIZE = 8
CVECTOR_SIZE = 4
POINTER_SIZE = 4

# POLY_* GPU primitive sizes (with 4-byte tag included)
POLY_F3_SIZE = 4 + 16    # tag + (r,g,b,code + 3 vertices) = 20
POLY_F4_SIZE = 4 + 20    # tag + (r,g,b,code + 4 vertices) = 24
POLY_FT3_SIZE = 4 + 28   # tag + color,code + 3*(xy + uv) + clut,tpage,pad = 32
POLY_FT4_SIZE = 4 + 36   # tag + color,code + 4*(xy + uv) + clut,tpage,pads = 40
POLY_G3_SIZE = 4 + 24    # tag + 3*(color + xy) = 28
POLY_G4_SIZE = 4 + 32    # tag + 4*(color + xy) = 36
POLY_GT3_SIZE = 4 + 36   # tag + 3*(color + xy + uv) + clut,tpage,pad = 40
POLY_GT4_SIZE = 4 + 48   # tag + 4*(color + xy + uv) + clut,tpage,pads = 52

# Polygon structure sizes
PF3_SIZE = 3 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POLY_F3_SIZE  # 52
PF4_SIZE = 4 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POLY_F4_SIZE  # 64
PFT3_SIZE = 3 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POINTER_SIZE + POLY_FT3_SIZE  # 68
PFT4_SIZE = 4 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POINTER_SIZE + POLY_FT4_SIZE  # 84
PG3_SIZE = 3 * SVECTOR_SIZE + 3 * SVECTOR_SIZE + POLY_G3_SIZE  # 76
PG4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + POLY_G4_SIZE  # 100
PGT3_SIZE = 3 * SVECTOR_SIZE + 3 * SVECTOR_SIZE + POINTER_SIZE + POLY_GT3_SIZE  # 92
PGT4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + 4 * CVECTOR_SIZE + POINTER_SIZE + POLY_GT4_SIZE  # 136


@dataclass
class ParsedVertex:
    """A vertex with position, normal, and optional color"""
    position:  Tuple[float, float, float]
    normal: Tuple[float, float, float]
    color:  Optional[Tuple[float, float, float, float]] = None


@dataclass
class ParsedFace:
    """A triangle face with vertex indices"""
    indices: List[int]  # 3 vertex indices


class AMFParser: 
    """Parser for AMF model files"""

    def __init__(self, data: bytes):
        self.data = data
        self.header:  Optional[AMFHeader] = None
        self.texture_names: List[str] = []
        self.vertices: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.colors: List[Tuple[float, float, float, float]] = []
        self.faces: List[List[int]] = []  # Each face is a list of vertex indices
        self.face_normals: List[List[int]] = []  # Normal indices per face
        self. face_colors: List[List[int]] = []  # Color indices per face (optional)

    def parse(self):
        """Parse the AMF file"""
        # Parse header
        self.header = AMFHeader.from_bytes(self.data, 0)
        print(f"AMF Header:")
        print(f"  Textures: {self. header.used_textures}")
        print(f"  Chunks: {self.header.x} x {self.header.z}")
        print(f"  Bounds: ({self.header.bounds.minX}, {self.header.bounds.minZ}) to "
              f"({self.header.bounds.maxX}, {self.header.bounds.maxZ})")

        # Parse texture names (8 bytes each)
        offset = 8 + 16  # Header size + WorldBounds size
        for i in range(self.header.used_textures):
            name_bytes = self.data[offset:offset + 8]
            name = name_bytes. rstrip(b'\x00').decode('ascii', errors='replace')
            self.texture_names.append(name)
            offset += 8

        print(f"  Texture names: {self.texture_names}")

        # Parse chunk table and chunks
        chunk_count = self.header. x * self.header.z
        chunk_table_offset = offset

        # The chunk table contains relative offsets/info for each chunk
        chunk_data_offset = chunk_table_offset + chunk_count * 4

        for chunk_idx in range(chunk_count):
            self._parse_chunk(chunk_idx, chunk_data_offset)
            # Calculate offset for next chunk based on current chunk's content
            chunk_header = ChunkHeader.from_bytes(self. data, chunk_data_offset)
            chunk_size = 16 + 32  # header (16) + 8 pointers (32)
            chunk_size += chunk_header. F4_amount * PF4_SIZE
            chunk_size += chunk_header.G4_amount * PG4_SIZE
            chunk_size += chunk_header.FT4_amount * PFT4_SIZE
            chunk_size += chunk_header.GT4_amount * PGT4_SIZE
            chunk_size += chunk_header.F3_amount * PF3_SIZE
            chunk_size += chunk_header.G3_amount * PG3_SIZE
            chunk_size += chunk_header.FT3_amount * PFT3_SIZE
            chunk_size += chunk_header.GT3_amount * PGT3_SIZE
            chunk_data_offset += chunk_size

    def _parse_chunk(self, chunk_idx: int, offset: int):
        """Parse a single chunk"""
        chunk_header = ChunkHeader. from_bytes(self.data, offset)

        print(f"\nChunk {chunk_idx}:")
        print(f"  F4: {chunk_header. F4_amount}, G4: {chunk_header.G4_amount}")
        print(f"  FT4: {chunk_header.FT4_amount}, GT4: {chunk_header. GT4_amount}")
        print(f"  F3: {chunk_header. F3_amount}, G3: {chunk_header.G3_amount}")
        print(f"  FT3: {chunk_header.FT3_amount}, GT3: {chunk_header.GT3_amount}")

        # Skip the 8 pointers (32 bytes) that are set at runtime
        poly_offset = offset + 16 + 32  # Header (16) + 8 pointers (32)

        # Parse each polygon type in order (matching C code)
        poly_offset = self._parse_f4_polys(poly_offset, chunk_header. F4_amount)
        poly_offset = self._parse_g4_polys(poly_offset, chunk_header.G4_amount)
        poly_offset = self._parse_ft4_polys(poly_offset, chunk_header.FT4_amount)
        poly_offset = self._parse_gt4_polys(poly_offset, chunk_header.GT4_amount)
        poly_offset = self._parse_f3_polys(poly_offset, chunk_header.F3_amount)
        poly_offset = self._parse_g3_polys(poly_offset, chunk_header.G3_amount)
        poly_offset = self._parse_ft3_polys(poly_offset, chunk_header.FT3_amount)
        poly_offset = self._parse_gt3_polys(poly_offset, chunk_header.GT3_amount)

    def _add_vertex(self, sv: SVECTOR) -> int:
        """Add a vertex and return its 0-based index"""
        self.vertices.append(sv.to_float())
        return len(self.vertices) - 1

    def _add_normal(self, sv: SVECTOR) -> int:
        """Add a normal and return its 0-based index"""
        self.normals. append(sv.to_float())
        return len(self. normals) - 1

    def _add_color(self, cv: CVECTOR) -> int:
        """Add a color and return its 0-based index"""
        self.colors.append(cv. to_float())
        return len(self.colors) - 1

    def _parse_f4_polys(self, offset: int, count:  int) -> int:
        """Parse flat-shaded quads (PF4)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self. data, offset + 24)
            n = SVECTOR.from_bytes(self.data, offset + 32)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx = self._add_normal(n)

            # Duplicate normal for each vertex
            self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            # Add quad as two triangles (PS1 quads are v0-v1-v2-v3 in Z pattern)
            self.faces.append([idx0, idx1, idx2])
            self.faces.append([idx1, idx3, idx2])
            self.face_normals.append([n_idx, n_idx + 1, n_idx + 2])
            self.face_normals.append([n_idx + 1, n_idx + 3, n_idx + 2])

            offset += PF4_SIZE
        return offset

    def _parse_g4_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded quads (PG4)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self. data, offset)
            v1 = SVECTOR.from_bytes(self. data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR. from_bytes(self.data, offset + 24)
            n0 = SVECTOR.from_bytes(self.data, offset + 32)
            n1 = SVECTOR.from_bytes(self. data, offset + 40)
            n2 = SVECTOR.from_bytes(self.data, offset + 48)
            n3 = SVECTOR.from_bytes(self. data, offset + 56)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx0 = self._add_normal(n0)
            n_idx1 = self._add_normal(n1)
            n_idx2 = self._add_normal(n2)
            n_idx3 = self._add_normal(n3)

            self. faces.append([idx0, idx1, idx2])
            self.faces.append([idx1, idx3, idx2])
            self.face_normals.append([n_idx0, n_idx1, n_idx2])
            self.face_normals. append([n_idx1, n_idx3, n_idx2])

            offset += PG4_SIZE
        return offset

    def _parse_ft4_polys(self, offset: int, count: int) -> int:
        """Parse flat-shaded textured quads (PFT4)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self. data, offset + 24)
            n = SVECTOR.from_bytes(self.data, offset + 32)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx = self._add_normal(n)

            self._add_normal(n)
            self._add_normal(n)
            self._add_normal(n)

            self.faces.append([idx0, idx1, idx2])
            self.faces. append([idx1, idx3, idx2])
            self.face_normals. append([n_idx, n_idx + 1, n_idx + 2])
            self.face_normals.append([n_idx + 1, n_idx + 3, n_idx + 2])

            offset += PFT4_SIZE
        return offset

    def _parse_gt4_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded textured quads (PGT4)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n0 = SVECTOR. from_bytes(self.data, offset + 32)
            n1 = SVECTOR.from_bytes(self.data, offset + 40)
            n2 = SVECTOR.from_bytes(self.data, offset + 48)
            n3 = SVECTOR.from_bytes(self. data, offset + 56)
            c0 = CVECTOR.from_bytes(self.data, offset + 64)
            c1 = CVECTOR. from_bytes(self.data, offset + 68)
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

    def _parse_f3_polys(self, offset:  int, count: int) -> int:
        """Parse flat-shaded triangles (PF3)"""
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

            self.faces. append([idx0, idx1, idx2])
            self.face_normals. append([n_idx, n_idx + 1, n_idx + 2])

            offset += PF3_SIZE
        return offset

    def _parse_g3_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded triangles (PG3)"""
        for _ in range(count):
            v0 = SVECTOR. from_bytes(self.data, offset)
            v1 = SVECTOR. from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n0 = SVECTOR.from_bytes(self.data, offset + 24)
            n1 = SVECTOR.from_bytes(self. data, offset + 32)
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
        """Parse flat-shaded textured triangles (PFT3)"""
        for _ in range(count):
            v0 = SVECTOR. from_bytes(self.data, offset)
            v1 = SVECTOR. from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n = SVECTOR.from_bytes(self. data, offset + 24)

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

    def _parse_gt3_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded textured triangles (PGT3)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            n0 = SVECTOR.from_bytes(self.data, offset + 24)
            n1 = SVECTOR. from_bytes(self.data, offset + 32)
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

    def export_gltf(self, filename: str):
        """Export parsed model to glTF 2.0 format"""
        if not self.vertices:
            print("No vertices to export!")
            return

        # Build interleaved vertex data (position + normal per vertex)
        # For glTF, we need to flatten the indexed mesh into a non-indexed format
        # since our normals are per-face-vertex, not per-vertex

        positions = []
        normals = []
        indices = []

        vertex_index = 0
        for face_idx, face in enumerate(self.faces):
            face_normal_indices = self.face_normals[face_idx] if face_idx < len(self.face_normals) else [0, 0, 0]

            for i, v_idx in enumerate(face):
                pos = self.vertices[v_idx]
                positions.extend(pos)

                n_idx = face_normal_indices[i] if i < len(face_normal_indices) else 0
                if n_idx < len(self.normals):
                    normal = self.normals[n_idx]
                else:
                    normal = (0.0, 1.0, 0.0)  # Default up normal
                normals.extend(normal)

                indices.append(vertex_index)
                vertex_index += 1

        # Calculate bounds
        min_pos = [float('inf'), float('inf'), float('inf')]
        max_pos = [float('-inf'), float('-inf'), float('-inf')]

        for i in range(0, len(positions), 3):
            for j in range(3):
                min_pos[j] = min(min_pos[j], positions[i + j])
                max_pos[j] = max(max_pos[j], positions[i + j])

        # Create binary buffer
        position_bytes = struct.pack(f'<{len(positions)}f', *positions)
        normal_bytes = struct.pack(f'<{len(normals)}f', *normals)
        index_bytes = struct.pack(f'<{len(indices)}I', *indices)

        # Pad to 4-byte alignment
        def pad_to_4(data:  bytes) -> bytes:
            padding = (4 - len(data) % 4) % 4
            return data + b'\x00' * padding

        position_bytes = pad_to_4(position_bytes)
        normal_bytes = pad_to_4(normal_bytes)
        index_bytes = pad_to_4(index_bytes)

        # Combine all buffers
        buffer_data = position_bytes + normal_bytes + index_bytes
        buffer_base64 = base64.b64encode(buffer_data).decode('ascii')

        # Build glTF structure
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "amf2gltf. py - UnnamedHB1 AMF Converter"
            },
            "scene": 0,
            "scenes": [
                {
                    "name": "Scene",
                    "nodes": [0]
                }
            ],
            "nodes": [
                {
                    "name": "AMFModel",
                    "mesh": 0
                }
            ],
            "meshes": [
                {
                    "name": "AMFMesh",
                    "primitives": [
                        {
                            "attributes":  {
                                "POSITION": 0,
                                "NORMAL": 1
                            },
                            "indices": 2,
                            "mode": 4  # TRIANGLES
                        }
                    ]
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,  # FLOAT
                    "count": len(positions) // 3,
                    "type": "VEC3",
                    "min": min_pos,
                    "max": max_pos
                },
                {
                    "bufferView": 1,
                    "byteOffset": 0,
                    "componentType": 5126,  # FLOAT
                    "count": len(normals) // 3,
                    "type": "VEC3"
                },
                {
                    "bufferView": 2,
                    "byteOffset":  0,
                    "componentType":  5125,  # UNSIGNED_INT
                    "count": len(indices),
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset": 0,
                    "byteLength": len(position_bytes),
                    "target": 34962  # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset": len(position_bytes),
                    "byteLength": len(normal_bytes),
                    "target": 34962  # ARRAY_BUFFER
                },
                {
                    "buffer": 0,
                    "byteOffset":  len(position_bytes) + len(normal_bytes),
                    "byteLength": len(index_bytes),
                    "target": 34963  # ELEMENT_ARRAY_BUFFER
                }
            ],
            "buffers": [
                {
                    "uri": f"data:application/octet-stream;base64,{buffer_base64}",
                    "byteLength": len(buffer_data)
                }
            ]
        }

        # Write glTF JSON file
        with open(filename, 'w') as f:
            json.dump(gltf, f, indent=2)

        print(f"\nExported to {filename}")
        print(f"  Vertices: {len(positions) // 3}")
        print(f"  Normals:  {len(normals) // 3}")
        print(f"  Triangles: {len(indices) // 3}")

    def export_glb(self, filename: str):
        """Export parsed model to GLB (binary glTF) format"""
        if not self.vertices:
            print("No vertices to export!")
            return

        # Build interleaved vertex data
        positions = []
        normals = []
        indices = []

        vertex_index = 0
        for face_idx, face in enumerate(self.faces):
            face_normal_indices = self.face_normals[face_idx] if face_idx < len(self.face_normals) else [0, 0, 0]

            for i, v_idx in enumerate(face):
                pos = self. vertices[v_idx]
                positions. extend(pos)

                n_idx = face_normal_indices[i] if i < len(face_normal_indices) else 0
                if n_idx < len(self.normals):
                    normal = self.normals[n_idx]
                else: 
                    normal = (0.0, 1.0, 0.0)
                normals.extend(normal)

                indices.append(vertex_index)
                vertex_index += 1

        # Calculate bounds
        min_pos = [float('inf'), float('inf'), float('inf')]
        max_pos = [float('-inf'), float('-inf'), float('-inf')]

        for i in range(0, len(positions), 3):
            for j in range(3):
                min_pos[j] = min(min_pos[j], positions[i + j])
                max_pos[j] = max(max_pos[j], positions[i + j])

        # Create binary buffer
        position_bytes = struct.pack(f'<{len(positions)}f', *positions)
        normal_bytes = struct.pack(f'<{len(normals)}f', *normals)
        index_bytes = struct.pack(f'<{len(indices)}I', *indices)

        # Pad to 4-byte alignment
        def pad_to_4(data: bytes) -> bytes:
            padding = (4 - len(data) % 4) % 4
            return data + b'\x00' * padding

        position_bytes_padded = pad_to_4(position_bytes)
        normal_bytes_padded = pad_to_4(normal_bytes)
        index_bytes_padded = pad_to_4(index_bytes)

        buffer_data = position_bytes_padded + normal_bytes_padded + index_bytes_padded

        # Build glTF structure (without embedded buffer URI)
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "amf2gltf.py - UnnamedHB1 AMF Converter"
            },
            "scene": 0,
            "scenes": [
                {
                    "name": "Scene",
                    "nodes": [0]
                }
            ],
            "nodes":  [
                {
                    "name":  "AMFModel",
                    "mesh": 0
                }
            ],
            "meshes":  [
                {
                    "name":  "AMFMesh",
                    "primitives": [
                        {
                            "attributes": {
                                "POSITION": 0,
                                "NORMAL": 1
                            },
                            "indices": 2,
                            "mode": 4
                        }
                    ]
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "byteOffset": 0,
                    "componentType": 5126,
                    "count": len(positions) // 3,
                    "type": "VEC3",
                    "min": min_pos,
                    "max": max_pos
                },
                {
                    "bufferView": 1,
                    "byteOffset":  0,
                    "componentType":  5126,
                    "count":  len(normals) // 3,
                    "type":  "VEC3"
                },
                {
                    "bufferView": 2,
                    "byteOffset":  0,
                    "componentType":  5125,
                    "count":  len(indices),
                    "type": "SCALAR"
                }
            ],
            "bufferViews": [
                {
                    "buffer": 0,
                    "byteOffset":  0,
                    "byteLength": len(position_bytes_padded),
                    "target": 34962
                },
                {
                    "buffer": 0,
                    "byteOffset": len(position_bytes_padded),
                    "byteLength": len(normal_bytes_padded),
                    "target": 34962
                },
                {
                    "buffer": 0,
                    "byteOffset":  len(position_bytes_padded) + len(normal_bytes_padded),
                    "byteLength": len(index_bytes_padded),
                    "target": 34963
                }
            ],
            "buffers": [
                {
                    "byteLength": len(buffer_data)
                }
            ]
        }

        # Serialize JSON
        json_str = json.dumps(gltf, separators=(',', ': '))
        json_bytes = json_str.encode('utf-8')

        # Pad JSON to 4-byte alignment
        json_padding = (4 - len(json_bytes) % 4) % 4
        json_bytes_padded = json_bytes + b' ' * json_padding

        # Pad binary buffer to 4-byte alignment
        bin_padding = (4 - len(buffer_data) % 4) % 4
        buffer_data_padded = buffer_data + b'\x00' * bin_padding

        # GLB structure: 
        # Header (12 bytes): magic, version, length
        # JSON chunk:  length, type, data
        # BIN chunk: length, type, data

        header_length = 12
        json_chunk_length = 8 + len(json_bytes_padded)
        bin_chunk_length = 8 + len(buffer_data_padded)
        total_length = header_length + json_chunk_length + bin_chunk_length

        with open(filename, 'wb') as f:
            # Header
            f.write(struct.pack('<I', 0x46546C67))  # magic "glTF"
            f.write(struct.pack('<I', 2))  # version
            f.write(struct.pack('<I', total_length))  # total length

            # JSON chunk
            f.write(struct.pack('<I', len(json_bytes_padded)))  # chunk length
            f. write(struct.pack('<I', 0x4E4F534A))  # chunk type "JSON"
            f.write(json_bytes_padded)

            # BIN chunk
            f.write(struct.pack('<I', len(buffer_data_padded)))  # chunk length
            f.write(struct.pack('<I', 0x004E4942))  # chunk type "BIN\0"
            f. write(buffer_data_padded)

        print(f"\nExported to {filename}")
        print(f"  Vertices:  {len(positions) // 3}")
        print(f"  Normals:  {len(normals) // 3}")
        print(f"  Triangles: {len(indices) // 3}")
        print(f"  File size: {total_length} bytes")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input. amf [output. gltf|output.glb]")
        print("\nConverts AMF files from UnnamedHB1 PS1 homebrew to glTF 2.0 format.")
        print("\nOutput format is determined by file extension:")
        print("  .gltf - JSON format with embedded base64 buffer")
        print("  .glb  - Binary glTF format (default)")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = os.path.splitext(input_file)[0] + ".glb"

    # Read input file
    with open(input_file, 'rb') as f:
        data = f.read()

    print(f"Reading {input_file} ({len(data)} bytes)")

    # Parse and convert
    parser = AMFParser(data)
    parser.parse()

    # Export based on extension
    if output_file.lower().endswith('.gltf'):
        parser.export_gltf(output_file)
    else:
        parser. export_glb(output_file)


if __name__ == "__main__": 
    main()