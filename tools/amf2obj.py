#!/usr/bin/env python3
"""
AMF to OBJ Converter for UnnamedHB1 PlayStation 1 Model Format

This script converts AMF (Animated Model Format) files from the UnnamedHB1
PS1 homebrew game to Wavefront OBJ format. 

AMF format specification derived from:
https://github.com/achrostel/UnnamedHB1

Usage:
    python amf2obj. py input.amf [output.obj]
"""

import struct
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class SVECTOR:
    """Short vector (8 bytes) - used for vertices and normals"""
    vx: int  # int16_t
    vy: int  # int16_t
    vz:  int  # int16_t
    pad:  int  # int16_t (flags:  depth check type, subdiv depth, culling mode)

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
    r:  int  # uint8_t
    g: int  # uint8_t
    b: int  # uint8_t
    cd: int  # uint8_t (code/alpha)

    @classmethod
    def from_bytes(cls, data: bytes, offset:  int = 0) -> 'CVECTOR':
        r, g, b, cd = struct.unpack_from('<BBBB', data, offset)
        return cls(r, g, b, cd)


@dataclass
class WorldBounds:
    """World bounding box (16 bytes)"""
    minX: int  # int32_t
    minZ: int  # int32_t
    maxX: int  # int32_t
    maxZ:  int  # int32_t

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'WorldBounds': 
        minX, minZ, maxX, maxZ = struct. unpack_from('<iiii', data, offset)
        return cls(minX, minZ, maxX, maxZ)


@dataclass
class AMFHeader:
    """AMF file header (24 bytes)"""
    used_textures: int  # uint32_t (highest bit for Texture* already set)
    x:  int  # uint16_t (chunk count in X)
    z: int  # uint16_t (chunk count in Z)
    bounds: WorldBounds

    @classmethod
    def from_bytes(cls, data:  bytes, offset: int = 0) -> 'AMFHeader':
        used_textures, x, z = struct.unpack_from('<IHH', data, offset)
        bounds = WorldBounds. from_bytes(data, offset + 8)
        return cls(used_textures & 0x7FFFFFFF, x, z, bounds)


@dataclass
class ChunkHeader:
    """Chunk header with polygon counts (16 bytes for counts)"""
    F4_amount: int   # uint16_t
    G4_amount: int   # uint16_t
    FT4_amount: int  # uint16_t
    GT4_amount: int  # uint16_t
    F3_amount: int   # uint16_t
    G3_amount: int   # uint16_t
    FT3_amount:  int  # uint16_t
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
# From PSn00bSDK psxgpu.h - each has a uint32_t tag prefix
POLY_F3_SIZE = 4 + 16    # tag + (r,g,b,code + 3 vertices) = 20
POLY_F4_SIZE = 4 + 20    # tag + (r,g,b,code + 4 vertices) = 24
POLY_FT3_SIZE = 4 + 28   # tag + color,code + 3*(xy + uv) + clut,tpage,pad = 32
POLY_FT4_SIZE = 4 + 36   # tag + color,code + 4*(xy + uv) + clut,tpage,pads = 40
POLY_G3_SIZE = 4 + 24    # tag + 3*(color + xy) = 28
POLY_G4_SIZE = 4 + 32    # tag + 4*(color + xy) = 36
POLY_GT3_SIZE = 4 + 36   # tag + 3*(color + xy + uv) + clut,tpage,pad = 40
POLY_GT4_SIZE = 4 + 48   # tag + 4*(color + xy + uv) + clut,tpage,pads = 52

# PF3: 3 SVECTOR (vertices) + 1 SVECTOR (normal) + POLY_F3
# = 24 + 8 + 20 = 52
PF3_SIZE = 3 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POLY_F3_SIZE

# PF4: 4 SVECTOR (vertices) + 1 SVECTOR (normal) + POLY_F4
# = 32 + 8 + 24 = 64
PF4_SIZE = 4 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POLY_F4_SIZE

# PFT3: 3 SVECTOR (vertices) + 1 SVECTOR (normal) + Texture* + POLY_FT3
# = 24 + 8 + 4 + 32 = 68
PFT3_SIZE = 3 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POINTER_SIZE + POLY_FT3_SIZE

# PFT4: 4 SVECTOR (vertices) + 1 SVECTOR (normal) + Texture* + POLY_FT4
# = 32 + 8 + 4 + 40 = 84
PFT4_SIZE = 4 * SVECTOR_SIZE + 1 * SVECTOR_SIZE + POINTER_SIZE + POLY_FT4_SIZE

# PG3: 3 SVECTOR (vertices) + 3 SVECTOR (normals) + POLY_G3
# = 24 + 24 + 28 = 76
PG3_SIZE = 3 * SVECTOR_SIZE + 3 * SVECTOR_SIZE + POLY_G3_SIZE

# PG4: 4 SVECTOR (vertices) + 4 SVECTOR (normals) + POLY_G4
# = 32 + 32 + 36 = 100
PG4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + POLY_G4_SIZE

# PGT3: 3 SVECTOR (vertices) + 3 SVECTOR (normals) + Texture* + POLY_GT3
# = 24 + 24 + 4 + 40 = 92
PGT3_SIZE = 3 * SVECTOR_SIZE + 3 * SVECTOR_SIZE + POINTER_SIZE + POLY_GT3_SIZE

# PGT4: 4 SVECTOR (vertices) + 4 SVECTOR (normals) + 4 CVECTOR (colors) + Texture* + POLY_GT4
# = 32 + 32 + 16 + 4 + 52 = 136
PGT4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + 4 * CVECTOR_SIZE + POINTER_SIZE + POLY_GT4_SIZE


class AMFParser:
    """Parser for AMF model files"""

    def __init__(self, data: bytes):
        self.data = data
        self.header:  Optional[AMFHeader] = None
        self.texture_names: List[str] = []
        self.vertices: List[Tuple[float, float, float]] = []
        self.normals: List[Tuple[float, float, float]] = []
        self.faces: List[List[int]] = []  # Each face is a list of vertex indices
        self.face_normals: List[List[int]] = []  # Normal indices per face

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

        print(f"  Texture names:  {self.texture_names}")

        # Parse chunk table and chunks
        chunk_count = self.header. x * self.header.z
        chunk_table_offset = offset

        # The chunk table contains relative offsets/info for each chunk
        # Based on the C code, the actual chunk data starts after the table
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
        # Read chunk header (first 16 bytes for counts, then 32 bytes for pointers which we skip)
        chunk_header = ChunkHeader.from_bytes(self. data, offset)

        print(f"\nChunk {chunk_idx}:")
        print(f"  F4: {chunk_header.F4_amount}, G4: {chunk_header.G4_amount}")
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
        poly_offset = self._parse_ft3_polys(poly_offset, chunk_header. FT3_amount)
        poly_offset = self._parse_gt3_polys(poly_offset, chunk_header. GT3_amount)

    def _add_vertex(self, sv: SVECTOR) -> int:
        """Add a vertex and return its 1-based index (for OBJ format)"""
        self.vertices.append(sv.to_float())
        return len(self.vertices)

    def _add_normal(self, sv: SVECTOR) -> int:
        """Add a normal and return its 1-based index (for OBJ format)"""
        self.normals.append(sv.to_float())
        return len(self.normals)

    def _parse_f4_polys(self, offset: int, count:  int) -> int:
        """Parse flat-shaded quads (PF4)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self. data, offset + 24)
            n = SVECTOR.from_bytes(self.data, offset + 32)

            # Add vertices
            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx = self._add_normal(n)

            # Add quad as two triangles (PS1 quads are v0-v1-v2-v3 in Z pattern)
            self.faces.append([idx0, idx1, idx2])
            self.faces.append([idx1, idx3, idx2])
            self.face_normals.append([n_idx, n_idx, n_idx])
            self.face_normals.append([n_idx, n_idx, n_idx])

            offset += PF4_SIZE
        return offset

    def _parse_g4_polys(self, offset: int, count: int) -> int:
        """Parse gouraud-shaded quads (PG4)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n0 = SVECTOR.from_bytes(self.data, offset + 32)
            n1 = SVECTOR.from_bytes(self. data, offset + 40)
            n2 = SVECTOR.from_bytes(self. data, offset + 48)
            n3 = SVECTOR.from_bytes(self.data, offset + 56)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx0 = self._add_normal(n0)
            n_idx1 = self._add_normal(n1)
            n_idx2 = self._add_normal(n2)
            n_idx3 = self._add_normal(n3)

            self.faces.append([idx0, idx1, idx2])
            self.faces. append([idx1, idx3, idx2])
            self.face_normals. append([n_idx0, n_idx1, n_idx2])
            self.face_normals. append([n_idx1, n_idx3, n_idx2])

            offset += PG4_SIZE
        return offset

    def _parse_ft4_polys(self, offset: int, count: int) -> int:
        """Parse flat-shaded textured quads (PFT4)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR. from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n = SVECTOR.from_bytes(self.data, offset + 32)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx = self._add_normal(n)

            self.faces.append([idx0, idx1, idx2])
            self.faces. append([idx1, idx3, idx2])
            self.face_normals. append([n_idx, n_idx, n_idx])
            self.face_normals.append([n_idx, n_idx, n_idx])

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
            # Skip 4 CVECTORs (16 bytes) and texture pointer (4 bytes)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            idx3 = self._add_vertex(v3)
            n_idx0 = self._add_normal(n0)
            n_idx1 = self._add_normal(n1)
            n_idx2 = self._add_normal(n2)
            n_idx3 = self._add_normal(n3)

            self.faces.append([idx0, idx1, idx2])
            self.faces.append([idx1, idx3, idx2])
            self.face_normals.append([n_idx0, n_idx1, n_idx2])
            self.face_normals. append([n_idx1, n_idx3, n_idx2])

            offset += PGT4_SIZE
        return offset

    def _parse_f3_polys(self, offset: int, count: int) -> int:
        """Parse flat-shaded triangles (PF3)"""
        for _ in range(count):
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            n = SVECTOR. from_bytes(self.data, offset + 24)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx = self._add_normal(n)

            self.faces.append([idx0, idx1, idx2])
            self.face_normals.append([n_idx, n_idx, n_idx])

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
            v0 = SVECTOR.from_bytes(self. data, offset)
            v1 = SVECTOR.from_bytes(self. data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            n = SVECTOR.from_bytes(self.data, offset + 24)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx = self._add_normal(n)

            self.faces.append([idx0, idx1, idx2])
            self.face_normals.append([n_idx, n_idx, n_idx])

            offset += PFT3_SIZE
        return offset

    def _parse_gt3_polys(self, offset:  int, count: int) -> int:
        """Parse gouraud-shaded textured triangles (PGT3)"""
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

    def export_obj(self, filename: str):
        """Export parsed model to OBJ format"""
        with open(filename, 'w') as f:
            f.write(f"# Converted from AMF (UnnamedHB1 PS1 Format)\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Normals: {len(self.normals)}\n")
            f.write(f"# Faces: {len(self.faces)}\n\n")

            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            f.write("\n")

            # Write normals
            for n in self.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            f.write("\n")

            # Write faces (with normals)
            for face, normals in zip(self.faces, self. face_normals):
                if len(face) == 3 and len(normals) == 3:
                    f.write(f"f {face[0]}//{normals[0]} "
                            f"{face[1]}//{normals[1]} "
                            f"{face[2]}//{normals[2]}\n")
                else:
                    # Fallback without normals
                    f.write(f"f {' '.join(map(str, face))}\n")

        print(f"\nExported to {filename}")
        print(f"  Vertices: {len(self.vertices)}")
        print(f"  Normals: {len(self.normals)}")
        print(f"  Faces: {len(self.faces)}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.amf [output.obj]")
        print("\nConverts AMF files from UnnamedHB1 PS1 homebrew to OBJ format.")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else: 
        output_file = os.path.splitext(input_file)[0] + ".obj"

    # Read input file
    with open(input_file, 'rb') as f:
        data = f.read()

    print(f"Reading {input_file} ({len(data)} bytes)")

    # Parse and convert
    parser = AMFParser(data)
    parser.parse()
    parser.export_obj(output_file)


if __name__ == "__main__": 
    main()