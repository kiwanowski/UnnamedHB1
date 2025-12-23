#!/usr/bin/env python3
"""
AMF to OBJ Converter for UnnamedHB1 PlayStation 1 Model Format

This script converts AMF (Animated Model Format) files from the UnnamedHB1
PS1 homebrew game to Wavefront OBJ format.  

AMF format specification derived from:
https://github.com/achrostel/UnnamedHB1

Usage:
    python amf2obj. py input.amf [output.obj]
    python amf2obj.py input.amf [output.obj] --debug  # Enable verbose polygon debug
"""

import struct
import sys
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Global debug flag
DEBUG_POLYGONS = False


@dataclass
class SVECTOR:
    """Short vector (8 bytes) - used for vertices and normals"""
    vx: int  # int16_t
    vy: int  # int16_t
    vz: int  # int16_t
    pad: int  # int16_t (flags:  depth check type, subdiv depth, culling mode)

    @classmethod
    def from_bytes(cls, data: bytes, offset: int = 0) -> 'SVECTOR': 
        vx, vy, vz, pad = struct.unpack_from('<hhhh', data, offset)
        return cls(vx, vy, vz, pad)

    def to_float(self, scale: float = 1.0 / 4096.0) -> Tuple[float, float, float]: 
        """Convert fixed-point to floating point coordinates"""
        return (self.vx * scale, self. vy * scale, self.vz * scale)

    def __str__(self) -> str:
        fx, fy, fz = self.to_float()
        return f"({self.vx}, {self.vy}, {self.vz}, pad={self.pad}) -> ({fx:.4f}, {fy:.4f}, {fz:.4f})"


@dataclass
class CVECTOR:
    """Color vector (4 bytes)"""
    r: int  # uint8_t
    g: int  # uint8_t
    b:  int  # uint8_t
    cd: int  # uint8_t (code/alpha)

    @classmethod
    def from_bytes(cls, data: bytes, offset:  int = 0) -> 'CVECTOR':
        r, g, b, cd = struct.unpack_from('<BBBB', data, offset)
        return cls(r, g, b, cd)

    def __str__(self) -> str:
        return f"RGBA({self.r}, {self.g}, {self.b}, {self.cd})"


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
        bounds = WorldBounds.from_bytes(data, offset + 8)
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

    def total_polygons(self) -> int:
        return (self.F4_amount + self.G4_amount + self.FT4_amount + self.GT4_amount +
                self.F3_amount + self. G3_amount + self.FT3_amount + self. GT3_amount)


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


def debug_print(*args, **kwargs):
    """Print only if debug mode is enabled"""
    if DEBUG_POLYGONS: 
        print(*args, **kwargs)


def hex_dump(data:  bytes, offset: int, length: int, prefix: str = "    ") -> str:
    """Create a hex dump of raw bytes for debugging"""
    raw = data[offset:offset + length]
    hex_str = ' '.join(f'{b:02X}' for b in raw)
    return f"{prefix}Raw bytes @ 0x{offset: 08X}: {hex_str}"


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
        
        # Debug statistics
        self.polygon_count = 0
        self.chunk_polygon_counts = {}

    def parse(self):
        """Parse the AMF file"""
        # Parse header
        self. header = AMFHeader.from_bytes(self.data, 0)
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

        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"POLYGON SUMMARY")
        print(f"{'='*60}")
        print(f"Total polygons parsed: {self.polygon_count}")
        for chunk_idx, counts in self.chunk_polygon_counts.items():
            print(f"  Chunk {chunk_idx}:  {sum(counts. values())} polygons")
            for poly_type, count in counts.items():
                if count > 0:
                    print(f"    {poly_type}: {count}")

    def _parse_chunk(self, chunk_idx: int, offset: int):
        """Parse a single chunk"""
        # Read chunk header (first 16 bytes for counts, then 32 bytes for pointers which we skip)
        chunk_header = ChunkHeader.from_bytes(self. data, offset)

        print(f"\n{'='*60}")
        print(f"CHUNK {chunk_idx} @ offset 0x{offset: 08X}")
        print(f"{'='*60}")
        print(f"  Polygon counts:")
        print(f"    F4 (flat quad):              {chunk_header. F4_amount}")
        print(f"    G4 (gouraud quad):           {chunk_header.G4_amount}")
        print(f"    FT4 (flat textured quad):    {chunk_header. FT4_amount}")
        print(f"    GT4 (gouraud textured quad): {chunk_header.GT4_amount}")
        print(f"    F3 (flat tri):               {chunk_header.F3_amount}")
        print(f"    G3 (gouraud tri):            {chunk_header.G3_amount}")
        print(f"    FT3 (flat textured tri):     {chunk_header.FT3_amount}")
        print(f"    GT3 (gouraud textured tri):  {chunk_header.GT3_amount}")
        print(f"  Total polygons in chunk: {chunk_header.total_polygons()}")

        # Initialize chunk polygon count tracking
        self.chunk_polygon_counts[chunk_idx] = {
            'F4': 0, 'G4':  0, 'FT4': 0, 'GT4': 0,
            'F3': 0, 'G3': 0, 'FT3': 0, 'GT3': 0
        }

        # Skip the 8 pointers (32 bytes) that are set at runtime
        poly_offset = offset + 16 + 32  # Header (16) + 8 pointers (32)

        # Parse each polygon type in order (matching C code)
        poly_offset = self._parse_f4_polys(poly_offset, chunk_header. F4_amount, chunk_idx)
        poly_offset = self._parse_g4_polys(poly_offset, chunk_header. G4_amount, chunk_idx)
        poly_offset = self._parse_ft4_polys(poly_offset, chunk_header.FT4_amount, chunk_idx)
        poly_offset = self._parse_gt4_polys(poly_offset, chunk_header.GT4_amount, chunk_idx)
        poly_offset = self._parse_f3_polys(poly_offset, chunk_header. F3_amount, chunk_idx)
        poly_offset = self._parse_g3_polys(poly_offset, chunk_header.G3_amount, chunk_idx)
        poly_offset = self._parse_ft3_polys(poly_offset, chunk_header.FT3_amount, chunk_idx)
        poly_offset = self._parse_gt3_polys(poly_offset, chunk_header. GT3_amount, chunk_idx)

    def _add_vertex(self, sv: SVECTOR) -> int:
        """Add a vertex and return its 1-based index (for OBJ format)"""
        self.vertices.append(sv. to_float())
        return len(self.vertices)

    def _add_normal(self, sv: SVECTOR) -> int:
        """Add a normal and return its 1-based index (for OBJ format)"""
        self.normals.append(sv.to_float())
        return len(self.normals)

    def _parse_poly_gpu_data(self, offset: int, poly_type: str, size: int) -> dict:
        """Parse GPU primitive data from POLY_* structure"""
        gpu_data = {}
        
        # Tag is first 4 bytes
        tag = struct.unpack_from('<I', self. data, offset)[0]
        gpu_data['tag'] = tag
        
        pos = offset + 4
        
        if poly_type in ('F3', 'F4'):
            # Flat shaded:  r, g, b, code, then xy pairs
            r, g, b, code = struct.unpack_from('<BBBB', self.data, pos)
            gpu_data['color'] = CVECTOR(r, g, b, code)
            pos += 4
            
            vert_count = 3 if poly_type == 'F3' else 4
            gpu_data['xy'] = []
            for i in range(vert_count):
                x, y = struct.unpack_from('<hh', self.data, pos)
                gpu_data['xy'].append((x, y))
                pos += 4
                
        elif poly_type in ('G3', 'G4'):
            # Gouraud shaded: (r,g,b,code + xy) per vertex
            vert_count = 3 if poly_type == 'G3' else 4
            gpu_data['colors'] = []
            gpu_data['xy'] = []
            for i in range(vert_count):
                r, g, b, code = struct.unpack_from('<BBBB', self.data, pos)
                gpu_data['colors'].append(CVECTOR(r, g, b, code))
                pos += 4
                x, y = struct. unpack_from('<hh', self. data, pos)
                gpu_data['xy'].append((x, y))
                pos += 4
                
        elif poly_type in ('FT3', 'FT4'):
            # Flat textured: r,g,b,code, then (xy, uv) per vertex, clut, tpage
            r, g, b, code = struct.unpack_from('<BBBB', self. data, pos)
            gpu_data['color'] = CVECTOR(r, g, b, code)
            pos += 4
            
            vert_count = 3 if poly_type == 'FT3' else 4
            gpu_data['xy'] = []
            gpu_data['uv'] = []
            for i in range(vert_count):
                x, y = struct.unpack_from('<hh', self.data, pos)
                gpu_data['xy']. append((x, y))
                pos += 4
                u, v = struct.unpack_from('<BB', self.data, pos)
                gpu_data['uv'].append((u, v))
                pos += 2
                if i == 0:
                    clut = struct.unpack_from('<H', self.data, pos)[0]
                    gpu_data['clut'] = clut
                    pos += 2
                elif i == 1:
                    tpage = struct. unpack_from('<H', self.data, pos)[0]
                    gpu_data['tpage'] = tpage
                    pos += 2
                else:
                    pos += 2  # padding
                    
        elif poly_type in ('GT3', 'GT4'):
            # Gouraud textured: (r,g,b,code, xy, uv) per vertex, clut, tpage
            vert_count = 3 if poly_type == 'GT3' else 4
            gpu_data['colors'] = []
            gpu_data['xy'] = []
            gpu_data['uv'] = []
            for i in range(vert_count):
                r, g, b, code = struct.unpack_from('<BBBB', self. data, pos)
                gpu_data['colors'].append(CVECTOR(r, g, b, code))
                pos += 4
                x, y = struct. unpack_from('<hh', self. data, pos)
                gpu_data['xy'].append((x, y))
                pos += 4
                u, v = struct.unpack_from('<BB', self.data, pos)
                gpu_data['uv'].append((u, v))
                pos += 2
                if i == 0:
                    clut = struct.unpack_from('<H', self.data, pos)[0]
                    gpu_data['clut'] = clut
                    pos += 2
                elif i == 1:
                    tpage = struct.unpack_from('<H', self.data, pos)[0]
                    gpu_data['tpage'] = tpage
                    pos += 2
                else:
                    pos += 2  # padding
        
        return gpu_data

    def _debug_polygon(self, poly_type: str, poly_idx: int, chunk_idx: int, offset: int,
                       vertices: List[SVECTOR], normals: List[SVECTOR],
                       colors:  Optional[List[CVECTOR]] = None,
                       texture_ptr: Optional[int] = None,
                       gpu_offset: Optional[int] = None,
                       gpu_size: Optional[int] = None):
        """Print detailed debug information for a polygon"""
        self.polygon_count += 1
        self.chunk_polygon_counts[chunk_idx][poly_type] += 1
        
        print(f"\n  --- {poly_type} Polygon #{poly_idx} (Global #{self.polygon_count}) ---")
        print(f"  Offset: 0x{offset: 08X}")
        
        # Vertices
        print(f"  Vertices ({len(vertices)}):")
        for i, v in enumerate(vertices):
            print(f"    V{i}: {v}")
        
        # Normals
        print(f"  Normals ({len(normals)}):")
        for i, n in enumerate(normals):
            print(f"    N{i}: {n}")
        
        # Colors (if present)
        if colors: 
            print(f"  Colors ({len(colors)}):")
            for i, c in enumerate(colors):
                print(f"    C{i}: {c}")
        
        # Texture pointer (if present)
        if texture_ptr is not None:
            print(f"  Texture pointer: 0x{texture_ptr:08X}")
            tex_idx = texture_ptr & 0x7FFFFFFF
            if tex_idx < len(self.texture_names):
                print(f"    -> Texture name: {self.texture_names[tex_idx]}")
        
        # GPU primitive data
        if gpu_offset is not None and gpu_size is not None:
            gpu_type = poly_type.replace('P', '')  # Remove 'P' prefix if any
            gpu_data = self._parse_poly_gpu_data(gpu_offset, gpu_type, gpu_size)
            print(f"  GPU Primitive (POLY_{gpu_type}) @ 0x{gpu_offset:08X}:")
            print(f"    Tag: 0x{gpu_data. get('tag', 0):08X}")
            if 'color' in gpu_data:
                print(f"    Color: {gpu_data['color']}")
            if 'colors' in gpu_data:
                for i, c in enumerate(gpu_data['colors']):
                    print(f"    Color{i}: {c}")
            if 'xy' in gpu_data:
                for i, xy in enumerate(gpu_data['xy']):
                    print(f"    XY{i}: ({xy[0]}, {xy[1]})")
            if 'uv' in gpu_data: 
                for i, uv in enumerate(gpu_data['uv']):
                    print(f"    UV{i}: ({uv[0]}, {uv[1]})")
            if 'clut' in gpu_data:
                print(f"    CLUT: 0x{gpu_data['clut']: 04X}")
            if 'tpage' in gpu_data: 
                print(f"    TPage: 0x{gpu_data['tpage']:04X}")
        
        # Raw hex dump
        if poly_type == 'F4':
            size = PF4_SIZE
        elif poly_type == 'G4': 
            size = PG4_SIZE
        elif poly_type == 'FT4':
            size = PFT4_SIZE
        elif poly_type == 'GT4':
            size = PGT4_SIZE
        elif poly_type == 'F3': 
            size = PF3_SIZE
        elif poly_type == 'G3': 
            size = PG3_SIZE
        elif poly_type == 'FT3':
            size = PFT3_SIZE
        elif poly_type == 'GT3': 
            size = PGT3_SIZE
        else:
            size = 64
        
        debug_print(hex_dump(self. data, offset, min(size, 64)))

    def _parse_f4_polys(self, offset: int, count: int, chunk_idx: int) -> int:
        """Parse flat-shaded quads (PF4)"""
        if count > 0:
            print(f"\n  Parsing {count} F4 (flat-shaded quad) polygons...")
        
        for i in range(count):
            start_offset = offset
            v0 = SVECTOR. from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self. data, offset + 24)
            n = SVECTOR.from_bytes(self. data, offset + 32)
            
            gpu_offset = offset + 40  # After 4 vertices + 1 normal

            # Debug output
            self._debug_polygon('F4', i, chunk_idx, start_offset,
                              vertices=[v0, v1, v2, v3],
                              normals=[n],
                              gpu_offset=gpu_offset,
                              gpu_size=POLY_F4_SIZE)

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

    def _parse_g4_polys(self, offset: int, count: int, chunk_idx: int) -> int:
        """Parse gouraud-shaded quads (PG4)"""
        if count > 0:
            print(f"\n  Parsing {count} G4 (gouraud-shaded quad) polygons...")
        
        for i in range(count):
            start_offset = offset
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR. from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n0 = SVECTOR.from_bytes(self.data, offset + 32)
            n1 = SVECTOR.from_bytes(self. data, offset + 40)
            n2 = SVECTOR.from_bytes(self.data, offset + 48)
            n3 = SVECTOR. from_bytes(self.data, offset + 56)
            
            gpu_offset = offset + 64  # After 4 vertices + 4 normals

            # Debug output
            self._debug_polygon('G4', i, chunk_idx, start_offset,
                              vertices=[v0, v1, v2, v3],
                              normals=[n0, n1, n2, n3],
                              gpu_offset=gpu_offset,
                              gpu_size=POLY_G4_SIZE)

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
            self.face_normals.append([n_idx1, n_idx3, n_idx2])

            offset += PG4_SIZE
        return offset

    def _parse_ft4_polys(self, offset: int, count: int, chunk_idx: int) -> int:
        """Parse flat-shaded textured quads (PFT4)"""
        if count > 0:
            print(f"\n  Parsing {count} FT4 (flat-shaded textured quad) polygons...")
        
        for i in range(count):
            start_offset = offset
            v0 = SVECTOR. from_bytes(self.data, offset)
            v1 = SVECTOR. from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self.data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n = SVECTOR.from_bytes(self.data, offset + 32)
            
            # Texture pointer
            tex_ptr = struct.unpack_from('<I', self. data, offset + 40)[0]
            gpu_offset = offset + 44  # After 4 vertices + 1 normal + texture ptr

            # Debug output
            self._debug_polygon('FT4', i, chunk_idx, start_offset,
                              vertices=[v0, v1, v2, v3],
                              normals=[n],
                              texture_ptr=tex_ptr,
                              gpu_offset=gpu_offset,
                              gpu_size=POLY_FT4_SIZE)

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

    def _parse_gt4_polys(self, offset: int, count: int, chunk_idx: int) -> int:
        """Parse gouraud-shaded textured quads (PGT4)"""
        if count > 0:
            print(f"\n  Parsing {count} GT4 (gouraud-shaded textured quad) polygons...")
        
        for i in range(count):
            start_offset = offset
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            v3 = SVECTOR.from_bytes(self.data, offset + 24)
            n0 = SVECTOR. from_bytes(self.data, offset + 32)
            n1 = SVECTOR.from_bytes(self.data, offset + 40)
            n2 = SVECTOR.from_bytes(self.data, offset + 48)
            n3 = SVECTOR.from_bytes(self. data, offset + 56)
            
            # Colors
            c0 = CVECTOR.from_bytes(self.data, offset + 64)
            c1 = CVECTOR. from_bytes(self.data, offset + 68)
            c2 = CVECTOR.from_bytes(self.data, offset + 72)
            c3 = CVECTOR.from_bytes(self. data, offset + 76)
            
            # Texture pointer
            tex_ptr = struct.unpack_from('<I', self. data, offset + 80)[0]
            gpu_offset = offset + 84  # After vertices, normals, colors, texture ptr

            # Debug output
            self._debug_polygon('GT4', i, chunk_idx, start_offset,
                              vertices=[v0, v1, v2, v3],
                              normals=[n0, n1, n2, n3],
                              colors=[c0, c1, c2, c3],
                              texture_ptr=tex_ptr,
                              gpu_offset=gpu_offset,
                              gpu_size=POLY_GT4_SIZE)

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
            self.face_normals.append([n_idx1, n_idx3, n_idx2])

            offset += PGT4_SIZE
        return offset

    def _parse_f3_polys(self, offset: int, count: int, chunk_idx: int) -> int:
        """Parse flat-shaded triangles (PF3)"""
        if count > 0:
            print(f"\n  Parsing {count} F3 (flat-shaded triangle) polygons...")
        
        for i in range(count):
            start_offset = offset
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR. from_bytes(self.data, offset + 16)
            n = SVECTOR.from_bytes(self.data, offset + 24)
            
            gpu_offset = offset + 32  # After 3 vertices + 1 normal

            # Debug output
            self._debug_polygon('F3', i, chunk_idx, start_offset,
                              vertices=[v0, v1, v2],
                              normals=[n],
                              gpu_offset=gpu_offset,
                              gpu_size=POLY_F3_SIZE)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx = self._add_normal(n)

            self.faces.append([idx0, idx1, idx2])
            self.face_normals.append([n_idx, n_idx, n_idx])

            offset += PF3_SIZE
        return offset

    def _parse_g3_polys(self, offset: int, count: int, chunk_idx: int) -> int:
        """Parse gouraud-shaded triangles (PG3)"""
        if count > 0:
            print(f"\n  Parsing {count} G3 (gouraud-shaded triangle) polygons...")
        
        for i in range(count):
            start_offset = offset
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            n0 = SVECTOR.from_bytes(self.data, offset + 24)
            n1 = SVECTOR.from_bytes(self. data, offset + 32)
            n2 = SVECTOR.from_bytes(self.data, offset + 40)
            
            gpu_offset = offset + 48  # After 3 vertices + 3 normals

            # Debug output
            self._debug_polygon('G3', i, chunk_idx, start_offset,
                              vertices=[v0, v1, v2],
                              normals=[n0, n1, n2],
                              gpu_offset=gpu_offset,
                              gpu_size=POLY_G3_SIZE)

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

    def _parse_ft3_polys(self, offset: int, count: int, chunk_idx: int) -> int:
        """Parse flat-shaded textured triangles (PFT3)"""
        if count > 0:
            print(f"\n  Parsing {count} FT3 (flat-shaded textured triangle) polygons...")
        
        for i in range(count):
            start_offset = offset
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            n = SVECTOR. from_bytes(self.data, offset + 24)
            
            # Texture pointer
            tex_ptr = struct.unpack_from('<I', self.data, offset + 32)[0]
            gpu_offset = offset + 36  # After 3 vertices + 1 normal + texture ptr

            # Debug output
            self._debug_polygon('FT3', i, chunk_idx, start_offset,
                              vertices=[v0, v1, v2],
                              normals=[n],
                              texture_ptr=tex_ptr,
                              gpu_offset=gpu_offset,
                              gpu_size=POLY_FT3_SIZE)

            idx0 = self._add_vertex(v0)
            idx1 = self._add_vertex(v1)
            idx2 = self._add_vertex(v2)
            n_idx = self._add_normal(n)

            self.faces. append([idx0, idx1, idx2])
            self.face_normals. append([n_idx, n_idx, n_idx])

            offset += PFT3_SIZE
        return offset

    def _parse_gt3_polys(self, offset: int, count: int, chunk_idx: int) -> int:
        """Parse gouraud-shaded textured triangles (PGT3)"""
        if count > 0:
            print(f"\n  Parsing {count} GT3 (gouraud-shaded textured triangle) polygons...")
        
        for i in range(count):
            start_offset = offset
            v0 = SVECTOR.from_bytes(self.data, offset)
            v1 = SVECTOR.from_bytes(self.data, offset + 8)
            v2 = SVECTOR.from_bytes(self. data, offset + 16)
            n0 = SVECTOR.from_bytes(self.data, offset + 24)
            n1 = SVECTOR. from_bytes(self.data, offset + 32)
            n2 = SVECTOR.from_bytes(self.data, offset + 40)
            
            # Texture pointer
            tex_ptr = struct. unpack_from('<I', self.data, offset + 48)[0]
            gpu_offset = offset + 52  # After 3 vertices + 3 normals + texture ptr

            # Debug output
            self._debug_polygon('GT3', i, chunk_idx, start_offset,
                              vertices=[v0, v1, v2],
                              normals=[n0, n1, n2],
                              texture_ptr=tex_ptr,
                              gpu_offset=gpu_offset,
                              gpu_size=POLY_GT3_SIZE)

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
                f. write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

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
        print(f"  Faces:  {len(self. faces)}")


def main():
    global DEBUG_POLYGONS
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.amf [output.obj] [--debug]")
        print("\nConverts AMF files from UnnamedHB1 PS1 homebrew to OBJ format.")
        print("\nOptions:")
        print("  --debug    Enable verbose hex dump output for each polygon")
        sys.exit(1)

    # Check for debug flag
    if '--debug' in sys.argv:
        DEBUG_POLYGONS = True
        sys.argv.remove('--debug')
        print("Debug mode enabled:  verbose polygon output\n")

    input_file = sys. argv[1]

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