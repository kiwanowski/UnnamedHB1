#!/usr/bin/env python3
"""
glTF to AMF Converter for UnnamedHB1 PlayStation 1 Model Format

This script converts glTF 2.0 files to AMF (Animated Model Format) files
for the UnnamedHB1 PS1 homebrew game.

Only G4 (gouraud-shaded quads) polygons are supported. 
Triangles are automatically merged into quads where possible. 

AMF format specification derived from:
https://github.com/achrostel/UnnamedHB1

Usage:
    python gltf2amf.py input.gltf [output.amf]
    python gltf2amf.py input.glb [output.amf]

Dependencies:
    None (uses only standard library)
"""

import struct
import sys
import os
import json
import base64
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Set
from collections import defaultdict


# Structure sizes
SVECTOR_SIZE = 8
CVECTOR_SIZE = 4
POINTER_SIZE = 4

# POLY_G4 GPU primitive size (with 4-byte tag included)
POLY_G4_SIZE = 4 + 32  # tag + 4*(color + xy) = 36

# PG4 structure size:  4 vertices + 4 normals + POLY_G4
PG4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + POLY_G4_SIZE  # 100

# Fixed-point scale (PS1 uses 12-bit fractional part)
FIXED_POINT_SCALE = 4096.0


@dataclass
class SVECTOR:
    """Short vector (8 bytes) - used for vertices and normals"""
    vx: int  # int16_t
    vy: int  # int16_t
    vz: int  # int16_t
    pad: int  # int16_t (flags)

    def to_bytes(self) -> bytes:
        return struct.pack('<hhhh', self.vx, self.vy, self.vz, self.pad)

    @classmethod
    def from_float(cls, x: float, y:  float, z: float, scale: float, pad: int = 0) -> 'SVECTOR':
        """Convert floating point coordinates to fixed-point"""
        vx = int(round(x * scale))
        vy = int(round(y * scale))
        vz = int(round(z * scale))
        # Clamp to int16 range
        vx = max(-32768, min(32767, vx))
        vy = max(-32768, min(32767, vy))
        vz = max(-32768, min(32767, vz))
        return cls(vx, vy, vz, pad)


@dataclass
class Vertex:
    """A vertex with position and normal"""
    position:  Tuple[float, float, float]
    normal: Tuple[float, float, float]

    def position_key(self, precision: int = 5) -> Tuple[int, ...]:
        """Return a hashable key for position comparison"""
        scale = 10 ** precision
        return (
            int(round(self.position[0] * scale)),
            int(round(self.position[1] * scale)),
            int(round(self.position[2] * scale))
        )


@dataclass
class Triangle:
    """A triangle with 3 vertices"""
    v0: int  # vertex index
    v1: int
    v2: int
    used: bool = False

    def get_vertices(self) -> List[int]:
        return [self.v0, self.v1, self.v2]

    def get_edge(self, edge_idx: int) -> Tuple[int, int]:
        """Get edge vertices (0=v0-v1, 1=v1-v2, 2=v2-v0)"""
        edges = [(self.v0, self.v1), (self.v1, self.v2), (self.v2, self.v0)]
        return edges[edge_idx]

    def get_opposite_vertex(self, edge_idx: int) -> int:
        """Get vertex opposite to the given edge"""
        opposites = [self.v2, self.v0, self.v1]
        return opposites[edge_idx]


@dataclass
class Quad:
    """A quad with 4 vertices in PS1 Z-pattern order"""
    v0: int
    v1: int
    v2: int
    v3: int


class GLTFParser: 
    """Parser for glTF/GLB files"""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.gltf:  Dict[str, Any] = {}
        self.binary_data: bytes = b''
        self.vertices: List[Vertex] = []
        self.triangles: List[Triangle] = []

    def parse(self):
        """Parse the glTF or GLB file"""
        if self.filepath.lower().endswith('.glb'):
            self._parse_glb()
        else: 
            self._parse_gltf()

        self._extract_mesh_data()

    def _parse_glb(self):
        """Parse GLB (binary glTF) file"""
        with open(self.filepath, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x46546C67:  # "glTF"
                raise ValueError("Invalid GLB magic number")

            version = struct.unpack('<I', f.read(4))[0]
            if version != 2:
                raise ValueError(f"Unsupported glTF version: {version}")

            total_length = struct.unpack('<I', f.read(4))[0]

            # Read chunks
            while f.tell() < total_length:
                chunk_length = struct.unpack('<I', f.read(4))[0]
                chunk_type = struct.unpack('<I', f.read(4))[0]
                chunk_data = f.read(chunk_length)

                if chunk_type == 0x4E4F534A:  # "JSON"
                    self.gltf = json.loads(chunk_data.decode('utf-8'))
                elif chunk_type == 0x004E4942:  # "BIN\0"
                    self.binary_data = chunk_data

    def _parse_gltf(self):
        """Parse glTF JSON file"""
        with open(self.filepath, 'r') as f:
            self.gltf = json.load(f)

        # Load binary buffer if referenced
        if 'buffers' in self.gltf:
            for buffer in self.gltf['buffers']: 
                if 'uri' in buffer:
                    uri = buffer['uri']
                    if uri.startswith('data:'):
                        # Embedded base64 data
                        base64_start = uri.find(',') + 1
                        self.binary_data = base64.b64decode(uri[base64_start:])
                    else:
                        # External file
                        buffer_path = os.path.join(os.path.dirname(self.filepath), uri)
                        with open(buffer_path, 'rb') as f:
                            self.binary_data = f.read()

    def _get_accessor_data(self, accessor_idx: int) -> List[Any]:
        """Get data from an accessor"""
        accessor = self.gltf['accessors'][accessor_idx]
        buffer_view = self.gltf['bufferViews'][accessor['bufferView']]

        byte_offset = buffer_view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
        count = accessor['count']
        component_type = accessor['componentType']
        accessor_type = accessor['type']

        # Determine component format
        component_formats = {
            5120: ('b', 1),  # BYTE
            5121: ('B', 1),  # UNSIGNED_BYTE
            5122: ('h', 2),  # SHORT
            5123: ('H', 2),  # UNSIGNED_SHORT
            5125: ('I', 4),  # UNSIGNED_INT
            5126: ('f', 4),  # FLOAT
        }

        fmt, size = component_formats[component_type]

        # Determine number of components
        type_counts = {
            'SCALAR': 1,
            'VEC2': 2,
            'VEC3': 3,
            'VEC4': 4,
            'MAT2': 4,
            'MAT3': 9,
            'MAT4': 16,
        }

        num_components = type_counts[accessor_type]
        stride = buffer_view.get('byteStride', size * num_components)

        data = []
        for i in range(count):
            offset = byte_offset + i * stride
            values = struct.unpack_from(f'<{num_components}{fmt}', self.binary_data, offset)
            if num_components == 1:
                data.append(values[0])
            else:
                data.append(values)

        return data

    def _extract_mesh_data(self):
        """Extract vertices and triangles from the mesh"""
        if 'meshes' not in self.gltf or len(self.gltf['meshes']) == 0:
            raise ValueError("No meshes found in glTF file")

        # Process first mesh (could be extended to handle multiple)
        mesh = self.gltf['meshes'][0]

        for primitive in mesh['primitives']:
            if primitive. get('mode', 4) != 4:  # Only triangles
                print(f"Warning:  Skipping non-triangle primitive (mode={primitive.get('mode', 4)})")
                continue

            # Get position data
            pos_accessor_idx = primitive['attributes']['POSITION']
            positions = self._get_accessor_data(pos_accessor_idx)

            # Get normal data (or generate defaults)
            if 'NORMAL' in primitive['attributes']:
                norm_accessor_idx = primitive['attributes']['NORMAL']
                normals = self._get_accessor_data(norm_accessor_idx)
            else:
                normals = [(0.0, 1.0, 0.0)] * len(positions)

            # Build vertex list
            base_vertex_idx = len(self.vertices)
            for pos, norm in zip(positions, normals):
                self.vertices.append(Vertex(position=pos, normal=norm))

            # Get indices
            if 'indices' in primitive: 
                idx_accessor_idx = primitive['indices']
                indices = self._get_accessor_data(idx_accessor_idx)
            else:
                # Non-indexed geometry
                indices = list(range(len(positions)))

            # Build triangle list
            for i in range(0, len(indices), 3):
                if i + 2 < len(indices):
                    self.triangles.append(Triangle(
                        v0=base_vertex_idx + indices[i],
                        v1=base_vertex_idx + indices[i + 1],
                        v2=base_vertex_idx + indices[i + 2]
                    ))

        print(f"Loaded {len(self.vertices)} vertices and {len(self.triangles)} triangles")


class TriangleToQuadMerger: 
    """Merges triangles into quads based on shared edge positions"""

    def __init__(self, vertices: List[Vertex], triangles: List[Triangle]):
        self.vertices = vertices
        self.triangles = triangles
        self.quads: List[Quad] = []
        self.remaining_triangles:  List[Triangle] = []

    def _get_position_key(self, vertex_idx: int) -> Tuple[int, ... ]:
        """Get position key for a vertex"""
        return self.vertices[vertex_idx].position_key()

    def _get_edge_key(self, v1_idx: int, v2_idx: int) -> Tuple[Tuple[int, ... ], Tuple[int, ...]]:
        """Get a canonical edge key based on positions (sorted)"""
        k1 = self._get_position_key(v1_idx)
        k2 = self._get_position_key(v2_idx)
        return tuple(sorted([k1, k2]))

    def merge(self) -> Tuple[List[Quad], List[Triangle]]:
        """Merge triangles into quads where possible"""
        # Build edge (by position) to triangle map
        edge_to_triangles: Dict[Any, List[Tuple[int, int]]] = defaultdict(list)

        for tri_idx, tri in enumerate(self. triangles):
            for edge_idx in range(3):
                v1_idx, v2_idx = tri.get_edge(edge_idx)
                edge_key = self._get_edge_key(v1_idx, v2_idx)
                edge_to_triangles[edge_key].append((tri_idx, edge_idx))

        # Debug:  print edge sharing info
        shared_edges = {k: v for k, v in edge_to_triangles.items() if len(v) >= 2}
        print(f"Found {len(shared_edges)} shared edges between triangles")

        # Find pairs of triangles sharing an edge
        for tri_idx, tri in enumerate(self. triangles):
            if tri. used:
                continue

            best_match = None
            best_score = float('inf')

            for edge_idx in range(3):
                v1_idx, v2_idx = tri.get_edge(edge_idx)
                edge_key = self._get_edge_key(v1_idx, v2_idx)

                for other_tri_idx, other_edge_idx in edge_to_triangles[edge_key]: 
                    if other_tri_idx == tri_idx:
                        continue

                    other_tri = self.triangles[other_tri_idx]
                    if other_tri.used:
                        continue

                    # Calculate planarity score (lower is better)
                    score = self._calculate_planarity_score(tri, other_tri, edge_idx, other_edge_idx)

                    if score < best_score: 
                        best_score = score
                        best_match = (other_tri_idx, edge_idx, other_edge_idx)

            if best_match is not None and best_score < 0.5:  # Planarity threshold
                other_tri_idx, edge_idx, other_edge_idx = best_match
                other_tri = self.triangles[other_tri_idx]

                quad = self._create_quad(tri, other_tri, edge_idx, other_edge_idx)
                self.quads.append(quad)

                tri.used = True
                other_tri.used = True

        # Collect remaining triangles
        for tri in self.triangles:
            if not tri.used:
                self.remaining_triangles.append(tri)

        print(f"Merged into {len(self.quads)} quads, {len(self.remaining_triangles)} remaining triangles")

        return self.quads, self.remaining_triangles

    def _calculate_planarity_score(self, tri1: Triangle, tri2: Triangle,
                                   edge_idx1: int, edge_idx2: int) -> float:
        """Calculate how planar the resulting quad would be (0 = perfectly planar)"""
        # Get the 4 vertex positions
        v0_idx = tri1.get_opposite_vertex(edge_idx1)
        edge = tri1.get_edge(edge_idx1)
        v1_idx = edge[0]
        v2_idx = edge[1]
        v3_idx = tri2.get_opposite_vertex(edge_idx2)

        v0 = self.vertices[v0_idx].position
        v1 = self.vertices[v1_idx].position
        v2 = self.vertices[v2_idx].position
        v3 = self.vertices[v3_idx].position

        # Calculate normals of both triangles
        n1 = self._cross_product(self._subtract(v1, v0), self._subtract(v2, v0))
        n2 = self._cross_product(self._subtract(v2, v3), self._subtract(v1, v3))

        # Normalize
        n1 = self._normalize(n1)
        n2 = self._normalize(n2)

        if n1 is None or n2 is None:
            return float('inf')

        # Dot product (1 = parallel, 0 = perpendicular)
        dot = abs(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2])

        return 1.0 - dot

    def _subtract(self, a:  Tuple[float, ... ], b: Tuple[float, ...]) -> Tuple[float, ...]:
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def _cross_product(self, a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]: 
        return (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        )

    def _normalize(self, v: Tuple[float, ...]) -> Optional[Tuple[float, ...]]:
        length = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
        if length < 1e-10:
            return None
        return (v[0] / length, v[1] / length, v[2] / length)

    def _create_quad(self, tri1: Triangle, tri2: Triangle,
                     edge_idx1: int, edge_idx2: int) -> Quad:
        """Create a quad from two triangles sharing an edge

        PS1 quad vertex order is Z-pattern: 
          v0 -- v1
           |  /  |
          v2 -- v3

        Triangle 1: v0, v1, v2 (edge v1-v2 shared)
        Triangle 2: v1, v3, v2 (edge v1-v2 shared)
        """
        # Get shared edge vertices from tri1
        edge = tri1.get_edge(edge_idx1)

        # v0 is opposite to shared edge in tri1
        # v3 is opposite to shared edge in tri2
        v0_idx = tri1.get_opposite_vertex(edge_idx1)
        v3_idx = tri2.get_opposite_vertex(edge_idx2)

        # Determine correct assignment of edge vertices to v1 and v2
        # We need to maintain consistent winding
        e0, e1 = edge

        # Check triangle winding to assign v1 and v2 correctly
        # In tri1: v0 -> e0 -> e1 or v0 -> e1 -> e0
        tri1_verts = [tri1.v0, tri1.v1, tri1.v2]
        v0_pos = tri1_verts.index(v0_idx)
        next_vert = tri1_verts[(v0_pos + 1) % 3]

        if next_vert == e0:
            v1_idx, v2_idx = e0, e1
        else:
            v1_idx, v2_idx = e1, e0

        return Quad(v0=v0_idx, v1=v1_idx, v2=v2_idx, v3=v3_idx)


class AMFWriter:
    """Writer for AMF files with G4 polygon support"""

    def __init__(self, vertices:  List[Vertex], quads: List[Quad]):
        self.vertices = vertices
        self.quads = quads

    def write(self, filepath: str):
        """Write the AMF file"""
        if not self.quads:
            print("No quads to write!")
            return

        # Calculate world bounds from vertices
        min_x = min_z = float('inf')
        max_x = max_z = float('-inf')

        for v in self.vertices:
            pos = v.position
            # Convert to fixed point for bounds
            fx = int(pos[0] * FIXED_POINT_SCALE)
            fz = int(pos[2] * FIXED_POINT_SCALE)
            min_x = min(min_x, fx)
            max_x = max(max_x, fx)
            min_z = min(min_z, fz)
            max_z = max(max_z, fz)

        # Clamp to int32 range
        min_x = max(-2147483648, min(2147483647, int(min_x)))
        min_z = max(-2147483648, min(2147483647, int(min_z)))
        max_x = max(-2147483648, min(2147483647, int(max_x)))
        max_z = max(-2147483648, min(2147483647, int(max_z)))

        with open(filepath, 'wb') as f:
            # Write AMF header (24 bytes)
            # used_textures (uint32), x chunks (uint16), z chunks (uint16)
            f.write(struct.pack('<I', 0))  # No textures
            f. write(struct.pack('<H', 1))  # 1 chunk in X
            f.write(struct.pack('<H', 1))  # 1 chunk in Z

            # WorldBounds (16 bytes)
            f.write(struct.pack('<iiii', min_x, min_z, max_x, max_z))

            # No texture names (0 textures)

            # Chunk table (4 bytes per chunk, just 1 chunk)
            f.write(struct.pack('<I', 0))  # Chunk offset/info placeholder

            # Chunk header (16 bytes for polygon counts)
            # Order: F4, G4, FT4, GT4, F3, G3, FT3, GT3
            f.write(struct.pack('<HHHHHHHH',
                                0,               # F4_amount
                                len(self.quads), # G4_amount
                                0,               # FT4_amount
                                0,               # GT4_amount
                                0,               # F3_amount
                                0,               # G3_amount
                                0,               # FT3_amount
                                0))              # GT3_amount

            # 8 pointers (32 bytes) - set to 0, filled at runtime
            f.write(b'\x00' * 32)

            # Write G4 polygons
            for quad in self.quads:
                self._write_pg4(f, quad)

        file_size = os.path.getsize(filepath)
        print(f"\nExported to {filepath}")
        print(f"  Quads: {len(self.quads)}")
        print(f"  File size: {file_size} bytes")

    def _write_pg4(self, f, quad: Quad):
        """Write a PG4 (gouraud-shaded quad) structure

        PG4 layout:
        - 4 SVECTOR vertices (32 bytes)
        - 4 SVECTOR normals (32 bytes)
        - POLY_G4 GPU primitive (36 bytes)
        Total: 100 bytes
        """
        # Get vertex data
        v0 = self.vertices[quad.v0]
        v1 = self.vertices[quad.v1]
        v2 = self.vertices[quad.v2]
        v3 = self.vertices[quad.v3]

        # Write 4 vertices
        for v in [v0, v1, v2, v3]: 
            sv = SVECTOR.from_float(v.position[0], v.position[1], v. position[2], scale=64.0)
            f.write(sv.to_bytes())

        # Write 4 normals
        for v in [v0, v1, v2, v3]: 
            # Normals use different scale (unit vectors)
            sn = SVECTOR.from_float(v.normal[0], v.normal[1], v.normal[2], scale=2048.0)
            f.write(sn.to_bytes())

        # Write POLY_G4 GPU primitive (36 bytes)
        # Structure: tag(4) + 4*(color(3) + pad(1) + xy(4)) = 4 + 4*8 = 36
        # For G4: each vertex has its own color

        # Tag (4 bytes) - length and GPU command
        tag = (8 << 24) | 0x38  # len=8 words, code for POLY_G4
        f.write(struct.pack('<I', tag))

        # 4 vertex entries:  color (r,g,b,code) + xy coordinates
        for i, v in enumerate([v0, v1, v2, v3]):
            # Color (default to white/gray)
            r, g, b = 128, 128, 128
            code = 0x38 if i == 0 else 0  # GPU code only in first color
            f.write(struct.pack('<BBBB', r, g, b, code))

            # XY screen coordinates (set to 0, calculated at runtime)
            f.write(struct.pack('<hh', 0, 0))


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.gltf|input.glb [output.amf]")
        print("\nConverts glTF 2.0 files to AMF format for UnnamedHB1 PS1 homebrew.")
        print("\nFeatures:")
        print("  - Supports only G4 (gouraud-shaded quads)")
        print("  - Automatically merges triangles into quads")
        print("  - Remaining triangles that can't be merged are discarded")
        sys.exit(1)

    input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else: 
        output_file = os.path.splitext(input_file)[0] + ".amf"

    print(f"Reading {input_file}")

    # Parse glTF
    parser = GLTFParser(input_file)
    parser.parse()

    # Merge triangles to quads
    merger = TriangleToQuadMerger(parser.vertices, parser.triangles)
    quads, remaining = merger.merge()

    if remaining:
        print(f"Warning: {len(remaining)} triangles could not be merged into quads and will be discarded")

    if not quads: 
        print("Error: No quads could be created from the input mesh")
        sys.exit(1)

    # Write AMF
    writer = AMFWriter(parser.vertices, quads)
    writer.write(output_file)


if __name__ == "__main__": 
    main()