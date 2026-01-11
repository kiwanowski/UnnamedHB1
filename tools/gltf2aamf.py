#!/usr/bin/env python3
"""
glTF to AAMF Converter for UnnamedHB1 PlayStation 1 Animated Model Format

This script converts glTF 2.0 files to AAMF (Animated AMF) format for the 
UnnamedHB1 PS1 homebrew game.  Only supports G4 (gouraud-shaded quads) polygon type. 
Triangles are automatically merged into quads where possible. 

AAMF format specification derived from:
https://github.com/achrostel/UnnamedHB1

Usage:
    python gltf2aamf.py input. gltf [output.aamf]

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
from typing import List, Tuple, Optional, Dict, Any, Set


# ============================================================================
# Constants
# ============================================================================

# Fixed-point scale factor (4. 12 format for PS1 GTE)
FIXED_POINT_SCALE = 4096.0

# Structure sizes
SVECTOR_SIZE = 8   # 4 int16_t values
MATRIX_SIZE = 32   # 9 int16_t (18 bytes) + 2 padding + 3 int32_t (12 bytes) = 32 bytes
POLY_G4_SIZE = 36  # With 4-byte tag:  code/colors + 4 vertices

# PG4 total size:  4 SVECTORs (vertices) + 4 SVECTORs (normals) + POLY_G4
PG4_SIZE = 4 * SVECTOR_SIZE + 4 * SVECTOR_SIZE + POLY_G4_SIZE  # 32 + 32 + 36 = 100


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Vertex: 
    """3D vertex with position, normal, and optional color"""
    position:  Tuple[float, float, float]
    normal: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    
    def __hash__(self):
        return hash((self. position, self.normal))
    
    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return False
        return self.position == other. position and self.normal == other.normal


@dataclass
class Triangle:
    """Triangle with 3 vertices"""
    v0: Vertex
    v1: Vertex
    v2: Vertex
    
    def get_edge(self, index: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]: 
        """Get edge by index (0, 1, 2). Returns (start_pos, end_pos)"""
        vertices = [self.v0, self.v1, self.v2]
        return (vertices[index]. position, vertices[(index + 1) % 3].position)
    
    def get_opposite_vertex(self, edge_index: int) -> Vertex: 
        """Get the vertex opposite to the given edge"""
        vertices = [self.v0, self.v1, self.v2]
        return vertices[(edge_index + 2) % 3]
    
    def get_edge_vertices(self, edge_index: int) -> Tuple[Vertex, Vertex]: 
        """Get the two vertices forming the given edge"""
        vertices = [self.v0, self.v1, self.v2]
        return (vertices[edge_index], vertices[(edge_index + 1) % 3])


@dataclass
class Quad:
    """Quad with 4 vertices in PS1 Z-pattern order (v0-v1-v2-v3)"""
    v0: Vertex  # Top-left
    v1: Vertex  # Top-right
    v2: Vertex  # Bottom-left
    v3: Vertex  # Bottom-right


@dataclass 
class Bone:
    """Bone data with geometry and hierarchy info"""
    index: int
    parent_index: int
    quads: List[Quad] = field(default_factory=list)


@dataclass
class KeyframeTransform:
    """Keyframe transformation data"""
    translation: Tuple[float, float, float]
    rotation:  Tuple[float, float, float, float]  # Quaternion (x, y, z, w)
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Animation:
    """Animation data with keyframes per bone"""
    name:  str
    keyframe_count: int
    # keyframes[bone_idx][keyframe_idx]
    keyframes:  List[List[KeyframeTransform]] = field(default_factory=list)


# ============================================================================
# Triangle to Quad Merging
# ============================================================================

def edges_match(edge1: Tuple[Tuple, Tuple], edge2: Tuple[Tuple, Tuple], tolerance: float = 1e-6) -> bool:
    """Check if two edges are the same (in either direction)"""
    def pos_equal(p1: Tuple, p2: Tuple) -> bool:
        return all(abs(a - b) < tolerance for a, b in zip(p1, p2))
    
    return (pos_equal(edge1[0], edge2[1]) and pos_equal(edge1[1], edge2[0])) or \
           (pos_equal(edge1[0], edge2[0]) and pos_equal(edge1[1], edge2[1]))


def is_planar(v0: Vertex, v1: Vertex, v2: Vertex, v3: Vertex, tolerance: float = 0.1) -> bool:
    """Check if 4 vertices are roughly coplanar"""
    # Calculate normal of first triangle
    e1 = (v1.position[0] - v0.position[0], v1.position[1] - v0.position[1], v1.position[2] - v0.position[2])
    e2 = (v2.position[0] - v0.position[0], v2.position[1] - v0.position[1], v2.position[2] - v0.position[2])
    
    normal = (
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0]
    )
    
    length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if length < 1e-10:
        return True  # Degenerate triangle
    
    normal = (normal[0]/length, normal[1]/length, normal[2]/length)
    
    # Check if v3 lies on the plane
    d = normal[0]*v0.position[0] + normal[1]*v0.position[1] + normal[2]*v0.position[2]
    dist = abs(normal[0]*v3.position[0] + normal[1]*v3.position[1] + normal[2]*v3.position[2] - d)
    
    return dist < tolerance


def merge_triangles_to_quads(triangles: List[Triangle]) -> Tuple[List[Quad], List[Triangle]]:
    """
    Merge adjacent triangles into quads where possible.
    Returns (quads, remaining_triangles)
    """
    quads = []
    used = set()
    remaining = []
    
    # Build adjacency map based on shared edges
    edge_to_triangle:  Dict[Tuple, List[Tuple[int, int]]] = {}  # edge -> [(tri_idx, edge_idx), ...]
    
    for tri_idx, tri in enumerate(triangles):
        for edge_idx in range(3):
            edge = tri.get_edge(edge_idx)
            # Normalize edge direction for consistent hashing
            edge_key = tuple(sorted([edge[0], edge[1]], key=lambda p: (p[0], p[1], p[2])))
            if edge_key not in edge_to_triangle:
                edge_to_triangle[edge_key] = []
            edge_to_triangle[edge_key].append((tri_idx, edge_idx))
    
    # Find pairs of triangles sharing an edge
    for edge_key, tri_list in edge_to_triangle.items():
        if len(tri_list) != 2:
            continue
        
        (idx1, edge_idx1), (idx2, edge_idx2) = tri_list
        
        if idx1 in used or idx2 in used:
            continue
        
        tri1 = triangles[idx1]
        tri2 = triangles[idx2]
        
        # Get the shared edge vertices and opposite vertices
        shared_v1, shared_v2 = tri1.get_edge_vertices(edge_idx1)
        opp1 = tri1.get_opposite_vertex(edge_idx1)
        opp2 = tri2.get_opposite_vertex(edge_idx2)
        
        # Check if the resulting quad would be planar
        if not is_planar(shared_v1, shared_v2, opp1, opp2):
            continue
        
        # Create quad in PS1 Z-pattern order: 
        # v0 -- v1
        #  |  X  |
        # v2 -- v3
        #
        # For two triangles sharing edge (shared_v1, shared_v2):
        # Triangle 1: opp1, shared_v1, shared_v2 (or some rotation)
        # Triangle 2: opp2, shared_v2, shared_v1 (or some rotation)
        #
        # The quad should be:  opp1, shared_v1, shared_v2, opp2
        # But we need to order them correctly for PS1's Z-pattern
        
        # Determine proper winding based on the original triangle vertices
        # PS1 quad order: v0-v1 is top edge, v2-v3 is bottom edge
        # Triangles:  v0-v1-v2 and v1-v3-v2
        
        # The shared edge is the diagonal of the quad
        # opp1 and opp2 are the corners not on the shared edge
        
        # For proper Z-pattern, we need to figure out which corner is which
        # Let's use the original triangle winding as a guide
        
        # Simple approach: create quad with vertices in order that forms proper winding
        quad = Quad(
            v0=opp1,
            v1=shared_v1,
            v2=shared_v2,
            v3=opp2
        )
        
        quads.append(quad)
        used.add(idx1)
        used.add(idx2)
    
    # Collect remaining unpaired triangles
    for idx, tri in enumerate(triangles):
        if idx not in used: 
            remaining.append(tri)
    
    return quads, remaining


def triangle_to_degenerate_quad(tri: Triangle) -> Quad: 
    """Convert a triangle to a degenerate quad by duplicating one vertex"""
    # Duplicate v2 as v3
    return Quad(
        v0=tri. v0,
        v1=tri. v1,
        v2=tri. v2,
        v3=tri. v2  # Degenerate - duplicates v2
    )


# ============================================================================
# glTF Parser
# ============================================================================

class GltfParser: 
    """Parser for glTF 2.0 files"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.gltf:  Dict[str, Any] = {}
        self.buffers: List[bytes] = []
        self.bones: List[Bone] = []
        self.animations: List[Animation] = []
        
    def load(self):
        """Load and parse the glTF file"""
        with open(self.filename, 'r') as f:
            self.gltf = json.load(f)
        
        self._load_buffers()
        self._parse_scene()
        self._parse_animations()
    
    def _load_buffers(self):
        """Load binary buffer data"""
        for buffer_info in self.gltf. get('buffers', []):
            uri = buffer_info. get('uri', '')
            if uri.startswith('data:'):
                # Base64 embedded data
                header, data = uri.split(',', 1)
                self.buffers.append(base64.b64decode(data))
            else: 
                # External file
                buffer_path = os.path.join(os.path.dirname(self.filename), uri)
                with open(buffer_path, 'rb') as f:
                    self.buffers. append(f.read())
    
    def _get_accessor_data(self, accessor_idx: int) -> List: 
        """Get data from an accessor"""
        accessor = self.gltf['accessors'][accessor_idx]
        buffer_view = self. gltf['bufferViews'][accessor['bufferView']]
        
        buffer_data = self.buffers[buffer_view['buffer']]
        byte_offset = buffer_view. get('byteOffset', 0) + accessor. get('byteOffset', 0)
        
        component_type = accessor['componentType']
        accessor_type = accessor['type']
        count = accessor['count']
        
        # Component type to struct format
        component_formats = {
            5120: 'b',   # BYTE
            5121: 'B',   # UNSIGNED_BYTE
            5122: 'h',   # SHORT
            5123: 'H',   # UNSIGNED_SHORT
            5125: 'I',   # UNSIGNED_INT
            5126: 'f',   # FLOAT
        }
        
        # Type to component count
        type_counts = {
            'SCALAR': 1,
            'VEC2': 2,
            'VEC3': 3,
            'VEC4': 4,
            'MAT4': 16,
        }
        
        fmt = component_formats[component_type]
        components = type_counts[accessor_type]
        component_size = struct.calcsize(fmt)
        
        byte_stride = buffer_view.get('byteStride', component_size * components)
        
        data = []
        for i in range(count):
            offset = byte_offset + i * byte_stride
            if components == 1:
                value = struct.unpack_from(f'<{fmt}', buffer_data, offset)[0]
                data.append(value)
            else:
                values = struct.unpack_from(f'<{components}{fmt}', buffer_data, offset)
                data.append(tuple(values))
        
        return data
    
    def _parse_scene(self):
        """Parse the scene and extract mesh/skin data"""
        # Find the main mesh and skin
        mesh_node = None
        skin_data = None
        
        for node in self.gltf. get('nodes', []):
            if 'mesh' in node:
                mesh_node = node
                if 'skin' in node:
                    skin_data = self. gltf['skins'][node['skin']]
                break
        
        if not mesh_node: 
            print("No mesh found in glTF file!")
            return
        
        # Get mesh data
        mesh = self. gltf['meshes'][mesh_node['mesh']]
        
        # Parse each primitive
        all_triangles_by_bone:  Dict[int, List[Triangle]] = {}
        
        for primitive in mesh. get('primitives', []):
            positions = self._get_accessor_data(primitive['attributes']['POSITION'])
            
            normals = [(0.0, 1.0, 0.0)] * len(positions)
            if 'NORMAL' in primitive['attributes']: 
                normals = self._get_accessor_data(primitive['attributes']['NORMAL'])
            
            # Get joint weights if available
            joints = [(0, 0, 0, 0)] * len(positions)
            weights = [(1.0, 0.0, 0.0, 0.0)] * len(positions)
            
            if 'JOINTS_0' in primitive['attributes']:
                joints = self._get_accessor_data(primitive['attributes']['JOINTS_0'])
            if 'WEIGHTS_0' in primitive['attributes']: 
                weights = self._get_accessor_data(primitive['attributes']['WEIGHTS_0'])
            
            # Get indices
            indices = []
            if 'indices' in primitive: 
                indices = self._get_accessor_data(primitive['indices'])
            else:
                indices = list(range(len(positions)))
            
            # Build triangles grouped by bone
            for i in range(0, len(indices), 3):
                if i + 2 >= len(indices):
                    break
                
                i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
                
                v0 = Vertex(positions[i0], normals[i0])
                v1 = Vertex(positions[i1], normals[i1])
                v2 = Vertex(positions[i2], normals[i2])
                
                # Determine which bone this triangle belongs to (use dominant weight)
                bone_idx = joints[i0][0]  # Use first vertex's primary joint
                
                if bone_idx not in all_triangles_by_bone: 
                    all_triangles_by_bone[bone_idx] = []
                
                all_triangles_by_bone[bone_idx].append(Triangle(v0, v1, v2))
        
        # Build bone hierarchy from skin
        bone_parents = {}
        joint_nodes = []
        
        if skin_data:
            joint_nodes = skin_data. get('joints', [])
            
            # Build parent mapping from node hierarchy
            for joint_idx, node_idx in enumerate(joint_nodes):
                node = self.gltf['nodes'][node_idx]
                # Find parent by looking through all nodes
                bone_parents[joint_idx] = joint_idx  # Default:  self-parent (root)
                
                for potential_parent_idx, potential_parent in enumerate(self.gltf['nodes']):
                    if 'children' in potential_parent and node_idx in potential_parent['children']: 
                        # Found the parent node, now find its joint index
                        if potential_parent_idx in joint_nodes:
                            bone_parents[joint_idx] = joint_nodes.index(potential_parent_idx)
                        break
        
        # If no skin, create a single bone
        if not bone_parents:
            bone_parents[0] = 0
        
        # Convert triangles to quads for each bone
        for bone_idx in sorted(all_triangles_by_bone. keys()):
            triangles = all_triangles_by_bone[bone_idx]
            quads, remaining = merge_triangles_to_quads(triangles)
            
            print(f"Bone {bone_idx}:  {len(triangles)} triangles -> {len(quads)} quads, {len(remaining)} remaining triangles")
            
            # Convert remaining triangles to degenerate quads
            for tri in remaining:
                quads.append(triangle_to_degenerate_quad(tri))
            
            parent_idx = bone_parents. get(bone_idx, bone_idx)
            bone = Bone(index=bone_idx, parent_index=parent_idx, quads=quads)
            self.bones.append(bone)
        
        # Ensure we have at least one bone
        if not self.bones:
            self.bones.append(Bone(index=0, parent_index=0, quads=[]))
    
    def _parse_animations(self):
        """Parse animation data from glTF"""
        for anim_data in self.gltf.get('animations', []):
            name = anim_data. get('name', 'anim')[: 8]  # Truncate to 8 chars
            
            # Find the number of keyframes (from the first sampler's input)
            if not anim_data.get('samplers'):
                continue
            
            first_sampler = anim_data['samplers'][0]
            times = self._get_accessor_data(first_sampler['input'])
            keyframe_count = len(times)
            
            # Initialize keyframes for all bones
            bone_keyframes: List[List[KeyframeTransform]] = []
            for _ in range(len(self.bones)):
                bone_keyframes.append([
                    KeyframeTransform(
                        translation=(0.0, 0.0, 0.0),
                        rotation=(0.0, 0.0, 0.0, 1.0)
                    ) for _ in range(keyframe_count)
                ])
            
            # Parse channels
            for channel in anim_data. get('channels', []):
                sampler = anim_data['samplers'][channel['sampler']]
                target = channel['target']
                
                node_idx = target. get('node', 0)
                path = target.get('path', '')
                
                # Map node index to bone index
                # In glTF exported by aamf2gltf, bone nodes start at index 2
                bone_idx = node_idx - 2
                if bone_idx < 0 or bone_idx >= len(self.bones):
                    continue
                
                output_data = self._get_accessor_data(sampler['output'])
                
                for kf_idx, value in enumerate(output_data):
                    if kf_idx >= keyframe_count: 
                        break
                    
                    if path == 'translation':
                        bone_keyframes[bone_idx][kf_idx] = KeyframeTransform(
                            translation=value,
                            rotation=bone_keyframes[bone_idx][kf_idx].rotation
                        )
                    elif path == 'rotation':
                        bone_keyframes[bone_idx][kf_idx] = KeyframeTransform(
                            translation=bone_keyframes[bone_idx][kf_idx].translation,
                            rotation=value
                        )
            
            animation = Animation(
                name=name,
                keyframe_count=keyframe_count,
                keyframes=bone_keyframes
            )
            self.animations. append(animation)


# ============================================================================
# AAMF Writer
# ============================================================================

class AAMFWriter:
    """Writer for AAMF (Animated AMF) files"""
    
    def __init__(self, bones: List[Bone], animations: List[Animation]):
        self.bones = bones
        self.animations = animations
    
    def _float_to_fixed(self, value: float) -> int:
        """Convert float to 4.12 fixed-point int16"""
        result = int(value * FIXED_POINT_SCALE)
        # Clamp to int16 range
        return max(-32768, min(32767, result))
    
    def _float_to_fixed32(self, value:  float) -> int:
        """Convert float to fixed-point int32 for translations"""
        result = int(value * FIXED_POINT_SCALE)
        # Clamp to int32 range
        return max(-2147483648, min(2147483647, result))
    
    def _write_svector(self, vx: int, vy: int, vz: int, pad: int = 0) -> bytes:
        """Write an SVECTOR (4 int16_t = 8 bytes)"""
        return struct.pack('<hhhh', vx, vy, vz, pad)
    
    def _write_matrix(self, transform: KeyframeTransform) -> bytes:
        """Write a MATRIX (32 bytes) from a KeyframeTransform"""
        # Convert quaternion to rotation matrix
        qx, qy, qz, qw = transform.rotation
        
        # Rotation matrix from quaternion
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
            [1.0 - 2.0*(yy + zz), 2.0*(xy - wz), 2.0*(xz + wy)],
            [2.0*(xy + wz), 1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
            [2.0*(xz - wy), 2.0*(yz + wx), 1.0 - 2.0*(xx + yy)]
        ]
        
        # Write rotation matrix (9 int16_t = 18 bytes)
        data = b''
        for row in range(3):
            for col in range(3):
                data += struct.pack('<h', self._float_to_fixed(m[row][col]))
        
        # 2 bytes padding
        data += struct.pack('<h', 0)
        
        # Write translation (3 int32_t = 12 bytes)
        tx, ty, tz = transform.translation
        data += struct.pack('<iii', 
            self._float_to_fixed32(tx),
            self._float_to_fixed32(ty),
            self._float_to_fixed32(tz)
        )
        
        return data
    
    def _write_poly_g4(self) -> bytes:
        """Write a POLY_G4 primitive (36 bytes with tag)"""
        # Tag (4 bytes)
        data = struct.pack('<I', 0x38000000)  # POLY_G4 GPU command
        
        # 4 vertices with colors
        # r0, g0, b0, code; x0, y0
        # r1, g1, b1, pad0; x1, y1
        # r2, g2, b2, pad1; x2, y2
        # r3, g3, b3, pad2; x3, y3
        for i in range(4):
            r, g, b = 128, 128, 128  # Default gray
            code_or_pad = 0x38 if i == 0 else 0  # GPU code for first vertex only
            x, y = 0, 0  # Screen coords set at runtime
            data += struct.pack('<BBBB', r, g, b, code_or_pad)
            data += struct.pack('<hh', x, y)
        
        return data
    
    def _write_pg4(self, quad: Quad) -> bytes:
        """Write a PG4 structure (100 bytes)"""
        data = b''
        
        # Write 4 vertex SVECTORs
        for v in [quad.v0, quad.v1, quad.v2, quad.v3]: 
            vx = self._float_to_fixed(v. position[0])
            vy = self._float_to_fixed(v.position[1])
            vz = self._float_to_fixed(v.position[2])
            data += self._write_svector(vx, vy, vz)
        
        # Write 4 normal SVECTORs
        for v in [quad.v0, quad.v1, quad.v2, quad.v3]:
            nx = self._float_to_fixed(v.normal[0])
            ny = self._float_to_fixed(v.normal[1])
            nz = self._float_to_fixed(v.normal[2])
            data += self._write_svector(nx, ny, nz)
        
        # Write POLY_G4
        data += self._write_poly_g4()
        
        return data
    
    def _write_amf_for_bone(self, bone: Bone) -> bytes:
        """Write embedded AMF data for a single bone"""
        data = b''
        
        # AMF Header (24 bytes)
        used_textures = 0
        x_chunks = 1
        z_chunks = 1
        
        # Header:  used_textures (4) + x (2) + z (2) + WorldBounds (16)
        data += struct.pack('<I', used_textures)
        data += struct.pack('<HH', x_chunks, z_chunks)
        
        # WorldBounds (16 bytes)
        data += struct.pack('<iiii', -1000, -1000, 1000, 1000)
        
        # No texture names since used_textures = 0
        
        # Chunk table (1 chunk = 4 bytes offset placeholder)
        # The offset points to the chunk data relative to chunk table start
        data += struct.pack('<I', 4)  # Offset to first (and only) chunk
        
        # Chunk header (16 bytes for counts + 32 bytes for 8 pointers = 48 bytes)
        g4_count = len(bone. quads)
        
        # Polygon counts:  F4, G4, FT4, GT4, F3, G3, FT3, GT3
        data += struct. pack('<HHHHHHHH', 0, g4_count, 0, 0, 0, 0, 0, 0)
        
        # 8 pointers (set at runtime, write zeros)
        data += struct.pack('<IIIIIIII', 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Write PG4 polygon data
        for quad in bone.quads:
            data += self._write_pg4(quad)
        
        return data
    
    def _write_animation(self, anim: Animation) -> bytes:
        """Write animation data"""
        data = b''
        
        # Animation header
        # name (8 bytes)
        name_bytes = anim.name.encode('ascii')[:8]
        name_bytes = name_bytes.ljust(8, b'\x00')
        data += name_bytes
        
        # keyframeamount (4 bytes)
        data += struct.pack('<I', anim.keyframe_count)
        
        # keyframe pointer (4 bytes, set at runtime)
        data += struct.pack('<I', 0)
        
        # Write keyframes:  organized as [bone][keyframe]
        for bone_idx in range(len(self.bones)):
            for kf_idx in range(anim.keyframe_count):
                if bone_idx < len(anim.keyframes) and kf_idx < len(anim.keyframes[bone_idx]):
                    transform = anim.keyframes[bone_idx][kf_idx]
                else:
                    transform = KeyframeTransform(
                        translation=(0.0, 0.0, 0.0),
                        rotation=(0.0, 0.0, 0.0, 1.0)
                    )
                data += self._write_matrix(transform)
        
        return data
    
    def write(self, filename: str):
        """Write the AAMF file"""
        data = b''
        
        # Header
        bone_count = len(self.bones)
        anim_count = len(self. animations)
        
        data += struct.pack('<HH', bone_count, anim_count)
        
        # Bone parent table
        for bone in self.bones:
            data += struct. pack('<HH', bone.index, bone.parent_index)
        
        # Bone AMF data blocks
        for bone in self.bones:
            amf_data = self._write_amf_for_bone(bone)
            block_size = 4 + len(amf_data)  # Include size field itself
            data += struct.pack('<I', block_size)
            data += amf_data
        
        # Animation data blocks
        for anim in self.animations:
            anim_data = self._write_animation(anim)
            block_size = 4 + len(anim_data)
            data += struct. pack('<I', block_size)
            data += anim_data
        
        # Write to file
        with open(filename, 'wb') as f:
            f.write(data)
        
        print(f"\nExported to {filename}")
        print(f"  Bones: {bone_count}")
        print(f"  Animations: {anim_count}")
        total_quads = sum(len(bone.quads) for bone in self.bones)
        print(f"  Total quads (G4): {total_quads}")
        print(f"  File size: {len(data)} bytes")


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.gltf [output. aamf]")
        print("\nConverts glTF 2.0 files to AAMF format for UnnamedHB1 PS1 homebrew.")
        print("\nFeatures:")
        print("  - Supports only G4 (gouraud-shaded quad) polygon type")
        print("  - Automatically merges adjacent triangles into quads")
        print("  - Remaining triangles are converted to degenerate quads")
        print("  - Preserves skeletal animation data")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else: 
        output_file = os.path.splitext(input_file)[0] + ". aamf"
    
    print(f"Reading {input_file}")
    
    parser = GltfParser(input_file)
    parser.load()
    
    print(f"\nParsed glTF:")
    print(f"  Bones:  {len(parser. bones)}")
    print(f"  Animations: {len(parser.animations)}")
    
    writer = AAMFWriter(parser.bones, parser.animations)
    writer.write(output_file)


if __name__ == "__main__": 
    main()