#!/usr/bin/env python3
"""
glTF to AAMF Converter for UnnamedHB1 PlayStation 1 Animated Model Format
"""

import struct
import sys
import os
import json
import base64
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set


FIXED_POINT_SCALE = 4096.0
DEBUG_MODE = False


def debug_print(*args, **kwargs):
    if DEBUG_MODE: 
        print(*args, **kwargs)


@dataclass
class Vertex:
    position: Tuple[float, float, float]
    normal:  Tuple[float, float, float] = (0.0, 1.0, 0.0)
    
    def __hash__(self):
        return hash((self.position, self.normal))
    
    def __eq__(self, other):
        if not isinstance(other, Vertex):
            return False
        return self. position == other.position and self.normal == other.normal


@dataclass
class Triangle:
    v0: Vertex
    v1: Vertex
    v2: Vertex
    
    def get_edge(self, index:  int):
        vertices = [self.v0, self.v1, self.v2]
        return (vertices[index]. position, vertices[(index + 1) % 3].position)
    
    def get_opposite_vertex(self, edge_index: int):
        vertices = [self.v0, self.v1, self.v2]
        return vertices[(edge_index + 2) % 3]
    
    def get_edge_vertices(self, edge_index: int):
        vertices = [self.v0, self.v1, self.v2]
        return (vertices[edge_index], vertices[(edge_index + 1) % 3])


@dataclass
class Quad:
    v0: Vertex
    v1: Vertex
    v2: Vertex
    v3: Vertex
    source_triangle_indices:  Tuple[int, int] = (-1, -1)
    is_degenerate: bool = False


@dataclass
class Bone:
    index: int
    parent_index: int
    quads: List[Quad] = field(default_factory=list)


@dataclass
class KeyframeTransform: 
    translation:  Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass
class Animation:
    name: str
    keyframe_count: int
    keyframes: List[List[KeyframeTransform]] = field(default_factory=list)


def is_planar(v0, v1, v2, v3, tolerance=0.1):
    e1 = (v1.position[0] - v0.position[0], v1.position[1] - v0.position[1], v1.position[2] - v0.position[2])
    e2 = (v2.position[0] - v0.position[0], v2.position[1] - v0.position[1], v2.position[2] - v0.position[2])
    
    normal = (
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0]
    )
    
    length = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
    if length < 1e-10:
        return True
    
    normal = (normal[0]/length, normal[1]/length, normal[2]/length)
    d = normal[0]*v0.position[0] + normal[1]*v0.position[1] + normal[2]*v0.position[2]
    dist = abs(normal[0]*v3.position[0] + normal[1]*v3.position[1] + normal[2]*v3.position[2] - d)
    
    return dist < tolerance


def merge_triangles_to_quads(triangles):
    quads = []
    used = set()
    remaining = []
    
    edge_to_triangle = {}
    
    for tri_idx, tri in enumerate(triangles):
        for edge_idx in range(3):
            edge = tri.get_edge(edge_idx)
            edge_key = tuple(sorted([edge[0], edge[1]], key=lambda p: (p[0], p[1], p[2])))
            if edge_key not in edge_to_triangle:
                edge_to_triangle[edge_key] = []
            edge_to_triangle[edge_key].append((tri_idx, edge_idx))
    
    for edge_key, tri_list in edge_to_triangle.items():
        if len(tri_list) != 2:
            continue
        
        (idx1, edge_idx1), (idx2, edge_idx2) = tri_list
        
        if idx1 in used or idx2 in used:
            continue
        
        tri1 = triangles[idx1]
        tri2 = triangles[idx2]
        
        shared_v1, shared_v2 = tri1.get_edge_vertices(edge_idx1)
        opp1 = tri1.get_opposite_vertex(edge_idx1)
        opp2 = tri2.get_opposite_vertex(edge_idx2)
        
        if not is_planar(shared_v1, shared_v2, opp1, opp2):
            continue
        
        tri1_verts = [tri1.v0, tri1.v1, tri1.v2]
        
        opp1_pos = -1
        for i, v in enumerate(tri1_verts):
            if v. position == opp1.position:
                opp1_pos = i
                break
        
        if opp1_pos == -1:
            continue
        
        next_vert = tri1_verts[(opp1_pos + 1) % 3]
        prev_vert = tri1_verts[(opp1_pos + 2) % 3]
        
        quad = Quad(
            v0=opp1, v1=prev_vert, v2=next_vert, v3=opp2,
            source_triangle_indices=(idx1, idx2), is_degenerate=False
        )
        
        quads.append(quad)
        used.add(idx1)
        used.add(idx2)
    
    for idx, tri in enumerate(triangles):
        if idx not in used:
            remaining.append(tri)
    
    return quads, remaining


def triangle_to_degenerate_quad(tri, tri_idx=-1):
    return Quad(
        v0=tri. v0, v1=tri.v1, v2=tri.v2, v3=tri.v2,
        source_triangle_indices=(tri_idx, -1), is_degenerate=True
    )


class GltfParser: 
    def __init__(self, filename):
        self.filename = filename
        self.gltf = {}
        self.buffers = []
        self.bones = []
        self.animations = []
        self.node_to_bone = {}
        self.joint_nodes = []
        
    def load(self):
        if self.filename.lower().endswith('.glb'):
            self._load_glb()
        else:
            with open(self.filename, 'r') as f:
                self. gltf = json.load(f)
            self._load_buffers()
        
        self._parse_scene()
        self._parse_animations()
    
    def _load_glb(self):
        with open(self.filename, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x46546C67:
                raise ValueError("Invalid GLB magic number")
            
            struct.unpack('<I', f.read(4))[0]  # version
            length = struct.unpack('<I', f. read(4))[0]
            
            while f.tell() < length:
                chunk_length = struct.unpack('<I', f. read(4))[0]
                chunk_type = struct.unpack('<I', f. read(4))[0]
                chunk_data = f.read(chunk_length)
                
                if chunk_type == 0x4E4F534A: 
                    self. gltf = json.loads(chunk_data.decode('utf-8'))
                elif chunk_type == 0x004E4942:
                    self.buffers.append(chunk_data)
    
    def _load_buffers(self):
        for buffer_info in self.gltf.get('buffers', []):
            uri = buffer_info. get('uri', '')
            if uri.startswith('data:'):
                _, data = uri.split(',', 1)
                self.buffers.append(base64.b64decode(data))
            elif uri: 
                buffer_path = os.path.join(os.path.dirname(self.filename), uri)
                with open(buffer_path, 'rb') as f:
                    self.buffers.append(f. read())
    
    def _get_accessor_data(self, accessor_idx):
        accessor = self.gltf['accessors'][accessor_idx]
        buffer_view = self. gltf['bufferViews'][accessor['bufferView']]
        
        buffer_data = self. buffers[buffer_view['buffer']]
        byte_offset = buffer_view. get('byteOffset', 0) + accessor. get('byteOffset', 0)
        
        component_type = accessor['componentType']
        accessor_type = accessor['type']
        count = accessor['count']
        
        component_formats = {5120: 'b', 5121: 'B', 5122: 'h', 5123: 'H', 5125: 'I', 5126: 'f'}
        type_counts = {'SCALAR': 1, 'VEC2':  2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
        
        fmt = component_formats[component_type]
        components = type_counts[accessor_type]
        component_size = struct.calcsize(fmt)
        byte_stride = buffer_view.get('byteStride', component_size * components)
        
        data = []
        for i in range(count):
            offset = byte_offset + i * byte_stride
            if components == 1:
                data.append(struct.unpack_from(f'<{fmt}', buffer_data, offset)[0])
            else:
                data. append(tuple(struct.unpack_from(f'<{components}{fmt}', buffer_data, offset)))
        
        return data
    
    def _get_animated_nodes(self) -> List[int]:
        """Get all node indices that have animation channels, in order of first appearance."""
        animated_nodes = []
        seen = set()
        for anim_data in self.gltf.get('animations', []):
            for channel in anim_data.get('channels', []):
                node_idx = channel. get('target', {}).get('node')
                if node_idx is not None and node_idx not in seen: 
                    animated_nodes.append(node_idx)
                    seen.add(node_idx)
        return animated_nodes
    
    def _parse_scene(self):
        mesh_node = None
        skin_data = None
        
        for node in self.gltf.get('nodes', []):
            if 'mesh' in node:
                mesh_node = node
                if 'skin' in node:
                    skin_data = self. gltf['skins'][node['skin']]
                break
        
        if not mesh_node: 
            print("No mesh found in glTF file!")
            return
        
        if skin_data:
            self. joint_nodes = skin_data.get('joints', [])
        
        mesh = self.gltf['meshes'][mesh_node['mesh']]
        
        # Collect triangles by joint index
        all_triangles_by_joint:  Dict[int, List[Triangle]] = {}
        
        for primitive in mesh.get('primitives', []):
            positions = self._get_accessor_data(primitive['attributes']['POSITION'])
            
            normals = [(0.0, 1.0, 0.0)] * len(positions)
            if 'NORMAL' in primitive['attributes']: 
                normals = self._get_accessor_data(primitive['attributes']['NORMAL'])
            
            joints_data = [(0, 0, 0, 0)] * len(positions)
            if 'JOINTS_0' in primitive['attributes']: 
                joints_data = self._get_accessor_data(primitive['attributes']['JOINTS_0'])
            
            indices = self._get_accessor_data(primitive['indices']) if 'indices' in primitive else list(range(len(positions)))
            
            for i in range(0, len(indices), 3):
                if i + 2 >= len(indices):
                    break
                
                i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
                v0 = Vertex(positions[i0], normals[i0])
                v1 = Vertex(positions[i1], normals[i1])
                v2 = Vertex(positions[i2], normals[i2])
                
                joint_idx = joints_data[i0][0]
                
                if joint_idx not in all_triangles_by_joint:
                    all_triangles_by_joint[joint_idx] = []
                all_triangles_by_joint[joint_idx].append(Triangle(v0, v1, v2))
        
        # Get joints with geometry (sorted by joint index)
        joints_with_geometry = sorted(all_triangles_by_joint. keys())
        
        # Get animated nodes
        animated_nodes = self._get_animated_nodes()
        
        print(f"\n=== DEBUG INFO ===")
        print(f"Skin joints array (node indices): {self.joint_nodes}")
        print(f"Joints with triangles: {joints_with_geometry}")
        print(f"Animated nodes (in order): {animated_nodes}")
        
        # Map animated nodes directly to bones with geometry
        # animated_nodes[0] -> bone 0, animated_nodes[1] -> bone 1, etc.
        num_bones = len(joints_with_geometry)
        for bone_idx, node_idx in enumerate(animated_nodes[: num_bones]):
            self.node_to_bone[node_idx] = bone_idx
            print(f"  Animated node {node_idx} -> Bone {bone_idx}")
        
        print(f"Node to bone mapping: {self.node_to_bone}")
        print(f"==================\n")
        
        # Build parent mapping based on joint hierarchy
        bone_parents = {}

        # Create a mapping from node index to joint index (index in joint_nodes array)
        node_to_joint_idx = {node_idx: joint_idx for joint_idx, node_idx in enumerate(self. joint_nodes)}

        # Create a mapping from old joint index to new bone index
        old_joint_to_new_bone = {old_joint_idx: new_bone_idx for new_bone_idx, old_joint_idx in enumerate(joints_with_geometry)}

        for new_bone_idx, old_joint_idx in enumerate(joints_with_geometry):
            bone_parents[new_bone_idx] = new_bone_idx  # default: self (root)
            
            if old_joint_idx < len(self.joint_nodes):
                node_idx = self.joint_nodes[old_joint_idx]
                
                # Find the parent node in the glTF hierarchy
                for potential_parent_node_idx, potential_parent in enumerate(self.gltf['nodes']):
                    if 'children' in potential_parent and node_idx in potential_parent['children']: 
                        # Check if this parent node is also a joint in the skin
                        if potential_parent_node_idx in node_to_joint_idx:
                            parent_joint_idx = node_to_joint_idx[potential_parent_node_idx]
                            # Check if that joint has geometry and thus became a bone
                            if parent_joint_idx in old_joint_to_new_bone: 
                                bone_parents[new_bone_idx] = old_joint_to_new_bone[parent_joint_idx]
                        break
        
        # Create bones
        for new_bone_idx, old_joint_idx in enumerate(joints_with_geometry):
            triangles = all_triangles_by_joint. get(old_joint_idx, [])
            quads, remaining = merge_triangles_to_quads(triangles)
            
            for tri_idx, tri in enumerate(remaining):
                quads.append(triangle_to_degenerate_quad(tri, tri_idx))
            
            parent_idx = bone_parents.get(new_bone_idx, new_bone_idx)
            bone = Bone(index=new_bone_idx, parent_index=parent_idx, quads=quads)
            self.bones.append(bone)
            
            print(f"Bone {new_bone_idx}: {len(triangles)} triangles -> {len(quads)} quads (parent={parent_idx}, was joint {old_joint_idx})")
        
        if not self.bones:
            self.bones.append(Bone(index=0, parent_index=0, quads=[]))
    
    def _parse_animations(self):
        print(f"\n=== ANIMATION DEBUG ===")
        print(f"Node to bone mapping: {self.node_to_bone}")
        
        for anim_data in self. gltf.get('animations', []):
            name = anim_data.get('name', 'anim')[: 8]
            
            if not anim_data.get('samplers'):
                continue
            
            first_sampler = anim_data['samplers'][0]
            times = self._get_accessor_data(first_sampler['input'])
            keyframe_count = len(times)
            
            print(f"\nAnimation '{name}':  {keyframe_count} keyframes")
            
            bone_keyframes = []
            for _ in range(len(self.bones)):
                bone_keyframes.append([
                    KeyframeTransform(
                        translation=(0.0, 0.0, 0.0),
                        rotation=(0.0, 0.0, 0.0, 1.0)
                    ) for _ in range(keyframe_count)
                ])
            
            for channel in anim_data.get('channels', []):
                sampler = anim_data['samplers'][channel['sampler']]
                target = channel['target']
                
                node_idx = target. get('node', 0)
                path = target.get('path', '')
                
                if node_idx in self. node_to_bone:
                    bone_idx = self. node_to_bone[node_idx]
                    print(f"  Channel:  node {node_idx} -> bone {bone_idx}, path '{path}' [MAPPED]")
                    
                    output_data = self._get_accessor_data(sampler['output'])
                    
                    for kf_idx, value in enumerate(output_data):
                        if kf_idx >= keyframe_count:
                            break
                        
                        current = bone_keyframes[bone_idx][kf_idx]
                        
                        if path == 'translation':
                            bone_keyframes[bone_idx][kf_idx] = KeyframeTransform(
                                translation=value, rotation=current.rotation
                            )
                        elif path == 'rotation':
                            bone_keyframes[bone_idx][kf_idx] = KeyframeTransform(
                                translation=current.translation, rotation=value
                            )
                else:
                    print(f"  Channel: node {node_idx} -> ??? path '{path}' [SKIPPED]")
            
            self.animations.append(Animation(name=name, keyframe_count=keyframe_count, keyframes=bone_keyframes))
        
        print(f"=======================\n")


class AAMFWriter: 
    def __init__(self, bones, animations):
        self.bones = bones
        self.animations = animations
    
    def _float_to_fixed(self, value):
        result = int(value * FIXED_POINT_SCALE)
        return max(-32768, min(32767, result))
    
    def _float_to_fixed32(self, value):
        result = int(value * FIXED_POINT_SCALE)
        return max(-2147483648, min(2147483647, result))
    
    def _write_svector(self, vx, vy, vz, pad=0):
        return struct.pack('<hhhh', vx, vy, vz, pad)
    
    def _write_matrix(self, transform):
        qx, qy, qz, qw = transform.rotation
        
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz
        
        m = [
            [1.0 - 2.0*(yy + zz), 2.0*(xy - wz), 2.0*(xz + wy)],
            [2.0*(xy + wz), 1.0 - 2.0*(xx + zz), 2.0*(yz - wx)],
            [2.0*(xz - wy), 2.0*(yz + wx), 1.0 - 2.0*(xx + yy)]
        ]
        
        data = b''
        for row in range(3):
            for col in range(3):
                data += struct.pack('<h', self._float_to_fixed(m[row][col]))
        
        data += struct.pack('<h', 0)
        
        tx, ty, tz = transform.translation
        data += struct. pack('<iii',
            self._float_to_fixed32(tx),
            self._float_to_fixed32(ty),
            self._float_to_fixed32(tz)
        )
        
        return data
    
    def _write_poly_g4(self):
        data = struct.pack('<I', 0x38000000)
        for i in range(4):
            data += struct.pack('<BBBB', 128, 128, 128, 0x38 if i == 0 else 0)
            data += struct.pack('<hh', 0, 0)
        return data
    
    def _write_pg4(self, quad):
        data = b''
        for v in [quad.v0, quad.v1, quad.v2, quad.v3]: 
            data += self._write_svector(
                self._float_to_fixed(v.position[0]),
                self._float_to_fixed(v.position[1]),
                self._float_to_fixed(v.position[2])
            )
        for v in [quad.v0, quad.v1, quad.v2, quad.v3]:
            data += self._write_svector(
                self._float_to_fixed(v. normal[0]),
                self._float_to_fixed(v.normal[1]),
                self._float_to_fixed(v.normal[2])
            )
        data += self._write_poly_g4()
        return data
    
    def _write_amf_for_bone(self, bone):
        data = b''
        data += struct.pack('<I', 0)
        data += struct. pack('<HH', 1, 1)
        data += struct.pack('<iiii', -1000, -1000, 1000, 1000)
        data += struct.pack('<I', 4)
        
        g4_count = len(bone. quads)
        data += struct.pack('<HHHHHHHH', 0, g4_count, 0, 0, 0, 0, 0, 0)
        data += struct.pack('<IIIIIIII', 0, 0, 0, 0, 0, 0, 0, 0)
        
        for quad in bone.quads:
            data += self._write_pg4(quad)
        
        return data
    
    def _write_animation(self, anim):
        data = b''
        data += anim.name.encode('ascii')[:8]. ljust(8, b'\x00')
        data += struct.pack('<I', anim.keyframe_count)
        data += struct.pack('<I', 0)
        
        for bone_idx in range(len(self.bones)):
            for kf_idx in range(anim. keyframe_count):
                if bone_idx < len(anim.keyframes) and kf_idx < len(anim.keyframes[bone_idx]):
                    transform = anim.keyframes[bone_idx][kf_idx]
                else:
                    transform = KeyframeTransform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
                data += self._write_matrix(transform)
        
        return data
    
    def write(self, filename):
        data = b''
        
        bone_count = len(self. bones)
        anim_count = len(self.animations)
        
        data += struct.pack('<HH', bone_count, anim_count)
        
        for bone in self. bones:
            data += struct.pack('<HH', bone.index, bone.parent_index)
        
        for bone in self.bones:
            amf_data = self._write_amf_for_bone(bone)
            data += struct.pack('<I', 4 + len(amf_data))
            data += amf_data
        
        for anim in self.animations:
            anim_data = self._write_animation(anim)
            data += struct.pack('<I', 4 + len(anim_data))
            data += anim_data
        
        with open(filename, 'wb') as f:
            f.write(data)
        
        print(f"\nExported to {filename}")
        print(f"  Bones: {bone_count}")
        print(f"  Animations: {anim_count}")
        print(f"  Total quads: {sum(len(b.quads) for b in self.bones)}")
        print(f"  File size: {len(data)} bytes")


def main():
    global DEBUG_MODE
    
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} input.gltf [output. aamf] [--debug]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = None
    
    for arg in sys.argv[2:]:
        if arg == '--debug':
            DEBUG_MODE = True
        elif not output_file:
            output_file = arg
    
    if not output_file:
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