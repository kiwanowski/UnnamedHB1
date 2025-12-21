#!/usr/bin/env python3
"""
AMF to OBJ Converter - Working version for G4 polygons
Converts PlayStation 1 AMF model files to Wavefront OBJ format
"""

import struct
import sys
from dataclasses import dataclass

@dataclass
class WorldBounds:
    minX: int
    minZ: int
    maxX: int
    maxZ: int

@dataclass
class AMFHeader:
    used_textures:  int
    x:  int
    z: int
    bounds: WorldBounds

def parse_amf_header(data):
    """Parse AMF file header"""
    used_textures = struct.unpack_from('<I', data, 0)[0]
    x, z = struct.unpack_from('<HH', data, 4)
    minX, minZ, maxX, maxZ = struct.unpack_from('<iiii', data, 8)
    
    bounds = WorldBounds(minX, minZ, maxX, maxZ)
    header = AMFHeader(used_textures, x, z, bounds)
    
    return header

def read_svector(data, offset):
    """Read a PSX SVECTOR (4 int16 values:  x, y, z, pad)"""
    x, y, z, pad = struct.unpack_from('<hhhh', data, offset)
    return (x, y, z, pad), offset + 8

def read_cvector(data, offset):
    """Read a PSX CVECTOR (4 uint8 values: r, g, b, pad)"""
    r, g, b, pad = struct.unpack_from('<BBBB', data, offset)
    return (r, g, b, pad), offset + 4

def convert_psx_coord(value):
    """Convert PSX coordinate to float"""
    return value / 256.0

def normalize_vector(x, y, z):
    """Normalize a 3D vector"""
    length = (x*x + y*y + z*z) ** 0.5
    if length > 0:
        return (x/length, y/length, z/length)
    return (0, 0, 1)

class AMFConverter:
    def __init__(self, data):
        self.data = data
        self.vertices = []
        self.normals = []
        self.colors = []
        self.faces = []
        
    def parse(self):
        """Parse AMF file"""
        # Parse header
        header = parse_amf_header(self.data)
        print(f"AMF Header:")
        print(f"  Used textures: {header.used_textures}")
        print(f"  Grid size: {header.x} x {header.z}")
        print(f"  Bounds: minX={header.bounds.minX}, minZ={header.bounds.minZ}, maxX={header.bounds.maxX}, maxZ={header.bounds.maxZ}")
        
        # Parse texture names (8 bytes each)
        offset = 0x18  # After header (24 bytes)
        texture_names = []
        for i in range(header.used_textures):
            name_bytes = self.data[offset + i*8 : offset + i*8 + 8]
            name = name_bytes.decode('ascii', errors='ignore').rstrip('\x00')
            texture_names.append(name)
        print(f"  Texture names:  {texture_names}")
        
        # Chunk pointer table starts after texture names
        chunk_table_offset = offset + header.used_textures * 8
        chunk_count = header.x * header.z
        print(f"  Chunk count: {chunk_count}")
        print(f"  Chunk table offset: 0x{chunk_table_offset:04x}")
        
        # The chunk data starts after the chunk pointer table
        chunk_data_base = chunk_table_offset + chunk_count * 4
        print(f"  Chunk data base: 0x{chunk_data_base:04x}")
        
        vertex_index = 1  # OBJ uses 1-based indexing
        
        # Process each chunk
        for chunk_idx in range(chunk_count):
            # First chunk is right after the pointer table
            chunk_pos = chunk_data_base
            
            if chunk_pos + 48 > len(self.data):
                print(f"  Chunk {chunk_idx} exceeds file size, skipping")
                break
            
            # Read polygon type counts (16 bytes = 8 uint16)
            counts_offset = chunk_pos
            F4, G4, FT4, GT4, F3, G3, FT3, GT3 = struct. unpack_from('<8H', self.data, counts_offset)
            
            print(f"\nChunk {chunk_idx}:")
            print(f"  F4={F4}, G4={G4}, FT4={FT4}, GT4={GT4}")
            print(f"  F3={F3}, G3={G3}, FT3={FT3}, GT3={GT3}")
            
            # Polygon data starts after the chunk header
            # Header = 8 counts (16 bytes) + 8 pointers (32 bytes) = 48 bytes
            poly_data_offset = counts_offset + 48
            
            print(f"  Polygon data starts at offset: 0x{poly_data_offset:04x}")
            
            # PG4 structure (from model.h):
            # typedef struct {
            #     SVECTOR v0, v1, v2, v3;     // 4 * 8 = 32 bytes
            #     SVECTOR n0, n1, n2, n3;     // 4 * 8 = 32 bytes  
            #     CVECTOR c0, c1, c2, c3;     // 4 * 4 = 16 bytes
            #     POLY_G4 pol;                // ~20 bytes
            # } PG4;
            # Total ≈ 100 bytes per PG4
            
            # Parse F4 polygons (flat-shaded quads)
            PF4_SIZE = 32 + 20  # 4 vertices + POLY_F4
            for i in range(F4):
                if poly_data_offset + PF4_SIZE > len(self.data):
                    break
                poly_data_offset += PF4_SIZE
            
            # Parse G4 polygons (Gouraud-shaded quads)
            PG4_SIZE = 32 + 32 + 16 + 20  # vertices + normals + colors + POLY_G4
            print(f"  Parsing {G4} G4 polygons...")
            
            for i in range(G4):
                if poly_data_offset + PG4_SIZE > len(self.data):
                    print(f"    Polygon {i}:  Not enough data (need {PG4_SIZE} bytes, have {len(self. data) - poly_data_offset})")
                    break
                
                start_offset = poly_data_offset
                
                # Read 4 vertices (SVECTOR each = 8 bytes)
                v0, poly_data_offset = read_svector(self.data, poly_data_offset)
                v1, poly_data_offset = read_svector(self.data, poly_data_offset)
                v2, poly_data_offset = read_svector(self.data, poly_data_offset)
                v3, poly_data_offset = read_svector(self.data, poly_data_offset)
                
                # Read 4 normals (SVECTOR each = 8 bytes)
                n0, poly_data_offset = read_svector(self.data, poly_data_offset)
                n1, poly_data_offset = read_svector(self.data, poly_data_offset)
                n2, poly_data_offset = read_svector(self.data, poly_data_offset)
                n3, poly_data_offset = read_svector(self. data, poly_data_offset)
                
                # Read 4 colors (CVECTOR each = 4 bytes)
                c0, poly_data_offset = read_cvector(self.data, poly_data_offset)
                c1, poly_data_offset = read_cvector(self.data, poly_data_offset)
                c2, poly_data_offset = read_cvector(self.data, poly_data_offset)
                c3, poly_data_offset = read_cvector(self.data, poly_data_offset)
                
                # Skip POLY_G4 structure (~20 bytes)
                poly_data_offset += 20
                
                if i < 3:  # Debug first few polygons
                    print(f"    Polygon {i}:")
                    print(f"      v0=({v0[0]}, {v0[1]}, {v0[2]})")
                    print(f"      v1=({v1[0]}, {v1[1]}, {v1[2]})")
                    print(f"      v2=({v2[0]}, {v2[1]}, {v2[2]})")
                    print(f"      v3=({v3[0]}, {v3[1]}, {v3[2]})")
                    print(f"      c0=({c0[0]}, {c0[1]}, {c0[2]})")
                
                # Convert vertices to world coordinates
                v0_idx = vertex_index
                self.vertices.append((convert_psx_coord(v0[0]), convert_psx_coord(v0[1]), convert_psx_coord(v0[2])))
                self.vertices.append((convert_psx_coord(v1[0]), convert_psx_coord(v1[1]), convert_psx_coord(v1[2])))
                self.vertices.append((convert_psx_coord(v2[0]), convert_psx_coord(v2[1]), convert_psx_coord(v2[2])))
                self.vertices.append((convert_psx_coord(v3[0]), convert_psx_coord(v3[1]), convert_psx_coord(v3[2])))
                
                # Convert normals
                self.normals.append(normalize_vector(n0[0], n0[1], n0[2]))
                self.normals.append(normalize_vector(n1[0], n1[1], n1[2]))
                self.normals.append(normalize_vector(n2[0], n2[1], n2[2]))
                self.normals.append(normalize_vector(n3[0], n3[1], n3[2]))
                
                # Store colors (normalized to 0-1 range)
                self.colors.append((c0[0]/255.0, c0[1]/255.0, c0[2]/255.0))
                self.colors.append((c1[0]/255.0, c1[1]/255.0, c1[2]/255.0))
                self.colors.append((c2[0]/255.0, c2[1]/255.0, c2[2]/255.0))
                self.colors.append((c3[0]/255.0, c3[1]/255.0, c3[2]/255.0))
                
                # Create two triangles from quad (0-1-2 and 0-2-3)
                self.faces.append((v0_idx, v0_idx+1, v0_idx+2, v0_idx+3))
                #self.faces.append((v0_idx+1, v0_idx+2, v0_idx+3))
                
                vertex_index += 4
            
            print(f"  Processed {G4} G4 polygons")
            
            # Parse other polygon types if needed... 
            # (FT4, GT4, F3, G3, FT3, GT3)
        
        print(f"\nExtracted:")
        print(f"  {len(self.vertices)} vertices")
        print(f"  {len(self.normals)} normals")
        print(f"  {len(self.faces)} faces")
    
    def write_obj(self, output_path):
        """Write OBJ file"""
        with open(output_path, 'w') as f:
            f.write("# AMF to OBJ conversion\n")
            f.write(f"# Vertices: {len(self.vertices)}\n")
            f.write(f"# Faces: {len(self.faces)}\n\n")
            
            # Write vertices
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write normals
            if self.normals:
                f.write("\n")
                for n in self.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
            # Write faces
            f.write("\n")
            for face in self.faces:
                if self.normals:
                    f.write(f"f {face[0]}//{face[0]} {face[1]}//{face[1]} {face[3]}//{face[3]} {face[2]}//{face[2]}\n")
                else:
                    f. write(f"f {face[0]} {face[1]} {face[3]} {face[2]}\n")
        
        print(f"\nOBJ file written to:  {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python amf_to_obj. py <input. amf> [output.obj]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.rsplit('.', 1)[0] + '.obj'
    
    # Read AMF file
    if input_file == '-':
        data = sys.stdin.buffer.read()
    else:
        with open(input_file, 'rb') as f:
            data = f.read()
    
    print(f"Loaded {len(data)} bytes")
    print(f"First 64 bytes (hex): {data[:64].hex()}\n")
    
    # Convert
    converter = AMFConverter(data)
    converter.parse()
    
    if len(converter.vertices) > 0:
        converter.write_obj(output_file)
        print("\nConversion complete!")
    else:
        print("\nERROR: No vertices found.")

if __name__ == "__main__":
    main()