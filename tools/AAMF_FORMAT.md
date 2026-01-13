# AAMF Format Specification

This document describes both the legacy and new skinned AAMF (Animated AMF) format for the UnnamedHB1 PlayStation 1 game.

## Overview

AAMF is a custom binary format for storing animated skeletal meshes optimized for PS1 hardware. There are two versions:

1. **Legacy AAMF** - One complete AMF mesh per bone (rigid binding)
2. **Skinned AAMF (SAMF)** - Single unified mesh with vertex skinning weights (smooth deformation)

## Legacy AAMF Format

The original format stores a separate complete mesh for each bone in the skeleton.

### Header (4 bytes)
```
uint16_t bone_count;
uint16_t anim_count;
```

### Bone Parent Table
```
For each bone (bone_count):
  uint16_t bone_idx;
  uint16_t parent_idx;
```

### Bone Data Blocks
```
For each bone (bone_count):
  uint32_t block_size;
  [AMF mesh data - see AMF format]
```

### Animation Data Blocks
```
For each animation (anim_count):
  uint32_t block_size;
  char name[8];
  uint32_t keyframe_count;
  uint32_t keyframe_pointer;  // Runtime only
  
  Keyframes (bone_count * keyframe_count):
    MATRIX transform;  // 32 bytes each
```

### Limitations
- Each vertex belongs to exactly one bone (weight = 1.0)
- No smooth deformation at joints
- Duplicated vertices at bone boundaries
- Larger file sizes due to mesh duplication

## Skinned AAMF Format (SAMF v2)

The new format uses a single unified mesh with vertex skinning weights for smooth deformation.

### Header (16 bytes)
```
uint32_t magic;          // 'SAMF' (0x464D4153) or 'AAMF' (0x464D4141)
uint16_t version;        // 2 for skinned format
uint16_t bone_count;     // Number of bones
uint16_t anim_count;     // Number of animations
uint32_t vertex_count;   // Total vertices in unified mesh
uint16_t face_count;     // Total triangular faces
```

### Bone Parent Table
```
For each bone (bone_count):
  uint16_t bone_idx;
  uint16_t parent_idx;
```

### Bind Pose Transforms
```
For each bone (bone_count):
  MATRIX bind_pose;  // 32 bytes - bind pose transform
```

### Unified Mesh Data

#### Vertices
```
For each vertex (vertex_count):
  SVECTOR position;  // 8 bytes (4 x int16_t)
```

#### Normals
```
For each vertex (vertex_count):
  SVECTOR normal;  // 8 bytes (4 x int16_t)
```

#### Skin Weights
```
For each vertex (vertex_count):
  uint8_t bone_indices[4];   // Up to 4 bone influences
  uint8_t weights[4];        // Normalized 0-255 (divide by 255 to get 0.0-1.0)
```

**Constraints:**
- Maximum 4 bone influences per vertex (PS1 hardware limitation)
- Weights normalized to sum to 255 (or 1.0 when converted)
- Unused slots filled with 0

#### Faces
```
For each face (face_count):
  uint16_t v0, v1, v2;  // Triangle vertex indices
```

### Animation Data Blocks
Same format as legacy AAMF:
```
For each animation (anim_count):
  uint32_t block_size;
  char name[8];
  uint32_t keyframe_count;
  uint32_t keyframe_pointer;  // Runtime only
  
  Keyframes (bone_count * keyframe_count):
    MATRIX transform;  // 32 bytes each
```

### Advantages
- Smooth deformation at joints
- Shared vertices reduce memory usage
- Supports multi-bone influences
- Compatible with modern 3D tools (glTF export/import)

## Data Types

### SVECTOR (8 bytes)
```c
struct SVECTOR {
    int16_t vx;   // X coordinate (fixed-point 4.12)
    int16_t vy;   // Y coordinate (fixed-point 4.12)
    int16_t vz;   // Z coordinate (fixed-point 4.12)
    int16_t pad;  // Padding
};
```

### MATRIX (32 bytes)
```c
struct MATRIX {
    int16_t m[3][3];  // 3x3 rotation matrix (fixed-point 4.12)
    int16_t pad;      // Padding
    int32_t t[3];     // Translation vector (fixed-point 4.12)
};
```

### SkinWeight (8 bytes)
```c
struct SkinWeight {
    uint8_t bone_indices[4];  // Up to 4 bone influences
    uint8_t weights[4];       // Normalized weights (0-255)
};
```

## Fixed-Point Format

All coordinates and matrix values use 4.12 fixed-point format:
- Integer part: 4 bits (sign + 3 bits)
- Fractional part: 12 bits
- Scale: multiply float by 4096.0 to convert to fixed-point

Example:
```
1.0 → 4096 (0x1000)
0.5 → 2048 (0x0800)
-1.0 → -4096 (0xF000)
```

## Tools

### Python Converters

#### aamf2gltf.py
Converts AAMF files to glTF 2.0 format.

**Usage:**
```bash
python aamf2gltf.py input.aamf [output.gltf]
```

**Features:**
- Auto-detects legacy vs skinned format
- Exports proper vertex weights for skinned meshes
- Maintains backward compatibility with legacy AAMF
- Exports skeletal animations

#### gltf2aamf.py
Converts glTF 2.0 files to AAMF format.

**Usage:**
```bash
# Export legacy format (one mesh per bone)
python gltf2aamf.py input.gltf [output.aamf]

# Export skinned format (unified mesh with weights)
python gltf2aamf.py input.gltf [output.aamf] --skinned
```

**Options:**
- `--skinned` - Export in skinned AAMF format (SAMF v2)
- `--debug` - Enable debug output

**Features:**
- Reads glTF skinned meshes with proper vertex weights
- Validates bone influence limits (max 4 per vertex)
- Extracts skeleton hierarchy and animations
- Normalizes vertex weights

### C Runtime

#### Legacy AAMF
```c
#include "anim_model.h"

AAMF model;
aamfInitData(&model, aamf_file_data);
activateAnimation(&model, "walk");
setAnimationKeyframe(&model, frame_number);
```

#### Skinned AAMF
```c
#include "anim_model.h"

SAMF model;
MATRIX bone_matrices[MAX_BONES];

samfInitData(&model, samf_file_data);
activateAnimationSkinned(&model, "walk");
setAnimationKeyframeSkinned(&model, frame_number, bone_matrices);

// TODO: Implement vertex skinning with GTE
// Apply bone_matrices to vertices using skin weights
```

## Vertex Skinning Algorithm

For rendering skinned meshes on PS1, you need to:

1. **Update bone transforms** from animation keyframes
2. **Transform each vertex** using weighted bone influences:

```c
for each vertex v:
    result = (0, 0, 0)
    total_weight = 0
    
    for i in 0..3:
        bone_idx = v.skin_weights.bone_indices[i]
        weight = v.skin_weights.weights[i] / 255.0
        
        if weight > 0:
            // Transform vertex by bone matrix
            transformed = bone_matrices[bone_idx] * v.position
            result += transformed * weight
            total_weight += weight
    
    // Normalize (should already sum to 1.0)
    if total_weight > 0:
        result /= total_weight
    
    output_vertex = result
```

On PS1 hardware, use GTE (Geometry Transfer Engine) for matrix multiplication.

## Round-Trip Conversion

The tools support round-trip conversion for validation:

```bash
# glTF → SAMF → glTF
python gltf2aamf.py model.gltf model.aamf --skinned
python aamf2gltf.py model.aamf model_roundtrip.gltf

# Compare original and round-trip glTF files
# Vertex positions, normals, weights should match
```

## File Size Comparison

Example for a character with 4 bones and 300 triangles:

**Legacy AAMF:**
- 4 separate meshes (75 triangles each)
- ~600 unique vertices (duplicated at joints)
- File size: ~40KB

**Skinned AAMF:**
- 1 unified mesh (300 triangles)
- ~200 shared vertices
- File size: ~15KB (62% reduction)

## References

- PSn00bSDK: https://github.com/Lameguy64/PSn00bSDK
- glTF 2.0 Specification: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- PS1 GTE Documentation: PSX Programming Guide
