# Skinned AAMF Format - Usage Examples

This guide demonstrates how to use the new skinned AAMF format converters.

## Prerequisites

- Python 3.6 or later
- glTF 2.0 file with skinned mesh and animations

## Converting glTF to Skinned AAMF

### Basic Usage

Convert a glTF file with skinned mesh to the new SAMF format:

```bash
python gltf2aamf.py character.gltf character.aamf --skinned
```

This will:
- Extract the skinned mesh with vertex weights
- Validate bone influences (max 4 per vertex)
- Export animations
- Create a smaller file compared to legacy format

### Legacy Format (Backward Compatible)

To export in the legacy format (one mesh per bone):

```bash
python gltf2aamf.py character.gltf character.aamf
```

## Converting AAMF to glTF

The converter automatically detects the format:

```bash
# Works with both legacy and skinned formats
python aamf2gltf.py character.aamf character.gltf
```

## Round-Trip Verification

Verify that conversions preserve data:

```bash
# Start with a glTF file
python gltf2aamf.py original.gltf test.aamf --skinned

# Convert back to glTF
python aamf2gltf.py test.aamf roundtrip.gltf

# Compare the two glTF files
# Vertex positions, normals, and weights should match
```

## Example Workflow

### 1. Create a Skinned Character in Blender

1. Model your character
2. Create an armature (skeleton)
3. Paint vertex weights
4. Create animations
5. Export as glTF 2.0 with these settings:
   - Format: glTF Separate (.gltf + .bin + textures)
   - Include: Selected Objects
   - Transform: +Y Up
   - Geometry: Apply Modifiers, UVs, Normals
   - Animation: Export Deformation Bones Only

### 2. Convert to SAMF Format

```bash
python gltf2aamf.py character.gltf character.aamf --skinned
```

Expected output:
```
Reading character.gltf
Export mode: Skinned AAMF (SAMF)
Bone 0: parent=0
Bone 1: parent=0
Bone 2: parent=1
...

Parsed skinned mesh:
  Bones: 8
  Vertices: 426
  Faces: 142

Exported skinned AAMF to character.aamf
  Format: SAMF v2 (skinned)
  Bones: 8
  Animations: 3
  Vertices: 426
  Faces: 142
  Max bone influences per vertex: 4
  File size: 76976 bytes
```

### 3. Use in PS1 Game (C Code)

```c
#include "anim_model.h"

// Load skinned model
SAMF character;
MATRIX bone_matrices[8];  // One per bone

samfInitData(&character, character_aamf_data);

// Activate walk animation
activateAnimationSkinned(&character, "walk");

// In your game loop
void update_character(int frame) {
    // Update bone transforms
    setAnimationKeyframeSkinned(&character, frame, bone_matrices);
    
    // TODO: Implement vertex skinning
    // For each vertex:
    //   1. Get vertex.position and vertex.skin_weights
    //   2. Transform by up to 4 bone matrices
    //   3. Blend using vertex.weights
    //   4. Output transformed vertex
}
```

### 4. Verify Output (Optional)

Convert back to glTF to verify in Blender:

```bash
python aamf2gltf.py character.aamf character_verify.gltf
```

Open `character_verify.gltf` in Blender to verify:
- Mesh geometry is correct
- Vertex weights are preserved
- Animations play correctly

## Format Comparison

### Legacy AAMF
- **Pros**: Simple, well-tested
- **Cons**: Larger files, rigid binding, visible seams at joints
- **Use Case**: Simple models, static poses, backward compatibility

### Skinned AAMF (SAMF)
- **Pros**: Smooth deformation, smaller files, shared vertices
- **Cons**: Requires vertex skinning implementation in renderer
- **Use Case**: Characters, organic models, smooth animation

## File Size Example

For a simple character with 4 bones:

| Format | Vertices | File Size | Notes |
|--------|----------|-----------|-------|
| Legacy AAMF | 600 (duplicated) | 40 KB | One mesh per bone |
| Skinned AAMF | 200 (shared) | 15 KB | Single unified mesh |
| **Reduction** | **67%** | **62%** | Significant savings |

## Tips

1. **Keep bone count low**: PS1 has limited memory
2. **Limit influences**: Max 4 bones per vertex for performance
3. **Optimize mesh**: Reduce polygon count for PS1 hardware
4. **Test animations**: Verify in glTF before converting
5. **Use debug mode**: Add `--debug` flag for detailed output

## Troubleshooting

### "Max bone influences per vertex: 8"

Your model has too many bone influences. In Blender:
1. Select your mesh
2. Go to Weight Paint mode
3. Use Weights → Limit Total to max 4
4. Export again

### File size not reduced

Your model may already be optimized or have many separate parts.
The skinned format works best when:
- Vertices are shared across bones
- The mesh has smooth deformation
- Multiple bones affect the same vertices

### Animation not working

Make sure:
- Animations are baked (not using constraints)
- All bones in the armature are exported
- Animation is assigned to the armature, not individual bones

## Advanced: Batch Conversion

Convert multiple files:

```bash
# Convert all glTF files to skinned AAMF
for file in models/*.gltf; do
    python gltf2aamf.py "$file" "${file%.gltf}.aamf" --skinned
done

# Verify all conversions
for file in models/*.aamf; do
    python aamf2gltf.py "$file" "verified/${file%.aamf}.gltf"
done
```

## References

- [AAMF_FORMAT.md](AAMF_FORMAT.md) - Complete format specification
- [glTF 2.0 Specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- [PSn00bSDK Documentation](https://github.com/Lameguy64/PSn00bSDK)
