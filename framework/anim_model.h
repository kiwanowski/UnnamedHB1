#ifndef ANIM_MODEL_H
#define ANIM_MODEL_H

#include <psxgte.h>
#include <inline_c.h>

#include "scene.h"
#include "model.h"

typedef struct {
    MATRIX mat;
} Keyframe;

typedef struct {
    char name[8];
    uint32_t keyframeamount;
    Keyframe* keyframe;
} Animation;

typedef struct {
    uint16_t boneamount; 
    uint16_t animamount; 
    Node* root;
    Node** nodes; // 0 is body
    Animation** anims;
    uint32_t active_anim;
} AAMF;

// Skinned AAMF format structures
typedef struct {
    uint8_t bone_indices[4];   // Up to 4 bone influences
    uint8_t weights[4];        // Normalized weights (0-255)
} SkinWeight;

typedef struct {
    uint32_t magic;            // 'AAMF' or 'SAMF'
    uint16_t version;          // 2 for skinned format
    uint16_t bone_count;
    uint16_t anim_count;
    uint32_t vertex_count;
    uint16_t face_count;
} SAMFHeader;

typedef struct {
    uint16_t bone_idx;
    uint16_t parent_idx;
} BoneParent;

typedef struct {
    uint16_t v0, v1, v2;       // Triangle vertex indices
} Face;

typedef struct {
    SAMFHeader header;
    BoneParent* bone_parents;   // Array of bone parent relationships
    MATRIX* bind_poses;         // Array of bind pose transforms
    SVECTOR* vertices;          // Unified vertex array
    SVECTOR* normals;           // Unified normal array
    SkinWeight* skin_weights;   // Vertex skin weights
    Face* faces;                // Triangle faces
    Animation** anims;          // Animation data
    uint32_t active_anim;
} SAMF;

void initAnimatedModels();

void aamfInitData(AAMF* aamf, unsigned char* data);
void samfInitData(SAMF* samf, unsigned char* data);

void activateAnimation(AAMF* aamf, char name[8]);
void setAnimationKeyframe(AAMF* aamf, uint16_t keyframe);

// Skinned AAMF functions
void activateAnimationSkinned(SAMF* samf, char name[8]);
void setAnimationKeyframeSkinned(SAMF* samf, uint16_t keyframe, MATRIX* bone_matrices);

#endif // _ANIM_MODEL_H