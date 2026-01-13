#include "anim_model.h"
#include <string.h>

extern unsigned char enemy_aamf_data[];
AAMF enemy_aamf[5];

void initAnimatedModels() {
    for (int i = 0; i < 5; ++i) {
        aamfInitData(&enemy_aamf[i], enemy_aamf_data);
    }
}

void aamfInitData(AAMF* aamf, unsigned char* data) {
    uint16_t* d16 = (uint16_t*) data;
    aamf->boneamount = *(d16);
    aamf->animamount = *(d16+1);
    aamf->nodes = malloc(sizeof(Node*) * aamf->boneamount);
    aamf->anims = malloc(sizeof(Animation) * aamf->animamount);
    aamf->active_anim = 0;
    aamf->root = new_node(NULL, NULL);

    uint32_t offset = 0;
    for (uint16_t i = 0; i < aamf->boneamount; ++i) {
        AMF* amf = malloc(sizeof(AMF));
        amfInitData(amf, data+(4+aamf->boneamount*4 + offset + 4));
        // printf("%d off %08x, %.8s\n", i, 4+aamf->boneamount*4+offset+4, data+(4+aamf->boneamount*4 + offset + 4 + 0x18));
        offset += *(uint32_t*)(data+4+aamf->boneamount*4 + offset);
        uint16_t a = *(d16+2+i*2+0);
        uint16_t b = *(d16+2+i*2+1);
        assert(a == i && b <= a);
        if (a == b) aamf->nodes[i] = add_child(aamf->root, amf);
        else aamf->nodes[i] = add_child(aamf->nodes[b], amf);
    }

    for (uint16_t i = 0; i < aamf->animamount; ++i) {
        Animation* anim = (Animation*)(data+4+aamf->boneamount*4 + offset + 4);
        anim->keyframe = (Keyframe*)(data+4+aamf->boneamount*4 + offset + 4 + 16);
        offset += *(uint32_t*)(data+4+aamf->boneamount*4 + offset);
        aamf->anims[i] = anim;
    }
}

void activateAnimation(AAMF* aamf, char name[8]) {
    for (int i = 0; i < aamf->animamount; ++i) {
        if (memcmp(name, aamf->anims[i]->name, 8) == 0) {
            aamf->active_anim = i;
            return;
        }
    } 
}

void setAnimationKeyframe(AAMF* aamf, uint16_t keyframe) {
    uint16_t keyframeamount = aamf->anims[aamf->active_anim]->keyframeamount;
    keyframe = keyframe % keyframeamount;
    // printf("f %d\n", keyframe);
    for (int i = 0; i < aamf->boneamount; ++i) {
        Keyframe* k = &aamf->anims[aamf->active_anim]->keyframe[keyframe + i*keyframeamount];
        
        aamf->nodes[i]->mtx = k->mat;
    }
}

// Skinned AAMF (SAMF) functions

void samfInitData(SAMF* samf, unsigned char* data) {
    uint32_t* d32 = (uint32_t*) data;
    uint16_t* d16 = (uint16_t*) data;
    
    // Check magic
    samf->header.magic = *d32;
    if (samf->header.magic != 0x464D4153 && // 'SAMF'
        samf->header.magic != 0x464D4141) {  // 'AAMF'
        // Invalid format
        return;
    }
    
    // Parse header (16 bytes)
    samf->header.version = *(d16 + 2);
    samf->header.bone_count = *(d16 + 3);
    samf->header.anim_count = *(d16 + 4);
    samf->header.vertex_count = *(d32 + 2) >> 16 | (*(d32 + 3) & 0xFFFF);
    samf->header.face_count = *(d16 + 7);
    
    uint32_t offset = 16;
    
    // Parse bone parent table
    samf->bone_parents = (BoneParent*)(data + offset);
    offset += samf->header.bone_count * 4;
    
    // Parse bind pose transforms
    samf->bind_poses = (MATRIX*)(data + offset);
    offset += samf->header.bone_count * 32;
    
    // Parse unified mesh data
    samf->vertices = (SVECTOR*)(data + offset);
    offset += samf->header.vertex_count * 8;
    
    samf->normals = (SVECTOR*)(data + offset);
    offset += samf->header.vertex_count * 8;
    
    samf->skin_weights = (SkinWeight*)(data + offset);
    offset += samf->header.vertex_count * 8;
    
    samf->faces = (Face*)(data + offset);
    offset += samf->header.face_count * 6;
    
    // Parse animations
    samf->anims = malloc(sizeof(Animation*) * samf->header.anim_count);
    samf->active_anim = 0;
    
    for (uint16_t i = 0; i < samf->header.anim_count; ++i) {
        uint32_t block_size = *(uint32_t*)(data + offset);
        Animation* anim = (Animation*)(data + offset + 4);
        anim->keyframe = (Keyframe*)(data + offset + 4 + 16);
        samf->anims[i] = anim;
        offset += block_size;
    }
}

void activateAnimationSkinned(SAMF* samf, char name[8]) {
    for (int i = 0; i < samf->header.anim_count; ++i) {
        if (memcmp(name, samf->anims[i]->name, 8) == 0) {
            samf->active_anim = i;
            return;
        }
    }
}

void setAnimationKeyframeSkinned(SAMF* samf, uint16_t keyframe, MATRIX* bone_matrices) {
    uint16_t keyframeamount = samf->anims[samf->active_anim]->keyframeamount;
    keyframe = keyframe % keyframeamount;
    
    // Update bone matrices from animation keyframes
    for (int i = 0; i < samf->header.bone_count; ++i) {
        Keyframe* k = &samf->anims[samf->active_anim]->keyframe[keyframe + i*keyframeamount];
        bone_matrices[i] = k->mat;
    }
}
