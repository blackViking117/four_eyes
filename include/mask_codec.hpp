/****************************************************************************
 * mask_codec.hpp  -- single source of truth for the mono8 obstacle mask.
 *
 * One byte per pixel:
 *     bits [7:3]  instance id : 1..31   (0 => background)
 *     bits [2:0]  class id    : 0..7    (your obstacle labels are 0..5)
 *
 * value 0 == background. Instances are painted 1-based, so any real obstacle
 * pixel is >= (1<<3)=8; the value 0 can never be a painted obstacle.
 *
 * Include this in BOTH:
 *   - mask_builder.hpp / main_cam_opt.cpp  (encode, when g_mask.single_channel)
 *   - fusion node (decode, in projectAllCameras)
 * so the encoder and decoder are guaranteed to agree.
 ****************************************************************************/
#pragma once
#include <cstdint>

namespace maskcodec {

constexpr int INST_BITS  = 5;
constexpr int CLASS_BITS = 3;
constexpr int MAX_INST   = (1 << INST_BITS)  - 1;   // 31 instances per camera
constexpr int MAX_CLASS  = (1 << CLASS_BITS) - 1;   // class 0..7
constexpr int CLASS_MASK = MAX_CLASS;               // 0x07

// Pack (instance, class) -> mono8. Caller should cap inst at MAX_INST already;
// we clamp defensively so a stray index can never wrap into another instance.
inline uint8_t encode(int inst, int cls) {
    if (inst < 1)         inst = 1;
    if (inst > MAX_INST)  inst = MAX_INST;
    cls &= CLASS_MASK;
    return static_cast<uint8_t>((inst << CLASS_BITS) | cls);
}

// Unpack mono8 -> (instance, class). Returns false for background (value 0).
inline bool decode(uint8_t v, int& inst, int& cls) {
    if (v == 0) return false;
    inst = v >> CLASS_BITS;
    cls  = v &  CLASS_MASK;
    return inst >= 1;
}

} // namespace maskcodec