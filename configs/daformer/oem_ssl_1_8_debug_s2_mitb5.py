# ---------------------------------------------------------------
# DAPCN-SSL debug run, Solution 2: DynAnchor AFTER decoder fusion
#
# Differs from oem_ssl_1_8_debug_mitb5.py in EXACTLY ONE setting:
#     anchor_after_fusion = True
# Everything else (data, split, seed, iters, batch, losses, RCS) is
# inherited, so any difference in the debug metrics is attributable
# to this switch alone.
#
# WHY THIS RUN EXISTS
# -------------------
# In Solution 1 (anchor_after_fusion=False) the prototypes live in
# encoder space (512-d) and are bridged to the classifier's 256-d
# space by `proto_to_decoder`, an nn.Linear.  That Linear is used ONLY
# inside `_compute_prototype_probs`, whose output is consumed by
# `torch.max` to build pseudo-label *targets* — never by a loss.  It
# therefore receives no gradient and stays at its random
# initialisation forever.  Verified on the Solution 1 checkpoint at
# iter 2000: AdamW had created no moment buffers for
# proto_to_decoder.weight/bias (state=False, steps=0) while every
# other trainable parameter had them.
#
# The consequence was measurable: across all ten 1000-iter
# checkpoints of the Solution 1 run, proto_mIoU sat at 2.5-3.1% with
# no trend while teacher_mIoU rose to 53%, and blend_mIoU tracked
# teacher_mIoU to within 0.0002 — the correction was a no-op.
#
# Solution 2 removes the broken bridge entirely: DynAnchor operates on
# the decoder's fused features, so prototypes are already 256-d and
# `conv_seg` applies to them directly (`proto_to_decoder` is None).
# The prototypes themselves DO receive gradient, via DAPGLoss.
#
# WHAT TO COMPARE
# ---------------
# The question this run answers is narrow: does p^c carry any class
# information once the random projection is gone?  Read debug/proto_mIoU
# first — if it stays near 3%, the problem is not the bridge and the
# correction needs a training signal of its own, not a rewiring.
#
# NOTE ON COST: the fused map is at stride 4 (128x128 for a 512 crop),
# so DynAnchor now sees 64x more tokens per image than in Solution 1
# (which used the stride-32 encoder stage). Expect higher memory and a
# slower iteration; batch size is deliberately left unchanged so the
# comparison stays clean.
# ---------------------------------------------------------------

_base_ = ['./oem_ssl_1_8_debug_mitb5.py']

uda = dict(anchor_after_fusion=True)

name = 'oem_ssl_1_8_debug_s2_mitb5'
name_uda = 'dapcn_ssl_affinity_rcs_s2'
