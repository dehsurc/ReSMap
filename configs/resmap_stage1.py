# Stage 1 (6 epochs): camera, satellite and fused BEV segmentation with PTF.
# No vector head. 2 GPUs x 3 samples.
_base_ = ['./_base_/resmap_nusc_60x30.py']

model = dict(mm_head_cfg=None)
data = dict(val=dict(eval_semantic=True), test=dict(eval_semantic=True))

optimizer = dict(
    type='AdamW',
    lr=3.5e-4,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=0.05)
runner = dict(type='IterBasedRunner', max_iters=27840)
work_dir = './work_dirs/resmap_stage1'
