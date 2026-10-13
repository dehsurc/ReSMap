# Stage 2 (6 epochs): OCMQ head on top of the frozen stage-1 BEV model.
_base_ = ['./_base_/resmap_nusc_60x30.py']

model = dict(freeze_bev=True, freeze_sat_encoder=True)
load_from = './work_dirs/resmap_stage1/latest.pth'

optimizer = dict(
    type='AdamW',
    lr=1.4e-4,
    paramwise_cfg=dict(
        custom_keys=dict(
            img_backbone=dict(lr_mult=0.0),
            sat_encoder=dict(lr_mult=0.0),
            seg_decoder=dict(lr_mult=0.0),
            seg_decoder_sat=dict(lr_mult=0.0),
            seg_decoder_fused=dict(lr_mult=0.0),
            mm_head=dict(lr_mult=1.0))),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=0.1)
runner = dict(type='IterBasedRunner', max_iters=27840)
evaluation = dict(save_best='mAP', rule='greater')
work_dir = './work_dirs/resmap_stage2'
