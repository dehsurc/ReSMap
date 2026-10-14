# ReSMap on the nuScenes geographically disjoint split (newsplit), 60 m x 30 m ROI.
# Shared by the three training stages.
plugin = True
plugin_dir = 'plugin/'

data_root = './datasets/nuscenes'
aid4ad_root = './datasets/nuscenes/AID4AD_frames'
train_ann = data_root + '/nuscenes_map_infos_train_newsplit.pkl'
val_ann = data_root + '/nuscenes_map_infos_val_newsplit.pkl'

img_norm_cfg = dict(mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_size = (480, 800)
cat2id = dict(ped_crossing=0, divider=1, boundary=2)
num_class = 3
roi_size = (60, 30)
bev_h = 50
bev_w = 100
pc_range = [-30.0, -15.0, -3, 30.0, 15.0, 5]
canvas_size = (200, 100)
bev_embed_dims = 256
embed_dims = 512
num_points = 20
meta = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False,
            use_external=False, output_format='vector')

model = dict(
    type='ReSMap',
    roi_size=roi_size,
    bev_h=bev_h,
    bev_w=bev_w,
    history_steps=4,
    test_time_history_steps=20,
    mem_select_dist_ranges=[1, 5, 10, 15],
    backbone_cfg=dict(
        type='BEVFormerBackbone',
        roi_size=roi_size,
        bev_h=bev_h,
        bev_w=bev_w,
        use_grid_mask=True,
        history_steps=4,
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d'),
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)),
        img_neck=dict(
            type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=bev_embed_dims,
            start_level=0,
            add_extra_convs=True,
            num_outs=3,
            norm_cfg=dict(type='BN2d'),
            relu_before_extra_convs=True),
        transformer=dict(
            type='PerceptionTransformer',
            embed_dims=bev_embed_dims,
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=2,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    attn_cfgs=[
                        dict(type='TemporalSelfAttention', embed_dims=bev_embed_dims, num_levels=1),
                        dict(
                            type='SpatialCrossAttention',
                            deformable_attention=dict(
                                type='MSDeformableAttention3D',
                                embed_dims=bev_embed_dims,
                                num_points=8,
                                num_levels=3),
                            embed_dims=bev_embed_dims)
                    ],
                    feedforward_channels=bev_embed_dims * 2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=bev_embed_dims // 2,
            row_num_embed=bev_h,
            col_num_embed=bev_w)),
    seg_cfg=dict(
        type='MapSegHead',
        num_classes=num_class,
        in_channels=bev_embed_dims,
        embed_dims=bev_embed_dims,
        bev_size=(bev_w, bev_h),
        canvas_size=canvas_size,
        loss_seg=dict(type='MaskFocalLoss', use_sigmoid=True, loss_weight=10.0),
        loss_dice=dict(type='MaskDiceLoss', loss_weight=1.0)),
    sat_encoder_cfg=dict(
        type='SatelliteEncoder',
        backbone_cfg=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            norm_cfg=dict(type='BN2d'),
            norm_eval=True,
            style='pytorch'),  # randomly initialized
        out_channels=bev_embed_dims,
        bev_size=(bev_h, bev_w)),
    # BEV masking of the fused-branch inputs (training only)
    use_modality_masking=True,
    mask_cam_zero_prob=0.1,
    mask_cam_noise_prob=0.1,
    mask_cam_block_prob=0.05,
    mask_sat_zero_prob=0.05,
    mask_noise_scale=1.0,
    # per-scene evidence drop in the OCMQ gate (training only)
    scene_drop_probs=(0.3, 0.3, 0.1),
    mm_head_cfg=dict(
        num_classes=num_class,
        in_channels=bev_embed_dims,
        embed_dims=embed_dims,
        num_points=num_points,
        num_queries=100,
        roi_size=roi_size,
        decoder_cfg=dict(num_layers=6, num_heads=8, ffn_dims=1024, dropout=0.1),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=5.0),
        loss_reg=dict(type='LinesL1Loss', loss_weight=50.0, beta=0.01),
        assigner=dict(
            type='HungarianLinesAssigner',
            cls_cost=dict(type='FocalLossCost', weight=5.0),
            reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01)),
        sync_cls_avg_factor=True,
        bg_cls_weight=0.0,
        different_heads=False))

collect_meta_keys = ('token', 'ego2img', 'sample_idx', 'ego2global_translation',
                     'ego2global_rotation', 'img_shape', 'scene_name')
train_pipeline = [
    dict(type='VectorizeMap', coords_dim=2, roi_size=roi_size, sample_num=num_points,
         normalize=True, permute=True),
    dict(type='RasterizeMap', roi_size=roi_size, coords_dim=2, canvas_size=canvas_size,
         thickness=3, semantic_mask=True),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='LoadAID4ADSatelliteImage', data_root=aid4ad_root, normalize=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages', size=img_size, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'vectors', 'semantic_mask', 'sat_img'],
         meta_keys=collect_meta_keys),
]
test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='LoadAID4ADSatelliteImage', data_root=aid4ad_root, normalize=True),
    dict(type='ResizeMultiViewImages', size=img_size, change_intrinsics=True),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    # camera failure at test time; set by tools/test.py --camera-drop
    dict(type='ApplySensorCorruption', enabled=False, camera_mask=0, camera_mode='zero'),
    dict(type='FormatBundleMap'),
    dict(type='Collect3D', keys=['img', 'sat_img'], meta_keys=collect_meta_keys),
]
# GT used by the evaluator
eval_config = dict(
    type='NuscDataset',
    data_root=data_root,
    ann_file=val_ann,
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(type='VectorizeMap', coords_dim=2, simplify=True, normalize=False, roi_size=roi_size),
        dict(type='RasterizeMap', roi_size=roi_size, coords_dim=2, canvas_size=canvas_size,
             thickness=3, semantic_mask=True),
        dict(type='FormatBundleMap'),
        dict(type='Collect3D', keys=['vectors', 'semantic_mask'], meta_keys=collect_meta_keys),
    ],
    interval=1)

eval_semantic = False
test_dataset = dict(
    type='NuscDataset',
    data_root=data_root,
    ann_file=val_ann,
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=test_pipeline,
    eval_config=eval_config,
    test_mode=True,
    seq_split_num=1,
    eval_semantic=eval_semantic)
data = dict(
    samples_per_gpu=3,  # total batch size 6
    workers_per_gpu=8,
    train=dict(
        type='NuscDataset',
        data_root=data_root,
        ann_file=train_ann,
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-2,
        multi_frame=5,
        sampling_span=10),
    val=test_dataset,
    test=test_dataset)

# 27,840 training samples / batch 6 = 4,640 iterations per epoch
iters_per_epoch = 4640
checkpoint_config = dict(interval=iters_per_epoch)
evaluation = dict(interval=iters_per_epoch)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
SyncBN = True
