dataset_type = 'CocoDataset'
img_size = (1024, 1024)
img_norm_cfg = dict(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255], std=[0.229 * 255, 0.224 * 255, 0.225 * 255], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.5),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


# Bacteria = "Enterococci"

# root = f'/home/ikrylov/datasets/sc_counting/Vitens-{Bacteria}-coco/'

# root = '/media/cluster_fs/datasets/sc_counting/wgisd/'
# classes = (Bacteria.lower(), )
# classes = ("CDY", "CFR", "CSV", "SVB", "SYH")
# #classes = ("green", "blue", "red")

classes = ${CLASSES}
train_ann_file = ${TRAIN_ANN_FILE}
train_data_root = ${TRAIN_DATA_ROOT}
val_ann_file = ${VAL_ANN_FILE}
val_data_root = ${VAL_DATA_ROOT}
test_ann_file = ${TEST_ANN_FILE}
test_data_root = ${TEST_DATA_ROOT}
work_dir= ${WORK_DIR}
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        adaptive_repeat_times=True,
        dataset=dict(
            type=dataset_type,
            classes=classes,
            ann_file=train_ann_file,
            img_prefix=train_data_root,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=val_ann_file,
        img_prefix=val_data_root,
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=test_ann_file,
        img_prefix=test_data_root,
        test_mode=True,
        pipeline=test_pipeline))

width_mult = 1.0
model = dict(
    type='DensityEstimator',
    backbone=dict(
        type='resnet50',
        out_indices=(1, 2, 3, 4),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True,
    ),
    neck=None,
    bbox_head=dict(
        type='DensityHead',
        num_classes=len(classes),
        in_channels=[256, 512, 1024, 2048],
        feature_channels=32,
        losses=[
            # dict(type='CountingLoss'),
            dict(type='GSMLoss', h=img_size[1], w=img_size[0]),
            # dict(type='OTLoss', c_size=img_size[0], stride=8),
            # dict(type='TVLoss', h=img_size[1], w=img_size[0], stride=8),
        ],
        )
    )

cudnn_benchmark = True
evaluation = dict(interval=1, metric='MAE', save_best='MAE', rule='less')
optimizer = dict(
    type='SGD',
    lr=0.1,
    momentum=0.9,
    weight_decay=0.00001)
optimizer_config = dict()
lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='MAE',
    rule='less',
    patience=5,
    iteration_patience=500,
    interval=1,
    # policy='CosineAnnealing',
    min_lr=0.0000000000000001,
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=1.0 / 10)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
runner = dict(type='EpochRunnerWithCancel', max_epochs=300)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from=None
resume_from = None
workflow = [('train', 1)]

custom_hooks = [
    dict(type='EarlyStoppingHook', patience=8, iteration_patience=1000, metric='MAE', interval=1, priority=75, rule='less')
]