_base_ = ['../../../../../../MMDeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py']

onnx_config = dict(
    input_names=['image'],
    output_names=['boxes', 'labels'],
    dynamic_axes=dict({
        'image': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'boxes': {
            0: 'batch',
            1: 'num_dets',
        },
        'labels': {
            0: 'batch',
            1: 'num_dets',
        },
    }, _delete_=True),
    strip_doc_string=False,
)
