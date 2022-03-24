_base_ = ['../../../../mmdeploy/submodule/configs/mmdet/instance-seg/instance-seg_openvino_dynamic-800x1344.py']

onnx_config = dict(
    input_names=['image'],
    output_names=['boxes', 'labels', 'masks'],
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
        'masks': {
            0: 'batch',
            1: 'num_dets',
            2: 'height',
            3: 'width'
        },
    }, _delete_=True),
    strip_doc_string=False,
)
