_base_ = ['../../../../../../MMDeploy/configs/mmdet/_base_/base_openvino_dynamic-800x1344.py']

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

'''
backend_config = dict(
    mo_args=dict({
        '--mean_values': [0, 0, 0],
        '--scale_values': [255, 255, 255],
        '--data_type': 'FP32',
    }),
    mo_flags=['--disable_fusing'],
)
'''
