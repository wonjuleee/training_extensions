_base_ = ['../../../../../MMDeploy/configs/mmdet/_base_/base_openvino_dynamic-800x1344.py']

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 300, 300]))])

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
)
