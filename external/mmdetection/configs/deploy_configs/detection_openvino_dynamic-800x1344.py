_base_ = ['./base/detection_openvino.py']

backend_config = dict(
    model_inputs=[dict(opt_shapes=dict(input=[1, 3, 800, 1344]))])
