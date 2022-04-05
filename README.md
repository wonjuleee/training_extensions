# OpenVINO™ Training Extensions
[![python](https://img.shields.io/badge/python-3.8%2B-green)]()
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)]()
[![mypy](https://img.shields.io/badge/%20type_checker-mypy-%231674b1?style=flat)]()
[![openvino](https://img.shields.io/badge/openvino-2021.4-purple)]()

OpenVINO™ Training Extensions provide a convenient environment to train
Deep Learning models and convert them using the [OpenVINO™
toolkit](https://software.intel.com/en-us/openvino-toolkit) for optimized
inference.

## Prerequisites
* Ubuntu 18.04 / 20.04
* Python 3.8+
* [CUDA Toolkit 11.1](https://developer.nvidia.com/cuda-11.1.1-download-archive) - for training on GPU

## Repository components
* [OTE SDK](ote_sdk)
* [OTE CLI](ote_cli)
* [OTE Algorithms](external)

## Quick start guide
In order to get started with OpenVINO™ Training Extensions click [here](QUICK_START_GUIDE.md).

## License
Deep Learning Deployment Toolkit is licensed under [Apache License Version 2.0](LICENSE).
By contributing to the project, you agree to the license and copyright terms therein
and release your contribution under these terms.

## Misc

Models that were previously developed can be found on the [misc](https://github.com/openvinotoolkit/training_extensions/tree/misc) branch.

## Contributing

Please read the [Contributing guide](CONTRIBUTING.md) before starting work on a pull request.

## Known limitations

Currently, training, exporting, evaluation scripts for TensorFlow\*-based models and the most of PyTorch\*-based models from the [misc](#misc) branch are exploratory and are not validated.

- The following repositories are used: [MMDeploy dev-v0.4.0](https://github.com/open-mmlab/mmdeploy/tree/dev-v0.4.0), [MMDetection 2.22.0](https://github.com/open-mmlab/mmdetection).
- The gen3_mobilenetV2_ATSS and gen3_resnet50_VFNet templates are supported.
- There may be problems with other templates and models (SSD, YOLOXS).
- NNCF is not supported.

The following tests fail:
- training_extensions/external/mmdetection/
    - tests/test_ote_api.py::API::test_nncf_optimize_progress_tracking
    - tests/test_ote_inference.py::TestInference::test_inference - TypeError: SingleStageDetector
- training_extensions/external/mmdetection/submodule/
    - ote/tests/test_models.py::PublicModelsTestCase::test_onnx_retinanet_effd0_bifpn_1x_coco
    - ote/tests/test_models.py::PublicModelsTestCase::test_onnx_ssd300_coco
    - ote/tests/test_models.py::PublicModelsTestCase::test_openvino_retinanet_effd0_bifpn_1x_coco
    - ote/tests/test_models.py::PublicModelsTestCase::test_openvino_ssd300_coco
    - ote/tests/test_models.py::PublicModelsTestCase::test_pytorch_retinanet_effd0_bifpn_1x_coco
    - ote/tests/test_models.py::PublicModelsTestCase::test_pytorch_ssd300_coco
- training_extensions/
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_train[Custom_Object_Detection_YOLOX]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_train[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_export[Custom_Object_Detection_YOLOX]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_export[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_eval[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_eval_openvino[Custom_Object_Detection_YOLOX]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_demo[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_deploy_openvino[Custom_Object_Detection_YOLOX]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_deploy_openvino[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_deploy_openvino[Custom_Object_Detection_Gen3_ATSS]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_hpo[Custom_Object_Detection_YOLOX]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_hpo[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_ote_hpo[Custom_Object_Detection_Gen3_ATSS]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_nncf_optimize[Custom_Object_Detection_YOLOX]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_nncf_optimize[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_nncf_export[Custom_Object_Detection_YOLOX]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_nncf_export[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_nncf_eval[Custom_Object_Detection_YOLOX]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_nncf_eval[Custom_Object_Detection_Gen3_SSD]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_detection.py::TestToolsDetection::test_notebook
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_train[Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_train[Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_export[Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_export[Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_eval[Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_eval[Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_demo[Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_demo[Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_deploy_openvino[Custom_Counting_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_instance_segmentation.py::TestToolsInstanceSegmentation::test_ote_deploy_openvino[Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_train[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_train[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_export[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_export[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_eval[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_eval[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_demo[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_demo[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_deploy_openvino[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_EfficientNetB2B]
    - tests/ote_cli/external/mmdetection/test_ote_cli_tools_rotated_detection.py::TestToolsRotatedDetection::test_ote_deploy_openvino[Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50]


---
\* Other names and brands may be claimed as the property of others.
