"""This file defines the ModelConfiguration, ModelEntity and Model classes"""

# INTEL CONFIDENTIAL
#
# Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were provided to
# you ("License"). Unless the License provides otherwise, you may not use, modify, copy,
# publish, distribute, disclose or transmit this software or the related documents
# without Intel's prior written permission.
#
# This software and the related documents are provided as is,
# with no express or implied warranties, other than those that are expressly stated
# in the License.
import datetime
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from ote_sdk.configuration import ConfigurableParameters
from ote_sdk.entities.id import ID
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.label_schema import LabelSchemaEntity
from ote_sdk.entities.metrics import NullPerformance, Performance
from ote_sdk.entities.model_template import TargetDevice
from ote_sdk.entities.url import URL
from ote_sdk.usecases.adapters.model_adapter import IDataSource, ModelAdapter
from ote_sdk.utils.time_utils import now

if TYPE_CHECKING:
    # pylint: disable=ungrouped-imports
    from ote_sdk.entities.datasets import DatasetEntity


class ModelPrecision(IntEnum):
    """Represents the ModelPrecision of a Model"""

    INT4 = auto()
    INT8 = auto()
    FP16 = auto()
    FP32 = auto()


class ModelConfiguration:
    """
    This class represents the task configuration which was used to generate a specific model.

    Those are the parameters that a task may need in order to use the model.

    :param configurable_parameters: Task configurable parameters used to generate the model
    :param label_schema: Label schema inside the project used to generate the model
    """

    configurable_parameters: ConfigurableParameters
    labels: List[LabelEntity]
    label_schema: LabelSchemaEntity

    def __init__(
        self,
        configurable_parameters: ConfigurableParameters,
        label_schema: LabelSchemaEntity,
    ):
        self.configurable_parameters = configurable_parameters
        self.label_schema = label_schema


class ModelStatus(IntEnum):
    """Indicates the status of the last training result"""

    NOT_READY = auto()  # Model is not ready to be trained
    TRAINED_NO_STATS = auto()  # Model is trained but not evaluated yet
    SUCCESS = auto()  # Model trained successfully and improved
    FAILED = (
        auto()
    )  # Model failed during training, e.g. an error occurred or user cancelled training
    NOT_IMPROVED = auto()  # Model trained successfully but didn't improve


class ModelFormat(IntEnum):
    """Indicate the format of the model"""

    OPENVINO = auto()
    BASE_FRAMEWORK = auto()
    ONNX = auto()


class ModelOptimizationType(IntEnum):
    """Represents optimization type that is used to optimize the model"""

    NONE = auto()
    MO = auto()
    NNCF = auto()
    POT = auto()


class OptimizationMethod(IntEnum):
    """Represents optimization method that is used to optimize the model"""

    FILTER_PRUNING = auto()
    QUANTIZATION = auto()


# pylint: disable=too-many-instance-attributes, too-many-public-methods
class ModelEntity:
    """Represents the Entity of a Model"""

    # TODO: add tags and allow filtering on those in modelrepo
    # pylint: disable=too-many-arguments,too-many-locals; Requires refactor
    def __init__(
        self,
        train_dataset: "DatasetEntity",
        configuration: ModelConfiguration,
        *,
        creation_date: Optional[datetime.datetime] = None,
        performance: Optional[Performance] = None,
        previous_trained_revision: Optional["ModelEntity"] = None,
        previous_revision: Optional["ModelEntity"] = None,
        version: int = 1,
        tags: Optional[List[str]] = None,
        model_status: ModelStatus = ModelStatus.SUCCESS,
        model_format: ModelFormat = ModelFormat.OPENVINO,
        training_duration: float = 0.0,
        model_adapters: Optional[Dict[str, ModelAdapter]] = None,
        exportable_code: Optional[bytes] = None,
        precision: Optional[List[ModelPrecision]] = None,
        latency: int = 0,
        fps_throughput: int = 0,
        target_device: TargetDevice = TargetDevice.CPU,
        target_device_type: Optional[str] = None,
        optimization_type: ModelOptimizationType = ModelOptimizationType.NONE,
        optimization_methods: List[OptimizationMethod] = None,
        optimization_objectives: Dict[str, str] = None,
        performance_improvement: Dict[str, float] = None,
        model_size_reduction: float = 0.0,
        _id: Optional[ID] = None,
    ):
        _id = ID() if _id is None else _id
        performance = NullPerformance() if performance is None else performance
        creation_date = now() if creation_date is None else creation_date

        optimization_methods = (
            [] if optimization_methods is None else optimization_methods
        )
        optimization_objectives = (
            {} if optimization_objectives is None else optimization_objectives
        )
        performance_improvement = (
            {} if performance_improvement is None else performance_improvement
        )

        tags = [] if tags is None else tags
        precision = [ModelPrecision.FP32] if precision is None else precision

        if model_adapters is None:
            model_adapters = {}

        self.__id = _id
        self.__creation_date = creation_date
        self.__train_dataset = train_dataset
        self.__previous_trained_revision = previous_trained_revision
        self.__previous_revision = previous_revision
        self.__version = version
        self.__tags = tags
        self.__model_status = model_status
        self.__model_format = model_format
        self.__performance = performance
        self.__training_duration = training_duration
        self.__configuration = configuration
        self.__model_adapters = model_adapters
        self.__exportable_code = exportable_code
        self.model_adapters_to_delete: List[ModelAdapter] = []
        self.__precision = precision
        self.__latency = latency
        self.__fps_throughput = fps_throughput
        self.__target_device = target_device
        self.__target_device_type = target_device_type
        self.__optimization_type = optimization_type
        self.__optimization_methods = optimization_methods
        self.__optimization_objectives = optimization_objectives
        self.__performance_improvement = performance_improvement
        self.__model_size_reduction = model_size_reduction

    @property
    def id(self) -> ID:
        """Gets or sets the id of a Model"""
        return self.__id

    @id.setter
    def id(self, value: ID):
        self.__id = value

    @property
    def configuration(self) -> ModelConfiguration:
        """Gets or sets the configuration of the Model"""
        return self.__configuration

    @configuration.setter
    def configuration(self, value: ModelConfiguration):
        self.__configuration = value

    @property
    def creation_date(self) -> datetime.datetime:
        """Gets or sets the creation_date of the Model"""
        return self.__creation_date

    @creation_date.setter
    def creation_date(self, value: datetime.datetime):
        self.__creation_date = value

    @property
    def train_dataset(self) -> "DatasetEntity":
        """Gets or sets the current Training Dataset"""
        return self.__train_dataset

    @train_dataset.setter
    def train_dataset(self, value: "DatasetEntity"):
        self.__train_dataset = value

    @property
    def previous_trained_revision(self) -> Union[None, "ModelEntity"]:
        """
        Gets or sets the previous model
        Returns None if no previous_trained_revision has been created
        """
        return self.__previous_trained_revision

    @previous_trained_revision.setter
    def previous_trained_revision(self, value: "ModelEntity"):
        self.__previous_trained_revision = value

    @property
    def previous_revision(self) -> Union[None, "ModelEntity"]:
        """
        Gets or sets the previous model
        """
        return self.__previous_revision

    @previous_revision.setter
    def previous_revision(self, value: "ModelEntity"):
        self.__previous_revision = value

    @property
    def version(self) -> int:
        """Gets or sets the version"""
        return self.__version

    @version.setter
    def version(self, value: int):
        self.__version = value

    @property
    def tags(self) -> List[str]:
        """Gets or sets the tags of the Model"""
        return self.__tags

    @tags.setter
    def tags(self, value: List[str]):
        self.__tags = value

    @property
    def model_status(self) -> ModelStatus:
        """Shows the status of the latest training"""
        return self.__model_status

    @model_status.setter
    def model_status(self, value: ModelStatus):
        self.__model_status = value

    @property
    def model_format(self) -> ModelFormat:
        """Gets the model format"""
        return self.__model_format

    @model_format.setter
    def model_format(self, value: ModelFormat):
        self.__model_format = value

    @property
    def performance(self) -> Performance:
        """Gets or sets the current Performance of the Model"""
        return self.__performance

    @performance.setter
    def performance(self, value: Performance):
        self.__performance = value

    @property
    def training_duration(self) -> float:
        """Gets or sets the current training duration"""
        return self.__training_duration

    @training_duration.setter
    def training_duration(self, value: float):
        self.__training_duration = value

    @property
    def precision(self) -> List[ModelPrecision]:
        """
        Get or set the precision for the model.
        This has effect on accuracy, latency and throughput of the model.
        """
        return self.__precision

    @precision.setter
    def precision(self, value: List[ModelPrecision]):
        self.__precision = value

    @property
    def latency(self) -> int:
        """
        Get or set the latency of the model.
        Unit is milliseconds (ms)
        """
        return self.__latency

    @latency.setter
    def latency(self, value: int):
        self.__latency = value

    @property
    def fps_throughput(self) -> int:
        """
        Get or set the throughput of the model
        Unit is frames per second (fps)
        """
        return self.__fps_throughput

    @fps_throughput.setter
    def fps_throughput(self, value: int):
        self.__fps_throughput = value

    @property
    def target_device(self) -> TargetDevice:
        """
        Get or set the device on which the model will be deployed
        """
        return self.__target_device

    @target_device.setter
    def target_device(self, value: TargetDevice):
        self.__target_device = value

    @property
    def target_device_type(self) -> Optional[str]:
        """
        Get or set the type of the target device used by the model
        """
        return self.__target_device_type

    @target_device_type.setter
    def target_device_type(self, value: str):
        self.__target_device_type = value

    @property
    def optimization_methods(self) -> Optional[List[OptimizationMethod]]:
        """
        Get or set the optimization methods used on the model
        """
        return self.__optimization_methods

    @optimization_methods.setter
    def optimization_methods(self, value: List[OptimizationMethod]):
        self.__optimization_methods = value

    @property
    def optimization_type(self) -> ModelOptimizationType:
        """
        Get or set the optimization type used for the model
        """
        return self.__optimization_type

    @optimization_type.setter
    def optimization_type(self, value: ModelOptimizationType):
        self.__optimization_type = value

    @property
    def optimization_objectives(self) -> Optional[Dict[str, str]]:
        """
        Get or set the optimization level of the model
        """
        return self.__optimization_objectives

    @optimization_objectives.setter
    def optimization_objectives(self, value: Dict[str, str]):
        self.__optimization_objectives = value

    @property
    def performance_improvement(self) -> Optional[Dict[str, float]]:
        """
        Get or set the performance improvement of the model
        """
        return self.__performance_improvement

    @performance_improvement.setter
    def performance_improvement(self, value: Dict[str, float]):
        self.__performance_improvement = value

    @property
    def model_size_reduction(self) -> float:
        """
        Get or set the reduction in model size by optimizing
        """
        return self.__model_size_reduction

    @model_size_reduction.setter
    def model_size_reduction(self, value: float):
        self.__model_size_reduction = value

    @property
    def exportable_code(self) -> Optional[bytes]:
        """
        Get the exportable_code
        """
        return self.__exportable_code

    @exportable_code.setter
    def exportable_code(self, data: bytes):
        self.__exportable_code = data

    def get_data(self, key: str) -> bytes:
        """
        Fetches byte data for a certain model.
        :param key: key to fetch data for
        :return:
        """
        return self.__model_adapters[key].data

    def set_data(self, key: str, data: Union[bytes, IDataSource], skip_deletion=False):
        """
        Sets the data for a specified key, either from a binary blob or from a data source. If the key already exists
        it appends existing data url to a list of urls that will be removed upon saving the model. Skip deletion
        parameter should only be true if replacing bytes data with a file.
        """
        if not skip_deletion:
            self.delete_data(key)
        self.__model_adapters[key] = ModelAdapter(data)

    def delete_data(self, key: str):
        """
        This function is used to delete data sources that are on the filesystem. If the key exists the model adapter
        will be appended to a list of model adapter that will be removed once the model is saved by the repo. Note that
        an optimized model must contain at least 1 DataSource otherwise you are left with an invalid optimized model.
        """
        if key in self.__model_adapters:
            self.model_adapters_to_delete.append(self.__model_adapters[key])
            del self.__model_adapters[key]

    @property
    def model_adapters(self) -> Dict[str, ModelAdapter]:
        """
        Returns the dictionary of model adapters for each data key.
        """
        return self.__model_adapters

    @property
    def weight_paths(self) -> Dict[str, URL]:
        """
        Returns the the path to URLs for each data key. Note that this function will raise an error if
        the model was not saved to a database.
        """
        return {
            key: model_adapter.data_source.binary_url
            for key, model_adapter in self.model_adapters.items()
            if model_adapter.from_file_storage
        }

    def is_optimized(self) -> bool:
        """
        Returns a boolean indicating if the model has been optimized or not
        """
        if self.optimization_type == ModelOptimizationType.NONE:
            return False
        return True

    def __eq__(self, other):
        if isinstance(other, ModelEntity):
            return (
                self.id == other.id
                and self.train_dataset == other.train_dataset
                and self.performance == other.performance
            )
        return False
