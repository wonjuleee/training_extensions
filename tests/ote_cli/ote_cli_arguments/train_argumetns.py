"""
Model training and optimization tools arguments
"""
# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
from copy import deepcopy
from typing import Dict, List
from collections import OrderedDict


class TrainOptimizeArguments:
    def __init__(self, ote_dir):
        self.ote_dir = ote_dir
        self.default_train_args_paths = {
            '--train-ann-file': 'data/airport/annotation_example_train.json',
            '--train-data-roots': 'data/airport/train',
            '--val-ann-file': 'data/airport/annotation_example_train.json',
            '--val-data-roots': 'data/airport/train',
            '--test-ann-files': 'data/airport/annotation_example_train.json',
            '--test-data-roots': 'data/airport/train',
        }
        self.parameters = {
            "train_ann_files": ["--train-ann-files", self.fabricate_path("--train-ann-file")],
            "train_data_roots": ["--train-data-roots", self.fabricate_path("--train-data-roots")],
            "val_ann_files": ["--val-ann-file", self.fabricate_path("--val-ann-file")],
            "val_data_roots": ["--val-data-roots", self.fabricate_path("--val-data-roots")],
            "save_model_to": ["--save-model-to", None],
            "enable_hpo": ["--enable-hpo", None],
            "params": ["params", None],
            "save_performance": ["--save-performance", None],
            "learning_parameters_batch_size": ["--learning_parameters.batch_size", None],
            "learning_parameters_learning_rate": ["--learning_parameters.learning_rate", None],
            "learning_parameters_learning_rate_warmup_iters": ["--learning_parameters.learning_rate_warmup_iters", None],
            "learning_parameters_num_iters": ["--learning_parameters.num_iters", None],
            "postprocessing_confidence_threshold": ["--postprocessing.confidence_threshold", None],
            "postprocessing_result_based_confidence_threshold": ["--postprocessing.result_based_confidence_threshold", None],
            "nncf_optimization_enable_quantization": ["--nncf_optimization.enable_quantization", None],
            "nncf_optimization_enable_pruning": ["--nncf_optimization.enable_pruning", None],
            "nncf_optimization_maximal_accuracy_degradation": ["--nncf_optimization.maximal_accuracy_degradation", None],
        }

        self.used_params = {} # .set(key)

        """# required common arguments
        self.train_ann_files = None
        self.train_data_roots = None
        self.val_ann_files = None
        self.val_data_roots = None
        self.save_model_to = None

        # not required train common arguments
        self.enable_hpo = None
        self.hpo_time_ratio = None
        self.params = None

        # not required optimization common arguments
        self.save_performance = None

        # not required common argument
        self.params = None

        # mmdetection template parameters
        self.learning_parameters_batch_size = None
        self.learning_parameters_learning_rate = None
        self.learning_parameters_learning_rate_warmup_iters = None
        self.learning_parameters_num_iters = None
        self.postprocessing_confidence_threshold = None
        self.postprocessing_result_based_confidence_threshold = None
        self.nncf_optimization_enable_quantization = None
        self.nncf_optimization_enable_pruning = None
        self.nncf_optimization_maximal_accuracy_degradation = None"""

    def fabricate_path(self, arg):
        return f'{os.path.join(self.ote_dir, self.default_train_args_paths[arg])}'


class CommandLineGenerator:
    def __init__(self, parameters: Dict[str,List[str]]):
        assert isinstance(parameters, OrderedDict)
        self.parameters = parameters
        self.used = dict()

    def command_line(self):
        res = []
        for k, v in self.parameters.items():
            if k not in self.used:
                continue
            if callable(v):
                cur_params = v(self.used(k))
            else:
                cur_params = deepcopy(v)

            res.expand(cur_params)
        return res

    def set(self, name, value=True):
        if name not in self.parameters:
            raise RuntimeError("") # TODO
        self.used[name] = value
        return self


class RequiredArguments:
    def __init__(self, ote_dir):
        self.ote_dir = ote_dir
        self.comand_line_generator = CommandLineGenerator(OrderedDict([
            ("--train-ann-files", ['--train-ann-file', self.fabricate_path('--train-ann-file')]),

        ]))

    def set(self, name):
        self.comand_line_generator.set(name)

    def command_line(self):
        return self.comand_line_generator.command_line()


"""class ArgumentsForge:
    def __init__(self, ote_dir, train_args=None):
        self.ote_dir = ote_dir
        self.required_arguments = RequiredArguments(self.ote_dir, self.train_args)
        if train_args is None:
            self.train_args = TrainOptimizeArguments(ote_dir)
        else:
            self.train_args = train_args

    @property
    def add_required_args(self):
        return self.required_arguments  # RequiredArguments(self.ote_dir, self.train_args)

    @property
    def opt_train_args(self):
        return OptTrainArguments(self.ote_dir, self.train_args)

    @property
    def opt_optimize_args(self):
        return OptTrainArguments(self.ote_dir, self.train_args)

    @property
    def mmdetection_args(self):
        return MmdetectionTemplateArguments(self.ote_dir, self.train_args)

    def create(self):
        return self.train_args


class RequiredArguments(ArgumentsForge):
    def __init__(self, ote_dir, train_args):
        super().__init__(ote_dir, train_args)

    @property
    def train_ann_files(self):
        self.train_args.train_ann_files = True
        return self

    @property
    def train_data_roots(self):
        self.train_args.train_data_roots = True
        return self

    @property
    def val_ann_files(self):
        self.train_args.val_ann_files = True
        return self

    @property
    def val_data_roots(self):
        self.train_args.val_data_roots = True
        return self

    def save_model_to(self, value):
        self.train_args.save_model_to = value
        return self


class OptTrainArguments(ArgumentsForge):
    def __init__(self, ote_dir, train_args):
        super().__init__(ote_dir, train_args)

    @property
    def enable_hpo(self):
        self.train_args.enable_hpo = True
        return self

    def hpo_time_ratio(self, value):
        self.train_args.hpo_time_ratio = value
        return self


class OptOptimizeArguments(ArgumentsForge):
    def __init__(self, ote_dir, train_args):
        super().__init__(ote_dir, train_args)

    def save_performance(self, value):
        self.train_args.hpo_time_ratio = value
        return self


class MmdetectionTemplateArguments(ArgumentsForge):
    def __init__(self, ote_dir, train_args):
        super().__init__(ote_dir, train_args)

    @property
    def params(self):
        self.train_args.params = True
        return self

    def learning_parameters_batch_size(self, value):
        self.train_args.learning_parameters_batch_size = value
        return self

    def learning_parameters_learning_rate(self, value):
        self.train_args.learning_parameters_batch_size = value
        return self

    def learning_parameters_learning_rate_warmup_iters(self, value):
        self.train_args.learning_parameters_learning_rate_warmup_iters = value
        return self

    def learning_parameters_num_iters(self, value):
        self.train_args.learning_parameters_num_iters = value
        return self

    def postprocessing_confidence_threshold(self, value):
        self.train_args.postprocessing_confidence_threshold = value
        return self

    def postprocessing_result_based_confidence_threshold(self, value):
        self.train_args.postprocessing_result_based_confidence_threshold = value
        return self

    def nncf_optimization_enable_quantization(self, value):
        self.train_args.learning_parameters_batch_size = value
        return self

    def nncf_optimization_enable_pruning(self, value):
        self.train_args.learning_parameters_batch_size = value
        return self

    def nncf_optimization_maximal_accuracy_degradation(self, value):
        self.train_args.nncf_optimization_maximal_accuracy_degradation = value
        return self"""