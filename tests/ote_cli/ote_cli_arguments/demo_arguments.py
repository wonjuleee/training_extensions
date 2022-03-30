"""
Model inference demonstration tool arguments
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


class DemoArguments:
    def __init__(self):

        # required common args
        self.input = None
        self.load_weights = None

        # not required common args
        self.fit_to_size = None
        self.fit_to_size_height = None
        self.fit_to_size_width = None
        self.loop = None
        self.delay = None
        self.display_perf = None

        # mmdetection template params
        self.params = None
        self.learning_parameters_batch_size = None
        self.learning_parameters_learning_rate = None
        self.learning_parameters_learning_rate_warmup_iters = None
        self.learning_parameters_num_iters = None
        self.postprocessing_confidence_threshold = None
        self.postprocessing_result_based_confidence_threshold = None
        self.nncf_optimization_enable_quantization = None
        self.nncf_optimization_enable_pruning = None
        self.nncf_optimization_maximal_accuracy_degradation = None

    def output(self):
        ret = []
        # required common args
        if self.input:
            ret.append('--input')
            ret.append(self.input)
        if self.load_weights:
            ret.append('--load-weights')
            ret.append(self.load_weights)

        # not required common args
        if self.fit_to_size:
            ret.append('--fit-to-size')
            ret.append(self.fit_to_size_height)
            ret.append(self.fit_to_size_width)
        if self.loop:
            ret.append('--loop')
            ret.append(self.loop)
        if self.delay:
            ret.append('--delay')
            ret.append(self.delay)
        if self.display_perf:
            ret.append('--display-perf')
            ret.append(self.display_perf)
        if self.params:
            ret.append('params')

        # mmdetection template params
        if self.postprocessing_confidence_threshold:
            ret.append('--postprocessing.confidence_threshold')
            ret.append(self.learning_parameters_batch_size)
        if self.postprocessing_result_based_confidence_threshold:
            ret.append('--postprocessing.result_based_confidence_threshold')
            ret.append(self.learning_parameters_learning_rate)
        return ret


class DemoArgumentsForge:
    def __init__(self, args=None):
        if args is None:
            self.args = DemoArguments()
        else:
            self.args = args

    @property
    def add_required_args(self):
        return RequiredArguments(self.args)

    @property
    def add_not_required_args(self):
        return NotRequiredArguments(self.args)

    @property
    def mmdetection_args(self):
        return MmdetectionTemplateArguments(self.args)

    def create(self):
        return self.args


class RequiredArguments(DemoArgumentsForge):
    def __init__(self, args):
        super().__init__(args)

    def input(self, value):
        self.args.input = value
        return self

    def load_weights(self, value):
        self.args.train_data_roots = value
        return self


class NotRequiredArguments(DemoArgumentsForge):
    def __init__(self, args):
        super().__init__(args)

    def fit_to_size(self, height, width):
        self.args.fit_to_size_height = height
        self.args.fit_to_size_width = width
        return self

    def delay(self, value):
        self.args.delay = value
        return self

    def display_perf(self, value):
        self.args.delay = value
        return self


class MmdetectionTemplateArguments(DemoArgumentsForge):
    def __init__(self, args):
        super().__init__(args)

    @property
    def params(self):
        self.args.params = True
        return self

    def postprocessing_confidence_threshold(self, value):
        self.args.postprocessing_confidence_threshold = value
        return self

    def postprocessing_result_based_confidence_threshold(self, value):
        self.args.postprocessing_result_based_confidence_threshold = value
        return self
