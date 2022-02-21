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

from copy import deepcopy
import os
from subprocess import run

import pytest

from ote_cli.registry import Registry
from common import wrong_paths

ote_dir = os.getcwd()


@pytest.fixture()
def templates(algo_be):
    return Registry('external').filter(task_type=algo_be).templates


def test_ote_export_no_weights(templates):
    error_string = "ote export: error: the following arguments are required: --load-weights"
    for template in templates:
        command_line = ['ote',
                        'export',
                        template.model_template_id,
                        f'--save-model-to',
                        f'./exported_{template.model_template_id}']
        assert error_string in str(run(command_line, capture_output=True).stderr)


def test_ote_export_no_save_to(templates):
    error_string = "ote export: error: the following arguments are required: --save-model-to"
    for template in templates:
        command_line = ['ote',
                        'export',
                        template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth']
        assert error_string in str(run(command_line, capture_output=True).stderr)


def test_ote_export_wrong_paths(templates):
    for template in templates:
        command_line = ['ote',
                        'export',
                        template.model_template_id,
                        '--load-weights',
                        './trained_default_template/weights.pth',
                        f'--save-model-to',
                        f'./exported_{template.model_template_id}']
        for i in [4, 6]:
            for case in wrong_paths.values():
                temp = deepcopy(command_line)
                temp[i] = case
                assert "Path is not valid" in str(run(temp, capture_output=True).stderr)


def test_ote_export_no_template():
    error_string = "ote export: error: the following arguments are required: template, --load-weights, --save-model-to"
    command_line = ['ote',
                    'export']
    assert error_string in str(run(command_line, capture_output=True).stderr)