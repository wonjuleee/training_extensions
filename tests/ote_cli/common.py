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

import shutil
import json
import os
import logging
from subprocess import run  # nosec

from ote_sdk.usecases.exportable_code.utils import get_git_commit_hash

args_paths = {
    '--train-ann-file': 'data/airport/annotation_example_train.json',
    '--train-data-roots': 'data/airport/train',
    '--val-ann-file': 'data/airport/annotation_example_train.json',
    '--val-data-roots': 'data/airport/train',
    '--test-ann-files': 'data/airport/annotation_example_train.json',
    '--test-data-roots': 'data/airport/train',
}

wrong_paths = {
               'empty': '',
               'not_printable': '\x11',
               # 'null_symbol': '\x00' It is caught on subprocess level
               }

logger = logging.getLogger(__name__)


def get_template_rel_dir(template):
    return os.path.dirname(os.path.relpath(template.model_template_path))


def get_some_vars(template, root):
    template_dir = get_template_rel_dir(template)
    task_type = template.task_type
    work_dir = os.path.join(root, str(task_type))
    template_work_dir = os.path.join(work_dir, template_dir)
    os.makedirs(template_work_dir, exist_ok=True)
    algo_backend_dir = '/'.join(template_dir.split('/')[:2])

    return work_dir, template_work_dir, algo_backend_dir


def create_venv(algo_backend_dir, work_dir, template_work_dir):
    venv_dir = f'{work_dir}/venv'
    if not os.path.exists(venv_dir):
        assert run([f'./{algo_backend_dir}/init_venv.sh', venv_dir]).returncode == 0, "Exit code must be 0"
        install_ote_cli_cmd = [f'{work_dir}/venv/bin/python', '-m', 'pip', 'install', '-e', 'ote_cli']
        assert run(install_ote_cli_cmd).returncode == 0, "Exit code must be 0"


def extract_export_vars(path):
    vars_ = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('export ') and '=' in line:
                line = line.replace('export ', '').split('=')
                assert len(line) == 2
                vars_[line[0].strip()] = line[1].strip()
    return vars_


def collect_env_vars(work_dir):
    vars_ = extract_export_vars(f'{work_dir}/venv/bin/activate')
    vars_.update({'PATH':f'{work_dir}/venv/bin/:' + os.environ['PATH']})
    if 'HTTP_PROXY' in os.environ:
        vars_.update({'HTTP_PROXY': os.environ['HTTP_PROXY']})
    if 'HTTPS_PROXY' in os.environ:
        vars_.update({'HTTPS_PROXY': os.environ['HTTPS_PROXY']})
    if 'NO_PROXY' in os.environ:
        vars_.update({'NO_PROXY': os.environ['NO_PROXY']})
    return vars_


def patch_demo_py(src_path, dst_path):
    with open(src_path) as read_file:
        content = [line for line in read_file]
        replaced = False
        for i, line in enumerate(content):
            if 'visualizer = Visualizer(media_type)' in line:
                content[i] = line.rstrip() + '; visualizer.show = show\n'
                replaced = True
        assert replaced
        content = ['def show(self):\n', '    pass\n\n'] + content
        with open(dst_path, 'w') as write_file:
            write_file.write(''.join(content))


def remove_ote_sdk_from_requirements(path):
    with open(path, encoding='UTF-8') as read_file:
        content = ''.join([line for line in read_file if 'ote_sdk' not in line])

    with open(path, 'w', encoding='UTF-8') as write_file:
        write_file.write(content)


def check_ote_sdk_commit_hash_in_requirements(path):
    with open(path, encoding='UTF-8') as read_file:
        content = [line for line in read_file if 'ote_sdk' in line]
    if len(content) != 1:
        raise RuntimeError(f"Invalid ote_sdk requirements (0 or more than 1 times mentioned): {path}")

    git_commit_hash = get_git_commit_hash()
    assert git_commit_hash in content[0], "OTE SDK commit hash must be in requirement"


def path_exist_assert(path: str) -> None:
    path_exist_assert(path), f"Path {path} must be exists after execution"


def ote_common(template, root, tool, cmd_args):
    work_dir, __, _ = get_some_vars(template, root)
    command_line = ['ote',
                    tool,
                    *cmd_args]
    ret = run(command_line, env=collect_env_vars(work_dir), capture_output=True)
    output = {'exit_code': int(ret.returncode), 'stdout': str(ret.stdout), 'stderr': str(ret.stderr)}
    return output


def ote_train_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--train-ann-file',
                    f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                    '--train-data-roots',
                    f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                    '--val-ann-file',
                    f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                    '--val-data-roots',
                    f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                    '--save-model-to',
                    f'{template_work_dir}/trained_{template.model_template_id}']
    command_line.extend(args['train_params'])
    ret = ote_common(template, root, 'train', command_line)

    logger.debug(f"Command arguments: ote train {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(f'{template_work_dir}/trained_{template.model_template_id}/weights.pth')
    path_exist_assert(f'{template_work_dir}/trained_{template.model_template_id}/label_schema.json')


def ote_hpo_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    if os.path.exists(f"{template_work_dir}/hpo"):
        shutil.rmtree(f"{template_work_dir}/hpo")
    command_line = [template.model_template_id,
                    '--train-ann-file',
                    f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                    '--train-data-roots',
                    f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                    '--val-ann-file',
                    f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                    '--val-data-roots',
                    f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                    '--save-model-to',
                    f'{template_work_dir}/hpo_trained_{template.model_template_id}',
                    '--enable-hpo',
                    '--hpo-time-ratio',
                    '1']
    command_line.extend(args['train_params'])
    ret = ote_common(template, root, 'train', command_line)

    logger.debug(f"Command arguments: ote train {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    hpopt_status_path = f'{template_work_dir}/hpo/hpopt_status.json'

    path_exist_assert(hpopt_status_path)
    with open(hpopt_status_path, "r") as f:
        assert json.load(f).get('best_config_id', None) is not None, \
            f"Json file must be available by path {hpopt_status_path}"
    path_exist_assert(f'{template_work_dir}/hpo_trained_{template.model_template_id}/weights.pth')
    path_exist_assert(f'{template_work_dir}/hpo_trained_{template.model_template_id}/label_schema.json')


def ote_export_testing(template, root):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    f'--save-model-to',
                    f'{template_work_dir}/exported_{template.model_template_id}']

    ret = ote_common(template, root, 'export', command_line)

    logger.debug(f"Command arguments: ote export {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"

    path_exist_assert(f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml')
    path_exist_assert(f'{template_work_dir}/exported_{template.model_template_id}/openvino.bin')
    path_exist_assert(f'{template_work_dir}/exported_{template.model_template_id}/label_schema.json')


def ote_eval_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    '--save-performance',
                    f'{template_work_dir}/trained_{template.model_template_id}/performance.json']
    ret = ote_common(template, root, 'eval', command_line)

    logger.debug(f"Command arguments: ote eval {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(f'{template_work_dir}/trained_{template.model_template_id}/performance.json')


def ote_eval_openvino_testing(template, root, ote_dir, args, threshold):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    exported_performance_path = f'{template_work_dir}/exported_{template.model_template_id}/performance.json'
    trained_performance_path = f'{template_work_dir}/trained_{template.model_template_id}/performance.json'
    path_exist_assert(trained_performance_path)
    command_line = [template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    '--save-performance',
                    exported_performance_path]
    ret = ote_common(template, root, 'eval', command_line)

    logger.debug(f"Command arguments: ote eval {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"

    path_exist_assert(exported_performance_path)
    with open(trained_performance_path) as read_file:
        trained_performance = json.load(read_file)
    with open(exported_performance_path) as read_file:
        exported_performance = json.load(read_file)

    for k in trained_performance.keys():
        assert trained_performance[k] != 0, f"Trained performance {trained_performance[k]} for {k} must not be 0"
        performance = abs(trained_performance[k] - exported_performance[k]) / trained_performance[k]
        assert_info = f"Performance must be <= {threshold} {trained_performance[k]=}, {exported_performance[k]=}"
        assert performance <= threshold, assert_info


def ote_demo_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    '--input',
                    os.path.join(ote_dir, args['--input']),
                    '--delay',
                    '1']
    ret = ote_common(template, root, 'demo', command_line)

    logger.debug(f"Command arguments: ote demo {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"


def ote_demo_openvino_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    '--input',
                    os.path.join(ote_dir, args['--input']),
                    '--delay',
                    '1']
    ret = ote_common(template, root, 'demo', command_line)

    logger.debug(f"Command arguments: ote demo {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"


def ote_deploy_openvino_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    deployment_dir = f'{template_work_dir}/deployed_{template.model_template_id}'
    command_line = [template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    f'--save-model-to',
                    deployment_dir]
    ret = ote_common(template, root, 'deploy', command_line)

    logger.debug(f"Command arguments: ote deploy {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"

    assert run(['unzip', 'openvino.zip'],
               cwd=deployment_dir).returncode == 0, "Exit code must be 0"
    assert run(['python3', '-m', 'venv', 'venv'],
               cwd=os.path.join(deployment_dir, 'python')).returncode == 0, "Exit code must be 0"
    assert run(['python3', '-m', 'pip', 'install', 'wheel'],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0, "Exit code must be 0"

    check_ote_sdk_commit_hash_in_requirements(os.path.join(deployment_dir, 'python', 'requirements.txt'))

    # Remove ote_sdk from requirements.txt, since merge commit
    # (that is created on CI) is not pushed to github and that's why cannot be cloned.
    # Install ote_sdk from local folder instead.
    # Install the demo_package with --no-deps since,
    # requirements.txt has been embedded to the demo_package during creation.
    remove_ote_sdk_from_requirements(os.path.join(deployment_dir, 'python', 'requirements.txt'))
    cmd_line = ['python3', '-m', 'pip', 'install', '-e', os.path.join(os.path.dirname(__file__), '..', '..', 'ote_sdk')]
    assert run(cmd_line,
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0, "Exit code must be 0"
    assert run(['python3', '-m', 'pip', 'install', '-r', os.path.join(deployment_dir, 'python', 'requirements.txt')],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0, "Exit code must be 0"
    assert run(['python3', '-m', 'pip', 'install', 'demo_package-0.0-py3-none-any.whl', '--no-deps'],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0, "Exit code must be 0"

    # Patch demo since we are not able to run cv2.imshow on CI.
    patch_demo_py(os.path.join(deployment_dir, 'python', 'demo.py'),
                  os.path.join(deployment_dir, 'python', 'demo_patched.py'))

    assert run(['python3', 'demo_patched.py', '-m', '../model/model.xml', '-i', os.path.join(ote_dir, args['--input'])],
               cwd=os.path.join(deployment_dir, 'python'),
               env=collect_env_vars(os.path.join(deployment_dir, 'python'))).returncode == 0, "Exit code must be 0"


def ote_eval_deployment_testing(template, root, ote_dir, args, threshold):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    deployed_performance_path = f'{template_work_dir}/deployed_{template.model_template_id}/performance.json'
    exported_performance_path = f'{template_work_dir}/exported_{template.model_template_id}/performance.json'
    path_exist_assert(exported_performance_path)
    command_line = [template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/deployed_{template.model_template_id}/openvino.zip',
                    '--save-performance',
                    deployed_performance_path]
    ret = ote_common(template, root, 'eval', command_line)

    logger.debug(f"Command arguments: ote eval {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(deployed_performance_path)
    with open(exported_performance_path) as read_file:
        exported_performance = json.load(read_file)
    with open(deployed_performance_path) as read_file:
        deployed_performance = json.load(read_file)

    for k in exported_performance.keys():
        assert exported_performance[k] != 0, f"Trained performance {exported_performance[k]} for {k} must not be 0"
        performance = abs(exported_performance[k] - deployed_performance[k]) / exported_performance[k]
        assert_info = f"Performance must be <= {threshold}: {deployed_performance[k]=}, {exported_performance[k]=}"
        assert performance <= threshold, assert_info


def ote_demo_deployment_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/deployed_{template.model_template_id}/openvino.zip',
                    '--input',
                    os.path.join(ote_dir, args['--input']),
                    '--delay',
                    '1']
    ret = ote_common(template, root, 'demo', command_line)

    logger.debug(f"Command arguments: ote demo {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"


def pot_optimize_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
    # TODO:fk create tests against parameters
    command_line = [template.model_template_id,
                    '--train-ann-file',
                    f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                    '--train-data-roots',
                    f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                    '--val-ann-file',
                    f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                    '--val-data-roots',
                    f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/exported_{template.model_template_id}/openvino.xml',
                    '--save-model-to',
                    f'{template_work_dir}/pot_{template.model_template_id}',
                    ]
    ret = ote_common(template, root, 'optimize', command_line)

    logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(f'{template_work_dir}/pot_{template.model_template_id}/openvino.xml')
    path_exist_assert(f'{template_work_dir}/pot_{template.model_template_id}/openvino.bin')
    path_exist_assert(f'{template_work_dir}/pot_{template.model_template_id}/label_schema.json')


def pot_eval_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/pot_{template.model_template_id}/openvino.xml',
                    '--save-performance',
                    f'{template_work_dir}/pot_{template.model_template_id}/performance.json']
    ret = ote_common(template, root, 'eval', command_line)

    logger.debug(f"Command arguments: ote eval {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(f'{template_work_dir}/pot_{template.model_template_id}/performance.json')


def nncf_optimize_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, algo_backend_dir = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--train-ann-file',
                    f'{os.path.join(ote_dir, args["--train-ann-file"])}',
                    '--train-data-roots',
                    f'{os.path.join(ote_dir, args["--train-data-roots"])}',
                    '--val-ann-file',
                    f'{os.path.join(ote_dir, args["--val-ann-file"])}',
                    '--val-data-roots',
                    f'{os.path.join(ote_dir, args["--val-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/trained_{template.model_template_id}/weights.pth',
                    '--save-model-to',
                    f'{template_work_dir}/nncf_{template.model_template_id}',
                    '--save-performance',
                    f'{template_work_dir}/nncf_{template.model_template_id}/train_performance.json',
                    ]
    command_line.extend(args['train_params'])
    ret = ote_common(template, root, 'optimize', command_line)

    logger.debug(f"Command arguments: ote optimize {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(f'{template_work_dir}/nncf_{template.model_template_id}/weights.pth')
    path_exist_assert(f'{template_work_dir}/nncf_{template.model_template_id}/label_schema.json')


def nncf_export_testing(template, root):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--load-weights',
                    f'{template_work_dir}/nncf_{template.model_template_id}/weights.pth',
                    f'--save-model-to',
                    f'{template_work_dir}/exported_nncf_{template.model_template_id}']
    ret = ote_common(template, root, 'export', command_line)

    logger.debug(f"Command arguments: ote export {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(f'{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml')
    path_exist_assert(f'{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.bin')
    path_exist_assert(f'{template_work_dir}/exported_nncf_{template.model_template_id}/label_schema.json')
    original_bin_path = f'{template_work_dir}/exported_{template.model_template_id}/openvino.bin'
    original_bin_size = os.path.getsize(original_bin_path)
    compressed_bin_path = f'{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.bin'
    compressed_bin_size = os.path.getsize(compressed_bin_path)
    assert_info = f"{compressed_bin_size=} must be < {original_bin_size=}"
    assert compressed_bin_size < original_bin_size, assert_info


def nncf_eval_testing(template, root, ote_dir, args, threshold):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    nncf_performance_path = f'{template_work_dir}/nncf_{template.model_template_id}/performance.json'
    nncf_train_performance_path = f'{template_work_dir}/nncf_{template.model_template_id}/train_performance.json'
    path_exist_assert(nncf_train_performance_path)
    command_line = [template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/nncf_{template.model_template_id}/weights.pth',
                    '--save-performance',
                    nncf_performance_path]
    ret = ote_common(template, root, 'eval', command_line)

    logger.debug(f"Command arguments: ote eval {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(nncf_performance_path)
    with open(nncf_train_performance_path) as read_file:
        trained_performance = json.load(read_file)
    with open(nncf_performance_path) as read_file:
        evaluated_performance = json.load(read_file)

    for k in trained_performance.keys():
        assert trained_performance[k] != 0, f"Trained performance {trained_performance[k]} for {k} must not be 0"
        performance = abs(trained_performance[k] - evaluated_performance[k]) / trained_performance[k]
        assert_info = f"Performance must be <= {threshold}: {trained_performance[k]=}, {evaluated_performance[k]=}"
        assert performance <= threshold, assert_info


def nncf_eval_openvino_testing(template, root, ote_dir, args):
    work_dir, template_work_dir, _ = get_some_vars(template, root)
    command_line = [template.model_template_id,
                    '--test-ann-file',
                    f'{os.path.join(ote_dir, args["--test-ann-files"])}',
                    '--test-data-roots',
                    f'{os.path.join(ote_dir, args["--test-data-roots"])}',
                    '--load-weights',
                    f'{template_work_dir}/exported_nncf_{template.model_template_id}/openvino.xml',
                    '--save-performance',
                    f'{template_work_dir}/exported_nncf_{template.model_template_id}/performance.json']
    ret = ote_common(template, root, 'eval', command_line)

    logger.debug(f"Command arguments: ote eval {' '.join(str(it) for it in command_line)}")
    logger.debug(f"Stdout: {ret['stdout']}\n")
    logger.debug(f"Stderr: {ret['stderr']}\n")
    logger.debug(f"Exit_code: {ret['exit_code']}\n")

    assert ret['exit_code'] == 0, "Exit code must be 0"
    path_exist_assert(f'{template_work_dir}/exported_nncf_{template.model_template_id}/performance.json')
