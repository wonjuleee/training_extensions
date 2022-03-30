import argparse
import glob
import os
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--work-dir', required=True)

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    experiments = [
        {
            "work_dir": "Vitens-Coliform",
            "root": "/media/cluster_fs/datasets/sc_counting/Vitens-Coliform-coco",
            "train-ann-file": "annotations/instances_train.json",
            "train-data-root": "images/train/",
            "val-ann-file": "annotations/instances_val.json",
            "val-data-root": "images/val/",
            "test-ann-file": "annotations/instances_test.json",
            "test-data-root": "images/test/",
            "classes": ("coliform", )
        },
        {
            "work_dir": "Vitens-Aeromonas",
            "root": "/media/cluster_fs/datasets/sc_counting/Vitens-Aeromonas-coco",
            "train-ann-file": "annotations/instances_train.json",
            "train-data-root": "images/train/",
            "val-ann-file": "annotations/instances_val.json",
            "val-data-root": "images/val/",
            "test-ann-file": "annotations/instances_test.json",
            "test-data-root": "images/test/",
            "classes": ("aeromonas", ),
        },
        {
            "work_dir": "Vitens-Enterococci",
            "root": "/media/cluster_fs/datasets/sc_counting/Vitens-Enterococci-coco",
            "train-ann-file": "annotations/instances_train.json",
            "train-data-root": "images/train/",
            "val-ann-file": "annotations/instances_val.json",
            "val-data-root": "images/val/",
            "test-ann-file": "annotations/instances_test_rectified.json",
            "test-data-root": "images/test/",
            "classes": ("enterococci", ),
        },
        {
            "work_dir": "MinneApple",
            "root": "/media/cluster_fs/datasets/sc_counting/MinneApple/detection/train",
            "train-ann-file": "train_coco.json",
            "train-data-root": "images",
            "val-ann-file": "val_coco.json",
            "val-data-root": "images",
            "test-ann-file": "test_coco.json",
            "test-data-root": "images",
            "classes": ("apple", ),
        },
        {
            "work_dir": "WGISD1",
            "root": "/media/cluster_fs/datasets/sc_counting/wgisd",
            "train-ann-file": "train_1_class.json",
            "train-data-root": "data",
            "val-ann-file": "test_1_class.json",
            "val-data-root": "data",
            "test-ann-file": "test_1_class.json",
            "test-data-root": "data",
            "classes": ("grape", ),
        },
        {
            "work_dir": "DOTA",
            "root": "/media/cluster_fs/datasets/sc_counting/DOTA",
            "train-ann-file": "anno/DOTA_train.json",
            "train-data-root": "train/images-jpeg",
            "val-ann-file": "anno/DOTA_val.json",
            "val-data-root": "val/images-jpeg",
            "test-ann-file": "anno/DOTA_val.json",
            "test-data-root": "val/images-jpeg",
            "classes": (
                "plane", "baseball-diamond", "bridge", "ground-track-field",
                "small-vehicle", "large-vehicle", "ship", "tennis-court",
                "basketball-court", "storage-tank", "soccer-ball-field",
                "roundabout", "harbor", "swimming-pool", "helicopter"
                ),
        },
        {
            "work_dir": "Vitens-Legionella",
            "root": "/media/cluster_fs/datasets/sc_counting/Vitens-Legionella-coco",
            "train-ann-file": "annotations/instances_train.json",
            "train-data-root": "images/train/",
            "val-ann-file": "annotations/instances_val.json",
            "val-data-root": "images/val/",
            "test-ann-file": "annotations/instances_test.json",
            "test-data-root": "images/test/",
            "classes": ("legionella", )
        },
        {
            "work_dir": "WGISD5",
            "root": "/media/cluster_fs/datasets/sc_counting/wgisd",
            "train-ann-file": "train_5_classes.json",
            "train-data-root": "data",
            "val-ann-file": "test_5_classes.json",
            "val-data-root": "data",
            "test-ann-file": "test_5_classes.json",
            "test-data-root": "data",
            "classes": ("CDY", "CFR", "CSV", "SVB", "SYH"),
        },
    ]

    for exp in experiments:
        work_dir = os.path.join(args.work_dir, exp['work_dir'])
        os.makedirs(work_dir, exist_ok=False)
        overrides = {
            '${WORK_DIR}': '"' + os.path.join(work_dir, 'output') + '"',
            '${TRAIN_ANN_FILE}': '"' + os.path.join(exp['root'], exp['train-ann-file']) + '"',
            '${TRAIN_DATA_ROOT}': '"' + os.path.join(exp['root'], exp['train-data-root']) + '"',
            '${VAL_ANN_FILE}': '"' + os.path.join(exp['root'], exp['val-ann-file']) + '"',
            '${VAL_DATA_ROOT}': '"' + os.path.join(exp['root'], exp['val-data-root']) + '"',
            '${TEST_ANN_FILE}': '"' + os.path.join(exp['root'], exp['test-ann-file']) + '"',
            '${TEST_DATA_ROOT}': '"' + os.path.join(exp['root'], exp['test-data-root']) + '"',
            '${CLASSES}': str(exp["classes"]),
        }

        def instantiate_config(content, overrides, config_path):
            modified_content = list(content)
            for k, v in overrides.items():
                if not any(k in line for line in modified_content):
                    raise RuntimeError(f"{k} not in content.")
                modified_content = [line.replace(k, v) for line in modified_content]

            with open(config_path, 'w') as write_file:
                write_file.write(''.join(modified_content))

        def run_test_py(config_path, checkpoint_path, log_path):
            res = subprocess.run(['python3', 'tools/test.py', config_path, checkpoint_path, '--eval', 'MAE'], capture_output=True, env=os.environ.copy())
            with open(log_path, "w") as write_file:
                write_file.write(res.stdout.decode())

        with open(args.config, 'r') as read_file:
            content = [line for line in read_file]

        config_path = os.path.join(work_dir, os.path.basename(args.config))

        instantiate_config(content, overrides, config_path)
        subprocess.run(['python3', 'tools/train.py', config_path], env=os.environ.copy())

        ckpt_dir = os.path.join(work_dir, 'output')
        best_ckpt = glob.glob(f"{ckpt_dir}/best_*")
        assert len(best_ckpt) == 1

        run_test_py(config_path, best_ckpt[0], os.path.join(ckpt_dir, 'res_on_TEST.txt'))

        for subset in ('TRAIN', 'VAL'):
            overrides['${TEST_ANN_FILE}'] = overrides['${' + subset + '_ANN_FILE}']
            overrides['${TEST_DATA_ROOT}'] = overrides['${' + subset + '_DATA_ROOT}']
            instantiate_config(content, overrides, config_path + f'.{subset}.py')
            run_test_py(config_path + f'.{subset}.py', best_ckpt[0], os.path.join(ckpt_dir, f'res_on_{subset}.txt'))

if __name__ == '__main__':
    main()
