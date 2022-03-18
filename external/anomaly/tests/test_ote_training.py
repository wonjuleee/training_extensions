import subprocess
import sys
import os
from pathlib import Path
from ote_anomalib.tools.sample import OteAnomalyTask

CURPATH = Path(__file__).absolute()
model_template_path = "./anomaly_classification/configs/padim/template.yaml"
dataset_path = "/mnt/ote_data/anomaly_datasets/mvtec"
seed = 0
task = OteAnomalyTask(dataset_path=dataset_path, seed=seed, model_template_path=model_template_path)
task.train()
task.export()

#subprocess.run(f"{sys.executable} ../ote_anomalib/tools/sample.py --model_template_path=../anomaly_classification/configs/padim/template.yaml --dataset_path=/mnt/ote_data/anomaly_datasets/mvtec", check=True, shell=True)
