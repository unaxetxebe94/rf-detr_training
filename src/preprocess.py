import os
import yaml
import json
from pathlib import Path
from src.utils import set_seed

# Obtenemos el nombre del dataset de entrenamiento
with open("params.yaml") as f:
    params = yaml.safe_load(f)
set_seed(params["seed"])

task_name = params["task-name"]
dataset_train_path = Path("data", task_name, "train")


# Obtenemos los gts del entrenamiento para obtener las categorÃ­as
annotations_path = Path(dataset_train_path, "_annotations.coco.json")
with open(annotations_path, mode="r") as f:
    coco = json.load(f)
categories = coco["categories"]
output = {}


# Simplemente conseguimos el mapeo de cat_id --> cat_name
for cat in categories:
    if cat["id"] not in output:
        output[cat["id"]] = cat["name"]


# Escribimos el mapeo en disco para luego poder leerlo desde el test
output_path = Path("data", task_name, "category_map.json")
with open(output_path, mode="w") as f:
    json.dump(output, f)