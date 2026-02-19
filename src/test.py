import os
import yaml
import json
import wandb
from pathlib import Path
from utils import set_seed
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

# Leer el yaml de parÃ¡metros para saber a que run pertenece en entrenamiento
with open("params.yaml") as f:
    config = yaml.safe_load(f)
set_seed(config["seed"])

run_name_path = Path("trainings", config["task-name"], "run_info.json")

run = wandb.init(project="rf-detr", job_type="testing", name=run_name_path)

models = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge
    }
model = models[str.lower(config["model-type"])](pretrain_weights=f"trainings/{config["task-name"]}/checkpoint_best_total.pth")

category_map_path = Path("data", config["task-name"], "category_map.json")
with open(category_map_path, mode="r") as f:
    category_map = json.load(f)

# Obtenemos las imÃ¡genes del test para ver las predicciones
files_in_test_dir = os.listdir(Path("data", config["task-name"], "test"))
images_to_test = []
for file in files_in_test_dir:
    if file.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff")):
        images_to_test.append(file)

# Hacemos las predicciones y evaluÃ¡mos el modelo
log_list = []
for img_path in images_to_test:
    results = model.predict(img_path)

    # Creamos el objeto de imagen con sus boxes para WandB
    img_log = wandb.Image(img_path, boxes={
        "predictions": {
            "box_data": [
                {
                    "position": {"xmin": b[0], "ymin": b[1], "xmax": b[2], "ymax": b[3]},
                    "class_id": int(cat),
                    "box_caption": f"Score: {score:.2f}"
                } for b, score, cat in zip(results.boxes, results.scores, results.classes)
            ],
            "class_labels": category_map
        }
    })
    log_list.append(img_log)

# Subimos los logs a los servidores de W&B
wandb.log({"test_predictions": log_list})
wandb.finish()