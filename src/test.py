import os
import yaml
import shutil
import json
import wandb
import numpy as np
from PIL import Image
from pathlib import Path
from utils import set_seed, read_params
from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
import logging
from logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)

# Leer el yaml de parÃ¡metros para saber a que run pertenece en entrenamiento
params = read_params()
set_seed(params["seed"])

run_name_path = Path("trainings", "temp", "run_info.json")
with open(run_name_path, mode="r") as f:
    run_name = json.load(f)["run_name"]

# Sincronizamos el run de entrenamiento con el de test
api = wandb.Api()
runs = api.runs(
    f"unaxetxebe94-upv-ehu/{params['train']['project']}",
    filters={"display_name": run_name}
)

if runs.length > 0:
    try:
        run_id = runs[0].id
        run = wandb.init(project=params['train']['project'], id=run_id, resume="must", job_type="test")
    except Exception as e:
        raise Exception("Error conectandose al run de entrenamiento desde el test:", e)
else:
    run = wandb.init(project=params['train']['project'], name=run_name)
    logger.error("No se encontró un run con ese nombre")

# Inicializamos el modelo
models = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge
    }
model = None
for ckpt in ["checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"]:
    try:
        model = models[str.lower(params["model-type"])](pretrain_weights=f"trainings/training/{ckpt}")
        break
    except Exception as e:
        print(f"Falló {ckpt}: {e}")
if model == None: raise Exception("No se ha cargado el modelo")

# Obtenemos category_map
category_map_path = Path("data", "temp", "category_map.json")
with open(category_map_path, mode="r") as f:
    category_map_ = json.load(f)
category_map = {int(id): cat for id, cat in category_map_.items()}

# Obtenemos las imÃ¡genes del test para ver las predicciones
test_dir = Path("data", 'formatted', "test")
os.makedirs(test_dir, exist_ok=True)
files_in_test_dir = os.listdir(test_dir)
images_to_test = []
for file in files_in_test_dir:
    if file.endswith((".jpg", ".jpeg", ".png", ".webp", ".tiff")):
        images_to_test.append(str(test_dir / file))
# Leemos los gts
with open(test_dir / "_annotations.coco.json") as f:
    coco_gts = json.load(f)

# Construimos un dict {filename -> lista de anotaciones COCO} para acceso rápido
coco_images = {img["id"]: img["file_name"] for img in coco_gts["images"]}
gt_by_filename = {}
for ann in coco_gts["annotations"]:
    fname = coco_images[ann["image_id"]]
    gt_by_filename.setdefault(fname, []).append(ann)

# Hacemos las predicciones y evaluÃ¡mos el modelo
log_list = []
filenames = []

for img_path in images_to_test:
    results = model.predict(img_path)
    filenames.append(img_path)

    # Abrir imagen con PIL para pasar el objeto y conocer el tamaño
    pil_img = Image.open(img_path).convert("RGB")
    w, h = pil_img.size  # width, height

    # Extraer boxes, scores y clases en numpy (soporta torch tensors, listas, etc.)
    boxes = results.xyxy  # asumimos formato [x1,y1,x2,y2]
    scores = results.confidence
    classes = results.class_id

    # Convertir a numpy arrays si vienen como tensores
    try:
        # PyTorch tensor
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
    except Exception:
        pass
    boxes = np.array(boxes, dtype=float)

    try:
        if hasattr(scores, "cpu"):
            scores = scores.cpu().numpy()
    except Exception:
        pass
    scores = np.array(scores, dtype=float)

    try:
        if hasattr(classes, "cpu"):
            classes = classes.cpu().numpy()
    except Exception:
        pass
    classes = np.array(classes, dtype=int)

    box_data = []
    # Si no hay detecciones, boxes podría ser shape (0,) o (0,4)
    if boxes.size != 0 and boxes.ndim == 2 and boxes.shape[1] == 4:
        # Detectar si las coordenadas están normalizadas (0..1)
        max_coord = boxes.max()
        normalized = (max_coord <= 1.0)

        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
            # si normalizado, convertir a píxeles
            if normalized:
                x1 *= w; y1 *= h; x2 *= w; y2 *= h

            # convertir y clampear
            x1 = float(max(0.0, min(x1, w)))
            y1 = float(max(0.0, min(y1, h)))
            x2 = float(max(0.0, min(x2, w)))
            y2 = float(max(0.0, min(y2, h)))

            # crear el dict siguiendo el ejemplo y requisitos de W&B
            bd = {
                "position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
                "class_id": int(cls),
                "box_caption": f"{category_map.get(int(cls), str(int(cls)))} ({float(score):.3f})",
                "domain": "pixel",
                "scores": {"score": float(score)}
            }
            box_data.append(bd)

    # Construir ground truth box_data desde las anotaciones COCO
    # Las anotaciones COCO usan formato [x, y, width, height] en píxeles
    gt_box_data = []
    img_filename = Path(img_path).name
    for ann in gt_by_filename.get(img_filename, []):
        x, y, bw, bh = ann["bbox"]
        x1_gt = float(max(0.0, min(x, w)))
        y1_gt = float(max(0.0, min(y, h)))
        x2_gt = float(max(0.0, min(x + bw, w)))
        y2_gt = float(max(0.0, min(y + bh, h)))
        cat_id = ann["category_id"]
        gt_bd = {
            "position": {"minX": x1_gt, "minY": y1_gt, "maxX": x2_gt, "maxY": y2_gt},
            "class_id": int(cat_id),
            "box_caption": category_map.get(int(cat_id), str(int(cat_id))),
            "domain": "pixel",
        }
        gt_box_data.append(gt_bd)

    # Si no hay boxes, box_data queda vacío — no rompe la visualización
    img_log = wandb.Image(pil_img, boxes={
        "predictions": {
            "box_data": box_data,
            "class_labels": category_map  # debe ser {int: "label"}
        },
        "ground_truth": {
            "box_data": gt_box_data,
            "class_labels": category_map
        }
    })
    log_list.append(img_log)

# Subimos (puedes hacer primero con una sola imagen para debug)
wandb.log({"test_predictions": log_list})
wandb.finish()

logger.info("Se ha terminado el experimento")