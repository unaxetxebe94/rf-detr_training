import random, numpy as np, torch, yaml, json, os
from pathlib import Path
import logging
from logger import get_logger

logger = get_logger(__name__, level=logging.DEBUG)

def set_seed(seed=42):
    # 1. Semillas básicas de Python y CPU
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 2. Semillas de GPU (todas las disponibles)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Para sistemas multi-GPU
    
    # 3. Configuración para evitar TF32 (TensorFloat-32)
    # Las GPUs modernas (RTX 30xx/40xx) usan TF32 por defecto para ir más rápido, 
    # pero esto reduce la precisión y puede causar resultados distintos.
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    # 4. Forzar determinismo en algoritmos (Crucial para ViTs)
    # Esto obliga a usar algoritmos deterministas para multiplicaciones de matrices (GEMM)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 5. Configuración de cuBLAS (Esencial si usas torch.use_deterministic_algorithms)
    # Sin esto, muchas operaciones de álgebra lineal lanzarán error si intentas ser determinista
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # 6. Activación del modo determinista estricto de PyTorch
    # Si alguna operación NO puede ser determinista, PyTorch lanzará un error para avisarte.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"⚠️ Aviso: Algunas operaciones no soportan determinismo estricto: {e}")

    print(f"✅ Semilla {seed} configurada. Entorno blindado para reproducción.")



def read_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)



def save_mapping():
        params = read_params()
        dataset_train_path = Path(params["final-data"], "train")
        
        annotations_path = dataset_train_path / "_annotations.coco.json"
        with open(annotations_path, mode="r") as f:
            coco = json.load(f)
        categories = coco["categories"]
        output = {}
        
        for cat in categories:
            if cat["id"] not in output:
                output[cat["id"]] = cat["name"]

        output_path = Path("data", "temp", "category_map.json")
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, mode="w") as f:
            json.dump(output, f)

        logger.debug("Se ha guardado el category_map")
