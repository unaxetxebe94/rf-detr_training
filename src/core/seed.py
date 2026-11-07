import os
import random
import logging
import torch
import numpy as np

from core.logger import get_logger


logger = get_logger(__name__, level=logging.DEBUG)


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Para sistemas multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print(f"⚠️ Aviso: Algunas operaciones no soportan determinismo estricto: {e}")

    print(f"✅ Semilla {seed} configurada. Entorno blindado para reproducción.")