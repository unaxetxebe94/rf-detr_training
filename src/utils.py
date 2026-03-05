import random, numpy as np, torch, yaml

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def read_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)