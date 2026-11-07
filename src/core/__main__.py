import sys
from .train import main as train
from .fuse_datasets import main as run_dataset_fusion
from .mlflow import initialize_mlflow, stop_mlflow

from ..preprocess.run_preprocess import run_preprocess
from ..test.run_test import run_test


def main():
    try:
        initialize_mlflow()
        run_preprocess()
        run_dataset_fusion()
        train()
        run_test()
    except Exception:
        print("Error ejecutando el pipeline")
        raise
    finally:
        stop_mlflow()


if __name__ == "__main__":
    sys.exit(main())
    