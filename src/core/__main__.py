import sys

from ..test.run_test import run_test
from .mlflow import mlflow_run
from .train import main as train


def main():
    with mlflow_run("pipeline"):
        try:
            # run_preprocess()
            # run_dataset_fusion()
            checkpoint = train()
            run_test(["--weights", str(checkpoint)])
        except Exception:
            print("Error ejecutando el pipeline")
            raise


if __name__ == "__main__":
    sys.exit(main())
