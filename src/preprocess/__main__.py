import argparse
import sys

from .run_preprocess import run_preprocess


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preprocesado incremental: solo regenera lo que cambió desde la última ejecución.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Ignora el manifiesto y reconstruye desde cero las salidas de todas las fuentes.",
    )
    args = parser.parse_args(argv)
    return run_preprocess(full=args.full)


if __name__ == "__main__":
    sys.exit(main())
