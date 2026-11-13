"""Pre-training checks of the view: every referenced file exists (quick) or decodes with its size (deep)."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ...preprocess.annotations import SPLITS, read_annotations
from .cache import read_tile_array
from .view import ANNOTATIONS_FILE

MAX_REPORTED = 20


def _check(path: Path, width: int | None, height: int | None, deep: bool) -> str | None:
    try:
        if path.stat().st_size == 0:
            return f"{path}: fichero vacío"
        if deep:
            array = read_tile_array(path)
            if width is not None and (array.shape[1], array.shape[0]) != (width, height):
                return f"{path}: tamaño {array.shape[1]}x{array.shape[0]}, esperado {width}x{height}"
    except FileNotFoundError:
        return f"{path}: no existe"
    except Exception as error:
        return f"{path}: {error}"
    return None


def verify_view(dataset_dir: Path, *, deep: bool = False, n_jobs: int = 8) -> int:
    """Raise if any image of the view is missing/empty (or undecodable with ``deep``); return the files checked."""
    dataset_dir = Path(dataset_dir)
    jobs = []
    for split in SPLITS:
        coco = read_annotations(dataset_dir / split / ANNOTATIONS_FILE)
        jobs.extend(
            (dataset_dir / split / image["file_name"], image["width"], image["height"]) for image in coco["images"]
        )
        jobs.extend((dataset_dir / split / source["file_name"], None, None) for source in coco.get("source_images", []))
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as executor:
        problems = [problem for problem in executor.map(lambda job: _check(*job, deep), jobs) if problem]
    if problems:
        listed = "\n  ".join(problems[:MAX_REPORTED])
        raise RuntimeError(
            f"La vista {dataset_dir} tiene {len(problems)} ficheros inválidos (ejecuta con --full para regenerar):\n"
            f"  {listed}"
        )
    return len(jobs)
