import json
import os
import copy
import yaml
import logging
from logger import get_logger
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pyvips
from tqdm import tqdm


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


class Resizer:
    """
    Resizes TIFF images and their COCO annotations by a given factor.
    Optionally detects and crops to a bright central ROI before resizing.

    Args:
        input_folder  (str | Path): Folder containing .tiff images and a COCO JSON.
        output_folder (str | Path): Destination folder for processed images and JSON.
        resize_factor (float)     : Scale factor, e.g. 0.5 halves each dimension.
        apply_roi     (bool)      : Whether to detect and crop to the bright ROI.
        roi_threshold (int)       : Pixel brightness threshold (0-255) used to
                                    separate the dark background from the ROI.
                                    Pixels above this value are considered ROI.
        roi_padding   (int)       : Extra pixels added around the detected ROI bbox
                                    (clamped to image bounds).
        num_workers   (int)       : Number of parallel threads for image resizing.
    """

    def __init__(
        self,
        input_folder: str | Path,
        output_folder: str | Path,
        resize_factor: float,
        apply_roi: bool = False,
        roi_threshold: int = 20,
        roi_padding: int = 0,
        num_workers: int = os.cpu_count() or 4,
    ):
        self.input_folder = Path(input_folder)
        self.input_coco = self.input_folder / "_annotations.coco.json"
        self.output_folder = Path(output_folder)
        self.output_coco = self.output_folder / "_annotations.coco.json"
        self.resize_factor = resize_factor
        self.apply_roi = apply_roi
        self.roi_threshold = roi_threshold
        self.roi_padding = roi_padding
        self.num_workers = num_workers
        self.logger = get_logger(__name__, level=logging.DEBUG)

        self.output_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Execute the full pipeline and return the new COCO dict."""
        with open(self.input_coco, "r") as f:
            coco = json.load(f)

        # Map image_id -> annotation list for efficient per-image processing
        id_to_anns: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            id_to_anns.setdefault(ann["image_id"], []).append(ann)

        self.logger.info(
            f"Processing {len(coco['images'])} images "
            f"(factor={self.resize_factor}, apply_roi={self.apply_roi})"
        )

        new_images, new_annotations = self._process_images_parallel(
            coco["images"], id_to_anns
        )

        new_coco = copy.deepcopy(coco)
        new_coco["images"] = new_images
        new_coco["annotations"] = new_annotations

        self.output_coco.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_coco, "w") as f:
            json.dump(new_coco, f, indent=2)

        self.logger.info(f"Saved annotations -> {self.output_coco}")
        return new_coco

    # ------------------------------------------------------------------
    # Parallel orchestration
    # ------------------------------------------------------------------

    def _process_images_parallel(
        self,
        images: list[dict],
        id_to_anns: dict[int, list[dict]],
    ) -> tuple[list[dict], list[dict]]:
        """Process all images in parallel; collect updated images + annotations."""
        result_images: list[dict | None] = [None] * len(images)
        result_anns: list[list[dict]] = [[] for _ in images]

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._process_single_image,
                    img,
                    id_to_anns.get(img["id"], []),
                ): i
                for i, img in enumerate(images)
            }
            with tqdm(total=len(images), unit="img") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        new_img, new_img_anns = future.result()
                        result_images[idx] = new_img
                        result_anns[idx] = new_img_anns
                    except Exception as exc:
                        fname = images[idx]["file_name"]
                        self.logger.error(f"{fname}: {exc}")
                        result_images[idx] = images[idx]
                        result_anns[idx] = id_to_anns.get(images[idx]["id"], [])
                    pbar.update(1)

        flat_anns = [ann for sublist in result_anns for ann in sublist]
        return result_images, flat_anns

    # ------------------------------------------------------------------
    # Single-image pipeline
    # ------------------------------------------------------------------

    def _process_single_image(
        self, img_info: dict, annotations: list[dict]
    ) -> tuple[dict, list[dict]]:
        """
        Full pipeline for one image:
          1. Load
          2. (Optional) detect ROI and crop
          3. Resize
          4. Save
          5. Adjust annotations
        """
        src_path = self.input_folder / img_info["file_name"]
        dst_path = self.output_folder / img_info["file_name"]
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        image = pyvips.Image.new_from_file(str(src_path), access="sequential")

        roi: ROI | None = None
        if self.apply_roi:
            roi = self._detect_roi(src_path, image.width, image.height)
            image = image.crop(roi.x, roi.y, roi.w, roi.h)

        resized = image.resize(self.resize_factor)

        resized.write_to_file(
            str(dst_path),
            compression="lzw",
            predictor="horizontal",
            tile=True,
            tile_width=256,
            tile_height=256,
            pyramid=False,
        )

        new_info = copy.copy(img_info)
        new_info["width"] = resized.width
        new_info["height"] = resized.height

        new_anns = self._adjust_annotations(
            annotations,
            roi=roi,
            out_w=resized.width,
            out_h=resized.height,
        )

        return new_info, new_anns

    # ------------------------------------------------------------------
    # ROI detection
    # ------------------------------------------------------------------

    def _detect_roi(self, src_path: Path, orig_w: int, orig_h: int) -> ROI:
        """
        Detect the bright central ROI in a TIFF image.

        Strategy:
          - Load a small thumbnail (fast even for huge TIFFs via pyvips streaming).
          - Convert to grayscale numpy array.
          - Threshold above `roi_threshold` to build a binary mask.
          - Find the bounding box of all foreground pixels.
          - Map coordinates back to the original resolution.

        Returns:
            ROI with coordinates in the original image space.
        """
        THUMB_SIZE = 1024  # longest side; pyvips keeps aspect ratio
        thumb = pyvips.Image.thumbnail(str(src_path), THUMB_SIZE)
        scale_x = orig_w / thumb.width
        scale_y = orig_h / thumb.height

        # Convert to grayscale
        gray = thumb.colourspace("b-w") if thumb.bands > 1 else thumb

        # pyvips -> numpy (H, W, 1) uint8 -> squeeze to (H, W)
        arr: np.ndarray = np.ndarray(
            buffer=gray.write_to_memory(),
            dtype=np.uint8,
            shape=(gray.height, gray.width, gray.bands),
        ).squeeze()

        # Binary mask: pixels brighter than threshold
        mask = arr > self.roi_threshold

        rows_any = mask.any(axis=1)
        cols_any = mask.any(axis=0)

        if not rows_any.any():
            self.logger.warning(
                f"{src_path.name}: no ROI found with threshold={self.roi_threshold}."
                " Falling back to full image."
            )
            return ROI(0, 0, orig_w, orig_h)

        row_min = int(np.argmax(rows_any))
        row_max = int(len(rows_any) - 1 - np.argmax(rows_any[::-1]))
        col_min = int(np.argmax(cols_any))
        col_max = int(len(cols_any) - 1 - np.argmax(cols_any[::-1]))

        # Map thumbnail coords -> original resolution and apply padding
        x  = max(0,      int(col_min * scale_x) - self.roi_padding)
        y  = max(0,      int(row_min * scale_y) - self.roi_padding)
        x2 = min(orig_w, int((col_max + 1) * scale_x) + self.roi_padding)
        y2 = min(orig_h, int((row_max + 1) * scale_y) + self.roi_padding)

        roi = ROI(x=x, y=y, w=x2 - x, h=y2 - y)
        self.logger.debug(f"{src_path.name}: detected ROI -> {roi}")
        return roi

    # ------------------------------------------------------------------
    # Annotation adjustment
    # ------------------------------------------------------------------

    def _adjust_annotations(
        self,
        annotations: list[dict],
        roi: ROI | None,
        out_w: int,
        out_h: int,
    ) -> list[dict]:
        """
        Adjust all spatial fields in annotations after an optional ROI crop
        followed by a resize.

        Order of operations (mirrors the image pipeline):
          1. Shift coordinates by (-roi.x, -roi.y).
          2. Clip to ROI bounds  [0, roi.w] x [0, roi.h].
          3. Scale by resize_factor.

        Annotations whose bbox falls entirely outside the ROI are discarded.
        """
        f = self.resize_factor
        new_anns = []

        for ann in annotations:
            new_ann = copy.deepcopy(ann)

            # ---- bbox ------------------------------------------------
            if "bbox" in ann:
                bbox = self._adjust_bbox(ann["bbox"], roi, f)
                if bbox is None:
                    self.logger.debug(
                        f"Discarding annotation id={ann.get('id')} (outside ROI)"
                    )
                    continue
                new_ann["bbox"] = bbox
                # Recompute area from the adjusted (and possibly clipped) bbox
                new_ann["area"] = bbox[2] * bbox[3]

            # ---- polygon segmentation --------------------------------
            if "segmentation" in ann and isinstance(ann["segmentation"], list):
                new_ann["segmentation"] = self._adjust_segmentation(
                    ann["segmentation"], roi, f
                )

            # ---- area-only annotations (no bbox) ---------------------
            elif "area" in ann and "bbox" not in ann:
                new_ann["area"] = ann["area"] * (f ** 2)

            new_anns.append(new_ann)

        return new_anns

    def _adjust_bbox(
        self,
        bbox: list[float],
        roi: ROI | None,
        f: float,
    ) -> list[float] | None:
        """
        Shift, clip and scale a COCO [x, y, w, h] bbox.
        Returns None if the bbox is completely outside the ROI.
        """
        x, y, w, h = bbox
        x2, y2 = x + w, y + h

        if roi is not None:
            x  -= roi.x;  x2 -= roi.x
            y  -= roi.y;  y2 -= roi.y

            # Clip to ROI canvas
            x  = max(0.0, x);   x2 = min(float(roi.w), x2)
            y  = max(0.0, y);   y2 = min(float(roi.h), y2)

            if x2 <= x or y2 <= y:
                return None  # Completely outside

        return [x * f, y * f, (x2 - x) * f, (y2 - y) * f]

    def _adjust_segmentation(
        self,
        segmentation: list[list[float]],
        roi: ROI | None,
        f: float,
    ) -> list[list[float]]:
        """
        Shift polygon vertices by the ROI offset and scale by f.
        Coordinates are clamped to >= 0 (upper-left clip; full clip would
        require polygon intersection which is rarely needed for COCO masks).
        """
        off_x = roi.x if roi else 0
        off_y = roi.y if roi else 0

        adjusted = []
        for poly in segmentation:
            new_poly = []
            for i, coord in enumerate(poly):
                if i % 2 == 0:  # x
                    new_poly.append(max(0.0, coord - off_x) * f)
                else:           # y
                    new_poly.append(max(0.0, coord - off_y) * f)
            adjusted.append(new_poly)
        return adjusted


# ----------------------------------------------------------------------
# Quick usage
# ----------------------------------------------------------------------
if __name__ == "__main__":

    with open("params.yaml", mode="r") as f:
        params = yaml.safe_load(f)

    input_folder = Path("data", params["task-name"])
    output_folder = input_folder / "formatted"

    resizer = Resizer(
        input_folder=input_folder,
        output_folder=output_folder,
        resize_factor=1.0,
        apply_roi=params["preprocess"]["apply-roi"],
        roi_threshold=20,
        roi_padding=16,
        num_workers=os.cpu_count() // 2,
    )
    resizer.run()