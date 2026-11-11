# Código de entrenamiento de RF-DETR

Este código se usa para entrenar modelos RF-DETR en el proyecto B1375. Este proyecto se encarga de la preparación de datos, el entrenamiento del modelo y su evaluación. 

## Vocabulario que se usa en los comentarios del código
* **Tile**: Las imágenes originales son de $16k \times 21k$ píxeles, más o menos. Evidentemente, no se puede entrenar una red neuronal con imágenes de ese tamaño, por lo que lo solemos dividir en trozos. A cada trozo le llamamos **tile**-s ya que a la inferencia por trozos se le llama **tile inference**. 

## Entorno del proyecto (uv)

El proyecto usa [uv](https://docs.astral.sh/uv/) para que el entorno sea 100% reproducible:

* `pyproject.toml` → declaración de las dependencias directas y grupos de desarrollo.
* `uv.lock` → versiones exactas de **todas** las dependencias (con hashes); se debe **versionar en git**.
* `.python-version` → versión exacta de Python (3.12); uv la descarga e instala automáticamente si hace falta.

### Cómo reproducir el entorno

```bash
uv sync          # crea .venv e instala exactamente lo que hay en uv.lock
```

Cualquier comando se ejecuta a través de `uv run`, que garantiza que se usa el entorno bloqueado:

```bash
uv run pytest                  # ejecutar los tests
uv run python -m src.test      # evaluación con ponderes: uv run python -m src.test --weights trainings/training/checkpoint_best_total.pth
uv run python -m src.preprocess
uv run python -m src.core.train
```

Para actualizar dependencias (se regenera `uv.lock` y hay que hacer commit):

```bash
uv add paquete-nuevo       # añadir dependencia a [project]
uv add --dev ruff          # añadir dependencia de desarrollo
uv lock --upgrade          # actualizar uv.lock a versiones más nuevas
```

El pipeline (`run_pipeline.ps1` / `run_pipeline.sh`) ya ejecuta todas las etapas con `uv run`, así que solo hace falta tener `uv` en el PATH (en Windows: `winget install astral-sh.uv` o `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`).

### MLflow

El pipeline guarda experimentos y runs en el almacén de archivos `mlflow` de la raíz del proyecto. Imprime su URI absoluta y el `run_id`; el logger de entrenamiento y el test comparten esa run. Una ejecución correcta termina como `FINISHED` y un error como `FAILED`. También se registran los parámetros efectivos y las rutas del entrenamiento y del checkpoint.

Abre la interfaz **desde la raíz del repositorio**, indicando ese mismo backend:

```powershell
$env:MLFLOW_ALLOW_FILE_STORE = "true"
$trackingUri = uv run python -c "from pathlib import Path; print((Path.cwd() / 'mlflow').as_uri())"
uv run mlflow ui --backend-store-uri "$trackingUri"
```

No uses `cd .\mlflow; uv run mlflow ui` sin backend: puede abrir una base SQLite distinta, mostrando solo `Default` o experimentos sin los runs existentes. No hace falta migrar ni borrar las bases SQLite para consultar los runs del almacén de archivos.

## Archivos importantes

### `params.yaml`

Este es el archivo que se usa para cambiar el comportamiento del pipeline. Aquí se definen de forma declarativa todos los parámetros de tanto la preparación de datos, como del entrenamiento y de los tests. Aquí te dejo un ejemplo de como se ve el archivo:

```yaml
experiment-name: test_top_bottom  # MLflow experiment name
model-type: large  # Variante de RF-DETR que se quiere entrenar. Puede ser: nano, small, medium o large
seed: 42  # Semilla definida para que los experimentos sean reproducibles

data:
  sources:
    - C:\datasets\source_a
    - C:\datasets\source_b
  training-dataset-path: C:\datasets\training_dataset

# Parámetros del preprocesamiento
preprocess:
  classes: null  # All available classes; or a subset such as ["scratch", "dent"]
  saving-prob: 1.0  # Solo train: probabilidad por imagen de guardar todos sus tiles; si no, solo los que tienen defectos
  min-retained-dimension-ratio: 0.10  # Descarta anotaciones cortadas si ancho o alto queda por debajo de este ratio
  resize: 0.33  # Antes de dividir la imagen en tiles, se le aplica un resize a las imágenes
  apply-roi: false  # Si es true, primero se recortará la imagen de la imagen original y solo se trozeará esa parte. Si es false, el troceo será sobre la imagen entera. 
  train-ratio: 0.7
  test-ratio: 0.15
  val-ratio: 0.15
  # Aumentos de datos adicionales a los predeterminados del entrenamiento de RF-DETR
  augmentations-per-image: 2  # Cuántas veces se repite la misma imagen en cada época de entrenamiento.
  max-transforms-per-sample: 5  # Cuantas transformaciones se le pueden aplicar como máximo a una imagen de entrenamiento

# Variables de entrenamiento
train:
  lr: 0.0001
  lr_encoder: 0.00015
  batch_size: 4
  grad_accum_steps: 6
  epochs: 15
  weight_decay: 1e-4
  tensorboard: true
```

### `run_pipeline.ps1`

El pipeline se ejecuta desde un script de PowerShell, el archivo `.\run_pipeline.ps1`. Este archivo simplemente ejecuta diferentes scripts de python: 

1. Script de preparación de datos.
1. Script de entrenamiento.
1. ...

Es algo así como el programa principal o el *main*. Usamos PowerShell en vez de Python simplemente para aprovechar y ejecutar comandos de git en el propio script. 

## Código fuente en Python

El código fuente se encuentra en `.\src`. Vamos a desglosar todos los archivos:

### `logger.py` 

Simplemente guarda las funciones relacionadas con el logging. 

### `utils.py`

Guarda funciones de ayuda que se usan a lo largo del pipeline. 

### `src\preprocess\__main__.py`

Run `uv run python -m src.preprocess` to prepare all configured sources, fuse them and select the training classes. No separate fusion command is needed. The training/test entry point remains unchanged; rerun preprocessing explicitly when changing sources or selected classes.

1. `splitter.py` stratifies original images into train/valid/test. Tiles from the same original never cross splits.
2. `tile_creation.py` creates tiles and annotations. Train samples empty tiles once per original using `saving-prob`; valid/test retain all tiles.
3. `augmenter.py` augments train only.
4. `annotations.py` corrects boxes and harmonizes category IDs across each source's splits.
5. `fusion.py` merges only the sources from the current preprocessing run, in `data.sources` order. Old source folders are not included. Category names are compared using `lower()` and case-equivalent categories are merged.
6. `class_filter.py` applies `preprocess.classes` **after fusion**, updating tile and original-image annotations and recomputing original-image `has_defects`.

Set `preprocess.classes` to a nonempty list of category names, for example `["Scratch", "DENT"]`. Names must be a subset of the fused catalog and are matched using `lower()`. Unknown names, empty/invalid lists and case-insensitive duplicates raise errors. `null` (the default) retains all available classes. Configuration names must be quoted if they are numeric, for example `["0", "2"]`.

The final dataset is written to `data.training-dataset-path` with the same lowercase category catalog in train/valid/test. IDs are contiguous from 1, following the configured list order (or fused catalog order for `null`), so RF-DETR's zero-based prediction labels map back using `label + 1`. Images remain in the dataset when their removed annotations leave them negative. Raw sources and intermediate per-source datasets are not filtered.

Test `source_images` contains full original images resized without ROI cropping, their annotations and each tile's origin. `tile_to_source_scale` maps tile coordinates (including `tile_origin`) to the resized full image, including when a small image or ROI was enlarged to the tile size. Fusion and class selection preserve these files and coordinates. Rerun preprocessing to add this metadata to existing datasets; incremental runs update the JSON without rewriting unchanged images. `src.core.fuse_datasets` remains a compatibility entry point for fusing already-preprocessed source folders and applying the configured class selection.

#### Preprocesado incremental

Cada fuente guarda `sources\<nombre>_source<i>\_preprocess_manifest.json` con, por imagen original, la huella del archivo (tamaño, mtime y SHA-256), su split y los archivos generados. En cada ejecución los JSON se reconstruyen desde las anotaciones originales (barato) y solo se escriben imágenes donde hace falta:

| Cambio | Qué se regenera |
| --- | --- |
| Imagen nueva | Se asigna a un split (las existentes conservan el suyo; la nueva va al split más por debajo de su ratio, priorizando etiquetas) y se generan sus tiles, aumentos y `source_image`. |
| Archivo de imagen modificado | Sus tiles, aumentos y `source_image`. |
| Anotaciones modificadas o categorías nuevas | Solo los JSON; no se reescriben tiles. |
| Imagen eliminada | Se borran sus archivos (también en el dataset fusionado). |
| `model-type`, `resize`, `apply-roi` | Todos los tiles (y `source_images` si cambia `resize`). |
| `saving-prob` | Solo se añaden o borran los tiles vacíos de train afectados. |
| `augmentations-per-image`, `max-transforms-per-sample` | Solo los aumentos. |
| `seed` o ratios | Se rehace el split; los tiles que cambian de split se mueven. |

La fusión solo copia imágenes nuevas o modificadas y borra las que el JSON ya no referencia. El muestreo de `saving-prob` y los aumentos usan una semilla por imagen/tile, así que no dependen del resto de imágenes. Un manifiesto ausente, corrupto o de otra versión provoca la reconstrucción completa de esa fuente. Para forzarla en todas:

```powershell
uv run python -m src.preprocess --full
```

### `src\test\__main__.py`

La evaluación se organiza por responsabilidad; `run_test.py` solo interpreta los argumentos y lanza el test:

| Módulo en `src\test` | Responsabilidad |
| --- | --- |
| `__main__.py`, `run_test.py` | Entrada CLI, selección del checkpoint, carga del modelo y ciclo de la run de MLflow. |
| `evaluation.py` | Coordinación del dataset, calibración sobre `valid` y evaluación por tiles y puertas. |
| `categories.py` | Alineación de categorías con entrenamiento y originales; validación del orden del checkpoint. |
| `predictions.py` | Inferencia, conversión de cajas e IDs y filtrado por confianza. |
| `tile_evaluation.py` | Ground truth, métricas e imágenes anotadas por tile. |
| `image_evaluation.py` | Disponibilidad de puertas completas, reconstrucción, NMS y evaluación por puerta. |
| `reporting.py` | Registro de resultados en MLflow y escritura de `metrics.json`. |
| `thresholds.py`, `metrics.py`, `matching.py`, `iou.py`, `drawing.py` | Algoritmos de umbrales, métricas, matching, geometría y renderizado. |

Para evaluar con un modelo ya cargado, se puede importar `evaluate_dataset` de `src.test.evaluation`; el import existente desde `src.test.run_test` sigue disponible.

Ejemplo: `python -m src.test --weights trainings\training\checkpoint_best_total.pth`.

`--weights` carga exactamente ese checkpoint y busca `training_categories.json` en su mismo directorio. Sin este argumento se usa el entrenamiento más reciente con nombre de fecha. El pipeline integrado pasa directamente el checkpoint que acaba de entrenar, sin buscar otros entrenamientos.

Para evaluar directamente con un umbral fijo para todas las clases: `python -m src.test --global-threshold 0.35`. Acepta valores entre 0 y 1, no lee `valid` ni busca umbrales optimos, y aplica el valor tanto a tiles como a puertas. Sin este argumento se mantiene la calibracion automatica descrita a continuacion.

Los umbrales se calibran exclusivamente con los tiles de `valid`, nunca con `test`. Se comparan los umbrales independientes por clase con un umbral global, barriendo de 0.05 a 0.95 en pasos de 0.05. Gana la estrategia con mayor F1 micro de deteccion (TP/FP/FN acumulados de todas las clases, IoU 0.3); en empate se usa el umbral global. El test por tile y por puerta aplica la estrategia ganadora sin reajustarla. `threshold_calibration` en `metrics.json` guarda ambos F1 de validacion, los candidatos ganadores y la estrategia seleccionada; `per_class_thresholds` contiene los umbrales finalmente aplicados. En la estrategia por clase, las clases sin anotaciones en validacion mantienen el umbral 0.5. Se requiere un split `valid` separado, con anotaciones.

Antes de calibrar o evaluar, las categorias de `valid` y `test` se alinean por nombre sin distinguir mayusculas/minusculas con `train\_annotations.coco.json`, usando los IDs del entrenamiento. Se eliminan las anotaciones de clases ausentes en train, tanto de tiles como de `source_images`, y se recalcula `has_defects` por puerta. Las predicciones con IDs ajenos al entrenamiento se descartan. La alineacion se hace en memoria, sin modificar los JSON; el dataset de train debe corresponder al checkpoint evaluado.

For original images, evaluation reads the original COCO files from `data.sources` in `params.yaml`. It identifies each original by `source_image_id` (or a unique `original_file_name`), recovers each annotation's original category by annotation ID, and matches category names using `lower()` before aligning with train. If an annotation ID has changed, it matches the resized original box with a one-pixel tolerance, accepting only an unambiguous category. Existing resized `source_images` boxes are preserved; disk JSON files are not changed. Originals must be available; ambiguous images or annotations raise errors rather than comparing incompatible IDs.

Los nuevos entrenamientos guardan `training_categories.json` junto a los checkpoints. El test usa esta copia si existe junto a `--weights`; para pesos antiguos o copiados se puede indicar el COCO usado realmente al entrenar con `--train-annotations ruta\train\_annotations.coco.json` (tiene prioridad sobre la copia). Se comprueba el orden de los nombres almacenados en el checkpoint, cuando estan disponibles, y se detiene la evaluacion si no coincide. No se deduce una permutacion de clases a partir de los aciertos del test ni se asume que `model.class_names` use los mismos indices que las predicciones. `category_id_to_name` en `metrics.json` registra el mapeo aplicado.

El matching de deteccion empareja primero por clase e IoU, con predicciones ordenadas por confianza. Solo despues identifica confusiones de clase entre las predicciones y anotaciones restantes. Una prediccion incorrecta no puede consumir una anotacion que tenga una coincidencia correcta; una confusion real sigue contando como FP y FN. Este criterio se comparte entre tiles, puertas y comparacion de umbrales.

Se mantienen las métricas y carpetas por tile. En `test_output\metrics.json`, `image_metrics` añade TP/FP/FN/TN, precisión, recall, F1, accuracy y especificidad por puerta; `image_detection_metrics` añade las métricas de detección, incluidos mAP/mAR, sobre las cajas reconstruidas en la imagen completa. Los duplicados de tiles solapados se suprimen mediante NMS por clase con IoU 0.5.

Las imágenes completas anotadas, con GT y predicciones lado a lado, se guardan en `test_output\doors\TP`, `FP`, `FN` y `TN`. Esta clasificación es binaria: basta cualquier predicción que supere su umbral de confianza, sin exigir coincidencia de clase ni IoU con GT. Las métricas de detección sí evalúan clase y solapamiento; una clasificación incorrecta cuenta como FP y FN.

Si faltan `source_images`, metadatos de reconstrucción o archivos de imágenes completas, el test avisa y evalúa **solo los tiles**, sin consultar los originales ni exigir repetir el preprocesado. Se mantienen la calibración sobre `valid`, las métricas y las carpetas por tile. `evaluation_scope` indica `tiles` y `door_evaluation_skip_reason` explica la omisión; no se generan `image_metrics`, `image_detection_metrics` ni métricas por puerta en MLflow. Una carpeta `doors` de una ejecución anterior no representa ese nuevo test y no se borra automáticamente.

Con datos completos, `evaluation_scope` es `tiles_and_doors` y se conserva la evaluación por puerta. Los errores de categorías, anotaciones corruptas, tiles ausentes o inferencia no se ocultan mediante este fallback. Regenerar los datasets antiguos solo es necesario si se quieren obtener también resultados por puerta.

### Inferencia de una imagen grande

```powershell
uv run python -m src.scripts.infer_image "D:\imagenes\puerta.png" `
    --weights trainings\2026-11-09_14-30-11\checkpoint_best_total.pth `
    --output-dir inference_output\puerta `
    --threshold 0.5
```

No necesita anotaciones de la imagen de entrada ni ejecuta entrenamiento o MLflow. Reutiliza el ROI, `preprocess.resize` y el troceado del preprocesado, incluidos los recortes solapados del borde, e infiere todos los tiles. Usa `model-type` de `params.yaml` (o `--model-type`) para cargar el modelo y seleccionar el tamano de tile; estos parametros deben corresponder al entrenamiento del checkpoint. Sin `--weights`, selecciona el ultimo entrenamiento con nombre de fecha.

No lee archivos de anotaciones COCO ni `training_categories.json`: solo necesita la imagen, los pesos y la configuracion de preprocesado. Dibuja unicamente las cajas predichas, sin etiquetas ni texto.

Guarda `predictions.json` con cajas `[x, y, ancho, alto]` en **pixeles de la imagen original**, deshaciendo el escalado y el desplazamiento del ROI. Los duplicados se eliminan mediante NMS por clase (`--nms-iou`, por defecto 0.5). `preview.png` muestra las detecciones sobre la imagen completa, reducida a un maximo de 2048 pixeles por lado (`--preview-size`) para no cargar la imagen gigante en memoria. Sin `--output-dir`, guarda en `inference_output\<nombre de imagen>`.

### FP/FN confidence trade-off charts

Run from the repository root:

```powershell
uv run python -m src.scripts.plot_confidence_tradeoff
```

The script uses the model type from `params.yaml`, the latest dated training checkpoint, and the configured dataset's `valid` and `test` splits. It does not train, change parameters, or create MLflow runs. To select a checkpoint and output directory explicitly:

```powershell
uv run python -m src.scripts.plot_confidence_tradeoff `
    --weights trainings\2026-11-09_14-30-11\checkpoint_best_total.pth `
    --output-dir charts\confidence_tradeoff `
    --step 0.01
```

Additional options are `--dataset-dir`, `--train-annotations`, `--level {tiles,images,all}`, and repeatable `--original-data-dir`. Training categories come from the snapshot next to the checkpoint when available, and their order is checked against the model. COCO grouping metadata (`supercategory: none`) is not treated as a defect class.

The baseline is calibrated **only on validation tiles**, using the same F1 comparison between global and per-class thresholds as the regular test. All charts share that baseline: it is not independently optimized for original-image detection or binary decisions. The same offset is added to every class threshold, decreasing until the first class reaches **0**, and increasing until the first reaches **1**. The baseline and exact endpoints are always included. For example, baseline thresholds `[0.2, 0.8]` produce endpoints `[0, 0.6]` and `[0.4, 1]`.

The horizontal axis shows **percentage-point changes**, not absolute thresholds: 0 means the baseline, and -5 subtracts 0.05 from every class threshold. Each chart shows the full sweep and a +/-10-point detail around the baseline. The default step is 0.01 (1 percentage point). Inference runs only once per validation/test tile with a minimum confidence of 0; all curves reuse those predictions, subject to the model's detection-count limit.

The default `--level all` exports the tile chart and both original-image charts when metadata is available:

| Output location under `DATASET\test_output\confidence_tradeoff` | Counting unit |
| --- | --- |
| Root directory | Detection errors per tile, not unique physical defects or rejected images. |
| `original_image_detection` | Detection errors against original-image boxes, after reconstructing tile coordinates and applying per-class NMS at IoU 0.5. |
| `original_image_binary` | Accept/reject decisions per original image: FP = defect-free images rejected; FN = defective images accepted. |

Each location contains `confidence_tradeoff.png` for sharing, `.svg` for vector presentations, `.csv` with counts and class thresholds, and `.json` with calibration, checkpoint, dataset and sweep metadata. Binary CSV/JSON reports also include TN.

Detection charts use class-aware matching at IoU 0.3; a wrong class counts as both FP and FN. Binary decisions follow the regular image evaluation: **any** retained defect prediction rejects the image, without class or IoU matching. An image is defective if its aligned ground-truth annotations contain at least one trained defect class. Overlapping tile detections are suppressed per class before original-image matching; raw inference and reconstruction are shared across the sweep.

Original-image charts require `source_images` ground-truth metadata and every tile's `source_image_id` and `tile_origin`. Full-image pixel files are not needed because the script does not render those images. Original categories are aligned using the configured `data.sources` COCO files; use repeated `--original-data-dir` arguments to override them. Invalid or ambiguous category mappings still raise an error.

If metadata is absent or incomplete, `--level all` prints an English warning, records the skip reason in the tile JSON, and exports only tile curves. `--level images` instead fails explicitly, without substituting tile results for image results:

```powershell
uv run python -m src.scripts.plot_confidence_tradeoff --level images
```

Use `--level tiles` to request only tile curves. Only requested/available reports are updated; files from earlier runs are not deleted automatically.

These charts show the sample size and baseline counts, not monetary costs. Choosing a threshold from the curves uses `test` as a decision set; an independent sample is needed for a final unbiased assessment. Synthetic tests cover reconstruction, NMS, category alignment and binary decisions; real-image validation requires a dataset containing original-image metadata.
