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
uv run python -m src.preprocess --is-master
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
task-name: test_top_bottom  # Es el nombre que se le da a la ejecución
model-type: large  # Variante de RF-DETR que se quiere entrenar. Puede ser: nano, small, medium o large
seed: 42  # Semilla definida para que los experimentos sean reproducibles

# Como los datos están distribuidos en dos ordenadores, se tienen que definir dos fuentes de datos: data-src1 y data-src2. 
data-src1: E:\Images for labelling\Top  # Dataset local
data-src2: \\10.139.140.20\Bottom # Dataset remoto
final-data: C:\training_dataset  # En este path se guardará el dataset final que se usará para entrenar el modelo, después de unificar y preprocesar ambas fuentes de datos

# Parámetros del preprocesamiento
preprocess:
  requires-preprocess: true  # Si se debería de ejecutar el preprocesado o no
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
  run_test: true
  eval_max_dets: 300
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

Coordina funciones separadas por responsabilidad, sin clases de pasos ni logger:

1. `splitter.py` divide las **imágenes originales** en train/valid/test mediante estratificación. Los tiles de una misma puerta nunca se reparten entre splits.
2. `tile_creation.py` recorta los tiles y sus anotaciones. En train sortea una sola vez por imagen: con `saving-prob` guarda todos sus tiles y, en caso contrario, solo los que contienen defectos. Valid y test guardan todos, también los de puertas sin defectos y los tiles negros.
3. `augmenter.py` aumenta únicamente train.
4. `label_corrector.py` corrige las cajas y mantiene el mismo mapeo de categorías en todos los splits.

En test, `source_images` contiene copias completas de las puertas con el mismo `resize`, sin recorte ROI. El COCO conserva sus anotaciones y el origen de cada tile en esas coordenadas. La fusión master/slave mantiene estas copias y sus metadatos.

### `src\test\__main__.py`

Ejemplo: `python -m src.test --weights trainings\training\checkpoint_best_total.pth`.

`--weights` carga exactamente ese checkpoint y busca `training_categories.json` en su mismo directorio. Sin este argumento se usa el entrenamiento más reciente con nombre de fecha. El pipeline integrado pasa directamente el checkpoint que acaba de entrenar, sin buscar otros entrenamientos.

Para evaluar directamente con un umbral fijo para todas las clases: `python -m src.test --global-threshold 0.35`. Acepta valores entre 0 y 1, no lee `valid` ni busca umbrales optimos, y aplica el valor tanto a tiles como a puertas. Sin este argumento se mantiene la calibracion automatica descrita a continuacion.

Los umbrales se calibran exclusivamente con los tiles de `valid`, nunca con `test`. Se comparan los umbrales independientes por clase con un umbral global, barriendo de 0.05 a 0.95 en pasos de 0.05. Gana la estrategia con mayor F1 micro de deteccion (TP/FP/FN acumulados de todas las clases, IoU 0.3); en empate se usa el umbral global. El test por tile y por puerta aplica la estrategia ganadora sin reajustarla. `threshold_calibration` en `metrics.json` guarda ambos F1 de validacion, los candidatos ganadores y la estrategia seleccionada; `per_class_thresholds` contiene los umbrales finalmente aplicados. En la estrategia por clase, las clases sin anotaciones en validacion mantienen el umbral 0.5. Se requiere un split `valid` separado, con anotaciones.

Antes de calibrar o evaluar, las categorias de `valid` y `test` se alinean por nombre exacto con `train\_annotations.coco.json`, usando los IDs del entrenamiento. Se eliminan las anotaciones de clases ausentes en train, tanto de tiles como de `source_images`, y se recalcula `has_defects` por puerta. Las predicciones con IDs ajenos al entrenamiento se descartan. La alineacion se hace en memoria, sin modificar los JSON; el dataset de train debe corresponder al checkpoint evaluado.

Para las puertas, el test lee primero los JSON originales de `data-src1` y `data-src2` configurados en `params.yaml`. Identifica cada puerta por su `source_image_id` (o por `original_file_name` si es unico), recupera la categoria original de cada anotacion por su `id` y la empareja por nombre exacto con las categorias del test antes de alinearla con train. Si el ID de anotacion ha cambiado, busca su caja original redimensionada usando las dimensiones de ambas imagenes, con tolerancia de un pixel por redondeo; solo acepta coincidencias que identifiquen un unico nombre de categoria. Conserva las cajas ya redimensionadas de `source_images` y no modifica los JSON de disco. Los originales deben estar disponibles; si una puerta o anotacion no se puede identificar, la evaluacion se detiene en lugar de comparar IDs incompatibles.

Los nuevos entrenamientos guardan `training_categories.json` junto a los checkpoints. El test usa esta copia si existe junto a `--weights`; para pesos antiguos o copiados se puede indicar el COCO usado realmente al entrenar con `--train-annotations ruta\train\_annotations.coco.json` (tiene prioridad sobre la copia). Se comprueba el orden de los nombres almacenados en el checkpoint, cuando estan disponibles, y se detiene la evaluacion si no coincide. No se deduce una permutacion de clases a partir de los aciertos del test ni se asume que `model.class_names` use los mismos indices que las predicciones. `category_id_to_name` en `metrics.json` registra el mapeo aplicado.

El matching de deteccion empareja primero por clase e IoU, con predicciones ordenadas por confianza. Solo despues identifica confusiones de clase entre las predicciones y anotaciones restantes. Una prediccion incorrecta no puede consumir una anotacion que tenga una coincidencia correcta; una confusion real sigue contando como FP y FN. Este criterio se comparte entre tiles, puertas y comparacion de umbrales.

Se mantienen las métricas y carpetas por tile. En `test_output\metrics.json`, `image_metrics` añade TP/FP/FN/TN, precisión, recall, F1, accuracy y especificidad por puerta; `image_detection_metrics` añade las métricas de detección, incluidos mAP/mAR, sobre las cajas reconstruidas en la imagen completa. Los duplicados de tiles solapados se suprimen mediante NMS por clase con IoU 0.5.

Las imágenes completas anotadas, con GT y predicciones lado a lado, se guardan en `test_output\doors\TP`, `FP`, `FN` y `TN`. Esta clasificación es binaria: basta cualquier predicción que supere su umbral de confianza, sin exigir coincidencia de clase ni IoU con GT. Las métricas de detección sí evalúan clase y solapamiento; una clasificación incorrecta cuenta como FP y FN.

Si faltan `source_images`, metadatos de reconstrucción o archivos de imágenes completas, el test avisa y evalúa **solo los tiles**, sin consultar los originales ni exigir repetir el preprocesado. Se mantienen la calibración sobre `valid`, las métricas y las carpetas por tile. `evaluation_scope` indica `tiles` y `door_evaluation_skip_reason` explica la omisión; no se generan `image_metrics`, `image_detection_metrics` ni métricas por puerta en MLflow. Una carpeta `doors` de una ejecución anterior no representa ese nuevo test y no se borra automáticamente.

Con datos completos, `evaluation_scope` es `tiles_and_doors` y se conserva la evaluación por puerta. Los errores de categorías, anotaciones corruptas, tiles ausentes o inferencia no se ocultan mediante este fallback. Regenerar los datasets antiguos solo es necesario si se quieren obtener también resultados por puerta.
