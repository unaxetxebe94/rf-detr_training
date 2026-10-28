# Código de entrenamiento de RF-DETR

Este código se usa para entrenar modelos RF-DETR en el proyecto B1375. Este proyecto se encarga de la preparación de datos, el entrenamiento del modelo y su evaluación. 

## Vocabulario que se usa en los comentarios del código
* **Tile**: Las imágenes originales son de $16k \times 21k$ píxeles, más o menos. Evidentemente, no se puede entrenar una red neuronal con imágenes de ese tamaño, por lo que lo solemos dividir en trozos. A cada trozo le llamamos **tile**-s ya que a la inferencia por trozos se le llama **tile inference**. 

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

Se mantienen las métricas y carpetas por tile. En `test_output\metrics.json`, `image_metrics` añade TP/FP/FN/TN, precisión, recall, F1, accuracy y especificidad por puerta; `image_detection_metrics` añade las métricas de detección, incluidos mAP/mAR, sobre las cajas reconstruidas en la imagen completa. Los duplicados de tiles solapados se suprimen mediante NMS por clase con IoU 0.5.

Las imágenes completas anotadas, con GT y predicciones lado a lado, se guardan en `test_output\doors\TP`, `FP`, `FN` y `TN`. Esta clasificación es binaria: basta cualquier predicción que supere su umbral de confianza, sin exigir coincidencia de clase ni IoU con GT. Las métricas de detección sí evalúan clase y solapamiento; una clasificación incorrecta cuenta como FP y FN.

Es necesario regenerar los datasets antiguos para disponer del split por puerta, las imágenes completas y sus coordenadas.
