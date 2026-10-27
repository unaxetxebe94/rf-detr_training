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
  saving-prob: 1.0  # Probabilidad de que se guarde un tile aunque no tenga ninguna anotación
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

### `preprocess_pipeline.py`

El preprocesado es seguramente la parte más compleja del pipeline en cuanto a código. Este script simplemente funciona como un coordinador de las diferentes clases que creo en `.\src\preprocess_stage`. Cada clase se encarga de un paso distinto del preprocesado:

*  **TileCreator**: Se encarga de coger las imágenes enormes de $16000\times21000$ píxeles, y de dividirlos en trozos. También divide las anotaciones en tiles, calculando el solapamiento entre el trozo de la puerta y cada anotación.
* **Splitter**: Divide todos los trozos en los splits de train, test y valid, intentando dividir las anotaciones de forma equitativa.
* **Augmenter**: El entrenamiento estándar de RF-DETR usa muy pocas aumentaciones, por lo que en la carpeta de entrenamiento, por cada imagen creamos algunas copias con ciertas transformaciones.
* **LabelCorrector**: Clase que se cerciora de que las IDs en las anotaciones empiezan en 0 y son secuenciales, en vez de tener un ID 7 y que el siguiente sea 20.



