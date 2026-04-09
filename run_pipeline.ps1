<#
.SYNOPSIS
    Ejecuta el pipeline completo de entrenamiento RF-DETR con versionado de datos via DVC.

.PARAMETER CondaEnv
    Nombre del entorno conda a usar. Por defecto: "rf-detr".

.EXAMPLE
    .\run_pipeline.ps1
    .\run_pipeline.ps1 -CondaEnv "mi-entorno"
#>

param(
    [string]$CondaEnv = "rf-detr"
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

function Write-Step {
    param([string]$Msg)
    $ts = Get-Date -Format "HH:mm:ss"
    Write-Host ""
    Write-Host "[$ts] ======================================" -ForegroundColor Cyan
    Write-Host "[$ts]  $Msg" -ForegroundColor Cyan
    Write-Host "[$ts] ======================================" -ForegroundColor Cyan
}

function Write-Ok   { param([string]$M); Write-Host "  OK  $M" -ForegroundColor Green  }
function Write-Warn { param([string]$M); Write-Host "  WARNING  $M" -ForegroundColor Yellow }
function Write-Err  { param([string]$M); Write-Host "  ERROR  $M" -ForegroundColor Red    }

function Invoke-Stage {
    param([string]$StageName, [string]$Script)
    Write-Step "Etapa: $StageName"
    conda run -n $CondaEnv python $Script
    if ($LASTEXITCODE -ne 0) {
        Write-Err "La etapa '$StageName' ha fallado (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }
    Write-Ok "$StageName completado."
}


# ─────────────────────────────────────────────────────────────────────────────
# Preguntar si se quiere guardar el dataset en git
# ─────────────────────────────────────────────────────────────────────────────

Write-Step "Versionado del dataset"

# COMMIT
$response = Read-Host "¿Quieres guardar la base de datos en git para reproducir el experimento (COMMIT)? ([y]/n)"

if ($response -eq "y" -or $response -eq "") {
    Write-Host "Guardando estado en git..." -ForegroundColor Yellow

    mkdir annotations -Force
    Copy-Item "data\formatted\train\_annotations.coco.json" "annotations\train_annotations.json" -Force
    Copy-Item "data\formatted\test\_annotations.coco.json" "annotations\test_annotations.json" -Force
    Copy-Item "data\formatted\val\_annotations.coco.json" "annotations\val_annotations.json" -Force

    git add .
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Warn "No hay cambios para commitear."
    } else {
        git commit -m "Snapshot dataset antes de ejecutar experimento $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        Write-Ok "Snapshot guardado en git."
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Warn "No se pudo hacer commit (quizá no hay cambios)."
    }
    else {
        Write-Ok "Snapshot guardado en git."
    }

    # PUSH
    $response = Read-Host "¿Quieres hacer PUSH de los cambios? ([y]/n)"

    if ($response -eq "y" -or $response -eq "") {
        Write-Host "Subiendo cambios al remoto..." -ForegroundColor Yellow

        git push

        if ($LASTEXITCODE -ne 0) {
            Write-Warn "No se pudo hacer push"
        }
        else {
            Write-Ok "Snapshot guardado en git."
        }
    }
    elseif ($response -eq "n") {
        Write-Warn "Se ejecutará el pipeline sin versionar el dataset."
    }
    else {
        Write-Err "Respuesta inválida. Usa 'y' o 'n'."
        exit 1
    }
}
elseif ($response -eq "n") {
    Write-Warn "Se ejecutará el pipeline sin versionar el dataset."
}
else {
    Write-Err "Respuesta inválida. Usa 'y' o 'n'."
    exit 1
}


# ─────────────────────────────────────────────────────────────────────────────
# Preguntar si se tienen que preparar los datos
# ─────────────────────────────────────────────────────────────────────────────

Write-Step "Preparación de datos"

$response = Read-Host "¿Los datos ya están preparados? ([y]/n)"

if ($response -eq "y" -or $response -eq "") {
    Write-Host "No se ejecutará ni el preprocesado ni la fusión" -ForegroundColor Yellow

    $isDataPrepared = $true
}
elseif ($response -eq "n") {
    Write-Warn "Se ejecutará el preprocesado"
    $isDataPrepared = $false
}
else {
    Write-Err "Respuesta inválida. Usa 'y' o 'n'."
    exit 1
}


# ─────────────────────────────────────────────────────────────────────────────
# Etapas del pipeline
# ─────────────────────────────────────────────────────────────────────────────

if (-not $isDataPrepared) {
    # 0. Preprocesado
    Invoke-Stage "Preprocesado" "src/preprocess_pipeline.py"

    # 1. Fusionar los datasets del master y slave
    Invoke-Stage "Fusión de datasets" "src/fuse_datasets.py"
}

# 2. Entrenamiento
Invoke-Stage "Entrenamiento" "src/train.py"

# 3. Procesado de resultados + logging a W&B
Invoke-Stage "Procesado de resultados" "src/process_results.py"

# 4. Test
Invoke-Stage "Test" "src/test.py"


# ─────────────────────────────────────────────────────────────────────────────
# Resumen final
# ─────────────────────────────────────────────────────────────────────────────

$Elapsed = (Get-Date) - $StartTime
Write-Step "Pipeline completado"
Write-Ok "Tiempo total: $($Elapsed.ToString('hh\:mm\:ss'))"
Write-Host ""

$timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
Copy-Item "trainings\training\checkpoint_best_total.pth" "$timestamp.pth"

$response = Read-Host "¿Quieres eliminar los archivos del dataset, entrenamiento y resultados? (y/[n])"

if ($response -eq "y") {
    Write-Host "Eliminando archivos..." -ForegroundColor Yellow

    Remove-Item -r data\*
    Remove-Item -r .\trainings

    if ($LASTEXITCODE -ne 0) {
        Write-Warn "No se pudieron eliminar todos los archivos"
        Write-Error "Recuerda eliminar archivos para que el espacio en disco no se llene"
    }
    else {
        Write-Ok "Archivos eliminados."
    }
}
elseif ($response -eq "n" -or $response -eq "") {
    Write-Warn "Los datos se mantendrán en disco."
}
else {
    Write-Err "Respuesta inválida. Usa 'y' o 'n'."
    exit 1
}