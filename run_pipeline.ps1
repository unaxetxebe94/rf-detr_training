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
$commit = Read-Host "¿Quieres guardar la base de datos en git para reproducir el experimento (COMMIT)? ([y]/n)"
$push = "n"
$commitMessage = ""
if ($commit -eq "y" -or $commit -eq "") {
    # Mensaje del commit
    $commitMessage = Read-Host "Escribe el mensaje del commit"
    # PUSH
    $push = Read-Host "¿Quieres hacer PUSH de los cambios? ([y]/n)"
}
elseif ($commit -eq "n") {
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
$useSlave = "n"
if ($response -eq "y" -or $response -eq "") {
    $isDataPrepared = $true
    Write-Host "No se ejecutará ni el preprocesado ni la fusión" -ForegroundColor Yellow
}
elseif ($response -eq "n") {
    Write-Warn "Se ejecutará el preprocesado"
    $isDataPrepared = $false

    $useSlave = Read-Host "¿Quieres usar las imágenes del Slave? ([y]/n)"
    if ($useSlave -eq "y" -or $useSlave -eq "") {
        Write-Warn "Se funsionarán los datasets"
    } elseif ($useSlave -ne "n") {
        Write-Err "Respuesta inválida. Usa 'y' o 'n'."
        exit 1
    }
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

    if ($useSlave) {
        # 1. Fusionar los datasets del master y slave
        Invoke-Stage "Fusión de datasets" "src/fuse_datasets.py"
    } else {
        $basePath = "E:\rf-detr_training\data"
        $formattedPath = Join-Path $basePath "formatted"
        $sourcePath = Join-Path $basePath "preprocessed_src1"

        # Si existe 'formatted', eliminarlo primero
        if (Test-Path $formattedPath) {
            Remove-Item $formattedPath -Recurse -Force
        }

        # Renombrar carpeta
        Rename-Item $sourcePath "formatted"
    }
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


# Lo de git
if ($commit -eq "y" -or $commit -eq "") {
    Write-Host "Guardando estado en git..." -ForegroundColor Yellow

    mkdir annotations -Force
    Copy-Item "data\formatted\train\_annotations.coco.json" "annotations\train_annotations.json" -Force
    Copy-Item "data\formatted\test\_annotations.coco.json" "annotations\test_annotations.json" -Force
    Copy-Item "data\formatted\valid\_annotations.coco.json" "annotations\valid_annotations.json" -Force

    git add .
    git diff --cached --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Warn "No hay cambios para commitear."
    } else {
        git commit -m "Snapshot dataset antes de ejecutar experimento $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'): $($commitMessage)"
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Warn "No se pudo hacer commit (quizá no hay cambios)."
    }
    else {
        Write-Ok "Snapshot guardado en git."
    }

    # PUSH
    if ($push -eq "y" -or $push -eq "") {
        Write-Host "Subiendo cambios al remoto..." -ForegroundColor Yellow

        git push

        if ($LASTEXITCODE -ne 0) {
            Write-Warn "No se pudo hacer push"
        }
        else {
            Write-Ok "Snapshot guardado en git."
        }
    }
    elseif ($push -eq "n") {
        Write-Warn "Se ejecutará el pipeline sin versionar el dataset."
    }
    else {
        Write-Err "Respuesta inválida. Usa 'y' o 'n'."
        exit 1
    }
}