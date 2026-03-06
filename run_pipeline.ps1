<#
.SYNOPSIS
    Ejecuta el pipeline completo de entrenamiento RF-DETR con versionado de datos via DVC.

.PARAMETER CondaEnv
    Nombre del entorno conda a usar. Por defecto: "rf-detr".

.EXAMPLE
    .\run_pipeline.ps1
    .\run_pipeline.ps1 -SkipPreprocess -SkipDVC
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
    git commit -m "Snapshot dataset antes de ejecutar experimento $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

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
# Etapas del pipeline
# ─────────────────────────────────────────────────────────────────────────────

# 0. Preprocesado
Invoke-Stage "Preprocesado" "src/preprocess_pipeline.py"

# 1. Fusionar los datasets del master y slave
# Invoke-Stage "Fusión de datasets" "src/fuse_datasets.py"

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
Write-Ok "Experimento: $TaskName"
Write-Host ""