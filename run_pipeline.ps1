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
# Etapas del pipeline
# ─────────────────────────────────────────────────────────────────────────────

# 1. Preprocesado
Invoke-Stage "Preprocesado" "src/preprocess_pipeline.py"

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