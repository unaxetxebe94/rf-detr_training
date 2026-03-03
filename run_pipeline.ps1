<#
.SYNOPSIS
    Ejecuta el pipeline completo de entrenamiento RF-DETR con versionado de datos via DVC.

.PARAMETER SkipPreprocess
    Salta la etapa de preprocesado (usa el dataset ya existente).

.PARAMETER SkipDVC
    Salta el versionado de datos con DVC (útil para iteraciones rápidas sin cambios en los datos).

.PARAMETER SkipTest
    Salta la etapa de test final.

.PARAMETER CondaEnv
    Nombre del entorno conda a usar. Por defecto: "rf-detr".

.EXAMPLE
    .\run_pipeline.ps1
    .\run_pipeline.ps1 -SkipPreprocess -SkipDVC
    .\run_pipeline.ps1 -CondaEnv "mi-entorno"
#>

param(
    [switch]$SkipPreprocess,
    [switch]$SkipDVC,
    [switch]$SkipTest,
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
    Write-Host "[$ts] ══════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "[$ts]  $Msg" -ForegroundColor Cyan
    Write-Host "[$ts] ══════════════════════════════════════" -ForegroundColor Cyan
}

function Write-Ok   { param([string]$M); Write-Host "  ✔  $M" -ForegroundColor Green  }
function Write-Warn { param([string]$M); Write-Host "  ⚠  $M" -ForegroundColor Yellow }
function Write-Err  { param([string]$M); Write-Host "  ✖  $M" -ForegroundColor Red    }

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
# Leer params.yaml con Python (evita dependencia de módulo YAML en PowerShell)
# ─────────────────────────────────────────────────────────────────────────────

Write-Step "Leyendo parámetros del pipeline"

$PyReadParams = @"
import yaml, json, sys
with open('params.yaml') as f:
    p = yaml.safe_load(f)
print(json.dumps({
    'data_src':           p['data-src'],
    'task_name':          p['task-name'],
    'requires_preprocess': p['preprocess']['requires-preprocess']
}))
"@

$ParamsJson = conda run -n $CondaEnv python -c $PyReadParams
if ($LASTEXITCODE -ne 0) {
    Write-Err "No se pudo leer params.yaml."
    exit 1
}

$Params           = $ParamsJson | ConvertFrom-Json
$DataSrc          = $Params.data_src
$TaskName         = $Params.task_name
$RequiresPreproc  = $Params.requires_preprocess

Write-Ok "task-name          : $TaskName"
Write-Ok "data-src           : $DataSrc"
Write-Ok "requires-preprocess: $RequiresPreproc"

# ─────────────────────────────────────────────────────────────────────────────
# DVC: versionar data-src antes del experimento
# ─────────────────────────────────────────────────────────────────────────────

if (-not $SkipDVC) {
    Write-Step "Versionando data-src con DVC"

    # Comprobar que el directorio data-src existe
    if (-not (Test-Path $DataSrc)) {
        Write-Err "El directorio data-src no existe: $DataSrc"
        exit 1
    }

    # dvc add trackea el directorio (crea/actualiza el .dvc)
    Write-Host "  → dvc add '$DataSrc'" -ForegroundColor DarkCyan
    dvc add $DataSrc
    if ($LASTEXITCODE -ne 0) { Write-Err "dvc add falló."; exit 1 }

    # dvc push sube los datos al remote local configurado
    Write-Host "  → dvc push" -ForegroundColor DarkCyan
    dvc push
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "dvc push falló o no hay remote configurado. Continuando sin push."
    }

    # Calcular el hash actual del .dvc para usarlo en el commit message
    $DvcFile = "$DataSrc.dvc"
    # Los archivos .dvc se generan relativo al repo, buscar el nombre de archivo
    $DvcFileName = (Split-Path $DataSrc -Leaf) + ".dvc"

    # Hacer git add de los ficheros DVC relevantes
    Write-Host "  → git add *.dvc .gitignore" -ForegroundColor DarkCyan
    git add "*.dvc" ".gitignore" 2>$null
    git add "$DvcFileName" 2>$null

    # Solo commitear si hay cambios staged
    $GitStatus = git status --porcelain
    if ($GitStatus) {
        $CommitMsg = "data: snapshot de '$TaskName' antes del experimento [$(Get-Date -Format 'yyyy-MM-dd HH:mm')]"
        Write-Host "  → git commit -m `"$CommitMsg`"" -ForegroundColor DarkCyan
        git commit -m $CommitMsg
        if ($LASTEXITCODE -ne 0) { Write-Warn "git commit falló. ¿Está configurado git?"; }
        else { Write-Ok "Snapshot de datos commiteado en git." }
    } else {
        Write-Warn "No hay cambios en los archivos DVC. El snapshot de datos no ha cambiado."
    }
} else {
    Write-Warn "Versionado DVC omitido (-SkipDVC)."
}

# ─────────────────────────────────────────────────────────────────────────────
# Etapas del pipeline
# ─────────────────────────────────────────────────────────────────────────────

# 1. Preprocesado (opcional)
if ($RequiresPreproc -and (-not $SkipPreprocess)) {
    Invoke-Stage "Preprocesado" "src/preprocess_pipeline.py"
} elseif ($SkipPreprocess) {
    Write-Warn "Preprocesado omitido (-SkipPreprocess)."
} else {
    Write-Warn "Preprocesado omitido (requires-preprocess: false en params.yaml)."
}

# 2. Entrenamiento
Invoke-Stage "Entrenamiento" "src/train.py"

# 3. Procesado de resultados + logging a W&B
Invoke-Stage "Procesado de resultados" "src/process_results.py"

# 4. Test
if (-not $SkipTest) {
    Invoke-Stage "Test" "src/test.py"
} else {
    Write-Warn "Test omitido (-SkipTest)."
}

# ─────────────────────────────────────────────────────────────────────────────
# Resumen final
# ─────────────────────────────────────────────────────────────────────────────

$Elapsed = (Get-Date) - $StartTime
Write-Step "Pipeline completado"
Write-Ok "Tiempo total: $($Elapsed.ToString('hh\:mm\:ss'))"
Write-Ok "Experimento: $TaskName"
Write-Host ""