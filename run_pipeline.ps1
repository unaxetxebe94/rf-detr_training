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

Import-Module powershell-yaml
$config = Get-Content params.yaml -Raw | ConvertFrom-Yaml
$dataSrc2 = $config["data-src2"]
$finalData = $config["final-data"]

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
    param(
        [string]$StageName,
        [string]$Script,
        [string[]]$Arguments = @()
    )

    Write-Step "Etapa: $StageName"

    $cmdArgs = @(
        "run",
        "--no-capture-output",
        "-n",
        $CondaEnv,
        "python",
        "-u",
        $Script
    ) + $Arguments

    conda @cmdArgs

    if ($LASTEXITCODE -ne 0) {
        Write-Err "La etapa '$StageName' ha fallado (exit code $LASTEXITCODE)."
        exit $LASTEXITCODE
    }

    Write-Ok "$StageName completado."
}

function Invoke-ParallelPreprocess {
    Write-Step "Preprocesado (paralelo)"

    # Preprocesado en el master
    $p1 = Start-Process `
        -FilePath "conda" `
        -ArgumentList @(
            "run",
            "--no-capture-output",
            "-n",
            $CondaEnv,
            "python",
            "-u",
            "src/preprocess_pipeline.py",
            "--is-master"
        ) `
        -PassThru `
        -NoNewWindow

    # Preprocesado en el esclavo
    $p2 = Start-Process `
        -FilePath "conda" `
        -ArgumentList @(
            "run",
            "--no-capture-output",
            "-n",
            $CondaEnv,
            "python",
            "-u",
            "src/preprocess_pipeline.py"
        ) `
        -PassThru `
        -NoNewWindow

    Wait-Process -Id $p1.Id, $p2.Id

    $p1.Refresh()
    $p2.Refresh()

    if ($p1.ExitCode -ne 0) {
        Write-Err "Preprocesado de data-src1 falló (exit code $($p1.ExitCode))."
        exit $p1.ExitCode
    }

    if ($p2.ExitCode -ne 0) {
        Write-Err "Preprocesado data-src2 falló (exit code $($p2.ExitCode))."
        exit $p2.ExitCode
    }

    Write-Ok "Preprocesados completados."
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

$response = Read-Host "¿Los datos ya están preparados? (y/[n])"
$useSlave = "n"
if ($response -eq "y") {
    $isDataPrepared = $true
    Write-Host "No se ejecutará ni el preprocesado ni la fusión" -ForegroundColor Yellow
}
elseif ($response -eq "n" -or $response -eq "") {
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

    if ($useSlave -eq "y") {
        # 0. Preprocesado en paralelo 
        Invoke-Stage "Fusión de datasets" "src/preprocess_pipeline.py" @("--is-master")
        # 1. Mover los archivos de src2 a dst2
        $slavePath = Join-Path $finalData "slave"
        New-Item -ItemType Directory -Force -Path $slavePath | Out-Null
        Move-Item -Path $dataSrc2\* -Destination $slavePath -Force
        # 2. Fusionar los datasets del master y slave
        Invoke-Stage "Fusión de datasets" "src/fuse_datasets.py"
    } else {
        Invoke-Stage "Fusión de datasets" "src/preprocess_pipeline.py" @("--is-master")

        $masterPath = Join-Path $finalData "master"

        # Mover todo el contenido de master a final-data
        Get-ChildItem -Path $masterPath -Force | ForEach-Object {
            Move-Item -Path $_.FullName -Destination $finalData -Force
        }

        # Eliminar la carpeta master vacía
        Remove-Item -Path $masterPath -Recurse -Force
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