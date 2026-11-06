#!/bin/bash
#
# Port en bash de run_pipeline.ps1: ejecuta el pipeline completo de
# entrenamiento RF-DETR con versionado de datos en git.
#
# Las dependencias y la versión de Python se fijan con uv (pyproject.toml +
# uv.lock + .python-version). Antes de la primera ejecución, y siempre que
# cambien las dependencias, hay que sincronizar el entorno con: uv sync
# Todas las etapas se ejecutan con `uv run` para garantizar que se usa
# exactamente el entorno bloqueado en uv.lock.
#
set -u

# ─────────────────────────────────────────────
# Leer params.yaml con Python (como Get-Content params.yaml | ConvertFrom-Yaml)
# ─────────────────────────────────────────────

read_param_yaml () {
  uv run python - <<EOF
import yaml
with open("params.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg.get("data-src2", ""))
print(cfg.get("final-data", ""))
EOF
}

readarray -t config < <(read_param_yaml)

DATA_SRC2=${config[0]:-}
FINAL_DATA=${config[1]:-}

if [ -z "$DATA_SRC2" ] || [ -z "$FINAL_DATA" ]; then
  write_err "No se han podido leer 'data-src2' y/o 'final-data' desde params.yaml."
  exit 1
fi

START_TIME=$(date +%s)

# ─────────────────────────────────────────────
# Helpers (Write-Step / Write-Ok / Write-Warn / Write-Err / Invoke-Stage)
# ─────────────────────────────────────────────

CYAN=$'\e[36m'
GREEN=$'\e[32m'
YELLOW=$'\e[33m'
RED=$'\e[31m'
RESET=$'\e[0m'

write_step () {
  printf '\n[%s] ======================================\n' "$(date +%H:%M:%S)"
  printf '[%s] %s %s\n' "$(date +%H:%M:%S)" "${CYAN}$1${RESET}" ""
  printf '[%s] ======================================\n' "$(date +%H:%M:%S)"
}

write_ok   () { printf '  %s %s\n' "${GREEN}OK${RESET}" "$1"; }
write_warn () { printf '  %s %s\n' "${YELLOW}WARNING${RESET}" "$1"; }
write_err  () { printf '  %s %s\n' "${RED}ERROR${RESET}" "$1"; }

# invoke_stage "Nombre etapa" "ruta/script.py" [argumentos...]
invoke_stage () {
  local stage_name=$1
  local script=$2
  shift 2

  write_step "Etapa: $stage_name"

  uv run --no-capture-output python -u "$script" "$@"
  local exit_code=$?

  if [ "$exit_code" -ne 0 ]; then
    write_err "La etapa '$stage_name' ha fallado (exit code $exit_code)."
    exit "$exit_code"
  fi

  write_ok "$stage_name completado."
}

# Invoke-ParallelPreprocess: master con uv, esclavo con python del sistema
invoke_parallel_preprocess () {
  write_step "Preprocesado (paralelo)"

  uv run --no-capture-output python -u src/preprocess/__main__.py --is-master &
  local p1=$!

  python -u src/preprocess/__main__.py &
  local p2=$!

  echo "master (uv) pid: $p1"
  echo "slave (python) pid: $p2"

  local e1=0 e2=0
  wait "$p1" || e1=$?
  wait "$p2" || e2=$?

  echo "Master ExitCode = $e1"
  echo "Slave ExitCode = $e2"

  if [ "$e1" -ne 0 ]; then
    write_err "Preprocesado de data-src1 falló (exit code $e1)."
    exit "$e1"
  fi

  if [ "$e2" -ne 0 ]; then
    write_err "Preprocesado data-src2 falló (exit code $e2)."
    exit "$e2"
  fi

  write_ok "Preprocesados completados."
}

# ─────────────────────────────────────────────
# Preguntar si se quiere guardar el dataset en git
# ─────────────────────────────────────────────

write_step "Versionado del dataset"

read -r -p "¿Quieres guardar la base de datos en git para reproducir el experimento (COMMIT)? ([y]/n): " commit
commit=${commit:-y}

push="n"
commit_message=""
case "$commit" in
  y|Y)
    read -r -p "Escribe el mensaje del commit: " commit_message
    read -r -p "¿Quieres hacer PUSH de los cambios? ([y]/n): " push
    ;;
  n|N)
    write_warn "Se ejecutará el pipeline sin versionar el dataset."
    ;;
  *)
    write_err "Respuesta inválida. Usa 'y' o 'n'."
    exit 1
    ;;
esac

# ─────────────────────────────────────────────
# Preguntar si se tienen que preparar los datos
# ─────────────────────────────────────────────

write_step "Preparación de datos"

read -r -p "¿Los datos ya están preparados? (y/[n]): " response

is_data_prepared=false
use_slave="n"
case "$response" in
  y|Y)
    is_data_prepared=true
    write_warn "No se ejecutará ni el preprocesado ni la fusión"
    ;;
  n|N|'')
    write_warn "Se ejecutará el preprocesado"

    read -r -p "¿Quieres usar las imágenes del Slave? ([y]/n): " use_slave
    use_slave=${use_slave:-y}
    case "$use_slave" in
      y|Y)
        write_warn "Se funsionarán los datasets"
        ;;
      n|N)
        ;;
      *)
        write_err "Respuesta inválida. Usa 'y' o 'n'."
        exit 1
        ;;
    esac
    ;;
  *)
    write_err "Respuesta inválida. Usa 'y' o 'n'."
    exit 1
    ;;
esac

# ─────────────────────────────────────────────
# Etapas del pipeline
# ─────────────────────────────────────────────

if ! "$is_data_prepared"; then

  if [ "$use_slave" = "y" ] || [ "$use_slave" = "Y" ]; then

    if [ -e "$FINAL_DATA" ]; then
      rm -rf -- "$FINAL_DATA"
    fi

    # 0. Preprocesado en paralelo (master y slave)
    invoke_parallel_preprocess

    # 1. Mover los archivos de src2 a final-data/slave
    slave_path="$FINAL_DATA/slave"
    mkdir -p -- "$slave_path"
    find "$DATA_SRC2/formatted" -mindepth 1 -maxdepth 1 -print0 2>/dev/null |
      xargs -0 -r mv -f -t "$slave_path"

    # 2. Fusionar los datasets del master y slave
    invoke_stage "Fusión de datasets" "src/core/fuse_datasets.py"
  else
    invoke_stage "Fusión de datasets" "src/preprocess/__main__.py" --is-master

    master_path="$FINAL_DATA/master"

    # Mover todo el contenido de master a final-data
    find "$master_path" -mindepth 1 -maxdepth 1 -print0 2>/dev/null |
      xargs -0 -r mv -f -t "$FINAL_DATA"

    # Eliminar la carpeta master vacía
    rm -rf -- "$master_path"
  fi
fi

# 2. Entrenamiento
invoke_stage "Entrenamiento" "src/core/train.py"

# 3. Procesado de resultados + logging a W&B
invoke_stage "Procesado de resultados" "src/core/process_results.py"

# 4. Test
invoke_stage "Test" "src/test/__main__.py"


# ─────────────────────────────────────────────
# Resumen final
# ─────────────────────────────────────────────

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
write_step "Pipeline completado"
write_ok "Tiempo total: $(date -u -d "@$ELAPSED" +%H:%M:%S)"
echo ""

timestamp=$(date +%Y-%m-%d_%H-%M-%S)
cp trainings/training/checkpoint_best_total.pth "$timestamp.pth" \
  || write_warn "No se pudo copiar trainings/training/checkpoint_best_total.pth"


# ─────────────────────────────────────────────
# Lo de git
# ─────────────────────────────────────────────

if [ "$commit" = "y" ] || [ "$commit" = "Y" ]; then
  write_warn "Guardando estado en git..."

  git add .

  if git diff --cached --quiet; then
    write_warn "No hay cambios para commitear."
  else
    if git commit -m "Snapshot dataset antes de ejecutar experimento $(date '+%Y-%m-%d %H:%M:%S'): $commit_message"; then
      write_ok "Snapshot guardado en git."
    else
      write_warn "No se pudo hacer commit (quizá no hay cambios)."
    fi
  fi

  # PUSH
  case "$push" in
    y|Y|'')
      write_warn "Subiendo cambios al remoto..."
      if git push; then
        write_ok "Snapshot subido al remoto (push)."
      else
        write_warn "No se pudo hacer push"
      fi
      ;;
    n|N)
      ;;
    *)
      write_err "Respuesta inválida. Usa 'y' o 'n'."
      exit 1
      ;;
  esac
fi
