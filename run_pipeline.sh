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
# Uso: ./run_pipeline.sh
#
set -u

START_TIME=$(date +%s)

# ─────────────────────────────────────────────
# Leer params.yaml con Python (como Get-Content params.yaml | ConvertFrom-Yaml)
# ─────────────────────────────────────────────

read_config_yaml () {
  uv run python - <<EOF
import yaml
with open("params.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print(cfg["data"]["training-dataset-path"])
EOF
}

FINAL_DATA=$(read_config_yaml)

if [ -z "$FINAL_DATA" ]; then
  echo "No se ha podido leer 'data.training-dataset-path' desde params.yaml."
  exit 1
fi

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
  printf '[%s]  %s\n' "$(date +%H:%M:%S)" "${CYAN}$1${RESET}"
  printf '[%s] ======================================\n' "$(date +%H:%M:%S)"
}

write_ok   () { printf '  %s %s\n' "${GREEN}OK${RESET}" "$1"; }
write_warn () { printf '  %s %s\n' "${YELLOW}WARNING${RESET}" "$1"; }
write_err  () { printf '  %s %s\n' "${RED}ERROR${RESET}" "$1"; }

# invoke_stage "Nombre etapa" "ruta/script.py" [argumentos...]
# Equivalente a Invoke-Stage del .ps1.
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

# ─────────────────────────────────────────────
# Preguntar si se quiere guardar el dataset en git
# ─────────────────────────────────────────────

write_step "Versionado del dataset"

# COMMIT
read -r -p "¿Quieres guardar la base de datos en git para reproducir el experimento (COMMIT)? ([y]/n): " commit
commit=${commit:-y}

push="n"
commit_message=""
case "$commit" in
  y|Y)  # el .ps1 también acepta la respuesta vacía como "y"
    # Mensaje del commit
    read -r -p "Escribe el mensaje del commit: " commit_message
    # PUSH
    read -r -p "¿Quieres hacer PUSH de los cambios? ([y]/n): " push
    push=${push:-y}
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
case "$response" in
  y|Y)
    is_data_prepared=true
    write_warn "No se ejecutará ni el preprocesado ni la fusión"
    ;;
  n|N|'')  # el .ps1 también acepta la respuesta vacía como "n"
    write_warn "Se ejecutará el preprocesado"
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

  # El preprocesado escribe el dataset resultante en FINAL_DATA.
  if [ -e "$FINAL_DATA" ]; then
    rm -rf -- "$FINAL_DATA"
  fi

  # 0. Preprocesado de todas las fuentes declaradas en data.sources
  invoke_stage "Preprocesado" "src/preprocess/__main__.py"

  # 1. Fusionar los datasets preprocesados de todas las fuentes
  invoke_stage "Fusión de datasets" "src/core/fuse_datasets.py"
fi

# 2. Entrenamiento
invoke_stage "Entrenamiento" "src/core/train.py"

# 3. Test
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
cp trainings/training/checkpoint_best_total.pth "$timestamp.pth"

# ─────────────────────────────────────────────
# Git: snapshot del dataset antes del experimento
# ─────────────────────────────────────────────

if [ "$commit" = "y" ]; then
  write_warn "Guardando estado en git..."

  git add .

  if git diff --cached --quiet; then
    write_warn "No hay cambios para commitear."
  else
    git commit -m "Snapshot dataset antes de ejecutar experimento $(date '+%Y-%m-%d %H:%M:%S'): $commit_message"
    commit_exit=$?
    if [ "$commit_exit" -ne 0 ]; then
      write_warn "No se pudo hacer commit (quizá no hay cambios)."
    else
      write_ok "Snapshot guardado en git."
    fi
  fi

  # PUSH (el .ps1 trata la respuesta vacía como "y")
  case "$push" in
    y|Y)
      write_warn "Subiendo cambios al remoto..."

      git push

      if [ "$?" -ne 0 ]; then
        write_warn "No se pudo hacer push"
      else
        write_ok "Snapshot guardado en git."
      fi
      ;;
    n|N)
      write_warn "Se ejecutará el pipeline sin versionar el dataset."
      ;;
    *)
      write_err "Respuesta inválida. Usa 'y' o 'n'."
      exit 1
      ;;
  esac
fi
