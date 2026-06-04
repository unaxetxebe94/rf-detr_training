#!/bin/bash
set -e

CONFIG_FILE=${1:-pipeline_config.yaml}

# ─────────────────────────────────────────────
# Leer YAML con Python
# ─────────────────────────────────────────────

read_yaml () {
  python - <<EOF
import yaml
with open("$CONFIG_FILE") as f:
    cfg = yaml.safe_load(f)

print(cfg["conda_env"])
print(cfg["git"]["commit"])
print(cfg["git"]["push"])
print(cfg["git"]["message"])
print(cfg["data"]["prepared"])
print(cfg["data"]["use_slave"])
EOF
}

# Cargar variables
readarray -t cfg < <(read_yaml)

CONDA_ENV=${cfg[0]}
COMMIT=${cfg[1]}
PUSH=${cfg[2]}
COMMIT_MESSAGE=${cfg[3]}
PREPARED=${cfg[4]}
USE_SLAVE=${cfg[5]}

START_TIME=$(date +%s)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

write_step () {
  echo ""
  echo "======================================"
  echo " $1"
  echo "======================================"
}

invoke_stage () {
  local name=$1
  local script=$2

  write_step "$name"
  conda run -n "$CONDA_ENV" python "$script"

  if [ $? -ne 0 ]; then
    echo "ERROR en $name"
    exit 1
  fi
}

# ─────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────

write_step "Pipeline iniciado"

if [ "$PREPARED" != "True" ] && [ "$PREPARED" != "true" ]; then

  invoke_stage "Preprocesado" "src/preprocess_pipeline.py"

  if [ "$USE_SLAVE" == "True" ] || [ "$USE_SLAVE" == "true" ]; then
    invoke_stage "Fusión datasets" "src/fuse_datasets.py"
  else
    rm -rf data/formatted 2>/dev/null || true
    mv data/preprocessed_src1 data/formatted
  fi
fi

invoke_stage "Entrenamiento" "src/train.py"
invoke_stage "Procesado resultados" "src/process_results.py"
invoke_stage "Test" "src/test.py"

# ─────────────────────────────────────────────
# Post-procesado
# ─────────────────────────────────────────────

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
cp trainings/training/checkpoint_best_total.pth "$timestamp.pth"

# ─────────────────────────────────────────────
# Git
# ─────────────────────────────────────────────

if [ "$COMMIT" == "True" ] || [ "$COMMIT" == "true" ]; then

  mkdir -p annotations

  cp data/formatted/train/_annotations.coco.json annotations/train_annotations.json
  cp data/formatted/test/_annotations.coco.json annotations/test_annotations.json
  cp data/formatted/valid/_annotations.coco.json annotations/valid_annotations.json

  git add .

  if ! git diff --cached --quiet; then
    git commit -m "Snapshot $(date): $COMMIT_MESSAGE"
  fi

  if [ "$PUSH" == "True" ] || [ "$PUSH" == "true" ]; then
    git push || echo "Push falló"
  fi
fi

# ─────────────────────────────────────────────
# Tiempo total
# ─────────────────────────────────────────────

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

write_step "Pipeline completado"
echo "Tiempo total: $(date -u -d @$ELAPSED +%H:%M:%S)"