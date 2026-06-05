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

print(cfg["git"]["commit"])
print(cfg["git"]["push"])
print(cfg["git"]["message"])
print(cfg["data"]["prepared"])
print(cfg["data"]["data-src1"])
print(cfg["data"]["data-src2"])
print(cfg["data"]["final-dir"])
EOF
}

# Cargar variables
readarray -t cfg < <(read_yaml)

COMMIT=${cfg[0]}
PUSH=${cfg[1]}
COMMIT_MESSAGE=${cfg[2]}
PREPARED=${cfg[3]}
DATA_SRC1=${cfg[4]}
DATA_SRC2=${cfg[5]}
FINAL_DIR=${cfg[6]}

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
  python "$script"

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

  if [ "$USE_SLAVE" == "True" ] || [ "$USE_SLAVE" == "true" ]; then
    invoke_stage "Fusión datasets" "src/fuse_datasets.py"
  else
    rm -r $FINAL_DIR || True
    mv $DATA_SRC1 $FINAL_DIR
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

  cp $FINAL_DIR/train/_annotations.coco.json annotations/train_annotations.json
  cp $FINAL_DIR/test/_annotations.coco.json annotations/test_annotations.json
  cp $FINAL_DIR/valid/_annotations.coco.json annotations/valid_annotations.json

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