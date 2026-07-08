# Usage:
#   ./run_latent.sh OUTDIR [SELECT_FROM]

# Example:
#   ./run_latent.sh outputs/run1 stones

OUTDIR="${1:?Need OUTDIR}"
SELECT_FROM="${2:-}"
mkdir -p "$OUTDIR"

DATASETS=(
  "monkeyhill" "/ABS/PATH/monkeyhill_seis.npz" "/ABS/PATH/monkeyhill_label.npz" "1"
  "stones"     "/ABS/PATH/stones_seis.npz"     "/ABS/PATH/stones_label.npz"     "0"
  "nakika"     "/ABS/PATH/nakika_seis.npz"     "/ABS/PATH/nakika_label.npz"     "2"

)

# Hardcode model/training params here (optional)

IMG_SIZE=512
EPOCHS=20
BATCH=16
LR=0.001
VAL_SPLIT=0.15
CHECKPOINT="all_512_512.h5"

# -----------------------------
# BUILD COMMAND
# -----------------------------

# Add datasets (consume DATASETS array in groups of 4)
i=0
while (( i < ${#DATASETS[@]} )); do
  NAME="${DATASETS[i]}"; SEIS="${DATASETS[i+1]}"; LAB="${DATASETS[i+2]}"; FLAG="${DATASETS[i+3]}"
  CMD+=(--dataset "$NAME" "$SEIS" "$LAB" "$FLAG")
  i=$((i+4))
done

# Optionally select a dataset
if [[ -n "$SELECT_FROM" ]]; then
  CMD+=(--select_from "$SELECT_FROM")
fi

python cAE_training.py
  --outdir "$OUTDIR"
  --img_size "$IMG_SIZE"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --lr "$LR"
  --val_split "$VAL_SPLIT"
  --checkpoint "$CHECKPOINT"
 