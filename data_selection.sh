# Usage:
#   ./run_selection.sh OUTDIR LATENT_CSV [KERNEL] [C]

# Example:
#   ./run_selection.sh outputs/sel cAE_latent_space.csv sigmoid 0.01

OUTDIR="path/where/to/save/outputs"
LATENT_CSV="path/to/csv/generated/in/phase1"
KERNEL="sigmoid" #kernel choice for SVM
C="0.01" #regularization parameter value
BAD_LABEL_VALUE='0'

mkdir -p "$OUTDIR"

python cAE_data_selection.py \
  --latent_csv "$LATENT_CSV" \
  --outdir "$OUTDIR" \
  --kernel "$KERNEL" \
  --C "$C" \
  --out_csv "selection_output.csv" \
  --bad_label_value "$BAD_LABEL_VALUE"\
  --save_report
 