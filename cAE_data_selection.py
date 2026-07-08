import os
import argparse
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

def parse_args():
   ap = argparse.ArgumentParser(description="Latent-space SVM selection/removal script")
   ap.add_argument("--latent_csv", required=True,
                   help="CSV containing latent vectors + metadata (e.g., cAE_latent_space.csv)")
   ap.add_argument("--outdir", required=True, help="Output directory")
   ap.add_argument("--out_csv", default="selection_output.csv",
                   help="Single output CSV filename (inside outdir)")
   # SVM params
   ap.add_argument("--kernel", default="sigmoid",
                   choices=["sigmoid", "rbf", "linear", "poly"],
                   help="SVM kernel")
   ap.add_argument("--C", type=float, default=0.01,
                   help="SVM regularization parameter C")
   # Column names (so we don’t depend on one exact CSV schema forever)
   ap.add_argument("--label_col", default="train_labels",
                   help="Column containing original dataset labels (e.g., train_labels)")
   ap.add_argument("--slice_col", default="slice_indexes",
                   help="Column containing slice index (e.g., slice_indexes)")
   ap.add_argument("--in_cross_col", default="in_or_cross_imgs",
                   help="Column containing inline/crossline indicator (e.g., in_or_cross_imgs)")
   # How to define “good space”: map label 2 -> 1, so {1,2} are good; 0 is stones bad
   ap.add_argument("--bad_label_value", type=int, default=0,
                   help="Value representing the bad dataset label (stones). Default 0")
   ap.add_argument("--merge_label_value", type=int, default=2,
                   help="Label value to merge into good class (e.g., 2 -> 1). Default 2")
   ap.add_argument("--good_label_value", type=int, default=1,
                   help="The good class value after merging. Default 1")
   ap.add_argument("--save_report", action="store_true",
                   help="Save a text report with accuracy + classification report")
   return ap.parse_args()

def main():
   args = parse_args()
   os.makedirs(args.outdir, exist_ok=True)
   df = pd.read_csv(args.latent_csv)
   # --- Build binary class label: good vs bad ---
   if args.label_col not in df.columns:
       raise ValueError(f"Missing label column '{args.label_col}' in {args.latent_csv}")
   y_raw = df[args.label_col].copy()
   # Merge label 2 -> 1 (as in your original script)
   y_bin = y_raw.copy()
   y_bin[y_bin == args.merge_label_value] = args.good_label_value
   y_bin = y_bin.astype(int)
   # --- Latent feature columns ---
   # - drop obvious meta columns if present
   meta_cols = {args.label_col, "label", args.slice_col, args.in_cross_col}
   candidate_cols = [c for c in df.columns if c not in meta_cols]
   # If user’s CSV includes extra string columns, keep only numeric
   X_df = df[candidate_cols].select_dtypes(include=[np.number])
   if X_df.shape[1] == 0:
       raise ValueError("No numeric latent columns found. Check your latent CSV format.")
   X = X_df.to_numpy(dtype=np.float32)
   # --- Scale + fit SVM on ALL points---
   scaler = StandardScaler()
   Xs = scaler.fit_transform(X)
   svm_model = SVC(kernel=args.kernel, C=args.C)
   svm_model.fit(Xs, y_bin)
   y_pred = svm_model.predict(Xs)
   acc = accuracy_score(y_bin, y_pred)
   rep = classification_report(y_bin, y_pred, digits=4)
   # --- Two-way removal logic ---
   # bad dataset points that fell into good space  (bad -> predicted good)
   bad_to_good = (y_bin == args.bad_label_value) & (y_pred == args.good_label_value)
   # good dataset points that fell into bad space  (good -> predicted bad)
   good_to_bad = (y_bin == args.good_label_value) & (y_pred == args.bad_label_value)
   remove_mask = bad_to_good | good_to_bad
   # Build output table (single file)
   out = pd.DataFrame({
       "orig_label": y_raw,
       "binary_label": y_bin,
       "pred_label": y_pred,
       "remove": remove_mask.astype(int),
       "remove_reason": np.where(
           bad_to_good, "bad_points_in_good_space",
           np.where(good_to_bad, "good_points_in_bad_space", "keep")
       )
   })
   # Attach metadata if available
   if args.slice_col in df.columns:
       out["slice_index"] = df[args.slice_col].values
   if args.in_cross_col in df.columns:
       out["in_or_cross"] = df[args.in_cross_col].values
   # Save
   out_path = os.path.join(args.outdir, args.out_csv)
   out.to_csv(out_path, index=False)
   # Optional report
   if args.save_report:
       report_path = os.path.join(args.outdir, "svm_selection_report.txt")
       with open(report_path, "w", encoding="utf-8") as f:
           f.write(f"Kernel: {args.kernel}\n")
           f.write(f"C: {args.C}\n")
           f.write(f"Accuracy (train-fit on all points): {acc:.6f}\n\n")
           f.write(rep)
           f.write("\n\nCounts:\n")
           f.write(f"bad_to_good: {int(bad_to_good.sum())}\n")
           f.write(f"good_to_bad: {int(good_to_bad.sum())}\n")
           f.write(f"remove_total: {int(remove_mask.sum())}\n")
   print("[OK] Saved:", out_path)
   print(f"[INFO] remove_total={int(remove_mask.sum())} | bad_to_good={int(bad_to_good.sum())} | good_to_bad={int(good_to_bad.sum())}")
   print(f"[INFO] Accuracy={acc:.6f}")

if __name__ == "__main__":
   main()