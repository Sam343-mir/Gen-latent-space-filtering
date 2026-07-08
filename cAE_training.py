import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
 
from convautoencoder_leakyrelu import ConvAutoencoder
from generating_slices import scalenorm, both_inline_crossline
 
 
def load_arr0(npz_path: str) -> np.ndarray:
    return np.load(npz_path)["arr_0"]
 
 
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
 
 
def plot_tsne(df_tsne: pd.DataFrame, out_png: str):
    plt.figure(figsize=(10, 8))
    for lab in sorted(df_tsne["label"].unique()):
        sub = df_tsne[df_tsne["label"] == lab]
        plt.scatter(sub["x"], sub["y"], s=10, alpha=0.8)
    plt.title("Latent Space (t-SNE)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
 
 
def main():
    ap = argparse.ArgumentParser()
 
    # repeatable: --dataset NAME SEISMIC_NPZ LABEL_NPZ FLAG
    ap.add_argument(
        "--dataset",
        action="append",
        nargs=4,
        metavar=("NAME", "SEISMIC_NPZ", "LABEL_NPZ", "FLAG"),
        required=True,
        help="Add dataset block: NAME SEISMIC_NPZ LABEL_NPZ FLAG",
    )
 
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--img_size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--tsne_seed", type=int, default=2)
 
    ap.add_argument("--checkpoint", default="all_512_512.h5")
    ap.add_argument("--force_retrain", action="store_true")
 
    args = ap.parse_args()
    ensure_dir(args.outdir)
 
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(args.outdir, ckpt_path)
 
    # -------------------------
    # 1) Load + slice ALL datasets
    # -------------------------
    all_imgs = []
    meta_rows = []
 
    for (name, seismic_npz, label_npz, flag_str) in args.dataset:
        flag = int(flag_str)
 
        print(f"[INFO] Loading '{name}'")
        s = load_arr0(seismic_npz)
        gt = load_arr0(label_npz)
 
        s = scalenorm(s)
        imgs, _, labels, in_or_cross, slice_index = both_inline_crossline(s, gt, flag)
 
        all_imgs.append(imgs)
 
        for i in range(imgs.shape[0]):
            meta_rows.append(
                {
                    "dataset_name": name,
                    "train_labels": int(labels[i]),          # <-- selection script expects train_labels
                    "slice_indexes": int(slice_index[i]),    # <-- expects slice_indexes
                    "in_or_cross_imgs": int(in_or_cross[i])  # <-- expects in_or_cross_imgs
                }
            )
 
    X = np.concatenate(all_imgs, axis=0).astype("float32")
    meta = pd.DataFrame(meta_rows)
 
    # -------------------------
    # 2) Train or load AE
    # -------------------------
    h = w = args.img_size
    d = 1
    _, _, autoencoder = ConvAutoencoder.build(h, w, d)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.lr),
        loss="mean_squared_error",
    )
 
    if (not args.force_retrain) and os.path.exists(ckpt_path):
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        autoencoder = keras.models.load_model(ckpt_path)
    else:
        print("[INFO] Training AE...")
        train_X, valid_X = train_test_split(X, test_size=args.val_split, random_state=13)
 
        cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        )
 
        autoencoder.fit(
            train_X, train_X,
            epochs=args.epochs,
            batch_size=args.batch,
            shuffle=True,
            validation_data=(valid_X, valid_X),
            callbacks=[cb],
        )
        autoencoder = keras.models.load_model(ckpt_path)
 
    # -------------------------
    # 3) Latent extraction
    # -------------------------
    encoder = autoencoder.get_layer("Encoder")
    latent = encoder.predict(X, batch_size=args.batch, verbose=1)
 
    # Save latent as NPY (optional but useful)
    np.save(os.path.join(args.outdir, "latent_space.npy"), latent)
 
    # Save latent CSV (THIS is what selection consumes)
    latent_cols = [f"z_{i}" for i in range(latent.shape[1])]
    df_latent = pd.concat(
        [meta.reset_index(drop=True), pd.DataFrame(latent, columns=latent_cols)],
        axis=1
    )
    latent_csv_path = os.path.join(args.outdir, "cAE_latent_space.csv")
    df_latent.to_csv(latent_csv_path, index=False)
 
    # -------------------------
    # 4) t-SNE (for visualization)
    # -------------------------
    tsne = TSNE(n_components=2, random_state=args.tsne_seed)
    z2 = tsne.fit_transform(latent)
 
    df_tsne = pd.DataFrame({"x": z2[:, 0], "y": z2[:, 1], "label": df_latent["train_labels"].values})
    df_tsne = pd.concat([df_tsne, meta.reset_index(drop=True)], axis=1)
 
    df_tsne.to_csv(os.path.join(args.outdir, "latent_tsne.csv"), index=False)
    plot_tsne(df_tsne, os.path.join(args.outdir, "tsne_space.png"))
 
    print("\n[OK] Phase-1 outputs saved in:", args.outdir)
    print(" -", os.path.basename(ckpt_path))
    print(" - cAE_latent_space.csv")
    print(" - latent_space.npy")
    print(" - latent_tsne.csv")
    print(" - tsne_space.png")
 
 
if __name__ == "__main__":
    main()