"""Build Galaxy_dataset_3class (elliptical / merger / spiral) from Galaxy Zoo 2 labels.
Run from the 'finetuning tmi-102 project' folder:  python build_dataset_3class.py
Old Galaxy_dataset is NOT touched."""
import os, shutil, zipfile
import numpy as np, pandas as pd

CSV  = "rawdata/training_solutions_rev1/training_solutions_rev1.csv"
IMGS = "rawdata/images_training_rev1/images_training_rev1"
OUT  = "Galaxy_dataset_3class"
SEED = 42

# ---- labelling rules -------------------------------------------------------
ELL_T, SPI_T = 0.80, 0.80      # same thresholds as Week 3 (Class1.1 smooth, Class1.2 features/disk)
MERGER_T     = 0.30            # Class8.6 = "merger" (already weighted by the 'odd feature' branch)
# ----------------------------------------------------------------------------

df = pd.read_csv(CSV)

is_merger = df["Class8.6"] >= MERGER_T
is_ell    = (df["Class1.1"] >= ELL_T) & ~is_merger      # merger takes priority -> no image in 2 classes
is_spi    = (df["Class1.2"] >= SPI_T) & ~is_merger

merger = df[is_merger].assign(cls="merger")
ell    = df[is_ell].assign(cls="elliptical")
spi_all = df[is_spi]
n_big  = min(len(ell), len(spi_all))                    # keep ellipticals and spirals equal, like Week 3
ell    = ell.sample(n=n_big, random_state=SEED)
spi    = spi_all.sample(n=n_big, random_state=SEED).assign(cls="spiral")
print("pools -> elliptical", len(ell), "| spiral", len(spi), "| merger", len(merger))

# ---- stratified 70/15/15 split, per class ----------------------------------
rng = np.random.default_rng(SEED)
parts = []
for c, g in pd.concat([ell, spi, merger]).groupby("cls"):
    g = g.iloc[rng.permutation(len(g))].copy()
    n = len(g); n_tr, n_va = int(0.70 * n), int(0.15 * n)
    g["split"] = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
    parts.append(g)
man = pd.concat(parts).reset_index(drop=True)

# ---- copy images -----------------------------------------------------------
for s in ["train", "val", "test"]:          # resumable: existing files are skipped, nothing is deleted
    for c in ["elliptical", "merger", "spiral"]:
        os.makedirs(f"{OUT}/{s}/{c}", exist_ok=True)
from concurrent.futures import ThreadPoolExecutor
def _copy(t):
    gid, c, s = t
    src, dst = f"{IMGS}/{gid}.jpg", f"{OUT}/{s}/{c}/{gid}.jpg"
    if os.path.exists(dst):
        return 0
    if not os.path.exists(src):
        return 1
    shutil.copy(src, dst)
    return 0
with ThreadPoolExecutor(32) as ex:                 # I/O bound -> threads help a lot on a slow disk
    missing = sum(ex.map(_copy, zip(man["GalaxyID"], man["cls"], man["split"])))
print("missing source images:", missing)

man[["GalaxyID", "cls", "split", "Class1.1", "Class1.2", "Class8.6"]].to_csv(f"{OUT}/manifest.csv", index=False)
print(man.groupby(["split", "cls"]).size().unstack())

# ---- zip for Kaggle upload (jpg is already compressed -> store only) --------
with zipfile.ZipFile("Galaxy_dataset_3class.zip", "w", zipfile.ZIP_STORED) as z:
    for root, _, files in os.walk(OUT):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, "."))
print("zip written:", round(os.path.getsize("Galaxy_dataset_3class.zip") / 1e6), "MB")
