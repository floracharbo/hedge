import os
from pathlib import Path
import shutil

hedge_inputs_dir = Path("hedge_inputs")
if not hedge_inputs_dir.exists():
    os.mkdir(hedge_inputs_dir)

for dir in ["factors", "clusters"]:
    if not (hedge_inputs_dir / dir).exists():
        os.mkdir(hedge_inputs_dir / dir)

shutil.copy("results/percentiles.pickle", "hedge_inputs/percentiles.pickle")

for param in ["gamma_prms", "mean_residual", "f_min", "f_max", "EV_p_pos", "EV_p_zero2pos",
              "EV_fs_brackets", "EV_mid_fs_brackets"]:
    shutil.copy(
        f"results/factors/{param}.pickle",
        f"hedge_inputs/factors/{param}.pickle"
    )

for param in ["p_clus", "p_trans", "n_clus"]:
    shutil.copy(
        f"results/clusters/{param}.pickle",
        f"hedge_inputs/clusters/{param}.pickle"
    )

if os.path.exists("hedge_inputs/profiles"):
    shutil.rmtree("hedge_inputs/profiles")
shutil.copytree("results/profiles", "hedge_inputs/profiles")
