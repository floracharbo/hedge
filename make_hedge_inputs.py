import os
from pathlib import Path
import shutil

hedge_inputs_dir = Path("hedge_inputs")
if not hedge_inputs_dir.exists():
    os.mkdir(hedge_inputs_dir)

for dir in ["factors", "clusters"]:
    if not (hedge_inputs_dir / dir).exists():
        os.mkdir(hedge_inputs_dir / dir)

for param in ["gamma_prms", "f_mean", "f_min", "f_max"]:
    shutil.copy(
        f"results/factors/{param}.pickle",
        f"hedge_inputs/factors/{param}.pickle"
    )

for param in ["p_clus", "p_trans"]:
    shutil.copy(
        f"results/clusters/{param}.pickle",
        f"hedge_inputs/clusters/{param}.pickle"
    )