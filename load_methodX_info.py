import datetime
import numpy as np
from pathlib import Path
import pickle

path0 = Path("/Users/floracharbonnier/Documents/GitHub/private_HEDGE/results")
path = path0 / "factors/n_transitions_EV.pickle"
with open(path, "rb") as file:
    n_transitions = pickle.load(file)

path = path0 / "n_data_type_EV.pickle"
with open(path, "rb") as file:
    n_data_type = pickle.load(file)

print(f"n_transitions['EV'] / n_data_type['EV'] {n_transitions['EV'] / n_data_type['EV']}")

path = path0 / "n_ids_EV.npy"
n_ids = np.load(path).item()

path = path0 / "n_ids_0_EV.npy"
n_ids_0 = np.load(path).item()

print(f"n_ids/n_ids_0 = {n_ids/n_ids_0}")

path = path0 / "range_dates_EV.npy"
range_dates = np.load(path)

print(f"start dates {datetime.date(2010, 1, 1) + datetime.timedelta(days=range_dates[0])}")
print(f"end dates {datetime.date(2010, 1, 1) + datetime.timedelta(days=range_dates[1])}")
