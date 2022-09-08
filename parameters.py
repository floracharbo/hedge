"""Load and perform preprocessing for input parameters."""
import datetime
import multiprocessing as mp
import os
from pathlib import Path
from typing import List, Tuple

import yaml

from utils import initialise_dict


def _formatting(prm: dict, run_config: dict) \
        -> Tuple[dict, dict]:
    """Format separator, line_terminator, n_rows, test_cell."""
    for key in ["separator", "line_terminator"]:
        for key2, value in prm[key].items():
            if value == "\\t":
                prm[key][key2] = "\t"
            elif value == "\\n":
                prm[key][key2] = "\n"

    for key, value in run_config["n_rows"].items():
        if value != "all":
            run_config["n_rows"][key] = int(value)

    test_cell_keys = list(prm["test_cell_translation"].keys())
    for key in test_cell_keys:
        if not isinstance(key, str):
            prm["test_cell_translation"][str(key)] \
                = prm["test_cell_translation"][key]
            del prm["test_cell_translation"][key]

    return prm, run_config


def _update_paths(paths: dict, prm: dict, run_config: dict) \
        -> dict:
    current_path = os.getcwd()
    location = (
        "local"
        if current_path[0: len(run_config["local"])] == run_config["local"]
        else "remote"
    )
    paths["data_path"] = Path(run_config["data_path"][location])
    paths["save_path"] = Path(current_path) / paths["save_folder"]
    paths["debug_path"] = Path(current_path) / paths["debug_folder"]
    paths["homes_path"] = {}  # the path to the file with homes information
    for data_source in prm["data_sources"]:
        paths["homes_path"][data_source] = (
            paths["data_path"] / paths["homes_file"][data_source]
        )

    # paths to the main data files with demand / generation data points
    paths["var_path"] = {}
    for data_type in prm["data_types"]:
        paths["var_path"][data_type] \
            = paths["data_path"] / paths["var_file"][data_type]

    return paths


def _make_dirs(paths: dict, data_types: List[str]):
    for path in ["save_path", "debug_path"]:
        if not os.path.exists(paths[path]):
            os.mkdir(paths[path])

    for folder in ["profiles", "clusters", "factors"]:
        path = paths['save_path'] / folder
        if not os.path.exists(path):
            os.mkdir(path)
    for data_type in data_types:
        path = paths["save_path"] / "profiles" / f"norm_{data_type}"
        if not os.path.exists(path):
            os.mkdir(path)
    path = paths["save_path"] / "profiles" / "EV_avail"
    if not os.path.exists(path):
        os.mkdir(path)


def get_parameters() -> Tuple[dict, dict, dict]:
    """Load input parameter files and perform pre-processing."""
    with open("inputs/parameters.yaml") as file:
        prm = yaml.safe_load(file)
    with open("inputs/paths.yaml") as file:
        paths = yaml.safe_load(file)
    with open("inputs/run_config.yaml") as file:
        run_config = yaml.safe_load(file)

    prm["data_types"] = run_config["data_types"]
    prm, run_config = _formatting(prm, run_config)

    prm["behaviour_types"] = [data_type for data_type in prm["behaviour_types"]
                              if data_type in prm["data_types"]]
    prm["CLNR_types"] = [data_type for data_type in prm["data_types"]
                         if prm["data_type_source"][data_type] == "CLNR"]

    # information about date and time
    prm["dT"] = 60 * 24 / run_config["n"]  # interval length in minutes
    prm["datetime_entries"] \
        = prm["date_entries"] + prm["time_entries"]
    prm["date0"] = datetime.date(2010, 1, 1)

    # possible types of transition between week day types (week/weekend)
    prm["day_trans"] = []
    for prev_day in prm["weekday_type"]:
        for next_day in prm["weekday_type"]:
            prm["day_trans"].append(f"{prev_day}2{next_day}")

    # for data filling in, which value are we substituting
    prm["direct_mult"] = {"prev": -1, "next": 1}
    # duration of the jumps in time in minutes
    prm["jump_dt"] = {"week": 2 * 24 * 60, "day": 24 * 60}
    prm["bound_delta"] = {
        "before": - prm["dT"],
        "after": prm["dT"],
        "replacement": 0
    }
    prm["replacement_types"] = []
    for direct in prm["directs"]:
        for jump in prm["jumps"]:
            prm["replacement_types"].append(direct + "_" + jump)

    # instructions for importing data
    # columns corresponding to each type of information in the data files
    prm["i_cols"] = initialise_dict(prm["data_types"], "empty_dict")
    prm["sequence_entries"] = initialise_dict(prm["data_types"])

    for data_type in prm["CLNR_types"]:
        prm["sequence_entries"][data_type] \
            = prm["sequence_entries_CLNR"] + [data_type]
        for name_col, i_col in zip(["id", "dtm", data_type],
                                   prm["i_cols_CLNR"]):
            prm["i_cols"][data_type][name_col] = i_col

    prm["sequence_entries"]["EV"] = prm["sequence_entries_NTS"]
    for name_col, i_col in zip(prm["name_cols_NTS"], prm["i_cols_NTS"]):
        prm["i_cols"]["EV"][name_col] = i_col

    # year                  : survey year
    # id                    : home unique id fo find home-specific data
    # weekday               : Day of the travel week (1-7)
    # mode                  : 5.0 (home car-driver), 7.0 (home car-passenger)
    # dist                  : Trip distance - miles
    # purposeFrom/purposeTo : 23 categories; home = 23
    # start_h/end_h         : Trip start/end time - hours component
    # start_min/end_min     : Trip start/end time - minutes component
    # check_time            : Trip start time band - 24 hourly bands
    # duration              : Total trip travelling time - minutes

    paths = _update_paths(paths, prm, run_config)

    # number of CPUs available to parallelise
    prm["n_cpu"] = (
        mp.cpu_count() if run_config["n_cpu"] is None else run_config["n_cpu"]
    )

    _make_dirs(paths, prm["data_types"])

    prm["bat"]["min_charge"] = prm["bat"]["cap"] * prm["bat"]["SoCmin"]

    # transfer run_config to prm
    for key in [
        "n_rows",
        "parallel",
        "lin_interp",
        "do_test_filling_in",
        "do_heat_map",
        "n",
        "n_clusters",
        "n_intervals"
    ]:
        prm[key] = run_config[key]

    return prm, paths, run_config
