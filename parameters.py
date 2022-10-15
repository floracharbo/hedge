"""Load and perform preprocessing for input parameters."""
import datetime
import multiprocessing as mp
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import yaml
from scipy.stats import alpha, chi, chi2, gamma, maxwell, norm

from utils import initialise_dict


def _import_columns_info(prm):

    prm["i_cols"] = initialise_dict(prm["data_types"], "empty_dict")
    for data_type in prm["CLNR_types"]:
        for name_col, i_col in zip(["id", "dtm", data_type],
                                   prm["i_cols_CLNR"]):
            prm["i_cols"][data_type][name_col] = i_col
    if "EV" in prm["data_types"]:
        for name_col, i_col in zip(prm["name_cols_NTS"], prm["i_cols_NTS"]):
            prm["i_cols"]["EV"][name_col] = i_col

    str_to_type = {
        'int': np.int32,
        'str': str,
        'flt': float
    }
    prm["dtypes"] = initialise_dict(prm["data_types"], "empty_dict")
    for data_type in prm["data_types"]:
        for i, name in enumerate(prm["i_cols"][data_type].keys()):
            data_source = prm["data_type_source"][data_type]
            type_ = str_to_type[prm["type_cols"][data_source][i]]
            prm["dtypes"][data_type][name] = type_

    return prm


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


def _init_data_filling(prm, run_config):
    prm["direct_mult"] = {"prev": -1, "next": 1}
    # duration of the jumps in time in minutes
    prm["jump_dt"] = {
        "week": 7 * 24 * 60,
        "day": 24 * 60,
        "two_days": 2 * 24 * 60
    }

    prm["bound_delta"] = {
        "before": - prm["step_len"],
        "after": prm["step_len"],
        "replacement": 0
    }
    prm["replacement_types"] = []
    for direct in prm["directs"]:
        for fill_type in prm["jumps"].keys():
            for jump in prm["jumps"][fill_type]:
                if f"{direct}_{jump}" not in prm["replacement_types"]:
                    prm["replacement_types"].append(f"{direct}_{jump}")

    prm["fill_types_choice"] = [
        fill_type for fill_type in prm["fill_types"]
        if fill_type != "lin_interp"
    ]

    if run_config["test_factor_distr"]:
        distributions = {
            'alpha': alpha,
            'chi': chi,
            'chi2': chi2,
            'gamma': gamma,
            'maxwell': maxwell,
            'norm': norm
        }
        if run_config["candidate_factor_distributions"] is None:
            prm["candidate_factor_distributions"] = distributions
        else:
            prm["candidate_factor_distributions"] = {}
            for candidate in run_config["candidate_distr"]:
                if candidate in distributions:
                    prm["candidate_factor_distributions"][candidate] \
                        = distributions[candidate]
                else:
                    print(f"Please add {candidate} to the "
                          f"distributions dictionary in parameters.py")

    return prm


def _update_paths(prm: dict, run_config: dict) \
        -> dict:
    current_path = os.getcwd()
    location = (
        "local"
        if current_path[0: len(run_config["local"])] == run_config["local"]
        else "remote"
    )
    prm["data_path"] = Path(run_config["data_path"][location])
    prm["save_folder"] = prm["save_folder"] + f"_n{run_config['n']}"
    for folder in ["save", "debug", "outs"]:
        prm[f"{folder}_path"] = Path(current_path) / prm[f"{folder}_folder"]
    prm["homes_path"] = {}  # the path to the file with homes information
    for data_source in prm["data_sources"]:
        prm["homes_path"][data_source] = (
            prm["data_path"] / prm["homes_file"][data_source]
        )

    # paths to the main data files with demand / generation data points
    prm["var_path"] = {}
    for data_type in prm["data_types"]:
        prm["var_path"][data_type] \
            = prm["data_path"] / prm["var_file"][data_type]

    return prm


def _make_dirs(prm: dict):
    for path in ["save_path", "debug_path", "outs_path"]:
        if not os.path.exists(prm[path]):
            os.mkdir(prm[path])

    for folder in ["profiles", "clusters", "factors"]:
        path = prm['save_path'] / folder
        if not os.path.exists(path):
            os.mkdir(path)
    for data_type in prm["data_types"]:
        path = prm["save_path"] / "profiles" / f"norm_{data_type}"
        if not os.path.exists(path):
            os.mkdir(path)
    path = prm["save_path"] / "profiles" / "EV_avail"
    if not os.path.exists(path):
        os.mkdir(path)


def get_parameters() -> Tuple[dict, dict]:
    """Load input parameter files and perform pre-processing."""
    with open("inputs/parameters.yaml") as file:
        prm = yaml.safe_load(file)
    with open("inputs/run_config.yaml") as file:
        run_config = yaml.safe_load(file)

    prm["data_types"] = run_config["data_types"]
    prm, run_config = _formatting(prm, run_config)

    # useful list of data types
    prm["behaviour_types"] = [
        data_type for data_type in prm["behaviour_types"]
        if data_type in prm["data_types"]
    ]
    prm["CLNR_types"] = [
        data_type for data_type in prm["data_types"]
        if prm["data_type_source"][data_type] == "CLNR"
    ]

    # information about date and time
    prm["step_len"] = 60 * 24 / run_config["n"]  # interval length in minutes
    prm["datetime_entries"] \
        = prm["date_entries"] + prm["time_entries"]
    prm["date0"] = datetime.date(2010, 1, 1)

    # for data filling in, which value are we substituting
    prm = _init_data_filling(prm, run_config)

    # possible types of transition between week day types (week/weekend)
    prm["day_trans"] = []
    for prev_day in prm["weekday_type"]:
        for next_day in prm["weekday_type"]:
            prm["day_trans"].append(f"{prev_day}2{next_day}")

    # instructions for importing data
    # columns corresponding to each type of information in the data files
    prm = _import_columns_info(prm)

    prm["sequence_entries"] = initialise_dict(prm["data_types"])

    for data_type in prm["CLNR_types"]:
        prm["sequence_entries"][data_type] \
            = prm["sequence_entries_CLNR"] + [data_type]

    if "EV" in prm["data_types"]:
        prm["sequence_entries"]["EV"] = prm["sequence_entries_NTS"]

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

    prm = _update_paths(prm, run_config)

    # number of CPUs available to parallelise
    prm["n_cpu"] = (
        mp.cpu_count() if run_config["n_cpu"] is None else run_config["n_cpu"]
    )

    _make_dirs(prm)

    prm["EV"]["min_charge"] = prm["EV"]["cap"] * prm["EV"]["SoCmin"]

    # transfer run_config to prm
    for key in [
        "n_rows",
        "parallel",
        "fill_type",
        "do_test_filling_in",
        "prob_test_filling_in",
        "do_heat_map",
        "n",
        "n_clus",
        "n_intervals",
        "plots",
        "test_factor_distr",
        "save_intermediate_outs",
        "max_size_chunk"
    ]:
        prm[key] = run_config[key]

    if not prm["plots"]:
        prm["do_heat_map"] = False

    return prm, run_config
