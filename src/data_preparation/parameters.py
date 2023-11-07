"""Load and perform preprocessing for input parameters."""
import datetime
import multiprocessing as mp
import os
from pathlib import Path
from typing import Tuple

import yaml
from scipy.stats import alpha, chi, chi2, gamma, maxwell, norm

from src.utils import initialise_dict, run_id


def _import_columns_info(prm):

    prm["i_cols"] = initialise_dict(prm["data_types"], "empty_dict")
    if 'loads' in prm["data_types"]:
        for name_col, i_col in zip(["id", "dtm", 'loads'],  prm["i_cols_CLNR"]):
            prm["i_cols"]['loads'][name_col] = i_col
    if "car" in prm["data_types"]:
        for name_col, i_col in zip(prm["name_cols_NTS"], prm["i_cols_NTS"]):
            prm["i_cols"]["car"][name_col] = i_col
    if 'gen' in prm['data_types']:
        if prm['var_file']['gen'][-len('parquet'):] == 'parquet':
            names = ['id', 'dtm', 'gen']
            i_cols = prm['i_cols_CLNR']
        elif prm['var_file']['gen'] == 'EXPORT HourlyData - Customer Endpoints.csv':
            names = ['SerialNo', 'd_y', 'd_m', 'd_d', 'd_w', 't_h', 't_m', 'P_GEN_MIN', 'P_GEN_MAX']
            i_cols = prm['i_cols_gen']
        elif prm['var_file']['gen'] == '15minute_data_austin.csv':
            names = ['id', 'dtm', 'gen']
            i_cols = prm['i_cols_gen']
        else:
            names = ['id', 'Measurement Description', 'dtm', 'gen']
            i_cols = prm['i_cols_gen']
        for name_col, i_col in zip(names, i_cols):
            prm["i_cols"]["gen"][name_col] = i_col

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
    run_config['n_rows0'] = run_config['n_rows'].copy()
    run_config['n_consecutive_days'] = sorted(run_config['n_consecutive_days'], reverse=True)

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
    prm["save_hedge"] = Path("data") / "hedge_inputs" / f"{run_id(run_config)}"
    prm['paths']["input_folder"] = Path(prm['paths']["input_folder"])
    prm['paths']['hedge_inputs'] = \
        prm['paths']["input_folder"] / prm['paths']['hedge_inputs_folder'] \
        / f"n{run_config['syst']['H']}"
    prm["save_other"] = Path("data") / "other_outputs" / f"{run_id(run_config)}"
    prm["homes_path"] = {}  # the path to the file with homes information
    for data_source in prm["data_sources"]:
        prm["homes_path"][data_source] = (
            Path('data') / 'data_preparation_inputs' / prm["homes_file"][data_source]
        )

    # paths to the main data files with demand / generation data points
    prm["var_path"] = {}
    for data_type in prm["data_types"]:
        prm["var_path"][data_type] \
            = Path('data') / 'data_preparation_inputs' / prm["var_file"][data_type]

    prm["outs_path"] = prm["save_other"] / "outs"

    return prm


def _make_dirs(prm: dict):
    for folder in ["hedge_inputs", "other_outputs"]:
        if not os.path.exists(Path("data") / folder):
            os.mkdir(Path("data") / folder)
    for folder in ["save_hedge", "save_other"]:
        if not os.path.exists(prm[folder]):
            os.mkdir(prm[folder])
        for sub_folder in ["profiles", "clusters", "factors"]:
            path = prm[folder] / sub_folder
            if not os.path.exists(path):
                os.mkdir(path)

    if not os.path.exists(prm["outs_path"]):
        os.mkdir(prm["outs_path"])

    for folder in [f"norm_{data_type}" for data_type in prm["data_types"]] + ["car_avail"]:
        path = prm["save_hedge"] / "profiles" / folder
        if not os.path.exists(path):
            os.mkdir(path)


def get_parameters() -> Tuple[dict, dict]:
    """Load input parameter files and perform pre-processing."""
    with open("config_parameters/fixed_parameters.yaml") as file:
        prm = yaml.safe_load(file)
    with open("config_parameters/data_preparation_config.yaml") as file:
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
    prm["step_len"] = 60 * 24 / run_config["syst"]["H"]  # interval length in minutes
    prm['n'] = run_config["syst"]["H"]
    prm['syst'] = {'H': run_config["syst"]["H"]}
    prm["datetime_entries"] \
        = prm["date_entries"] + prm["time_entries"]
    prm["date0"] = datetime.date(2010, 1, 1)

    # for data filling in, which value are we substituting
    prm = _init_data_filling(prm, run_config)

    # possible types of transition between week day types (week/weekend)
    prm["day_trans"] = []
    for prev_day in prm["weekday_types"]:
        for next_day in prm["weekday_types"]:
            prm["day_trans"].append(f"{prev_day}2{next_day}")
    # instructions for importing data
    # columns corresponding to each type of information in the data files
    prm = _import_columns_info(prm)

    prm["sequence_entries"] = initialise_dict(prm["data_types"])

    for data_type in prm["CLNR_types"]:
        prm["sequence_entries"][data_type] \
            = prm["sequence_entries_CLNR"] + [data_type]

    if "car" in prm["data_types"]:
        prm["sequence_entries"]["car"] = prm["sequence_entries_NTS"]

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
    prm['n'] = run_config['syst']['H']

    _make_dirs(prm)

    prm["car"]["min_charge"] = prm["car"]["cap"] * prm["car"]["SoCmin"]

    # transfer run_config to prm
    for key in [
        "n_rows",
        "n_rows0",
        "parallel",
        "fill_type",
        "do_test_filling_in",
        "prob_test_filling_in",
        "do_heat_map",
        "n_clus",
        "n_intervals",
        "plots",
        "test_factor_distr",
        "save_intermediate_outs",
        "max_size_chunk",
        'gan_generation_profiles',
        'kurtosis',
        'brackets_definition',
        'high_res',
        'n_consecutive_days',
        'months',
        'train_set_size'
    ]:
        prm[key] = run_config[key]

    for key in ['max_power_cutoff', 'max_daily_energy_cutoff']:
        prm['car'][key] = run_config['car'][key]
    prm['H'] = run_config['syst']['H']
    if not prm["plots"]:
        prm["do_heat_map"] = False

    return prm
