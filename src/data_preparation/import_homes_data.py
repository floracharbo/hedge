"""Function import_homes_data imports home-specific information."""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils import dmy_to_cum_day, initialise_dict, str_to_cum_day


def _make_data(
        prm: dict,
        data_source: str,
        test_cell: dict,
        str_save: List[str],
) -> Tuple[list, dict, dict]:
    # import data

    start_id, end_id = [{} for _ in range(2)]

    home_data = pd.read_csv(
        prm["homes_path"][data_source],
        usecols=prm["i_cols_homes_info"][data_source],
        names=prm["name_cols_homes_info"][data_source],
        sep=prm["separator"][data_source],
        lineterminator=prm["line_terminator"][data_source],
        skiprows=1,
    )

    # obtain the start and end dates as number of days since 01/01/2010
    if data_source == "CLNR":
        home_data["start_no"] = home_data["start"].map(str_to_cum_day)
        home_data["end_no"] = home_data["end"].map(str_to_cum_day)
    elif data_source == "NTS":
        home_data["start_no"] = home_data.apply(
            lambda x: dmy_to_cum_day(
                x.start_day, x.start_month, x.survey_year),
            axis=1,
        )
        home_data["end_no"] = home_data.apply(
            lambda x: dmy_to_cum_day(
                x.end_day, x.end_month, x.survey_year),
            axis=1,
        )

    valid_test_cells = list(prm["test_cell_translation"].keys())
    home_type = {}
    for row in range(len(home_data)):
        id_ = int(home_data["id"][row])
        if (data_source == "CLNR"
                and home_data["test_cell"][row] in valid_test_cells) \
                or data_source == "NTS":
            start_id[id_] = home_data["start_no"][row]
            end_id[id_] = home_data["end_no"][row]
            if data_source == "CLNR":
                test_cell[id_] = home_data["test_cell"][row]
        if data_source == "NTS":
            home_type[id_] = home_data["home_type"][row]

    for label, obj in zip(str_save, [start_id, end_id, test_cell]):
        with open(prm["save_other"] / f"{label}.pickle", "wb") as file:
            pickle.dump(obj, file)
    if data_source == "NTS":
        with open(prm["save_other"] / "home_type_NTS.pickle", "wb") as file:
            pickle.dump(home_type, file)

    return [start_id, end_id], test_cell, home_type


def _load_data(
        save_path: Path,
        str_save: List[str],
        data_source: str,
        home_type: dict
) -> Tuple[List[dict], dict, dict]:
    # if data was already loaded and computed before,
    # load it from the saved files
    start_id, end_id, test_cell \
        = [np.load(save_path / f"{label}.npy",
                   allow_pickle=True).item()
           for label in str_save]

    if data_source == "NTS":
        with open(save_path / "home_type_NTS.pickle", "rb") as file:
            home_type = pickle.load(file)

    return [start_id, end_id], test_cell, home_type


def import_homes_data(prm: dict) -> Tuple[dict, List[dict], dict]:
    """
    Import home-specific information.

    e.g. valid dates, home ID, home type.
    """
    # import validity ddata for CLNR and NTS
    start_ids, end_ids = [
        initialise_dict(prm["data_sources"], "empty_dict") for _ in range(2)
    ]
    # id with no data will have start_id[dt][iID] = end_id[dt][iID] = None
    # NTS
    # id_: household id,
    # survey_year: year,
    # start_day = day of month (1-31),
    # start_month: month of year (1-12),
    # home_type: 1.0: Urban, 2.0: Rural, 3.0: Scotland, -10.0: DEAD, -8.0: NA
    str_save_root = ["start_avails", "end_avails", "test_cell"]
    home_type: Dict[int, int] = {}
    test_cell: Dict[int, str] = {}

    for data_source in prm["data_sources"]:
        str_save = [str_ + str(data_source) for str_ in str_save_root]
        make_homes_data = not os.path.exists(
            prm["save_other"] / f"start_avails_{data_source}.npy"
        )
        if data_source == "NTS" and not os.path.exists(
            prm["save_other"] / "homes_type_NTS.npy"
        ):
            make_homes_data = True

        if make_homes_data:
            start_end_id, test_cell, home_type_ \
                = _make_data(
                    prm, data_source, test_cell, str_save)
            if data_source == "NTS":
                home_type = home_type_
        else:
            start_end_id, test_cell, home_type = _load_data(
                prm["save_other"], str_save, data_source, home_type
            )

        start_ids[data_source], end_ids[data_source] = start_end_id

    return home_type, [start_ids, end_ids], test_cell
