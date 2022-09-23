"""
Cut up data into smaller blocks.

For manageable import and processing
"""

import csv
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

from utils import initialise_dict


def _load_out(prm, ids):
    out = []
    for label in prm["outs_labels"]:
        with open(prm["outs_path"] / f"{label}_{ids[0]}.pickle", "rb") \
                as file:
            out.append(pickle.load(file))

    return out


def _get_split_ids(read, n_rows, i_col_id):
    ids_indexes = {}
    for row, i_row in zip(read, range(n_rows)):
        if row[i_col_id] not in ids_indexes:
            ids_indexes[row[i_col_id]] = []
        ids_indexes[row[i_col_id]].append(i_row)

    return ids_indexes


def _get_indexes_blocks(prm, data_type, n_rows):
    i_col_id = prm["i_cols"][data_type]["id"]
    data_source = prm["data_type_source"][data_type]
    # obtain all the rows at which the id changes
    with open(prm["var_path"][data_type]) as file:
        # open main variable data file
        read = csv.reader(file,
                          delimiter=prm["separator"][data_source])
        ids_indexes = _get_split_ids(read, n_rows[data_type], i_col_id)

    n_per_block = n_rows[data_type] / prm["n_cpu"]
    block_indexes = initialise_dict(range(prm["n_cpu"]), "empty_list")
    i_block = 0
    for id, id_indexes in ids_indexes.items():
        if len(block_indexes[i_block] + id_indexes) > n_per_block:
            i_block = min(i_block + 1, prm["n_cpu"] - 1)
        block_indexes[i_block] = block_indexes[i_block] + id_indexes

    # for cutting up the overall data
    # we are aiming for equivalent burden per CPU
    path = prm["save_path"] \
        / f"block_indexes_{data_type}_n_rows{n_rows[data_type]}.pickle"
    with open(path, "wb") as file:
        pickle.dump(
            block_indexes,
            file
        )

    return block_indexes


def _get_n_rows(
        data_source: str,
        data_type: str,
        prm: dict,
) -> int:
    """Obtain the number of rows of data."""
    path_n_rows = prm["save_path"] / f"n_rows_all_{data_type}.npy"
    if os.path.exists(path_n_rows):
        n_rows = int(np.load(path_n_rows))
    else:
        n_rows = 0
        # open main variable data file
        with open(prm["var_path"][data_type]) as file:
            read = csv.reader(file, delimiter=prm["separator"][data_source])
            for _ in read:
                n_rows += 1

        np.save(path_n_rows, n_rows)

    return n_rows


def add_out(prm,
            out: Tuple[List,
                       List[float],
                       Dict[str, int],
                       np.ndarray,
                       List[int]],
            days: list,
            all_abs_error: Dict[str, np.ndarray],
            types_replaced: dict,
            all_data: np.ndarray,
            data_type: str,
            granularities: List[int],
            range_dates: list,
            n_ids: int,
            ids
            ) -> list:
    """Concatenate import_segments outputs that were imported separately."""
    if out[0] is None:
        out = _load_out(prm, ids)

    assert len(out[0]) > 0, f"len(out[0]) = {len(out[0])}"
    days = days + out[0]

    if prm["data_type_source"][data_type] == "CLNR":
        if prm["do_test_filling_in"]:
            for fill_type in prm["fill_types"]:
                all_abs_error[fill_type] \
                    = np.concatenate(
                    (all_abs_error[fill_type], out[1][fill_type])
                )
            for fill_type in prm["fill_types_choice"]:
                for replacement in prm["replacement_types"]:
                    types_replaced[fill_type][replacement] \
                        += out[2][fill_type][replacement]

    if out[3] is not None:
        all_data += out[3]

    granularities += [
        granularity for granularity in out[4]
        if granularity not in granularities
    ]
    if out[5][0] < range_dates[0]:
        range_dates[0] = out[5][0]
    if out[5][1] > range_dates[1]:
        range_dates[1] = out[5][1]

    n_ids += out[6]

    return [
        days, all_abs_error, types_replaced, all_data,
        granularities, range_dates, n_ids
    ]
