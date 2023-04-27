"""
Cut up data into smaller blocks.

For manageable import and processing
"""

import csv
import os
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data_preparation.filling_in import stats_filling_in
from src.utils import data_id, initialise_dict, list_potential_paths_outs, save_fig


def _load_out(prm, data_type, chunk_rows):
    out = []
    for label in prm["outs_labels"]:
        file_name = f"{label}_{data_id(prm, data_type)}_{chunk_rows[0]}_{chunk_rows[1]}.pickle"
        potential_paths = list_potential_paths_outs(prm, data_type)
        for potential_path in potential_paths:
            file_path = potential_path / file_name
            if file_path.exists():
                with open(file_path, "rb") as file:
                    out.append(pickle.load(file))
                break

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
    for id_indexes in ids_indexes.values():
        if len(block_indexes[i_block] + id_indexes) > n_per_block:
            i_block = min(i_block + 1, prm["n_cpu"] - 1)
        block_indexes[i_block] = block_indexes[i_block] + id_indexes

    # for cutting up the overall data
    # we are aiming for equivalent burden per CPU
    path = prm["save_other"] \
        / f"block_indexes_{data_type}_n_rows{n_rows[data_type]}.pickle"
    with open(path, "wb") as file:
        pickle.dump(
            block_indexes,
            file
        )

    return block_indexes


def get_n_rows(
        data_type: str,
        prm: dict,
) -> int:
    """Obtain the number of rows of data."""
    path_n_rows = prm["save_other"] / f"n_rows_all_{data_type}.npy"
    if os.path.exists(path_n_rows):
        n_rows = int(np.load(path_n_rows))
    else:
        data_source = prm["data_type_source"][data_type]
        n_rows = 0
        # open main variable data file
        with open(prm["var_path"][data_type]) as file:
            read = csv.reader(file, delimiter=prm["separator"][data_source])
            for _ in read:
                n_rows += 1

        np.save(path_n_rows, n_rows)

    return n_rows


def add_out(
        prm,
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
        chunk_rows
) -> list:
    """Concatenate import_segments outputs that were imported separately."""
    if out[0] is None:
        out = _load_out(prm, data_type, chunk_rows)

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


def save_outs(outs, prm, data_type, chunks_rows):
    """Once import_segment is done, bring it all together."""
    days_ = []
    all_abs_error = initialise_dict(prm["fill_types"], "empty_np_array")
    types_replaced \
        = initialise_dict(prm["fill_types_choice"], "empty_dict")
    for fill_type in prm["fill_types_choice"]:
        types_replaced[fill_type] \
            = initialise_dict(prm["replacement_types"], "zero")

    all_data = np.zeros((prm["n"], 366))
    granularities = []
    range_dates = [1e6, - 1e6]
    n_ids = 0
    for out, chunk_rows in zip(outs, chunks_rows):
        [days_, all_abs_error, types_replaced,
         all_data, granularities, range_dates, n_ids] \
            = add_out(prm, out, days_, all_abs_error,
                      types_replaced, all_data, data_type,
                      granularities, range_dates, n_ids, chunk_rows)
        assert len(days_) > 0, f"in save_outs len(days_) {len(days_)}"

    if prm["data_type_source"][data_type] == "CLNR":
        np.save(
            prm["outs_path"] / f"granularities_{data_type}",
            granularities
        )

    if prm["do_heat_map"] and prm['plots']:
        if len(np.shape(all_data)) == 2:
            fig = plt.figure()
            ax = sns.heatmap(all_data)
            ax.set_title("existing data")
            fig_save_path = prm["save_other"] / f"existing_data_{data_type}"
            save_fig(fig, prm, fig_save_path)
            plt.close("all")

        else:
            print(f"{data_type} np.shape(all_data) {np.shape(all_data)}")
        print(f"np.sum(all_data) {np.sum(all_data)} {data_type}")
    if (
            prm["do_test_filling_in"]
            and prm["data_type_source"][data_type] == "CLNR"
    ):
        stats_filling_in(
            prm, data_type, all_abs_error, types_replaced)

    np.save(prm["outs_path"] / f"len_all_days_{data_type}", len(days_))
    np.save(prm["outs_path"] / f"n_ids_{data_type}", n_ids)
    np.save(prm["outs_path"] / f"range_dates_{data_type}", range_dates)

    assert len(days_) > 0, f"in save_outs len(days_) {len(days_)}"

    return days_


def save_intermediate_out(prm, out, chunk_rows, data_type):
    """
    Save individual output from parallel streams to file.

    Clear memory in the meantime.
    """
    if prm["save_intermediate_outs"]:
        for obj, label in zip(out, prm["outs_labels"]):
            with open(
                prm["outs_path"]
                / f"{label}_{data_id(prm, data_type)}_{chunk_rows[0]}_{chunk_rows[1]}.pickle",
                "wb"
            ) as file:
                pickle.dump(obj, file)
        out = [None] * 7

    return out


def get_data_chunks(prm, data_type):
    """
    Break down rows into chunks to be parallelised.

    Without interrupting id sequences.
    """
    unique_ids_path \
        = prm["save_other"] \
        / f"unique_ids_{data_type}_{prm['n_rows'][data_type]}.npy"
    chunks_path \
        = prm["save_other"] \
        / f"chunks_rows_{data_type}_{prm['n_rows'][data_type]}_max{prm['max_size_chunk']}.npy"
    if unique_ids_path.is_file() and chunks_path.is_file():
        # if the unique_ids have already been computed, load them
        chunks_rows = np.load(chunks_path)
    else:
        idx, ids = _get_rows_ids(prm, data_type, unique_ids_path)
        chunks_rows = _rows_ids_to_chunks(
            prm, idx, ids, data_type, chunks_path
        )

    return chunks_rows


def _get_rows_ids(prm, data_type, unique_ids_path):
    data_source = prm["data_type_source"][data_type]
    current_id = None
    with open(
            prm["var_path"][data_type],
            newline=prm["line_terminator"][data_source]
    ) as file:
        reader = csv.reader(
            file, delimiter=prm["separator"][data_source]
        )
        next(reader, None)  # skip the headers
        for no_row, row in enumerate(reader):
            if current_id is None:
                current_id = int(row[0])
                ids = np.array([current_id], dtype=np.int32)
                idx = np.array([no_row], dtype=np.int32)
            row_id = int(row[0])
            if row_id != current_id:
                idx = np.append(idx, no_row)
                ids = np.append(ids, row_id)
                current_id = row_id
            # no_row += 1

            if no_row > prm["n_rows"][data_type]:
                break

    idx = np.append(idx, prm["n_rows"][data_type])

    unique_ids = list(set(ids))
    np.save(unique_ids_path, unique_ids)
    np.save(
        prm["save_other"] / f"n_ids_0_{data_id(prm, data_type)}",
        len(unique_ids)
    )

    return idx, ids


def _rows_ids_to_chunks(prm, idx, ids, data_type, chunks_path):
    """Break down the data in chunks."""
    n_chunks = min(len(ids), prm["n_cpu"])
    n_rows_per_chunks = min(prm['n_rows'][data_type] / n_chunks, prm["max_size_chunk"])
    chunks_rows = []
    start_current_chunk = 0
    n_current_chunk = 0
    for i, (start_idx, end_idx) in enumerate(zip(idx[:-1], idx[1:])):
        n_current_chunk += (end_idx - start_idx)
        if n_current_chunk > n_rows_per_chunks or i == len(idx) - 1:
            chunks_rows.append([start_current_chunk, end_idx])
            start_current_chunk = end_idx
            n_current_chunk = 0
    np.save(chunks_path, chunks_rows)

    return chunks_rows
