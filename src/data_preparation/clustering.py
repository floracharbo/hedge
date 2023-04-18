"""
Object Categoriser.

For imported data, characterise its categories and properties.

Methods are:
- __init__: Add relevant properties to object and initialise methods.
- split_day_types: Split week day and lweekend days.
- clustering: Cluster the data profiles in behaviour groups.
- scaling_factors: Obtain scaling factors corresponding to the data.
- _correlation_factors: Obtain and plot scaling factor correlations.
- _print_error_factors: Print info if Exception in computing pearsonr
- _scaling_factors_transition: Characterise factors transitions and save info.
- _save_scaling_factors: Save scaling factor correlation data.
- _cluster_and_plot: Cluster profiles and plot for given data and day types.
- _initialise_cluster_dicts: Initialise dictionaries for storing cluster info.
- _transition_probabilities: Get probabilities of transtioning between
day types and clusters.
"""
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data_preparation.factors_generation import (
    compute_factors_clusters_generators, compute_profile_generators)
from src.utils import initialise_dict


def _get_n_trans(n_data_type, data_type, days, n_trans, banks, n_consecutive_days=2):
    if data_type == 'gen':
        for month in range(12):
            for i in range(n_consecutive_days):
                banks[data_type][month][f"f{i}of{n_consecutive_days}"] = []

    for i in range(n_data_type[data_type] - (n_consecutive_days - 1)):
        consecutive_days = [days[data_type][d] for d in range(i, i + n_consecutive_days)]
        same_id = all(consecutive_days[d]['id'] == consecutive_days[0]["id"] for d in range(n_consecutive_days))
        subsequent_days = all(
            consecutive_days[d]["cum_day"] == consecutive_days[0]["cum_day"] + d for d in range(n_consecutive_days)
        )
        if same_id and subsequent_days:  # record transition
            if data_type == 'gen':
                i_month = consecutive_days[0]['month'] - 1
                for i, day_ in enumerate(consecutive_days):
                    if day_["factor"] < 0:
                        print(f"gen {i} Negative factor: {day_['factor']}")
                    banks[data_type][i_month][f"f{i}of{n_consecutive_days}"].append(day_["factor"])
            else:
                day_types, index_wdt = [
                    [day_[key] for day_ in consecutive_days]
                    for key in ["day_type", "index_wdt"]
                ]
                transition = f"{day_types[-2]}2{day_types[-1]}"
                clusters = [
                    consecutive_days[d]["cluster"] for d in range(n_consecutive_days)
                ]
                idx = tuple(clusters[d] for d in range(n_consecutive_days))
                n_trans[data_type][transition][idx] += 1
                for i, day_ in enumerate(consecutive_days):
                    banks[data_type][transition][f"f{i}of{n_consecutive_days}"].append(day_["factor"])

    return banks, n_trans


def _transition_probabilities(
    data_type: str,
    days: List[dict],
    p_clus: Dict[str, Dict[str, float]],
    n_trans: Dict[str, Dict[str, np.ndarray]],
    p_trans: Dict[str, Dict[str, np.ndarray]],
    banks: Dict[str, dict],
    n_clus_all_: int,
    prm: Dict[str, Any],
    n_data_type: Dict[str, int],
    n_consecutive_days: int,
) -> Tuple[
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, dict]
]:
    """Get probabilities of transtioning between day types and clusters."""
    for transition in prm["day_trans"]:
        shape = tuple(n_clus_all_ for _ in range(n_consecutive_days))
        n_trans[data_type][transition] = np.zeros(shape)
        p_trans[data_type][transition] = np.zeros(shape)

    banks, n_trans = _get_n_trans(
        n_data_type, data_type, days, n_trans, banks, n_consecutive_days
    )
    if len(banks['loads']['wd2wd'][f'f0of{n_consecutive_days}']) == 0:
        print(f"len(banks['loads']['wd2wd']['f0of{n_consecutive_days}']) == 0 after _get_n_trans")

    for c0 in range(n_clus_all_):
        for c1 in range(n_clus_all_):
            if n_consecutive_days == 2:
                for transition in prm["day_trans"]:
                    p_trans[data_type][transition][c0, c1] = (
                        n_trans[data_type][transition][c0, c1]
                        / sum(n_trans[data_type][transition][c0])
                    ) if sum(n_trans[data_type][transition][c0]) > 0 else None
            elif n_consecutive_days == 3:
                for c2 in range(n_clus_all_):
                    for transition in prm["day_trans"]:
                        p_trans[data_type][transition][c0, c1, c2] = (
                                n_trans[data_type][transition][c0, c1, c2]
                                / sum(n_trans[data_type][transition][c0, c1])
                        ) if sum(n_trans[data_type][transition][c0, c1]) > 0 else None
            else:
                print(f"implement n_consecutive_days = {n_consecutive_days}")

    with open(prm["save_hedge"] / "clusters" / f"{data_type}_p_trans_n_consecutive_days{n_consecutive_days}", "wb") as file:
        pickle.dump(p_trans, file)

    if n_consecutive_days == 2:
        v_max = np.max(
            [
                np.max(p_trans[data_type][transition])
                for transition in prm["day_trans"]
            ]
        )
        v_min = np.min(
            [
                np.min(p_trans[data_type][transition])
                for transition in prm["day_trans"]
            ]
        )

        if prm["plots"]:
            for transition in prm["day_trans"]:
                _plot_heat_map_p_trans(
                    p_trans, transition, data_type, v_min, v_max, prm
                )

            # plot once with the colour bar
            _plot_heat_map_p_trans(
                p_trans, transition, data_type, v_min,
                v_max, prm, colourbar=True
            )

        p_clus[data_type] = {}
        for day_type in prm["weekday_type"]:
            pcluss = [
                banks[data_type][day_type][k]["p_clus"] for k in range(n_clus_all_)
            ]
            p_clus[data_type][day_type] = pcluss
        if prm['gan_generation_factors_clusters']:
            compute_factors_clusters_generators(
                prm, n_data_type, data_type, days, p_clus, p_trans, n_clus_all_
            )

    return p_clus, p_trans, n_trans, banks


def _plot_heat_map_p_trans(
        p_trans, transition, data_type, v_min,
        v_max, prm, colourbar=False
):
    if prm['plots']:
        fig = plt.figure()
        sns.heatmap(
            p_trans[data_type][transition], vmin=v_min, vmax=v_max,
            cmap="RdBu_r", yticklabels=False,
            xticklabels=False, cbar=colourbar, square=True
        )
        plt.tick_params(left=False, bottom=False)
        plt.tight_layout()
        plt.gca().tick_params(left=False, bottom=False)
        fig.savefig(
            prm["save_other"] / "clusters" / f"p_trans_heatmap_{data_type}_{transition}"
        )
        plt.close("all")


def _plot_clusters(
        transformed_features,
        norm_vals,
        data_type,
        day_type,
        bank,
        vals_k,
        prm,
        save_path: Path):
    # plot clusters
    xs = np.arange(0, prm["n"])
    if np.shape(norm_vals) != (len(transformed_features), prm["n"]):
        print(
            f"check np.shape(norm_vals) = {np.shape(norm_vals)} "
            f"same as ({len(transformed_features)}, {prm['n']})"
        )
    statistical_indicators = {k: {} for k in range(prm["n_clus"][data_type])}
    for k in range(prm["n_clus"][data_type]):
        if bank[k]["n_clus"] != 0:
            for statistical_indicator in ['p10', 'p50', 'p90', 'mean']:
                statistical_indicators[k][statistical_indicator] = np.zeros(prm["n"])
            for time in range(prm["n"]):
                for percentile in [10, 50, 90]:
                    statistical_indicators[k][f'p{percentile}'][time] = np.percentile(
                        vals_k[k][:, time], percentile
                    )
                statistical_indicators[k]['mean'][time] = np.mean(vals_k[k][:, time])
            if prm["plots"]:
                for stylised in [True, False]:
                    fig = plt.figure()
                    plt.fill_between(
                        xs,
                        statistical_indicators[k]['p90'],
                        color="blue",
                        alpha=0.1,
                        label="10th to 90th percentile",
                    )
                    plt.fill_between(xs, statistical_indicators[k]['p10'], color="w")
                    if stylised:
                        title = f"Cluster {k} demand {day_type} stylised"
                        plt.plot(
                            xs, statistical_indicators[k]['mean'],
                            color="blue", label="mean", lw=3
                        )
                        plt.tight_layout()
                        plt.axis('off')
                    else:
                        plt.plot(xs, statistical_indicators[k]['p50'], color="red", label="median")
                        plt.plot(xs, statistical_indicators[k]['mean'], color="blue", label="mean")
                        plt.legend()
                        title = f"Cluster {k} demand {day_type}"
                        plt.title(title)
                    fig.savefig(
                        save_path / "clusters" / title.replace(" ", "_")
                    )
                plt.close("all")

    return statistical_indicators


def _get_vals_k(labels, norm_vals, n_clus):
    vals_k, idx_k_clustered = {}, {}
    for k in range(n_clus):
        idx_k_clustered[k] = [i for i, label in enumerate(labels) if label == k]
        vals_k[k] = np.array([norm_vals[i] for i in idx_k_clustered[k]])

    return vals_k, idx_k_clustered


def _get_cdfs(distances, label, prm, bank):
    min_cdfs_, max_cdfs_ = [], []

    for i, distance in enumerate(distances):
        if distance is None:
            bank[i]["cdfs"] = None
            continue

        # plot histogram distances around the cluster
        if prm["plots"]:
            fig, ax1 = plt.subplots()
            plt.hist(
                distance, 100,
                density=True, facecolor='k', alpha=0.75, label='data'
            )
            plt.ylabel('Probability', color='k')
            ax2 = ax1.twinx()
            ax2.hist(
                distance, 100, density=True, cumulative=True, label='CDF',
                histtype='step', alpha=0.8, color='r'
            )
            plt.ylabel('Cumulative probability', color='r')
            plt.xlabel('Distances to cluster')
            title = f"Histogram of distance to the cluster centers {label} {i}"
            plt.title(title)
            plt.grid(True)
            fig.savefig(prm["save_other"] / "clusters" / title.replace(" ", "_"))
            plt.close("all")

        # get cumulative probability for each profile
        # getting data of the histogram
        count, bins_edges = np.histogram(distance, bins=1000)
        # finding the PDF of the histogram using count values
        pdf = count / sum(count)
        # finding the CDF of the histogram
        cdfs = np.cumsum(pdf)
        # matching each distance data point to its bin probability
        all_cdfs = [
            [cdf for cdf, bin_start in zip(cdfs + cdfs[-1], bins_edges)
             if dist >= bin_start][0]
            for dist in distance
        ]
        bank[i]["cdfs"] = all_cdfs
        min_cdfs_.append(all_cdfs[0])
        max_cdfs_.append(all_cdfs[-1])

    return bank, min_cdfs_, max_cdfs_


def _get_p_clus(vals_k, n_days, data_type, n_zeros):
    n_clus = max(list(vals_k.keys())) + 1
    n_clus_all = (
        n_clus
        if data_type == "loads"
        else n_clus + 1
    )
    bank = initialise_dict(range(n_clus_all), "empty_dict")

    for k, val_k in vals_k.items():
        bank[k]["n_clus"] = len(val_k)
        bank[k]["p_clus"] = len(val_k) / n_days
    if data_type == 'car':
        bank[n_clus_all - 1]["n_clus"] = n_zeros
        bank[n_clus_all - 1]["p_clus"] = n_zeros / n_days

    return bank, n_clus_all


def _get_profs(
        to_cluster: List[int],
        bank: dict,
        days_: List[dict],
        data_type: str,
        idx_k_clustered: List[int],
        cluster_distance: list
) -> dict:
    """Get data banks of clusters."""
    n_k = max(list(bank.keys())) + 1
    n_steps = len(days_[0][data_type])
    for k, bank_ in bank.items():
        if data_type == 'car' and k == n_k - 1:
            bank_["profs"] = [[0 for _ in range(n_steps)]]
            bank_["dists"] = [[0 for _ in range(n_steps)]]
        else:
            idx_k_all = [to_cluster[i] for i in idx_k_clustered[k]]
            bank_["profs"] = [days_[i][f"norm_{data_type}"] for i in idx_k_all]
            assert all(
                sum(prof) == 0 or abs(sum(prof) - 1) < 1e-3
                for prof in bank_["profs"]
            ), f"{data_type} normalised profile should sum to 1"
            if data_type == "car":
                bank_["avail"] = [days_[i]["avail"] for i in idx_k_all]
            bank_["dists"] = [cluster_distance[i][k] for i in idx_k_clustered[k]]

    return bank


def _cluster_module(
        transformed_features: list,
        n_clus: int,
        to_cluster: list,
        days_: list,
        i_zeros: list,
) -> Tuple[list, list, int, list]:
    # actual clustering
    clusobj = KMeans(n_clusters=n_clus, n_init=100, random_state=0)
    obj = clusobj.fit(transformed_features)

    n_zeros = len(days_) - len(to_cluster)
    # obtain main centroids of each cluster
    # centers_unordered = obj.cluster_centers_
    # centers_alloc = obj.fit_predict(centers_unordered)
    # centers = [centers_unordered[np.where(centers_alloc == c)[0][0]]
    #            for c in range(prm["n_clus"][data_type])]
    labels = obj.fit_predict(transformed_features)
    for label, i in zip(labels, to_cluster):
        days_[i]["cluster"] = label
    for i in i_zeros:
        days_[i]["cluster"] = n_clus

    # obtain distances to centroid for each profile
    # -> ndarray of shape (n_samples, n_clus[data_type])
    # KMeans.transform() returns an array of distances
    # of each sample to the cluster center
    cluster_distances = obj.transform(transformed_features)

    return labels, cluster_distances, n_zeros, days_


def _elbow_method(transformed_features, data_type, day_type, prm):
    # elbow method
    if not (prm['save_other'] / "clusters" / "Elbow_method_{data_type}_{day_type}.npy").is_file():
        wcss = []
        maxn_clus = min(10, len(transformed_features))
        for i in range(1, maxn_clus):
            kmeans = KMeans(
                n_clusters=i,
                init="k-means++",
                max_iter=300,
                n_init=10,
                random_state=0,
            )
            kmeans.fit(transformed_features)
            wcss.append(kmeans.inertia_)
        if prm['plots']:
            fig = plt.figure()
            plt.plot(range(1, maxn_clus), wcss)
            title = f"Elbow Method {data_type} {day_type}"
            plt.title(title)
            plt.xlabel("Number of clusters")
            plt.ylabel("WCSS")
            fig.savefig(prm['save_other'] / "clusters" / title.replace(" ", "_"))
            plt.close("all")


def _get_features(days_, data_type, prm):
    # prepare data features
    # only cluster if positive values
    to_cluster = [
        i
        for i in range(len(days_))
        if days_[i][data_type] is not None and sum(days_[i][data_type]) > 0
    ]
    if data_type == "car":
        i_zeros = [i for i in range(len(days_))
                   if days_[i][data_type] is None
                   or sum(days_[i][data_type]) == 0]
        assert len(to_cluster) + len(i_zeros) == len(days_), \
            f"len(to_cluster) {len(to_cluster)} " \
            f"+ len(i_zeros) {len(i_zeros)} != " \
            f"len(days_) {len(days_)}"
    else:
        i_zeros = []

    features = []
    norm_vals = []
    for i in to_cluster:
        norm_day = days_[i][f"norm_{data_type}"]
        if data_type == "loads":
            peak = np.max(norm_day)
            t_peak = np.argmax(norm_day)
            values = [
                np.mean(
                    norm_day[
                        int(interval[0] * prm["n"] / 24):
                        int(interval[1] * prm["n"] / 24)
                    ]
                )
                for interval in prm["dem_intervals"]
            ]
            if len(norm_day) != prm["n"]:
                print(f"len(to_cluster) " f"= {len(to_cluster)}")
            features.append([peak, t_peak] + values)

        elif data_type == "car":
            features.append(
                days_[i][data_type][
                    int(6 * 60 / prm["step_len"]):
                    int(22 * 60 / prm["step_len"])]
            )
        norm_vals.append(days_[i][f"norm_{data_type}"])
    transformed_features = StandardScaler().fit_transform(features)

    return transformed_features, to_cluster, i_zeros, norm_vals


def _initialise_cluster_transition_dicts(prm, n_consecutive_days, banks):
    """Initialise dictionaries for storing cluster info."""
    n_trans, p_trans = [
        initialise_dict(["loads", "car"], "empty_dict") for _ in range(2)
    ]
    for data_type in ["loads", "car"]:
        n_trans[data_type], p_trans[data_type] = [
            initialise_dict(prm["day_trans"]) for _ in range(2)
        ]
        for transition in prm["day_trans"]:
            for i in range(n_consecutive_days):
                banks[data_type][transition][f"f{i}of{n_consecutive_days}"] = []

    return n_trans, p_trans, banks


def _initialise_cluster_dicts(prm):
    """Initialise dictionaries for storing cluster info."""
    p_clus, n_zeros, n_clus_all, banks = [
        {data_type: {} for data_type in ["loads", "car"]}
        for _ in range(4)
    ]
    for data_type in ["loads", "car"]:
        p_clus[data_type], n_zeros[data_type] = [
            {weekday_type: [] for weekday_type in prm["weekday_type"]}
            for _ in range(2)
        ]
        banks[data_type] = {day_type: {} for day_type in prm["weekday_type"] + prm["day_trans"]}

    min_cdfs, max_cdfs = [
        {data_type: {} for data_type in prm["data_types"]}
        for _ in range(2)
    ]

    return p_clus, n_zeros, n_clus_all, banks, min_cdfs, max_cdfs


def _group_gen_month(days_, prm):
    bank = initialise_dict(range(12), "empty_dict")
    cluster_distances = []
    for i_month, month in enumerate(range(1, 13)):
        i_days = [i for i, day in enumerate(days_) if day["month"] == month]
        bank[i_month]["profs"] = [days_[i]["norm_gen"] for i in i_days]
        for property_ in ["factor", "id", "cum_day"]:
            bank[i_month][property_] = [days_[i][property_] for i in i_days]

        path = prm["save_hedge"] / "profiles" / "norm_gen"
        np.save(
            path / f"i_month{i_month}",
            bank[i_month]["profs"],
        )

        if len(bank[i_month]["profs"]) > 0:
            transformed_features \
                = StandardScaler().fit_transform(bank[i_month]["profs"])
            clusobj = KMeans(n_clusters=1, n_init=100, random_state=0)
            obj = clusobj.fit(transformed_features)
            cluster_distances.append(obj.transform(transformed_features))
        else:
            cluster_distances.append(None)

    bank, min_cdfs, max_cdfs = _get_cdfs(cluster_distances, "gen", prm, bank)

    for i_month in range(12):
        np.save(
            prm["save_hedge"] / "clusters"
            / f"cdfs_clus_gen_{i_month}",
            bank[i_month]["cdfs"],
        )

    return bank, min_cdfs, max_cdfs


def _save_clustering(
        prm: dict,
        save_path: Path,
        banks: dict,
        p_clus: dict,
        p_trans: dict,
        min_cdfs: dict,
        max_cdfs: dict
):
    prof_path = save_path / "profiles"
    for data_type in prm["behaviour_types"]:
        for day_type in prm["weekday_type"]:
            for k in range(prm["n_clus"][data_type]):
                np.save(
                    save_path / "clusters"
                    / f"cdfs_clus_{data_type}_{day_type}_{k}",
                    banks[data_type][day_type][k]["cdfs"],
                )
                path = prof_path / f"norm_{data_type}"
                np.save(
                    path / f"c{k}_{day_type}",
                    banks[data_type][day_type][k]["profs"],
                )

                assert not np.all(
                    np.array(banks[data_type][day_type][k]["profs"]) == 0
                ), f"{data_type} {k} {day_type} all zeros"
                if data_type == "car":
                    path = prof_path / "car_avail"
                    np.save(
                        path / f"c{k}_{day_type}",
                        banks[data_type][day_type][k]["avail"],
                    )

    if "car" in prm["data_types"]:
        k = prm["n_clus"]["car"]
        for day_type in prm["weekday_type"]:
            np.save(
                prof_path / "car_avail" / f"c{k}_{day_type}",
                [np.ones((prm["n"],))]
            )
            np.save(
                prof_path / "norm_car" / f"c{k}_{day_type}",
                [np.zeros((prm["n"],))]
            )

    path = save_path / "clusters"
    for var, label in zip([p_clus, p_trans, min_cdfs, max_cdfs],
                          ["p_clus", "p_trans", "min_cdfs", "max_cdfs"]):
        if label == 'min_cdfs':
            print(f"save min_cdfs {min_cdfs}")
        with open(path / f"{label}.pickle", "wb") as file:
            pickle.dump(var, file)

    with open(path / "n_clus.pickle", "wb") as file:
        pickle.dump(prm["n_clus"], file)


def split_day_types(days, prm, n_data_type):
    """Split week day and weekend days."""
    n_day_type = initialise_dict(prm["behaviour_types"], "empty_dict")
    for data_type in prm["behaviour_types"]:
        for day_type in prm["weekday_type"]:
            days[f"{data_type}_{day_type}"] = []
        for i in range(n_data_type[data_type]):
            day = days[data_type][i]
            if data_type == "car":
                day_number = day["weekday"]
            else:
                delta_days = int(day["cum_day"] - 1)
                day_number = (
                    datetime(2010, 1, 1)
                    + timedelta(delta_days)
                ).weekday()

            day_type = "wd" if day_number < 5 else "we"
            day["day_type"] = day_type
            day["index_wdt"] = len(days[f"{data_type}_{day_type}"]) - 1

            days[f"{data_type}_{day_type}"].append(day)

        # obtain length of bank for each day type
        for day_type in prm["weekday_type"]:
            n_day_type[data_type][day_type] = len(
                days[f"{data_type}_{day_type}"])

    return days, n_day_type


def clustering(days, prm, n_data_type):
    """Cluster the data profiles in behaviour groups."""
    done_clustering = {
        data_type: {day_type: False for day_type in prm["weekday_type"]}
        for data_type in prm["behaviour_types"]
    }
    days, n_day_type = split_day_types(days, prm, n_data_type)
    p_clus, n_zeros, n_clus_all, banks, min_cdfs, max_cdfs = _initialise_cluster_dicts(prm)
    for n_consecutive_days in [3, 2]:
        print(f"clustering with {n_consecutive_days} consecutive days")
        n_trans, p_trans, banks = _initialise_cluster_transition_dicts(
            prm, n_consecutive_days, banks
        )
        enough_data = {}

        # n_clus_all includes zeros
        for data_type in prm["behaviour_types"]:
            print(f"data_type {data_type}")
            enough_data[data_type] = True
            for day_type in prm["weekday_type"]:
                print(f"day_type {day_type}")
                days_ = days[f"{data_type}_{day_type}"]
                if not done_clustering[data_type][day_type]:
                    transformed_features, to_cluster, i_zeros, norm_vals = _get_features(
                        days_, data_type, prm
                    )
                    if len(transformed_features) < 2:
                        print(f"len(transformed_features) {len(transformed_features)} "
                              f"data_type {data_type} "
                              f"day_type {day_type} insufficient for clustering")
                        banks[data_type][day_type] = None
                        enough_data[data_type] = False
                        continue
                    if prm["plots"]:
                        _elbow_method(
                            transformed_features, data_type, day_type, prm
                        )
                    [
                        labels, cluster_distance, n_zeros_, days[f"{data_type}_{day_type}"]
                    ] = _cluster_module(
                        transformed_features, prm["n_clus"][data_type],
                        to_cluster, days_, i_zeros
                    )
                    vals_k, idx_k_clustered = _get_vals_k(
                        labels, norm_vals,
                        prm["n_clus"][data_type]
                    )
                    done_clustering[data_type][day_type] = True
                    banks_, n_clus_all[data_type] = _get_p_clus(
                        vals_k, n_day_type[data_type][day_type],
                        data_type, n_zeros_
                    )
                    banks_ = _get_profs(
                        to_cluster, banks_, days_, data_type,
                        idx_k_clustered, cluster_distance
                    )
                    distances = [bank_["dists"] for bank_ in banks_.values()]
                    if data_type == "car":
                        distances = distances[:-1]
                    [
                        banks_, min_cdfs[data_type][day_type], max_cdfs[data_type][day_type]
                    ] = _get_cdfs(distances, f"{data_type} {day_type}", prm, banks_)
                    statistical_indicators = _plot_clusters(
                        transformed_features, norm_vals, data_type,
                        day_type, banks_, vals_k, prm, prm["save_other"]
                    )
                    if prm['gan_generation_profiles']:
                        for k in range(prm["n_clus"][data_type]):
                            print(f"k {k}")
                            compute_profile_generators(
                                vals_k[k], prm["n"], k, statistical_indicators,
                                data_type, prm['save_other'], prm
                            )

                    banks[data_type][day_type] = banks_
                    n_zeros[data_type][day_type] = n_zeros_

            # obtain probabilities transition between day types and clusters
            if not enough_data[data_type]:
                print(f"not enough data {data_type}")
                continue

            p_clus, p_trans, n_trans, banks = _transition_probabilities(
                data_type, days, p_clus, n_trans, p_trans,
                banks, n_clus_all[data_type], prm, n_data_type, n_consecutive_days
            )
    if "gen" in prm["data_types"]:
        banks["gen"], min_cdfs["gen"], max_cdfs["gen"] = _group_gen_month(days["gen"], prm)
        for n_consecutive_days in [3, 2]:
            banks, _ = _get_n_trans(
                n_data_type, 'gen', days, n_trans, banks, n_consecutive_days
            )

    # transitions probabilities
    _save_clustering(
        prm, prm["save_hedge"], banks, p_clus, p_trans, min_cdfs, max_cdfs
    )
    if len(banks['loads']['wd2wd'][f'f0of{n_consecutive_days}']) == 0:
        print(f"len(banks['loads']['wd2wd'][f0of{n_consecutive_days}]) == 0 in clustering")

    return banks
