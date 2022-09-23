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
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils import initialise_dict


def _get_n_trans(n_data_type, data_type, days, n_trans, banks):
    for i in range(n_data_type[data_type] - 1):
        day, next_day = [days[data_type][i_] for i_ in [i, i + 1]]
        same_id = day["id"] == next_day["id"]
        subsequent_days = day["cum_day"] + 1 == next_day["cum_day"]
        if same_id and subsequent_days:  # record transition
            day_types, index_wdt = [
                [day_[key] for day_ in [day, next_day]]
                for key in ["day_type", "index_wdt"]
            ]
            transition = f"{day_types[0]}2{day_types[1]}"
            clusters = [days[f"{data_type}_{day_types[i]}"][
                        index_wdt[i]]["cluster"]
                        for i in range(2)]
            n_trans[data_type][transition][clusters[0], clusters[1]] += 1
            for i, day_ in enumerate([day, next_day]):
                banks[data_type][transition][f"f{i}"].append(
                    day_["factor"])

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
) -> Tuple[Dict[str, Dict[str, float]],
           Dict[str, Dict[str, np.ndarray]],
           Dict[str, Dict[str, np.ndarray]],
           Dict[str, dict]]:
    """Get probabilities of transtioning between day types and clusters."""
    for transition in prm["day_trans"]:
        n_trans[data_type][transition] = np.zeros((n_clus_all_, n_clus_all_))
        p_trans[data_type][transition] = np.zeros((n_clus_all_, n_clus_all_))

    banks, n_trans = _get_n_trans(
        n_data_type, data_type, days, n_trans, banks
    )

    for c0 in range(n_clus_all_):
        for c1 in range(n_clus_all_):
            for transition in prm["day_trans"]:
                p_trans[data_type][transition][c0, c1] = (
                    n_trans[data_type][transition][c0, c1]
                    / sum(n_trans[data_type][transition][c0])
                )

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
                p_trans, transition, data_type, v_min, v_max, prm["save_path"]
            )

        # plot once with the colour bar
        _plot_heat_map_p_trans(
            p_trans, transition, data_type, v_min,
            v_max, prm["save_path"], colourbar=True
        )

    p_clus[data_type] = {}
    for day_type in prm["weekday_type"]:
        pcluss = [
            banks[data_type][day_type][k]["p_clus"] for k in range(n_clus_all_)
        ]
        p_clus[data_type][day_type] = pcluss

    return p_clus, p_trans, n_trans, banks


def _plot_heat_map_p_trans(
        p_trans, transition, data_type, v_min,
        v_max, save_path, colourbar=False
):
    fig = plt.figure()
    sns.heatmap(
        p_trans[data_type][transition], vmin=v_min, vmax=v_max,
        cmap="RdBu_r", yticklabels=False,
        xticklabels=False, cbar=colourbar, square=True
    )
    plt.tick_params(left=False, bottom=False)
    plt.tight_layout()
    fig.savefig(
        save_path / "clusters" / f"p_trans_heatmap_{data_type}_{transition}"
    )
    plt.gca().tick_params(left=False, bottom=False)
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
    for k in range(prm["n_clusters"][data_type]):
        if bank[k]["n_clus"] != 0:
            d10, d50, d90, mean = [[0] * prm["n"] for _ in range(4)]
            for t in range(prm["n"]):
                d10[t] = np.percentile(vals_k[k][:, t], 10)
                d90[t] = np.percentile(vals_k[k][:, t], 90)
                d50[t] = np.percentile(vals_k[k][:, t], 50)
                mean[t] = np.mean(vals_k[k][:, t])
            if prm["plots"]:
                for stylised in [True, False]:
                    fig = plt.figure()
                    plt.fill_between(
                        xs,
                        d90,
                        color="blue",
                        alpha=0.1,
                        label="10th to 90th percentile",
                    )
                    plt.fill_between(xs, d10, color="w")
                    if stylised:
                        title = f"Cluster {k} demand {day_type} stylised"
                        plt.plot(xs, mean, color="blue", label="mean", lw=3)
                        plt.tight_layout()
                        plt.axis('off')
                    else:
                        plt.plot(xs, d50, color="red", label="median")
                        plt.plot(xs, mean, color="blue", label="mean")
                        plt.legend()
                        title = f"Cluster {k} demand {day_type}"
                        plt.title(title)
                    fig.savefig(
                        save_path / "clusters" / title.replace(" ", "_")
                    )
                plt.close("all")


def _get_vals_k(labels, norm_vals, n_clusters):
    vals_k, idx_k_clustered = {}, {}
    for k in range(n_clusters):
        idx_k_clustered[k] \
            = [i for i, label in enumerate(labels) if label == k]
        vals_k[k] = np.array([norm_vals[i] for i in idx_k_clustered[k]])

    return vals_k, idx_k_clustered


def _get_cdfs(distances, label, save_path, bank, plots):
    min_cdfs_, max_cdfs_ = [], []

    for i, distance in enumerate(distances):
        if distance is None:
            bank[i]["cdfs"] = None
            continue

        # plot histogram distances around the cluster
        if plots:
            fig, ax1 = plt.subplots()
            plt.hist(distance, 100,
                     density=True, facecolor='k', alpha=0.75, label='data')
            plt.ylabel('Probability', color='k')
            ax2 = ax1.twinx()
            ax2.hist(distance, 100, density=True, cumulative=True, label='CDF',
                     histtype='step', alpha=0.8, color='r')
            plt.ylabel('Cumulative probability', color='r')
            plt.xlabel('Distances to cluster')
            title = f"Histogram of distance to the cluster centers {label} {i}"
            plt.title(title)
            plt.grid(True)
            fig.savefig(save_path / "clusters" / title.replace(" ", "_"))
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
    n_clusters = max(list(vals_k.keys())) + 1
    n_clus_all = (
        n_clusters
        if data_type == "dem"
        else n_clusters + 1
    )
    bank = initialise_dict(range(n_clus_all), "empty_dict")

    for k, val_k in vals_k.items():
        bank[k]["n_clus"] = len(val_k)
        bank[k]["p_clus"] = len(val_k) / n_days
    if data_type == 'EV':
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
        if data_type == 'EV' and k == n_k - 1:
            bank_["profs"] = [[0 for _ in range(n_steps)]]
            bank_["dists"] = [[0 for _ in range(n_steps)]]
        else:
            idx_k_all = [to_cluster[i] for i in idx_k_clustered[k]]
            bank_["profs"] = [days_[i][f"norm_{data_type}"] for i in idx_k_all]
            assert all(
                [
                    sum(prof) == 0 or abs(sum(prof) - 1) < 1e-3
                    for prof in bank_["profs"]
                ]
            ), f"{data_type} normalised profile should sum to 1"

            if data_type == "EV":
                bank_["avail"] = [days_[i]["avail"] for i in idx_k_all]
            bank_["dists"] \
                = [cluster_distance[i][k] for i in idx_k_clustered[k]]

    return bank


def _cluster_module(
        transformed_features: list,
        n_clusters: int,
        to_cluster: list,
        days_: list,
        i_zeros: list,
) -> Tuple[list, list, int, list]:
    # actual clustering
    clusobj = KMeans(n_clusters=n_clusters, n_init=100)
    obj = clusobj.fit(transformed_features)

    n_zeros = len(days_) - len(to_cluster)

    # obtain main centroids of each cluster
    # centers_unordered = obj.cluster_centers_
    # centers_alloc = obj.fit_predict(centers_unordered)
    # centers = [centers_unordered[np.where(centers_alloc == c)[0][0]]
    #            for c in range(prm["n_clusters"][data_type])]
    labels = obj.fit_predict(transformed_features)
    for label, i in zip(labels, to_cluster):
        days_[i]["cluster"] = label
    for i in i_zeros:
        days_[i]["cluster"] = n_clusters

    # obtain distances to centroid for each profile
    # -> ndarray of shape (n_samples, n_clusters[dt])
    # KMeans.transform() returns an array of distances
    # of each sample to the cluster center
    cluster_distances = obj.transform(transformed_features)

    return labels, cluster_distances, n_zeros, days_


def _elbow_method(transformed_features, data_type, day_type, save_path):
    # elbow method
    if not os.path.exists(f"Elbow_method_{data_type}_{day_type}.npy"):
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

        fig = plt.figure()
        plt.plot(range(1, maxn_clus), wcss)
        title = f"Elbow Method {data_type} {day_type}"
        plt.title(title)
        plt.xlabel("Number of clusters")
        plt.ylabel("WCSS")
        fig.savefig(save_path / "clusters" / title.replace(" ", "_"))
        plt.close("all")


def _get_features(days_, data_type, prm):
    # prepare data features
    # only cluster if positive values
    to_cluster = [
        i
        for i in range(len(days_))
        if days_[i][data_type] is not None and sum(days_[i][data_type]) > 0
    ]
    if data_type == "EV":
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
        if data_type == "dem":
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

        elif data_type == "EV":
            features.append(
                days_[i][data_type][
                    int(6 * 60 / prm["dT"]):
                    int(22 * 60 / prm["dT"])]
            )
        norm_vals.append(days_[i][f"norm_{data_type}"])
    transformed_features = StandardScaler().fit_transform(features)

    return transformed_features, to_cluster, i_zeros, norm_vals


def _initialise_cluster_dicts(prm):
    """Initialise dictionaries for storing cluster info."""
    n_trans, p_trans, p_clus, n_zeros, n_clus_all, banks = [
        initialise_dict(["dem", "EV"], "empty_dict") for _ in range(6)
    ]

    for data_type in ["dem", "EV"]:
        n_trans[data_type], p_trans[data_type] = [
            initialise_dict(prm["day_trans"]) for _ in range(2)
        ]
        p_clus[data_type], n_zeros[data_type] = [
            initialise_dict(prm["weekday_type"]) for _ in range(2)
        ]
        banks[data_type] = initialise_dict(
            prm["weekday_type"] + prm["day_trans"])
        for transition in prm["day_trans"]:
            banks[data_type][transition] = initialise_dict(
                [f"f{i}" for i in range(2)]
            )

    return n_trans, p_trans, p_clus, n_zeros, n_clus_all, banks


def _group_gen_month(days_, save_path, plots):
    bank = initialise_dict(range(12), "empty_dict")
    cluster_distances = []
    for i_month, month in enumerate(range(1, 13)):
        i_days = [i for i, day in enumerate(days_) if day["month"] == month]
        bank[i_month]["profs"] = [days_[i]["norm_gen"] for i in i_days]
        for property_ in ["factor", "id", "cum_day"]:
            bank[i_month][property_] = [days_[i][property_] for i in i_days]

        path = save_path / "profiles" / "norm_gen"
        np.save(
            path / f"i_month{i_month}",
            bank[i_month]["profs"],
        )

        if len(bank[i_month]["profs"]) > 0:
            transformed_features \
                = StandardScaler().fit_transform(bank[i_month]["profs"])
            clusobj = KMeans(n_clusters=1, n_init=100)
            obj = clusobj.fit(transformed_features)
            cluster_distances.append(obj.transform(transformed_features))
        else:
            cluster_distances.append(None)

    bank, min_cdfs, max_cdfs \
        = _get_cdfs(cluster_distances, "gen", save_path, bank, plots)

    for i_month in range(12):
        np.save(
            save_path / "clusters"
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
            for k in range(prm["n_clusters"][data_type]):
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
                if data_type == "EV":
                    path = prof_path / "EV_avail"
                    np.save(
                        path / f"c{k}_{day_type}",
                        banks[data_type][day_type][k]["avail"],
                    )

    if "EV" in prm["data_types"]:
        k = prm["n_clusters"]["EV"]
        for day_type in prm["weekday_type"]:
            np.save(
                prof_path / "EV_avail" / f"c{k}_{day_type}",
                [np.ones((prm["n"],))]
            )
            np.save(
                prof_path / "norm_EV" / f"c{k}_{day_type}",
                [np.zeros((prm["n"],))]
            )

    path = save_path / "clusters"
    for var, label in zip([p_clus, p_trans, min_cdfs, max_cdfs],
                          ["p_clus", "p_trans", "min_cdfs", "max_cdfs"]):
        with open(path / f"{label}.pickle", "wb") as file:
            pickle.dump(var, file)

    with open(path / "n_clusters.pickle", "wb") as file:
        pickle.dump(prm["n_clusters"], file)


def split_day_types(days, prm, n_data_type):
    """Split week day and weekend days."""
    n_day_type = initialise_dict(prm["behaviour_types"], "empty_dict")
    for data_type in prm["behaviour_types"]:
        for day_type in prm["weekday_type"]:
            days[f"{data_type}_{day_type}"] = []
        for i in range(n_data_type[data_type]):
            day = days[data_type][i]
            if data_type == "EV":
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
    days, n_day_type = split_day_types(days, prm, n_data_type)

    (
        n_trans,
        p_trans,
        p_clus,
        n_zeros,
        n_clus_all,
        banks,
    ) = _initialise_cluster_dicts(prm)

    enough_data = {}
    min_cdfs, max_cdfs = [initialise_dict(prm["data_types"], "empty_dict")
                          for _ in range(2)]
    # n_clus_all includes zeros
    for data_type in prm["behaviour_types"]:
        enough_data[data_type] = True
        for day_type in prm["weekday_type"]:
            days_ = days[f"{data_type}_{day_type}"]
            transformed_features, to_cluster, i_zeros, norm_vals \
                = _get_features(days_, data_type, prm)
            if len(transformed_features) < 2:
                print(f"len(transformed_features) {len(transformed_features)} "
                      f"data_type {data_type} "
                      f"day_type {day_type} insufficient for clustering")
                banks[data_type][day_type] = None
                enough_data[data_type] = False
                continue
            if prm["plots"]:
                _elbow_method(
                    transformed_features, data_type, day_type, prm["save_path"]
                )
            [labels, cluster_distance, n_zeros_,
             days[f"{data_type}_{day_type}"]] \
                = _cluster_module(
                transformed_features, prm["n_clusters"][data_type],
                to_cluster, days_, i_zeros)
            vals_k, idx_k_clustered \
                = _get_vals_k(labels, norm_vals,
                              prm["n_clusters"][data_type])
            banks_, n_clus_all[data_type] = _get_p_clus(
                vals_k, n_day_type[data_type][day_type],
                data_type, n_zeros_)
            banks_ = _get_profs(
                to_cluster, banks_, days_, data_type,
                idx_k_clustered, cluster_distance)
            distances = [bank_["dists"] for bank_ in banks_.values()]
            if data_type == "EV":
                distances = distances[:-1]
            [banks_,
             min_cdfs[data_type][day_type],
             max_cdfs[data_type][day_type]] = _get_cdfs(
                distances, f"{data_type} {day_type}",
                prm["save_path"], banks_, prm["plots"]
            )
            _plot_clusters(transformed_features, norm_vals, data_type,
                           day_type, banks_, vals_k, prm, prm["save_path"])
            banks[data_type][day_type] = banks_
            n_zeros[data_type][day_type] = n_zeros_
        # obtain probabilities transition between day types and clusters
        if not enough_data[data_type]:
            print(f"not enough data {data_type}")
            continue
        p_clus, p_trans, n_trans, banks = _transition_probabilities(
            data_type, days, p_clus, n_trans, p_trans,
            banks, n_clus_all[data_type], prm, n_data_type,
        )

    if "gen" in prm["data_types"]:
        banks["gen"], min_cdfs["gen"], max_cdfs["gen"] \
            = _group_gen_month(days["gen"], prm["save_path"], prm["plots"])

    # transitions probabilities
    _save_clustering(
        prm, prm["save_path"], banks, p_clus, p_trans, min_cdfs, max_cdfs
    )

    return banks
