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
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.data_preparation.profile_generation import compute_profile_generators
from src.utils import initialise_dict, save_fig


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
        fig_save_path = prm["save_other"] / "clusters" / f"p_trans_heatmap_{data_type}_{transition}"
        save_fig(fig, prm, fig_save_path)
        plt.close("all")


def _get_vals_k(labels, norm_vals, n_clus, ev_avail=None):
    vals_k, idx_k_clustered, ev_avail_k = {}, {}, {}
    for k in range(n_clus):
        idx_k_clustered[k] = [i for i, label in enumerate(labels) if label == k]
        vals_k[k] = np.array([norm_vals[i] for i in idx_k_clustered[k]])
        if len(ev_avail) > 0:
            ev_avail_k[k] = np.array([ev_avail[i] for i in idx_k_clustered[k]])

    return vals_k, idx_k_clustered, ev_avail_k


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


class Clusterer:
    def __init__(self, prm):
        for info in ['behaviour_types', 'data_types', 'weekday_types', 'day_trans', 'n']:
            setattr(self, info, prm[info])
        setattr(self, 'prm', prm)
        self.done_clustering = {
            data_type: {day_type: False for day_type in self.weekday_types}
            for data_type in self.behaviour_types
        }
        self.n_day_type = {data_type: {} for data_type in self.behaviour_types}
        self._initialise_cluster_dicts()

    def _split_day_types(self, days, n_data_type):
        """Split week day and weekend days."""
        for data_type in self.behaviour_types:
            for day_type in self.weekday_types:
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
            for day_type in self.weekday_types:
                self.n_day_type[data_type][day_type] = len(days[f"{data_type}_{day_type}"])

        return days

    def _initialise_cluster_dicts(self):
        """Initialise dictionaries for storing cluster info."""
        for info in ['p_clus', 'n_zeros', 'n_clus_all', 'banks']:
            setattr(self, info, {data_type: {} for data_type in ["loads", "car"]})

        for data_type in ["loads", "car"]:
            self.p_clus[data_type], self.n_zeros[data_type] = [
                {weekday_type: [] for weekday_type in self.weekday_types}
                for _ in range(2)
            ]
            self.banks[data_type] = {
                day_type: {} for day_type in self.weekday_types + self.day_trans
            }

        for info in [
            'min_cdfs', 'max_cdfs', 'clus_dist_bin_edges', 'clus_dist_cdfs',
            'fitted_kmeans_obj', 'fitted_scalers'
        ]:
            setattr(self, info, {data_type: {} for data_type in self.data_types})

    def _initialise_cluster_transition_dicts(self, n_consecutive_days):
        """Initialise dictionaries for storing cluster info."""
        for info in ['n_trans', 'p_trans']:
            setattr(self, info, {data_type: {} for data_type in ["loads", "car"]})

            for data_type in self.behaviour_types:
                self.__dict__[info][data_type] = {day_trans: [] for day_trans in self.day_trans}

                for transition in self.day_trans:
                    for i in range(n_consecutive_days):
                        self.banks[data_type][transition][f"f{i}of{n_consecutive_days}"] = []

    def _cluster_data_and_day_type(self, data_type, day_type, days):
        days_ = days[f"{data_type}_{day_type}"]
        [
            transformed_features, to_cluster, i_zeros, norm_vals, ev_avail
        ] = self._get_features(days_, data_type, day_type)
        self.n_zeros[data_type][day_type] = len(days_) - len(to_cluster)

        if len(transformed_features) < 2:
            print(f"len(transformed_features) {len(transformed_features)} "
                  f"data_type {data_type} "
                  f"day_type {day_type} insufficient for clustering")
            self.banks[data_type][day_type] = None
            self.enough_data[data_type] = False

        if self.prm["plots"]:
            self._elbow_method(transformed_features, data_type, day_type)

        labels, cluster_distance, days[f"{data_type}_{day_type}"] = self._cluster_module(
            transformed_features, to_cluster, days_, i_zeros, data_type, day_type
        )
        vals_k, idx_k_clustered, ev_avail_k = _get_vals_k(
            labels, norm_vals, self.prm["n_clus"][data_type], ev_avail
        )
        self.done_clustering[data_type][day_type] = True
        banks_ = self._get_p_clus(vals_k, day_type, data_type, self.n_zeros[data_type][day_type])
        banks_ = _get_profs(
            to_cluster, banks_, days_, data_type,
            idx_k_clustered, cluster_distance
        )
        distances = [bank_["dists"] for bank_ in banks_.values()]
        if data_type == "car":
            distances = distances[:-1]
        [
            banks_, self.min_cdfs[data_type][day_type], self.max_cdfs[data_type][day_type],
            self.clus_dist_bin_edges[data_type][day_type], self.clus_dist_cdfs[data_type][day_type]
        ] = self._get_cdfs(distances, f"{data_type} {day_type}", banks_)
        statistical_indicators = self._plot_clusters(
            transformed_features, norm_vals, data_type, day_type, banks_, vals_k
        )
        if self.prm['gan_generation_profiles']:
            self._generate_gan_profiles_behaviour_type(
                day_type, data_type, ev_avail_k, vals_k, statistical_indicators
            )
        self.banks[data_type][day_type] = banks_

    def _get_features(self, days_, data_type, day_type):
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
        ev_avail = []
        for i in to_cluster:
            norm_day = days_[i][f"norm_{data_type}"]
            if data_type == "loads":
                peak = np.max(norm_day)
                t_peak = np.argmax(norm_day)
                values = [
                    np.mean(
                        norm_day[
                            int(interval[0] * self.n / 24):
                            int(interval[1] * self.n / 24)
                        ]
                    )
                    for interval in self.prm["dem_intervals"]
                ]
                if len(norm_day) != self.n:
                    print(f"len(to_cluster) " f"= {len(to_cluster)}")
                features.append([peak, t_peak] + values)

            elif data_type == "car":
                start_idx, end_idx = [int(hour * 60 / self.prm["step_len"]) for hour in [6, 22]]
                features.append(days_[i][data_type][start_idx: end_idx])
                ev_avail.append(days_[i]["avail"])
            norm_vals.append(days_[i][f"norm_{data_type}"])
        fitted_scaler = StandardScaler().fit(features)
        transformed_features = fitted_scaler.transform(features)

        self.fitted_scalers[data_type][day_type] = fitted_scaler

        return transformed_features, to_cluster, i_zeros, norm_vals, ev_avail

    def _elbow_method(self, transformed_features, data_type, day_type):
        # elbow method
        file = self.prm['save_other'] / "clusters" / f"Elbow_method_{data_type}_{day_type}.npy"
        if not file.is_file():
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
            if self.prm['plots']:
                fig = plt.figure()
                plt.plot(range(1, maxn_clus), wcss)
                title = f"Elbow Method {data_type} {day_type}"
                plt.title(title)
                plt.xlabel("Number of clusters")
                plt.ylabel("WCSS")
                fig_save_path = self.prm['save_other'] / "clusters" / title.replace(" ", "_")
                save_fig(fig, self.prm, fig_save_path)
                plt.close("all")

    def _get_cdfs(self, distances, label, bank):
        min_cdfs_, max_cdfs_, bins_edges_, cdfs_ = [[] for _ in range(4)]

        for i, distance in enumerate(distances):
            if distance is None:
                bank[i]["cdfs"] = None
                continue

            # plot histogram distances around the cluster
            if self.prm["plots"]:
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
                fig_save_path = self.prm["save_other"] / "clusters" / title.replace(" ", "_")
                save_fig(fig, self.prm, fig_save_path)
                plt.close("all")

            # get cumulative probability for each profile
            # getting data of the histogram
            count, bins_edges = np.histogram(distance, bins=1000)
            bins_edges_.append(bins_edges)
            # finding the PDF of the histogram using count values
            pdf = count / sum(count)
            # finding the CDF of the histogram
            cdfs = np.cumsum(pdf)
            cdfs_.append(cdfs)
            # matching each distance data point to its bin probability
            all_cdfs = [cdfs[np.where(dist >= bins_edges[:-1])[0][-1]] for dist in distance]
            bank[i]["cdfs"] = all_cdfs
            min_cdfs_.append(np.min(all_cdfs))
            max_cdfs_.append(np.max(all_cdfs))

        return bank, min_cdfs_, max_cdfs_, bins_edges_, cdfs_

    def _get_percentiles(self, data):
        # data should be [n_clus, n_profiles, n_steps]
        n_clus = len(data)
        statistical_indicators = {k: {} for k in range(n_clus)}
        for k in range(n_clus):
            if len(data[k]) > 0:
                print(f"k {k} len(data[k]) = {len(data[k])}")
                statistical_indicators[k]['min'] = np.min(data[k])
                statistical_indicators[k]['max'] = np.max(data[k])
                for statistical_indicator in ['p10', 'p25', 'p50', 'p75', 'p90', 'mean']:
                    statistical_indicators[k][statistical_indicator] = np.zeros(self.n)
                for time in range(self.n):
                    for percentile in [10, 25, 50, 75, 90]:
                        statistical_indicators[k][f'p{percentile}'][time] = np.percentile(
                            data[k][:, time], percentile
                        )
                    statistical_indicators[k]['mean'][time] = np.mean(data[k][:, time])
                fig = plt.figure()
                plt.plot(statistical_indicators[k]['mean'], label='mean')
                fig.savefig(self.prm['save_other'] / 'clusters' / f'mean_{k}.png')
                plt.close('all')
            else:
                print(f"Cluster {k} is empty")

        return statistical_indicators

    def _plot_clusters(
            self,
            transformed_features,
            norm_vals,
            data_type,
            day_type,
            bank,
            vals_k,
    ):
        # plot clusters
        xs = np.arange(0, self.n)
        if np.shape(norm_vals) != (len(transformed_features), self.n):
            print(
                f"check np.shape(norm_vals) = {np.shape(norm_vals)} "
                f"same as ({len(transformed_features)}, {self.n})"
            )
        statistical_indicators = self._get_percentiles(vals_k)

        ymax = np.max(
            [
                statistical_indicators[k]['p90'] for k in range(self.prm["n_clus"][data_type])
                if bank[k]["n_clus"] != 0
            ]
        )
        for k in range(self.prm['n_clus'][data_type]):
            if bank[k]['n_clus'] != 0 and self.prm["plots"]:
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
                        title = f"Cluster {k} {data_type} {day_type} stylised"
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
                    plt.ylim(0, ymax)
                    fig_save_path = self.prm["save_other"] / "clusters" / title.replace(" ", "_")
                    save_fig(fig, self.prm, fig_save_path)
                plt.close("all")

        return statistical_indicators

    def _group_gen_month(self, days_):
        self.banks['gen'] = initialise_dict(range(13), "empty_dict")
        self.fitted_kmeans_obj["gen"], self.fitted_scalers['gen'], cluster_distances = [], [], []
        for i_month, month in enumerate(range(1, 14)):
            if month == 13:
                i_days = range(len(days_))
            else:
                i_days = [i for i, day in enumerate(days_) if day["month"] == month]
            self.banks['gen'][i_month]["profs"] = np.array([days_[i]["norm_gen"] for i in i_days])
            self.banks['gen'][i_month]["gen"] = np.array([days_[i]["gen"] for i in i_days])
            for property_ in ["factor", "id", "cum_day"]:
                self.banks['gen'][i_month][property_] = [days_[i][property_] for i in i_days]

            path = self.prm["save_hedge"] / "profiles" / "norm_gen"
            np.save(
                path / f"i_month{i_month}",
                self.banks['gen'][i_month]["profs"],
            )

            if len(self.banks['gen'][i_month]["profs"]) > 0:
                start_idx, end_idx = [int(hour * 60 / self.prm["step_len"]) for hour in [6, 22]]
                features = self.banks['gen'][i_month]["profs"][:, start_idx: end_idx]
                fitted_scaler = StandardScaler().fit(features)
                self.fitted_scalers['gen'].append(fitted_scaler)
                transformed_features = fitted_scaler.transform(features)
                clusobj = KMeans(n_clusters=1, n_init=100, random_state=0)
                fitted_kmeans_obj = clusobj.fit(transformed_features)
                self.fitted_kmeans_obj["gen"].append(fitted_kmeans_obj)
                cluster_distances.append(fitted_kmeans_obj.transform(transformed_features))
            else:
                cluster_distances.append(None)

        [
            self.banks['gen'], self.min_cdfs["gen"], self.max_cdfs["gen"],
            self.clus_dist_bin_edges['gen'], self.clus_dist_cdfs['gen']
        ] = self._get_cdfs(cluster_distances, "gen", self.banks['gen'])

        for i_month in range(13):
            np.save(
                self.prm["save_hedge"] / "clusters"
                / f"cdfs_clus_gen_{i_month}",
                self.banks['gen'][i_month]["cdfs"],
            )

    def _transition_probabilities(
            self,
            data_type: str,
            days: List[dict],
            n_data_type: Dict[str, int],
            n_consecutive_days: int,
    ) -> Tuple[
        Dict[str, Dict[str, float]],
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, dict]
    ]:
        """Get probabilities of transtioning between day types and clusters."""
        n_clus_all_ = self.n_clus_all[data_type]
        for transition in self.day_trans:
            shape = tuple(n_clus_all_ for _ in range(n_consecutive_days))
            self.n_trans[data_type][transition] = np.zeros(shape)
            self.p_trans[data_type][transition] = np.zeros(shape)

        self._get_n_trans(
            n_data_type, data_type, days, n_consecutive_days
        )
        if data_type == 'loads' and len(self.banks['loads']['wd2wd'][f'f0of{n_consecutive_days}']) == 0:
            print(f"loads wd2wd f0of{n_consecutive_days}: len banks is 0 after _get_n_trans")

        for c0 in range(n_clus_all_):
            for c1 in range(n_clus_all_):
                if n_consecutive_days == 2:
                    for transition in self.day_trans:
                        self.p_trans[data_type][transition][c0, c1] = (
                                self.n_trans[data_type][transition][c0, c1]
                                / sum(self.n_trans[data_type][transition][c0])
                        ) if sum(self.n_trans[data_type][transition][c0]) > 0 else None
                elif n_consecutive_days == 3:
                    for c2 in range(n_clus_all_):
                        for transition in self.day_trans:
                            self.p_trans[data_type][transition][c0, c1, c2] = (
                                    self.n_trans[data_type][transition][c0, c1, c2]
                                    / sum(self.n_trans[data_type][transition][c0, c1])
                            ) if sum(self.n_trans[data_type][transition][c0, c1]) > 0 else None
                else:
                    print(f"implement n_consecutive_days = {n_consecutive_days}")

        with open(
                self.prm["save_hedge"]
                / "clusters"
                / f"{data_type}_p_trans_n_consecutive_days{n_consecutive_days}", "wb"
        ) as file:
            pickle.dump(self.p_trans, file)

        if n_consecutive_days == 2:
            v_max = np.max(
                [
                    np.max(self.p_trans[data_type][transition])
                    for transition in self.day_trans
                ]
            )
            v_min = np.min(
                [
                    np.min(self.p_trans[data_type][transition])
                    for transition in self.day_trans
                ]
            )

            if self.prm["plots"]:
                for transition in self.day_trans:
                    _plot_heat_map_p_trans(
                        self.p_trans, transition, data_type, v_min, v_max, self.prm
                    )

                # plot once with the colour bar
                _plot_heat_map_p_trans(
                    self.p_trans, transition, data_type, v_min,
                    v_max, self.prm, colourbar=True
                )

            self.p_clus[data_type] = {}
            for day_type in self.weekday_types:
                pcluss = [
                    self.banks[data_type][day_type][k]["p_clus"] for k in range(n_clus_all_)
                ]
                self.p_clus[data_type][day_type] = pcluss

    def _get_p_clus(self, vals_k, day_type, data_type, n_zeros):
        n_days = self.n_day_type[data_type][day_type]
        n_clus = max(list(vals_k.keys())) + 1
        self.n_clus_all[data_type] = (
            n_clus
            if data_type == "loads"
            else n_clus + 1
        )
        bank = initialise_dict(range(self.n_clus_all[data_type]), "empty_dict")

        for k, val_k in vals_k.items():
            bank[k]["n_clus"] = len(val_k)
            bank[k]["p_clus"] = len(val_k) / n_days
        if data_type == 'car':
            bank[self.n_clus_all[data_type] - 1]["n_clus"] = n_zeros
            bank[self.n_clus_all[data_type] - 1]["p_clus"] = n_zeros / n_days

        return bank

    def _save_clustering(self):
        save_path = self.prm["save_hedge"]
        prof_path = save_path / "profiles"
        for data_type in self.behaviour_types:
            for day_type in self.weekday_types:
                for k in range(self.prm["n_clus"][data_type]):
                    np.save(
                        save_path / "clusters"
                        / f"cdfs_clus_{data_type}_{day_type}_{k}",
                        self.banks[data_type][day_type][k]["cdfs"],
                    )
                    path = prof_path / f"norm_{data_type}"
                    np.save(
                        path / f"c{k}_{day_type}",
                        self.banks[data_type][day_type][k]["profs"],
                    )

                    assert not np.all(
                        np.array(self.banks[data_type][day_type][k]["profs"]) == 0
                    ), f"{data_type} {k} {day_type} all zeros"
                    if data_type == "car":
                        path = prof_path / "car_avail"
                        np.save(
                            path / f"c{k}_{day_type}",
                            self.banks[data_type][day_type][k]["avail"],
                        )

        if "car" in self.data_types:
            k = self.prm["n_clus"]["car"]
            for day_type in self.weekday_types:
                np.save(
                    prof_path / "car_avail" / f"c{k}_{day_type}",
                    [np.ones((self.n,))]
                )
                np.save(
                    prof_path / "norm_car" / f"c{k}_{day_type}",
                    [np.zeros((self.n,))]
                )

        path = save_path / "clusters"
        for info in [
            "p_clus", "p_trans", "min_cdfs", "max_cdfs", "clus_dist_bin_edges", "clus_dist_cdfs",
            "fitted_kmeans_obj", "fitted_scalers"
        ]:
            with open(path / f"{info}.pickle", "wb") as file:
                pickle.dump(getattr(self, info), file)

        with open(path / "n_clus.pickle", "wb") as file:
            pickle.dump(self.prm["n_clus"], file)

    def _get_n_trans(self, n_data_type, data_type, days, n_consecutive_days=2):
        if data_type == 'gen':
            for month in range(12):
                for i in range(n_consecutive_days):
                    self.banks[data_type][month][f"f{i}of{n_consecutive_days}"] = []

        for i in range(n_data_type[data_type] - (n_consecutive_days - 1)):
            consecutive_days = [days[data_type][d] for d in range(i, i + n_consecutive_days)]
            same_id = all(
                consecutive_days[d]['id'] == consecutive_days[0]["id"]
                for d in range(n_consecutive_days)
            )
            subsequent_days = all(
                consecutive_days[d]["cum_day"] == consecutive_days[0]["cum_day"] + d
                for d in range(n_consecutive_days)
            )
            if same_id and subsequent_days:  # record transition
                if data_type == 'gen':
                    i_month = consecutive_days[0]['month'] - 1
                    for i, day_ in enumerate(consecutive_days):
                        if day_["factor"] < 0:
                            print(f"gen {i} Negative factor: {day_['factor']}")
                        self.banks[data_type][i_month][f"f{i}of{n_consecutive_days}"].append(
                            day_["factor"]
                        )
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
                    self.n_trans[data_type][transition][idx] += 1
                    for i, day_ in enumerate(consecutive_days):
                        self.banks[data_type][transition][f"f{i}of{n_consecutive_days}"].append(
                            day_["factor"]
                        )

    def _generate_gan_profiles_behaviour_type(
            self, day_type, data_type, ev_avail_k, vals_k, statistical_indicators
    ):
        for k in range(self.prm["n_clus"][data_type]):
            print(f"k {k}")
            if data_type == 'car':
                percentage_car_avail = np.sum(ev_avail_k[k]) / np.multiply(*np.shape(ev_avail_k[k]))
                average_non_zero_trip = np.mean(vals_k[k][vals_k[k] > 0])
                print(f"k {k} % car available = {percentage_car_avail}")
                print(f"average trip non zero {average_non_zero_trip}")
            else:
                percentage_car_avail, average_non_zero_trip = None, None
            compute_profile_generators(
                vals_k[k], k, statistical_indicators,
                data_type, day_type, self.prm,
                percentage_car_avail, average_non_zero_trip
            )

    def _cluster_module(
            self,
            transformed_features: list,
            to_cluster: list,
            days_: list,
            i_zeros: list,
            data_type,
            day_type,
    ) -> Tuple[list, list, int, list]:
        # actual clustering
        n_clus = self.prm["n_clus"][data_type]
        clusobj = KMeans(n_clusters=n_clus, n_init=100, random_state=0)
        fitted_kmeans_obj = clusobj.fit(transformed_features)

        labels = fitted_kmeans_obj.fit_predict(transformed_features)
        for label, i in zip(labels, to_cluster):
            days_[i]["cluster"] = label
        for i in i_zeros:
            days_[i]["cluster"] = n_clus

        # obtain distances to centroid for each profile
        # -> ndarray of shape (n_samples, n_clus[data_type])
        # KMeans.transform() returns an array of distances
        # of each sample to the cluster center
        cluster_distances = fitted_kmeans_obj.transform(transformed_features)
        self.fitted_kmeans_obj[data_type][day_type] = fitted_kmeans_obj

        return labels, cluster_distances, days_

    def clustering(self, days, n_data_type):
        """Cluster the data profiles in behaviour groups."""
        days = self._split_day_types(days, n_data_type)
        for n_consecutive_days in self.prm['n_consecutive_days']:
            print(f"clustering with {n_consecutive_days} consecutive days")
            self._initialise_cluster_transition_dicts(n_consecutive_days)
            self.enough_data = {}
            for data_type in self.behaviour_types:
                print(f"data_type {data_type}")
                self.enough_data[data_type] = True
                for day_type in self.weekday_types:
                    print(f"day_type {day_type}")
                    if not self.done_clustering[data_type][day_type]:
                        self._cluster_data_and_day_type(data_type, day_type, days)

                # obtain probabilities transition between day types and clusters
                if not self.enough_data[data_type]:
                    print(f"not enough data {data_type}")
                    continue

                self._transition_probabilities(
                    data_type, days, n_data_type, n_consecutive_days
                )

        if "gen" in self.data_types:
            self._group_gen_month(days["gen"])
            for n_consecutive_days in self.prm['n_consecutive_days']:
                self._get_n_trans(n_data_type, 'gen', days, n_consecutive_days)
            vals_k = {
                i_month: np.array(self.banks["gen"][i_month]['profs']) for i_month in range(13)
            }
            for i_month in [12]:
                print(f"i_month {i_month} {np.shape(vals_k[i_month])}")
                fig = plt.figure()
                for i in range(min(len(vals_k[i_month]), int(1e3))):
                    plt.plot(vals_k[i_month][i, :], color="grey", alpha=0.1)
                plt.title(f"i_month {i_month}")
                fig.savefig(f"gen_{i_month}.png")
                plt.close(fig)
            statistical_indicators = self._get_percentiles(vals_k)
            if self.prm['gan_generation_profiles']:
                for i_month in [12]:
                    compute_profile_generators(
                        self.banks["gen"][i_month]["profs"], i_month, statistical_indicators,
                        'gen', '', self.prm
                    )

        # transitions probabilities
        self._save_clustering()
        if 'loads' in self.data_types and len(self.banks['loads']['wd2wd'][f'f0of{n_consecutive_days}']) == 0:
            print(f"len(self.banks['loads']['wd2wd'][f0of{n_consecutive_days}]) == 0 in clustering")

        return self.banks
