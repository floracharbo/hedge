import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
from torch import nn
from tqdm import tqdm

from hedge import car_loads_to_availability
from utils import list_potential_paths, save_fig

th.manual_seed(111)


def checks_nan(real_inputs, real_outputs, generated_outputs):
    if any(any(th.isnan(real_inputs_i)) for real_inputs_i in real_inputs):
        print('nan in real_inputs')
        sys.exit()
    if any(any(th.isnan(real_outputs_i)) for real_outputs_i in real_outputs):
        print('nan in real_outputs')
        sys.exit()
    if any(any(th.isnan(generated_outputs_i)) for generated_outputs_i in generated_outputs):
        print('nan in generated_outputs')
        sys.exit()


class GAN_Trainer():
    def __init__(self, training_params, prm):
        for param, value in training_params.items():
            self.__dict__[param] = value
        if 'value_type' in training_params:
            self.update_value_type(self.value_type)
        self.prm = prm

    def update_value_type(self, value_type):
        save_folder = f"{self.data_type}_{value_type}_generation"
        self.value_type = value_type
        self.save_path = self.prm['save_other'] / save_folder
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.normalised = False

    def get_ideal_ev_avail_tol(self, generated_outputs):
        percentages_available = []
        potential_tols = [1e-4, 1e-3, 1e-2, 1e-1]
        for tol in potential_tols:
            percentages_available.append(
                self.car_output_to_percentage_availability(generated_outputs, tol)
            )
        ideal_tol = potential_tols[np.argmin(percentages_available)]
        print(f"ideal tol: {ideal_tol}")
        np.save(self.save_path / f"ideal_tol_{self.ext}.npy", ideal_tol)

    def get_saving_label(self):
        saving_label \
            = f"{self.data_type} {self.day_type} {self.k} {self.value_type} " \
              f"lr_start {self.lr_start:.2e} " \
              f"lr_discriminator_ratio {self.lr_discriminator_ratio:.2e} " \
              f"n_items_generated {self.n_items_generated} " \
              f"noise0{self.noise0:.2f}_noise_end{self.noise_end:.2f}".replace('.', '_')
        saving_label += f" cluster {self.k}"
        if self.lr_decay != 1:
            saving_label += f" lr_end {self.lr_end:.2e}".replace('.', '_')
        if self.dim_latent_noise != 1:
            saving_label += f" dim_latent_noise {self.dim_latent_noise}"
        for label in ['generator', 'discriminator']:
            nn_type = self.__dict__[f'nn_type_{label}']
            if nn_type != 'linear':
                saving_label += f" nn_type_{label} {nn_type}"

        return saving_label

    def add_training_data(
        self, inputs=None, outputs=None, outputs_test=None,
        p_clus=None, p_trans=None, n_clus=None, plot=True
    ):
        self.train_data_length = len(outputs)
        self.outputs = th.tensor(outputs)
        if outputs_test is not None:
            self.outputs_test = th.tensor(outputs_test)

        if self.normalised:
            for i in range(self.train_data_length):
                if inputs is not None:
                    self.inputs[i] = self.inputs[i] / self.inputs[i][0]
                self.outputs[i] = self.outputs[i] / self.inputs[i][0]
                self.outputs_test[i] = self.outputs_test[i] / self.inputs[i][0]

        if plot:
            self.plot_inputs()

        self.mean_real_output = np.nanmean(self.outputs)
        self.std_real_output = np.nanstd(self.outputs)
        self.mean_test_output = np.nanmean(self.outputs_test)
        self.std_test_output = np.nanstd(self.outputs_test)

        self.train_data = self.outputs
        self.test_data = self.outputs_test

        if p_clus is not None:
            self.p_clus0, self.p_trans0, self.n_clus = p_clus, p_trans, n_clus

    def plot_inputs(self):
        if self.prm['plots']:
            fig = plt.figure()
            for i in range(min(20, len(self.outputs))):
                plt.plot(self.outputs[i])

            title = f"{self.get_saving_label()} series"
            plt.title(title)
            title = title.replace(' ', '_')
            save_fig(fig, self.prm, self.save_path / title)
            plt.close('all')

    def get_train_loader(self):
        i_not_nans = [
            i for i, train_data_i in enumerate(self.train_data)
            if not any(th.isnan(train_data_i))
        ]
        train_data_not_nan = self.train_data[i_not_nans]
        train_data_cut = \
            train_data_not_nan[: - (train_data_not_nan.size()[0] % self.n_items_generated)] \
            if (train_data_not_nan.size()[0] % self.n_items_generated) > 0 \
            else train_data_not_nan
        train_data_cut = train_data_cut.view(
            (-1, train_data_cut.size()[1] * self.n_items_generated)
        )
        self.train_loader = th.utils.data.DataLoader(
            train_data_cut, batch_size=self.batch_size, shuffle=True
        )

    def update_n_items_generated(self):
        self.size_output_generator = self.size_output_generator_one_item * self.n_items_generated
        self.size_input_discriminator = self.size_input_discriminator_one_item \
            * self.n_items_generated
        self.size_input_generator = self.size_input_generator_one_item

    def split_inputs_and_outputs(self, train_data):
        real_inputs = th.randn(self.batch_size_, self.dim_latent_noise)
        real_outputs = train_data

        return real_inputs, real_outputs

    def merge_inputs_and_outputs(self, real_inputs, generated_outputs, real_outputs=None):
        if real_outputs is None:
            real_samples = None
        else:
            real_samples = real_outputs
        generated_samples = generated_outputs

        return generated_samples, real_samples

    def insert_generated_outputs_in_inputs_list(self, inputs, outputs):
        n_inputs = int(self.size_input_generator / self.n_items_generated)
        samples = th.zeros(
            (self.batch_size_, self.size_input_generator + self.n_items_generated)
        )
        for i_batch in range(self.batch_size_):
            for i_item in range(self.n_items_generated):
                samples[i_batch, i_item * (n_inputs + 1): i_item * (n_inputs + 1) + n_inputs] \
                    = inputs[i_batch, i_item * n_inputs: (i_item + 1) * n_inputs]
                samples[i_batch, i_item * (n_inputs + 1) + n_inputs] \
                    = outputs[i_batch, i_item]

        return samples

    def initialise_generator_and_discriminator(self):
        self.update_n_items_generated()
        self.generator = Generator(self)
        self.discriminator = Discriminator(self)
        self.loss_function = nn.BCELoss()
        self.optimizer_discriminator = th.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_start * self.lr_discriminator_ratio
        )
        self.optimizer_generator = th.optim.Adam(self.generator.parameters(), lr=self.lr_start)

    def train_discriminator(self, real_inputs, real_outputs, test):
        generated_outputs = self.generator(real_inputs, test=test)
        generated_samples_labels = th.zeros((self.batch_size_, 1))

        checks_nan(real_inputs, real_outputs, generated_outputs)

        generated_samples, real_samples = self.merge_inputs_and_outputs(
            real_inputs, generated_outputs, real_outputs
        )
        all_samples = th.cat((real_samples, generated_samples))
        all_samples_labels = th.cat(
            (self.get_real_samples_labels(), generated_samples_labels)
        )
        # Training the discriminator
        self.discriminator.zero_grad()
        output_discriminator = self.discriminator(all_samples.to(th.float32))
        loss_discriminator = self.loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        self.optimizer_discriminator.step()

        return loss_discriminator

    def get_real_samples_labels(self):
        return th.ones((self.batch_size_, 1))

    def _compute_statistical_indicators_generated_profiles(self, generated_samples):
        generated_samples_2d = generated_samples.view(
            self.batch_size_ * self.n_items_generated, -1
        )
        percentiles_generated = {}
        for percentile in self.percentiles:
            percentiles_generated[f'p{percentile}'] = th.quantile(
                generated_samples_2d,
                percentile/100,
                axis=0
            )
        percentiles_generated['mean'] = th.mean(generated_samples_2d, axis=0)

        return percentiles_generated

    def _compute_diff_forecast_test(
            self, sum_diff_forecast_test, n_forecast_test,
            t, generated_samples_2d, i_test, i_generated
    ):
        if t > 0:
            diff = self.outputs_test[i_test, t] - generated_samples_2d[i_generated, t - 1]
            if th.abs(diff) > 1e-4:
                sum_diff_forecast_test += th.abs(
                    (self.outputs_test[i_test, t] - generated_samples_2d[i_generated, t]) / diff
                )
                n_forecast_test += 1

        return sum_diff_forecast_test, n_forecast_test

    def _compute_metrics_episode(self, episode, generated_samples):
        mean_real_t = th.mean(self.outputs, dim=0)[~self.zero_values]
        std_real_t = th.std(self.outputs, dim=0)[~self.zero_values]
        mean_test_t = th.mean(self.outputs_test, dim=0)[~self.zero_values]
        std_test_t = th.std(self.outputs_test, dim=0)[~self.zero_values]
        generated_samples_2d = generated_samples.view(
            self.batch_size_ * self.n_items_generated, -1
        )
        mean_generated_t = th.mean(generated_samples_2d, dim=0)
        std_generated_t = th.std(generated_samples_2d, dim=0)
        episode['ave_diff_mean'] = th.mean(th.square(mean_generated_t - mean_real_t))
        episode['ave_diff_mean_test'] = th.mean(th.square(mean_generated_t - mean_test_t))
        n_real = len(self.outputs)
        n_generated = int(self.batch_size_ * self.n_items_generated)
        n_sum = min(n_generated, n_real)
        n_sum_test = min(n_generated, len(self.outputs_test))
        [
            sum_diff_x_mean, sum_diff_x2, sum_diff_y_mean, sum_diff_y2,
            sum_diff_x_y_2, sum_x2, sum_diff_forecast, n_forecast
        ] = [0] * 8
        [
            sum_diff_x_mean_test, sum_diff_x2_test, sum_diff_x_y_2_test,
            sum_x2_test, sum_diff_forecast_test, n_forecast_test
        ] = [0] * 6
        for t in range(self.n_profile):
            for i in range(n_sum):
                i_real = np.random.choice(n_real)
                i_generated = np.random.choice(n_generated)
                sum_diff_x_mean += self.outputs[i_real, t] - mean_real_t[t]
                sum_diff_x2 += th.square(self.outputs[i_real, t] - mean_real_t[t])
                sum_diff_y_mean += generated_samples_2d[i_generated, t] - mean_generated_t[t]
                sum_diff_y2 += th.square(generated_samples_2d[i_generated, t] - mean_generated_t[t])
                sum_diff_x_y_2 += th.square(
                    self.outputs[i_real, t] - generated_samples_2d[i_generated, t]
                )
                sum_x2 += th.square(self.outputs[i_real, t])
                if t > 0:
                    f = generated_samples_2d[i_generated, t - 1]
                    if th.abs(self.outputs[i_real, t] - f) > 1e-4:
                        sum_diff_forecast += th.abs(
                            (self.outputs[i_real, t] - generated_samples_2d[i_generated, t]) /
                            (self.outputs[i_real, t] - f)
                        )
                        n_forecast += 1

                if i < n_sum_test:
                    i_test = np.random.choice(n_sum_test)
                    sum_diff_x_mean_test += self.outputs_test[i_test, t] - mean_test_t[t]
                    sum_diff_x2_test += th.square(self.outputs_test[i_test, t] - mean_test_t[t])
                    sum_diff_x_y_2_test += th.square(
                        self.outputs_test[i_test, t] - generated_samples_2d[i_generated, t]
                    )
                    sum_x2_test += th.square(self.outputs_test[i_test, t])
                    sum_diff_forecast_test, n_forecast_test = self._compute_diff_forecast_test(
                        sum_diff_forecast_test, n_forecast_test, t, generated_samples_2d,
                        i_test, i_generated
                    )

        episode['pcc'] = (sum_diff_x_mean * sum_diff_y_mean) / th.sqrt(sum_diff_x2 * sum_diff_y2)
        episode['prd'] = th.sqrt(sum_diff_x_y_2 / sum_x2)
        episode['rmse'] = th.sqrt(1/th.tensor(n_sum * self.n_profile) * sum_diff_x_y_2)
        episode['mrae'] = 1/n_forecast * sum_diff_forecast
        episode['diff_mean'] = sum(
            (mean_generated_t[t] - mean_real_t[t]) ** 2 for t in range(self.n_profile)
        )
        episode['diff_std'] = sum(
            (std_generated_t[t] - std_real_t[t]) ** 2 for t in range(self.n_profile)
        )
        episode['pcc_test'] = (sum_diff_x_mean_test * sum_diff_y_mean) \
            / th.sqrt(sum_diff_x2_test * sum_diff_y2)
        episode['prd_test'] = th.sqrt(sum_diff_x_y_2_test / sum_x2_test)
        episode['rmse_test'] = th.sqrt(
            1 / th.tensor(n_sum_test * self.n_profile) * sum_diff_x_y_2_test
        )
        episode['mrae_test'] = 1/n_forecast_test * sum_diff_forecast_test
        episode['diff_mean_test'] = sum(
            (mean_generated_t[t] - mean_test_t[t]) ** 2 for t in range(self.n_profile)
        )
        episode['diff_std_test'] = sum(
            (std_generated_t[t] - std_test_t[t]) ** 2 for t in range(self.n_profile)
        )

        return episode

    def _diff_sum_profile_1(self, generated_samples, i, j):
        return th.sum(generated_samples[j, i * self.n_profile: (i + 1) * self.n_profile]) - 1

    def train_generator(self, real_inputs, final_n, epoch, test, plot):
        episode = {}
        self.generator.zero_grad()
        generated_outputs = self.generator(real_inputs.to(th.float32), test)
        generated_samples, _ = self.merge_inputs_and_outputs(real_inputs, generated_outputs)
        output_discriminator_generated = self.discriminator(generated_samples)
        episode['loss_generator'] = self.loss_function(
            output_discriminator_generated, self.get_real_samples_labels()
        )
        percentiles_generated \
            = self._compute_statistical_indicators_generated_profiles(generated_samples)
        episode['loss_percentiles'] = 0
        for key in [f'p{percentile}' for percentile in self.percentiles] + ['mean']:
            episode['loss_percentiles'] += th.sum(
                th.square(
                    percentiles_generated[key]
                    - th.from_numpy(self.percentiles_inputs_train[self.k][key][~self.zero_values])
                )
            )
        episode['loss_generator'] += episode['loss_percentiles'] * self.weight_diff_percentiles
        episode = self._compute_metrics_episode(episode, generated_samples)
        divergences_from_1 = th.stack(
            [th.stack(
                [
                    abs(self._diff_sum_profile_1(generated_samples, i, j))
                    for i in range(self.n_items_generated)
                ]
            ) for j in range(self.batch_size_)]
        )
        episode['mean_err_1'] = th.mean(divergences_from_1)
        episode['std_err_1'] = th.std(divergences_from_1)
        episode['share_large_err_1'] = th.sum(divergences_from_1 > 1) \
            / (self.n_items_generated * self.batch_size_)

        episode['loss_sum_profiles'] = th.sum(
            th.stack(
                [
                    th.stack(
                        [
                            self._diff_sum_profile_1(generated_samples, i, j) ** 2
                            for i in range(self.n_items_generated)
                        ]
                    ) for j in range(self.batch_size_)
                ]
            )
        )
        episode['loss_generator'] += episode['loss_sum_profiles'] * self.weight_sum_profiles

        episode['loss_generator'].backward()
        self.optimizer_generator.step()
        if final_n and epoch % self.n_epochs_test == 0 and plot:
            for compared_with_test_set in [True, False]:
                self.plot_statistical_indicators_profiles(
                    percentiles_generated, epoch, compared_with_test_set=compared_with_test_set
                )
        episode['means_outputs'] = th.mean(generated_outputs)
        episode['stds_outputs'] = th.std(generated_outputs)

        return generated_outputs, episode, generated_samples

    def plot_statistical_indicators_profiles(
            self, percentiles_generated, epoch, compared_with_test_set=False,
    ):
        if self.prm['plots']:
            if compared_with_test_set:
                percentiles_inputs = self.percentiles_inputs_test
            else:
                percentiles_inputs = self.percentiles_inputs_train
            fig = plt.figure()
            for color, percentiles_, label in zip(
                    ['b', 'g'],
                    [percentiles_generated, percentiles_inputs[self.k]],
                    ['generated', 'original']
            ):
                for indicator in [f'p{percentile}' for percentile in self.percentiles] + ['mean']:
                    percentiles_np = percentiles_[indicator] if label == 'original' \
                        else percentiles_[indicator].detach().numpy()
                    if label == 'generated':
                        y = np.zeros(self.prm['n'])
                        y[~self.zero_values] = percentiles_np
                    else:
                        y = percentiles_np
                    plt.plot(
                        y,
                        color=color,
                        linestyle='--' if indicator[0] == 'p' else '-',
                        label=label if indicator == 'mean' else None,
                        alpha=1 if indicator in ['p50', 'mean'] else 0.5
                    )

            plt.legend()
            title = f"{self.get_saving_label()} profiles generated vs original epoch {epoch}"
            if self.normalised:
                title += ' normalised'
            if compared_with_test_set:
                title += ' test set'
            plt.title(title)
            title = title.replace(' ', '_')
            if epoch in [0, self.n_epochs - 1]:
                fig.savefig(
                    self.save_path / f"{title}.pdf", bbox_inches='tight', format='pdf', dpi=1200
                )
            else:
                fig.savefig(self.save_path / title)
            plt.close('all')

    def plot_generated_samples_start_epoch(self, generated_samples, epoch):
        saving_label = self.get_saving_label()
        if self.prm['plots']:
            fig = plt.figure()
            for i in range(self.batch_size_):
                plt.plot(generated_samples[i].detach().numpy()[1:])
            title = f'{saving_label} generated samples epoch {epoch}'
            if self.normalised:
                title += ' normalised'
            plt.title(title)
            title = title.replace(' ', '_')
            save_fig(fig, self.prm, self.save_path / title)
            plt.close('all')

    def update_noise_and_lr_generator(self, epoch):
        if self.initial_noise is not None and epoch < self.n_epochs_initial_noise:
            self.generator.noise_factor = self.initial_noise
            self.discriminator.noise_factor = self.initial_noise

        elif self.noise_reduction_type == 'exp':
            self.generator.noise_factor *= self.generator.noise_reduction_exp
            self.discriminator.noise_factor *= self.generator.noise_reduction_exp

        elif self.noise_reduction_type == 'linear':
            noise_epoch = self.noise0 + epoch / self.n_epochs * (self.noise_end - self.noise0)
            self.generator.noise_factor = noise_epoch
            self.discriminator.noise_factor = noise_epoch

        else:
            print(f"noise reduction type {self.noise_reduction_type} not implemented")

        if self.initial_lr is not None and epoch < self.n_epochs_initial_lr:
            for g in self.optimizer_generator.param_groups:
                g['lr'] = self.initial_lr
            for g in self.optimizer_discriminator.param_groups:
                g['lr'] = self.initial_lr
        else:
            for g in self.optimizer_generator.param_groups:
                g['lr'] = self.lr_start * self.lr_decay ** epoch
            for g in self.optimizer_discriminator.param_groups:
                g['lr'] = self.lr_start * self.lr_decay ** epoch * self.lr_discriminator_ratio

    def plot_metrics_over_time(self, episodes, epoch, test=False):
        if not self.prm['plots']:
            return

        title = f"{self.get_saving_label()} metrics over time"
        if self.normalised:
            title += ' normalised'
        if test:
            title += ' test'
        fig, axs = plt.subplots(3, 2)
        for label, x, y in zip(
            ['pcc', 'prd', 'rmse', 'mrae', 'diff_mean', 'diff_std'],
            [0, 0, 1, 1, 2, 2],
            [0, 1, 0, 1, 0, 1]
        ):
            axs[x, y].plot(episodes[label][:epoch].detach().numpy())
            axs[x, y].set_xlabel("Epochs")
            axs[x, y].set_ylabel(label)

        title = title.replace(' ', '_')
        save_fig(fig, self.prm, self.save_path / title)
        plt.close('all')

    def plot_losses_over_time(self, episodes, epoch, test=False):
        if not self.prm['plots']:
            return
        title = f"{self.get_saving_label()} losses over time"
        if self.normalised:
            title += ' normalised'
        if test:
            title += " test"
        colours = sns.color_palette()
        fig, ax = plt.subplots()
        twin = ax.twinx()
        labels = ["loss_generator", "loss_percentiles"]
        # if self.data_type != 'gen':
        if True:
            labels.append("loss_sum_profiles")
        alphas = [1, 0.5]
        ps = []
        for i, (label, alpha) in enumerate(zip(labels, alphas)):
            p, = ax.plot(
                episodes[label][:epoch + 1].detach().numpy(),
                color=colours[i], label=label, alpha=alpha
            )
            ps.append(p)
            with open(self.save_path / f"{label}.pickle", 'wb') as file:
                pickle.dump(episodes[label], file)
        p3, = twin.plot(
            episodes['loss_discriminator'][:epoch + 1].detach().numpy(),
            color=colours[3], label="Discriminator losses"
        )
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Generator losses")
        ax.set_yscale('log')
        twin.set_ylabel("Discriminator losses")
        ax.yaxis.label.set_color(ps[0].get_color())
        twin.yaxis.label.set_color(p3.get_color())
        ax.legend(handles=[ps[0], ps[1], p3])
        ax.set_title(epoch)
        title = title.replace(' ', '_')
        save_fig(fig, self.prm, self.save_path / title)
        plt.close('all')

    def plot_noise_over_time(self):
        if self.prm['plots']:
            fig = plt.figure()
            noises = [
                self.generator.noise0 * self.generator.noise_reduction ** epoch
                for epoch in range(self.n_epochs)
            ]
            plt.plot(noises)
            title = f"{self.get_saving_label()} noise over time"
            plt.title(title)
            save_fig(fig, self.prm, self.save_path / title)
            plt.close('all')

    def car_output_to_percentage_availability(self, generated_outputs, tol=None):
        tol = self.tol if tol is None else tol
        ev_avail = np.ones(np.shape(generated_outputs))
        for i in range(len(generated_outputs)):
            ev_avail[i], generated_outputs[i] = car_loads_to_availability(
                generated_outputs[i], tol=tol
            )
        percentage_availability = np.sum(ev_avail) / np.multiply(*np.shape(ev_avail))
        print(
            f"% car available generated =  {percentage_availability}"
        )
        if isinstance(generated_outputs, np.ndarray):
            print(f"average trip non zero {np.mean(generated_outputs[generated_outputs > 0])}")
        else:
            print(f"average trip non zero {th.mean(generated_outputs[generated_outputs > 0])}")

        return percentage_availability

    def plot_final_hist_generated_vs_real(self, generated_outputs, real_outputs, epoch):
        generated_outputs = generated_outputs.detach().numpy()
        if self.data_type == 'car':
            self.car_output_to_percentage_availability(generated_outputs)
        if self.prm['plots']:
            nbins = 100
            generated_outputs_reshaped = np.array(generated_outputs).flatten()
            real_outputs_reshaped = np.array(real_outputs.detach().numpy(), dtype=int).flatten()

            fig = plt.figure()
            plt.hist(
                [generated_outputs_reshaped, real_outputs_reshaped],
                bins=nbins, alpha=0.5, label=['generated', 'real'], color=['red', 'blue']
            )
            plt.legend()
            title = f"{self.get_saving_label()} hist generated vs real epoch {epoch}"
            if self.normalised:
                title += ' normalised'
            title = title.replace(' ', '_')
            save_fig(fig, self.prm, self.save_path / title)
            plt.close('all')

    def barplot_compare_metrics_train_test_dataset(self, episodes, epoch):
        labels = [
            'pcc',
            'prd', 'rmse', 'mrae',
            # 'ave_diff_mean'
        ]
        labels_for_plot = ['PCC', 'PRD', 'RMSE', 'MRAE']
        train_values = np.ones(len(labels))
        test_values = [episodes[f"{label}_test"][epoch]/episodes[label][epoch] for label in labels]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, train_values, width, label='Against training set')
        rects2 = ax.bar(x + width / 2, test_values, width, label='Against testing set')

        ax.set_xticks(x)
        ax.set_xticklabels(labels_for_plot)
        ax.legend(loc='upper left')
        ax.set_ylabel('Values normalised by training set values')
        title = f"{self.get_saving_label()} metrics train vs test epoch {epoch}"
        title = title.replace(' ', '_')
        fig.savefig(
            self.save_path / f"{title}.pdf", bbox_inches='tight', format='pdf', dpi=1200
        )
        plt.close('all')

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        fig.tight_layout()

        plt.show()

    def train(self, plot=True):
        self.get_train_loader()
        n_train_loader = len(self.train_loader)
        self.initialise_generator_and_discriminator()
        episode_entries = [
            'loss_generator', 'loss_discriminator', 'loss_percentiles',
            'means_outputs', 'stds_outputs',
        ]
        for entry in ['pcc', 'prd', 'rmse', 'mrae', 'diff_mean', 'diff_std', 'ave_diff_mean']:
            episode_entries.append(entry)
            episode_entries.append(f"{entry}_test")
        episode_entries += ['loss_sum_profiles',  'mean_err_1', 'std_err_1', 'share_large_err_1']
        episodes = {
            info: th.zeros(self.n_epochs * n_train_loader) for info in episode_entries
        }
        episodes_test = {
            info: th.zeros(int(np.floor(self.n_epochs / self.n_epochs_test)))
            for info in episode_entries
        }

        potential_paths = list_potential_paths(
            self.prm, [self.data_type], data_folder='other_outputs', sub_data_folder='profiles'
        )
        for potential_path in potential_paths:
            path = potential_path / f"norm_{self.data_type}"
            files = {
                'episodes': path / f"episodes{self.ext}.pt",
                'idx': path / f"episodes_idx{self.ext}.pt",
                'done': path / f"done{self.ext}.pt"
            }
            if self.recover_weights and all(os.path.exists(file) for file in files.values()):
                episodes_path = files['episodes']
                episodes = th.load(episodes_path)
                offset_idx = th.load(files['idx'])
                done = th.load(files['done'])
                if done:
                    return
                break
            else:
                offset_idx = 0
        idx = 0
        for epoch in tqdm(range(self.n_epochs)):
            test = True if epoch % self.n_epochs_test == 0 else False
            epoch_test = int(np.floor(epoch / self.n_epochs_test))
            for n, train_data in enumerate(self.train_loader):
                self.batch_size_ = len(train_data)
                real_inputs, real_outputs = self.split_inputs_and_outputs(train_data)
                real_outputs = real_outputs.view(
                    self.n_items_generated * self.batch_size_, -1
                )[:, ~self.zero_values].view(self.batch_size_, -1)
                loss_discriminator = self.train_discriminator(real_inputs, real_outputs, test)
                final_n = n == len(self.train_loader) - 1
                generated_outputs, episode, generated_samples = self.train_generator(
                    real_inputs, final_n, epoch, test, plot=plot
                )
                with th.no_grad():
                    episodes['loss_discriminator'][idx + offset_idx] = loss_discriminator
                for key in episode:
                    with th.no_grad():
                        episodes[key][idx + offset_idx] = episode[key]
                if test:
                    with th.no_grad():
                        episodes_test['loss_discriminator'][epoch_test] = loss_discriminator
                    for key in episode:
                        with th.no_grad():
                            episodes_test[key][epoch_test] = episode[key]
                    if self.data_type == 'car':
                        self.car_output_to_percentage_availability(generated_outputs)
                idx += 1
                test = False

            self.update_noise_and_lr_generator(epoch)
            if epoch % self.n_epochs_test == 0 and plot:
                self._save_model(episodes, idx, done=False, save_ext=epoch_test)
                self._plot_errors_normalisation_profiles(episodes, idx - 1)
                self.plot_losses_over_time(episodes, epoch)
                self.plot_metrics_over_time(episodes, epoch)
                self.plot_losses_over_time(episodes_test, epoch_test, test=True)
                self.plot_metrics_over_time(episodes_test, epoch_test, test=True)
            if episodes['loss_percentiles'][epoch] < self.tol_loss_percentiles:
                if self.data_type == 'car':
                    self.get_ideal_ev_avail_tol(generated_outputs)
                break

        if plot:
            self.plot_final_hist_generated_vs_real(generated_outputs, real_outputs, epoch)
            self._plot_errors_normalisation_profiles(episodes, idx - 1)
            self.plot_losses_over_time(episodes, epoch)
            self.plot_metrics_over_time(episodes, epoch)
            percentiles_generated \
                = self._compute_statistical_indicators_generated_profiles(generated_samples)
            self.plot_statistical_indicators_profiles(
                percentiles_generated, epoch
            )
            self.barplot_compare_metrics_train_test_dataset(episodes, epoch)
            self._save_model(episodes, idx, done=True)

    def _save_model(self, episodes, idx, done=False, save_ext=None):
        save_hedge_path = self.prm['save_hedge'] / 'profiles' / f"norm_{self.data_type}" \
            if save_ext is None else self.save_path
        save_other_path = self.prm['save_other'] / 'profiles' / f"norm_{self.data_type}" \
            if save_ext is None else self.save_path

        if save_ext is not None:
            model_ext = f"{self.ext}_{save_ext}.pt"
        else:
            model_ext = f"{self.ext}.pt"
        th.save(episodes, save_other_path / f"episodes{self.ext}.pt")
        th.save(idx, save_other_path / f"episodes_idx{self.ext}.pt")
        th.save(done, save_other_path / f"done{self.ext}.pt")
        try:
            th.save(
                self.generator.model,
                save_hedge_path
                / f"generator{model_ext}"
            )
            th.save(
                self.generator.model.state_dict(),
                save_other_path
                / f"generator_weights_{self.data_type}_{self.day_type}_{self.k}{self.ext}.pt"
            )
            th.save(
                self.discriminator.model,
                save_other_path
                / f"discriminator{self.ext}.pt"
            )
            th.save(
                self.discriminator.model.state_dict(),
                save_other_path
                / f"discriminator_weights{self.ext}.pt"
            )
        except Exception as ex1:
            try:
                th.save(
                    self.generator.fc,
                    save_hedge_path / f"generator_{self.get_saving_label()}_fc{self.ext}.pt"
                )
                th.save(
                    self.generator.conv,
                    save_hedge_path / f"generator_{self.get_saving_label()}_conv{self.ext}.pt"
                )
            except Exception as ex2:
                print(f"Could not save model weights: ex1 {ex1}, ex2 {ex2}")

    def _plot_errors_normalisation_profiles(self, episodes, idx):
        if not self.prm['plots']:
            return

        title = f"{self.get_saving_label()} normalisation errors over time"
        colours = sns.color_palette()
        fig, ax = plt.subplots(3)
        ax[0].plot(episodes['mean_err_1'][:idx + 1].detach().numpy(), color=colours[0])
        ax[0].set_title('mean error')
        ax[1].plot(episodes['std_err_1'][:idx + 1].detach().numpy(), color=colours[1])
        ax[1].set_title(f'std error {idx}')
        ax[2].plot(episodes['share_large_err_1'][:idx + 1].detach().numpy(), color=colours[2])
        ax[2].set_title(f'share large error > 1 {idx}')
        ax[2].set_xlabel("Epochs")
        title = title.replace(' ', '_')
        save_fig(fig, self.prm, self.save_path / title)
        plt.close('all')


class Discriminator(nn.Module):
    def __init__(self, gan_trainer):
        super().__init__()
        for attribute in ['size_input', 'nn_type', 'dropout']:
            setattr(self, attribute, getattr(gan_trainer, f"{attribute}_discriminator"))
        self.recover_weights = gan_trainer.recover_weights
        self.noise0 = gan_trainer.noise0
        self.save_hedge_path = gan_trainer.prm['save_hedge']
        self._initialise_model(gan_trainer)
        self.noise_factor = gan_trainer.noise0

    def _initialise_model(self, gan_trainer):
        size_input, nn_type, dropout = self.size_input, self.nn_type, self.dropout
        if nn_type == 'linear':
            self.model = nn.Sequential(
                nn.Linear(size_input, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )
            path = self.save_hedge_path / 'profiles' / f"norm_{gan_trainer.data_type}"
            file_name = \
                f"discriminator_weights_{gan_trainer.data_type}" \
                f"_{gan_trainer.day_type}_{gan_trainer.k}.pt"
            weights_path = path / file_name
            if os.path.exists(weights_path) and self.recover_weights:
                weights = th.load(weights_path)
                self.model.load_state_dict(weights)

        elif nn_type == 'cnn':
            self.model = nn.Sequential(
                nn.Conv1d(200, 200, kernel_size=3),
                nn.BatchNorm1d(num_features=size_input-2),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(size_input-2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

    def forward(self, x, test=False):
        output = self.model(x)
        noise = th.zeros(output.shape) if test else th.randn(output.shape) * self.noise_factor
        output = th.clamp(output + noise, min=0, max=1)

        return output


class Generator(nn.Module):
    def __init__(
            self, gan_trainer
    ):
        super().__init__()
        attribute_list = [
            'min',
            'max',
            'recover_weights',
            'size_output_generator_one_item',
            'tol',
            'data_type'
        ]
        for attribute in attribute_list:
            setattr(self, attribute, getattr(gan_trainer, attribute))
        self.save_hedge_path = gan_trainer.prm['save_hedge']
        self.hidden_dim = 256
        self.n_layers = 2
        self._initialise_model(gan_trainer)
        if gan_trainer.noise_reduction_type == 'exp':
            self.noise_reduction_exp = math.exp(
                math.log(gan_trainer.noise_end / gan_trainer.noise0) / gan_trainer.n_epochs
            )
        self.noise_factor = gan_trainer.noise0

    def _initialise_model(self, gan_trainer):
        for attribute in ['nn_type', 'size_output', 'size_input', 'dropout']:
            setattr(self, attribute, getattr(gan_trainer, f"{attribute}_generator"))
        multiplier = 1
        if self.nn_type == 'linear':
            self.model = nn.Sequential(
                nn.Linear(self.size_input, 16 * multiplier),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(16 * multiplier, 32 * multiplier),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(32 * multiplier, self.size_output),
                nn.Sigmoid(),
            )
            path = gan_trainer.prm['save_hedge'] / 'profiles' / f"norm_{gan_trainer.data_type}"
            file_name = \
                f"generator_weights_{gan_trainer.data_type}" \
                f"_{gan_trainer.day_type}_{gan_trainer.k}.pt"
            weights_path = path / file_name
            if os.path.exists(weights_path) and self.recover_weights:
                weights = th.load(weights_path)
                self.model.load_state_dict(weights)

        elif self.nn_type == 'cnn':
            self.fc = nn.Sequential(
                nn.Linear(self.size_input, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, self.size_output * 8),
                nn.ReLU(),
            )

            self.conv = nn.Sequential(
                nn.ConvTranspose1d(
                   self.size_output * 8, self.size_output * 4, kernel_size=3, stride=2, padding=1,
                ),
                nn.BatchNorm1d(self.size_output * 4),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.ConvTranspose1d(
                   self.size_output * 4, self.size_output * 2, kernel_size=3, stride=2, padding=1,
                ),
                nn.BatchNorm1d(self.size_output * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.ConvTranspose1d(
                   self.size_output * 2, self.size_output, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm1d(self.size_output),
                nn.ReLU(),
            )
        elif self.nn_type == 'rnn':
            # Defining the layers

            # RNN Layer
            self.rnn = nn.RNN(self.size_input, self.hidden_dim, self.n_layers, batch_first=True)
            # Fully connected layer
            self.fc = nn.Linear(self.hidden_dim, self.size_output)

        elif self.nn_type == 'lstm':
            self.lstm = nn.LSTM(self.size_input, self.hidden_dim, self.n_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_dim, self.size_output)

    def forward(self, x, test=False):
        if self.nn_type == 'linear':
            output = self.model(x)
        elif self.nn_type == 'cnn':
            x = self.fc(x)
            x = x.view(-1, 8 * self.size_output, 1)
            output = self.conv(x)
        elif self.nn_type == 'rnn':
            batch_size = x.size(0)
            # Initializing hidden state for first input using method defined below
            hidden = self.init_hidden(batch_size)
            # Passing in the input and hidden state into the model and obtaining outputs
            output, hidden = self.rnn(x, hidden)
            # Reshaping the outputs such that it can be fit into the fully connected layer
            output = output.contiguous().view(-1, self.hidden_dim)
            output = self.fc(output)
        elif self.nn_type == 'lstm':
            output, _ = self.lstm(x)
            output = self.fc(output)

        if self.data_type != 'car':
            output = output.reshape(-1, self.size_output_generator_one_item)
            output = th.div(
                output,
                th.sum(output, dim=1).reshape(-1, 1)
            ).reshape(-1, self.size_output)
        noise = th.zeros(output.shape) if test else th.randn(output.shape) * self.noise_factor
        if self.data_type != 'car':
            output = th.clamp(output + noise, min=self.min, max=self.max * 1.1)
        else:
            output = th.clamp(output + noise, min=self.min)

        return output

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = th.zeros(self.n_layers, self.hidden_dim)
        return hidden


def compute_profile_generators(
        profiles, k, percentiles_inputs_test, percentiles_inputs_train, data_type,
        day_type, prm, percentage_car_avail=None, average_non_zero_trip=None,
):
    print("profile generators")
    print(f"len(profiles): {len(profiles)}")
    n_train_data = int(len(profiles) * prm['train_set_size'])
    profiles_train = profiles[0: n_train_data]
    profiles_test = profiles[n_train_data:]

    zero_values = (
        (percentiles_inputs_train[k]['p90'] == 0)
        & (percentiles_inputs_train[k]['p10'] == 0)
        & (percentiles_inputs_train[k]['mean'] < 0.01)
    )
    ext = f"_{data_type}_{day_type}_{k}"
    params = {
        'profiles': True,
        'batch_size': 100,
        'n_epochs': int(1e8 / len(profiles_train)),
        'weight_sum_profiles': 1e-7,
        'weight_diff_percentiles': 100,
        'zero_values': zero_values,
        'n_profile': prm['n'] - sum(zero_values),
        'size_input_discriminator_one_item': prm['n'] - sum(zero_values),
        'size_output_generator_one_item': prm['n'] - sum(zero_values),
        'k': k,
        'n': prm['n'],
        'percentiles_inputs_train': percentiles_inputs_train,
        'percentiles_inputs_test': percentiles_inputs_test,
        'data_type': data_type,
        'n_items_generated': 50,
        'nn_type_generator': 'linear',
        'nn_type_discriminator': 'linear',
        'noise0': 1,
        'noise_end': 1e-4,
        'lr_start': 0.1,
        'lr_end': 0.001,
        'dropout_discriminator': 0.3,
        'dropout_generator': 0.15,
        'day_type': day_type,
        'dim_latent_noise': 1,
        'percentiles': [10, 25, 50, 75, 90],
        'noise_reduction_type': 'exp',
        'initial_noise': None,
        'initial_lr': None,
        'n_epochs_initial_lr': 100,
        'min': min(percentiles_inputs_train[k]['p10']),
        'max': max(percentiles_inputs_train[k]['p90']),
        'lr_discriminator_ratio': 1,
        'recover_weights': False,
        'n_epochs_test': 100,
        'tol': 1e-2,
        'ext': ext,
        'percentage_car_avail': percentage_car_avail,
        'average_non_zero_trip': average_non_zero_trip,
        'tol_loss_percentiles': 5e-1 / 100,
    }
    path = prm['save_hedge'] / 'profiles' / f"norm_{data_type}"
    np.save(path / f"zerovalues{ext}.npy", zero_values)
    np.save(path / f"min{ext}.npy", params['min'])
    np.save(path / f"max{ext}.npy", params['max'])

    if data_type == 'gen':
        params['noise0'] = 0.01
        params['noise_end'] = 1e-3
        params['lr_start'] = 1e-2
        params['lr_end'] = 1e-4
        # params['recover_weights'] = False

    elif data_type == 'loads':
        params['noise0'] = 0
        params['noise_end'] = 0
        params['noise_reduction_type'] = 'linear'
        params['lr_start'] = 1e-3
        params['lr_end'] = 1e-2
        params['initial_noise'] = 0.01
        params['n_epochs_initial_noise'] = 100
        params['dropout_generator'] = 0.5
        params['lr_discriminator_ratio'] = 1e-3
        params['tol_loss_percentiles'] = 5e-2 / 100
        # params['weight_diff_percentiles'] = 0
        # params['n_epochs'] = 2500
        # params['recover_weights'] = True
    elif data_type == 'car':
        params['noise0'] = 1e-2
        params['noise_end'] = 1e-2
        params['lr_start'] = 5e-3
        params['lr_end'] = 5e-5
        params['initial_noise'] = 0.01
        params['n_epochs_initial_noise'] = 100
        params['dropout_generator'] = 0.5
        params['lr_discriminator_ratio'] = 1e-3
        params['n_epochs_test'] = 10
        # params['recover_weights'] = True

    params['lr_decay'] = (params['lr_end'] / params['lr_start']) ** (1 / params['n_epochs'])
    params['size_input_generator_one_item'] = params['dim_latent_noise']
    gan_trainer = GAN_Trainer(params, prm)
    gan_trainer.update_value_type('profiles')
    gan_trainer.add_training_data(outputs=profiles_train, outputs_test=profiles_test)

    gan_trainer.train()

    return gan_trainer.generator, params
