import math
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
from torch import nn
from tqdm import tqdm

from src.hedge import car_loads_to_availability
from src.utils import save_fig

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

    def get_saving_label(self):
        saving_label \
            = f'{self.data_type} {self.day_type}, {self.value_type} lr_start {self.lr_start:.2e} ' \
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
        self, inputs=None, outputs=None, p_clus=None, p_trans=None, n_clus=None,
    ):
        self.train_data_length = len(outputs)
        self.outputs = th.tensor(outputs)

        if self.normalised:
            for i in range(self.train_data_length):
                if inputs is not None:
                    self.inputs[i] = self.inputs[i] / self.inputs[i][0]
                self.outputs[i] = self.outputs[i] / self.inputs[i][0]

        self.plot_inputs()

        self.mean_real_output = np.nanmean(self.outputs)
        self.std_real_output = np.nanstd(self.outputs)

        self.train_data = self.outputs

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
        self.generator = Generator(
            size_inputs=self.size_input_generator,
            size_outputs=self.size_output_generator,
            n_epochs=self.n_epochs,
            nn_type=self.nn_type_generator,
            batch_size=self.batch_size,
            noise0=self.noise0,
            noise_end=self.noise_end,
            dropout=self.dropout_generator,
            data_type=self.data_type
        )
        self.discriminator = Discriminator(
            size_inputs=self.size_input_discriminator,
            nn_type=self.nn_type_discriminator,
            dropout=self.dropout_discriminator,
        )
        self.loss_function = nn.BCELoss()

        self.optimizer_discriminator = th.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_start
        )
        self.optimizer_generator = th.optim.Adam(self.generator.parameters(), lr=self.lr_start)

    def train_discriminator(self, real_inputs, real_outputs):
        generated_outputs = self.generator(real_inputs.to(th.float32))
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
        n_samples = len(generated_samples_2d[0])
        percentiles_generated = {}
        for statistical_indicator in [f'p{percentile}' for percentile in self.percentiles] + ['mean']:
            percentiles_generated[statistical_indicator] = th.zeros(n_samples)
        for time in range(n_samples):
            for percentile in self.percentiles:
                percentiles_generated[f'p{percentile}'][time] = th.quantile(
                    generated_samples_2d[:, time],
                    percentile/100
                )
            percentiles_generated['mean'][time] = th.mean(generated_samples_2d[:, time])

        return percentiles_generated, generated_samples_2d, n_samples

    def train_generator(self, real_inputs, final_n, epoch):
        episode = {}
        self.generator.zero_grad()
        generated_outputs = self.generator(real_inputs.to(th.float32))
        generated_samples, _ = self.merge_inputs_and_outputs(real_inputs, generated_outputs)
        output_discriminator_generated = self.discriminator(generated_samples)
        episode['loss_generator'] = self.loss_function(
            output_discriminator_generated, self.get_real_samples_labels()
        )
        percentiles_generated, generated_samples_2d, n_samples \
            = self._compute_statistical_indicators_generated_profiles(generated_samples)
        episode['loss_percentiles'] = 0
        for key in [f'p{percentile}' for percentile in self.percentiles] + ['mean']:
            episode['loss_percentiles'] += th.sum(
                th.square(
                    percentiles_generated[key]
                    - th.from_numpy(self.percentiles_inputs[self.k][key])
                )
            ) * self.weight_diff_percentiles
        episode['loss_generator'] += episode['loss_percentiles']

        if self.data_type != 'gen':
            divergences_from_1 = th.stack(
                [th.stack(
                    [
                        abs(th.sum(generated_samples[j, i * self.prm['n']: (i + 1) * self.prm['n']]) - 1)
                        for i in range(self.n_items_generated)
                    ]
                ) for j in range(self.batch_size_)])
            episode['mean_err_1'] = th.mean(divergences_from_1)
            episode['std_err_1'] = th.std(divergences_from_1)
            episode['share_large_err_1'] = th.sum(divergences_from_1 > 1)/(self.n_items_generated * self.batch_size_)
            episode['loss_sum_profiles'] = th.sum(
                th.stack(
                    [th.stack(
                    [
                        (th.sum(generated_samples[j, i * self.prm['n']: (i + 1) * self.prm['n']]) - 1) ** 2
                        for i in range(self.n_items_generated)
                    ]
                )  for j in range(self.batch_size_)])
            ) * self.weight_sum_profiles
            episode['loss_generator'] += episode['loss_sum_profiles']

        episode['loss_generator'].backward()
        self.optimizer_generator.step()
        if final_n and epoch % 10 == 0:
            self.plot_statistical_indicators_profiles(
                percentiles_generated, epoch, n_samples
            )
        episode['means_outputs'] = th.mean(generated_outputs)
        episode['stds_outputs'] = th.std(generated_outputs)

        return generated_outputs, episode


    def plot_statistical_indicators_profiles(
            self, percentiles_generated, epoch, n_samples
    ):
        if self.prm['plots']:
            fig = plt.figure()
            xs = np.arange(n_samples)
            for color, percentiles_, label in zip(
                    ['b', 'g'],
                    [percentiles_generated, self.percentiles_inputs[self.k]],
                    ['generated', 'original']
            ):
                for indicator in [f'p{percentile}' for percentile in self.percentiles] + ['mean']:
                    percentiles_np = percentiles_[indicator] if label == 'original' \
                        else percentiles_[indicator].detach().numpy()
                    plt.plot(
                        xs,
                        percentiles_np,
                        color=color,
                        linestyle='--' if indicator[0] == 'p' else '-',
                        label=label if indicator == 'mean' else None,
                        alpha=1 if indicator in ['p50', 'mean'] else 0.5
                    )

            plt.legend()
            title = f"{self.get_saving_label()} profiles generated vs original epoch {epoch}"
            if self.normalised:
                title += ' normalised'
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
        self.generator.noise_factor *= self.generator.noise_reduction
        for g in self.optimizer_generator.param_groups:
            g['lr'] = self.lr_start * self.lr_decay ** epoch
        for g in self.optimizer_discriminator.param_groups:
            g['lr'] = self.lr_start * self.lr_decay ** epoch

    def plot_losses_over_time(self, episodes, epoch):
        if self.prm['plots']:
            title = f"{self.get_saving_label()} losses "
            title += "over time"
            if self.normalised:
                title += ' normalised'
            colours = sns.color_palette()
            fig, ax = plt.subplots()
            twin = ax.twinx()
            labels = ["loss_generator", "loss_percentiles"]
            if self.data_type != 'gen':
                labels.append("loss_sum_profiles")
            alphas = [1, 0.5]
            ps = []
            for i, (label, alpha) in enumerate(zip(labels, alphas)):
                p, = ax.plot(episodes[label][:epoch], color=colours[i], label=label, alpha=alpha)
                ps.append(p)
                with open(self.save_path / f"{label}.pickle", 'wb') as file:
                    pickle.dump(episodes[label], file)
            p3, = twin.plot(
                episodes['loss_discriminator'][:epoch], color=colours[3], label="Discriminator losses"
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

    def plot_final_hist_generated_vs_real(self, generated_outputs, real_outputs, epoch):
        generated_outputs = generated_outputs.detach().numpy()
        if self.data_type == 'car':
            ev_avail = np.ones(np.shape(generated_outputs))
            for i in range(len(generated_outputs)):
                ev_avail[i], generated_outputs[i] = car_loads_to_availability(generated_outputs[i])
            print(
                f"% car available generated = "
                f"{np.sum(ev_avail) / np.multiply(*np.shape(ev_avail))}"
            )
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

    def train(self):
        self.get_train_loader()
        n_train_loader = len(self.train_loader)
        self.initialise_generator_and_discriminator()
        episode_entries = [
                'loss_generator', 'loss_discriminator', 'loss_percentiles',
                'means_outputs', 'stds_outputs'
            ]
        if self.data_type != 'gen':
            episode_entries += ['loss_sum_profiles',  'mean_err_1', 'std_err_1', 'share_large_err_1']
        episodes = {
            info: np.zeros(self.n_epochs * n_train_loader) for info in episode_entries
        }
        idx = 0
        for epoch in tqdm(range(self.n_epochs)):
            for n, train_data in enumerate(self.train_loader):
                self.batch_size_ = len(train_data)
                real_inputs, real_outputs = self.split_inputs_and_outputs(train_data)
                episodes['loss_discriminator'][idx] = self.train_discriminator(real_inputs, real_outputs)
                final_n = n == len(self.train_loader) - 1
                generated_outputs, episode = self.train_generator(real_inputs, final_n, epoch)
                for key in episode:
                    episodes[key][idx] = episode[key].detach().numpy()
                idx += 1

            self.update_noise_and_lr_generator(epoch)
            if epoch % 2 == 0:
                self._save_model(ext=epoch)
                if self.data_type != 'gen':
                    self._plot_errors_normalisation_profiles(episodes, idx - 1)
                self.plot_losses_over_time(episodes, epoch)

            if episodes['loss_percentiles'][(epoch + 1) * n_train_loader - 1] < 9e-1:
                break

        self.plot_final_hist_generated_vs_real(generated_outputs, real_outputs, epoch)
        if self.data_type != 'gen':
            self._plot_errors_normalisation_profiles(episodes, idx - 1)
        self.plot_losses_over_time(episodes, epoch)
        self.plot_noise_over_time()
        print(
            f"mean generated outputs last 10: {np.mean(episodes['means_outputs'][-10:])}, "
            f"std {np.mean(episodes['stds_outputs'][-10:])}"
        )
        self._save_model()

    def _save_model(self, ext=''):
        path = self.prm['save_hedge'] / 'profiles' / f"norm_{self.data_type}" if ext == '' else self.save_path
        try:
            th.save(
                self.generator.model,
                path
                / f"generator_{self.data_type}_{self.day_type}_{self.k}{ext}.pt"
            )
        except Exception as ex1:
            try:
                th.save(
                    self.generator.fc,
                    path / f"generator_{self.get_saving_label()}{ext}_fc.pt"
                )
                th.save(
                    self.generator.conv,
                    path / f"generator_{self.get_saving_label()}{ext}_conv.pt"
                )
            except Exception as ex2:
                print(f"Could not save model weights: ex1 {ex1}, ex2 {ex2}")

    def _plot_errors_normalisation_profiles(self, episodes, idx):
        if not self.prm['plots']:
            return

        title = f"{self.get_saving_label()} normalisation errors over time"
        colours = sns.color_palette()
        fig, ax = plt.subplots(3)
        ax[0].plot(episodes['mean_err_1'][:idx + 1], color=colours[0])
        ax[0].set_title('mean error')
        ax[1].plot(episodes['std_err_1'][:idx + 1], color=colours[1])
        ax[1].set_title(f'std error {idx}')
        ax[2].plot(episodes['share_large_err_1'][:idx + 1], color=colours[2])
        ax[2].set_title(f'share large error > 1 {idx}')
        ax[2].set_xlabel("Epochs")
        title = title.replace(' ', '_')
        save_fig(fig, self.prm, self.save_path / title)
        plt.close('all')
        print(f"episodes['share_large_err_1'][{idx}] = {episodes['share_large_err_1'][idx]}")

class Discriminator(nn.Module):
    def __init__(self, size_inputs=1, nn_type='linear', dropout=0.3):
        super().__init__()
        self._initialise_model(size_inputs, nn_type, dropout)

    def _initialise_model(self, size_inputs, nn_type, dropout):
        if nn_type == 'linear':
            self.model = nn.Sequential(
                nn.Linear(size_inputs, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        elif nn_type == 'cnn':
            self.model = nn.Sequential(
                nn.Conv1d(200, 200, kernel_size=3),
                nn.BatchNorm1d(num_features=size_inputs-2),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(size_inputs-2, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

    def forward(self, x):
        output = self.model(x)

        return output


class Generator(nn.Module):
    def __init__(
            self, size_inputs=1, size_outputs=1,
            noise0=1, noise_end=5e-2, n_epochs=100,
            batch_size=100,
            nn_type='linear',
            dropout=0.3,
            data_type='car'
    ):
        super().__init__()

        self.hidden_dim = 256
        self.n_layers = 2
        self.data_type = data_type
        self._initialise_model(size_inputs, dropout, size_outputs, nn_type, batch_size)
        self.noise0 = noise0
        self.noise_reduction = math.exp(math.log(noise_end / noise0) / n_epochs)
        self.noise_factor = self.noise0

    def _initialise_model(self, size_inputs, dropout, size_outputs, nn_type, batch_size):
        self.nn_type = nn_type
        self.size_outputs = size_outputs
        if nn_type == 'linear':
            self.model = nn.Sequential(
                nn.Linear(size_inputs, 16),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, size_outputs),
                nn.Sigmoid(),
            )

        elif nn_type == 'cnn':
            self.fc = nn.Sequential(
                nn.Linear(size_inputs, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, size_outputs * 8),
                nn.ReLU(),
            )

            self.conv = nn.Sequential(
                nn.ConvTranspose1d(
                    size_outputs * 8, size_outputs * 4, kernel_size=3, stride=2, padding=1,
                ),
                nn.BatchNorm1d(size_outputs * 4),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.ConvTranspose1d(
                    size_outputs * 4, size_outputs * 2, kernel_size=3, stride=2, padding=1,
                ),
                nn.BatchNorm1d(size_outputs * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.ConvTranspose1d(
                    size_outputs * 2, size_outputs, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm1d(size_outputs),
                nn.ReLU(),
            )
        elif nn_type == 'rnn':
            # Defining the layers

            # RNN Layer
            self.rnn = nn.RNN(size_inputs, self.hidden_dim, self.n_layers, batch_first=True)
            # Fully connected layer
            self.fc = nn.Linear(self.hidden_dim, size_outputs)

        elif nn_type == 'lstm':
            self.lstm = nn.LSTM(size_inputs, self.hidden_dim, self.n_layers, batch_first=True)
            self.fc = nn.Linear(self.hidden_dim, size_outputs)

    def forward(self, x):
        if self.nn_type == 'linear':
            output = self.model(x)
        elif self.nn_type == 'cnn':
            x = self.fc(x)
            x = x.view(-1, 8 * self.size_outputs, 1)
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

        noise = th.randn(output.shape) * self.noise_factor
        output = th.clamp(output + noise, min=0, max=1)
        if self.data_type == 'gen':
            output = output.reshape(-1, 24)
            output = th.div(output, th.sum(output, dim=1).reshape(-1, 1)).reshape(-1, self.size_outputs)
        # output = th.exp(output)

        return output

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = th.zeros(self.n_layers, self.hidden_dim)
        return hidden


def compute_profile_generators(
        profiles, k, percentiles_inputs, data_type,
        day_type, prm
):
    print("profile generators")
    params = {
        'profiles': True,
        'batch_size': 100,
        'n_epochs': int(1e8 / len(profiles)),
        'weight_sum_profiles': 1e-7,
        'weight_diff_percentiles': 100,
        'size_input_discriminator_one_item': prm['n'],
        'size_output_generator_one_item': prm['n'],
        'k': k,
        'percentiles_inputs': percentiles_inputs,
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
    }

    params['lr_decay'] = (params['lr_end'] / params['lr_start']) ** (1 / params['n_epochs'])
    params['size_input_generator_one_item'] = params['dim_latent_noise']
    gan_trainer = GAN_Trainer(params, prm)
    gan_trainer.update_value_type('profiles')
    gan_trainer.add_training_data(outputs=profiles)
    gan_trainer.train()
