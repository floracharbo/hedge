import math
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from tqdm import tqdm

from src.data_preparation.scaling_factors import _transition_intervals
from src.utils import initialise_dict

th.manual_seed(111)


def obtain_cluster_transition_probabilities(n_clusters, n_consecutive_days, list_inputs, list_outputs):
    n_possible_inputs = n_clusters ** (n_consecutive_days - 1)
    n_transitions = np.zeros((n_possible_inputs, n_clusters))
    for i in range(len(list_inputs)):
        clusters_input_idx = 0
        for d in range(n_consecutive_days - 1):
            clusters_input_idx += list_inputs[i][-(d + 1)] * n_clusters ** d
        if isinstance(list_outputs[i], list) and len(list_outputs[i]) == 1:
            cluster_output_idx = int(list_outputs[i][0])
        else:
            cluster_output_idx = int(list_outputs[i])
        # clusters_input = list_inputs[i][0] * n_clus_all_ + list_inputs[i][1]
        n_transitions[int(clusters_input_idx)][cluster_output_idx] += 1
    sum_transitions = np.sum(n_transitions, axis=1)
    cluster_transition_propabilities = [
        n_transitions[c] / sum_transitions[c]
        for c in range(n_possible_inputs)
    ]

    return cluster_transition_propabilities

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

    def cluster_generated_output_to_idx(self, real_inputs, generated_outputs):
        n_possible_clusters = int(real_inputs.max()) + 1
        generated_outputs_ = th.zeros((self.batch_size_, self.n_items_generated))
        for i_batch in range(self.batch_size_):
            for i_item in range(self.n_items_generated):
                generated_outputs_[i_batch][i_item] \
                    = generated_outputs[
                      i_batch,
                      i_item * n_possible_clusters: (i_item + 1) * n_possible_clusters
                      ].argmax()

        return generated_outputs_

    def update_value_type(self, value_type):
        save_folder = f"{self.data_type}_{value_type}_generation"
        self.value_type = value_type
        self.save_path = self.general_saving_folder / save_folder
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.normalised = True if value_type == 'factors' else False
        if not self.profiles:
            self.size_output_generator_one_item = self.n_clusters if value_type == 'clusters' else 1

    def get_saving_label(self):
        saving_label = f'{self.data_type} {self.value_type}'
        if not self.profiles:
            saving_label += f" {self.transition} n_consecutive_days {self.n_consecutive_days}"
        else:
            saving_label += f" cluster {self.k}"
    
        if self.lr_decay != 1:
            saving_label += f" lr_decay {self.lr_decay:.3f}".replace('.', '_')

        return saving_label

    def add_training_data(
            self, inputs=None, outputs=None, p_clus=None, p_trans=None, n_clus=None,
    ):
        if self.value_type == 'factors':
            i_remove = [
                i for i in range(len(inputs))
                if any(f == 0 for f in inputs[i]) or outputs[i] == 0
            ]
            print(f"remove {len(i_remove) / len(outputs)} % of samples with 0")
            for it, i in enumerate(i_remove):
                if i - it == 0:
                    inputs = inputs[1:]
                    outputs = outputs[1:]
                else:
                    try:
                        inputs = np.concatenate((inputs[0: i - it], inputs[i - it + 1:]))
                        outputs = np.concatenate((outputs[0: i - it], outputs[i - it + 1:]))
                    except Exception as ex:
                        print(
                            f"ex {ex} self.data_type {self.data_type} self.value_type {self.value_type} "
                            f"it {it} i {i}"
                            f"np.shape(inputs) {np.shape(inputs)} np.shape(outputs) {np.shape(outputs)}"
                            f"inputs {inputs} outputs {outputs}"
                        )
                        sys.exit()

        self.train_data_length = len(outputs)
        self.outputs = th.tensor(outputs)

        if not self.profiles:
            self.inputs = inputs

        if self.normalised:
            for i in range(self.train_data_length):
                if inputs is not None:
                    self.inputs[i] = self.inputs[i] / self.inputs[i][0]
                self.outputs[i] = self.outputs[i] / self.inputs[i][0]

        self.plot_inputs()

        self.mean_real_output = np.nanmean(self.outputs)
        self.std_real_output = np.nanstd(self.outputs)

        if not self.profiles:
            self.train_data = th.zeros((self.train_data_length, self.n_consecutive_days))
            self.train_data[:, :-1] = th.tensor(self.inputs)
            self.train_data[:, -1] = th.tensor(
                np.array(self.outputs)
            ).view_as(self.train_data[:, -1])
        else:
            self.train_data = self.outputs

        if p_clus is not None:
            self.p_clus0, self.p_trans0, self.n_clus = p_clus, p_trans, n_clus

    def plot_inputs(self):
        fig = plt.figure()
        for i in range(20):
            if self.profiles:
                plt.plot(self.outputs[i])
            else:
                plt.plot(np.append(self.inputs[i], self.outputs[i]))

        title = f"{self.get_saving_label()} series"
        if not self.profiles:
            title += f" n_consecutive_days {self.n_consecutive_days}"
        if self.normalised:
            title += ' normalised'
        plt.title(title)
        title = title.replace(' ', '_')
        fig.savefig(self.save_path / title)
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
        self.size_input_discriminator = self.size_input_discriminator_one_item * self.n_items_generated
        self.size_input_generator = self.size_input_generator_one_item if self.profiles else self.size_input_generator_one_item * self.n_items_generated

    def split_inputs_and_outputs(self, train_data):
        if self.profiles:
            real_inputs = th.randn(self.batch_size_, 1)
            real_outputs = train_data
        elif self.n_items_generated > 1:
            i_inputs = np.array(
                [i for i in range(len(train_data[0])) if (i + 1) % self.n_consecutive_days != 0]
            )
            i_outputs = np.array(
                [i for i in range(len(train_data[0])) if (i + 1) % self.n_consecutive_days == 0]
            )
            real_inputs = train_data[:, i_inputs]
            real_outputs = train_data[:, i_outputs]

        else:
            real_inputs = train_data[:, :self.size_input_generator]
            real_outputs = train_data[:, self.size_input_generator:]

        return real_inputs, real_outputs

    def merge_inputs_and_outputs(self, real_inputs, generated_outputs, real_outputs=None):
        if real_outputs is None:
            real_samples = None
        elif self.profiles:
            real_samples = real_outputs
        else:
            real_samples = self.insert_generated_outputs_in_inputs_list(real_inputs, real_outputs)

        if self.profiles:
            generated_samples = generated_outputs
        else:
            generated_samples = self.insert_generated_outputs_in_inputs_list(real_inputs, generated_outputs)

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
            n_epochs=self.n_epochs
        )
        self.discriminator = Discriminator(size_inputs=self.size_input_discriminator)

        self.loss_function = nn.BCELoss()
        self.optimizer_discriminator = th.optim.Adam(
            self.discriminator.parameters(), lr=self.lr_start
        )
        self.optimizer_generator = th.optim.Adam(self.generator.parameters(), lr=self.lr_start)

    def train_discriminator(self, real_inputs, real_outputs):
        generated_outputs = self.generator(real_inputs.to(th.float32))
        if self.value_type == 'clusters':
            generated_outputs = self.cluster_generated_output_to_idx(real_inputs, generated_outputs)
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

    def plot_clusters_transitions_generated_vs_real(self, real_inputs, generated_outputs, epoch):
        # need to reshape inputs
        inputs_reshape = np.reshape(real_inputs, ((-1, self.n_consecutive_days - 1)))
        outputs_reshape = np.reshape(generated_outputs, ((-1, 1)))
        generated_ps = obtain_cluster_transition_probabilities(
            self.n_clusters, self.n_consecutive_days, inputs_reshape, outputs_reshape
        )
        fig, axs = plt.subplots(2)
        axs[0].imshow(generated_ps, label='generated')
        if 'cluster_transition_propabilities' in self.__dict__:
            axs[1].imshow(self.cluster_transition_propabilities, label='real')
        plt.legend()
        title = f"{self.get_saving_label()} hist clusters transitions generated vs real epoch {epoch}"
        if self.normalised:
            title += ' normalised'
        title = title.replace(' ', '_')
        fig.savefig(self.save_path / title)
        plt.close('all')

    def train_generator(self, real_inputs, real_outputs, final_n, epoch):
        self.generator.zero_grad()
        generated_outputs = self.generator(real_inputs.to(th.float32))

        if self.value_type == 'clusters':
            generated_outputs = self.cluster_generated_output_to_idx(real_inputs, generated_outputs)
        generated_samples, _ = self.merge_inputs_and_outputs(real_inputs, generated_outputs)
        # generated_samples = th.where(generated_samples < 0, 0, generated_samples)
        output_discriminator_generated = self.discriminator(generated_samples)

        loss_generator = self.loss_function(
            output_discriminator_generated, self.get_real_samples_labels()
        )
        if self.profiles:
            loss_generator += (
                th.sum(generated_samples) / (self.batch_size_ * self.n_items_generated) - 1
            ) ** 2 * self.weight_sum_profiles
        loss_generator.backward()
        self.optimizer_generator.step()
        if final_n:
            generated_samples_2d = np.reshape(
                generated_samples.detach().numpy(),
                (self.batch_size_ * self.n_items_generated, -1)
            )
            if self.value_type == 'clusters':
                self.plot_final_hist_generated_vs_real(generated_outputs, real_outputs, epoch)
                self.plot_clusters_transitions_generated_vs_real(real_inputs, generated_outputs, epoch)
            else:
                self.plot_generated_samples_start_epoch(generated_samples, epoch)

            if self.profiles:
                self.plot_statistical_indicators_profiles(generated_samples_2d, epoch)

            if epoch == self.n_epochs - 1:
                if self.value_type == 'clusters':
                    self.check_clusters_transition_probabilities_match_real(generated_samples_2d)
                else:
                    self.plot_final_hist_generated_vs_real(generated_outputs, real_outputs, epoch)
                if self.value_type == 'factors':
                    self.plot_heat_map_f_prev_next(generated_samples_2d, epoch=epoch)

        return loss_generator, generated_outputs

    def plot_heat_map_f_prev_next(self, generated_samples_2d, epoch=None):
        consecutive_factors = generated_samples_2d[:, :-1]
        # f_prev = generated_samples_2d[:, self.n_consecutive_days - 2]
        # f_next = generated_samples_2d[:, self.n_consecutive_days - 1]
        _transition_intervals(consecutive_factors, self.transition + f"_{epoch}", self.prm, self.data_type, n_consecutive_days=n_consecutive_days)

    def check_clusters_transition_probabilities_match_real(self, generated_samples_2d):
        if self.value_type == 'clusters':
            n_transitions = np.zeros((self.n_clusters, self.n_clusters))
            n_clus = np.zeros((self.n_clusters))

            for i in range(len(generated_samples_2d)):
                c1 = int(generated_samples_2d[i][-2])
                c2 = int(generated_samples_2d[i][-1])
                n_transitions[c1, c2] += 1
                n_clus[c2] += 1
            p_transitions_generated = [
                n_transitions[i] / np.sum(n_transitions[i]) for i in range(self.n_clusters)
            ]
            p_clus_generated = n_clus / np.sum(n_clus)
            print(f"p_transitions generated {p_transitions_generated}")
            print(f"p_clus_generated generated {p_clus_generated}")
            if 'p_clus0' in self.__dict__:
                print(f"p_clus inputs {self.p_clus0}")
            if 'p_trans0' in self.__dict__:
                print(f"p_trans inputs {self.p_trans0}")
            saving_label = self.get_saving_label()
            np.save(f"p_transitions_{saving_label}", p_transitions_generated)
            np.save(f"p_clus_generated_{saving_label}", p_clus_generated)

    def plot_statistical_indicators_profiles(self, generated_samples_2d, epoch):
        n_samples = len(generated_samples_2d[0])
        statistical_indicators_generated = {}
        for statistical_indicator in ['p10', 'p50', 'p90', 'mean']:
            statistical_indicators_generated[statistical_indicator] = np.zeros(n_samples)
        for time in range(n_samples):
            for percentile in [10, 50, 90]:
                statistical_indicators_generated[f'p{percentile}'][time] = np.percentile(
                    generated_samples_2d[:, time],
                    percentile
                )
            statistical_indicators_generated['mean'][time] = np.mean(generated_samples_2d[:, time])

        fig = plt.figure()
        xs = np.arange(n_samples)
        for color, statistical_indicators_, label in zip(
                ['b', 'g'],
                [statistical_indicators_generated, self.statistical_indicators_inputs[self.k]],
                ['generated', 'original']
        ):
            for indicator in ['p10', 'p50', 'p90', 'mean']:
                plt.plot(
                    xs,
                    statistical_indicators_[indicator],
                    color=color,
                    linestyle='--' if indicator[0] == 'p' else '-',
                    label=label if indicator == 'mean' else None,
                )

        plt.legend()
        title = f"{self.get_saving_label()} profiles generated vs original epoch {epoch}"
        if self.normalised:
            title += ' normalised'
        plt.title(title)
        title = title.replace(' ', '_')
        fig.savefig(self.save_path / title)
        plt.close('all')

    def plot_generated_samples_start_epoch(self, generated_samples, epoch):
        saving_label = self.get_saving_label()
        fig = plt.figure()
        for i in range(self.batch_size_):
            plt.plot(generated_samples[i].detach().numpy()[1:])
        title = f'{saving_label} generated samples epoch {epoch}'
        if self.normalised:
            title += ' normalised'
        plt.title(title)
        title = title.replace(' ', '_')
        fig.savefig(self.save_path / title)
        plt.close('all')

    def update_noise_and_lr_generator(self, epoch):
        self.generator.noise_factor *= self.generator.noise_reduction
        for g in self.optimizer_generator.param_groups:
            g['lr'] = self.lr_start * self.lr_decay ** epoch
        for g in self.optimizer_discriminator.param_groups:
            g['lr'] = self.lr_start * self.lr_decay ** epoch

    def plot_losses_over_time(
            self, losses_generator, losses_discriminator, means_outputs, stds_outputs
    ):
        title = f"{self.get_saving_label()} losses "
        if not self.profiles:
            title += "mean std "
        title += "over time"
        if self.normalised:
            title += ' normalised'
        if self.profiles:
            fig = plt.figure()
            plt.plot(losses_generator, label="losses_generator")
            plt.plot(losses_discriminator, label="losses_discriminator")
            plt.legend()
        else:
            fig, axs = plt.subplots(3)
            axs[0].plot(losses_generator, label="losses_generator")
            axs[0].plot(losses_discriminator, label="losses_discriminator")
            axs[0].legend()
            axs[1].plot(means_outputs, label="mean output")
            axs[1].legend()
            print(f"mean_real_output {self.mean_real_output}")
            axs[1].hlines(
                self.mean_real_output, 0, len(means_outputs),
                color='red', linestyle='dashed'
            )
            axs[2].plot(stds_outputs, label="std output")
            axs[2].legend()
            axs[2].hlines(
                self.std_real_output, 0, len(means_outputs),
                color='red', linestyle='dashed'
            )
            print(f"title {title}")
            print(f"np.mean(stds_outputs) {np.mean(stds_outputs)} vs self.std_real_output {self.std_real_output}")
            print(f"np.mean(means_outputs) {np.mean(means_outputs)} vs self.mean_real_output {self.mean_real_output}")


        title = title.replace(' ', '_')
        fig.savefig(self.save_path / title)
        plt.close('all')

    def plot_noise_over_time(self):
        fig = plt.figure()
        noises = [
            self.generator.noise0 * self.generator.noise_reduction ** epoch
            for epoch in range(self.n_epochs)
        ]
        plt.plot(noises)
        title = f"{self.get_saving_label()} noise over time"
        plt.title(title)
        fig.savefig(self.save_path / title)

    def plot_final_hist_generated_vs_real(self, generated_outputs, real_outputs, epoch):
        nbins = self.n_clusters if self.value_type == 'clusters' else 100
        generated_outputs_reshaped = np.array(generated_outputs.detach().numpy(), dtype=int).flatten()
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
        fig.savefig(self.save_path / title)
        plt.close('all')

    def train(self):
        self.get_train_loader()
        self.initialise_generator_and_discriminator()
        losses_generator, losses_discriminator = [], []
        means_outputs, stds_outputs = [], []

        for epoch in tqdm(range(self.n_epochs)):
            for n, train_data in enumerate(self.train_loader):
                self.batch_size_ = len(train_data)
                real_inputs, real_outputs = self.split_inputs_and_outputs(train_data)

                loss_discriminator = self.train_discriminator(real_inputs, real_outputs)
                final_n = n == len(self.train_loader) - 1
                loss_generator, generated_outputs = self.train_generator(
                    real_inputs, real_outputs, final_n, epoch
                )
                losses_generator.append(loss_generator.detach().numpy())
                losses_discriminator.append(loss_discriminator.detach().numpy())
                means_outputs.append(np.mean(generated_outputs.detach().numpy()))
                stds_outputs.append(np.std(generated_outputs.detach().numpy()))

            self.update_noise_and_lr_generator(epoch)

        self.plot_losses_over_time(
            losses_generator, losses_discriminator, means_outputs, stds_outputs
        )

        self.plot_noise_over_time()
        print(
            f"mean generated outputs last 10: {np.mean(means_outputs[-10:])}, "
            f"std {np.mean(stds_outputs[-10:])}"
        )
        th.save(self.generator.model, self.save_path / f"generator_{self.get_saving_label()}.pt")


class Discriminator(nn.Module):
    def __init__(self, size_inputs=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self, size_inputs=1, size_outputs=1, noise0=1, noise_end=5e-2, n_epochs=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(size_inputs, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, size_outputs),
        )
        self.noise0 = noise0
        self.noise_reduction = math.exp(math.log(noise_end/noise0)/n_epochs)
        self.noise_factor = self.noise0

    def forward(self, x):
        output = self.model(x)
        noise = th.randn(output.shape) * self.noise_factor
        output = output + noise
        output = th.exp(output)

        return output


def compute_profile_generators(
        profiles, n, k, statistical_indicators, data_type, general_saving_folder, prm
):
    print("profile generators")
    params = {
        'profiles': True,
        'batch_size': 100,
        'n_epochs': 200,
        'lr_start': 0.001,
        'lr_end': 0.0001,
        'weight_sum_profiles': 0.5 if data_type == 'loads' else 0,
        'size_input_discriminator_one_item': n,
        'size_input_generator_one_item': 1,
        'size_output_generator_one_item': n,
        'k': k,
        'statistical_indicators_inputs': statistical_indicators,
        'general_saving_folder': general_saving_folder,
        'data_type': data_type,
        'n_items_generated': 50,
    }
    params['lr_decay'] = (params['lr_end'] / params['lr_start']) ** (1 / params['n_epochs'])
    gan_trainer = GAN_Trainer(params, prm)
    gan_trainer.update_value_type('profiles')
    gan_trainer.add_training_data(outputs=profiles)
    gan_trainer.train()


def compute_factors_clusters_generators(
        prm, n_data_type, data_type, days, p_clus, p_trans, n_clus_all_
):
    print(f"data_type {data_type}")
    n_consecutive_days = 3
    training_params = {
        'batch_size': 100,
        'n_epochs': 20, #100,
        'lr_start': 0.001,
        'n_consecutive_days': n_consecutive_days,
        'data_type': data_type,
        'general_saving_folder': prm['save_other'],
        'size_input_generator_one_item': n_consecutive_days - 1,
        'size_output_generator_one_item': 1,
        'size_input_discriminator_one_item': n_consecutive_days,
        'n_items_generated': 50,
        'profiles': False,
        'lr_decay': 1,
        'n_clusters': n_clus_all_,
    }
    list_inputs, list_outputs = [initialise_dict(['factors', 'clusters']) for _ in range(2)]
    list_inputs['factors'], list_outputs['factors'] = [
        {el: [] for el in ['zero_start', 'non_zero_start']}
        for _ in range(2)
    ]
    list_inputs['clusters'], list_outputs['clusters'] = [
        {el: [] for el in prm['day_trans']}
        for _ in range(2)
    ]
    for i in range(n_data_type[data_type] - (n_consecutive_days - 1)):
        days_ = [days[data_type][i_] for i_ in range(i, i + n_consecutive_days)]
        same_id = all(days_[1 + i_]["id"] == days_[0]["id"] for i_ in range(n_consecutive_days - 1))
        subsequent_days = all(
            days_[0]["cum_day"] + i_ == days_[i_]["cum_day"]
            for i_ in range(1, n_consecutive_days)
        )
        if same_id and subsequent_days:  # record transition
            d1 = days_[-2]['day_type']
            d2 = days_[-1]['day_type']
            transition = f"{d1}2{d2}"
            # clusters
            day_types, index_wdt = [
                [day_[key] for day_ in days_]
                for key in ["day_type", "index_wdt"]
            ]
            clusters = [
                days[f"{data_type}_{day_types[i]}"][index_wdt[i]]["cluster"]
                for i in range(n_consecutive_days)
            ]

            list_inputs['clusters'][transition].append(clusters[:-1])
            list_outputs['clusters'][transition].append(clusters[-1])

            # factors
            c1_zero = 'zero_start' if data_type == 'car' and clusters[0] == n_clus_all_ - 1 else 'non_zero_start'
            c2_zero = data_type == 'car' and clusters[1] == n_clus_all_ - 1
            if not c2_zero:
                list_inputs['factors'][c1_zero].append(
                    [days_[i_]['factor'] for i_ in range(n_consecutive_days - 1)]
                )
                list_outputs['factors'][c1_zero].append([days_[-1]["factor"]])

    gan_trainer = GAN_Trainer(training_params, prm)

    for value_type in ['factors', 'clusters']:
        gan_trainer.update_value_type(value_type)
        for training_data, label in zip([list_inputs, list_outputs], ["inputs", "outputs"]):
            file_name = f"{label}_{data_type}_n_days{n_consecutive_days}_{value_type}.pickle"
            with open(gan_trainer.save_path / file_name, "wb") as f:
                pickle.dump(training_data, f)

    # gan_trainer.update_value_type('clusters')
    for transition in prm['day_trans']:
        print(f"clusters: transition {transition}")
        cluster_transition_propabilities = obtain_cluster_transition_probabilities(
            n_clus_all_, n_consecutive_days, list_inputs['clusters'][transition], list_outputs['clusters'][transition]
        )
        with open(prm["save_other"] / f"{data_type}_clusters_generation" / f"cluster_transition_probabilities_{transition}.pickle", "wb") as file:
            pickle.dump(cluster_transition_propabilities, file)
        # gan_trainer.cluster_transition_propabilities = cluster_transition_propabilities
        # gan_trainer.transition = transition
        # gan_trainer.add_training_data(
        #     list_inputs['clusters'][transition],
        #     list_outputs['clusters'][transition],
        #     p_clus,
        #     p_trans,
        #     n_clus_all_,
        # )
        #
        # gan_trainer.train()

    gan_trainer.update_value_type('factors')
    for c1_zero in ['zero_start', 'non_zero_start']:
        print(f"factors: {c1_zero}")
        gan_trainer.transition = c1_zero
        if len(list_outputs['factors'][c1_zero]) == 0:
            print(
                f"NO TRAINING: len(list_outputs['factors'][{c1_zero}]) {len(list_outputs['factors'][c1_zero])} "
                f"gan_trainer.data_type {gan_trainer.data_type} "
            )
        else:
            gan_trainer.add_training_data(
                list_inputs['factors'][c1_zero],
                list_outputs['factors'][c1_zero],
                p_clus,
                p_trans,
                n_clus_all_
            )
            gan_trainer.train()
