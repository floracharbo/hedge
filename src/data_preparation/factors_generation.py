import torch as th
from torch import nn
import pickle
import copy
import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
import sys

import math
import matplotlib.pyplot as plt

th.manual_seed(111)


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
            nn.Linear(32, size_outputs),
        )
        self.noise0 = noise0
        self.noise_reduction = math.exp(math.log(noise_end/noise0)/n_epochs)
        self.noise_factor = self.noise0

    def forward(self, x):
        output = self.model(x)
        noise = th.rand(output.shape) * self.noise_factor
        output = output + noise
        return output


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

def training(
    train_data_all, factors_generation_save_path,
    mean_real_output=None, std_real_output=None,
    size_input_discriminator=1,
    size_input_generator=1, size_output_generator=1,
    label_saving=None, normalised=True, k=0, profiles=False,
    statistical_indicators=None, batch_size=32, lr=0.001, n_epochs=20
):
    n_items_generated = 50
    train_data_all = th.tensor(train_data_all)
    i_not_nans = [i for i, train_data_i in enumerate(train_data_all) if not any(th.isnan(train_data_i))]
    train_data_all = train_data_all[i_not_nans]
    train_data_cut = train_data_all[:-(train_data_all.size()[0] % n_items_generated)] if (train_data_all.size()[0] % n_items_generated) > 0 else train_data_all
    train_data_cut = train_data_cut.view((-1, train_data_cut.size()[1] * n_items_generated))
    train_loader = th.utils.data.DataLoader(
        train_data_cut, batch_size=batch_size, shuffle=True
    )
    size_output_generator *= n_items_generated
    size_input_discriminator *= n_items_generated
    generator = Generator(
        size_inputs=size_input_generator, size_outputs=size_output_generator, n_epochs=n_epochs
    )
    discriminator = Discriminator(size_inputs=size_input_discriminator)

    loss_function = nn.BCELoss()

    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=lr)

    losses_generator = []
    losses_discriminator = []
    means_outputs = []
    stds_outputs = []
    for epoch in tqdm(range(n_epochs)):
        for n, train_data in enumerate(train_loader):
            batch_size_ = len(train_data)
            if profiles:
                real_inputs = th.rand(batch_size_, 1)
                real_outputs = train_data
            else:
                real_inputs = train_data[:, :size_input_generator]
                real_outputs = train_data[:, size_input_generator:]

            # Data for training the discriminator
            real_samples_labels = th.ones((batch_size_, 1))
            # latent_space_samples = th.randn((batch_size, 2))

            generated_outputs = generator(real_inputs.to(th.float32))
            generated_samples_labels = th.zeros((batch_size_, 1))

            checks_nan(real_inputs, real_outputs, generated_outputs)
            if profiles:
                real_samples = real_outputs
                generated_samples = generated_outputs
            else:
                real_samples = th.cat((real_inputs, real_outputs), dim=1)
                generated_samples = th.cat(
                    (real_inputs.view((batch_size_, size_input_generator)), generated_outputs.view((batch_size_, 1))),
                    dim=1
                )

            all_samples = th.cat((real_samples, generated_samples))
            if any(any(th.isnan(real_samples_i)) for real_samples_i in real_samples):
                print('nan in real_samples')
                sys.exit()
            all_samples_labels = th.cat(
                (real_samples_labels, generated_samples_labels)
            )
            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples.to(th.float32))
            try:
                loss_discriminator = loss_function(
                    output_discriminator, all_samples_labels
                )
            except Exception as ex:
                print(ex)
                print(
                    f"output_discriminator {output_discriminator} "
                    f"all_samples {all_samples} "
                    f"output_discriminator.size() {output_discriminator.size()}"
                )
                if any(any(th.isnan(real_samples_i)) for real_samples_i in real_samples):
                    print('nan in real_samples')
                if any(any(th.isnan(generated_samples_i)) for generated_samples_i in generated_samples):
                    print('nan in generated_samples')
                print(f"all_samples[9][1].isnan() {all_samples[9][1].isnan()}")
                sys.exit()

            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Training the generator
            generator.zero_grad()
            generated_outputs = generator(real_inputs.to(th.float32))
            if profiles:
                generated_samples = generated_outputs
            else:
                generated_samples = th.cat(
                    (real_inputs.view((batch_size_, size_input_generator)), generated_outputs.view((batch_size_, 1))),
                    dim=1
                )
            generated_samples = th.where(generated_samples < 0, 0, generated_samples)
            output_discriminator_generated = discriminator(generated_samples)

            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )

            loss_generator += (th.sum(generated_samples)/(batch_size_ * n_items_generated) - 1) ** 2 * 0.5
            loss_generator.backward()
            optimizer_generator.step()

            losses_generator.append(loss_generator.detach().numpy())
            losses_discriminator.append(loss_discriminator.detach().numpy())
            means_outputs.append(np.mean(generated_outputs.detach().numpy()))
            stds_outputs.append(np.std(generated_outputs.detach().numpy()))

        generator.noise_factor *= generator.noise_reduction

        fig = plt.figure()
        for i in range(batch_size_):
            plt.plot(generated_samples[i].detach().numpy()[1:])
        title = f'{label_saving} generated samples epoch {epoch}'
        if normalised:
            title += ' normalised'
        plt.title(title)
        title = title.replace(' ', '_')
        fig.savefig(factors_generation_save_path / title)
        plt.close('all')

        if profiles:
            generated_samples_2d = np.reshape(
                generated_samples.detach().numpy(),
                (batch_size_ * n_items_generated, -1)
            )
            print(f"generated_samples = {generated_samples_2d}")
            n = len(generated_samples_2d[0])
            statistical_indicators_generated = {}
            for statistical_indicator in ['p10', 'p50', 'p90', 'mean']:
                statistical_indicators_generated[statistical_indicator] = np.zeros(n)
            print(f"np.shape(generated_samples) {np.shape(generated_samples_2d)}")
            for time in range(n):
                for percentile in [10, 50, 90]:
                    statistical_indicators_generated[f'p{percentile}'][time] = np.percentile(
                        generated_samples_2d[:, time],
                        percentile
                    )
                statistical_indicators_generated['mean'][time] = np.mean(generated_samples_2d[:, time])

            fig = plt.figure()
            xs = np.arange(n)
            for color, statistical_indicators_, label in zip(['b', 'g'], [statistical_indicators_generated, statistical_indicators[k]], ['generated', 'original']):
                for percentile in [10, 50, 90]:
                    plt.plot(
                        xs,
                        statistical_indicators_[f'p{percentile}'],
                        color=color,
                        linestyle='-' if percentile==50 else '--',
                        label=label if percentile==50 else None,
                    )

                print(f"{label} statistical_indicators_['p50'] {statistical_indicators_['p50']}")
                print(f"{label} statistical_indicators_['p90'] {statistical_indicators_['p90']}")

            plt.legend()
            title = f"{label_saving} profiles generated vs original epoch {epoch}"
            if normalised:
                title += ' normalised'
            plt.title(title)
            title = title.replace(' ', '_')
            fig.savefig(factors_generation_save_path / title)
            plt.close('all')

    if not profiles:
        fig, axs = plt.subplots(3)
        axs[0].plot(losses_generator, label="losses_generator")
        axs[0].plot(losses_discriminator, label="losses_discriminator")
        axs[0].legend()
        axs[1].plot(means_outputs, label="mean output")
        axs[1].legend()
        print(f"mean_real_output {mean_real_output}")
        axs[1].hlines(mean_real_output, 0, len(means_outputs), color='red', linestyle='dashed')
        axs[2].plot(stds_outputs, label="std output")
        axs[2].legend()
        axs[2].hlines(std_real_output, 0, len(means_outputs), color='red', linestyle='dashed')
    else:
        fig = plt.figure()
        plt.plot(losses_generator, label="losses_generator")
        plt.plot(losses_discriminator, label="losses_discriminator")
        plt.legend()
    title = f"{label_saving} losses "
    if not profiles:
        title += "mean std "
    title+= "over time"
    if normalised:
        title += ' normalised'
    title = title.replace(' ', '_')
    fig.savefig(factors_generation_save_path / title)
    plt.close('all')

    fig = plt.figure()
    noises = [generator.noise0 * generator.noise_reduction ** epoch for epoch in range(n_epochs)]
    plt.plot(noises)
    plt.title('noise vs epoch')
    fig.savefig(factors_generation_save_path / 'noise_vs_epoch')

    print(f"Saved {title}, mean_real_output {mean_real_output}, std_real_output {std_real_output}")
    print(f"mean generated outputs last 10: {np.mean(means_outputs[-10:])}, std {np.mean(stds_outputs[-10:])}")

    fig = plt.figure()
    plt.hist(generated_outputs.detach().numpy(), bins=100, alpha=0.5, label='generated')
    plt.hist(real_outputs, bins=100, alpha=0.5, label='real')
    plt.legend()
    title = f"{label_saving} hist generated vs real"
    if normalised:
        title += ' normalised'
    title = title.replace(' ', '_')
    fig.savefig(factors_generation_save_path / title)
    plt.close('all')

    th.save(generator.model, factors_generation_save_path / f"generator_{label_saving}.pt")

def get_factors_generator(
        n_consecutive_days, list_inputs0, list_outputs0, factors_generation_save_path, data_type, value_type
):
    batch_size = 32
    lr = 0.001
    n_epochs = 20
    list_inputs0 = np.array(list_inputs0)
    list_outputs0 = np.array(list_outputs0)
    print(f"data_type {data_type}, value_type {value_type}")
    if data_type == 'loads':
        i_remove = [
            i for i in range(len(list_inputs0))
            if any(f == 0 for f in list_inputs0[i][1:]) or list_outputs0[i] == 0
        ]
        print(f"remove {len(i_remove)/len(list_inputs0)} % of samples with 0")
        for it, i in enumerate(i_remove):
            list_inputs0 = np.concatenate((list_inputs0[0: i - it], list_inputs0[i - it + 1:]))
    train_data_length = len(list_inputs0)

    if not factors_generation_save_path.exists():
        os.makedirs(factors_generation_save_path)

    for normalised in [True, False]:
        list_inputs = copy.deepcopy(list_inputs0)
        list_outputs = copy.deepcopy(list_outputs0)
        if normalised:
            for i in range(train_data_length):
                list_inputs[i, 1:] = list_inputs0[i, 1:] / list_inputs0[i, 1]
                list_outputs[i] = list_outputs0[i] / list_inputs0[i, 1]
        fig = plt.figure()
        for i in range(20):
            plt.plot(np.append(list_inputs[i][1:], list_outputs[i]))
        title = f"{data_type} n_consecutive_days {n_consecutive_days} example {value_type} series"
        if normalised:
            title += ' normalised'
        plt.title(title)
        title = title.replace(' ', '_')
        fig.savefig(factors_generation_save_path / title)
        plt.close('all')

        mean_real_output = np.nanmean(list_outputs)
        std_real_output = np.nanstd(list_outputs)
        train_data = th.zeros((train_data_length, n_consecutive_days + 1))
        train_data[:, :-1] = th.tensor(list_inputs)
        train_data[:, -1] = th.tensor(np.array(list_outputs)).view_as(train_data[:, -1])
        training(
            train_data, factors_generation_save_path, normalised=normalised,
            size_input_generator=n_consecutive_days, size_output_generator=1,
            size_input_discriminator=n_consecutive_days + 1,
            label_saving=f'{data_type} n_consecutive_days {n_consecutive_days} {value_type}',
            batch_size=batch_size, lr=lr, n_epochs=n_epochs, mean_real_output=mean_real_output,
            std_real_output=std_real_output
        )


def compute_profile_generator(profiles, profiles_generation_save_path, label_saving, n, k, statistical_indicators):
    if not profiles_generation_save_path.exists():
        profiles_generation_save_path.mkdir(parents=True)
    batch_size = 100
    lr = 0.0001
    n_epochs = 100
    training(
        profiles, profiles_generation_save_path,
        size_input_discriminator=n,
        size_input_generator=1, size_output_generator=n,
        label_saving=label_saving, normalised=False,
        k=k, profiles=True, statistical_indicators=statistical_indicators,
        batch_size=batch_size, lr=lr, n_epochs=n_epochs
    )


# save_other_path = Path("data/other_outputs/n24_loads_10000000_gen_10000000_car_10000000")

# n_consecutive_days = 3
# with open(save_other_path / "inputs_loads_n_days3.pickle", "rb") as f:
#     list_inputs0 = pickle.load(f)
# with open(save_other_path / "outputs_loads_n_days3.pickle", "rb") as f:
#     list_outputs0 = pickle.load(f)
# data_type = 'loads'
#
#
# get_factors_generator(n_consecutive_days, list_inputs0, list_outputs0, save_other_path, data_type)
