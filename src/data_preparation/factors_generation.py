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

def training(
        train_data, normalised, factors_generation_save_path,
        data_type, n_consecutive_days, mean_real_output, std_real_output
):
    batch_size = 32
    train_data = [train_data_i for train_data_i in train_data if not any(th.isnan(train_data_i))]
    train_loader = th.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    generator = Generator(n_consecutive_days)
    discriminator = Discriminator(n_consecutive_days)

    lr = 0.001
    num_epochs = 20
    loss_function = nn.BCELoss()

    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=lr)

    losses_generator = []
    losses_discriminator = []
    means_outputs = []
    stds_outputs = []
    for epoch in tqdm(range(num_epochs)):
        for n, train_data in enumerate(train_loader):
            batch_size_ = len(train_data)
            real_inputs = train_data[:, :-1]
            real_outputs = train_data[:, n_consecutive_days]
            # Data for training the discriminator
            real_samples_labels = th.ones((batch_size_, 1))
            # latent_space_samples = th.randn((batch_size, 2))

            generated_outputs = generator(real_inputs)
            generated_samples_labels = th.zeros((batch_size_, 1))
            real_samples = th.cat((real_inputs.view((batch_size_, n_consecutive_days)), real_outputs.view((batch_size_, 1))), dim=1)
            if any(any(th.isnan(real_inputs_i)) for real_inputs_i in real_inputs):
                print('nan in real_inputs')
                sys.exit()
            if any(th.isnan(real_outputs)):
                print('nan in real_outputs')
                sys.exit()
            if any(th.isnan(generated_outputs)):
                print('nan in generated_outputs')
                sys.exit()
            generated_samples = th.cat((real_inputs.view((batch_size_, n_consecutive_days)), generated_outputs.view((batch_size_, 1))),
                                       dim=1)
            all_samples = th.cat((real_samples, generated_samples))
            if any(any(th.isnan(real_samples_i)) for real_samples_i in real_samples):
                print('nan in real_samples')
                sys.exit()
            all_samples_labels = th.cat(
                (real_samples_labels, generated_samples_labels)
            )
            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
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
            generated_outputs = generator(real_inputs)
            generated_samples = th.cat((real_inputs.view((batch_size_, n_consecutive_days)), generated_outputs.view((batch_size_, 1))),
                                       dim=1)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            losses_generator.append(loss_generator.detach().numpy())
            losses_discriminator.append(loss_discriminator.detach().numpy())
            means_outputs.append(np.mean(generated_outputs.detach().numpy()))
            stds_outputs.append(np.std(generated_outputs.detach().numpy()))

        fig = plt.figure()
        for i in range(batch_size_):
            plt.plot(generated_samples[i].detach().numpy()[1:])
        title = f'{data_type} n_consecutive_days {n_consecutive_days} generated samples epoch {epoch}'
        if normalised:
            title += ' normalised'
        plt.title(title)
        title = title.replace(' ', '_')
        fig.savefig(factors_generation_save_path / title)

    fig, axs = plt.subplots(3)
    axs[0].plot(losses_generator, label="losses_generator")
    axs[0].plot(losses_discriminator, label="losses_discriminator")
    axs[0].legend()
    axs[1].plot(means_outputs, label="mean output")
    axs[1].legend()
    axs[1].hlines(mean_real_output, 0, len(means_outputs), color='red', linestyle='dashed')
    axs[2].plot(stds_outputs, label="std output")
    axs[2].legend()
    axs[2].hlines(std_real_output, 0, len(means_outputs), color='red', linestyle='dashed')
    title = f"{data_type} n_consecutive_days {n_consecutive_days} losses mean std over time"
    if normalised:
        title += ' normalised'
    title = title.replace(' ', '_')
    fig.savefig(factors_generation_save_path / title)
    print(f"Saved {title}, mean_real_output {mean_real_output}, std_real_output {std_real_output}")
    print(f"mean generated outputs last 10: {np.mean(means_outputs[-10:])}, std {np.mean(stds_outputs[-10:])}")

    fig = plt.figure()
    plt.hist(generated_outputs.detach().numpy(), bins=100, alpha=0.5, label='generated')
    plt.hist(real_outputs, bins=100, alpha=0.5, label='real')
    plt.legend()
    title = f"{data_type} n_consecutive_days {n_consecutive_days} hist generated vs real"
    if normalised:
        title += ' normalised'
    title = title.replace(' ', '_')
    fig.savefig(factors_generation_save_path / title)

    th.save(generator.model, factors_generation_save_path / f"generator_{data_type}_n_consecutive_days_{n_consecutive_days}.pt")

class Discriminator(nn.Module):
    def __init__(self, n_consecutive_days):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_consecutive_days + 1, 256),
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
    def __init__(self, n_consecutive_days):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_consecutive_days, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output = self.model(x)
        return output


def get_factors_generator(n_consecutive_days, list_inputs0, list_outputs0, factors_generation_save_path, data_type):
    list_inputs0 = np.array([[list_inputs0[i][-1]] + list_inputs0[i][:-1] for i in range(len(list_inputs0))])
    i_keep = [
        i for i in range(len(list_inputs0))
        if not any(f == 0 for f in list_inputs0[i][1:]) and list_outputs0[i] != 0
    ]
    list_inputs0 = list_inputs0[i_keep]
    list_outputs0 = np.array(list_outputs0)[i_keep]
    train_data_length = len(list_inputs0)

    if not factors_generation_save_path.exists():
        os.makedirs(factors_generation_save_path)

    for normalised in [True, False]:
        list_inputs = copy.deepcopy(list_inputs0)
        list_outputs = copy.deepcopy(list_outputs0)
        if normalised:
            for i in range(train_data_length):
                list_inputs[i][1:] = list_inputs0[i][1:] / list_inputs0[i][1]
                list_outputs[i] = list_outputs0[i] / list_inputs0[i][1]
        fig = plt.figure()
        for i in range(20):
            plt.plot(np.append(list_inputs[i][1:], list_outputs[i]))
        title = f"{data_type} n_consecutive_days {n_consecutive_days} example factor series"
        if normalised:
            title += ' normalised'
        plt.title(title)
        title = title.replace(' ', '_')
        fig.savefig(factors_generation_save_path / title)
        mean_real_output = np.nanmean(list_outputs)
        std_real_output = np.nanstd(list_outputs)
        train_data = th.zeros((train_data_length, n_consecutive_days + 1))
        train_data[:, :-1] = th.tensor(list_inputs)
        train_data[:, -1] = th.tensor(np.array(list_outputs)).view_as(train_data[:, -1])

        training(
            train_data, normalised, factors_generation_save_path,
            data_type, n_consecutive_days, mean_real_output, std_real_output
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
