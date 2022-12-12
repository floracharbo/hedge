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
    train_loader = th.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    generator = Generator()
    discriminator = Discriminator()

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
            try:
                real_samples = th.cat((real_inputs.view((batch_size_, n_consecutive_days)), real_outputs.view((batch_size_, 1))), dim=1)
            except Exception as ex:
                print(ex)
            generated_samples = th.cat((real_inputs.view((batch_size_, n_consecutive_days)), generated_outputs.view((batch_size_, 1))),
                                       dim=1)
            all_samples = th.cat((real_samples, generated_samples))
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
                    f"output_discriminator.size() {output_discriminator.size()}"
                )
                sys.exit()

            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Training the generator
            generator.zero_grad()
            generated_outputs = generator(real_inputs)
            generated_samples = th.cat((real_inputs.view((batch_size_, 3)), generated_outputs.view((batch_size_, 1))),
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
    axs[1].hlines(mean_real_output, 0, len(means_outputs))
    axs[2].plot(stds_outputs, label="std output")
    axs[2].legend()
    axs[2].hlines(std_real_output, 0, len(means_outputs))

    title = f"{data_type} n_consecutive_days {n_consecutive_days} losses mean std over time"
    if normalised:
        title += ' normalised'
    title = title.replace(' ', '_')
    fig.savefig(factors_generation_save_path / title)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 256),
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
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        output = self.model(x)
        return output


def get_factors_generator(n_consecutive_days, list_inputs0, list_outputs0, factors_generation_save_path, data_type):
    train_data_length = len(list_inputs0)
    list_inputs0 = np.array([[list_inputs0[i][-1]] + list_inputs0[i][:-1] for i in range(train_data_length)])
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
        title = "example factor series"
        if normalised:
            title += ' normalised'
        plt.title(title)
        title = title.replace(' ', '_')
        fig.savefig(title)
        mean_real_output = np.mean(list_outputs)
        std_real_output = np.std(list_outputs)
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
