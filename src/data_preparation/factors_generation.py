import torch as th
from torch import nn
import pickle

import math
import matplotlib.pyplot as plt

th.manual_seed(111)


def training(train_data, normalised):
    batch_size = 32
    train_loader = th.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    generator = Generator()
    discriminator = Discriminator()

    lr = 0.001
    num_epochs = 300
    loss_function = nn.BCELoss()

    optimizer_discriminator = th.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = th.optim.Adam(generator.parameters(), lr=lr)

    losses_generator = []
    losses_discriminator = []
    for epoch in range(num_epochs):
        for n, train_data in enumerate(train_loader):
            real_inputs = train_data[:, 0:3]
            real_outputs = train_data[:, 3]
            # Data for training the discriminator
            real_samples_labels = th.ones((batch_size, 1))
            # latent_space_samples = th.randn((batch_size, 2))

            generated_outputs = generator(real_inputs)
            generated_samples_labels = th.zeros((batch_size, 1))
            real_samples = th.cat((real_inputs.view((batch_size, 3)), real_outputs.view((batch_size, 1))), dim=1)
            generated_samples = th.cat((real_inputs.view((batch_size, 3)), generated_outputs.view((batch_size, 1))),
                                       dim=1)
            all_samples = th.cat((real_samples, generated_samples))
            all_samples_labels = th.cat(
                (real_samples_labels, generated_samples_labels)
            )
            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels
            )
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Training the generator
            generator.zero_grad()
            generated_outputs = generator(real_inputs)
            generated_samples = th.cat((real_inputs.view((batch_size, 3)), generated_outputs.view((batch_size, 1))),
                                       dim=1)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()
            if n % 10 == 0:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")

            losses_generator.append(loss_generator)
            losses_discriminator.append(loss_discriminator)

    fig = plt.figure()
    plt.plot(losses_generator, label="losses_generator")
    plt.plot(losses_discriminator, label="losses_discriminator")
    plt.legend()
    title = "losses over time"
    if normalised:
        title += ' normalised'
    fig.savefig(title)


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

with open(f'data/other_outputs/inputs_loads.pickle', 'rb') as f:
    list_inputs = pickle.load(f)
with open(f'data/other_outputs/outputs_loads.pickle', 'rb') as f:
    list_outputs = pickle.load(f)

train_data_length = len(list_inputs)
list_inputs = [[list_inputs[i][2]] + list_inputs[i][:2] for i in range(train_data_length)]

for normalised in [False, True]:
    if normalised:
        for i in train_data_length:
            list_inputs[i][1:3] = list_inputs[i][1:3]/list_inputs[i][1]
    fig = plt.figure()
    for i in range(20):
        plt.plot(list_inputs[i][1:3] + list_outputs[i])
    title = "example factor series"
    if normalised:
        title += ' normalised'
    plt.title(title)
    fig.savefig(title)
    train_data = th.zeros((train_data_length, 4))
    train_data[:, 0:3] = th.tensor(list_inputs)
    train_data[:, 3] = th.tensor(list_outputs).view_as(train_data[:, 3])

    training(train_data, normalised)

