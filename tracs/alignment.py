"""Conditional Normalizing Flows for aligning TraCS scores with Backbone predictor."""

#
# Copyright (c) Simon Kohaut, Felix Divo, Julius Hahnewald, and contributors
#
# This file is part of TraCS and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with TraCS.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

# Standard Library
from pickle import dump, load

# Third Party
import torch
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import ReversePermutation
from numpy.typing import NDArray
from tqdm import tqdm

# ProMis
from promis.geo import Collection


class Alignment:
    def __init__(
        self,
        number_of_states: int,
        number_of_hidden_features: int,
        number_of_layers: int,
        number_of_context_features: int,
    ):
        # Conditional Flow parameters
        self.number_of_states = number_of_states
        self.number_of_hidden_features = number_of_hidden_features
        self.number_of_layers = number_of_layers
        self.number_of_context_features = number_of_context_features

        # Need to ensure even number of features for nflows library
        if self.number_of_context_features > 2 and self.number_of_context_features % 2 == 1:
            self.number_of_context_features += 1

        # Flow setup
        transforms = []
        for _ in range(self.number_of_layers):
            transforms.append(ReversePermutation(features=self.number_of_states))
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=self.number_of_states,
                    hidden_features=self.number_of_hidden_features,
                    context_features=self.number_of_context_features,
                )
            )

        self.flow = Flow(CompositeTransform(transforms), StandardNormal(shape=[self.number_of_states]))

    @staticmethod
    def load(path) -> "Collection":
        with open(path, "rb") as file:
            return load(file)

    def save(self, path: str):
        with open(path, "wb") as file:
            dump(self, file)

    def sample(self, size: int, context: torch.Tensor):
        return self.flow.sample(size, context=context)

    def log_prob(self, states: NDArray, context: torch.Tensor):
        return self.flow.log_prob(states, context=context)

    def prob(self, states: NDArray, context: torch.Tensor):
        return self.log_prob(states, context).exp()

    def fit(
        self, samples: NDArray, context: torch.Tensor, number_of_epochs: int, batch_size: int
    ) -> NDArray:
        # Setup optimizer
        optimizer = torch.optim.Adam(self.flow.parameters())
        dataloader = torch.utils.data.DataLoader(
            torch.hstack([samples, context]),
            batch_size=batch_size,
            shuffle=True,
        )

        # Train on given samples and doubt features
        losses = []
        for _ in tqdm(range(number_of_epochs), desc="Learning Alignment", unit="epoch"):
            for batch in dataloader:
                optimizer.zero_grad()
                loss = -self.flow.log_prob(
                    batch[:, : self.number_of_states], context=batch[:, self.number_of_states :]
                ).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach().numpy())

        return losses