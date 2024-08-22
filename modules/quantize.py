import torch
import numpy as np
from distributions.gumbel import gumbel_softmax_sample
from torch import nn
from typing import NamedTuple
from typing import Tuple


class QuantizeOutput(NamedTuple):
    embeddings: torch.Tensor
    ids: torch.Tensor

class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        restart_threshold: int = 5,
        restart_probability: float = 0.5
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self._init_weights()

        # Initialize the usage counter and restart parameters
        self.usage_counter = torch.zeros(n_embed, dtype=torch.int)
        self.restart_threshold = restart_threshold
        self.restart_probability = restart_probability

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.2)

    def get_item_embeddings(self, item_ids) -> torch.Tensor:
        return self.embedding(item_ids)

    def update_usage_counter(self, ids: torch.Tensor) -> None:
        # Update the usage counter based on the ids tensor
        unique_ids = torch.unique(ids)
        for uid in unique_ids:
            self.usage_counter[uid] += 1

    def random_restart_unused_codes(self) -> None:
        # Identify unused codes
        unused_codes = (self.usage_counter < self.restart_threshold).nonzero(as_tuple=True)[0]

        # Randomly restart a subset of unused codes based on the restart probability
        num_codes_to_restart = int(len(unused_codes) * self.restart_probability)
        if num_codes_to_restart > 0:
            restart_indices = np.random.choice(unused_codes.cpu().numpy(), size=num_codes_to_restart, replace=False)
            new_values = torch.randn((num_codes_to_restart, self.embed_dim), device=self.device)
            self.embedding.weight.data[restart_indices] = new_values

            # Reset the usage counter for restarted codes
            self.usage_counter[restart_indices] = 0

    def forward(self, x, temperature, inference_only=False) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x.shape[-1] == self.embed_dim

        codebook = self.embedding.weight
        dist = (
            (x**2).sum(axis=1, keepdim=True) +
            (codebook.T**2).sum(axis=0, keepdim=True) -
            2 * x @ codebook.T
        )

        _, ids = (-dist).max(axis=1)

        if inference_only:
            emb = self.get_item_embeddings(ids)
        else:
            weights = gumbel_softmax_sample(
                -dist, temperature=temperature, device=self.device
            )
            emb = weights @ codebook

            # Update the usage counter and restart unused codes
            self.update_usage_counter(ids)
            self.random_restart_unused_codes()

        return QuantizeOutput(
            embeddings=emb,
            ids=ids
        )