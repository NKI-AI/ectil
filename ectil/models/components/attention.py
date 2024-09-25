from typing import Optional

import torch
from torch import nn
from torch.masked import masked_tensor


class Attention(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int = 1,
        non_linearity: nn.Module = nn.Tanh,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            non_linearity(),
            nn.Linear(hidden_features, out_features),
        )

    def _compute_attention_weights(self, attention_logits):
        attention_weights = nn.functional.softmax(attention_logits, dim=1)
        return attention_weights

    def forward(self, x: torch.Tensor):
        """Forward of attention module

        Args:
            x (torch.Tensor): features of dimension batch_size, bag_size, hidden_dimension
            **kwargs: Optional arguments for attention module. E.g. region for DistanceWeightedMean. Not used in this module, but passed to a sequential
                that holds a feature preprocessing component and the attention module. Since the feature preprocessing may need the regions, the regions are
                passed through all components and must be handled accordingly. In this case, we capture them through **kwargs and don't do anything with them,

        Returns:
            torch.Tensor: attention weights of dimension batch_size, out_features (1)
        """
        # x: batch_size * bag_size * instance_size
        attention_logits = self.attention(x) / self.temperature
        # attention_logits: batch_size * bag_size * out_features (often 1)
        attention_weights = self._compute_attention_weights(x, attention_logits)
        # take softmax over the bag_size dimension, to ensure that e.g. attention_weights[i].sum() == 1
        # attention_weights: batch_size * bag_size * out_features (often 1)
        return attention_weights


class GatedAttention(Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gate = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features), nn.Sigmoid()
        )

        self.attention_hidden = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_features), nn.Tanh()
        )

        self.attention = nn.Linear(self.hidden_features, self.out_features)

    def forward(self, x, **kwargs):
        """See Attention.forward() for docstring"""
        # x: batch_size * bag_size * instance_size
        attention_hidden_logits = self.attention_hidden(x)
        # attention_hidden_logits: batch_size * bag_size * hidden_features (often 1)

        attention_hidden_gate = self.gate(x)
        # attention_hidden_gate: batch_size * bag_size * hidden_features (often 1)

        attention_hidden_out = attention_hidden_logits * attention_hidden_gate
        # ^-- element-wise multiplication
        # attention_hidden_out: batch_size * bag_size * hidden_features (often 1)

        attention_logits = self.attention(attention_hidden_out)
        # attention_logits: batch_size * bag_size * hidden_features (often 1)

        attention_weights = self._compute_attention_weights(attention_logits)
        # take softmax over the bag_size dimension, to ensure that e.g. attention_weights[i].sum() == 1
        # attention_weights: batch_size * bag_size * out_features (often 1)
        return attention_weights
