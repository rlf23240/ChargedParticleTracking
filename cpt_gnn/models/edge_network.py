#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import torch
import torch.nn as nn


class EdgeNetwork(nn.Module):
    """
    Edge network is response for calculate edge weight
    that will be used in node network.

    Network is implement with MLP that output [0,1] value.
    """
    def __init__(self, input_dim, hidden_dim=8, hidden_activation=nn.Tanh):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            hidden_activation(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, nodes, in_node_adj_matrix, out_node_adj_matrix):
        # Extract features of in and out node of each edge.
        in_node_features = torch.bmm(in_node_adj_matrix.transpose(1, 2), nodes)
        out_node_features = torch.bmm(out_node_adj_matrix.transpose(1, 2), nodes)

        # Form network input.
        network_input = torch.cat([
            in_node_features,
            out_node_features
        ], dim=2)

        # Apply the network to each edge.
        return self.network(network_input).squeeze(-1)
