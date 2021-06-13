#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class NodeNetwork(nn.Module):
    """
    Node network is response for aggregating node feature from neighborhood
    and compute new features.

    Network is implement with MLP.
    """
    def __init__(self, input_dim, output_dim, hidden_activation=nn.Tanh):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim*3, output_dim),
            nn.LayerNorm(output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            hidden_activation(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            hidden_activation()
        )

    def forward(self, nodes, edge_wights, in_node_adj_matrix, out_node_adj_matrix):
        # Extract features of in and out node of each edge.
        in_node_features = torch.bmm(in_node_adj_matrix.transpose(1, 2), nodes)
        out_node_features = torch.bmm(out_node_adj_matrix.transpose(1, 2), nodes)

        # Weighting by edge weights.
        wighted_in_node_adj_matrix = in_node_adj_matrix * edge_wights[:, None]
        wighted_out_node_adj_matrix = out_node_adj_matrix * edge_wights[:, None]

        wighted_in_node_features = torch.bmm(wighted_in_node_adj_matrix, in_node_features)
        wighted_out_node_features = torch.bmm(wighted_out_node_adj_matrix, out_node_features)

        # Form network input.
        network_input = torch.cat([
            nodes,
            wighted_in_node_features,
            wighted_out_node_features,
        ], dim=2)

        # Apply the network to each node.
        return self.network(network_input)
