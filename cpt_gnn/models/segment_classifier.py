#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

from .edge_network import EdgeNetwork
from .node_network import NodeNetwork


class SegmentClassifier(nn.Module):
    """
    Segment classification graph neural network model.
    Consists of an input network, an edge network, and a node network.
    """
    def __init__(
        self,
        node_input_dim,
        node_hidden_dim=8,
        edge_hidden_dim=8,
        n_iter=3
    ):
        """
        :param node_input_dim: Input node feature size.
        :param node_hidden_dim: Node feature size to embed.
        :param edge_hidden_dim: Edge feature size to embed.
        :param n_iter: Number of iteration for recursive network.
        """
        super().__init__()
        self.n_iter = n_iter
        # Setup the input network
        self.node_input_network = nn.Sequential(
            nn.Linear(node_input_dim, node_hidden_dim),
            nn.Tanh()
        )
        # Setup the edge network
        self.edge_network = EdgeNetwork(
            node_input_dim=node_input_dim+node_hidden_dim,
            hidden_dim=edge_hidden_dim
        )
        # Setup the node layers
        self.node_network = NodeNetwork(
            node_input_dim=node_input_dim+node_hidden_dim,
            node_output_dim=node_hidden_dim,
            hidden_dim=node_hidden_dim
        )

        print("======Edge Network======")
        print(self.edge_network, '\n')

        print("======Node Network======")
        print(self.node_network, '\n')

    def forward(self, inputs):
        """
        Apply forward pass of the model.
        """
        # Since graph is variable size input,
        # batching is not support.
        if len(inputs[0]) > 1:
            raise ValueError(
                "Since graph is variable size input, only batch_size=1 support."
                "Please check your data."
            )

        nodes, in_node_adj_matrix, out_node_adj_matrix = inputs

        # Use input layers to produce hidden features.
        node_hidden_features = self.node_input_network(nodes)

        # Shortcut connect the input node and edge onto the hidden representation.
        node_combined_features = torch.cat([
            node_hidden_features, nodes
        ], dim=-1)

        # Create recursive network.
        for i in range(self.n_iter):
            # Apply edge network
            edge_weights = self.edge_network(
                node_combined_features,
                in_node_adj_matrix,
                out_node_adj_matrix
            )

            # Apply node network
            hidden_features = self.node_network(
                node_combined_features,
                edge_weights,
                in_node_adj_matrix,
                out_node_adj_matrix
            )

            node_combined_features = (
                # Residual network.
                node_combined_features +
                # Shortcut connect the input node onto the hidden representation.
                torch.cat([hidden_features, nodes], dim=-1)
            )

        # Apply edge network to get final score.
        return self.edge_network(
            node_combined_features,
            in_node_adj_matrix,
            out_node_adj_matrix
        )
