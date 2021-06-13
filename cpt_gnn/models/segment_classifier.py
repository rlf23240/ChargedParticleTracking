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
    def __init__(self, input_dim=2, hidden_dim=8, n_iter=3, hidden_activation=nn.Tanh):
        """
        :param input_dim: Input layer size. Should be same as number of node features.
        :param hidden_dim: Hidden layer size.
        :param n_iter: Number of iteration for recursive network.
        :param hidden_activation: Activation functions for hidden layers.
        """
        super().__init__()
        self.n_iter = n_iter
        # Setup the input network
        self.input_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            hidden_activation()
        )
        # Setup the edge network
        self.edge_network = EdgeNetwork(
            input_dim+hidden_dim,
            hidden_dim,
            hidden_activation
        )
        # Setup the node layers
        self.node_network = NodeNetwork(
            input_dim+hidden_dim,
            hidden_dim,
            hidden_activation
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
        if len(inputs) > 1:
            raise ValueError(
                "Since graph is variable size input, only batch_size=1 support."
                "Please check your data."
            )

        nodes, edges = inputs[0]

        # Process nodes and edges.
        nnodes = len(nodes)
        nedges = len(edges)

        # Create adjacency matrices that map hits to edges.
        # A hit mark as 1 if it is start node of an edge in in_node_adj_matrix,
        # and similarly, a hit mark as 1 if it is end node of an edge out_node_adj_matrix.
        in_node_adj_matrix = np.zeros(shape=(1, nnodes, nedges), dtype=np.uint8)
        out_node_adj_matrix = np.zeros(shape=(1, nnodes, nedges), dtype=np.uint8)
        for edge_idx, (in_node_id, out_node_id) in enumerate(edges):
            in_node_adj_matrix[:, in_node_id, edge_idx] = 1
            out_node_adj_matrix[:, out_node_id, edge_idx] = 1

        # Convert to PyTorch Tensor form.
        nodes = torch.Tensor([nodes])
        in_node_adj_matrix = torch.Tensor(in_node_adj_matrix)
        out_node_adj_matrix = torch.Tensor(out_node_adj_matrix)

        # Use input layers to produce hidden features.
        hidden_features = self.input_network(nodes)

        # Create combined features to perform residual network.
        combined_features = torch.cat([hidden_features, nodes], dim=-1)

        # Create recursive network.
        for i in range(self.n_iter):
            # Apply edge network
            edge_wights = self.edge_network(
                combined_features,
                in_node_adj_matrix,
                out_node_adj_matrix
            )

            # Apply node network
            hidden_features = self.node_network(
                combined_features,
                edge_wights,
                in_node_adj_matrix,
                out_node_adj_matrix
            )

            # Create combined features to perform residual network.
            combined_features = torch.cat([hidden_features, nodes], dim=-1)

        # Apply edge network to get final score.
        return self.edge_network(
            combined_features,
            in_node_adj_matrix,
            out_node_adj_matrix
        )
