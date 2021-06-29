#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        n_iter=3,
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
            nn.LayerNorm(node_hidden_dim),
            nn.Tanh(),
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.LayerNorm(node_hidden_dim),
            nn.Tanh(),
            nn.Linear(node_hidden_dim, node_hidden_dim),
            nn.LayerNorm(node_hidden_dim),
            nn.Tanh(),
        )
        input_n_param = sum(
            p.numel() for p in self.node_input_network.parameters()
        )

        # Setup the edge network
        self.edge_network = EdgeNetwork(
            node_input_dim=node_input_dim+node_hidden_dim,
            hidden_dim=edge_hidden_dim
        )
        edge_n_param = sum(
            p.numel() for p in self.edge_network.parameters()
        )

        # Setup the node layers
        self.node_network = NodeNetwork(
            node_input_dim=node_input_dim+node_hidden_dim,
            node_output_dim=node_hidden_dim,
            hidden_dim=node_hidden_dim
        )
        node_n_param = sum(
            p.numel() for p in self.node_network.parameters()
        )

        print("======Input Network======")
        print(self.node_input_network)
        print(f"Total {input_n_param} parameters.", '\n')

        print("======Edge Network======")
        print(self.edge_network)
        print(f"Total {edge_n_param} parameters.", '\n')

        print("======Node Network======")
        print(self.node_network, '\n')
        print(f"Total {node_n_param} parameters.", '\n')

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

        nodes, edges = inputs
        # Compute edges of a node.
        node_edges = self._node_edges(nodes, edges)

        # Use input layers to produce hidden features.
        node_hidden_features = self.node_input_network(nodes)

        # Shortcut connect the input node and edge onto the hidden representation.
        node_combined_features = torch.cat([
            2.0*node_hidden_features, nodes
        ], dim=-1)

        # Create recursive network.
        for i in range(self.n_iter):
            initial_hidden_features = node_hidden_features

            edge_weights = self.edge_network(
                node_combined_features,
                edges
            )

            # Apply node network
            node_hidden_features = self.node_network(
                node_combined_features,
                node_edges,
                edges,
                edge_weights
            )

            node_combined_features = (
                # Shortcut connect the input node and edge onto the hidden representation.
                torch.cat([
                    # Residual network.
                    initial_hidden_features + node_hidden_features,
                    nodes
                ], dim=-1)
            )

        # Apply edge network to get final score.
        return self.edge_network(
            node_combined_features,
            edges
        )

    @staticmethod
    def _node_edges(nodes, edges):
        """
        Compute array of edge indices of a node.

        For each batch, this method return a array of 2-tuple,
        index array of input edge and output edge.

        :param nodes: Node features.
        :param edges: Edges.
        :return: A 2-tuple array map node to index array of in and out edge.
        """
        n_batch = len(nodes)

        batches = []
        for batch_idx in range(n_batch):
            # Process a batch.
            n_node = len(nodes[batch_idx])
            n_edge = len(edges[batch_idx])

            # Get edge indices for each node.
            in_node_edge_indices = [[] for _ in range(n_node)]
            out_node_edge_indices = [[] for _ in range(n_node)]
            for edge_idx in range(n_edge):
                in_node_idx = edges[batch_idx, edge_idx, 0]
                in_node_edge_indices[in_node_idx].append(edge_idx)

                out_node_idx = edges[batch_idx, edge_idx, 1]
                out_node_edge_indices[out_node_idx].append(edge_idx)

            batches.append((
                in_node_edge_indices,
                out_node_edge_indices
            ))

        return batches
