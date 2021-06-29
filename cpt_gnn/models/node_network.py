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
    def __init__(self, node_input_dim, node_output_dim, hidden_dim):
        super().__init__()

        activation = nn.Tanh
        self.network = nn.Sequential(
            nn.Linear(node_input_dim*3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation(),
            nn.Linear(hidden_dim, node_output_dim),
            nn.LayerNorm(node_output_dim),
            activation()
        )

    def forward(self, nodes, node_edges, edges, edge_weights):
        # Extract features of in and out node of each edge and weighting by edge weights.
        edge_weighted_in_node_features, edge_weighted_out_node_features = self._weight_edges(
            nodes=nodes,
            edges=edges,
            weights=edge_weights
        )

        # Aggregate neighborhood features to node.
        aggregated_in_node_features, aggregated_out_node_features = self._aggregate_node_features(
            nodes=nodes,
            node_edges=node_edges,
            in_node_features=edge_weighted_in_node_features,
            out_node_features=edge_weighted_out_node_features
        )

        # Form network input.
        network_input = torch.cat([
            aggregated_in_node_features,
            aggregated_out_node_features,
            nodes
        ], dim=2)

        # Apply the network to each node.
        return self.network(network_input)

    @staticmethod
    def _weight_edges(nodes, edges, weights):
        n_batch = len(nodes)

        weighted_in_node_features = torch.stack([
            nodes[batch_idx, edges[batch_idx, :, 0], :]
            for batch_idx in range(n_batch)
        ]).transpose(1, 2) * weights[:, None]

        weighted_out_node_features = torch.stack([
            nodes[batch_idx, edges[batch_idx, :, 1], :]
            for batch_idx in range(n_batch)
        ]).transpose(1, 2) * weights[:, None]

        return weighted_in_node_features, weighted_out_node_features

    @staticmethod
    def _aggregate_node_features(nodes, node_edges, in_node_features, out_node_features):
        n_batch = len(nodes)

        # Aggregate neighborhood features to node.
        aggregated_in_node_features = []
        aggregated_out_node_features = []
        for batch_idx in range(n_batch):
            # Process a batch.
            n_node = len(nodes[batch_idx])

            # Aggregated node features.
            batch_aggregated_in_node_features = []
            batch_aggregated_out_node_features = []

            in_edge_indices, out_edge_indices = node_edges[batch_idx]
            for node_idx in range(n_node):
                # Handle in node features.
                batch_aggregated_in_node_features.append(
                    torch.sum(in_node_features[
                        batch_idx, :, in_edge_indices[node_idx]
                    ], dim=1)
                )

                # Handle out node features.
                batch_aggregated_out_node_features.append(
                    torch.sum(out_node_features[
                        batch_idx, :, out_edge_indices[node_idx]
                    ], dim=1)
                )

            # Stack all nodes.
            aggregated_in_node_features.append(
                torch.stack(batch_aggregated_in_node_features)
            )
            aggregated_out_node_features.append(
                torch.stack(batch_aggregated_out_node_features)
            )

        # Stack batch dim.
        aggregated_in_node_features = torch.stack(
            aggregated_in_node_features
        )
        aggregated_out_node_features = torch.stack(
            aggregated_out_node_features
        )

        return aggregated_in_node_features, aggregated_out_node_features
