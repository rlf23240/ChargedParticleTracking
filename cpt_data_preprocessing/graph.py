#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


class HitGraph:
    """
    A hit graph constructed from hits and pairs.
    """
    def __init__(
        self,
        hits: pd.DataFrame,
        pairs: pd.DataFrame,
        node_features: [str]
    ):
        """
        :param hits: Dataframe contain hit data.
        :param pairs: Dataframe contain edge data.
        :param node_features: Column that should be save as node features.
        """
        # Nodes.
        self.node_features = hits[node_features].to_numpy(dtype=np.float32)

        # Edges.
        node_indices = hits.set_index('hit_id').index
        self.edges = np.array([
            [node_indices.get_loc(hit1), node_indices.get_loc(hit2)]
            for hit1, hit2 in pairs[['hit_id_1', 'hit_id_2']].to_numpy()
        ], dtype=np.int64)

        # Truth label for edges.
        self.truth = pairs['truth'].to_numpy(dtype=np.float32)

        self.n_node = len(self.node_features)
        self.n_edge = len(self.edges)

        print(f"[HitGraph] Node count: {self.n_node}. Edge count: {self.n_edge}")

        # Create adjacency matrices that map nodes to edges.
        # A node mark as 1 if it is start node of an edge in in_node_adj_matrix,
        # and similarly, a node mark as 1 if it is end node of an edge out_node_adj_matrix.
        """self.in_node_adj_matrix = np.zeros(shape=(self.nnodes, self.nedges), dtype=np.float32)
        self.out_node_adj_matrix = np.zeros(shape=(self.nnodes, self.nedges), dtype=np.float32)
        for edge_idx, (in_node_id, out_node_id) in enumerate(self.edges):
            self.in_node_adj_matrix[in_node_id, edge_idx] = 1
            self.out_node_adj_matrix[out_node_id, edge_idx] = 1"""
