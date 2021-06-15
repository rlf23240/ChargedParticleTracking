#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


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
        self.node_features = hits[node_features].to_numpy()

        # Edges.
        node_indices = hits.set_index('hit_id').index
        self.edges = np.array([
            [node_indices.get_loc(hit1), node_indices.get_loc(hit2)]
            for hit1, hit2 in pairs[['hit_id_1', 'hit_id_2']].to_numpy()
        ], dtype=np.uint8)

        # Truth label for edges.
        self.truth = pairs['truth'].to_numpy()

        nnodes = len(self.node_features)
        nedges = len(self.edges)

        print(f"Node count: {nnodes}. Edge count: {nedges}")

        # Create adjacency matrices that map nodes to edges.
        # A node mark as 1 if it is start node of an edge in in_node_adj_matrix,
        # and similarly, a node mark as 1 if it is end node of an edge out_node_adj_matrix.
        self.in_node_adj_matrix = np.zeros(shape=(nnodes, nedges), dtype=np.uint8)
        self.out_node_adj_matrix = np.zeros(shape=(nnodes, nedges), dtype=np.uint8)
        for edge_idx, (in_node_id, out_node_id) in enumerate(self.edges):
            self.in_node_adj_matrix[in_node_id, edge_idx] = 1
            self.out_node_adj_matrix[out_node_id, edge_idx] = 1


class HitGraphDataset(Dataset):
    """
    PyTorch dataset for graph.

    This dataset is just enumerate though graph,
    you should use **HitGraphDataLoader** to unpack actual data.
    """
    def __init__(self, graphs: [HitGraph]):
        self.graphs = graphs
        self.data = [{
            # Convert to PyTorch Tensor first since it is a slow process.
            'nodes': torch.Tensor(graph.node_features),
            'in_node_adj_matrix': torch.Tensor(graph.in_node_adj_matrix),
            'out_node_adj_matrix': torch.Tensor(graph.out_node_adj_matrix),
            'truth': torch.Tensor(graph.truth)
        } for graph in graphs]

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.graphs)


class HitGraphGeneratorDataset(Dataset):
    """
    PyTorch dataset for graph.

    This dataset read hits and pairs file on demand.
    You should use **HitGraphDataLoader** to unpack actual data.
    """
    def __init__(
        self,
        hit_files: [Path],
        pair_files: [Path],
        node_features: [str]
    ):
        self.hit_files = hit_files
        self.pair_files = pair_files

        self.node_features = node_features

        self.data = [None]*len(hit_files)

    def __getitem__(self, index) -> T_co:
        print(f"Reading {self.hit_files[index]}...")
        hits = pd.read_csv(self.hit_files[index])

        print(f"Reading {self.pair_files[index]}...")
        pairs = pd.read_csv(self.pair_files[index])

        print(f"Read complete. Loading graph...")

        graph = HitGraph(
            hits=hits,
            pairs=pairs,
            node_features=self.node_features,
        )

        data = {
            # Convert to PyTorch Tensor first since it is a slow process.
            'nodes': torch.Tensor(graph.node_features),
            'in_node_adj_matrix': torch.Tensor(graph.in_node_adj_matrix),
            'out_node_adj_matrix': torch.Tensor(graph.out_node_adj_matrix),
            'truth': torch.Tensor(graph.truth)
        }

        print(f"Graph loaded.")

        return data

    def __len__(self):
        return len(self.hit_files)


class HitGraphDataLoader(DataLoader):
    """
    PyTorch data loader for graph.
    """
    def __init__(self, dataset: [HitGraphDataset, HitGraphGeneratorDataset]):
        def collate(batch_data):
            return (
                # Input data.
                (
                    torch.stack([data['nodes'] for data in batch_data]),
                    torch.stack([data['in_node_adj_matrix'] for data in batch_data]),
                    torch.stack([data['out_node_adj_matrix'] for data in batch_data]),
                ),
                # Truth label.
                torch.stack([data['truth'] for data in batch_data])
            )

        super().__init__(
            dataset=dataset,
            collate_fn=collate,
            # Batch size is always 1.
            batch_size=1,
            shuffle=True
        )
