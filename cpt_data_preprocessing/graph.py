#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


class HitGraph:
    """
    A hit graph constructed from hits and pairs.
    """
    def __init__(self, hits: pd.DataFrame, pairs: pd.DataFrame, features: [str]):
        """
        :param hits: Dataframe contain hit data.
        :param pairs: Dataframe contain edge data.
        :param features: Features that should be save as node features.
        """
        # Nodes.
        self.nodes = hits[features].to_numpy()

        # Edges.
        node_indices = hits.set_index('hit_id').index
        self.edges = np.array([
            [node_indices.get_loc(hit1), node_indices.get_loc(hit2)]
            for hit1, hit2 in pairs[['hit_id_1', 'hit_id_2']].to_numpy()
        ], dtype=np.uint8)

        # Truth label for edges.
        self.truth = pairs['truth'].to_numpy()


class HitGraphDataset(Dataset):
    """
    PyTorch dataset for graph.

    This dataset is just enumerate though graph,
    you should use **HitGraphDataLoader** to unpack actual data.
    """
    def __init__(self, graphs: [HitGraph]):
        self.graphs = graphs

    def __getitem__(self, index) -> T_co:
        return self.graphs[index]

    def __len__(self):
        return len(self.graphs)


class HitGraphDataLoader(DataLoader):
    """
    PyTorch data loader for graph.
    """
    def __init__(self, dataset: HitGraphDataset):
        def collate(graphs):
            return (
                [(graph.nodes, graph.edges) for graph in graphs],
                torch.Tensor([graph.truth for graph in graphs])
            )

        super().__init__(
            dataset=dataset,
            collate_fn=collate,
            # Batch size is always 1.
            batch_size=1
        )
