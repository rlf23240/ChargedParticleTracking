#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co

from cpt_data_preprocessing import HitGraph


def _extract_data(graph) -> dict:
    # Convert to PyTorch tensor.
    nodes = torch.from_numpy(graph.node_features)
    truth = torch.from_numpy(graph.truth)
    edges = torch.from_numpy(graph.edges)

    return {
        # Convert to PyTorch Tensor first since it is a slow process.
        'nodes': nodes,
        'edges': edges,
        'truth': truth
    }


def _collate(batch_data):
    return (
        # Input data.
        (
            torch.stack([data['nodes'] for data in batch_data]),
            torch.stack([data['edges'] for data in batch_data]),
        ),
        # Truth label.
        torch.stack([data['truth'] for data in batch_data])
    )


class HitGraphDataset(Dataset):
    """
    PyTorch dataset for graph.

    This dataset is just enumerate though graph,
    you should use **HitGraphDataLoader** to unpack actual data.
    """
    def __init__(self, graphs: [HitGraph]):
        self.graphs = graphs
        self.data = [_extract_data(graph) for graph in graphs]

    def __getitem__(self, index) -> T_co:
        return self.data[index]

    def __len__(self):
        return len(self.graphs)


class HitGraphGeneratorDataset(HitGraphDataset):
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
        super().__init__(graphs=[])

        self.hit_files = hit_files
        self.pair_files = pair_files

        self.node_features = node_features

        self.data = [None] * len(hit_files)

    def __getitem__(self, index) -> T_co:
        print(f"Reading {self.hit_files[index]}...")
        hits = pd.read_csv(self.hit_files[index])

        print(f"Reading {self.pair_files[index]}...")
        pairs = pd.read_csv(self.pair_files[index])

        print(f"Read complete. Loading graph...")

        data = _extract_data(HitGraph(
            hits=hits,
            pairs=pairs,
            node_features=self.node_features,
        ))

        print(f"Graph loaded.")

        return data

    def __len__(self):
        return len(self.hit_files)


class HitGraphDataLoader(DataLoader):
    """
    PyTorch data loader for graph.
    """
    def __init__(self, dataset: [HitGraphDataset, HitGraphGeneratorDataset]):
        super().__init__(
            dataset=dataset,
            collate_fn=_collate,
            # Batch size is always 1.
            batch_size=1,
            shuffle=True
        )
