#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optimizers
import torch.nn as nn
import torch.nn.functional as functional

import cpt_data_reader
from cpt_data_preprocessing import pair, filters
from cpt_data_preprocessing import HitGraph
from cpt_data_preprocessing import HitGraphDataset, HitGraphDataLoader
from cpt_plots import hit_pair_gnn_prediction_plot_2d
from cpt_gnn import SegmentClassifier, Trainer


def train_model(save: Path = None) -> nn.Module:
    """
    Train a GNN model.

    :param: save: Path to save model. None if no need to save.
    :return: Trained model.
    """
    # Prepare data.
    dataset = '../data/train_10evts/train_10evts/'
    volume = 8
    events = [
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009
    ]

    pd.set_option('display.max_columns', None)

    graphs = []
    for event in events:
        hits, particles, truth_labels = cpt_data_reader.read(
            dataset=dataset,
            event=event,
            volume=volume
        )

        # Pair hits to form edges.
        hits, pair_dfs = pair(
            hits=hits,
            start_layers=[2, 4, 6],
            end_layers=[4, 6, 8],
            node_filters=[
                filters.DBSCANFilter(
                    eps=0.05,
                    min_pts=20
                ),
            ],
            edge_filters=[
                filters.TransverseMomentumFilter(pt_min=0.5),
                filters.ClusterEdgeFilter(group='group_DBSCAN'),
                filters.RealEdgeLabeler(real_tracks=truth_labels)
            ]
        )

        # Concat all hit pairs to ignore layers.
        pairs = pd.concat(pair_dfs.values())

        graphs.append(HitGraph(hits, pairs, ['eta', 'phi']))

    # Form dataset.
    data_loader = HitGraphDataLoader(
        dataset=HitGraphDataset(graphs)
    )

    # Create model.
    model = SegmentClassifier()

    # Train model.
    trainer = Trainer(
        model=model,
        loss=functional.binary_cross_entropy,
        optimizer=optimizers.Adam(
            model.parameters()
        ),
        save=save
    )
    trainer.train(
        train_data=data_loader,
        epochs=1
    )
    torch.save(model, 'output/models/10evts_1epochs')

    return model


def validate(model: nn.Module, save: Path = None):
    """
    Validate model by use a single event.

    :param model: A model to be validate.
    :param save: Path to save plots. None if no need to save.
    :return:
    """
    dataset = '../data/train_10evts/train_10evts/'
    volume = 8

    hits, particles, truth_labels = cpt_data_reader.read(
        dataset=dataset,
        event=1000,
        volume=volume
    )

    # Pair hits to form edges.
    hits, pair_dfs = pair(
        hits=hits,
        start_layers=[2, 4, 6],
        end_layers=[4, 6, 8],
        node_filters=[
            filters.DBSCANFilter(
                eps=0.05,
                min_pts=20
            ),
        ],
        edge_filters=[
            filters.TransverseMomentumFilter(pt_min=0.5),
            filters.ClusterEdgeFilter(group='group_DBSCAN'),
            filters.RealEdgeLabeler(real_tracks=truth_labels)
        ]
    )

    # Concat all hit pairs to ignore layers.
    pairs = pd.concat(pair_dfs.values())

    hit_graph = HitGraph(hits, pairs, ['eta', 'phi'])

    data_loader = HitGraphDataLoader(
        HitGraphDataset([hit_graph])
    )

    # Mark model for evaluations.
    model.eval()

    # Use for loop to unpack data. It actually has only one graph.
    for batch_idx, (batch_input, batch_target) in enumerate(data_loader):
        # Evaluate model and remove batch dimension
        predictions = model(batch_input).squeeze().detach().numpy()

        hit_pair_gnn_prediction_plot_2d(
            hits=hits,
            pairs=pairs,
            predictions=predictions,
            truth=hit_graph.truth,
            save=save
        )


if __name__ == '__main__':
    model = train_model(save=Path('output/models/10evts_1epochs'))
    validate(model, save=Path('output/plots/gnn/10evts_1epochs.png'))
