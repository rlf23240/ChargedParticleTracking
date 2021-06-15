#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
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
from cpt_data_preprocessing import HitGraphDataset, HitGraphGeneratorDataset, HitGraphDataLoader
from cpt_plots import hit_pair_gnn_prediction_plot_2d
from cpt_gnn import SegmentClassifier, Trainer


def split(array: list, ratio: float):
    """
    Split array into two parts base on ratio.
    Mostly use on train test data split.

    :param array: Array to split.
    :param ratio: Ratio of length between 2 output array.
    :return: Two split array.
    """
    random.shuffle(array)

    if ratio > 1.0:
        return array, []
    if ratio < 0.0:
        return [], array

    count = int(len(array)*ratio)

    return array[:count], array[count:]


def train_model(model: nn.Module, save: Path = None) -> Trainer:
    """
    Train a GNN model.

    :param: save: Path to save model. None if no need to save.
    :return: Trainer contains a trained model.
    """
    # Prepare data.
    dataset = Path('output/pairs/100evts_volume8')
    volume = 8

    # events = range(1000, 1100)
    events = [
        1001, 1020, 1033, 1039, 1044, 1035, 1057, 1062, 1067, 1082,
    ]

    train_events, test_events = split(events, 0.67)

    # Form dataset.
    train_data_loader = HitGraphDataLoader(
        dataset=HitGraphGeneratorDataset(
            hit_files=[dataset/f'event{event:09}-hits.csv' for event in train_events],
            pair_files=[dataset/f'event{event:09}-pairs.csv' for event in train_events],
            node_features=['x', 'y', 'z']
        )
    )
    test_data_loader = HitGraphDataLoader(
        dataset=HitGraphGeneratorDataset(
            hit_files=[dataset/f'event{event:09}-hits.csv' for event in test_events],
            pair_files=[dataset/f'event{event:09}-pairs.csv' for event in test_events],
            node_features=['x', 'y', 'z']
        )
    )

    # Train model.
    trainer = Trainer(
        model=model,
        loss=functional.binary_cross_entropy,
        optimizer=optimizers.Adam(
            model.parameters(),
            lr=0.005
        ),
        save=save
    )
    trainer.train(
        train_data=train_data_loader,
        valid_data=test_data_loader,
        epochs=5
    )

    return trainer


def validate(model: nn.Module, save: Path = None):
    """
    Validate model by use a single event.

    :param model: Model to be validate.
    :param save: Path to save plots. None if no need to save.
    :return:
    """
    dataset = Path('output/pairs/100evts_volume8/')
    volume = 8
    event = 1001

    hits = pd.read_csv(dataset / f'event{event:09}-hits.csv')
    pairs = pd.read_csv(dataset / f'event{event:09}-pairs.csv')

    graph = HitGraph(
        hits=hits,
        pairs=pairs,
        node_features=['x', 'y', 'z'],
    )

    data_loader = HitGraphDataLoader(
        dataset=HitGraphDataset(
            graphs=[graph]
        )
    )

    # Mark model for evaluations.
    model.eval()

    # Use for loop to unpack data. It actually has only one graph.
    for batch_idx, (batch_input, batch_target) in enumerate(data_loader):
        # Evaluate model and remove batch dimension
        predictions = model(batch_input).squeeze().detach().numpy()

        save.mkdir(parents=True, exist_ok=True)

        hit_pair_gnn_prediction_plot_2d(
            hits=hits,
            pairs=pairs,
            predictions=predictions,
            truth=graph.truth,
            threshold=0.4,
            line_width=0.5,
            color_scheme=[
                [0, 1, 0, 1.0],
                [1, 0, 0, 0.1],
                [0, 0, 0, 0.0],
                [0, 0, 0, 0.0]
            ],
            save=save / 'true_false_positive.png'
        )

        hit_pair_gnn_prediction_plot_2d(
            hits=hits,
            pairs=pairs,
            predictions=predictions,
            truth=graph.truth,
            threshold=0.4,
            line_width=0.1,
            color_scheme=[
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0.1],
                [1, 1, 0, 0.1]
            ],
            save=save / 'true_false_negative.png'
        )

        hit_pair_gnn_prediction_plot_2d(
            hits=hits,
            pairs=pairs,
            predictions=predictions,
            truth=graph.truth,
            threshold=0.4,
            save=save / 'all.png'
        )


if __name__ == '__main__':
    # Create model.
    model = SegmentClassifier(
        node_input_dim=3,
        node_hidden_dim=8,
        edge_hidden_dim=8,
        n_iter=3
    )

    # Train new model.
    trainer = train_model(
        model=model,
        save=Path('output/models/iter3_10evts_5epochs')
    )

    # Load previous saved model.
    """model.load_state_dict(
        torch.load(
            'output/models/iter3_10evts_5epochs/model'
        )
    )"""

    validate(
        model=model,
        save=Path('output/plots/gnn/iter3_10evts_5epochs')
    )
