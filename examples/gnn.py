#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optimizers
import torch.nn as nn
import torch.nn.functional as functional

from cpt_data_preprocessing import HitGraph
from cpt_data_reader import HitGraphDataset, HitGraphGeneratorDataset, HitGraphDataLoader
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


def train_model(
    model: nn.Module,
    epoch: int,
    device: str = 'cpu',
    true_sample_train_weight: float = 1.0,
    false_sample_train_weight: float = 1.0,
    save: Path = None
) -> Trainer:
    """
    Train a GNN model.

    :param: model: Model to train.
    :param: epoch: Number of epochs.
    :param: device: Device use to train.
    :param: true_sample_train_weight: Train weight for true sample.
    :param: false_sample_train_weight: Train weight for false sample.
    :param: save: Path to save model. None if no need to save.
    :return: Trainer contains a trained model.
    """
    # Prepare data.
    dataset = Path('output/pairs/100evts_volume8')

    events = [
        1001, 1020, 1033, 1039, 1044, 1035, 1057, 1062, 1067, 1082,
        1061, 1091, 1063, 1077, 1047, 1078, 1076, 1018, 1056, 1048
    ]

    train_events, test_events = split(events, 0.67)

    # Form dataset.
    train_data_loader = HitGraphDataLoader(
        dataset=HitGraphGeneratorDataset(
            hit_files=[dataset/f'event{event:09}-hits.csv' for event in train_events],
            pair_files=[dataset/f'event{event:09}-pairs.csv' for event in train_events],
            node_features=['r', 'eta', 'phi']
        )
    )
    test_data_loader = HitGraphDataLoader(
        dataset=HitGraphGeneratorDataset(
            hit_files=[dataset/f'event{event:09}-hits.csv' for event in test_events],
            pair_files=[dataset/f'event{event:09}-pairs.csv' for event in test_events],
            node_features=['r', 'eta', 'phi']
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
        device=device,
        true_sample_train_weight=true_sample_train_weight,
        false_sample_train_weight=false_sample_train_weight,
        save=save
    )
    trainer.train(
        train_data=train_data_loader,
        valid_data=test_data_loader,
        epochs=epoch
    )

    return trainer


def validate(
    model: nn.Module,
    device: str = 'cpu',
    save: Path = None
):
    """
    Validate model by use a single event.

    :param model: Model to be validate.
    :param device: Device use for evaluation.
    :param save: Path to save plots. None if no need to save.
    :return:
    """
    dataset = Path('output/pairs/100evts_volume8/')
    volume = 8
    events = [1001, 1020, 1033]

    for event in events:
        hits = pd.read_csv(dataset / f'event{event:09}-hits.csv')
        pairs = pd.read_csv(dataset / f'event{event:09}-pairs.csv')

        graph = HitGraph(
            hits=hits,
            pairs=pairs,
            node_features=['r', 'eta', 'phi'],
        )

        data_loader = HitGraphDataLoader(
            dataset=HitGraphDataset(
                graphs=[graph]
            )
        )

        model.to(device)

        # Mark model for evaluations.
        model.eval()

        # Use for loop to unpack data. It actually has only one graph.
        for batch_idx, (batch_input, batch_target) in enumerate(data_loader):
            # Assign to device.
            batch_input = [input_values.to(device) for input_values in batch_input]
            batch_target = batch_target.to(device)

            # Evaluate model and remove batch dimension
            predictions = model(batch_input).squeeze().detach().cpu().numpy()

            subdir = (save / f'evt{event:09}')

            subdir.mkdir(parents=True, exist_ok=True)

            threshold = 0.5

            hit_pair_gnn_prediction_plot_2d(
                hits=hits,
                pairs=pairs,
                predictions=predictions,
                truth=graph.truth,
                threshold=threshold,
                line_width=0.5,
                color_scheme=[
                    [0, 1, 0, 1.0],
                    [1, 0, 0, 1.0],
                    [0, 0, 0, 0.0],
                    [1, 1, 0, 0.0]
                ],
                save=subdir / 'true_false_positive.png'
            )

            hit_pair_gnn_prediction_plot_2d(
                hits=hits,
                pairs=pairs,
                predictions=predictions,
                truth=graph.truth,
                threshold=threshold,
                line_width=0.5,
                color_scheme=[
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 0.1],
                    [1, 1, 0, 0.1]
                ],
                save=subdir / 'true_false_negative.png'
            )

            hit_pair_gnn_prediction_plot_2d(
                hits=hits,
                pairs=pairs,
                predictions=predictions,
                truth=graph.truth,
                threshold=threshold,
                save=subdir / 'all.png'
            )


def create_model(
    node_input_dim: int = 3,
    node_hidden_dim: int = 8,
    edge_hidden_dim: int = 8,
    n_iter: int = 8
):
    return SegmentClassifier(
        node_input_dim=node_input_dim,
        node_hidden_dim=node_hidden_dim,
        edge_hidden_dim=edge_hidden_dim,
        n_iter=n_iter
    )


def load_model(name: str):
    model = create_model()
    model.load_state_dict(
        torch.load(
            f'output/models/{name}/model'
        )
    )

    return model


def load_checkpoint(name: str, checkpoint_idx: int):
    model = create_model()

    # Load checkpoint.
    model.load_state_dict(
        torch.load(
            f'output/models/{name}/checkpoints/model_checkpoint_{checkpoint_idx:03}.pth.tar'
        )['model']
    )

    return model


if __name__ == '__main__':
    schemes = {
        'iter3_20evts_3epochs_gpu_weighted(0.1)_eta': {
            'model': {
                'node_input_dim': 3,
                'node_hidden_dim': 64,
                'edge_hidden_dim': 64,
                'n_iter': 3
            },
            'train': {
                'epoch': 50,
                'true_sample_train_weight': 1.0,
                'false_sample_train_weight': 0.1,
                'device': 'cpu'
            }
        },
    }

    for name, settings in schemes.items():
        print(f'\n======{name}======\n')

        # Create model.
        model = create_model(**(settings['model']))

        # Load previous saved model.
        # model = load_model(name)

        # Load checkpoint.
        # checkpoint_idx = 20
        # model = load_checkpoint(name, checkpoint_idx)

        # Train new model.
        trainer = train_model(
            model=model,
            save=Path(f'output/models/{name}'),
            **(settings['train'])
        )

        validate(
            model=model,
            save=Path(f'output/plots/gnn/{name}')
        )
