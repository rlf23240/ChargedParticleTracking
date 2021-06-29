#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import json
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optimizers
import torch.nn as nn
import torch.nn.functional as functional

from cpt_data_preprocessing import HitGraph
from cpt_data_reader import HitGraphDataset, HitGraphGeneratorDataset, HitGraphDataLoader
from cpt_plots import hit_pair_gnn_prediction_plot_2d
from cpt_plots import plot_acc_curve, plot_loss_curve, plot_auc_roc, plot_confusion_matrix
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
    features: [str] = None,
    epoch: int = 50,
    device: str = 'cpu',
    true_sample_train_weight: float = 1.0,
    false_sample_train_weight: float = 1.0,
    save: Path = None
) -> Trainer:
    """
    Train a GNN model.

    :param: model: Model to train.
    :param: features: Features use to train.
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

    features = features or ['r', 'eta', 'phi']

    # Form dataset.
    train_data_loader = HitGraphDataLoader(
        dataset=HitGraphGeneratorDataset(
            hit_files=[dataset/f'event{event:09}-hits.csv' for event in train_events],
            pair_files=[dataset/f'event{event:09}-pairs.csv' for event in train_events],
            node_features=features
        )
    )
    test_data_loader = HitGraphDataLoader(
        dataset=HitGraphGeneratorDataset(
            hit_files=[dataset/f'event{event:09}-hits.csv' for event in test_events],
            pair_files=[dataset/f'event{event:09}-pairs.csv' for event in test_events],
            node_features=features
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
    features: [str] = None,
    device: str = 'cpu',
    save: Path = None
):
    """
    Validate model by use a single event.

    :param model: Model to be validate.
    :param features: Features use for validate.
    :param device: Device use for evaluation.
    :param save: Path to save plots. None if no need to save.
    :return:
    """
    dataset = Path('output/pairs/100evts_volume8/')
    events = [1001, 1020, 1033]

    features = features or ['r', 'eta', 'phi']

    for event in events:
        hits = pd.read_csv(dataset / f'event{event:09}-hits.csv')
        pairs = pd.read_csv(dataset / f'event{event:09}-pairs.csv')

        graph = HitGraph(
            hits=hits,
            pairs=pairs,
            node_features=features
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

            if save is not None:
                subdir = (save / f'evt{event:09}')
                subdir.mkdir(parents=True, exist_ok=True)
            else:
                subdir = None

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
                save=subdir / 'true_false_positive.png' if save is not None else None
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
                save=subdir / 'true_false_negative.png' if save is not None else None
            )

            hit_pair_gnn_prediction_plot_2d(
                hits=hits,
                pairs=pairs,
                predictions=predictions,
                truth=graph.truth,
                threshold=threshold,
                save=subdir / 'all.png' if save is not None else None
            )


def plot_learning_curve(
    model: nn.Module,
    features: [str] = None,
    save: Path = None
):
    dataset = Path('output/pairs/100evts_volume8/')
    event = 1001

    features = features or ['r', 'eta', 'phi']

    hits = pd.read_csv(dataset / f'event{event:09}-hits.csv')
    pairs = pd.read_csv(dataset / f'event{event:09}-pairs.csv')

    if save is not None:
        subdir = (save / f'evt{event:09}')
        subdir.mkdir(parents=True, exist_ok=True)
    else:
        subdir = None

    graph = HitGraph(
        hits=hits,
        pairs=pairs,
        node_features=features
    )

    plot_acc_curve(
        history=history,
        save=subdir / 'acc.png' if save is not None else None
    )
    plot_loss_curve(
        history=history,
        save=subdir / 'loss.png' if save is not None else None
    )
    plot_auc_roc(
        model=model,
        graph=graph,
        save=subdir / 'roc.png' if save is not None else None
    )
    plot_confusion_matrix(
        model=model,
        graph=graph,
        save=subdir / 'confusion_matrix.png' if save is not None else None
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


def load_model(name: str, model_setting: dict):
    model = create_model(**model_setting)
    model.load_state_dict(
        torch.load(
            f'output/models/{name}/model'
        )
    )

    return model


def load_checkpoint(name: str, model_setting: dict, checkpoint_idx: int):
    model = create_model(**model_setting)

    # Load checkpoint.
    checkpoint = torch.load(
        f'output/models/{name}/checkpoints/model_checkpoint_{checkpoint_idx:03}.pth.tar',
        map_location=torch.device('cpu')
    )

    # Load model.
    model.load_state_dict(
        checkpoint['model']
    )

    # Load history.
    history = checkpoint['history']

    return model, history


if __name__ == '__main__':
    config_path = 'configs/20evts_50epochs_weight(0.1)_eta_light.json'

    with open(config_path) as fp:
        schemes = json.load(fp)

        for name, settings in schemes.items():
            print(f'\n======{name}======\n')

            # Create model.
            model = create_model(
                **(settings['model'])
            )
            
            # Load previous saved model.
            # model = load_model(name, settings['model'])

            # Load checkpoint.
            """checkpoint_idx = 49
            model, history = load_checkpoint(
                name=name,
                model_setting=settings['model'],
                checkpoint_idx=checkpoint_idx
            )"""

            # Train new model.
            trainer = train_model(
                model=model,
                save=Path(f'output/models/{name}'),
                **(settings['train'])
            )

            validate(
                model=model,
                save=Path(f'output/plots/gnn/{name}'),
                **(settings['validate'])
            )

            # Plot learning curves.
            """plot_learning_curve(
                model=model,
                features=settings['train']['features'],
                save=Path(f'output/plots/train/{name}')
            )"""

