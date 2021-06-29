#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import art3d


def hit_pair_plot_2d(hits, pair_dfs, save: Path = None):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for layer, df in pair_dfs.items():
        line_collection = mc.LineCollection(
            [((x1, y1), (x2, y2))
             for x1, y1, x2, y2 in df[[
                'x_1', 'y_1',
                'x_2', 'y_2'
              ]].to_numpy()
             ],
            linewidths=0.1,
            color=[0, 0, 1, 0.6]
        )
        ax.add_collection(line_collection)

    for layer_id, hits_on_layer in hits.groupby('layer_id'):
        ax.scatter(
            hits_on_layer['x'], hits_on_layer['y'],
            s=2.0, label=f'Layer {layer_id}', color=[0, 0, 1, 1]
        )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def hit_pair_plot_3d(hits, pair_dfs, save: Path = None):
    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={
            'projection': '3d'
        }
    )

    for layer, df in pair_dfs.items():
        line_collection = art3d.Line3DCollection(
            [((x1, y1, z1), (x2, y2, z2))
             for x1, y1, z1, x2, y2, z2 in df[[
                    'x_1', 'y_1', 'z_1',
                    'x_2', 'y_2', 'z_2'
                ]].to_numpy()
             ],
            linewidths=0.1,
            color=[0, 0, 1, 0.6]
        )
        ax.add_collection(line_collection)

    for layer_id, hits_on_layer in hits.groupby('layer_id'):
        ax.scatter(
            hits_on_layer['x'], hits_on_layer['y'], hits_on_layer['z'],
            s=2.0, label=f'Layer {layer_id}', color=[0, 0, 1, 1]
        )
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_ylabel('z')

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def hit_pair_gnn_prediction_plot_2d(
    hits,
    pairs,
    predictions,
    truth,
    threshold: float = 0.5,
    color_scheme: list = None,
    line_width: float = 0.1,
    save: Path = None
):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for layer_id, hits_on_layer in hits.groupby('layer_id'):
        ax.scatter(
            hits_on_layer['x'], hits_on_layer['y'],
            s=2.0, color=[0, 0, 1, 1]
        )

    # Confusion state.
    true_positive = []
    false_positive = []
    true_negative = []
    false_negative = []

    # Classify edges.
    for edge_idx, (x1, y1, z1, x2, y2, z2) in enumerate(pairs[[
        'x_1', 'y_1', 'z_1',
        'x_2', 'y_2', 'z_2'
    ]].to_numpy()):
        # FIXIT: truth[edge_idx] is True always return False.
        # Maybe have some operator overriding problem?
        if (predictions[edge_idx] > threshold) and (truth[edge_idx] > 0.5):
            true_positive.append(
                ((x1, y1), (x2, y2))
            )
        elif (predictions[edge_idx] <= threshold) and (truth[edge_idx] <= 0.5):
            true_negative.append(
                ((x1, y1), (x2, y2))
            )
        elif (predictions[edge_idx] > threshold) and (truth[edge_idx] <= 0.5):
            false_positive.append(
                ((x1, y1), (x2, y2))
            )
        elif (predictions[edge_idx] <= threshold) and (truth[edge_idx] > 0.5):
            false_negative.append(
                ((x1, y1), (x2, y2))
            )

    labels = [
        'True Positive',
        'False Positive',
        'True Negative',
        'False Negative'
    ]

    confusion = [
        true_positive,
        false_positive,
        true_negative,
        false_negative
    ]

    color_scheme = color_scheme or [
        [0, 1, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0.1],
        [1, 1, 0, 1]
    ]

    print('Summary:')
    for lines, label in zip(confusion, labels):
        print(f"{label}: {len(lines)}")
    print('\n')

    drawables = reversed(list(zip(confusion, color_scheme, labels)))
    for lines, color, label in drawables:
        # Draw only for non-zero alpha.
        if color[3] > 0:
            line_collection = mc.LineCollection(
                lines,
                linewidths=line_width,
                color=color,
                label=label
            )
            ax.add_collection(line_collection)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')

    ax.legend()

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
