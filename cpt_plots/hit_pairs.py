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

