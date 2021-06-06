#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def hit_position_plot_2d(hits: pd.DataFrame, save: Path = None):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for layer, hits_on_layer in hits.groupby('layer_id'):
        ax.scatter(
            hits_on_layer['x'].to_numpy(),
            hits_on_layer['y'].to_numpy(),
            s=1, label=f'Layer {layer}'
        )

    ax.axis('equal')
    ax.legend()

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def hit_position_plot_2d_no_group(hits, save: Path = None):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        hits['x'].to_numpy(),
        hits['y'].to_numpy(),
        s=1,
    )
    ax.axis('equal')

    if save is not None:
        plt.savefig(save)
    else:
        plt.show()


def hit_position_plot_3d(hits: pd.DataFrame, save: Path = None):
    fig, ax = plt.subplots(
        figsize=(8, 8),
        subplot_kw={
            'projection': '3d'
        }
    )

    for layer, hits_on_layer in hits.groupby('layer_id'):
        ax.scatter(
            hits_on_layer['x'],
            hits_on_layer['y'],
            hits_on_layer['z'],
            s=1, label=f'Layer {layer}'
        )

    ax.view_init(elev=80, azim=45)

    plt.legend()
    if save is not None:
        plt.savefig(save)
    else:
        plt.show()
