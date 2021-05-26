#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from mpl_toolkits.mplot3d import art3d


def hit_position_plot_3d(hits: pd.DataFrame):
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
    plt.savefig('hit_position_plot_3d.png')


def hit_position_plot_2d(hits: pd.DataFrame):
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

    plt.savefig('hit_position_plot_2d.png')


def hit_position_plot_2d_no_group(hits):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        hits['x'].to_numpy(),
        hits['y'].to_numpy(),
        s=1,
    )
    ax.axis('equal')

    plt.savefig('hit_position_plot_2d_no_group.png')


def hit_pair_plot_2d(hits, pair_dfs):
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

    plt.savefig('hit_pair_plot_2d.png')


def hit_pair_plot_3d(hits, pair_dfs):
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

    plt.savefig('hit_pair_plot_3d.png')


def hit_real_track_plot_2d(hits, pair_dfs):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for layer, df in pair_dfs.items():
        hit_particle = {
            hit: particles
            for hit, particles in hits[['hit_id', 'particle_id']].to_numpy()
        }
        line_collection = mc.LineCollection(
            [((x1, y1), (x2, y2))
             for h1, x1, y1, h2, x2, y2 in df[[
                    'hit_id_1', 'x_1', 'y_1',
                    'hit_id_2', 'x_2', 'y_2'
                ]].to_numpy()
             if hit_particle[h1] != 0 and hit_particle[h2] != 0
             if hit_particle[h1] == hit_particle[h2]
             ],
            linewidths=0.1,
            color=[0, 0, 1, 1]
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

    plt.savefig('hit_real_track_plot_2d.png')
