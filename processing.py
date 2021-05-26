#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd


def read(dataset, event, volume):
    """

    :param dataset: Name of dataset.
    :param event: Event ID.
    :param volume: Volume filter.
    :return: Three DataFrame: hits, particles and truth labels.
    """
    print(f"Reading {dataset}...")

    path = Path(dataset)

    hits = pd.read_csv(path / f'event{event:09}-hits.csv')
    selected_hits = hits[hits['volume_id'] == volume]
    selected_hits = selected_hits.assign(evtid=event)
    print(f"{len(selected_hits)} hit record read: ")
    print(selected_hits, "\n")

    particles = pd.read_csv(path / f'event{event:09}-particles.csv')
    print(f"{len(particles)} particle record read: ")
    print(particles, "\n")

    truth = pd.read_csv(path / f'event{event:09}-truth.csv')
    print(f"{len(truth)} truth label read: ")
    print(truth, "\n")

    return selected_hits, particles, truth


def pairing(
    hits: pd.DataFrame,
    output: Path = None,
    start_layer: [int] = None,
    end_layer: [int] = None
):
    layer_start = start_layer or [2, 4, 6]
    layer_end = end_layer or [4, 6, 8]

    layer_combinations = [
        (layer1, layer2)
        for layer1 in layer_start
        for layer2 in layer_end
        if layer2 > layer1
    ]
    print(f'Start pairing between:\n{layer_combinations}')

    df_edges = {}

    hits_on_layers = {
        layer_id: hits_on_layer
        for layer_id, hits_on_layer in hits.groupby('layer_id')
    }

    r2 = 100**2
    for layer1_id, layer2_id in layer_combinations:
        hits_on_layer1 = hits_on_layers[layer1_id]
        hits_on_layer2 = hits_on_layers[layer2_id]

        print(f'{layer1_id}: {len(hits_on_layer1)} x {layer2_id}: {len(hits_on_layer2)}')

        hit_pairs = pd.merge(
            hits_on_layer1.reset_index(),
            hits_on_layer2.reset_index(),
            how='inner',
            on='evtid',
            suffixes=('_1', '_2')
        )

        ds2 = (hit_pairs['x_2'] - hit_pairs['x_1'])**2 + \
              (hit_pairs['y_2'] - hit_pairs['y_1'])**2 + \
              (hit_pairs['z_2'] - hit_pairs['z_1'])**2

        df_pairs = hit_pairs[[
            #'evtid',
            'hit_id_1', 'hit_id_2',
            #'layer_id_1', 'layer_id_2',
            'x_1', 'y_1', 'z_1',
            'x_2', 'y_2', 'z_2',
        ]].assign(ds2=ds2)

        df_pairs = df_pairs[df_pairs['ds2'] < r2]

        if output is not None:
            df_pairs.to_csv(output / f'{layer1_id}_{layer2_id}.csv')
        else:
            df_edges[layer1_id, layer2_id] = df_pairs

    return df_edges
