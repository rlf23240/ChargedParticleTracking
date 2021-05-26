#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

from processing import read, pairing
from plots import (
    hit_position_plot_3d,
    hit_position_plot_2d,
    hit_position_plot_2d_no_group,
    hit_pair_plot_2d,
    hit_real_track_plot_2d
)

if __name__ == '__main__':
    dataset = 'data/train_10evts/train_10evts/'
    event = 1000
    volume = 8

    pd.set_option('display.max_columns', None)

    hits, particles, truth_labels = read(
        dataset='data/train_10evts/train_10evts/',
        event=event,
        volume=volume
    )

    print(hits.columns)
    print(particles.columns)
    print(truth_labels.columns)

    # Since number of hits is too large,
    # we cheat a little bit here by filter out known particles.
    particle_ids = particles['particle_id'].unique()[-500:]
    hit_ids = truth_labels[truth_labels['particle_id'].isin(particle_ids)]['hit_id'].to_numpy()
    hits_filtered = hits[hits['hit_id'].isin(hit_ids)]
    hits_filtered = pd.merge(
        hits_filtered, truth_labels,
        how='inner',
        on='hit_id'
    )

    pair_output = Path(dataset) / f'pairs/evt{event}_vol{volume}/'
    if pair_output.exists() is False:
        pair_output.mkdir(parents=True)

    pair_dfs = pairing(
        hits=hits_filtered,
        start_layer=[2, 4, 6],
        end_layer=[4, 6, 8]
    )

    hit_position_plot_3d(hits)
    hit_position_plot_2d(hits)
    hit_position_plot_2d_no_group(hits_filtered)
    hit_pair_plot_2d(hits_filtered, pair_dfs)
    hit_real_track_plot_2d(hits_filtered, pair_dfs)
