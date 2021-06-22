#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

import cpt_data_reader
from cpt_data_preprocessing import pair, filters
from cpt_plots import hit_pair_plot_2d

if __name__ == '__main__':
    dataset = '../data/train_10evts/train_10evts/'
    event = 1000
    volume = 8

    pd.set_option('display.max_columns', None)

    hits, particles, truth_labels = cpt_data_reader.read_event(
        dataset=dataset,
        event=event,
        volume=volume
    )

    hits, pair_dfs = pair(
        hits=hits,
        start_layers=[2, 4, 6],
        end_layers=[4, 6, 8],
        node_filters=[
            filters.NoiseFilter(
                real_tracks=truth_labels
            ),
            filters.SameLayerFilter(
                real_tracks=truth_labels,
            ),
            filters.RealNodeFilter(
                real_tracks=truth_labels,
                particles=particles
            ),
            # filters.DBSCANFilter(
            #     eps=0.05,
            #     min_pts=20
            # ),
        ],
        edge_filters=[
            # filters.TransverseMomentumFilter(pt_min=0.5),
            # filters.ClusterEdgeFilter(group='group_DBSCAN'),
            filters.RealEdgeFilter(
                real_tracks=truth_labels,
            )
        ]
    )

    output_dir = Path('output/plots/filters')
    output_dir.mkdir(parents=True, exist_ok=True)

    hit_pair_plot_2d(
        hits=hits,
        pair_dfs=pair_dfs,
        save=output_dir/'hit_pair_plot_2d.png'
    )
