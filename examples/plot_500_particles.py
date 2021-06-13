#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd

import cpt_data_reader
from cpt_data_preprocessing import pair, filters

from cpt_plots import (
    hit_position_plot_3d,
    hit_position_plot_2d,
    hit_position_plot_2d_no_group,
    hit_pair_plot_2d
)

if __name__ == '__main__':
    dataset = '../data/train_10evts/train_10evts/'
    event = 1000
    volume = 8

    pd.set_option('display.max_columns', None)

    hits, particles, truth_labels = cpt_data_reader.read(
        dataset=dataset,
        event=event,
        volume=volume
    )

    print(hits.columns)
    print(particles.columns)
    print(truth_labels.columns)

    hits_filtered, pair_dfs = pair(
        hits=hits,
        start_layers=[2, 4, 6],
        end_layers=[4, 6, 8],
        node_filters=[
            # Since number of hits is too large,
            # we cheat a little bit here by filter out known particles.
            filters.RealNodeFilter(
                n_particles=500,
                real_tracks=truth_labels,
                particles=particles
            )
        ],
        edge_filters=[
            # No physical reason. Just for visualization.
            # filters.DistanceFilter(distance=100),
            # For more accurate filter, use TransverseMomentumFilter.
            filters.TransverseMomentumFilter(pt_min=2.0),
        ]
    )

    hits_filtered, real_track_dfs = pair(
        hits=hits_filtered,
        start_layers=[2, 4, 6],
        end_layers=[4, 6, 8],
        node_filters=[
            # Since number of hits is too large,
            # we cheat a little bit here by filter out known particles.
            filters.RealNodeFilter(
                n_particles=500,
                real_tracks=truth_labels,
                particles=particles
            )
        ],
        edge_filters=[
            filters.RealEdgeFilter(
                real_tracks=truth_labels
            )
        ]
    )

    output_dir = Path('output/plots/500_particles')
    output_dir.mkdir(parents=True, exist_ok=True)

    hit_position_plot_3d(
        hits=hits_filtered,
        save=output_dir/'hit_position_plot_3d.png'
    )
    hit_position_plot_2d(
        hits=hits_filtered,
        save=output_dir/'hit_position_plot_2d.png'
    )
    hit_position_plot_2d_no_group(
        hits=hits_filtered,
        save=output_dir/'hit_position_plot_2d_no_group.png'
    )
    hit_pair_plot_2d(
        hits=hits_filtered,
        pair_dfs=pair_dfs,
        save=output_dir/'hit_pair_plot_2d.png'
    )
    hit_pair_plot_2d(
        hits=hits_filtered,
        pair_dfs=real_track_dfs,
        save=output_dir/'hit_real_track_plot_2d.png'
    )
