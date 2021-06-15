#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import pandas as pd

import cpt_data_reader
from cpt_data_preprocessing import pair, filters

if __name__ == '__main__':
    # Prepare data.
    dataset = '../data/train_100_events/'
    volume = 8

    # known bug: event 1087 have some problem.
    events = range(1000, 1010)

    output_dir = Path(f'output/pairs/2GeV/100evts_volume{volume}')
    output_dir.mkdir(parents=True, exist_ok=True)

    for event in events:
        hits, particles, truth_labels = cpt_data_reader.read(
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
                filters.DBSCANFilter(
                    eps=0.05,
                    min_pts=3
                ),
            ],
            edge_filters=[
                filters.TransverseMomentumFilter(pt_min=2),
                filters.ClusterEdgeFilter(group='group_DBSCAN'),
                filters.RealEdgeLabeler(
                    real_tracks=truth_labels
                )
            ]
        )

        pairs = pd.concat(pair_dfs.values())

        hits.to_csv(output_dir/f'event{event:09}-hits.csv')
        pairs.to_csv(output_dir/f'event{event:09}-pairs.csv')

    # Output configurations.
    # TODO: Get parameters from filter directly.
    with open(output_dir / 'config.json', 'w') as fp:
        json.dump({
            'layers': [(2, 4), (4, 6), (6, 8)],
            'pt_min': 2.0,
            'DBSCAN': {
                'eps': 0.05,
                'min_pts': 3,
            }
        }, fp)
