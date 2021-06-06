#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import cluster

from ..node_filter import NodeFilter


class NoiseFilter(NodeFilter):
    def filter_hits(self, hits) -> pd.DataFrame:
        # TODO: Implement noise filter.
        return hits


class SameLayerFilter(NodeFilter):
    def filter_hits(self, hits) -> pd.DataFrame:
        # TODO: Implement same layer filter.
        return hits


class RealNodeFilter(NodeFilter):
    """
    A filter reserve numbers of hits by truth label.

    This filter is use to generate truth label.
    """
    def __init__(self, real_tracks, particles, n_particles: int = 500):
        """
        :param real_tracks: A dataframe of truth label that maps hit to particle ID.
        :param particles: A dataframe contains parameters of each particle. Use to select particles.
        """
        self.real_tracks = real_tracks
        # Select particles.
        self.particle_ids = particles['particle_id'].unique()[-n_particles:]
        # Compute accepted hit IDs.
        self.hit_ids = real_tracks[
            real_tracks['particle_id'].isin(self.particle_ids)
        ]['hit_id'].to_numpy()

    def filter_hits(self, hits) -> pd.DataFrame:
        missing_columns = self.real_tracks.columns.difference(
            hits.columns
        ).tolist()
        particles = self.real_tracks[
            ['hit_id'] + missing_columns
        ]
        hits = pd.merge(
            hits,
            particles,
            on='hit_id',
            how='inner'
        )

        condition = (hits['particle_id'] != 0) & (hits['particle_id'].isin(self.particle_ids))
        hits = hits[condition]

        return hits


class DBSCANFilter(NodeFilter):
    """
    A filter base on DBSCAN algorithm.

    This filter mark each hit with DBSCAN cluster group and remove noise.

    You can combine this filter with **ClusterEdgeFilter** to filter edges.
    To filter edges, you need to specify group name with group_DBSCAN.
    """
    # HEP.TrkX+ min Pt[GeV] to (epsilon, min_pts).
    _pt_min = {
        2.00: (0.22, 3),
        1.50: (0.18, 3),
        1.00: (0.10, 3),
        0.75: (0.08, 3),
        0.60: (0.06, 3),
        0.50: (0.05, 3),
    }

    def __init__(self, eps: float = 0.05, min_pts: int = 20):
        self.epsilon = eps
        self.min_pts = min_pts

        self.clustering = cluster.DBSCAN(
            eps=self.epsilon,
            min_samples=self.min_pts
        )

    def filter_hits(self, hits) -> pd.DataFrame:
        predictions = self.clustering.fit_predict(
            hits[['eta', 'phi']]
        )
        hits = hits.assign(
            group_DBSCAN=predictions
        )

        print(f"[DBSCAN] Separate hits into {len(hits['group_DBSCAN'].unique())} groups.")

        return hits[hits['group_DBSCAN'] >= 0]
