#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from .edge_filter import EdgeFilter
from .node_filter import NodeFilter

# HEP.TrkX min Pt[GeV] to (phi_slope, z0[mm]).
_pt_min = {
    2.00: (6e-4, 100),
    1.50: (6e-4, 100),
    1.00: (6e-4, 100),
    0.75: (7.63e-4, 100),
    0.60: (7.63e-4, 100),
    0.50: (7.63e-4, 100),
}


class TransverseMomentumFilter(EdgeFilter):
    """
    Filter base on transverse momentum.

    Transverse momentum can be estimate by phi_slope and z0.
    We adapt HEP.TrkX approximation.
    """
    def __init__(self, pt_min: float = 2.0):
        if pt_min in _pt_min:
            self.phi_slope, self.z0 = _pt_min[pt_min]
        else:
            raise ValueError(
                f"Value not support. Supported values : {_pt_min.keys()}"
            )

    def filter_pairs(self, hit_pairs: pd.DataFrame) -> pd.DataFrame:
        condition = (abs(hit_pairs['phi_slope']) <= self.phi_slope) & (abs(hit_pairs['z0']) <= self.z0)

        return hit_pairs[condition]


class DistanceFilter(EdgeFilter):
    """
    Filter base on distance in 3D space.

    This filter does not have solid physics assumption
    unless you compute distance with theory.
    Hence this filter include here is just for rough estimation and visualization.
    """
    def __init__(self, distance: float = 100):
        self.distance = distance

    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        condition = hit_pairs['drho'] <= self.distance

        return hit_pairs[condition]


class MLFilter(EdgeFilter):
    """
    A filter base on pre-trained model.

    Model should be a keras model with single output in [0.0, 1.0].
    However, you can assign features and threshold to filter edges.

    Features should be one of default column of parameters,
    or you will need to create new filter override **compute_hit_pair_parameters**.

    Threshold is a value between 0 and 1,
    which tell filter remove edge with score below this value.
    This depends on how your model trained.
    """
    def __init__(self, model, features: [str], threshold: float):
        self.model = model
        self.features = features
        self.threshold = threshold

    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        predictions = self.model.predict(hit_pairs[self.features])
        condition = [prediction[0] >= self.threshold for prediction in predictions]

        return hit_pairs[condition]


class RealEdgeFilter(EdgeFilter):
    """
    A filter remove all edge except real track.

    This filter is use to generate truth label.
    """
    def __init__(self, real_tracks):
        """
        :param real_tracks: A dataframe of truth label that maps hit to particle ID.
        """
        self.real_tracks = real_tracks

    def compute_hit_pair_parameters(self, hit_pairs) -> pd.DataFrame:
        hit_pairs = super().compute_hit_pair_parameters(
            hit_pairs=hit_pairs
        )

        if self.columns_exist(hit_pairs, [
            'particle_id_1', 'particle_id_2'
        ]):
            return hit_pairs

        particles = self.real_tracks.rename(
            mapper=lambda column_name: column_name + '_1',
            axis='columns'
        )
        missing_columns = particles.columns.difference(
            hit_pairs.columns
        ).tolist()
        particles = particles[
            ['hit_id_1'] + missing_columns
        ]
        hit_pairs = pd.merge(
            hit_pairs,
            particles,
            on='hit_id_1'
        )

        particles = self.real_tracks.rename(
            mapper=lambda column_name: column_name + '_2',
            axis='columns'
        )
        missing_columns = particles.columns.difference(
            hit_pairs.columns
        ).tolist()
        particles = particles[
            ['hit_id_2'] + missing_columns
        ]
        hit_pairs = pd.merge(
            hit_pairs,
            particles,
            on='hit_id_2'
        )

        return hit_pairs

    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        condition = (hit_pairs["particle_id_1"] == hit_pairs["particle_id_2"]) \
            & (hit_pairs["particle_id_1"] != 0)

        return hit_pairs[condition]


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
