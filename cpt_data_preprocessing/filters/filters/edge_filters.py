#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sklearn.cluster

from ..edge_filter import EdgeFilter


class TransverseMomentumFilter(EdgeFilter):
    """
    Filter base on transverse momentum.

    Transverse momentum can be estimate by phi_slope and z0.
    We adapt HEP.TrkX approximation.
    """
    # HEP.TrkX+ min Pt[GeV] to (phi_slope, z0[mm]).
    _pt_min = {
        2.00: (6e-4, 1500),
        1.50: (6e-4, 1500),
        1.00: (6e-4, 1500),
        0.75: (7.63e-4, 2500),
        0.60: (7.63e-4, 2950),
        0.50: (7.63e-4, 2950),
    }

    def __init__(self, pt_min: float = 2.0):
        if pt_min in self._pt_min:
            self.phi_slope, self.z0 = self._pt_min[pt_min]
        else:
            raise ValueError(
                f"Value not support. Supported values : {self._pt_min.keys()}"
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
        self.distance_squared = distance**2

    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        d2 = (hit_pairs['x_2'] - hit_pairs['x_1']) ** 2 + \
             (hit_pairs['y_2'] - hit_pairs['y_1']) ** 2 + \
             (hit_pairs['z_2'] - hit_pairs['z_1']) ** 2

        condition = (d2 <= self.distance_squared)

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


class RealEdgeLabeler(EdgeFilter):
    """
    A filter do nothing but create truth label on columns.

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

        # If truth is already labeled.
        if 'truth' in hit_pairs.columns:
            return hit_pairs

        missing_columns = self.missing_columns(hit_pairs, [
            'particle_id_1', 'particle_id_2', 'truth'
        ])
        if len(missing_columns) == 0:
            return hit_pairs

        particles = self.real_tracks.rename(
            mapper=lambda column_name: column_name + '_1',
            axis='columns'
        )
        missing_columns = particles.columns.difference(
            hit_pairs.columns
        ).tolist()
        particles = particles[['hit_id_1', *missing_columns]]
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
        particles = particles[['hit_id_2', *missing_columns]]
        hit_pairs = pd.merge(
            hit_pairs,
            particles,
            on='hit_id_2'
        )

        # Remove noise.
        truth_label = (hit_pairs["particle_id_1"] != 0) & \
                      (hit_pairs["weight_1"] != 0) & \
                      (hit_pairs["weight_2"] != 0)
        # Check particle id.
        truth_label = truth_label & (hit_pairs["particle_id_1"] == hit_pairs["particle_id_2"])

        return hit_pairs.assign(
            truth=truth_label
        )

    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        return hit_pairs


class RealEdgeFilter(RealEdgeLabeler):
    """
    A filter remove all edge except real track.

    This filter is use to generate truth label.
    """
    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        condition = hit_pairs['truth']

        return hit_pairs[condition]


class ClusterEdgeFilter(EdgeFilter):
    """
    Filter edges connect hit come from different group.

    Group ID need to be calculated and append to hits before apply this filter.

    For example, use DBSCANFilter will remove noise and generate group ID named "group_DBSCAN",
    hence you initialize this filter with group="group_DBSCAN" to filter edges by DBSCAN group.
    """
    def __init__(self, group):
        self.group = group

    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        column1 = self.group + '_1'
        column2 = self.group + '_2'

        condition = (hit_pairs[column1] == hit_pairs[column2])

        return hit_pairs[condition]
