#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from .filters import NodeFilter, EdgeFilter


def _missing_columns(df: pd.DataFrame, columns: [str]) -> [str]:
    """
    Check missing columns in hit_pairs.

    :param df: DataFrame to be test.
    :param columns: Required columns.
    :return: List of missing columns.
    """
    return pd.Index(columns).difference(
        df.columns
    ).tolist()


def _compute_default_hit_parameters(hits: pd.DataFrame) -> pd.DataFrame:
    """
    Compute parameters of each hits.

    By default, This will compute:
        'r', 'phi', 'theta', 'eta'

    For additional parameters, you can compute in filters if needed.

    :param hits: Hits data frame.

    :return: A hit data frame join with parameters.
    """
    missing_columns = _missing_columns(hits, [
        'r', 'phi', 'theta', 'eta'
    ])
    if len(missing_columns) == 0:
        return hits

    z = hits['z']
    r = np.sqrt(hits['x'] ** 2 + hits['y'] ** 2)
    theta = np.arctan2(r, z)
    phi = np.arctan2(hits['x'], hits['y'])
    # Pseudorapidity
    eta = -np.log(np.tan(theta / 2))

    return hits.assign(
        r=r,
        phi=phi,
        theta=theta,
        eta=eta
    )


def _compute_default_hit_pair_parameters(hit_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute parameters of each hit pairs.

    By default, This will compute:
        'dr', 'dphi', 'dz', 'dtheta', 'deta',
        'dR', 'z0', 'phi_slope'

    For additional parameters, you can compute in filters if needed.

    :param hit_pairs: Hits data frame.

    :return: A hit pair data frame join with pair parameters.
    """

    missing_columns = _missing_columns(hit_pairs, [
        'dr', 'dphi', 'dz', 'dtheta', 'deta', 'dR', 'z0', 'phi_slope'
    ])
    if len(missing_columns) == 0:
        return hit_pairs

    dr = hit_pairs['r_2'] - hit_pairs['r_1']

    # In range [-pi, pi]
    dtheta = hit_pairs['theta_2'] - hit_pairs['theta_1']
    dtheta[dtheta > np.pi] -= 2 * np.pi
    dtheta[dtheta < -np.pi] += 2 * np.pi

    # In range [-pi, pi]
    dphi = hit_pairs['phi_2'] - hit_pairs['phi_1']
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < -np.pi] += 2 * np.pi

    dz = np.abs(hit_pairs['z_2'] - hit_pairs['z_1'])
    deta = np.abs(hit_pairs['eta_2'] - hit_pairs['eta_1'])
    dR = np.sqrt(deta ** 2 + dphi ** 2)
    z0 = hit_pairs['z_1'] - hit_pairs['r_1'] * dz / dr
    phi_slope = dphi / dr

    return hit_pairs.assign(
        dr=dr,
        dphi=dphi,
        dz=dz,
        dtheta=dtheta,
        deta=deta,
        dR=dR,
        z0=z0,
        phi_slope=phi_slope
    )


def _apply_node_filters(hits: pd.DataFrame, filters: [NodeFilter] = None):
    """
    Apply each node filter to hits.

    :param hits: Dataframe contains hit which filter will apply to.
    :param filters: Node filters will apply.
    :return: Filtered hit dataframe.
    """
    for filter in filters or []:
        print(f'Applying {type(filter).__name__} to {len(hits)} hits...')

        hits = filter.filter_hits(hits)

        print(f'{type(filter).__name__} complete and {len(hits)} hits remained.\n')

    return hits


def _apply_edge_filters(hit_pairs: pd.DataFrame, filters: [EdgeFilter] = None):
    """
    Apply each edge filter hit pairs.

    :param hit_pairs: Dataframe contains hit pairs which filter will apply to.
    :param filters: Edge filters will apply.
    :return: Filtered hit pair dataframe.
    """
    for filter in filters or []:
        print(f'Applying {type(filter).__name__} to {len(hit_pairs)} hit pairs...')

        hit_pairs = filter.compute_hit_pair_parameters(hit_pairs)
        hit_pairs = filter.filter_pairs(hit_pairs)

        print(f'{type(filter).__name__} complete and {len(hit_pairs)} pairs remained.\n')

    return hit_pairs


def _layer_combinations(
    start_layers: [int] = None,
    end_layers: [int] = None,
):
    """
    Compute layer combinations.

    This will compute all layer combinations which satisfy layer1 > layer2.

    :param start_layers: Start layers.
    :param end_layers: End layers.
    :return: List of layer combinations.
    """
    start_layers = start_layers or [2, 4, 6]
    end_layers = end_layers or [4, 6, 8]

    """layer_combinations = [
        (layer1, layer2)
        for layer1 in start_layers
        for layer2 in end_layers
        if layer2 > layer1
    ]"""
    layer_combinations = list(zip(start_layers, end_layers))

    return layer_combinations


def pair(
    hits: pd.DataFrame,
    start_layers: [int] = None,
    end_layers: [int] = None,
    node_filters: [NodeFilter] = None,
    edge_filters: [EdgeFilter] = None
):
    """
    Pair hits with given filters between layers.

    :param hits: Dataframe contain hits data.
    :param start_layers: Start layers.
    :param end_layers: End layers.
    :param node_filters: Node filters to remove unwanted nodes.
    :param edge_filters: Edge filters to remove unwanted edges.

    :return:
    Filtered hits dataframe and a dictionary contain pairing result.
    Dictionary keys are (Start layer ID, End layer ID) and values are Dataframe contains necessary parameters.
    """
    hits = _compute_default_hit_parameters(hits)
    hits = _apply_node_filters(hits, node_filters)

    # Group hits with layer.
    layer_combinations = _layer_combinations(start_layers, end_layers)
    hits_on_layers = {
        layer_id: hits_on_layer
        for layer_id, hits_on_layer in hits.groupby('layer_id')
    }

    print(f'Start pairing between:\n{layer_combinations}\n')

    edge_dfs = {}
    for layer1_id, layer2_id in layer_combinations:
        hits_on_layer1 = hits_on_layers.get(layer1_id, pd.DataFrame(
            columns=hits.columns
        ))
        hits_on_layer2 = hits_on_layers.get(layer2_id, pd.DataFrame(
            columns=hits.columns
        ))

        print(f'Pairing between '
              f'layer{layer1_id}({len(hits_on_layer1)} hits) x '
              f'layer{layer2_id}({len(hits_on_layer2)} hits)...')

        hit_pairs = pd.merge(
            hits_on_layer1.reset_index(),
            hits_on_layer2.reset_index(),
            how='inner',
            on='evtid',
            suffixes=('_1', '_2')
        )

        hit_pairs = _compute_default_hit_pair_parameters(hit_pairs)
        hit_pairs = _apply_edge_filters(hit_pairs, edge_filters)

        edge_dfs[layer1_id, layer2_id] = hit_pairs

    return hits, edge_dfs
