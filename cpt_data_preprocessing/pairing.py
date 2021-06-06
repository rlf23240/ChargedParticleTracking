#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from .filters import NodeFilter, EdgeFilter


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

    layer_combinations = [
        (layer1, layer2)
        for layer1 in start_layers
        for layer2 in end_layers
        if layer2 > layer1
    ]

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

    :return: A dictionary contain pairing result.
    Keys are (Start layer ID, End layer ID) and values are Dataframe contains necessary parameters.
    """
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

        hit_pairs = _apply_edge_filters(hit_pairs, edge_filters)

        edge_dfs[layer1_id, layer2_id] = hit_pairs

    return edge_dfs
