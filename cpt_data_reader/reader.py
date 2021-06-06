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