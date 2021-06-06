#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import pandas as pd


class EdgeFilter(ABC):
    def compute_hit_pair_parameters(self, hit_pairs) -> pd.DataFrame:
        """
        Override to compute necessary parameters of each hit pair.
        Default is do nothing.

        :param hit_pairs: DataFrame contains hit pairs.
        :return: New DataFrame contains hit_pairs and necessary parameters.
        """
        return hit_pairs

    @staticmethod
    def missing_columns(hit_pairs: pd.DataFrame, columns: [str]) -> [str]:
        """
        Check missing columns in hit_pairs.

        :param hit_pairs: DataFrame contains hit pairs.
        :param columns: Columns to check.
        :return: List of missing columns.
        """
        return pd.Index(columns).difference(
            hit_pairs.columns
        ).tolist()

    @abstractmethod
    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        """
        Filter hit pairs.
        Override to implement filter logic.

        :param hit_pairs: DataFrame contains hit pairs.
        :return: New filtered data frame.
        """
        pass
