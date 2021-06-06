#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import pandas as pd


class NodeFilter(ABC):
    @abstractmethod
    def filter_hits(self, hits: pd.DataFrame) -> pd.DataFrame:
        """
        Filter hit.
        Override to implement filter logic.

        :param hits: DataFrame contains hit pairs.
        :return: New filtered data frame.
        """
        pass
