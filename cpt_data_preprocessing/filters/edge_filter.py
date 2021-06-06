#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


def _compute_hit_pair_parameters(df_hit_pairs: pd.DataFrame, parameters: pd.Series = None):
    """
    Compute parameters of each hit pairs.
    By default, This will compute:
        'rho_1', 'r_1', 'phi_1', 'theta_1', 'eta_1',
        'rho_2', 'r_2', 'phi_2', 'theta_2', 'eta_2',
        'drho', 'dr', 'dphi', 'dz', 'dtheta', 'deta',
        'dR', 'z0', 'phi_slope'

    :param df_hit_pairs: Pair data frame.
    :param parameters: Required columns.
    Since all column are related, every column will be compute even if you didn't select it.
    If column not support, a KeyError will raise.
    Default is None and append for all support columns.

    :return: A data frame contains pair parameters.
    """
    r1 = np.sqrt(df_hit_pairs['x_1'] ** 2 +
                 df_hit_pairs['y_1'] ** 2)
    rho1 = np.sqrt(r1**2 + df_hit_pairs['z_1'] ** 2)
    z1 = df_hit_pairs['z_1']
    theta1 = np.arccos(z1 / rho1)
    phi1 = np.arctan(df_hit_pairs['y_1'] / df_hit_pairs['x_1'])
    # Pseudorapidity
    eta1 = -np.log(np.tan(theta1 / 2))

    r2 = np.sqrt(df_hit_pairs['x_2'] ** 2 +
                 df_hit_pairs['y_2'] ** 2)
    rho2 = np.sqrt(r2 ** 2 + df_hit_pairs['z_2'] ** 2)
    z2 = df_hit_pairs['z_2']
    theta2 = np.arccos(df_hit_pairs['z_2'] / rho2)
    phi2 = np.arctan(df_hit_pairs['y_2'] / df_hit_pairs['x_2'])
    # Pseudorapidity
    eta2 = -np.log(np.tan(theta2 / 2))

    dr = np.sqrt((df_hit_pairs['x_2'] - df_hit_pairs['x_1']) ** 2 +
                 (df_hit_pairs['y_2'] - df_hit_pairs['y_1']) ** 2)
    drho = np.sqrt(dr**2 + (df_hit_pairs['z_2'] - df_hit_pairs['z_1'])**2)

    # In range [-pi, pi]
    dtheta = theta2 - theta1
    dtheta[dtheta > np.pi] -= 2 * np.pi
    dtheta[dtheta < -np.pi] += 2 * np.pi

    # In range [-pi, pi]
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2 * np.pi
    dphi[dphi < -np.pi] += 2 * np.pi

    dz = np.abs(df_hit_pairs['z_2'] - df_hit_pairs['z_1'])
    deta = np.abs(eta2 - eta1)
    dR = np.sqrt(deta ** 2 + dphi ** 2)
    z0 = z1 - r1*dz/(r2-r1)
    phi_slope = dphi/(r2-r1)

    df_parameters = pd.DataFrame(data={
        'rho_1': rho1, 'r_1': r1, 'phi_1': phi1, 'theta_1': theta1, 'eta_1': eta1,
        'rho_2': rho1, 'r_2': r1, 'phi_2': phi1, 'theta_2': theta2, 'eta_2': eta2,
        'drho': drho, 'dr': dr, 'dphi': dphi, 'dz': dz, 'dtheta': dtheta, 'deta': deta,
        'dR': dR, 'z0': z0, 'phi_slope': phi_slope
    })

    if parameters is not None:
        return df_parameters[parameters]
    else:
        return df_parameters


class EdgeFilter(ABC):
    def compute_hit_pair_parameters(self, hit_pairs) -> pd.DataFrame:
        """
        Compute necessary parameters of each hit pair.
        This will compute:
            'rho_1', 'r_1', 'phi_1', 'theta_1', 'eta_1',
            'rho_2', 'r_2', 'phi_2', 'theta_2', 'eta_2',
            'drho', 'dr', 'dphi', 'dz', 'dtheta', 'deta',
            'dz', 'dR', 'z0', 'phi_slope'
        You can override this to compute other parameters.
        It is encouraged to check whether column already exist by calling **columns_exist**.

        :param hit_pairs: DataFrame contains hit pairs.
        :return: New DataFrame contains hit_pairs and necessary parameters.
        """

        # Default parameters.
        parameters = [
            'rho_1', 'r_1', 'phi_1', 'theta_1', 'eta_1',
            'rho_2', 'r_2', 'phi_2', 'theta_2', 'eta_2',
            'drho', 'dr', 'dphi', 'dz', 'dtheta', 'deta',
            'dR', 'z0', 'phi_slope'
        ]

        # If not exist, join default parameters.
        if self.columns_exist(hit_pairs, parameters) is False:
            df_parameters = _compute_hit_pair_parameters(hit_pairs)
            hit_pairs = hit_pairs.join(df_parameters)

        return hit_pairs

    @staticmethod
    def columns_exist(hit_pairs: pd.DataFrame, columns: [str]) -> bool:
        """
        Check columns is already exist in hit_pairs.

        :param hit_pairs: DataFrame contains hit pairs.
        :param columns: Columns to check.
        :return: True is all columns exists.
        """
        columns = pd.Series(columns)
        if all(columns.isin(hit_pairs.columns)):
            return True
        return False

    @abstractmethod
    def filter_pairs(self, hit_pairs) -> pd.DataFrame:
        """
        Filter hit pairs.
        Override to implement filter logic.

        :param hit_pairs: DataFrame contains hit pairs.
        :return: New filtered data frame.
        """
        pass
