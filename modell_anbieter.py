# =====================================================================================
# Anbietermodell
# =====================================================================================

import numpy as np


def profit(q_large, r, p_small, kappa):
    """
    Berechnet langfristigen durchschnittlichen Gewinn pro Periode G(r).
    """
    q = np.clip(q_large, 0.0, 1.0) 
    G_small = p_small - kappa
    G_large = 2.0 * (1.0 - np.asarray(r)) * p_small - kappa
    return ((1.0 - q) * G_small + q * G_large) / (1.0 + q)


def profit_without_discount(p_small, kappa):
    """
    Berechnet G(0).
    """
    return p_small - kappa


def profitability_threshold(p_small, kappa):
    """
    Berechnet Profitabilitätsschwelle kappa / (2 p_S) für p_S > 0.
    """
    return kappa / (2.0 * p_small)


def optimal_discount(r_values, profits, p_small, kappa):
    """
    Berechnet gewinnmaximierenden Rabatt r*.

    Ohne profitablen Rabatt (max G <= G(0)) gilt r* = NaN.
    """
    best = np.argmax(profits, axis=-1)
    profitable = np.max(profits, axis=-1) > profit_without_discount(p_small, kappa) + 1e-12
    return np.where(profitable, np.asarray(r_values)[best], np.nan)
