# ==========================================================================================
# Modellparameter, Parameterprüfung und Abbildungsraster
# ==========================================================================================

import math

import numpy as np


class ModelParams(dict):
    """
    Baseline-Parametrisierung und numerische Diskretisierung.
    """

    def __init__(
        self,
        p_small=1.0,     # Preis einer Kleinpackung p_S
        r=0.04,          # Mengenrabatt
        delta=0.95,      # Diskontfaktor
        lambda_=0.08,    # Informationskosten
        rho=0.80,        # Persistenz des AR(1)-Zustands
        sigma_x2=0.30,   # stationäre Varianz des Zustands
        kappa=0.20,      # packungsbezogene Kosten 
        n_states=101,    # Anzahl Rasterpunkte für x
        x_min=-2.5,
        x_max=2.5,
    ):
        super().__init__(
            p_small=p_small, r=r, delta=delta, rho=rho, sigma_x2=sigma_x2,
            kappa=kappa, n_states=n_states, x_min=x_min, x_max=x_max,
        )
        self["lambda"] = lambda_


def validate_params(params):
    """
    Prüft, ob die Parameter im festgelegten Definitionsbereich liegen.
    """
    for name in ("p_small", "r", "delta", "rho", "sigma_x2", "kappa", "x_min", "x_max"):
        if not math.isfinite(float(params[name])):
            raise ValueError(f"{name} muss endlich sein.")
    if math.isnan(float(params["lambda"])) or params["lambda"] < 0:
        raise ValueError("lambda muss nichtnegativ sein.")
    if params["p_small"] < 0 or params["kappa"] < 0:
        raise ValueError("p_small und kappa müssen nichtnegativ sein.")
    if not 0 <= params["r"] < 1:
        raise ValueError("r muss 0 <= r < 1 erfüllen.")
    if not 0 < params["delta"] < 1:
        raise ValueError("delta muss 0 < delta < 1 erfüllen.")
    if not 0 <= params["rho"] < 1:
        raise ValueError("rho muss 0 <= rho < 1 erfüllen.")
    if params["sigma_x2"] <= 0:
        raise ValueError("sigma_x2 muss positiv sein.")
    if (not math.isfinite(float(params["n_states"]))
            or int(params["n_states"]) != params["n_states"]
            or params["n_states"] < 2):
        raise ValueError("n_states muss eine ganze Zahl >= 2 sein.")
    if params["x_min"] >= params["x_max"]:
        raise ValueError("x_min muss kleiner als x_max sein.")


# ==========================================================================================
# Raster für Abbildungen
# ==========================================================================================

# Abb. 3a und 5a: Persistenzwerte
RHO_WERTE = (0.0, 0.4, 0.8, 0.95)

# Abb. 3b, 4, 5b und 6: Informationskosten
LAMBDA_WERTE = (0.0, 0.08, 0.3, np.inf)

# Abb. 3b und 5b: fixierte Persistenz
RHO_FIXIERT = 0.4

# Abb. 3–6: Rabattraster bis 15 %
R_GITTER = np.round(np.arange(0.0, 0.1501, 0.001), 3)

# Abb. 4 und 6: Persistenzraster
RHO_GITTER = np.unique(np.concatenate((np.round(np.arange(0.0, 0.901, 0.1), 2), RHO_WERTE)))

# Abb. 2: Gitterpunkte der Informationskurven
LAMBDA_GITTER_INFORMATION = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.3, 0.4)
RHO_GITTER_INFORMATION = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95)
