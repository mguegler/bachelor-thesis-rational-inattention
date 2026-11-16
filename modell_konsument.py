# ======================================================================================================================
# Dynamisches RI-Konsumentenmodell 
# ======================================================================================================================

import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import norm

from modell_parameter import validate_params

# Aktionsmenge und Spaltenreihenfolge aller Arrays
ACTIONS = ("S_A", "S_B", "L_A", "L_B")
SIGN = np.array([1.0, -1.0, 1.0, -1.0]) # begünstigendes Vorzeichen je nach Produkt
LARGE = np.array([False, False, True, True])


# Maximale Anzahl der Blahut-Arimoto-Schritte nach SLSQP und bei SLSQP-Ausfall,
# bricht bei hinreichend kleinem Fehler früher ab
POLISH_ITERATIONS = 20000


def discount_threshold(delta):
    """
    Berechnet Diskontierungsschwellenwert.
    """
    return (1.0 - delta) / 2.0


def _split_ties(values):
    """
    Stellt Gleichverteilung über numerisch gleichwertige optimale Aktionen sicher.
    """
    best = np.isclose(values, values.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12)
    return best / best.sum(axis=1, keepdims=True)


class DynamicRIModel:
    """
    Approximiert stationäre Lösung des dynamischen RI-Modells.
    """

    def __init__(self, params):
        validate_params(params)
        self.params = dict(params)
        self.lam = float(params["lambda"])
        self.delta = float(params["delta"])
        self.x = np.linspace(params["x_min"], params["x_max"], int(params["n_states"]))
        self.P = self._transition_matrix()

    # -------------------------------------------------------------------------------------------------------------------
    # Zustandsprozess
    # -------------------------------------------------------------------------------------------------------------------

    def _transition_matrix(self):
        """
        Diskretisierter AR(1)-Prozess.
        """
        rho = float(self.params["rho"])
        sigma = np.sqrt((1.0 - rho**2) * float(self.params["sigma_x2"]))
        edges = np.concatenate(([-np.inf], (self.x[1:] + self.x[:-1]) / 2, [np.inf])) # Randpunkte tragen Restmasse
        return np.diff(norm.cdf((edges[None, :] - rho * self.x[:, None]) / sigma), axis=1)

    # -------------------------------------------------------------------------------------------------------------------
    # Nutzen und Aktionswerte
    # -------------------------------------------------------------------------------------------------------------------

    def choice_values(self, V):
        """
        Berechnet Aktionswerte.
        """
        p_small, r = float(self.params["p_small"]), float(self.params["r"])
        price = np.where(LARGE, 2.0 * (1.0 - r) * p_small, p_small)     
        taste = SIGN * self.x[:, None] / 2.0
        u = taste - price                                               
        V_locked = taste + self.delta * V[:, None]                      
        continuation = np.where(LARGE, self.P @ V_locked, V[:, None])
        return u + self.delta * continuation                            

    # -------------------------------------------------------------------------------------------------------------------
    # Wahl unter Rational Inattention
    # -------------------------------------------------------------------------------------------------------------------

    def _log_partition(self, Z, q):
        """
        Berechnet L = Q/lambda + ln q und ln sum_a q(a) exp(Q(a,x)/lambda).
        """
        with np.errstate(divide="ignore"):
            L = Z + np.log(q)
        return L, logsumexp(L, axis=1)


    def _fixed_point(self, E, prior, q, max_iterations):
        """
        Fixpunktiteration q(a) = sum_x pi(x) p(a|x).

        Blahut-Arimoto; dient als Feinschliff nach SLSQP
        und als Fallback, falls SLSQP an einer Randlösung scheitert.
        """
        tiny = np.finfo(float).tiny
        for _ in range(max_iterations):
            q_new = q * (E.T @ (prior / np.maximum(E @ q, tiny)))
            if np.max(np.abs(q_new - q)) < 1e-13:
                return q_new
            q = q_new
        return q


    def _solve_q(self, E, prior, q0, polish=0):
        """
        SLSQP-Solver für q(a|x^-_h).

        Maximiert bei festem Q: lambda * sum_x pi(x|x^-_h) * ln sum_a q(a) exp(Q(a,x)/lambda).
        """
        tiny = np.finfo(float).tiny
        objective = lambda q: -self.lam * prior @ np.log(np.maximum(E @ q, tiny))
        gradient = lambda q: -self.lam * E.T @ (prior / np.maximum(E @ q, tiny))
        for start in (q0, np.full(4, 0.25)):
            result = minimize(
                objective, start, jac=gradient, method="SLSQP", bounds=[(0.0, 1.0)] * 4,
                constraints={"type": "eq", "fun": lambda q: q.sum() - 1.0},
                options={"ftol": 1e-12, "maxiter": 1000},
            )
            if result.success:
                q = np.clip(result.x, 0.0, None)
                break
        else:
            q, polish = q0, POLISH_ITERATIONS # Fallback auf Blahut-Arimoto
        q = self._fixed_point(E, prior, q, polish) 
        q = np.where(q < 1e-14, 0.0, q)
        return q / q.sum()


    def _bellman_step(self, V, q_init=None, polish=0):
        """
        Aktualisiert V(x^-), q(a|x^-) und p(a|x,x^-) (einzelner Bellman-Schritt).
        """
        Q = self.choice_values(V)
        n = len(self.x)

        if self.lam == 0.0: # Spezialfall kostenlose Informationsverarbeitung
            p_x = np.eye(4)[np.argmax(Q, axis=1)]
            return self.P @ Q.max(axis=1), self.P @ p_x, np.repeat(p_x[None], n, axis=0)

        if np.isinf(self.lam): # Spezialfall ohne zusätzliche Informationsverarbeitung
            expected_Q = self.P @ Q
            q = _split_ties(expected_Q)
            return expected_Q.max(axis=1), q, np.repeat(q[:, None, :], n, axis=1)

        Z = Q / self.lam
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        V_new, q, p = np.empty(n), np.empty((n, 4)), np.empty((n, n, 4))
        for h in range(n):
            q0 = np.full(4, 0.25) if q_init is None else np.maximum(q_init[h], 1e-10)
            q[h] = self._solve_q(E, self.P[h], q0 / q0.sum(), polish)
            L, log_sum = self._log_partition(Z, q[h])
            p[h] = np.exp(L - log_sum[:, None])                
            V_new[h] = self.lam * self.P[h] @ log_sum           
        return V_new, q, p


    def _mutual_information(self, p, q):
        """
        Berechnet die zusätzliche Informationsverarbeitung je Kontext.
        """
        q = q[:, None, :]
        mask = (p > 1e-15) & (q > 1e-15)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(mask, p * np.log(np.where(mask, p / q, 1.0)), 0.0)
        return np.einsum("hx,hxa->h", self.P, terms)

    # -------------------------------------------------------------------------------------------------------------------
    # Stationäre Verteilung an freien Entscheidungsknoten
    # -------------------------------------------------------------------------------------------------------------------

    def _context_distribution(self, p):
        """
        Berechnet stationäre Verteilung der Vorperiodenzustände an freien Entscheidungsknoten.
        """
        p_small = p[:, :, ~LARGE].sum(axis=2)
        p_large = p[:, :, LARGE].sum(axis=2)
        K = self.P * p_small + (self.P * p_large) @ self.P
        A = K.T - np.eye(len(K))
        A[-1] = 1.0
        b = np.zeros(len(K))
        b[-1] = 1.0
        return np.linalg.solve(A, b), K

    # -------------------------------------------------------------------------------------------------------------------
    # Stationäre Lösung
    # -------------------------------------------------------------------------------------------------------------------

    def solve_dynamic_model(self, tolerance=1e-7, max_iterations=10000, initial_results=None, ri_warning_tolerance=1e-5):
        """
        Iterativer Lösungsprozess bis zum Bellman-Fixpunkt (ggf. mit Warmstart).
        """
        V = np.zeros(len(self.x)) if initial_results is None else initial_results["V"].copy()
        q = None if initial_results is None else initial_results["q"]

        for iteration in range(1, max_iterations + 1):
            V_new, q, _ = self._bellman_step(V, q)
            error = np.max(np.abs(V_new - V))
            V = V_new
            if error < tolerance:
                break
        else:
            raise RuntimeError(f"Bellman-Iteration nicht konvergiert (Fehler {error:.2e}).")

        V_check, q, p = self._bellman_step(V, q, polish=POLISH_ITERATIONS)
        mu, K = self._context_distribution(p)
        ri_error = float(np.max(np.abs(np.einsum("hx,hxa->ha", self.P, p) - q)))
        if ri_error > ri_warning_tolerance:
            warnings.warn(f"RI-Fixpunktfehler {ri_error:.2e} über {ri_warning_tolerance:.2e}.",
                          RuntimeWarning, stacklevel=2)
        return {
            "params": self.params,
            "x_grid": self.x,
            "V": V,
            "q": q,                                             
            "p": p,                                             
            "mu": mu,
            "q_large": float(mu @ q[:, LARGE].sum(axis=1)),
            "information": float(mu @ self._mutual_information(p, q)),
            "iterations": iteration,
            "bellman_error": float(np.max(np.abs(V_check - V))),
            "ri_fixed_point_error": ri_error,
            "context_stationarity_error": float(np.max(np.abs(mu @ K - mu)))}