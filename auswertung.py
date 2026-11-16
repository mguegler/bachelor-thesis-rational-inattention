# ======================================================================================================
# Auswertung und Abbildungen
# ======================================================================================================

import csv
import hashlib
import inspect
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import modell_konsument
from modell_anbieter import (
    optimal_discount,
    profit,
    profit_without_discount,
    profitability_threshold,
)
from modell_konsument import ACTIONS, DynamicRIModel, discount_threshold
from modell_parameter import (
    LAMBDA_GITTER_INFORMATION,
    LAMBDA_WERTE,
    R_GITTER,
    RHO_FIXIERT,
    RHO_GITTER,
    RHO_GITTER_INFORMATION,
    RHO_WERTE,
)

CACHE_DIR = Path("resultate")

# ------------------------------------------------------------------------------------------------------
# Darstellung
# ------------------------------------------------------------------------------------------------------

# Farbe und Dicke der Kurven
ACTION_COLORS = ("#60A5FA", "#4ADE80", "#1E40AF", "#166534")
ACTION_LABELS = (r"$S_A$", r"$S_B$", r"$L_A$", r"$L_B$")
RHO_COLORS = ("#807dba", "#7160ab", "#5c3696", "#3f007d")
LAMBDA_COLORS = ("#f6b93b", "#ec6a0c", "#c0360c", "#7f0d14")
NEUTRAL_COLOR = "0.15"                              
REFERENCE_COLOR = "0.45"                             
LINEWIDTH = 1.5

# Abbildungsgrössen in Zoll
WIDE, TALL = (6.3, 2.8), (6.3, 3.0)                  

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "pdf.fonttype": 42})


def _reference_line(ax, value, label, vertical=True):
    """
    Gestrichelte Referenzlinie inkl. Beschriftung.
    """
    if vertical:
        ax.axvline(value, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0)
        ax.annotate(label, xy=(value, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -3), textcoords="offset points", va="top", color=REFERENCE_COLOR)
    else:
        ax.axhline(value, color=REFERENCE_COLOR, linestyle="--", linewidth=1.0)
        ax.annotate(label, xy=(1, value), xycoords=("axes fraction", "data"),
                    xytext=(-3, 3), textcoords="offset points", ha="right", color=REFERENCE_COLOR)


def _lambda_label(lam):
    """
    Legendenbeschriftung für Informationskosten.
    """
    if np.isinf(lam):
        return r"$\lambda\to\infty$"
    return rf"$\lambda={lam:g}$"


# ------------------------------------------------------------------------------------------------------
# Berechnung und Speicherung stationärer Grosspackungsanteil und zusätzliche Informationsverarbeitung
# ------------------------------------------------------------------------------------------------------

def _cache_dir(name, params, **grids):
    """
    Gibt Ordnerpfad zurück.
    """
    fixed = {k: float(v) for k, v in params.items() if k != "kappa"}
    key = {
        "model": inspect.getsource(modell_konsument),
        "params": fixed,
        "grids": {k: [float(v) for v in values] for k, values in grids.items()}}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{name}_{digest}"


def _demand_files(params, r_values, lam, rho):
    """
    Ergebnis- und Diagnosedatei einer (lambda, rho)-Zeile des Nachfrage-Sweeps.
    """
    fixed = {k: v for k, v in params.items() if k not in ("r", "rho", "lambda")}
    folder = _cache_dir("nachfrage", fixed, r=r_values)
    name = f"lambda={float(lam)!r}_rho={float(rho)!r}"
    return folder / f"{name}.npy", folder / f"{name}_diagnostik.npz"


def _demand_row(params, lam, rho, r_values):
    """
    Berechnet stationären Grosspackungsanteil für einzelnes (lambda, rho)-Paar.
    """
    r_delta = discount_threshold(params["delta"])
    q_large = np.zeros(len(r_values))            
    errors = {key: np.full(len(r_values), np.nan) for key in
              ("bellman_error", "ri_fixed_point_error", "context_stationarity_error")}
    previous = None
    for j, r in enumerate(r_values):
        if r < r_delta - 1e-12: # Kleinpackung dominiert --> q_L = 0
            continue
        run = dict(params, **{"lambda": lam, "rho": rho, "r": r})
        previous = DynamicRIModel(run).solve_dynamic_model(initial_results=previous)
        q_large[j] = previous["q_large"]
        for key in errors:
            errors[key][j] = previous[key]
    return q_large, errors


def compute_demand(params, r_values=R_GITTER, rho_values=RHO_GITTER, lambda_values=LAMBDA_WERTE):
    """
    Berechnet und speichert ganzes Raster stationärer Grosspackungsanteile [lambda, rho, r].
    """
    r_values, rho_values, lambda_values = map(np.asarray, (r_values, rho_values, lambda_values))
    for lam in lambda_values:
        for rho in rho_values:
            result_file, diag_file = _demand_files(params, r_values, lam, rho)
            if not result_file.exists():
                result_file.parent.mkdir(parents=True, exist_ok=True)
                q_large, errors = _demand_row(dict(params), float(lam), float(rho), r_values)
                np.savez(diag_file, **errors)
                np.save(result_file, q_large)

    q = np.array([[np.load(_demand_files(params, r_values, lam, rho)[0]) for rho in rho_values]
                  for lam in lambda_values])
    return {"r": r_values, "rho": rho_values, "lambda": lambda_values,
            "q_large": q, "params": dict(params)}


def compute_information(params, lambda_values=LAMBDA_GITTER_INFORMATION, rho_values=RHO_GITTER_INFORMATION):
    """
    Berechnet und speichert stationäre zusätzliche Informationsverarbeitung über lambda bzw. rho.
    """
    file = _cache_dir("information", params, lam=lambda_values, rho=rho_values).with_suffix(".npz")
    if file.exists():
        return dict(np.load(file))

    solve = lambda **changes: DynamicRIModel(dict(params, **changes)).solve_dynamic_model()["information"]
    data = {
        "lambda": np.asarray(lambda_values, dtype=float),
        "information_lambda": np.array([solve(**{"lambda": float(v)}) for v in lambda_values]),
        "rho": np.asarray(rho_values, dtype=float),
        "information_rho": np.array([solve(rho=float(v)) for v in rho_values]),
    }
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file, **data)
    return data


# ------------------------------------------------------------------------------------------------------
# Abgeleitete Grössen
# ------------------------------------------------------------------------------------------------------

def _check_demand_params(demand, params):
    """
    Verhindert die Verwendung von Grosspackungsanteil für falsche Modellparameter.
    """
    saved = demand.get("params")
    if saved is None or any(
        key not in saved or float(saved[key]) != float(value)
        for key, value in params.items() if key not in ("r", "rho", "lambda", "kappa")):
        raise ValueError("Grosspackungsanteil passt nicht zu den angegebenen Modellparametern.")


def _index(values, value):
    return int(np.flatnonzero(np.isclose(values, value))[0])


def _curve(demand, lam, rho):
    """
    Zieht q_L(r) für einzelnes (lambda, rho) aus der Nachfragematrix.
    """
    return demand["q_large"][_index(demand["lambda"], lam), _index(demand["rho"], rho)]


def _first_crossing(r, q, target, jump):
    """
    Berechnet kleinstes r, bei dem q_L(r) >= target.
    """
    above = np.flatnonzero(q >= target)
    if len(above) == 0:
        return np.nan
    j = above[0]
    if j == 0 or jump:
        return r[j]
    return r[j - 1] + (target - q[j - 1]) / (q[j] - q[j - 1]) * (r[j] - r[j - 1])


def binding_premium(demand, params, target=0.5):
    """
    Berechnet Bindungsprämie je (lambda, rho).
    """
    _check_demand_params(demand, params)
    r50 = np.array([[_first_crossing(demand["r"], _curve(demand, lam, rho), target,
                                     rho == 0 or lam == 0 or np.isinf(lam))
                     for rho in demand["rho"]] for lam in demand["lambda"]])
    return r50 - discount_threshold(params["delta"])


def profit_change(demand, params):
    """
    Berechnet prozentuale Gewinnveränderung gegenüber Szenario r = 0.
    """
    _check_demand_params(demand, params)
    G0 = profit_without_discount(params["p_small"], params["kappa"])
    if G0 == 0:
        raise ValueError("Prozentuale Gewinnveränderung bei G(0) = 0 nicht definiert.")
    G = profit(demand["q_large"], demand["r"], params["p_small"], params["kappa"])
    return 100.0 * (G - G0) / G0


def optimal_discounts(demand, params):
    """
    Berechnet gewinnmaximierenden Rabatt r*.

    Ohne profitablen Rabatt gilt r* = NaN.
    """
    _check_demand_params(demand, params)
    G = profit(demand["q_large"], demand["r"], params["p_small"], params["kappa"])
    return optimal_discount(demand["r"], G, params["p_small"], params["kappa"])


# ------------------------------------------------------------------------------------------------------
# Abbildungen
# ------------------------------------------------------------------------------------------------------

def _choice_panels(axes, result, prefix=""):
    """
    Zeichnet p(a|x,x^-) für x^- = 0 und x^- ≈ sigma_x in zwei Panels.
    """
    x = result["x_grid"]
    sigma_x = np.sqrt(result["params"]["sigma_x2"])
    for ax, x_minus in zip(axes, (0.0, sigma_x)):
        h = int(np.argmin(np.abs(x - x_minus)))
        context = r"$x^-=0$" if x_minus == 0 else rf"$x^-={x[h]:.2f}\approx\sigma_x$"
        for a in range(len(ACTIONS)):
            ax.plot(x, result["p"][h, :, a], color=ACTION_COLORS[a],
                    label=ACTION_LABELS[a], linewidth=LINEWIDTH)
        ax.axvline(0.0, color=REFERENCE_COLOR, linestyle="--", linewidth=0.8)
        ax.set(title=prefix + context, xlim=(-1.5, 1.5), ylim=(-0.02, 1.02), xticks=[-1, 0, 1])
    axes[0].set_ylabel(r"$p(a\mid x,x^-)$")


def _choice_figure(rows, titles):
    """
    Erstellt Abbildung für p(a|x,x^-) mit zwei Panels je Modelllösung und gemeinsamer Legende.
    """
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.3, 2.3 * len(rows) + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    for row_axes, result, title in zip(axes, rows, titles):
        _choice_panels(row_axes, result, title)
    for ax in axes[-1]:
        ax.set_xlabel(r"Aktueller Qualitätszustand $x$")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.25 / fig.get_figheight()))
    return fig


def fig1(result):
    """
    Erstellt Abb. 1: p(a|x,x^-) für (a) x^- = 0 und (b) x^- = sigma_x.
    """
    fig = _choice_figure([result], [""])
    for ax, label in zip(fig.axes, ("(a)", "(b)")):
        context = ax.get_title()                 
        ax.set_title(label)                      
        ax.set_title(context, loc="right")       
    return fig


def fig_limits(results):
    """
    Erstellt Abb. B1 und B2: p(a|x,x^-) für ausgewählte Grenzfälle.
    """
    labels, solutions = zip(*results)
    return _choice_figure(solutions, [f"{label}, " for label in labels])


def fig2(information):
    """
    Erstellt Abb. 2: I_bar über (a) lambda und (b) rho.
    """
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for ax, key, title, xlabel in (
        (axes[0], "lambda", "(a)", r"Informationskosten $\lambda$"),
        (axes[1], "rho", "(b)", r"Persistenz $\rho$")):
        ax.plot(information[key], information[f"information_{key}"],
                color=NEUTRAL_COLOR, linewidth=LINEWIDTH)
        ax.set(title=title, xlabel=xlabel)
    axes[0].set_ylabel(r"Informationsverarbeitung $\bar I$ (nats)")
    fig.tight_layout()
    return fig


def _discount_panels(demand, params, values, ylabel, reference, reference_label, legend_loc):
    """
    Zeichnet zwei Panels über Rabatt r, links mit variierter Persistenz,
    rechts mit variierten Informationskosten.
    """
    lam_fixed = params["lambda"]
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharex=True, sharey=True)
    for rho, color in zip(RHO_WERTE, RHO_COLORS):
        axes[0].plot(demand["r"], values(lam_fixed, rho), color=color,
                linewidth=LINEWIDTH, label=rf"$\rho={rho:g}$")
    for lam, color in zip(LAMBDA_WERTE, LAMBDA_COLORS):
        axes[1].plot(demand["r"], values(lam, RHO_FIXIERT), color=color,
                linewidth=LINEWIDTH, label=_lambda_label(lam))
    axes[0].set_title("(a)")
    axes[0].set_title(rf"$\lambda={lam_fixed:g}$", loc="right")
    axes[1].set_title("(b)")
    axes[1].set_title(rf"$\rho={RHO_FIXIERT:g}$", loc="right")
    for ax in axes:
        _reference_line(ax, reference, reference_label)
        ax.set_xlabel(r"Mengenrabatt $r$")
    axes[0].legend(loc=legend_loc)
    axes[1].legend(loc=legend_loc)
    axes[0].set_ylabel(ylabel)
    return fig, axes


def fig3(demand, params, legend_loc="best"):
    """
    Erstellt Abb. 3: Grosspackungsanteil über r.
    """
    _check_demand_params(demand, params)
    fig, axes = _discount_panels(
        demand, params, lambda lam, rho: _curve(demand, lam, rho),
        r"Stationärer Grosspackungsanteil $\overline{q_L}$",
        discount_threshold(params["delta"]), r"$r_\delta$", legend_loc)
    axes[0].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return fig


def _lambda_curves_over_rho(ax, demand, values):
    """
    Erstellt je lambda eine Kurve über rho.
    """
    colors = LAMBDA_COLORS
    for lam, row, color in zip(demand["lambda"], values, colors):
        ax.plot(demand["rho"], row, color=color,
                linewidth=LINEWIDTH, label=_lambda_label(lam))
    ax.set_xlabel(r"Persistenz $\rho$")


def fig4(demand, params):
    """
    Erstellt Abb. 4: Bindungsprämie über rho.
    """
    fig, ax = plt.subplots(figsize=TALL)
    _lambda_curves_over_rho(ax, demand, binding_premium(demand, params))
    ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
    ax.set_ylabel(r"Bindungsprämie $b=r_{50}-r_\delta$")
    ax.legend()
    fig.tight_layout()
    return fig


def fig5(demand, params, legend_loc="best"):
    """
    Erstellt Abb. 5: %ΔG über r.
    """
    change = profit_change(demand, params)
    values = lambda lam, rho: change[_index(demand["lambda"], lam), _index(demand["rho"], rho)]
    fig, axes = _discount_panels(
        demand, params, values, r"Gewinnveränderung $\%\Delta G(r)$",
        profitability_threshold(params["p_small"], params["kappa"]), r"$\kappa/(2p_S)$",
        legend_loc)
    for ax in axes:
        ax.axhline(0.0, color=REFERENCE_COLOR, linewidth=0.8)
    fig.tight_layout()
    return fig


def fig6(demand, params, legend_loc="upper right"):
    """
    Erstellt Abb. 6: r* über rho.
    """
    lower = discount_threshold(params["delta"])
    upper = profitability_threshold(params["p_small"], params["kappa"])
    fig, ax = plt.subplots(figsize=TALL)
    _lambda_curves_over_rho(ax, demand, optimal_discounts(demand, params))
    _reference_line(ax, upper, r"$\kappa/(2p_S)$", vertical=False)
    _reference_line(ax, lower, r"$r_\delta$", vertical=False)
    margin = 0.15 * (upper - lower)
    ax.set_ylim(lower - margin, upper + margin)
    ax.set_ylabel(r"Gewinnmaximierender Mengenrabatt $r^*$")
    # Legende im Bereich unterhalb der Linie kappa/(2p_S) platzieren.
    ax.legend(loc=legend_loc, bbox_to_anchor=(0.0, 0.0, 1.0, 0.85))
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------------
# Robustheits- und Plausibilitätsprüfungen
# ------------------------------------------------------------------------------------------------------

def _save_robustness(name, rows, folder):
    """
    Schreibt und speichert Tabelle als CSV-Tabelle.
    
    Legt bei Bedarf Ordner an. Keine neue Datei bei leeren Zeilen.
    """
    if not rows:
        return
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / name).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sweep_convergence_report(demand, params, output_dir=CACHE_DIR):
    """
    Speichert maximale Solver-Residuen aller numerisch gelösten Punkte des Nachfrage-Sweeps.
    """
    _check_demand_params(demand, params)
    rows = []
    for lam in demand["lambda"]:
        for rho in demand["rho"]:
            file = _demand_files(params, demand["r"], lam, rho)[1]
            if not file.exists():
                raise FileNotFoundError(
                    f"Solver-Diagnosen fehlen für lambda={lam:g}, rho={rho:g}. "
                    "Der alte Nachfrage-Cache enthält nur q_L. Residuen sind "
                    "ohne erneutes Lösen nicht rekonstruierbar.")
            with np.load(file) as data:
                values = {k: np.asarray(data[k]) for k in
                          ("bellman_error", "ri_fixed_point_error", "context_stationarity_error")}
            valid = np.isfinite(values["bellman_error"])
            if not np.any(valid):
                continue
            rows.append({
                "lambda": float(lam), "rho": float(rho), "n_geloest": int(valid.sum()),
                **{f"max_{k}": float(np.max(v[valid])) for k, v in values.items()}})
    if rows:
        rows.append({
            "lambda": "alle", "rho": "alle", "n_geloest": sum(row["n_geloest"] for row in rows),
            **{key: max(row[key] for row in rows) for key in rows[0] if key.startswith("max_")}})
    _save_robustness("robustheit_sweep_konvergenz.csv", rows, output_dir)
    return rows


def _solve_robustness(params, changes=None, tolerance=1e-7):
    """
    Löst Modell mit geänderten Parametern oder abweichender Toleranz für Robustheitsprüfung.
    """
    run = dict(params)
    if changes:
        run.update(changes)
    return DynamicRIModel(run).solve_dynamic_model(tolerance=tolerance)


def _check_numerics(params, *, state_configs=None, tolerances=(1e-9,),
                    convergence_cases=((0.95, 0.08, 0.04),), base=None,
                    output_dir=CACHE_DIR):
    """
    Prüft Stabilität der Lösung bei feinerem Gitter, grösserem Zustandsbereich und strengerer Toleranz.
    """
    p = dict(params)
    base_tol = 1e-7
    base = base if base is not None else _solve_robustness(p)
    if state_configs is None:
        n, xmin, xmax = int(p["n_states"]), p["x_min"], p["x_max"]
        step = (xmax - xmin) / (n - 1)

        extended_min, extended_max = -3.5, 3.5
        extended_n = int(round((extended_max - extended_min) / step)) + 1 
        state_configs = ((2 * n - 1, xmin, xmax),
                         (extended_n, extended_min, extended_max))

    configs = [("Baseline", {}, base_tol, base)]
    for n, xmin, xmax in state_configs:
        changes = {"n_states": int(n), "x_min": float(xmin), "x_max": float(xmax)}
        if any(changes[k] != p[k] for k in changes):
            configs.append(("Zustandsgitter/-bereich", changes, base_tol,
                            _solve_robustness(p, changes)))
    for tol in tolerances:
        if float(tol) != base_tol:
            configs.append(("Bellman-Toleranz", {}, float(tol),
                            _solve_robustness(p, tolerance=float(tol))))

    rows = []
    for label, changes, tol, sol in configs:
        settings = dict(p, **changes)
        rows.append({
            "Pruefung": label,
            "n_states": settings["n_states"], "x_min": settings["x_min"],
            "x_max": settings["x_max"], "tolerance": tol,
            "q_large": sol["q_large"], "information": sol["information"],
            "delta_q_large": sol["q_large"] - base["q_large"],
            "delta_information": sol["information"] - base["information"],
            "iterations": sol["iterations"], "bellman_error": sol["bellman_error"],
            "ri_fixed_point_error": sol["ri_fixed_point_error"],
            "context_stationarity_error": sol["context_stationarity_error"]})

    samples = []
    for rho, lam, r in convergence_cases:
        changes = {"rho": float(rho), "lambda": float(lam), "r": float(r)}
        reference = _solve_robustness(p, changes)
        for tol in tolerances:
            sol = _solve_robustness(p, changes, tolerance=float(tol))
            samples.append({
                "rho": rho, "lambda": lam, "r": r, "tolerance": tol,
                "delta_q_large": sol["q_large"] - reference["q_large"],
                "delta_information": sol["information"] - reference["information"],
                "bellman_error": sol["bellman_error"],
                "ri_fixed_point_error": sol["ri_fixed_point_error"],
                "context_stationarity_error": sol["context_stationarity_error"]})
    folder = Path(output_dir)
    _save_robustness("robustheit_numerik.csv", rows, folder)
    _save_robustness("robustheit_konvergenz_stichproben.csv", samples, folder)
    return {"numerik": rows, "konvergenz_stichproben": samples}


def _check_limit(params, *, limit_lambdas=(10.0, 100.0, np.inf),
                 limit_cases=((0.4, 0.04),), output_dir=CACHE_DIR):
    """
    Vergleicht grosse endliche Informationskosten gegen exaktes lambda-->inf.
    """
    rows = []
    for rho, r in limit_cases:
        settings = {"rho": float(rho), "r": float(r)}
        reference = _solve_robustness(params, dict(settings, **{"lambda": np.inf}))
        for lam in limit_lambdas:
            sol = (reference if np.isinf(lam) else _solve_robustness(
                params, dict(settings, **{"lambda": float(lam)})))
            rows.append({
                "rho": rho, "r": r, "lambda": lam,
                "q_large": sol["q_large"], "information": sol["information"],
                "delta_q_large_gegen_inf": sol["q_large"] - reference["q_large"],
                "delta_information_gegen_inf": sol["information"] - reference["information"],
                "bellman_error": sol["bellman_error"],
                "ri_fixed_point_error": sol["ri_fixed_point_error"]})
    folder = Path(output_dir)
    _save_robustness("robustheit_grenzfall.csv", rows, folder)
    return rows


def _check_analytics(params, *, base=None, demand=None, output_dir=CACHE_DIR):
    """
    Prüft Symmetrie, Diskontierungsschwellenwert und Profitabilitätsschwelle.
    """
    base = base if base is not None else _solve_robustness(params)
    rows = []
    if params["p_small"] > 0:
        r_delta = discount_threshold(params["delta"])
        for r in (0.0, r_delta / 2):
            sol = _solve_robustness(params, {"r": float(r)})
            rows.append({"Pruefung": "Dominanz r<r_delta", "r": r,
                         "Fehler": abs(sol["q_large"]), "Punkte_geprueft": 1,
                         "Punkte_ausgelassen": 0,
                         "Bemerkung": "Direktlösung ohne Nachfrage-Abkürzung"})
    x = base["x_grid"]
    if np.allclose(x, -x[::-1], rtol=0, atol=1e-12):
        swapped = [1, 0, 3, 2]
        mirror = base["p"][::-1, ::-1][:, :, swapped]
        prior = DynamicRIModel(base["params"]).P
        weighted = np.einsum("hx,hxa->h", prior, np.abs(base["p"] - mirror))
        q_mirror = base["q"][::-1][:, swapped]
        rows.extend([
            {"Pruefung": "Symmetrie Wahlregel (prior-gewichtet)", "r": params["r"],
             "Fehler": float(np.max(weighted)), "Punkte_geprueft": len(x),
             "Punkte_ausgelassen": 0, "Bemerkung": "Maximale gewichtete L1-Abweichung je Kontext"},
            {"Pruefung": "Symmetrie Ex-ante-Wahl (q)", "r": params["r"],
             "Fehler": float(np.max(np.abs(base["q"] - q_mirror))),
             "Punkte_geprueft": int(base["q"].size),
             "Punkte_ausgelassen": 0, "Bemerkung": "Maximale absolute Abweichung"}])

    if demand is not None and params["p_small"] > 0:
        _check_demand_params(demand, params)
        q = np.asarray(demand["q_large"], dtype=float)
        r = np.asarray(demand["r"], dtype=float)[None, None, :]
        G0 = profit_without_discount(params["p_small"], params["kappa"])
        difference = profit(q, r, params["p_small"], params["kappa"]) - G0
        threshold = profitability_threshold(params["p_small"], params["kappa"])

        q_tol, gain_tol = 1e-12, 1e-12 * max(1.0, abs(G0))
        clear = ((q > q_tol) &
                 (q * np.abs(params["kappa"] - 2 * r * params["p_small"])
                  / (1 + q) > gain_tol))
        expected_positive = r < threshold
        violations = (np.count_nonzero(clear & ((difference > gain_tol) != expected_positive))
                      + np.count_nonzero((q <= q_tol) & (np.abs(difference) > gain_tol)))
        tested = int(np.count_nonzero(clear) + np.count_nonzero(q <= q_tol))
        rows.append({
            "Pruefung": "Profitabilitaetsschwelle (Nachfrage-Sweep)", "r": "alle",
            "Fehler": int(violations), "Punkte_geprueft": tested,
            "Punkte_ausgelassen": int(q.size - tested),
            "Bemerkung": "Anzahl Vorzeichenverletzungen; unklare Werte toleranzbedingt ausgelassen"})
    folder = Path(output_dir)
    _save_robustness("robustheit_analytisch.csv", rows, folder)
    return rows

def economic_robustness(demand, params, *, kappa_values=(0.1, 0.2, 0.3),
                        demand_thresholds=(0.25, 0.5, 0.75),
                        sigma_x2_values=(), delta_values=(),
                        sensitivity_r_values=None, sensitivity_rho_values=None,
                        output_dir=CACHE_DIR):
    """
    Prüft Ergebnisse bei alternativen Grosspackungsschwellen und packungsbezogenen Kosten.
    """
    _check_demand_params(demand, params)
    variants = [("Baseline", dict(params), demand)]
    if sigma_x2_values or delta_values:
        r_values = (np.round(np.arange(0.0, 0.1501, 0.0025), 4)
                    if sensitivity_r_values is None else sensitivity_r_values)
        rho_values = RHO_WERTE if sensitivity_rho_values is None else sensitivity_rho_values
        lambda_values = demand["lambda"]
        variants = [("Baseline", dict(params), compute_demand(
            params, r_values, rho_values, lambda_values))]
        for value in sigma_x2_values:
            p = dict(params, sigma_x2=float(value))
            variants.append(("sigma_x2", p, compute_demand(
                p, r_values, rho_values, lambda_values)))
        for value in delta_values:
            p = dict(params, delta=float(value))
            variants.append(("delta", p, compute_demand(
                p, r_values, rho_values, lambda_values)))
    threshold_rows, kappa_rows = [], []
    for label, p, d in variants:
        for target in demand_thresholds:
            b = binding_premium(d, p, target=float(target))
            for i, lam in enumerate(d["lambda"]):
                for j, rho in enumerate(d["rho"]):
                    threshold_rows.append({
                        "Pruefung": label, "sigma_x2": p["sigma_x2"],
                        "delta": p["delta"], "Schwelle": target,
                        "lambda": float(lam), "rho": float(rho), "b": float(b[i, j])})
        for kappa in kappa_values:
            pp = dict(p, kappa=float(kappa))
            rstars = optimal_discounts(d, pp)
            G = profit(d["q_large"], d["r"], pp["p_small"], pp["kappa"])
            G0 = profit_without_discount(pp["p_small"], pp["kappa"])
            percent = (np.full(G.shape[:-1], np.nan) if G0 == 0
                       else 100.0 * (np.max(G, axis=-1) - G0) / G0)
            for i, lam in enumerate(d["lambda"]):
                for j, rho in enumerate(d["rho"]):
                    kappa_rows.append({
                        "Pruefung": label, "sigma_x2": p["sigma_x2"],
                        "delta": p["delta"], "kappa": float(kappa),
                        "lambda": float(lam), "rho": float(rho),
                        "r_star": float(rstars[i, j]),
                        "max_Gewinnveraenderung_prozent": float(percent[i, j])})
    folder = Path(output_dir)
    _save_robustness("robustheit_schwelle.csv", threshold_rows, folder)
    _save_robustness("robustheit_kappa.csv", kappa_rows, folder)
    return {"schwelle": threshold_rows, "kappa": kappa_rows}


def _check_discount_grid(params, *, discount_grids=None,
                         discount_cases=((0.4, 0.08), (0.8, 0.08), (0.4, np.inf)),
                         output_dir=CACHE_DIR):
    """
    Prüft, ob Bindungsprämie und gewinnmaximierender Rabatt vom gewählten Rabattraster abhängen.
    """
    rows = []
    if discount_grids is not None:
        for rho, lam in discount_cases:
            reference_b = reference_rstar = np.nan
            for i, grid in enumerate(discount_grids):
                grid = np.asarray(grid, dtype=float)
                if (grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid))
                        or np.any((grid < 0) | (grid >= 1)) or np.any(np.diff(grid) <= 0)):
                    raise ValueError("Rabattraster muss streng aufsteigend und in [0,1) liegen.")
                d = compute_demand(params, grid, (rho,), (lam,))
                b = float(binding_premium(d, params)[0, 0])
                rstar = float(optimal_discounts(d, params)[0, 0])
                if i == 0:
                    reference_b, reference_rstar = b, rstar
                rows.append({
                    "rho": rho, "lambda": lam, "n_r": len(grid),
                    "min_delta_r": float(np.min(np.diff(grid))),
                    "b": b, "r_star": rstar,
                    "delta_b": b - reference_b,
                    "delta_r_star": rstar - reference_rstar})
    folder = Path(output_dir)
    _save_robustness("robustheit_rabatt.csv", rows, folder)
    return rows


def robustness_checks(
    params, *, state_configs=None, tolerances=(1e-9,),
    convergence_cases=((0.95, 0.08, 0.04),),
    discount_grids=None,
    discount_cases=((0.4, 0.08), (0.8, 0.08), (0.4, np.inf)),
    limit_lambdas=(10.0, 100.0, np.inf), limit_cases=((0.4, 0.04),),
    demand=None, kappa_values=(0.1, 0.2, 0.3),
    demand_thresholds=(0.25, 0.5, 0.75),
    sigma_x2_values=(), delta_values=(),
    sensitivity_r_values=None, sensitivity_rho_values=None,
    output_dir=CACHE_DIR,
):
    """
    Führt alle Robustheitsprüfungen für Anhang A aus und speichert Ergebnisse als CSV-Tabellen.
    """
    folder = Path(output_dir)
    if demand is not None:
        sweep = sweep_convergence_report(demand, params, output_dir=folder)
    else:
        sweep = None
    base = _solve_robustness(params)
    result = _check_numerics(
        params, state_configs=state_configs, tolerances=tolerances,
        convergence_cases=convergence_cases, base=base, output_dir=folder)
    result["grenzfall"] = _check_limit(
        params, limit_lambdas=limit_lambdas, limit_cases=limit_cases, output_dir=folder)
    result["analytisch"] = _check_analytics(
        params, base=base, demand=demand, output_dir=folder)
    result["rabatt"] = _check_discount_grid(
        params, discount_grids=discount_grids, discount_cases=discount_cases,
        output_dir=folder)
    result["sweep_konvergenz"] = sweep
    result["oekonomisch"] = (economic_robustness(
        demand, params, kappa_values=kappa_values,
        demand_thresholds=demand_thresholds,
        sigma_x2_values=sigma_x2_values, delta_values=delta_values,
        sensitivity_r_values=sensitivity_r_values,
        sensitivity_rho_values=sensitivity_rho_values,
        output_dir=folder) if demand is not None else None)
    return result
