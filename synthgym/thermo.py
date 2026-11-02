"""Offline thermodynamics from the matminer `mp_nostruct_20181018` snapshot of the Materials Project.

Energies are 0 K DFT formation energies (eV/atom), 2018 corrections, no entropy. CO2 and H2O
release is therefore undercounted, so carbonate routes look uphill. Treat values as features
and as a coarse driving-force signal, never as ground truth.
"""

import gzip
import json
from functools import lru_cache

import numpy as np
from pymatgen.core import Composition
from scipy.optimize import nnls

from synthgym.data import RAW_DIR

MP_FILE = RAW_DIR / "mp_nostruct_20181018.json.gz"

VOLATILES = ["CO2", "H2O", "NH3", "NO2", "O2", "SO2"]


@lru_cache(maxsize=1)
def formation_table() -> dict[str, tuple[float, float]]:
    """Reduced formula -> (formation energy eV/atom, energy above hull eV/atom), lowest-energy polymorph."""
    with gzip.open(MP_FILE, "rt") as handle:
        raw = json.load(handle)
    formula_column = raw["columns"].index("formula")
    e_form_column = raw["columns"].index("e_form")
    e_hull_column = raw["columns"].index("e_hull")
    table: dict[str, tuple[float, float]] = {}
    for row in raw["data"]:
        formula = row[formula_column]
        if not isinstance(formula, str):
            formula = "NaN"
        try:
            reduced = Composition(formula).reduced_formula
        except Exception:
            continue
        e_form = row[e_form_column]
        e_hull = row[e_hull_column]
        if e_form is None:
            continue
        if reduced not in table or e_form < table[reduced][0]:
            table[reduced] = (float(e_form), float(e_hull))
    for element_gas in ["O2", "N2", "H2"]:
        table[element_gas] = (0.0, 0.0)
    return table


def formation_energy_per_fu(formula: str) -> float | None:
    """Formation energy of one formula unit in eV, or None if the compound is not in the snapshot."""
    entry = formation_table().get(Composition(formula).reduced_formula)
    if entry is None:
        return None
    return entry[0] * Composition(formula).num_atoms


def balance(reactants: list[str], products: list[str], byproducts: list[str] = VOLATILES) -> dict[str, float] | None:
    """Non-negative coefficients so that reactants -> products[0] (coefficient 1) + other products + byproducts.

    O2 may appear on either side. Returns None when no balance exists to within 1e-3 per element.
    """
    target = products[0]
    side_species = list(reactants) + ["O2"] + list(products[1:]) + list(byproducts)
    signs = [1.0] * (len(reactants) + 1) + [-1.0] * (len(products) - 1 + len(byproducts))
    elements = sorted({str(element) for formula in side_species + [target] for element in Composition(formula).elements})
    matrix = np.zeros((len(elements), len(side_species)))
    for column, (formula, sign) in enumerate(zip(side_species, signs)):
        composition = Composition(formula)
        for row, element in enumerate(elements):
            matrix[row, column] = sign * composition[element]
    target_composition = Composition(target)
    rhs = np.array([target_composition[element] for element in elements])
    coefficients, residual = nnls(matrix, rhs)
    if residual > 1e-3:
        return None
    result: dict[str, float] = {target: 1.0}
    for formula, sign, coefficient in zip(side_species, signs, coefficients):
        if coefficient > 1e-9:
            result[formula] = result.get(formula, 0.0) + sign * coefficient
    return result


def reaction_energy_per_atom(reactants: list[str], target: str) -> float | None:
    """Reaction energy (eV/atom of target) for reactants -> target + volatiles, or None if unknown."""
    coefficients = balance(reactants, [target])
    if coefficients is None:
        return None
    energy_change = 0.0
    for formula, coefficient in coefficients.items():
        energy = formation_energy_per_fu(formula)
        if energy is None:
            return None
        if formula == target:
            energy_change += energy
        else:
            energy_change -= coefficient * energy
    return energy_change / Composition(target).num_atoms
