"""Reaction extent law: how far a thermodynamically allowed reaction proceeds in one heat step.

    x = 1 - exp(-t_h * contact / tau)
    tau = tau0 * exp(-beta * (T_K / T_on - 1)) * exp(gamma * dG)
    T_on = alpha * min(Tm_K of the reacting phases)

tau0, beta, gamma are fitted on ARROWS3 phase fractions (scripts/fit_kinetics.py).
alpha is the Tamman-style onset ratio fitted on the Ceder training split.
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from pymatgen.core import Composition

KINETICS_FILE = Path(__file__).resolve().parent.parent / "models" / "kinetics.json"

# Melting or decomposition temperature in C, handbook values at round-number precision.
# Salts that decompose before melting use the decomposition temperature (Precursor Genome 2026 convention).
# Assumption: values are approximate; phases not listed take their anion-class default.
MELT_OR_DECOMPOSE_C = {
    "Li2CO3": 723, "Na2CO3": 851, "K2CO3": 891, "CaCO3": 825, "SrCO3": 1075, "BaCO3": 1100,
    "MgCO3": 350, "MnCO3": 400, "Y2(CO3)3": 500, "La2(CO3)3": 500, "CuCO3": 300,
    "TiO2": 1843, "ZrO2": 2715, "HfO2": 2758, "Al2O3": 2072, "SiO2": 1713, "MgO": 2852, "CaO": 2613,
    "SrO": 2531, "BaO": 1923, "BaO2": 800, "Y2O3": 2425, "La2O3": 2315, "CeO2": 2400, "Nd2O3": 2233,
    "Sm2O3": 2335, "Eu2O3": 2350, "Gd2O3": 2420, "Pr6O11": 2183, "Fe2O3": 1565, "Fe3O4": 1597,
    "FeO": 1377, "CuO": 1326, "Cu2O": 1235, "NiO": 1955, "CoO": 1933, "Co3O4": 895, "MnO2": 535,
    "Mn2O3": 888, "Mn3O4": 1567, "MnO": 1945, "Cr2O3": 2435, "V2O5": 690, "V2O3": 1940, "VO2": 1967,
    "Nb2O5": 1512, "Ta2O5": 1872, "MoO3": 795, "MoO2": 1100, "WO3": 1473, "ZnO": 1975, "SnO2": 1630,
    "Bi2O3": 817, "PbO": 888, "Sb2O3": 656, "TeO2": 732, "Ga2O3": 1900, "In2O3": 1910, "GeO2": 1115,
    "B2O3": 450, "P2O5": 340, "Li2O": 1438, "Na2O": 1132, "K2O": 740,
    "LiOH": 462, "NaOH": 318, "KOH": 406, "Mg(OH)2": 350, "Ca(OH)2": 512,
    "NH4H2PO4": 190, "(NH4)2HPO4": 155, "H3BO3": 170, "H3PO4": 213,
    "LiNO3": 255, "NaNO3": 308, "KNO3": 334, "Sr(NO3)2": 570, "Ba(NO3)2": 592, "Ca(NO3)2": 561,
    "LiF": 845, "NaF": 993, "KF": 858, "CaF2": 1418, "MgF2": 1263,
    "Li3PO4": 1205, "Li4P2O7": 880, "LiPO3": 656, "TiP2O7": 1400, "LiTi2(PO4)3": 1200, "LiTiPO5": 1100,
    "BaCuO2": 1000, "Y2Cu2O5": 1100, "BaY2CuO5": 1250, "Ba2YCu3O7": 1000, "Ba2Cu3O6": 950,
    "Na2MoO4": 687, "Na2Mo2O7": 612, "Na2TeO4": 800, "Na2Te3Mo3O16": 550,
    "Li2TiO3": 1533, "Li4Ti5O12": 1350, "BaTiO3": 1625, "SrTiO3": 2080, "CaTiO3": 1975,
}

CLASS_DEFAULT_C = {
    "carbonate": 800, "nitrate": 400, "hydroxide": 400, "ammonium": 200, "fluoride": 1100,
    "phosphate": 1100, "oxide": 1500, "other": 1000,
}

_MELT_BY_REDUCED = {Composition(formula).reduced_formula: value for formula, value in MELT_OR_DECOMPOSE_C.items()}


def anion_class(formula: str) -> str:
    elements = {str(element) for element in Composition(formula).elements}
    if "N" in elements and "H" in elements:
        return "ammonium"
    if "N" in elements and "O" in elements:
        return "nitrate"
    if "C" in elements and "O" in elements:
        return "carbonate"
    if "H" in elements and "O" in elements:
        return "hydroxide"
    if "F" in elements:
        return "fluoride"
    if "P" in elements and "O" in elements:
        return "phosphate"
    if "O" in elements:
        return "oxide"
    return "other"


def melt_k(formula: str) -> float:
    """Melting or decomposition temperature in kelvin, from the table or the anion-class default."""
    reduced = Composition(formula).reduced_formula
    celsius = _MELT_BY_REDUCED.get(reduced, CLASS_DEFAULT_C[anion_class(formula)])
    return celsius + 273.15


@dataclass
class Kinetics:
    tau0: float = 4.0
    beta: float = 6.0
    gamma: float = 8.0
    alpha: float = 0.6

    def save(self, path: Path = KINETICS_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path = KINETICS_FILE) -> "Kinetics":
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))


def extent(kinetics: Kinetics, delta_g: float, temperature_k: float, time_h: float, contact: float, reactants: list[str]) -> float:
    """Fraction of the reacting pair converted in this step, in [0, 1]."""
    onset_k = kinetics.alpha * min(melt_k(formula) for formula in reactants)
    exponent = -kinetics.beta * (temperature_k / onset_k - 1.0) + kinetics.gamma * delta_g
    exponent = max(min(exponent, 50.0), -50.0)
    tau = kinetics.tau0 * math.exp(exponent)
    return 1.0 - math.exp(-time_h * contact / tau)
