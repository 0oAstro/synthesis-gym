"""The simulated crucible: a hidden phase assemblage that heat steps react pairwise.

Physics: each reaction's products are the minimum-Gibbs-energy assemblage for the reacting phases,
found by a linear program over the phase library, with gases released and oxygen exchanged with the
atmosphere at a chemical potential set by its pO2. Kinetics: synthgym.kinetics.extent decides how
much of each allowed reaction happens in the step.
"""

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from pymatgen.core import Composition
from scipy.optimize import linprog

from synthgym.kinetics import Kinetics, extent
from synthgym.thermo import formation_table

BOLTZMANN_EV = 8.617e-5
GAS_ENTROPY_EV_PER_K = 2.1e-3
MAX_E_HULL = 0.10
MIN_DRIVING_FORCE = -0.02
TARGET_STABILIZATION_EV_PER_ATOM = 0.03
SUB_STEPS = 3
XRD_NOISE = 0.03
XRD_DETECTION_LIMIT = 0.03

GASES = ["CO2", "H2O", "NH3", "NO2", "N2", "SO2"]
OXYGEN_PRESSURE_ATM = {"air": 0.21, "O2": 1.0, "inert": 1e-5, "vacuum": 1e-8, "reducing": 1e-20}
ATMOSPHERES = list(OXYGEN_PRESSURE_ATM)


@dataclass(frozen=True)
class Phase:
    formula: str
    elements: frozenset[str]
    atoms: float
    energy: float          # formation energy per formula unit, eV
    weight: float          # g/mol per formula unit
    estimated: bool


def _composition_vector(formula: str) -> dict[str, float]:
    composition = Composition(formula)
    return {str(element): composition[element] for element in composition.elements}


@lru_cache(maxsize=1)
def _phase_index() -> list[Phase]:
    phases = []
    for formula, (e_form, e_hull) in formation_table().items():
        if e_hull > MAX_E_HULL:
            continue
        composition = Composition(formula)
        phases.append(Phase(
            formula=formula,
            elements=frozenset(str(element) for element in composition.elements),
            atoms=composition.num_atoms,
            energy=e_form * composition.num_atoms,
            weight=composition.weight,
            estimated=False,
        ))
    return phases


@lru_cache(maxsize=4096)
def library(elements: frozenset[str]) -> tuple[Phase, ...]:
    """All low-hull solid phases whose elements are a subset of `elements` (plus O)."""
    allowed = elements | {"O"}
    return tuple(phase for phase in _phase_index() if phase.elements <= allowed and phase.formula not in GASES + ["O2"])


def phase_for(formula: str) -> Phase:
    """The Phase record for a formula; estimates the energy as on-hull when the snapshot lacks it."""
    reduced = Composition(formula).reduced_formula
    entry = formation_table().get(reduced)
    composition = Composition(reduced)
    if entry is not None:
        return Phase(reduced, frozenset(str(e) for e in composition.elements), composition.num_atoms,
                     entry[0] * composition.num_atoms, composition.weight, False)
    elements = frozenset(str(e) for e in composition.elements)
    ground = _ground_state_energy(_composition_vector(reduced), library(elements))
    energy = ground - 0.02 * composition.num_atoms
    return Phase(reduced, elements, composition.num_atoms, energy, composition.weight, True)


def _ground_state_energy(composition: dict[str, float], phases: tuple[Phase, ...]) -> float:
    """Lowest 0 K energy of a closed composition over `phases` (no gas exchange)."""
    elements = sorted(composition)
    candidates = [phase for phase in phases if phase.elements <= set(elements)]
    if not candidates:
        return 0.0
    matrix = np.array([[_composition_vector(phase.formula).get(element, 0.0) for phase in candidates] for element in elements])
    rhs = np.array([composition[element] for element in elements])
    costs = np.array([phase.energy for phase in candidates])
    result = linprog(costs, A_eq=matrix, b_eq=rhs, bounds=(0, None), method="highs")
    if not result.success:
        return 0.0
    return float(result.fun)


def oxygen_potential(temperature_k: float, atmosphere: str) -> float:
    """Chemical potential of one O2 molecule in eV, relative to the 0 K O2 reference."""
    pressure = OXYGEN_PRESSURE_ATM[atmosphere]
    return -GAS_ENTROPY_EV_PER_K * temperature_k + BOLTZMANN_EV * temperature_k * math.log(pressure)


def gas_energy(formula: str, temperature_k: float) -> float:
    entry = formation_table().get(formula)
    base = entry[0] * Composition(formula).num_atoms if entry else 0.0
    return base - GAS_ENTROPY_EV_PER_K * temperature_k


@dataclass
class Reaction:
    reactants: dict[str, float]
    products: dict[str, float]     # solids only
    gases: dict[str, float]        # released (positive) or absorbed O2 (negative)
    delta_g: float                 # eV per atom of solid reactants


def stabilizing_shift(target: str, temperature_c: float, atmosphere: str) -> float:
    """Energy (eV per formula unit) to subtract from the target so it is just stable at its published conditions.

    Assumption, the ARROWS3 premise: a published target is the thermodynamic sink at the
    temperature and atmosphere it was made at. The 0 K snapshot can disagree (no O-deficient
    YBa2Cu3O6.3, no vibrational entropy), so the target is lowered to sit 0.03 eV/atom below its
    competing assemblage at those conditions. The shift is fixed per task, so other atmospheres
    and temperatures can still destabilise the target.
    """
    phase = phase_for(target)
    competing = equilibrium({target: 1.0}, temperature_c + 273.15, atmosphere, exclude_target=True)
    competing_energy = phase.energy + competing.delta_g * phase.atoms if competing else phase.energy
    gap = max(0.0, phase.energy - competing_energy)
    return gap + TARGET_STABILIZATION_EV_PER_ATOM * phase.atoms


def equilibrium(
    reactants: dict[str, float],
    temperature_k: float,
    atmosphere: str,
    target: str | None = None,
    shift: float = 0.0,
    exclude_target: bool = False,
) -> Reaction | None:
    """Minimum-Gibbs-energy products of `reactants` (moles) at T in an open atmosphere."""
    reactant_phases = {formula: phase_for(formula) for formula in reactants}
    elements: set[str] = set()
    for phase in reactant_phases.values():
        elements |= phase.elements
    solids = list({phase.formula: phase for phase in library(frozenset(elements))}.values())
    for phase in reactant_phases.values():
        if phase.formula not in {solid.formula for solid in solids}:
            solids.append(phase)
    if exclude_target:
        target = next(iter(reactants))
        solids = [solid for solid in solids if solid.formula != target]
        shift = 0.0
    elif target is not None and phase_for(target).elements <= elements | {"O"}:
        if target not in {solid.formula for solid in solids}:
            solids.append(phase_for(target))
    gases = [gas for gas in GASES if set(_composition_vector(gas)) <= elements | {"O"}]
    rows = sorted(elements | {"O"})
    columns = []
    costs = []
    for phase in solids:
        vector = _composition_vector(phase.formula)
        columns.append([vector.get(element, 0.0) for element in rows])
        costs.append(phase.energy - shift if phase.formula == target else phase.energy)
    for gas in gases:
        vector = _composition_vector(gas)
        columns.append([vector.get(element, 0.0) for element in rows])
        costs.append(gas_energy(gas, temperature_k))
    mu_o2 = oxygen_potential(temperature_k, atmosphere)
    oxygen_row = [2.0 if element == "O" else 0.0 for element in rows]
    columns.append(oxygen_row)
    costs.append(mu_o2)
    columns.append([-value for value in oxygen_row])
    costs.append(-mu_o2)
    matrix = np.array(columns).T
    rhs = np.zeros(len(rows))
    reactant_energy = 0.0
    reactant_atoms = 0.0
    for formula, moles in reactants.items():
        phase = reactant_phases[formula]
        vector = _composition_vector(phase.formula)
        for index, element in enumerate(rows):
            rhs[index] += moles * vector.get(element, 0.0)
        is_shifted_target = phase.formula == target and not exclude_target
        reactant_energy += moles * (phase.energy - shift if is_shifted_target else phase.energy)
        reactant_atoms += moles * phase.atoms
    result = linprog(np.array(costs), A_eq=matrix, b_eq=rhs, bounds=(0, None), method="highs")
    if not result.success or reactant_atoms <= 0:
        return None
    amounts = result.x
    products = {}
    for phase, amount in zip(solids, amounts[: len(solids)]):
        if amount > 1e-9:
            products[phase.formula] = float(amount)
    released = {}
    for gas, amount in zip(gases, amounts[len(solids): len(solids) + len(gases)]):
        if amount > 1e-9:
            released[gas] = float(amount)
    net_oxygen = float(amounts[-2] - amounts[-1])
    if abs(net_oxygen) > 1e-9:
        released["O2"] = net_oxygen
    delta_g = (float(result.fun) - reactant_energy) / reactant_atoms
    return Reaction(dict(reactants), products, released, delta_g)


@dataclass
class Crucible:
    """Hidden state of one sample."""

    target: str | None = None
    target_shift: float = 0.0
    moles: dict[str, float] = field(default_factory=dict)
    contact: float = 1.0
    charged_mass: float = 0.0
    released_mass: float = 0.0
    log: list[str] = field(default_factory=list)

    def add(self, precursors: dict[str, float]) -> None:
        for formula, amount in precursors.items():
            reduced = Composition(formula).reduced_formula
            self.moles[reduced] = self.moles.get(reduced, 0.0) + amount
            self.charged_mass += amount * Composition(reduced).weight
        self.contact = 1.0

    def regrind(self) -> None:
        self.contact = 1.0

    @property
    def mass_loss(self) -> float:
        if self.charged_mass <= 0:
            return 0.0
        return self.released_mass / self.charged_mass

    def weight_fractions(self) -> dict[str, float]:
        masses = {formula: amount * Composition(formula).weight for formula, amount in self.moles.items() if amount > 1e-9}
        total = sum(masses.values())
        if total <= 0:
            return {}
        return {formula: mass / total for formula, mass in sorted(masses.items(), key=lambda item: -item[1])}

    def heat(self, temperature_c: float, time_h: float, atmosphere: str, kinetics: Kinetics) -> None:
        temperature_k = temperature_c + 273.15
        sub_time = time_h / SUB_STEPS
        for _ in range(SUB_STEPS):
            self._react_once(temperature_k, sub_time, atmosphere, kinetics)
        self.contact *= 0.5

    def _candidate_groups(self) -> list[tuple[str, ...]]:
        present = [formula for formula, amount in self.moles.items() if amount > 1e-6]
        groups = [(formula,) for formula in present]
        for first_index in range(len(present)):
            for second_index in range(first_index + 1, len(present)):
                groups.append((present[first_index], present[second_index]))
        return groups

    def _react_once(self, temperature_k: float, time_h: float, atmosphere: str, kinetics: Kinetics) -> None:
        scored = []
        for group in self._candidate_groups():
            reaction = equilibrium({formula: self.moles[formula] for formula in group}, temperature_k, atmosphere, self.target, self.target_shift)
            if reaction is not None and reaction.delta_g < MIN_DRIVING_FORCE:
                scored.append((reaction.delta_g, group))
        scored.sort()
        for _, group in scored:
            if any(self.moles.get(formula, 0.0) <= 1e-6 for formula in group):
                continue
            reaction = equilibrium({formula: self.moles[formula] for formula in group}, temperature_k, atmosphere, self.target, self.target_shift)
            if reaction is None or reaction.delta_g >= MIN_DRIVING_FORCE:
                continue
            fraction = extent(kinetics, reaction.delta_g, temperature_k, time_h, self.contact, list(group))
            if fraction <= 1e-6:
                continue
            for formula, amount in reaction.reactants.items():
                self.moles[formula] -= fraction * amount
            for formula, amount in reaction.products.items():
                self.moles[formula] = self.moles.get(formula, 0.0) + fraction * amount
            for gas, amount in reaction.gases.items():
                self.released_mass += fraction * amount * Composition(gas).weight
        self.moles = {formula: amount for formula, amount in self.moles.items() if amount > 1e-9}


def xrd(crucible: Crucible, rng: np.random.Generator, noise: float = XRD_NOISE) -> dict[str, float]:
    """Noisy weight fractions as a refinement would report them; minor phases drop out."""
    observed = {}
    for formula, fraction in crucible.weight_fractions().items():
        reading = fraction + rng.normal(0.0, noise) if noise > 0 else fraction
        if reading >= XRD_DETECTION_LIMIT:
            observed[formula] = reading
    total = sum(observed.values())
    return {formula: round(float(value / total), 3) for formula, value in observed.items()} if total > 0 else {}
