from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ============================================================
# Deterministic sampling helpers
# ============================================================

def _sorted_int_array(values: set[int]) -> np.ndarray:
    """
    Convert a set of integers into a sorted NumPy array.
    """
    if not values:
        return np.array([], dtype=int)
    return np.array(sorted(values), dtype=int)


def _sample_uniform_from_array(arr: np.ndarray, rng: np.random.Generator) -> int:
    """
    Sample one element uniformly from a 1D NumPy array.
    """
    if arr.size == 0:
        raise ValueError("Cannot sample from an empty array.")
    idx = int(rng.integers(arr.size))
    return int(arr[idx])


def _sample_index_from_positive_weights(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> int:
    """
    Sample an index with probability proportional to positive weights.

    Uses one uniform random draw and a cumulative sum. This makes the
    sampling logic explicit and stable within a fixed code version.
    """
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Weights must have positive sum.")

    u = rng.random() * total
    cumsum = np.cumsum(weights)
    idx = int(np.searchsorted(cumsum, u, side="right"))

    # Numerical safety in case u lands extremely close to total
    if idx >= weights.size:
        idx = weights.size - 1

    return idx


# ============================================================
# Events
# ============================================================

@dataclass(frozen=True)
class BirthEvent:
    parent: int
    target: int


@dataclass(frozen=True)
class DeathEvent:
    site: int


@dataclass(frozen=True)
class MigrationEvent:
    source: int
    target: int


# ============================================================
# Lattices
# ============================================================

class Lattice(ABC):
    @property
    @abstractmethod
    def n_sites(self) -> int:
        pass

    @property
    @abstractmethod
    def coordination_number(self) -> int:
        pass

    @property
    @abstractmethod
    def site_area(self) -> float:
        pass

    @property
    @abstractmethod
    def coordinates(self) -> np.ndarray:
        pass

    @abstractmethod
    def neighbors(self, site: int) -> np.ndarray:
        pass

    @abstractmethod
    def zeros_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def reshape_state(self, values: np.ndarray) -> np.ndarray:
        pass

    def occupied_neighbor_count(self, occupancy: np.ndarray, site: int) -> int:
        neigh = self.neighbors(site)
        return int(np.sum(occupancy[neigh]))

    def empty_neighbor_sites(self, occupancy: np.ndarray, site: int) -> np.ndarray:
        neigh = self.neighbors(site)
        return neigh[~occupancy[neigh]]

    def occupied_neighbor_sites(self, occupancy: np.ndarray, site: int) -> np.ndarray:
        neigh = self.neighbors(site)
        return neigh[occupancy[neigh]]

    def default_center(self) -> np.ndarray:
        return self.coordinates.mean(axis=0)

    def disk_site_indices(
        self,
        radius: float,
        center: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        center_xy = self.default_center() if center is None else np.asarray(center, dtype=float)
        dr = self.coordinates - center_xy[None, :]
        dist2 = np.sum(dr * dr, axis=1)
        return np.flatnonzero(dist2 <= radius * radius)

    def make_disk_state(
        self,
        radius: float,
        center: Optional[Sequence[float]] = None,
        birth_time: float = 0.0,
    ) -> "CultureState":
        state = CultureState.empty(self)
        seed_sites = self.disk_site_indices(radius=radius, center=center)
        state.occupy_seed_sites(seed_sites, birth_time=birth_time)
        state.initialize_frontier(self)
        return state


class SquareLattice(Lattice):
    def __init__(self, nrows: int, ncols: int, periodic: bool = False):
        self.nrows = nrows
        self.ncols = ncols
        self.periodic = periodic
        self._coords = self._build_coordinates()
        self._neighbor_table = self._build_neighbor_table()

    @property
    def n_sites(self) -> int:
        return self.nrows * self.ncols

    @property
    def coordination_number(self) -> int:
        return 4

    @property
    def site_area(self) -> float:
        return 1.0

    @property
    def coordinates(self) -> np.ndarray:
        return self._coords

    def site_index(self, row: int, col: int) -> int:
        return row * self.ncols + col

    def to_row_col(self, site: int) -> tuple[int, int]:
        return site // self.ncols, site % self.ncols

    def neighbors(self, site: int) -> np.ndarray:
        return self._neighbor_table[site]

    def zeros_state(self) -> np.ndarray:
        return np.zeros(self.n_sites, dtype=bool)

    def reshape_state(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(self.nrows, self.ncols)

    def _build_coordinates(self) -> np.ndarray:
        coords = np.zeros((self.n_sites, 2), dtype=float)
        for site in range(self.n_sites):
            r, c = self.to_row_col(site)
            coords[site] = np.array([c, r], dtype=float)
        return coords

    def _build_neighbor_table(self) -> list[np.ndarray]:
        table: list[np.ndarray] = []
        for site in range(self.n_sites):
            r, c = self.to_row_col(site)
            neigh = []
            candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
            for rr, cc in candidates:
                if self.periodic:
                    rr %= self.nrows
                    cc %= self.ncols
                    neigh.append(self.site_index(rr, cc))
                else:
                    if 0 <= rr < self.nrows and 0 <= cc < self.ncols:
                        neigh.append(self.site_index(rr, cc))
            table.append(np.array(neigh, dtype=int))
        return table


class TriangularLattice(Lattice):
    def __init__(self, nrows: int, ncols: int, periodic: bool = False):
        self.nrows = nrows
        self.ncols = ncols
        self.periodic = periodic
        self._coords = self._build_coordinates()
        self._neighbor_table = self._build_neighbor_table()

    @property
    def n_sites(self) -> int:
        return self.nrows * self.ncols

    @property
    def coordination_number(self) -> int:
        return 6

    @property
    def site_area(self) -> float:
        return np.sqrt(3.0) / 2.0

    @property
    def coordinates(self) -> np.ndarray:
        return self._coords

    def site_index(self, row: int, col: int) -> int:
        return row * self.ncols + col

    def to_row_col(self, site: int) -> tuple[int, int]:
        return site // self.ncols, site % self.ncols

    def neighbors(self, site: int) -> np.ndarray:
        return self._neighbor_table[site]

    def zeros_state(self) -> np.ndarray:
        return np.zeros(self.n_sites, dtype=bool)

    def reshape_state(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(self.nrows, self.ncols)

    def _build_coordinates(self) -> np.ndarray:
        coords = np.zeros((self.n_sites, 2), dtype=float)
        dy = np.sqrt(3.0) / 2.0
        for site in range(self.n_sites):
            r, c = self.to_row_col(site)
            x = c + 0.5 * (r % 2)
            y = r * dy
            coords[site] = np.array([x, y], dtype=float)
        return coords

    def _build_neighbor_table(self) -> list[np.ndarray]:
        table: list[np.ndarray] = []
        for site in range(self.n_sites):
            r, c = self.to_row_col(site)
            neigh = []

            if r % 2 == 0:
                candidates = [
                    (r, c - 1),
                    (r, c + 1),
                    (r - 1, c - 1),
                    (r - 1, c),
                    (r + 1, c - 1),
                    (r + 1, c),
                ]
            else:
                candidates = [
                    (r, c - 1),
                    (r, c + 1),
                    (r - 1, c),
                    (r - 1, c + 1),
                    (r + 1, c),
                    (r + 1, c + 1),
                ]

            for rr, cc in candidates:
                if self.periodic:
                    rr %= self.nrows
                    cc %= self.ncols
                    neigh.append(self.site_index(rr, cc))
                else:
                    if 0 <= rr < self.nrows and 0 <= cc < self.ncols:
                        neigh.append(self.site_index(rr, cc))

            table.append(np.array(neigh, dtype=int))
        return table


class HoneycombLattice(Lattice):
    def __init__(self, nrows: int, ncols: int, periodic: bool = False):
        self.nrows = nrows
        self.ncols = ncols
        self.periodic = periodic
        self._coords = self._build_coordinates()
        self._neighbor_table = self._build_neighbor_table()

    @property
    def n_sites(self) -> int:
        return self.nrows * self.ncols

    @property
    def coordination_number(self) -> int:
        return 3

    @property
    def site_area(self) -> float:
        return 3.0 * np.sqrt(3.0) / 4.0

    @property
    def coordinates(self) -> np.ndarray:
        return self._coords

    def site_index(self, row: int, col: int) -> int:
        return row * self.ncols + col

    def to_row_col(self, site: int) -> tuple[int, int]:
        return site // self.ncols, site % self.ncols

    def neighbors(self, site: int) -> np.ndarray:
        return self._neighbor_table[site]

    def zeros_state(self) -> np.ndarray:
        return np.zeros(self.n_sites, dtype=bool)

    def reshape_state(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(self.nrows, self.ncols)

    def _build_coordinates(self) -> np.ndarray:
        coords = np.zeros((self.n_sites, 2), dtype=float)
        dy = np.sqrt(3.0) / 2.0
        for site in range(self.n_sites):
            r, c = self.to_row_col(site)
            x = c + 0.5 * (r % 2)
            y = r * dy
            coords[site] = np.array([x, y], dtype=float)
        return coords

    def _build_neighbor_table(self) -> list[np.ndarray]:
        table: list[np.ndarray] = []
        for site in range(self.n_sites):
            r, c = self.to_row_col(site)
            neigh = []

            if r % 2 == 0:
                candidates = [
                    (r, c - 1),
                    (r, c + 1),
                    (r - 1, c),
                ]
            else:
                candidates = [
                    (r, c - 1),
                    (r, c + 1),
                    (r + 1, c),
                ]

            for rr, cc in candidates:
                if self.periodic:
                    rr %= self.nrows
                    cc %= self.ncols
                    neigh.append(self.site_index(rr, cc))
                else:
                    if 0 <= rr < self.nrows and 0 <= cc < self.ncols:
                        neigh.append(self.site_index(rr, cc))

            table.append(np.array(neigh, dtype=int))
        return table


HexagonalLattice = HoneycombLattice


# ============================================================
# State
# ============================================================

@dataclass
class CultureState:
    occupancy: np.ndarray
    birth_time: np.ndarray
    parent_site: np.ndarray
    cell_id: np.ndarray
    parent_cell_id: np.ndarray
    next_cell_id: int = 0
    occupied_sites: set[int] = field(default_factory=set)
    active_sites: set[int] = field(default_factory=set)
    boundary_empty_sites: set[int] = field(default_factory=set)

    @classmethod
    def empty(cls, lattice: Lattice) -> "CultureState":
        n = lattice.n_sites
        return cls(
            occupancy=np.zeros(n, dtype=bool),
            birth_time=np.full(n, np.nan, dtype=float),
            parent_site=np.full(n, -1, dtype=int),
            cell_id=np.full(n, -1, dtype=int),
            parent_cell_id=np.full(n, -1, dtype=int),
            next_cell_id=0,
        )

    def copy(self) -> "CultureState":
        return CultureState(
            occupancy=self.occupancy.copy(),
            birth_time=self.birth_time.copy(),
            parent_site=self.parent_site.copy(),
            cell_id=self.cell_id.copy(),
            parent_cell_id=self.parent_cell_id.copy(),
            next_cell_id=self.next_cell_id,
            occupied_sites=set(self.occupied_sites),
            active_sites=set(self.active_sites),
            boundary_empty_sites=set(self.boundary_empty_sites),
        )

    @property
    def n_cells(self) -> int:
        return len(self.occupied_sites)

    def occupy_seed_sites(self, sites: np.ndarray, birth_time: float = 0.0) -> None:
        sites = np.asarray(sites, dtype=int)
        for site in sites:
            if self.occupancy[site]:
                continue
            self.occupancy[site] = True
            self.birth_time[site] = birth_time
            self.parent_site[site] = -1
            self.parent_cell_id[site] = -1
            self.cell_id[site] = self.next_cell_id
            self.next_cell_id += 1
            self.occupied_sites.add(int(site))

    def apply_birth(self, event: BirthEvent, t: float) -> None:
        parent = event.parent
        target = event.target

        if not self.occupancy[parent]:
            raise ValueError("Birth parent is not occupied.")
        if self.occupancy[target]:
            raise ValueError("Birth target is already occupied.")

        self.occupancy[target] = True
        self.birth_time[target] = t
        self.parent_site[target] = parent
        self.parent_cell_id[target] = self.cell_id[parent]
        self.cell_id[target] = self.next_cell_id
        self.next_cell_id += 1
        self.occupied_sites.add(int(target))

    def apply_death(self, event: DeathEvent) -> None:
        site = event.site
        if not self.occupancy[site]:
            raise ValueError("Death site is already empty.")

        self.occupancy[site] = False
        self.birth_time[site] = np.nan
        self.parent_site[site] = -1
        self.parent_cell_id[site] = -1
        self.cell_id[site] = -1
        self.occupied_sites.discard(int(site))

    def _recompute_frontier_site(self, lattice: Lattice, site: int) -> None:
        if self.occupancy[site]:
            neigh = lattice.neighbors(site)
            if np.any(~self.occupancy[neigh]):
                self.active_sites.add(int(site))
            else:
                self.active_sites.discard(int(site))
            self.boundary_empty_sites.discard(int(site))
        else:
            neigh = lattice.neighbors(site)
            if np.any(self.occupancy[neigh]):
                self.boundary_empty_sites.add(int(site))
            else:
                self.boundary_empty_sites.discard(int(site))
            self.active_sites.discard(int(site))

    def initialize_frontier(self, lattice: Lattice) -> None:
        self.active_sites.clear()
        self.boundary_empty_sites.clear()

        candidates = set(self.occupied_sites)
        for site in self.occupied_sites:
            candidates.update(int(x) for x in lattice.neighbors(site))

        for site in candidates:
            self._recompute_frontier_site(lattice, site)

    def update_frontier_local(self, lattice: Lattice, changed_site: int) -> None:
        affected = {int(changed_site)}
        first_shell = lattice.neighbors(changed_site)
        affected.update(int(x) for x in first_shell)

        # Second shell is included because changing one site can modify the
        # coordination number of neighboring empty sites, which can in turn
        # modify the effective rates of nearby parents/targets.
        for site in first_shell:
            affected.update(int(x) for x in lattice.neighbors(int(site)))

        for site in affected:
            self._recompute_frontier_site(lattice, site)


# ============================================================
# Rate modifiers
# ============================================================

class LocalRateModifier(ABC):
    @abstractmethod
    def factor(self, k: int, z: int) -> float:
        pass

    def effective_rate(self, base_rate: float, k: int, z: int) -> float:
        return float(base_rate) * self.factor(k, z)


class NoAdhesion(LocalRateModifier):
    def factor(self, k: int, z: int) -> float:
        return 1.0


class HardThresholdModifier(LocalRateModifier):
    def __init__(self, k_min: int):
        self.k_min = int(k_min)

    def factor(self, k: int, z: int) -> float:
        return 1.0 if k >= self.k_min else 0.0


class PowerLawModifier(LocalRateModifier):
    def __init__(self, alpha: float):
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        self.alpha = float(alpha)

    def factor(self, k: int, z: int) -> float:
        if k <= 0:
            return 0.0
        return (k / z) ** self.alpha


class BoltzmannModifier(LocalRateModifier):
    def __init__(self, beta: float):
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        self.beta = float(beta)

    def factor(self, k: int, z: int) -> float:
        if k <= 0:
            return 0.0
        return float(np.exp(-self.beta * (z - k)))


# ============================================================
# Birth kernels
# ============================================================

@dataclass
class BirthProposal:
    total_rate: float
    sites: np.ndarray
    site_rates: np.ndarray


class BirthKernel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def prepare(
        self,
        state: CultureState,
        lattice: Lattice,
        r_birth: float,
        birth_modifier: LocalRateModifier,
    ) -> BirthProposal:
        pass

    @abstractmethod
    def sample_event(
        self,
        proposal: BirthProposal,
        state: CultureState,
        lattice: Lattice,
        r_birth: float,
        birth_modifier: LocalRateModifier,
        rng: np.random.Generator,
    ) -> Optional[BirthEvent]:
        pass


class CellDrivenBirthKernel(BirthKernel):
    @property
    def name(self) -> str:
        return "cell_driven"

    def prepare(
        self,
        state: CultureState,
        lattice: Lattice,
        r_birth: float,
        birth_modifier: LocalRateModifier,
    ) -> BirthProposal:
        if len(state.active_sites) == 0 or r_birth <= 0.0:
            return BirthProposal(0.0, np.array([], dtype=int), np.array([], dtype=float))

        z = lattice.coordination_number
        parents = _sorted_int_array(state.active_sites)
        rates = np.empty(parents.size, dtype=float)

        no_adhesion = isinstance(birth_modifier, NoAdhesion)

        for i, parent in enumerate(parents):
            empty_neigh = lattice.empty_neighbor_sites(state.occupancy, int(parent))
            m = empty_neigh.size

            if m == 0:
                rates[i] = 0.0
                continue

            if no_adhesion:
                rates[i] = r_birth
            else:
                s = 0.0
                for target in empty_neigh:
                    k_target = lattice.occupied_neighbor_count(state.occupancy, int(target))
                    s += birth_modifier.effective_rate(r_birth, k_target, z)
                rates[i] = s / m

        total_rate = float(rates.sum())
        return BirthProposal(total_rate=total_rate, sites=parents, site_rates=rates)

    def sample_event(
        self,
        proposal: BirthProposal,
        state: CultureState,
        lattice: Lattice,
        r_birth: float,
        birth_modifier: LocalRateModifier,
        rng: np.random.Generator,
    ) -> Optional[BirthEvent]:
        if proposal.total_rate <= 0.0:
            return None

        parent_idx = _sample_index_from_positive_weights(proposal.site_rates, rng)
        parent = int(proposal.sites[parent_idx])

        empty_neigh = lattice.empty_neighbor_sites(state.occupancy, parent)
        if empty_neigh.size == 0:
            return None

        if isinstance(birth_modifier, NoAdhesion):
            target = _sample_uniform_from_array(empty_neigh, rng)
            return BirthEvent(parent=parent, target=target)

        z = lattice.coordination_number
        weights = np.empty(empty_neigh.size, dtype=float)

        for i, target in enumerate(empty_neigh):
            k_target = lattice.occupied_neighbor_count(state.occupancy, int(target))
            weights[i] = birth_modifier.effective_rate(r_birth, k_target, z)

        total = float(weights.sum())
        if total <= 0.0:
            return None

        target_idx = _sample_index_from_positive_weights(weights, rng)
        target = int(empty_neigh[target_idx])

        return BirthEvent(parent=parent, target=target)


class EmptyDrivenBirthKernel(BirthKernel):
    @property
    def name(self) -> str:
        return "empty_driven"

    def prepare(
        self,
        state: CultureState,
        lattice: Lattice,
        r_birth: float,
        birth_modifier: LocalRateModifier,
    ) -> BirthProposal:
        if len(state.boundary_empty_sites) == 0 or r_birth <= 0.0:
            return BirthProposal(0.0, np.array([], dtype=int), np.array([], dtype=float))

        z = lattice.coordination_number
        targets = _sorted_int_array(state.boundary_empty_sites)
        rates = np.empty(targets.size, dtype=float)

        no_adhesion = isinstance(birth_modifier, NoAdhesion)

        for i, target in enumerate(targets):
            k_target = lattice.occupied_neighbor_count(state.occupancy, int(target))
            if k_target == 0:
                rates[i] = 0.0
                continue

            if no_adhesion:
                rates[i] = r_birth * k_target
            else:
                local = birth_modifier.effective_rate(r_birth, k_target, z)
                rates[i] = k_target * local

        total_rate = float(rates.sum())
        return BirthProposal(total_rate=total_rate, sites=targets, site_rates=rates)

    def sample_event(
        self,
        proposal: BirthProposal,
        state: CultureState,
        lattice: Lattice,
        r_birth: float,
        birth_modifier: LocalRateModifier,
        rng: np.random.Generator,
    ) -> Optional[BirthEvent]:
        if proposal.total_rate <= 0.0:
            return None

        target_idx = _sample_index_from_positive_weights(proposal.site_rates, rng)
        target = int(proposal.sites[target_idx])

        occupied_neigh = lattice.occupied_neighbor_sites(state.occupancy, target)
        if occupied_neigh.size == 0:
            return None

        parent = _sample_uniform_from_array(occupied_neigh, rng)
        return BirthEvent(parent=parent, target=target)


# ============================================================
# Death kernel
# ============================================================

@dataclass
class DeathProposal:
    total_rate: float
    sites: np.ndarray
    site_rates: Optional[np.ndarray]
    uniform: bool


class DeathKernel:
    def prepare(
        self,
        state: CultureState,
        lattice: Lattice,
        r_death: float,
        death_modifier: LocalRateModifier,
    ) -> DeathProposal:
        if r_death <= 0.0 or state.n_cells == 0:
            return DeathProposal(
                total_rate=0.0,
                sites=np.array([], dtype=int),
                site_rates=None,
                uniform=True,
            )

        occupied = _sorted_int_array(state.occupied_sites)

        if isinstance(death_modifier, NoAdhesion):
            return DeathProposal(
                total_rate=float(r_death * occupied.size),
                sites=occupied,
                site_rates=None,
                uniform=True,
            )

        z = lattice.coordination_number
        rates = np.empty(occupied.size, dtype=float)
        for i, site in enumerate(occupied):
            k_site = lattice.occupied_neighbor_count(state.occupancy, int(site))
            rates[i] = death_modifier.effective_rate(r_death, k_site, z)

        return DeathProposal(
            total_rate=float(rates.sum()),
            sites=occupied,
            site_rates=rates,
            uniform=False,
        )

    def sample_event(
        self,
        proposal: DeathProposal,
        rng: np.random.Generator,
    ) -> Optional[DeathEvent]:
        if proposal.total_rate <= 0.0 or proposal.sites.size == 0:
            return None

        if proposal.uniform:
            site = _sample_uniform_from_array(proposal.sites, rng)
            return DeathEvent(site=site)

        idx = _sample_index_from_positive_weights(proposal.site_rates, rng)
        site = int(proposal.sites[idx])
        return DeathEvent(site=site)


# ============================================================
# Migration placeholder
# ============================================================

@dataclass
class MigrationProposal:
    total_rate: float


class MigrationKernel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def prepare(
        self,
        state: CultureState,
        lattice: Lattice,
        r_migration: float,
        migration_modifier: LocalRateModifier,
    ) -> MigrationProposal:
        pass

    @abstractmethod
    def sample_event(
        self,
        proposal: MigrationProposal,
        state: CultureState,
        lattice: Lattice,
        r_migration: float,
        migration_modifier: LocalRateModifier,
        rng: np.random.Generator,
    ) -> Optional[MigrationEvent]:
        pass


class NoMigration(MigrationKernel):
    @property
    def name(self) -> str:
        return "no_migration"

    def prepare(
        self,
        state: CultureState,
        lattice: Lattice,
        r_migration: float,
        migration_modifier: LocalRateModifier,
    ) -> MigrationProposal:
        return MigrationProposal(total_rate=0.0)

    def sample_event(
        self,
        proposal: MigrationProposal,
        state: CultureState,
        lattice: Lattice,
        r_migration: float,
        migration_modifier: LocalRateModifier,
        rng: np.random.Generator,
    ) -> Optional[MigrationEvent]:
        return None


# ============================================================
# Model config
# ============================================================

@dataclass
class ModelConfig:
    birth_kernel: BirthKernel
    r_birth: float
    r_death: float = 0.0
    r_migration: float = 0.0
    birth_modifier: LocalRateModifier = field(default_factory=NoAdhesion)
    death_modifier: LocalRateModifier = field(default_factory=NoAdhesion)
    migration_modifier: LocalRateModifier = field(default_factory=NoAdhesion)
    death_kernel: DeathKernel = field(default_factory=DeathKernel)
    migration_kernel: MigrationKernel = field(default_factory=NoMigration)

    def __post_init__(self) -> None:
        if self.r_birth < 0:
            raise ValueError("r_birth must be non-negative.")
        if self.r_death < 0:
            raise ValueError("r_death must be non-negative.")
        if self.r_migration < 0:
            raise ValueError("r_migration must be non-negative.")


# ============================================================
# Observables
# ============================================================

@dataclass
class InstantObservables:
    n_cells: int
    density: float
    mean_radius: float
    equiv_radius: float
    roughness: float
    n_active: int
    center_x: float
    center_y: float


def compute_instant_observables(state: CultureState, lattice: Lattice) -> InstantObservables:
    n = state.n_cells

    if n == 0:
        return InstantObservables(
            n_cells=0,
            density=0.0,
            mean_radius=np.nan,
            equiv_radius=0.0,
            roughness=np.nan,
            n_active=0,
            center_x=np.nan,
            center_y=np.nan,
        )

    occ_idx = _sorted_int_array(state.occupied_sites)
    occ_xy = lattice.coordinates[occ_idx]
    center = occ_xy.mean(axis=0)

    radii = np.linalg.norm(occ_xy - center[None, :], axis=1)
    mean_radius = float(radii.mean())

    equiv_radius = float(np.sqrt(n * lattice.site_area / np.pi))
    density = float(n / lattice.n_sites)

    n_active = len(state.active_sites)
    if n_active > 0:
        active_idx = _sorted_int_array(state.active_sites)
        active_xy = lattice.coordinates[active_idx]
        active_radii = np.linalg.norm(active_xy - center[None, :], axis=1)
        roughness = float(active_radii.std(ddof=0))
    else:
        roughness = np.nan

    return InstantObservables(
        n_cells=n,
        density=density,
        mean_radius=mean_radius,
        equiv_radius=equiv_radius,
        roughness=roughness,
        n_active=n_active,
        center_x=float(center[0]),
        center_y=float(center[1]),
    )


# ============================================================
# Result
# ============================================================

@dataclass
class SimulationResult:
    times: np.ndarray
    population: np.ndarray
    density: np.ndarray
    mean_radius: np.ndarray
    equiv_radius: np.ndarray
    roughness: np.ndarray
    n_active: np.ndarray
    center_x: np.ndarray
    center_y: np.ndarray
    front_velocity: np.ndarray
    snapshots: Optional[dict[float, np.ndarray]]
    final_state: CultureState
    event_count: int
    extinct: bool


# ============================================================
# Simulator
# ============================================================

class CultureSimulator:
    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    def run(
        self,
        initial_state: CultureState,
        config: ModelConfig,
        t_final: float,
        rng: np.random.Generator,
        observation_times: Optional[Sequence[float]] = None,
        store_snapshots: bool = False,
    ) -> SimulationResult:
        if t_final < 0:
            raise ValueError("t_final must be non-negative.")

        state = initial_state.copy()
        state.initialize_frontier(self.lattice)

        if observation_times is None:
            observation_times = [0.0, t_final]

        obs_times = np.array(sorted(set(float(x) for x in observation_times)), dtype=float)
        if obs_times[0] < 0 or obs_times[-1] > t_final:
            raise ValueError("observation_times must lie within [0, t_final].")

        times: list[float] = []
        population: list[float] = []
        density: list[float] = []
        mean_radius: list[float] = []
        equiv_radius: list[float] = []
        roughness: list[float] = []
        n_active: list[float] = []
        center_x: list[float] = []
        center_y: list[float] = []

        snapshots: Optional[dict[float, np.ndarray]] = {} if store_snapshots else None

        t = 0.0
        event_count = 0
        obs_idx = 0

        def record_current_state(t_record: float) -> None:
            obs = compute_instant_observables(state, self.lattice)

            times.append(t_record)
            population.append(obs.n_cells)
            density.append(obs.density)
            mean_radius.append(obs.mean_radius)
            equiv_radius.append(obs.equiv_radius)
            roughness.append(obs.roughness)
            n_active.append(obs.n_active)
            center_x.append(obs.center_x)
            center_y.append(obs.center_y)

            if snapshots is not None:
                snapshots[float(t_record)] = state.occupancy.copy()

        while obs_idx < len(obs_times) and obs_times[obs_idx] <= 0.0:
            record_current_state(float(obs_times[obs_idx]))
            obs_idx += 1

        while t < t_final:
            birth_prop = config.birth_kernel.prepare(
                state=state,
                lattice=self.lattice,
                r_birth=config.r_birth,
                birth_modifier=config.birth_modifier,
            )
            death_prop = config.death_kernel.prepare(
                state=state,
                lattice=self.lattice,
                r_death=config.r_death,
                death_modifier=config.death_modifier,
            )
            migration_prop = config.migration_kernel.prepare(
                state=state,
                lattice=self.lattice,
                r_migration=config.r_migration,
                migration_modifier=config.migration_modifier,
            )

            total_birth = birth_prop.total_rate
            total_death = death_prop.total_rate
            total_migration = migration_prop.total_rate
            total_rate = total_birth + total_death + total_migration

            if total_rate <= 0.0:
                while obs_idx < len(obs_times):
                    record_current_state(float(obs_times[obs_idx]))
                    obs_idx += 1
                break

            dt = rng.exponential(scale=1.0 / total_rate)

            if t + dt > t_final:
                t = t_final
                while obs_idx < len(obs_times) and obs_times[obs_idx] <= t:
                    record_current_state(float(obs_times[obs_idx]))
                    obs_idx += 1
                break

            t += dt

            u = rng.random() * total_rate

            if u < total_birth:
                event = config.birth_kernel.sample_event(
                    proposal=birth_prop,
                    state=state,
                    lattice=self.lattice,
                    r_birth=config.r_birth,
                    birth_modifier=config.birth_modifier,
                    rng=rng,
                )
                if event is not None:
                    state.apply_birth(event, t=t)
                    state.update_frontier_local(self.lattice, event.target)

            elif u < total_birth + total_death:
                event = config.death_kernel.sample_event(
                    proposal=death_prop,
                    rng=rng,
                )
                if event is not None:
                    state.apply_death(event)
                    state.update_frontier_local(self.lattice, event.site)

            else:
                raise NotImplementedError("Migration is not implemented yet.")

            event_count += 1

            while obs_idx < len(obs_times) and obs_times[obs_idx] <= t:
                record_current_state(float(obs_times[obs_idx]))
                obs_idx += 1

        if len(times) == 0:
            raise RuntimeError("No observations were recorded. Check observation_times.")

        times_arr = np.asarray(times, dtype=float)
        mean_radius_arr = np.asarray(mean_radius, dtype=float)

        if len(times_arr) >= 2:
            front_velocity = np.gradient(mean_radius_arr, times_arr)
        else:
            front_velocity = np.array([np.nan], dtype=float)

        final_state = state.copy()

        return SimulationResult(
            times=times_arr,
            population=np.asarray(population, dtype=float),
            density=np.asarray(density, dtype=float),
            mean_radius=mean_radius_arr,
            equiv_radius=np.asarray(equiv_radius, dtype=float),
            roughness=np.asarray(roughness, dtype=float),
            n_active=np.asarray(n_active, dtype=float),
            center_x=np.asarray(center_x, dtype=float),
            center_y=np.asarray(center_y, dtype=float),
            front_velocity=front_velocity,
            snapshots=snapshots,
            final_state=final_state,
            event_count=event_count,
            extinct=(final_state.n_cells == 0),
        )


# ============================================================
# Factories
# ============================================================

def make_birth_kernel(update_source: str) -> BirthKernel:
    update_source = update_source.lower().strip()
    if update_source == "cell_driven":
        return CellDrivenBirthKernel()
    if update_source == "empty_driven":
        return EmptyDrivenBirthKernel()
    raise ValueError(f"Unknown update source: {update_source}")


def make_lattice(kind: str, nrows: int, ncols: int, periodic: bool = False) -> Lattice:
    kind = kind.lower().strip()
    if kind == "square":
        return SquareLattice(nrows=nrows, ncols=ncols, periodic=periodic)
    if kind == "triangular":
        return TriangularLattice(nrows=nrows, ncols=ncols, periodic=periodic)
    if kind in {"hexagonal", "honeycomb"}:
        return HoneycombLattice(nrows=nrows, ncols=ncols, periodic=periodic)
    raise ValueError(f"Unknown lattice kind: {kind}")