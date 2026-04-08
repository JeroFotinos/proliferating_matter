from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ============================================================
# Deterministic sampling helpers
# ============================================================

def _sorted_int_array(values: set[int]) -> np.ndarray:
    if not values:
        return np.array([], dtype=int)
    return np.array(sorted(values), dtype=int)


def _sample_uniform_from_array(arr: np.ndarray, rng: np.random.Generator) -> int:
    if arr.size == 0:
        raise ValueError("Cannot sample from an empty array.")
    idx = int(rng.integers(arr.size))
    return int(arr[idx])


def _sample_index_from_positive_weights(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> int:
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Weights must have positive sum.")

    u = rng.random() * total
    cumsum = np.cumsum(weights)
    idx = int(np.searchsorted(cumsum, u, side="right"))
    if idx >= weights.size:
        idx = weights.size - 1
    return idx


# ============================================================
# Fenwick tree for exact O(log N) weighted updates/sampling
# ============================================================

class FenwickTree:
    """
    Fenwick tree (binary indexed tree) for nonnegative weights.

    Supports:
    - point updates in O(log N)
    - total sum in O(1)
    - exact weighted sampling by prefix inversion in O(log N)
    """

    def __init__(self, size: int):
        if size < 0:
            raise ValueError("size must be non-negative.")
        self.size = int(size)
        self.tree = np.zeros(self.size + 1, dtype=float)
        self.values = np.zeros(self.size, dtype=float)

    def total(self) -> float:
        return float(self.tree[self.size]) if self.size > 0 else 0.0

    def get(self, index: int) -> float:
        return float(self.values[index])

    def set(self, index: int, value: float) -> None:
        value = float(value)
        if value < 0.0:
            raise ValueError("Fenwick weights must be nonnegative.")

        delta = value - self.values[index]
        if delta == 0.0:
            return

        self.values[index] = value
        i = index + 1
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def build(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.size,):
            raise ValueError("build array has incorrect shape.")
        if np.any(values < 0.0):
            raise ValueError("Fenwick weights must be nonnegative.")

        self.tree.fill(0.0)
        self.values = values.copy()

        for i in range(self.size):
            j = i + 1
            self.tree[j] += self.values[i]
            parent = j + (j & -j)
            if parent <= self.size:
                self.tree[parent] += self.tree[j]

    def sample(self, rng: np.random.Generator) -> int:
        total = float(self.values.sum())
        if total <= 0.0:
            raise ValueError("Cannot sample from zero-total Fenwick tree.")

        target = rng.random() * total

        idx = 0
        bit = 1 << (self.size.bit_length() - 1) if self.size > 0 else 0
        running = 0.0

        while bit != 0:
            nxt = idx + bit
            if nxt <= self.size and running + self.tree[nxt] <= target:
                idx = nxt
                running += self.tree[nxt]
            bit >>= 1

        if idx >= self.size:
            idx = self.size - 1
        return int(idx)


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

            # Horizontal neighbors
            candidates = [
                (r, c - 1),
                (r, c + 1),
            ]

            # One vertical neighbor, chosen by checkerboard parity
            # so the edge is reciprocal and the graph is fully connected.
            if (r + c) % 2 == 0:
                candidates.append((r + 1, c))
            else:
                candidates.append((r - 1, c))

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

    def get_local_affected_sites(self, lattice: Lattice, changed_site: int) -> set[int]:
        """
        Return the local neighborhood whose frontier membership and/or local
        rates may change after flipping occupancy at changed_site.

        We include the second shell because birth rates can depend on the
        coordination number of nearby empty sites.
        """
        affected = {int(changed_site)}
        first_shell = lattice.neighbors(changed_site)
        affected.update(int(x) for x in first_shell)

        for site in first_shell:
            affected.update(int(x) for x in lattice.neighbors(int(site)))

        return affected

    def update_frontier_local(self, lattice: Lattice, changed_site: int) -> set[int]:
        affected = self.get_local_affected_sites(lattice, changed_site)
        for site in affected:
            self._recompute_frontier_site(lattice, site)
        return affected


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

class BirthKernel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute_site_rate(
        self,
        state: CultureState,
        lattice: Lattice,
        site: int,
        r_birth: float,
        birth_modifier: LocalRateModifier,
    ) -> float:
        pass

    @abstractmethod
    def sample_event_from_site(
        self,
        site: int,
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

    def compute_site_rate(
        self,
        state: CultureState,
        lattice: Lattice,
        site: int,
        r_birth: float,
        birth_modifier: LocalRateModifier,
    ) -> float:
        if r_birth <= 0.0:
            return 0.0
        if site not in state.active_sites:
            return 0.0

        if isinstance(birth_modifier, NoAdhesion):
            return float(r_birth)

        z = lattice.coordination_number
        empty_neigh = lattice.empty_neighbor_sites(state.occupancy, site)
        m = empty_neigh.size
        if m == 0:
            return 0.0

        s = 0.0
        for target in empty_neigh:
            k_target = lattice.occupied_neighbor_count(state.occupancy, int(target))
            s += birth_modifier.effective_rate(r_birth, k_target, z)

        return s / m

    def sample_event_from_site(
        self,
        site: int,
        state: CultureState,
        lattice: Lattice,
        r_birth: float,
        birth_modifier: LocalRateModifier,
        rng: np.random.Generator,
    ) -> Optional[BirthEvent]:
        if site not in state.active_sites:
            return None

        empty_neigh = lattice.empty_neighbor_sites(state.occupancy, site)
        if empty_neigh.size == 0:
            return None

        if isinstance(birth_modifier, NoAdhesion):
            target = _sample_uniform_from_array(empty_neigh, rng)
            return BirthEvent(parent=int(site), target=target)

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

        return BirthEvent(parent=int(site), target=target)


class EmptyDrivenBirthKernel(BirthKernel):
    @property
    def name(self) -> str:
        return "empty_driven"

    def compute_site_rate(
        self,
        state: CultureState,
        lattice: Lattice,
        site: int,
        r_birth: float,
        birth_modifier: LocalRateModifier,
    ) -> float:
        if r_birth <= 0.0:
            return 0.0
        if site not in state.boundary_empty_sites:
            return 0.0

        k_target = lattice.occupied_neighbor_count(state.occupancy, site)
        if k_target == 0:
            return 0.0

        if isinstance(birth_modifier, NoAdhesion):
            return float(r_birth * k_target)

        z = lattice.coordination_number
        local = birth_modifier.effective_rate(r_birth, k_target, z)
        return float(k_target * local)

    def sample_event_from_site(
        self,
        site: int,
        state: CultureState,
        lattice: Lattice,
        r_birth: float,
        birth_modifier: LocalRateModifier,
        rng: np.random.Generator,
    ) -> Optional[BirthEvent]:
        if site not in state.boundary_empty_sites:
            return None

        occupied_neigh = lattice.occupied_neighbor_sites(state.occupancy, site)
        if occupied_neigh.size == 0:
            return None

        parent = _sample_uniform_from_array(occupied_neigh, rng)
        return BirthEvent(parent=parent, target=int(site))


# ============================================================
# Death kernel
# ============================================================

class DeathKernel:
    def compute_site_rate(
        self,
        state: CultureState,
        lattice: Lattice,
        site: int,
        r_death: float,
        death_modifier: LocalRateModifier,
    ) -> float:
        if r_death <= 0.0:
            return 0.0
        if site not in state.occupied_sites:
            return 0.0

        if isinstance(death_modifier, NoAdhesion):
            return float(r_death)

        z = lattice.coordination_number
        k_site = lattice.occupied_neighbor_count(state.occupancy, site)
        return float(death_modifier.effective_rate(r_death, k_site, z))

    def sample_event_from_site(self, site: int) -> DeathEvent:
        return DeathEvent(site=int(site))


# ============================================================
# Migration placeholder
# ============================================================

class MigrationKernel(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass


class NoMigration(MigrationKernel):
    @property
    def name(self) -> str:
        return "no_migration"


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
# Instant observables
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
# Rate cache
# ============================================================

@dataclass
class RateCache:
    birth_rates: np.ndarray
    death_rates: np.ndarray
    migration_rates: np.ndarray
    birth_tree: FenwickTree
    death_tree: FenwickTree
    migration_tree: FenwickTree


# ============================================================
# Simulator
# ============================================================

class CultureSimulator:
    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    def _initialize_rate_cache(
        self,
        state: CultureState,
        config: ModelConfig,
    ) -> RateCache:
        n = self.lattice.n_sites

        birth_rates = np.zeros(n, dtype=float)
        death_rates = np.zeros(n, dtype=float)
        migration_rates = np.zeros(n, dtype=float)

        # Initialize only on currently eligible sets for births/deaths.
        if config.r_birth > 0.0:
            if isinstance(config.birth_kernel, CellDrivenBirthKernel):
                candidate_birth_sites = state.active_sites
            elif isinstance(config.birth_kernel, EmptyDrivenBirthKernel):
                candidate_birth_sites = state.boundary_empty_sites
            else:
                candidate_birth_sites = set()

            for site in candidate_birth_sites:
                birth_rates[site] = config.birth_kernel.compute_site_rate(
                    state=state,
                    lattice=self.lattice,
                    site=int(site),
                    r_birth=config.r_birth,
                    birth_modifier=config.birth_modifier,
                )

        if config.r_death > 0.0:
            for site in state.occupied_sites:
                death_rates[site] = config.death_kernel.compute_site_rate(
                    state=state,
                    lattice=self.lattice,
                    site=int(site),
                    r_death=config.r_death,
                    death_modifier=config.death_modifier,
                )

        birth_tree = FenwickTree(n)
        death_tree = FenwickTree(n)
        migration_tree = FenwickTree(n)

        birth_tree.build(birth_rates)
        death_tree.build(death_rates)
        migration_tree.build(migration_rates)

        return RateCache(
            birth_rates=birth_rates,
            death_rates=death_rates,
            migration_rates=migration_rates,
            birth_tree=birth_tree,
            death_tree=death_tree,
            migration_tree=migration_tree,
        )

    def _update_local_rate_cache(
        self,
        state: CultureState,
        config: ModelConfig,
        rate_cache: RateCache,
        affected_sites: set[int],
    ) -> None:
        for site in affected_sites:
            new_birth = config.birth_kernel.compute_site_rate(
                state=state,
                lattice=self.lattice,
                site=int(site),
                r_birth=config.r_birth,
                birth_modifier=config.birth_modifier,
            )
            if new_birth != rate_cache.birth_rates[site]:
                rate_cache.birth_rates[site] = new_birth
                rate_cache.birth_tree.set(int(site), new_birth)

            new_death = config.death_kernel.compute_site_rate(
                state=state,
                lattice=self.lattice,
                site=int(site),
                r_death=config.r_death,
                death_modifier=config.death_modifier,
            )
            if new_death != rate_cache.death_rates[site]:
                rate_cache.death_rates[site] = new_death
                rate_cache.death_tree.set(int(site), new_death)

        # Migration remains zero for now.

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

        rate_cache = self._initialize_rate_cache(state, config)

        if observation_times is None:
            observation_times = [0.0, t_final]

        obs_times = np.array(sorted(set(float(x) for x in observation_times)), dtype=float)
        if obs_times.size == 0:
            raise ValueError("observation_times must be non-empty.")
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
            total_birth = float(rate_cache.birth_rates.sum())
            total_death = float(rate_cache.death_rates.sum())
            total_migration = 0.0
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

            applied_event = False

            if u < total_birth:
                site = rate_cache.birth_tree.sample(rng)
                event = config.birth_kernel.sample_event_from_site(
                    site=int(site),
                    state=state,
                    lattice=self.lattice,
                    r_birth=config.r_birth,
                    birth_modifier=config.birth_modifier,
                    rng=rng,
                )

                if event is not None:
                    state.apply_birth(event, t=t)
                    affected_sites = state.update_frontier_local(self.lattice, event.target)
                    self._update_local_rate_cache(
                        state=state,
                        config=config,
                        rate_cache=rate_cache,
                        affected_sites=affected_sites,
                    )
                    applied_event = True

            elif u < total_birth + total_death:
                site = rate_cache.death_tree.sample(rng)
                event = config.death_kernel.sample_event_from_site(int(site))

                if event is not None:
                    state.apply_death(event)
                    affected_sites = state.update_frontier_local(self.lattice, event.site)
                    self._update_local_rate_cache(
                        state=state,
                        config=config,
                        rate_cache=rate_cache,
                        affected_sites=affected_sites,
                    )
                    applied_event = True

            else:
                raise NotImplementedError("Migration is not implemented yet.")

            if applied_event:
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