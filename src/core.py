from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ============================================================
# Basic event / result containers
# ============================================================

@dataclass(frozen=True)
class Transition:
    """
    A single birth event.

    Attributes
    ----------
    parent : int
        Occupied site that produced the daughter.
    target : int
        Empty site that becomes occupied.
    """
    parent: int
    target: int


@dataclass
class SimulationResult:
    """
    Stores the simulation output.
    """
    times: np.ndarray
    cell_counts: np.ndarray
    snapshots: dict[float, np.ndarray]
    final_state: "CultureState"
    event_count: int


# ============================================================
# Lattice
# ============================================================

class Lattice(ABC):
    """
    Abstract lattice / graph interface.
    """

    @property
    @abstractmethod
    def n_sites(self) -> int:
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


class SquareLattice(Lattice):
    """
    2D square lattice with von Neumann neighborhood (4-neighbors).
    """

    def __init__(self, nrows: int, ncols: int, periodic: bool = False):
        self.nrows = nrows
        self.ncols = ncols
        self.periodic = periodic
        self._neighbor_table = self._build_neighbor_table()

    @property
    def n_sites(self) -> int:
        return self.nrows * self.ncols

    def site_index(self, row: int, col: int) -> int:
        return row * self.ncols + col

    def to_coordinate(self, site: int) -> tuple[int, int]:
        return site // self.ncols, site % self.ncols

    def neighbors(self, site: int) -> np.ndarray:
        return self._neighbor_table[site]

    def zeros_state(self) -> np.ndarray:
        return np.zeros(self.n_sites, dtype=bool)

    def reshape_state(self, values: np.ndarray) -> np.ndarray:
        return values.reshape(self.nrows, self.ncols)

    def _build_neighbor_table(self) -> list[np.ndarray]:
        table: list[np.ndarray] = []

        for site in range(self.n_sites):
            r, c = self.to_coordinate(site)
            neigh = []

            candidates = [
                (r - 1, c),
                (r + 1, c),
                (r, c - 1),
                (r, c + 1),
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


# ============================================================
# State
# ============================================================

@dataclass
class CultureState:
    """
    State of the cell culture.

    Parameters
    ----------
    occupancy : np.ndarray
        Boolean vector of occupied sites.
    birth_time : np.ndarray
        Birth time of the cell currently occupying each site.
        Undefined for empty sites, stored as np.nan.
    parent_site : np.ndarray
        Parent site index for the cell currently occupying each site.
        For seed cells, stored as -1.
    cell_id : np.ndarray
        Unique id of the cell occupying each site, or -1 if empty.
    parent_cell_id : np.ndarray
        Unique id of the parent cell, or -1 for seed cells.
    """

    occupancy: np.ndarray
    birth_time: np.ndarray
    parent_site: np.ndarray
    cell_id: np.ndarray
    parent_cell_id: np.ndarray
    next_cell_id: int = 0

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
        )

    @property
    def n_cells(self) -> int:
        return int(self.occupancy.sum())

    def occupy_seed_sites(self, sites: np.ndarray, birth_time: float = 0.0) -> None:
        """
        Fill a collection of initial seed sites.
        """
        sites = np.asarray(sites, dtype=int)

        for site in sites:
            if self.occupancy[site]:
                continue

            self.occupancy[site] = True
            self.birth_time[site] = birth_time
            self.parent_site[site] = -1
            self.cell_id[site] = self.next_cell_id
            self.parent_cell_id[site] = -1
            self.next_cell_id += 1

    def apply_transition(self, transition: Transition, t: float) -> None:
        """
        Apply a birth event to the state.
        """
        target = transition.target
        parent = transition.parent

        if self.occupancy[target]:
            raise ValueError("Target site is already occupied.")

        if not self.occupancy[parent]:
            raise ValueError("Parent site is not occupied.")

        self.occupancy[target] = True
        self.birth_time[target] = t
        self.parent_site[target] = parent
        self.parent_cell_id[target] = self.cell_id[parent]
        self.cell_id[target] = self.next_cell_id
        self.next_cell_id += 1


# ============================================================
# Initial conditions
# ============================================================

class InitialCondition(ABC):
    @abstractmethod
    def generate(self, lattice: Lattice) -> CultureState:
        pass


class DiskSeed(InitialCondition):
    """
    Initial occupied disk ("sphere" in the user's wording for the 2D lattice).
    """

    def __init__(self, radius: float, center: Optional[tuple[int, int]] = None):
        self.radius = radius
        self.center = center

    def generate(self, lattice: Lattice) -> CultureState:
        if not isinstance(lattice, SquareLattice):
            raise NotImplementedError(
                "DiskSeed is currently implemented only for SquareLattice."
            )

        state = CultureState.empty(lattice)

        if self.center is None:
            center_row = lattice.nrows // 2
            center_col = lattice.ncols // 2
        else:
            center_row, center_col = self.center

        radius_sq = self.radius ** 2
        seed_sites = []

        for site in range(lattice.n_sites):
            r, c = lattice.to_coordinate(site)
            dr = r - center_row
            dc = c - center_col
            if dr * dr + dc * dc <= radius_sq:
                seed_sites.append(site)

        state.occupy_seed_sites(np.array(seed_sites, dtype=int), birth_time=0.0)
        return state

class DoubleDiskSeed(InitialCondition):
    """
    Initial condition consisting of two disks of equal radius, centered on the
    same row and displaced horizontally.

    Parameters
    ----------
    radius : float
        Radius of each disk in lattice-node units.
    center_row : int | None, default=None
        Common row for both disk centers. If None, use the lattice mid-row.
    center_col_left : int | None, default=None
        Column of the left disk center. If None, choose it automatically so
        that the pair is approximately centered in the lattice.
    horizontal_separation : float | None, default=None
        Distance between the disk centers, in lattice-node units.
        If None, use 2 * radius, corresponding to disks that are just tangent
        in the continuum picture.
    """

    def __init__(
        self,
        radius: float,
        center_row: Optional[int] = None,
        center_col_left: Optional[int] = None,
        horizontal_separation: Optional[float] = None,
    ):
        self.radius = radius
        self.center_row = center_row
        self.center_col_left = center_col_left
        self.horizontal_separation = horizontal_separation

    def generate(self, lattice: Lattice) -> CultureState:
        if not isinstance(lattice, SquareLattice):
            raise NotImplementedError(
                "DoubleDiskSeed is currently implemented only for SquareLattice."
            )

        state = CultureState.empty(lattice)

        radius_sq = self.radius ** 2
        row_center = lattice.nrows // 2 if self.center_row is None else self.center_row

        separation = 2 * self.radius if self.horizontal_separation is None else self.horizontal_separation

        if self.center_col_left is None:
            center_pair_col = lattice.ncols // 2
            left_col = int(round(center_pair_col - separation / 2))
        else:
            left_col = self.center_col_left

        right_col = int(round(left_col + separation))

        seed_sites = []

        for site in range(lattice.n_sites):
            r, c = lattice.to_coordinate(site)

            dr_left = r - row_center
            dc_left = c - left_col

            dr_right = r - row_center
            dc_right = c - right_col

            in_left = (dr_left * dr_left + dc_left * dc_left) <= radius_sq
            in_right = (dr_right * dr_right + dc_right * dc_right) <= radius_sq

            if in_left or in_right:
                seed_sites.append(site)

        state.occupy_seed_sites(np.array(seed_sites, dtype=int), birth_time=0.0)
        return state


# ============================================================
# Dynamics
# ============================================================

class GrowthModel(ABC):
    """
    Abstract growth model with incremental frontier updates.

    Each concrete model maintains:
    - `eligible_sites`: sites that carry the Poisson clocks for the model
      (active occupied sites for model 1, fillable empty sites for model 2)
    - `event_rate`: per-site rate, here equal to 1 / tau
    """

    def __init__(self, tau: float):
        if tau <= 0:
            raise ValueError("tau must be positive.")
        self.tau = float(tau)
        self.eligible_sites: set[int] = set()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def initialize(self, state: CultureState, lattice: Lattice) -> None:
        pass

    @abstractmethod
    def sample_transition(
        self,
        state: CultureState,
        lattice: Lattice,
        rng: np.random.Generator,
    ) -> Optional[Transition]:
        pass

    @abstractmethod
    def update_after_transition(
        self,
        state: CultureState,
        lattice: Lattice,
        transition: Transition,
    ) -> None:
        pass

    def total_rate(self) -> float:
        """
        Total hazard rate of the current configuration.
        """
        return len(self.eligible_sites) / self.tau


class CellDrivenGrowth(GrowthModel):
    """
    Model 1:
    Every occupied site with at least one empty neighbor reproduces at rate 1/tau.
    Upon reproduction, one empty neighboring site is chosen uniformly.
    """

    @property
    def name(self) -> str:
        return "cell_driven"

    def _is_active_cell(self, site: int, state: CultureState, lattice: Lattice) -> bool:
        if not state.occupancy[site]:
            return False
        neigh = lattice.neighbors(site)
        return np.any(~state.occupancy[neigh])

    def initialize(self, state: CultureState, lattice: Lattice) -> None:
        self.eligible_sites.clear()
        for site in np.flatnonzero(state.occupancy):
            if self._is_active_cell(site, state, lattice):
                self.eligible_sites.add(int(site))

    def sample_transition(
        self,
        state: CultureState,
        lattice: Lattice,
        rng: np.random.Generator,
    ) -> Optional[Transition]:
        if not self.eligible_sites:
            return None

        eligible = np.fromiter(self.eligible_sites, dtype=int)
        parent = int(rng.choice(eligible))

        neigh = lattice.neighbors(parent)
        empty_neigh = neigh[~state.occupancy[neigh]]

        if empty_neigh.size == 0:
            # Defensive fallback; frontier should prevent this
            return None

        target = int(rng.choice(empty_neigh))
        return Transition(parent=parent, target=target)

    def update_after_transition(
        self,
        state: CultureState,
        lattice: Lattice,
        transition: Transition,
    ) -> None:
        target = transition.target
        parent = transition.parent

        affected = {parent, target}
        affected.update(int(x) for x in lattice.neighbors(parent))
        affected.update(int(x) for x in lattice.neighbors(target))

        for site in affected:
            if self._is_active_cell(site, state, lattice):
                self.eligible_sites.add(site)
            else:
                self.eligible_sites.discard(site)


class EmptyDrivenGrowth_neighborIndependent(GrowthModel):
    """
    Model 2:
    Every empty site with at least one occupied neighbor gets filled at rate 1/tau.
    If selected, one occupied neighboring site is chosen uniformly as the parent.
    """

    @property
    def name(self) -> str:
        return "empty_driven"

    def _is_fillable_empty(self, site: int, state: CultureState, lattice: Lattice) -> bool:
        if state.occupancy[site]:
            return False
        neigh = lattice.neighbors(site)
        return np.any(state.occupancy[neigh])

    def initialize(self, state: CultureState, lattice: Lattice) -> None:
        self.eligible_sites.clear()

        occupied_sites = np.flatnonzero(state.occupancy)
        candidates = set()
        for occ in occupied_sites:
            for neigh in lattice.neighbors(int(occ)):
                candidates.add(int(neigh))

        for site in candidates:
            if self._is_fillable_empty(site, state, lattice):
                self.eligible_sites.add(site)

    def sample_transition(
        self,
        state: CultureState,
        lattice: Lattice,
        rng: np.random.Generator,
    ) -> Optional[Transition]:
        if not self.eligible_sites:
            return None

        eligible = np.fromiter(self.eligible_sites, dtype=int)
        target = int(rng.choice(eligible))

        neigh = lattice.neighbors(target)
        occupied_neigh = neigh[state.occupancy[neigh]]

        if occupied_neigh.size == 0:
            # Defensive fallback; frontier should prevent this
            return None

        parent = int(rng.choice(occupied_neigh))
        return Transition(parent=parent, target=target)

    def update_after_transition(
        self,
        state: CultureState,
        lattice: Lattice,
        transition: Transition,
    ) -> None:
        target = transition.target

        affected = {target}
        affected.update(int(x) for x in lattice.neighbors(target))

        for site in affected:
            if self._is_fillable_empty(site, state, lattice):
                self.eligible_sites.add(site)
            else:
                self.eligible_sites.discard(site)

class EmptyDrivenGrowth(GrowthModel):
    """
    Model 2:
    Every occupied-empty neighboring pair contributes a rate 1/tau.

    Equivalently, if an empty site has k occupied neighbors, then its total
    filling rate is k / tau.

    When an empty site is selected for filling, one of its occupied neighbors
    is chosen uniformly as the parent.
    """

    @property
    def name(self) -> str:
        return "empty_driven"

    def __init__(self, tau: float):
        super().__init__(tau)
        self.occupied_neighbor_count: dict[int, int] = {}

    def _count_occupied_neighbors(
        self,
        site: int,
        state: CultureState,
        lattice: Lattice,
    ) -> int:
        if state.occupancy[site]:
            return 0
        neigh = lattice.neighbors(site)
        return int(np.sum(state.occupancy[neigh]))

    def _is_fillable_empty(
        self,
        site: int,
        state: CultureState,
        lattice: Lattice,
    ) -> bool:
        return (not state.occupancy[site]) and (self._count_occupied_neighbors(site, state, lattice) > 0)

    def initialize(self, state: CultureState, lattice: Lattice) -> None:
        self.eligible_sites.clear()
        self.occupied_neighbor_count.clear()

        occupied_sites = np.flatnonzero(state.occupancy)
        candidates = set()

        for occ in occupied_sites:
            for neigh in lattice.neighbors(int(occ)):
                candidates.add(int(neigh))

        for site in candidates:
            if state.occupancy[site]:
                continue
            k = self._count_occupied_neighbors(site, state, lattice)
            if k > 0:
                self.eligible_sites.add(site)
                self.occupied_neighbor_count[site] = k

    def total_rate(self) -> float:
        """
        Total rate is sum_x k_x / tau over fillable empty sites x,
        where k_x is the number of occupied neighbors of x.
        """
        if not self.eligible_sites:
            return 0.0
        total_weight = sum(self.occupied_neighbor_count[site] for site in self.eligible_sites)
        return total_weight / self.tau

    def sample_transition(
        self,
        state: CultureState,
        lattice: Lattice,
        rng: np.random.Generator,
    ) -> Optional[Transition]:
        if not self.eligible_sites:
            return None

        eligible = np.fromiter(self.eligible_sites, dtype=int)
        weights = np.array(
            [self.occupied_neighbor_count[site] for site in eligible],
            dtype=float,
        )

        weight_sum = weights.sum()
        if weight_sum <= 0.0:
            return None

        probs = weights / weight_sum
        target = int(rng.choice(eligible, p=probs))

        neigh = lattice.neighbors(target)
        occupied_neigh = neigh[state.occupancy[neigh]]

        if occupied_neigh.size == 0:
            return None

        parent = int(rng.choice(occupied_neigh))
        return Transition(parent=parent, target=target)

    def update_after_transition(
        self,
        state: CultureState,
        lattice: Lattice,
        transition: Transition,
    ) -> None:
        target = transition.target

        affected = {target}
        affected.update(int(x) for x in lattice.neighbors(target))

        for site in affected:
            if state.occupancy[site]:
                self.eligible_sites.discard(site)
                self.occupied_neighbor_count.pop(site, None)
                continue

            k = self._count_occupied_neighbors(site, state, lattice)

            if k > 0:
                self.eligible_sites.add(site)
                self.occupied_neighbor_count[site] = k
            else:
                self.eligible_sites.discard(site)
                self.occupied_neighbor_count.pop(site, None)


# ============================================================
# Simulator
# ============================================================

class CultureSimulator:
    """
    Continuous-time simulator using the Gillespie direct method.
    """

    def __init__(self, lattice: Lattice):
        self.lattice = lattice

    def run(
        self,
        initial_state: CultureState,
        model: GrowthModel,
        t_final: float,
        rng: np.random.Generator,
        snapshot_times: Optional[list[float]] = None,
    ) -> SimulationResult:
        if t_final < 0:
            raise ValueError("t_final must be non-negative.")

        state = initial_state.copy()
        t = 0.0

        if snapshot_times is None:
            snapshot_times = [0.0, t_final]
        snapshot_times = sorted(set(float(x) for x in snapshot_times))

        model.initialize(state, self.lattice)

        times = [t]
        cell_counts = [state.n_cells]
        snapshots: dict[float, np.ndarray] = {}

        snapshot_idx = 0
        while snapshot_idx < len(snapshot_times) and snapshot_times[snapshot_idx] <= t:
            snapshots[snapshot_times[snapshot_idx]] = state.occupancy.copy()
            snapshot_idx += 1

        event_count = 0

        while t < t_final:
            total_rate = model.total_rate()

            if total_rate <= 0.0:
                while snapshot_idx < len(snapshot_times):
                    snapshots[snapshot_times[snapshot_idx]] = state.occupancy.copy()
                    snapshot_idx += 1
                break

            dt = rng.exponential(scale=1.0 / total_rate)

            if t + dt > t_final:
                t = t_final
                while snapshot_idx < len(snapshot_times) and snapshot_times[snapshot_idx] <= t:
                    snapshots[snapshot_times[snapshot_idx]] = state.occupancy.copy()
                    snapshot_idx += 1
                times.append(t)
                cell_counts.append(state.n_cells)
                break

            t += dt

            transition = model.sample_transition(state, self.lattice, rng)
            if transition is None:
                while snapshot_idx < len(snapshot_times):
                    snapshots[snapshot_times[snapshot_idx]] = state.occupancy.copy()
                    snapshot_idx += 1
                break

            state.apply_transition(transition, t=t)
            model.update_after_transition(state, self.lattice, transition)

            event_count += 1

            while snapshot_idx < len(snapshot_times) and snapshot_times[snapshot_idx] <= t:
                snapshots[snapshot_times[snapshot_idx]] = state.occupancy.copy()
                snapshot_idx += 1

            times.append(t)
            cell_counts.append(state.n_cells)

        for ts in snapshot_times:
            if ts not in snapshots:
                snapshots[ts] = state.occupancy.copy()

        return SimulationResult(
            times=np.asarray(times, dtype=float),
            cell_counts=np.asarray(cell_counts, dtype=int),
            snapshots=snapshots,
            final_state=state,
            event_count=event_count,
        )