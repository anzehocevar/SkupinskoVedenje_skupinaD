# pyright:strict
from copy import copy
from dataclasses import dataclass
from typing import Self, TypedDict
import itertools

import numpy as np
import pygame

from cb25d.render_environment import RenderEnvironment
from cb25d.simulation_framework import SimulationRecorder, SimulationRenderer


@dataclass(kw_only=True, slots=True)
class SimulationImplOriginal:
    # Constants
    c_eta: float
    """Coefficient of anisotropy."""
    c_gamma_ali: float
    """Strength of alignment.

    Possible values: 0.0, 1.2
    """
    c_gamma_att: float
    """Strength of attraction.

    Possible values: 0.0, 0.6
    """
    # k=1:
    #   Swarming: gamma_att, gamma_ali = (0.6, 0.6)
    #   Schooling: gamma_att, gamma_ali = (0.22, 0.6)
    #   Milling: gamma_att, gamma_ali = (0.37, 0.2)
    # k=2:
    #   Swarming: gamma_att, gamma_ali = (0.6, 0.2)
    #   Schooling: gamma_att, gamma_ali = (0.2, 0.3)
    c_gamma_rand: float
    """Noise intensity."""
    c_k: int
    """Number of neighbours to consider before every kick.

    All 3 states are possible for k=1.
    Only schooling and swarming are possible for k=2.
    """
    c_l_ali: float
    """Length of alignment.
    
    Possible values: 1.5, 3, 5
    """
    c_l_att: float
    """Length of attraction.
    
    Possible values: 1.5, 3, 5
    """
    c_tau_0: float
    """Relaxation time."""

    c_dist_critical: float
    """Critical distance for grouping. Default: = 4*l_att."""
    c_dist_merge: float
    """Groups with fish, separated by distance less than this will be merged. Default: min(l_att, l_ali)."""

    # Variables
    time: float
    """Will always be at the beginning of a kick unless the state is an interpolation."""
    rng: np.random.Generator
    """Random number generator with state, or just saved RNG state."""
    u_x_last: np.ndarray
    """The X coordinate of each fish at the beginning of its current kick."""
    u_y_last: np.ndarray
    """The Y coordinate of each fish at the beginning of its current kick."""
    phi: np.ndarray
    """The heading of each fish."""
    t_last: np.ndarray
    """Absolute time of each fish's kick start."""
    tau: np.ndarray
    """Length and duration of each fish's kick."""
    d_ij: np.ndarray
    """Pairwise distances between all fish."""
    group: np.ndarray
    """Index of group that each fish belongs to."""

    _dirty: bool = False
    """If true, this state has been snapshotted and some its members are referenced in another state.
    
    In this case, safely updating the state requires copying the mutable members and resetting the _dirty flag.
    """

    def _undirty(self):
        if not self._dirty:
            return
        bg = self.rng.bit_generator.__class__(0)
        bg.state = self.rng.bit_generator.state
        self.rng = np.random.Generator(bg)
        self.u_x_last = np.copy(self.u_x_last)
        self.u_y_last = np.copy(self.u_y_last)
        self.phi = np.copy(self.phi)
        self.t_last = np.copy(self.t_last)
        self.tau = np.copy(self.tau)
        self.d_ij = np.copy(self.d_ij)
        self.group = np.copy(self.group)
        self._dirty = False

    def compute_positions(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        t_since_last_kick = t - self.t_last
        phi_unitvec_x, phi_unitvec_y = np.cos(self.phi), np.sin(self.phi)
        scale = (
            self.tau
            * (1 - np.exp(-t_since_last_kick / self.c_tau_0))
            / (1 - np.exp(-self.tau / self.c_tau_0))
        )
        u_x = self.u_x_last + scale * phi_unitvec_x
        u_y = self.u_y_last + scale * phi_unitvec_y
        return u_x, u_y

    def compute_velocities(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        t_since_last_kick = t - self.t_last
        phi_unitvec_x, phi_unitvec_y = np.cos(self.phi), np.sin(self.phi)
        scale = (  # Partial derivative of position computation's scale with respect to t_since_last_kick
            self.tau
            * np.exp(-t_since_last_kick / self.c_tau_0)
            / (self.c_tau_0 - self.c_tau_0 * np.exp(-self.tau / self.c_tau_0))
        )
        v_x = scale * phi_unitvec_x
        v_y = scale * phi_unitvec_y
        return v_x, v_y

    def compute_groups(self) -> np.ndarray:
        n: int = self.u_x_last.shape[0]
        nearest_neighbours_indexes: np.ndarray = np.zeros((n, n-1), dtype=np.int64)    # N-1 because we know every fish is nearest to itself
        for i in range(n):
            nearest_neighbours_indexes[i] = np.argsort(self.d_ij[i])[1:]
        self.group = np.arange(n, dtype=np.int64)
        last_in_sequence: np.ndarray = self.group.copy()
        for i in range(n):
            i1: int = i
            i2: int = nearest_neighbours_indexes[i, 0]
            if self.d_ij[i1, i2] <= self.c_dist_critical:
                while nearest_neighbours_indexes[i2, 0] != i1:
                    i1 = i2
                    i2 = nearest_neighbours_indexes[i1, 0]
                last_in_sequence[i] = min(i1, i2)
        # group = group[last_in_sequence]
        for i in range(n):
            self.group[i] = self.group[last_in_sequence[i]]

        sets: list[set[int]] = []
        index_to_set_index: dict[int, int] = {}
        for i in range(n):
            for j in range(i+1, n):
                if self.d_ij[i, j] < self.c_dist_merge:
                    ii, jj = i in index_to_set_index.keys(), j in index_to_set_index.keys()
                    if ii and jj:
                        if (isi := index_to_set_index[i]) != (jsi := index_to_set_index[j]):
                            sets[isi].update(sets[jsi])
                            for j1 in sets[jsi]:
                                index_to_set_index[j1] = isi
                            sets[jsi].clear()
                    elif ii:
                        sets[index_to_set_index[i]].add(j)
                        index_to_set_index[j] = index_to_set_index[i]
                    elif jj:
                        sets[index_to_set_index[j]].add(i)
                        index_to_set_index[i] = index_to_set_index[j]
                    else:
                        sets.append({i, j})
                        index_to_set_index[i] = len(sets)-1
                        index_to_set_index[j] = len(sets)-1
        min_elems = [min(s) if len(s) > 0 else None for s in sets]
        for k in range(n):
            si = index_to_set_index.get(self.group[k], None)
            if si is not None:
                self.group[k] = min_elems[si]
        return self.group

    def step(self) -> None:
        self._undirty()

        # Find time and fish of next kick
        t_next = self.t_last + self.tau
        i = int(np.argmin(t_next))
        t = t_next[i]

        # Compute position of every fish at this time: u(t)
        u_x, u_y = self.compute_positions(t)

        # Compute distances from fish i to all other fish
        u_x_i, u_y_i = u_x[i], u_y[i]
        d_i = np.sqrt(np.square(u_x_i - u_x) + np.square(u_y_i - u_y))
        self.d_ij[i] = d_i
        self.d_ij[:, i] = d_i

        # Compute angle(s) of perception for fish i
        u_x_relative, u_y_relative = u_x - u_x_i, u_y - u_y_i
        theta = np.arctan2(u_y_relative, u_x_relative)
        psi = theta - self.phi[i]

        # Compute relative headings
        phi_relative = self.phi - self.phi[i]

        # Compute the heading angle changes
        d_i_sq = np.square(d_i)
        delta_phi = self.c_gamma_att * (
            (d_i * np.sin(psi)) / (1 + d_i_sq / np.square(self.c_l_att))
        ) + self.c_gamma_ali * (1 + self.c_eta * np.cos(psi)) * np.exp(
            -d_i_sq / np.square(self.c_l_ali)
        ) * np.sin(phi_relative)
        influence = np.abs(delta_phi)
        top_k_indexes = np.argpartition(influence, -self.c_k)[-self.c_k :]

        # Apply full kick movement for the selected fish
        self.u_x_last[i] = u_x[i]
        self.u_y_last[i] = u_y[i]

        # Compute new heading
        self.phi[i] = (
            self.phi[i]
            + self.c_gamma_rand * self.rng.normal()
            + np.sum(delta_phi[top_k_indexes])
        ) % (2 * np.pi)

        # Prepare for next kick
        self.t_last[i] = t
        self.tau[i] = (
            1.0
            * np.sqrt(2 / np.pi)
            * np.sqrt(-2.0 * np.log(self.rng.uniform() + 1e-16))
        )

        self.time = t

    def snapshot(self):
        self._dirty = True
        return copy(self)

    def interpolate(self, other: Self, t: float):
        ret = self.snapshot()
        ret.time = (1 - t) * self.time + t * other.time
        return ret


class _KwargsInitialConditions(TypedDict):
    c_l_att: float
    time: float
    rng: np.random.Generator
    u_x_last: np.ndarray
    u_y_last: np.ndarray
    phi: np.ndarray
    t_last: np.ndarray
    tau: np.ndarray
    d_ij: np.ndarray
    group: np.ndarray

def compute_pairwise_distances(u_x: np.ndarray, u_y: np.ndarray) -> np.ndarray:
    N: int = u_x.shape[0]
    d_ij: np.ndarray = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d_ij[i, j] = np.sqrt(np.square(u_x[i]-u_x[j]) + np.square(u_y[i]-u_y[j]))
    d_ij = d_ij + d_ij.T
    return d_ij

def generate_initial_conditions(
    *,
    seed: int,
    n: int,
    l_att: float,
) -> _KwargsInitialConditions:
    rng = np.random.default_rng(seed)
    """Uniformly random placement of fish in circle with centre (0, 0) and radius R"""
    R: float = (l_att / 2.0) * np.sqrt(n / np.pi)
    r = R * np.sqrt(rng.random(n))
    angle = rng.random(n) * 2 * np.pi
    u_x = r * np.cos(angle)
    u_y = r * np.sin(angle)
    phi = rng.random(n) * 2 * np.pi
    d_ij = compute_pairwise_distances(u_x, u_y)
    return {
        "c_l_att": l_att,
        "time": 0,
        "rng": rng,
        "u_x_last": u_x,
        "u_y_last": u_y,
        "phi": phi,
        "t_last": np.zeros(n),
        "tau": (
            0.5
            * np.sqrt(2 / np.pi)
            * np.sqrt(-2.0 * np.log(rng.uniform(size=n) + 1e-16))
        ),
        "d_ij": d_ij,
        "group": np.zeros(n, dtype=np.int64),
    }


@dataclass
class SimulationRendererOriginal(SimulationRenderer[SimulationImplOriginal]):
    size: float
    color: tuple[int, int, int]
    dir_width: float
    fixed_size: bool = False
    use_groups: bool = False
    """Determines if we should compute what fish belongs to what group. Heavy performance hit."""

    def draw(self, e: RenderEnvironment, state: SimulationImplOriginal):
        scale = 1 if self.fixed_size else e.scale
        u_x, u_y = state.compute_positions(state.time)
        v_x, v_y = state.compute_velocities(state.time)
        if self.use_groups:
            state.compute_groups()
            groups: np.ndarray = np.unique(state.group)
            index_colorspace: np.ndarray = np.linspace(0, 6*255, len(groups), endpoint=False).astype(int)
            group_to_index: np.ndarray = np.array(np.full(groups.max()+1, -1))
            group_to_index[groups] = np.arange(len(groups))
        for x, y, vx, vy, ix in zip(u_x, u_y, v_x, v_y, (group_to_index[state.group] if self.use_groups else itertools.repeat(0))):
            color: tuple[int, int, int] = (self.red[index_colorspace[ix]], self.green[index_colorspace[ix]], self.blue[index_colorspace[ix]]) if self.use_groups else self.color
            pygame.draw.circle(
                e.screen,
                color,
                e.w2s((x, y)),
                self.size / scale,
            )
            pygame.draw.line(
                e.screen,
                color,
                e.w2s((x, y)),
                e.w2s((x + vx, y + vy)),
                int(self.dir_width / scale),
            )


@dataclass
class SimulationRecorderOriginal(SimulationRecorder[SimulationImplOriginal]):
    # Config
    skip_first_n: int = 0

    # Statistics
    total_samples: int = 0
    total_dispersion: float = 0
    total_polarization: float = 0
    total_milling: float = 0

    def record(self, state: SimulationImplOriginal):
        self.total_samples += 1
        if self.total_samples <= self.skip_first_n:
            return

        u_x, u_y = state.compute_positions(state.time)
        v_x, v_y = state.compute_velocities(state.time)

        # Barycenter positon and velocity
        b_x, b_y = np.mean(u_x), np.mean(u_y)
        bv_x, bv_y = np.mean(v_x), np.mean(v_y)

        relative_pos = np.atan2(u_y - b_y, u_x - b_x)
        relative_heading = np.atan2(v_y - bv_y, v_x - bv_x)

        # Statistics
        self.total_dispersion += np.sqrt(np.mean((u_x - b_x) ** 2 + (u_y - b_y) ** 2))
        self.total_polarization += (
            np.sqrt(np.sum(np.cos(state.phi)) ** 2 + np.sum(np.sin(state.phi)) ** 2)
            / state.phi.size
        )
        self.total_milling += np.abs(np.mean(np.sin(relative_heading - relative_pos)))

    @property
    def samples(self) -> float:
        return self.total_samples - self.skip_first_n

    @property
    def results_available(self) -> bool:
        return self.samples > 0

    @property
    def dispersion(self) -> float:
        return self.total_dispersion / self.samples

    @property
    def polarization(self) -> float:
        return self.total_polarization / self.samples

    @property
    def milling(self) -> float:
        return self.total_milling / self.samples
