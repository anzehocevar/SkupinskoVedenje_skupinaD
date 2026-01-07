import itertools
from copy import copy
from dataclasses import dataclass, field
from typing import Self, TypedDict

import numpy as np
import pygame

from cb25d.render_environment import RenderEnvironment
from cb25d.simulation_framework import SimulationRecorder, SimulationRenderer


@dataclass(kw_only=True, slots=True)
class SimulationImplExtended:
    # Constants
    c_eta: float
    """Coefficient of anisotropy."""
    c_gamma_ali: float
    """Strength of alignment."""
    c_gamma_att: float
    """Strength of attraction."""
    c_gamma_rand: float
    """Noise intensity."""
    c_k: int
    """Number of neighbours to consider before every kick."""
    c_l_ali: float
    """Length of alignment."""
    c_l_att: float
    """Length of attraction."""
    c_tau_0: float
    """Relaxation time."""

    c_dist_critical: float
    """Critical distance for grouping."""
    c_dist_merge: float
    """Groups with fish, separated by distance less than this will be merged."""

    # Extended model variables/constants
    c_omega: float
    """Duty cycle (0.0 to 1.0). Ratio of burst phase duration to total cycle duration."""
    c_n_omega: int
    """Number of decision steps (heading updates) per cycle."""

    # Variables
    time: float
    rng: np.random.Generator
    u_x_last: np.ndarray
    """The X coordinate of each fish at the beginning of its current decision step."""
    u_y_last: np.ndarray
    """The Y coordinate of each fish at the beginning of its current decision step."""
    phi: np.ndarray
    """The heading of each fish."""
    t_last: np.ndarray
    """Absolute time of each fish's current decision step start."""
    tau: np.ndarray
    """Total duration of the current full swimming cycle (sum of burst and coast)."""
    t_cycle_start: np.ndarray
    """Absolute time when the current swimming cycle started."""
    step_count: np.ndarray
    """The index of the current decision step within the cycle [0, n_omega - 1]."""
    d_ij: np.ndarray
    """Pairwise distances between all fish."""
    group: np.ndarray
    """Index of group that each fish belongs to."""

    _dirty: bool = False

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
        self.t_cycle_start = np.copy(self.t_cycle_start)
        self.step_count = np.copy(self.step_count)
        self.d_ij = np.copy(self.d_ij)
        self._dirty = False

    def _get_v0(self) -> np.ndarray:
        """Calculates the initial burst velocity for the current cycle of each fish."""
        # v0 = tau / (omega*tau + tau0 * (1 - exp(-(1-omega)*tau/tau0)))
        # This ensures the total distance covered in one cycle (integral of v) equals tau.
        num = self.tau
        den = self.c_omega * self.tau + self.c_tau_0 * (
            1 - np.exp(-(1 - self.c_omega) * self.tau / self.c_tau_0)
        )
        return num / den

    def _get_displacement(self, dt: np.ndarray, tau: np.ndarray, v0: np.ndarray) -> np.ndarray:
        """Calculates the cumulative displacement S(dt) at time dt since cycle start."""
        t_burst = self.c_omega * tau
        
        # If dt <= t_burst: S = v0 * dt
        # If dt > t_burst: S = v0 * t_burst + v0 * tau0 * (1 - exp(-(dt - t_burst)/tau0))
        
        disp = np.where(
            dt <= t_burst,
            v0 * dt,
            v0 * t_burst + v0 * self.c_tau_0 * (1 - np.exp(-(dt - t_burst) / self.c_tau_0))
        )
        return disp
    
    def _burst_step_dt(self) -> np.ndarray:
        """Duration of one decision step during the burst phase."""
        return (self.c_omega * self.tau) / self.c_n_omega

    def compute_positions(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        v0 = self._get_v0()
        
        # Time relative to cycle start
        dt_now = t - self.t_cycle_start
        dt_prev = self.t_last - self.t_cycle_start
        
        # S(dt_now) - S(dt_prev) is the distance covered since the start of the current sub-step
        s_now = self._get_displacement(dt_now, self.tau, v0)
        s_prev = self._get_displacement(dt_prev, self.tau, v0)
        scale = s_now - s_prev
        
        phi_unitvec_x, phi_unitvec_y = np.cos(self.phi), np.sin(self.phi)
        u_x = self.u_x_last + scale * phi_unitvec_x
        u_y = self.u_y_last + scale * phi_unitvec_y
        return u_x, u_y

    def compute_velocities(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        v0 = self._get_v0()
        dt = t - self.t_cycle_start
        t_burst = self.c_omega * self.tau
        
        # v(dt) = v0 if dt <= t_burst else v0 * exp(-(dt - t_burst)/tau0)
        v_mag = np.where(
            dt <= t_burst,
            v0,
            v0 * np.exp(-(dt - t_burst) / self.c_tau_0)
        )
        
        phi_unitvec_x, phi_unitvec_y = np.cos(self.phi), np.sin(self.phi)
        v_x = v_mag * phi_unitvec_x
        v_y = v_mag * phi_unitvec_y
        return v_x, v_y

    def compute_groups(self) -> np.ndarray:
        n: int = self.u_x_last.shape[0]
        nearest_neighbours_indexes: np.ndarray = np.zeros((n, n - 1), dtype=np.int64)
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
        for i in range(n):
            self.group[i] = self.group[last_in_sequence[i]]

        sets: list[set[int]] = []
        index_to_set_index: dict[int, int] = {}
        for i in range(n):
            for j in range(i + 1, n):
                if self.d_ij[i, j] < self.c_dist_merge:
                    ii, jj = i in index_to_set_index, j in index_to_set_index
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
                        index_to_set_index[i] = len(sets) - 1
                        index_to_set_index[j] = len(sets) - 1
        min_elems = [min(s) if len(s) > 0 else None for s in sets]
        for k in range(n):
            si = index_to_set_index.get(self.group[k], None)
            if si is not None:
                self.group[k] = min_elems[si] # type: ignore
        return self.group

    def step(self) -> None:
        self._undirty()

        burst_step = self._burst_step_dt()
        t_next = self.t_last + burst_step

        i = int(np.argmin(t_next))
        t = t_next[i]

        dt = t - self.t_cycle_start[i]
        t_burst = self.c_omega * self.tau[i]

        # End of burst → start new cycle
        if dt > t_burst:
            self.step_count[i] = 0
            self.t_cycle_start[i] += self.tau[i]
            self.tau[i] = self.rng.rayleigh(np.sqrt(2 / np.pi))
            self.t_last[i] = self.t_cycle_start[i]
            self.time = self.t_cycle_start[i]
            return

        # Normal burst step
        u_x, u_y = self.compute_positions(t)

        u_x_i, u_y_i = u_x[i], u_y[i]
        d_i = np.sqrt((u_x_i - u_x)**2 + (u_y_i - u_y)**2)
        self.d_ij[i] = d_i
        self.d_ij[:, i] = d_i

        u_x_rel, u_y_rel = u_x - u_x_i, u_y - u_y_i
        theta = np.arctan2(u_y_rel, u_x_rel)
        psi = theta - self.phi[i]
        phi_rel = self.phi - self.phi[i]

        d_i_sq = d_i**2
        delta_phi = (
            self.c_gamma_att * (d_i * np.sin(psi)) / (1 + d_i_sq / self.c_l_att**2)
            + self.c_gamma_ali
            * (1 + self.c_eta * np.cos(psi))
            * np.exp(-d_i_sq / self.c_l_ali**2)
            * np.sin(phi_rel)
        )

        top_k = np.argpartition(np.abs(delta_phi), -self.c_k)[-self.c_k:]

        self.u_x_last[i] = u_x[i]
        self.u_y_last[i] = u_y[i]

        self.phi[i] = (
            self.phi[i]
            + self.c_gamma_rand * self.rng.normal()
            + np.sum(delta_phi[top_k])
        ) % (2 * np.pi)

        self.t_last[i] = t
        self.step_count[i] += 1
        self.time = t

        def snapshot(self):
            self._dirty = True
            return copy(self)

    def interpolate(self, other: Self, t: float):
        ret = self.snapshot()
        ret.time = (1 - t) * self.time + t * other.time
        return ret


class _KwargsInitialConditionsExt(TypedDict):
    c_l_att: float
    c_omega: float
    c_n_omega: int
    time: float
    rng: np.random.Generator
    u_x_last: np.ndarray
    u_y_last: np.ndarray
    phi: np.ndarray
    t_last: np.ndarray
    tau: np.ndarray
    t_cycle_start: np.ndarray
    step_count: np.ndarray
    d_ij: np.ndarray
    group: np.ndarray


def compute_pairwise_distances(u_x: np.ndarray, u_y: np.ndarray) -> np.ndarray:
    N: int = u_x.shape[0]
    d_ij: np.ndarray = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d_ij[i, j] = np.sqrt(np.square(u_x[i] - u_x[j]) + np.square(u_y[i] - u_y[j]))
    d_ij = d_ij + d_ij.T
    return d_ij


def generate_extended_initial_conditions(
    *,
    seed: int,
    n: int,
    l_att: float,
    omega: float = 0.0,
    n_omega: int = 1,
) -> _KwargsInitialConditionsExt:
    rng = np.random.default_rng(seed)
    R: float = (l_att / 2.0) * np.sqrt(n / np.pi)
    r = R * np.sqrt(rng.random(n))
    angle = rng.random(n) * 2 * np.pi
    u_x = r * np.cos(angle)
    u_y = r * np.sin(angle)
    phi = rng.random(n) * 2 * np.pi
    d_ij = compute_pairwise_distances(u_x, u_y)
    
    # Matching the Rayleigh sampling logic from original's generate_initial_conditions
    tau_vals = 0.5 * np.sqrt(2 / np.pi) * np.sqrt(-2.0 * np.log(rng.uniform(size=n) + 1e-16))
    
    return {
        "c_l_att": l_att,
        "c_omega": omega,
        "c_n_omega": n_omega,
        "time": 0,
        "rng": rng,
        "u_x_last": u_x,
        "u_y_last": u_y,
        "phi": phi,
        "t_last": np.zeros(n),
        "tau": tau_vals,
        "t_cycle_start": np.zeros(n),
        "step_count": np.zeros(n, dtype=np.int64),
        "d_ij": d_ij,
        "group": np.zeros(n, dtype=np.int64),
    }


@dataclass
class SimulationRendererExtended(SimulationRenderer[SimulationImplExtended]):
    size: float
    color: tuple[int, int, int]
    dir_width: float
    fixed_size: bool = False
    use_groups: bool = False

    _red: np.ndarray = field(init=False)
    _green: np.ndarray = field(init=False)
    _blue: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._generate_colorspace()

    def _generate_colorspace(self):
        red = np.zeros(6 * 255)
        green = np.zeros(6 * 255)
        blue = np.zeros(6 * 255)
        offset: int = 0
        for i in range(255):
            red[i + offset], green[i + offset], blue[i + offset] = 0, 255, i
        offset += 255
        for i in range(255):
            red[i + offset], green[i + offset], blue[i + offset] = 0, 255 - i, 255
        offset += 255
        for i in range(255):
            red[i + offset], green[i + offset], blue[i + offset] = i, 0, 255
        offset += 255
        for i in range(255):
            red[i + offset], green[i + offset], blue[i + offset] = 255, 0, 255 - i
        offset += 255
        for i in range(255):
            red[i + offset], green[i + offset], blue[i + offset] = 255, i, 0
        offset += 255
        for i in range(255):
            red[i + offset], green[i + offset], blue[i + offset] = 255 - i, 255, 0
        self._red = red
        self._green = green
        self._blue = blue

    def draw(self, e: RenderEnvironment, state: SimulationImplExtended):
        scale = 1 if self.fixed_size else e.scale
        u_x, u_y = state.compute_positions(state.time)
        v_x, v_y = state.compute_velocities(state.time)
        if self.use_groups:
            state.compute_groups()
            groups: np.ndarray = np.unique(state.group)
            index_colorspace: np.ndarray = np.linspace(
                0, 6 * 255, len(groups), endpoint=False
            ).astype(int)
            group_to_index: np.ndarray = np.array(np.full(groups.max() + 1, -1))
            group_to_index[groups] = np.arange(len(groups))
        
        for x, y, vx, vy, ix in zip(
            u_x,
            u_y,
            v_x,
            v_y,
            (group_to_index[state.group] if self.use_groups else itertools.repeat(0)), # type: ignore
        ):
            color: tuple[int, int, int] = ( # type: ignore
                (
                    self._red[index_colorspace[ix]], # type: ignore
                    self._green[index_colorspace[ix]], # type: ignore
                    self._blue[index_colorspace[ix]], # type: ignore
                )
                if self.use_groups
                else self.color
            )
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
class SimulationRecorderExtended(SimulationRecorder[SimulationImplExtended]):
    skip_first_n: int = 0
    use_groups: bool = False
    total_samples: int = 0
    total_dispersion: float = 0
    total_polarization: float = 0
    total_milling: float = 0
    n_groups: list[int] | None = None

    def record(self, state: SimulationImplExtended):
        self.total_samples += 1
        if self.total_samples <= self.skip_first_n:
            return

        u_x, u_y = state.compute_positions(state.time)
        v_x, v_y = state.compute_velocities(state.time)

        b_x, b_y = np.mean(u_x), np.mean(u_y)
        bv_x, bv_y = np.mean(v_x), np.mean(v_y)

        relative_pos = np.atan2(u_y - b_y, u_x - b_x)
        relative_heading = np.atan2(v_y - bv_y, v_x - bv_x)

        self.total_dispersion += np.sqrt(np.mean((u_x - b_x) ** 2 + (u_y - b_y) ** 2))
        self.total_polarization += (
            np.sqrt(np.sum(np.cos(state.phi)) ** 2 + np.sum(np.sin(state.phi)) ** 2)
            / state.phi.size
        )
        self.total_milling += np.abs(np.mean(np.sin(relative_heading - relative_pos)))

        if self.use_groups:
            if not self.n_groups:
                self.n_groups = []
            groups: np.ndarray = np.unique(state.compute_groups())
            self.n_groups.append(len(groups))

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