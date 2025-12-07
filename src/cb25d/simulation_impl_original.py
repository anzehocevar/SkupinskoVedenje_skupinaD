# pyright:strict
from copy import copy
from dataclasses import dataclass
from typing import Self, TypedDict

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
        d = np.sqrt(np.square(u_x_i - u_x) + np.square(u_y_i - u_y))

        # Compute angle(s) of perception for fish i
        u_x_relative, u_y_relative = u_x - u_x_i, u_y - u_y_i
        theta = np.arctan2(u_y_relative, u_x_relative)
        psi = theta - self.phi[i]

        # Compute relative headings
        phi_relative = self.phi - self.phi[i]

        # Compute the heading angle changes
        d_sq = np.square(d)
        delta_phi = self.c_gamma_att * (
            (d * np.sin(psi)) / (1 + d_sq / np.square(self.c_l_att))
        ) + self.c_gamma_ali * (1 + self.c_eta * np.cos(psi)) * np.exp(
            -d_sq / np.square(self.c_l_ali)
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
    }


@dataclass
class SimulationRendererOriginal(SimulationRenderer[SimulationImplOriginal]):
    size: float
    color: tuple[int, int, int]
    dir_width: float
    fixed_size: bool = False

    def draw(self, e: RenderEnvironment, state: SimulationImplOriginal):
        scale = 1 if self.fixed_size else e.scale
        u_x, u_y = state.compute_positions(state.time)
        v_x, v_y = state.compute_velocities(state.time)
        for x, y, vx, vy in zip(u_x, u_y, v_x, v_y):
            pygame.draw.circle(
                e.screen,
                self.color,
                e.w2s((x, y)),
                self.size / scale,
            )
            pygame.draw.line(
                e.screen,
                self.color,
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
