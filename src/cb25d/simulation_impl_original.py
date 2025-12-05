# pyright:strict
from copy import deepcopy
from dataclasses import dataclass
from typing import Self, TypedDict

import numpy as np
import pygame

from cb25d.render_environment import RenderEnvironment
from cb25d.simulation_framework import SimulationRenderer


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
    c_tau_n_mean: float
    """Kick duration and length (mean)."""
    c_tau_n_std: float
    """Kick duration and length (std)."""

    # Variables

    time: float
    """Will always be at the beginning of a kick unless the state is an interpolation."""
    rng: np.random.Generator
    """Random number generator with state."""
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

    def step(self) -> None:
        # Find time and fish of next kick
        t_next = self.t_last + self.tau
        i = int(np.argmin(t_next))
        t = t_next[i]

        # Compute time since all fish's last kicks
        s = t - self.t_last

        # Compute position of every fish at this time: u(t)
        phi_unitvec_x, phi_unitvec_y = np.cos(self.phi), np.sin(self.phi)
        scale = self.tau * (
            (1 - np.exp(-s / self.c_tau_0)) / (1 - np.exp(-self.tau / self.c_tau_0))
        )
        u_x = self.u_x_last + scale * phi_unitvec_x
        u_y = self.u_y_last + scale * phi_unitvec_y

        # Compute distances from fish i to all other fish
        u_x_i, u_y_i = u_x[i], u_y[i]
        d = np.sqrt(np.square(u_x_i - u_x) + np.square(u_y_i - u_y))

        # Compute angle(s) of perception for fish i
        u_x_relative, u_y_relative = u_x - u_x_i, u_y - u_y_i
        theta = np.arctan2(u_y_relative, u_x_relative)
        psi = theta - self.phi

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
            + self.c_gamma_rand
            * np.sqrt(-2.0 * np.log(self.rng.random() + 1e-16))
            * np.sin(2 * np.pi * self.rng.random())
            + np.sum(delta_phi[top_k_indexes])
        )

        # Prepare for next kick
        self.t_last[i] = t
        self.tau[i] = (
            0.5
            * np.sqrt(2 / np.pi)
            * np.sqrt(-2.0 * np.log(self.rng.uniform() + 1e-16))
        )

        self.time = t

    def snapshot(self):
        # I should optimize this, but it's fast enough for debugging
        return deepcopy(self)

    def interpolate(self, other: Self, t: float):
        ret = deepcopy(self)
        ret.time = (1 - t) * self.time + t * other.time
        return ret


class _KwargsInitialConditions(TypedDict):
    c_l_att: float
    c_tau_n_mean: float
    c_tau_n_std: float
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
    tau_n_mean: float,
    tau_n_std: float,
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
        "c_tau_n_mean": tau_n_mean,
        "c_tau_n_std": tau_n_std,
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
    pos_size: float
    pos_color: tuple[int, int, int]
    dir_len: float
    dir_width: float
    dir_color: tuple[int, int, int]

    def draw(self, e: RenderEnvironment, state: SimulationImplOriginal):
        s = state.time - state.t_last
        phi_unitvec_x, phi_unitvec_y = np.cos(state.phi), np.sin(state.phi)
        scale = state.tau * (
            (1 - np.exp(-s / state.c_tau_0)) / (1 - np.exp(-state.tau / state.c_tau_0))
        )
        u_x = state.u_x_last + scale * phi_unitvec_x
        u_y = state.u_y_last + scale * phi_unitvec_y
        for x, y, dx, dy in zip(
            u_x,
            u_y,
            np.cos(state.phi),
            np.sin(state.phi),
        ):
            dx, dy = self.dir_len * dx, self.dir_len * dy
            pygame.draw.circle(
                e.screen, self.pos_color, e.w2s((x, y)), self.pos_size / e.scale
            )
            pygame.draw.line(
                e.screen,
                self.dir_color,
                e.w2s((x, y)),
                e.w2s((x + dx, y + dy)),
                int(self.dir_width / e.scale),
            )
