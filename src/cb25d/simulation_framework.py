# pyright: strict
"""A high level description of the simulation logic.

Defined separately to make things more readable.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Self
import numpy as np

from cb25d.render_environment import RenderEnvironment


class SimulationImpl(Protocol):
    """Implements the simulation logic. Driven by an external simulation loop.

    Defined as a Protocol since implementation should be a numba jitclass.
    Implementation should use lazy computation since `.snapshot()` and `.interpolate()` will be getting called a lot.
    """

    @property
    def time(self) -> float:
        """The absolute time of the simulation at this instant, in seconds."""
        ...

    def step(self) -> None:
        """Advances the simulation by one step. What a step involves and how much time it takes is defined by the implementation."""
        ...

    def snapshot(self) -> Self:
        """Returns a copy of the current simulation state."""
        ...

    def interpolate(self, other: Self, t: float) -> Self:
        """Returns a linear interpolation between two states.

        Args:
            other: An object acquired by piping the current object through an arbitrary chain of `.snapshot()` and `.step()` calls.
            t: A number between 0 and 1. 0 means return an object equivalent to `self`, and 1 means return an object equivalent to `other`.

        Returns:
            An object between `self` and `other` according to `t`.
            The returned object might be a copy of `self` or `other`,
            so it isn't safe to call `.step()` on it without first creating a `.snapshot()`.
        """
        ...

def generate_colorspace() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red: np.ndarray = np.zeros(6*255)
    green: np.ndarray = np.zeros(6*255)
    blue: np.ndarray = np.zeros(6*255)
    offset: int = 0
    for i in range(255):
        red[i+offset], green[i+offset], blue[i+offset] = 0, 255, i
    offset += 255
    for i in range(255):
        red[i+offset], green[i+offset], blue[i+offset] = 0, 255-i, 255
    offset += 255
    for i in range(255):
        red[i+offset], green[i+offset], blue[i+offset] = i, 0, 255
    offset += 255
    for i in range(255):
        red[i+offset], green[i+offset], blue[i+offset] = 255, 0, 255-i
    offset += 255
    for i in range(255):
        red[i+offset], green[i+offset], blue[i+offset] = 255, i, 0
    offset += 255
    for i in range(255):
        red[i+offset], green[i+offset], blue[i+offset] = 255-i, 255, 0
    return red, green, blue

class SimulationRenderer[T: SimulationImpl](ABC):
    """Renders the simulation inside a pygame window."""
    red: np.ndarray
    green: np.ndarray
    blue: np.ndarray

    def __post_init__(self) -> None:
        self.red, self.green, self.blue = generate_colorspace()

    @abstractmethod
    def draw(self, e: RenderEnvironment, state: T) -> None: ...


class SimulationRecorder[T: SimulationImpl](ABC):
    """Records the statistics of a simulation."""

    @abstractmethod
    def record(self, state: T) -> None: ...
