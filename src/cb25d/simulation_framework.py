# pyright: strict
"""A high level description of the simulation logic.

Defined separately to make things more readable.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Self

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


class SimulationRenderer[T: SimulationImpl](ABC):
    """Renders the simulation inside a pygame window."""

    @abstractmethod
    def draw(self, e: RenderEnvironment, state: T) -> None: ...
