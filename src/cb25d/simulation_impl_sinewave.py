import math
from typing import Self

import pygame
from numba.experimental import jitclass

from cb25d.render_environment import RenderEnvironment
from cb25d.simulation_framework import SimulationRenderer


@jitclass
class SimulationImplSinewave:
    time: float

    def __init__(self, time: float):
        self.time = time

    def step(self) -> None:
        self.time += 1

    def snapshot(self):
        return SimulationImplSinewave(self.time)

    def interpolate(self, other: Self, t: float):
        return SimulationImplSinewave((1 - t) * self.time + t * other.time)


class SimulationRendererSinewave(SimulationRenderer[SimulationImplSinewave]):
    def draw(self, e: RenderEnvironment, state: SimulationImplSinewave):
        pygame.draw.circle(
            e.screen,
            "red",
            e.w2s((40, 60 + math.sin(state.time * math.pi) * 20)),
            40 / e.scale,
        )
