from contextlib import closing, contextmanager
from queue import Queue
from threading import Event, Thread

import pygame

from cb25d.simulation_framework import SimulationImpl, SimulationRenderer


class _Simulation[T: SimulationImpl]:
    """Runs a `SimulationImpl` in a background thread and sends new states through a queue."""

    _current_state: T
    _current_t: float
    _submitted_t: float
    _submitted_states: Queue
    _prerender_dt: float
    _event: Event
    _running: bool
    _thread: Thread

    def __init__(self, starting_state: T, prerender_dt: float):
        self._current_state = starting_state
        self._current_t = starting_state.time
        self._submitted_t = self._current_t
        self._submitted_states = Queue()
        self._submitted_states.put(starting_state.snapshot())
        self._prerender_dt = prerender_dt
        self._event = Event()
        self._running = True
        self._thread = Thread(target=self._run)
        self._thread.start()

    def _run(self):
        while self._running:
            if self._submitted_t + self._prerender_dt <= self._current_t:
                self._event.wait()
                continue
            self._current_state.step()
            self._current_t = self._current_state.time
            self._submitted_states.put(self._current_state.snapshot())

    def get_next_state(self) -> T:
        state = self._submitted_states.get()
        self._submitted_t = state.time
        self._event.set()
        return state

    def close(self):
        self._running = False
        self._event.set()
        self._thread.join()


@contextmanager
def _with_pygame():
    pygame.init()
    try:
        yield
    finally:
        pygame.quit()


def run_interactive_simulation[T: SimulationImpl](
    impl: T,
    renderer: SimulationRenderer[T],
) -> None:
    with _with_pygame(), closing(_Simulation(impl, 2)) as simulation:
        screen = pygame.display.set_mode((1280, 720))
        clock = pygame.time.Clock()
        running = True
        state_prev = impl
        state_next = impl
        t = state_prev.time
        dt = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if dt is not None:
                if dt > 0.1:
                    # Wtihout this part, the simulation could enter an infinite loop.
                    # If the simulation runs slower than real time, the window would just keep waiting for it to catch up.
                    # This freezes the simulation when the last frame took a while, allowing it to catch up.
                    print("Stutter")
                    dt = 0
                t += dt
            while (t_next := state_next.time) < t:
                state_prev = state_next
                state_next = simulation.get_next_state()
            t_prev = state_prev.time
            t_lerp = (t - t_prev) / (t_next - t_prev) if t_prev != t_next else 0
            state_lerp = state_prev.interpolate(state_next, t_lerp)

            screen.fill("white")
            renderer.draw(screen, state_lerp)

            pygame.display.flip()
            dt = clock.tick() / 1000
