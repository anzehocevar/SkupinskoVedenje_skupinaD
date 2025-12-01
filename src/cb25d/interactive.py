from contextlib import closing, contextmanager
from queue import SimpleQueue
from threading import Event, Thread

import pygame

from cb25d.render_environment import RenderEnvironment
from cb25d.simulation_framework import SimulationImpl, SimulationRenderer


class _Simulation[T: SimulationImpl]:
    """Runs a `SimulationImpl` in a background thread and sends new states through a queue."""

    _current_state: T
    _current_t: float
    _submitted_t: float
    _submitted_states: SimpleQueue[T]
    _prerender_dt: float
    _prerender_states: int
    _event: Event
    _running: bool
    _thread: Thread

    def __init__(self, starting_state: T, prerender_dt: float, prerender_states: int):
        self._current_state = starting_state
        self._current_t = starting_state.time
        self._submitted_t = self._current_t
        self._submitted_states = SimpleQueue()
        self._submitted_states.put(starting_state.snapshot())
        self._prerender_dt = prerender_dt
        self._prerender_states = prerender_states
        self._event = Event()
        self._running = True
        self._thread = Thread(target=self._run)
        self._thread.start()

    def _should_simulate(self) -> bool:
        return (
            self._current_t < self._submitted_t + self._prerender_dt
            or self._submitted_states.qsize() < self._prerender_states
        )

    def _run(self):
        while self._running:
            if not self._should_simulate():
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
    with _with_pygame(), closing(_Simulation(impl, 1, 2)) as simulation:
        screen = pygame.display.set_mode((1280, 720), pygame.DOUBLEBUF)
        clock = pygame.time.Clock()
        running = True

        state_prev = impl
        state_next = impl
        t = state_prev.time
        dt = None
        timescale = 1

        e = RenderEnvironment(
            screen=screen,
            left=0.0,
            top=0.0,
            scale=1.0,
            mouse_top=pygame.mouse.get_pos()[0],
            mouse_left=pygame.mouse.get_pos()[1],
            mouse_top_rel=0,
            mouse_left_rel=0,
        )

        while running:
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        running = False
                    case pygame.MOUSEWHEEL:
                        e.rescale(
                            e.scale
                            * 2 ** (event.precise_y * (1 if event.flipped else -1)),
                            (e.mouse_left, e.mouse_top),
                        )
                    case pygame.MOUSEBUTTONDOWN:
                        match event.button:
                            case pygame.BUTTON_MIDDLE:
                                e.rescale(1, (e.mouse_left, e.mouse_top))

            mouse_left, mouse_top = pygame.mouse.get_pos()
            e.mouse_left_rel, e.mouse_top_rel = (
                mouse_left - e.mouse_left,
                mouse_top - e.mouse_top,
            )
            e.mouse_left, e.mouse_top = (
                mouse_left,
                mouse_top,
            )

            if pygame.mouse.get_pressed()[0]:
                e.left -= e.mouse_left_rel * e.scale
                e.top -= e.mouse_top_rel * e.scale

            if dt is not None:
                dt *= timescale
                if dt > 0.1:
                    # Wtihout this part, the simulation could enter an infinite loop.
                    # If the simulation runs slower than real time, the window would just keep waiting for it to catch up.
                    # This freezes the simulation when the last frame took a while, allowing it to catch up.
                    dt = 0
                t += dt
            while (t_next := state_next.time) < t:
                state_prev = state_next
                state_next = simulation.get_next_state()
            t_prev = state_prev.time
            t_lerp = (t - t_prev) / (t_next - t_prev) if t_prev != t_next else 0
            state_lerp = state_prev.interpolate(state_next, t_lerp)

            screen.fill("white")

            renderer.draw(e, state_lerp)

            pygame.display.flip()
            dt = clock.tick() / 1000
