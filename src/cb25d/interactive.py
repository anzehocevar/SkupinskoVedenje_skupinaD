import importlib.resources
import math
from contextlib import closing, contextmanager
from fractions import Fraction
from queue import SimpleQueue
from threading import Event, Thread

import pygame
import pygame.freetype

import cb25d.ttf
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
            if self._should_simulate():
                self._current_state.step()
                self._current_t = self._current_state.time
                self._submitted_states.put(self._current_state.snapshot())

    @property
    def prerender_dt(self) -> float:
        return self._prerender_dt

    @prerender_dt.setter
    def prerender_dt(self, value: float):
        if value != self._prerender_dt:
            self._prerender_dt = value
            self._event.set()

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


def _draw_grid(e: RenderEnvironment):
    size = 20
    fgcolor = (200, 200, 200)

    w_left, w_top = e.s2w((0, 0))
    w_right, w_bottom = e.s2w(e.screen.get_size())

    log_scale = round(math.log10(max(w_right - w_left, w_bottom - w_top))) - 1
    log_scale_value = 10**log_scale
    n_decimals = max(0, -log_scale)

    w_left_now = math.ceil(w_left / log_scale_value) * log_scale_value
    while True:
        left_now = e.w2s((w_left_now, 0))[0]
        pygame.draw.line(
            e.screen, fgcolor, (left_now, 0), (left_now, e.screen.get_height())
        )
        s_mark_text, _ = e.font_ui.render(
            f"{w_left_now:0.{n_decimals}f}", fgcolor=fgcolor, size=size
        )
        e.screen.blit(
            s_mark_text,
            (
                left_now + 2,
                e.screen.get_height() - s_mark_text.get_height() - 1,
            ),
        )
        w_left_now_bkp = w_left_now
        w_left_now += log_scale_value
        if w_left_now == w_left_now_bkp or w_right <= w_left_now:
            break

    w_top_now = math.ceil(w_top / log_scale_value) * log_scale_value
    while True:
        top_now = e.w2s((0, w_top_now))[1]
        pygame.draw.line(
            e.screen, fgcolor, (0, top_now), (e.screen.get_width(), top_now)
        )
        s_mark_text, _ = e.font_ui.render(
            f"{w_top_now:0.{n_decimals}f}", fgcolor=fgcolor, size=size
        )
        e.screen.blit(s_mark_text, (1, top_now + 2))
        w_top_now_bkp = w_top_now
        w_top_now -= log_scale_value
        if w_top_now == w_top_now_bkp or w_bottom >= w_top_now:
            break


def _draw_clock(e: RenderEnvironment, t: float, timescale: Fraction, paused: bool):
    size = 24
    line_height = 28
    w_pad = line_height - size
    fgcolor = (255, 255, 255)
    bgcolor = (0, 0, 0, 200)

    h_last = e.screen.get_height()

    n_decimals = 2 + max(-2, round(-math.log10(timescale)))
    s_time_text, _ = e.font_ui.render(
        f"{t:0.{n_decimals}f}s", fgcolor=fgcolor, size=size
    )
    s_time_bg = pygame.Surface((s_time_text.get_width() + w_pad, line_height))
    s_time_bg.set_alpha(bgcolor[3])
    s_time_bg.fill(bgcolor)
    e.screen.blit(s_time_bg, (0, h_last - s_time_bg.get_height()))
    e.screen.blit(
        s_time_text,
        (
            (s_time_bg.get_width() - s_time_text.get_width()) / 2,
            h_last - (s_time_bg.get_height() + s_time_text.get_height()) / 2,
        ),
    )

    h_last -= line_height

    s_timescale_text, _ = e.font_ui.render(
        f"{timescale}x{' (paused)' if paused else ''}", fgcolor=fgcolor, size=size
    )
    s_timescale_bg = pygame.Surface((s_timescale_text.get_width() + w_pad, line_height))
    s_timescale_bg.set_alpha(bgcolor[3])
    s_timescale_bg.fill(bgcolor)
    e.screen.blit(s_timescale_bg, (0, h_last - s_timescale_bg.get_height()))
    e.screen.blit(
        s_timescale_text,
        (
            (s_timescale_bg.get_width() - s_timescale_text.get_width()) / 2,
            h_last - (s_timescale_bg.get_height() + s_timescale_text.get_height()) / 2,
        ),
    )


def run_interactive_simulation[T: SimulationImpl](
    impl: T,
    renderer: SimulationRenderer[T],
    center: tuple[float, float] = (0, 0),
    scale: float = 1.0,
) -> None:
    state_prev = impl.snapshot()
    state_next = state_prev

    with (
        _with_pygame(),
        closing(_Simulation(impl, 1, 2)) as simulation,
        importlib.resources.path(cb25d.ttf, "NotoSans.ttf") as p_font_ui,
    ):
        screen = pygame.display.set_mode((1280, 720), pygame.DOUBLEBUF)
        clock = pygame.time.Clock()
        running = True

        t = state_prev.time
        dt = None
        timescale = Fraction(1)
        paused = True

        e = RenderEnvironment(
            screen=screen,
            left=0.0,
            top=0.0,
            scale=scale,
            mouse_top=pygame.mouse.get_pos()[0],
            mouse_left=pygame.mouse.get_pos()[1],
            mouse_top_rel=0,
            mouse_left_rel=0,
            font_ui=pygame.freetype.Font(str(p_font_ui)),
        )
        e.recenter(center)
        e.font_ui.kerning = True

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
                                e.rescale(scale, (e.mouse_left, e.mouse_top))
                    case pygame.KEYDOWN:
                        match event.key:
                            case pygame.K_SPACE:
                                paused = not paused
                            case pygame.K_COMMA:
                                timescale /= 2
                            case pygame.K_PERIOD:
                                timescale *= 2
                            case pygame.K_SLASH:
                                timescale = Fraction(1)

            simulation.prerender_dt = timescale * 1.0

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

            if dt is not None and not paused:
                if dt > 0.1:
                    # Wtihout this part, the simulation could enter an infinite loop.
                    # If the simulation runs slower than real time, the window would just keep waiting for it to catch up.
                    # This freezes the simulation when the last frame took a while, allowing it to catch up.
                    dt = 0
                dt *= timescale
                t += dt
            while (t_next := state_next.time) < t:
                state_prev = state_next
                state_next = simulation.get_next_state()
            t_prev = state_prev.time
            t_lerp = (t - t_prev) / (t_next - t_prev) if t_prev != t_next else 0
            state_lerp = state_prev.interpolate(state_next, t_lerp)

            screen.fill("white")
            _draw_grid(e)
            renderer.draw(e, state_lerp)
            _draw_clock(e, t, timescale, paused)

            pygame.display.flip()
            dt = clock.tick() / 1000
