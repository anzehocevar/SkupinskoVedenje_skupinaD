from dataclasses import dataclass

from pygame import Surface
from pygame.freetype import Font


@dataclass
class RenderEnvironment:
    screen: Surface

    left: float
    top: float
    scale: float
    """World units per screen unit (lower value = higher zoom)."""

    mouse_top: int
    mouse_left: int
    mouse_top_rel: int
    mouse_left_rel: int

    font_ui: Font

    def i(self, xy: tuple[float, float]) -> tuple[int, int]:
        return int(xy[0]), int(xy[1])

    def w2s(self, xy: tuple[float, float]) -> tuple[float, float]:
        return (xy[0] - self.left) / self.scale, (xy[1] - self.top) / self.scale

    def s2w(self, xy: tuple[float, float]) -> tuple[float, float]:
        return self.left + xy[0] * self.scale, self.top + xy[1] * self.scale

    def rescale(self, new_scale: float, center: tuple[float, float]):
        center_world = self.s2w(center)
        self.scale = new_scale
        cx, cy = self.w2s(center_world)
        self.left += (cx - center[0]) * new_scale
        self.top += (cy - center[1]) * new_scale

    def recenter(self, center: tuple[float, float]):
        self.left, self.top = self.w2s(center)
        self.left -= self.screen.get_width() / 2 * self.scale
        self.top -= self.screen.get_height() / 2 * self.scale
