from typing import Iterator
import numpy as np
import numpy.typing as npt
import src.simulation
import pygame
import time

DISPLAY_SIZE: tuple[int, int] = (500, 500)
ITERATIONS_BETWEEN_VISUAL_UPDATES: int = 100
COLOR_POSITION: tuple[int, int, int] = (0, 255, 0)
SIZE_POSITION: int = 1
COLOR_DIRECTION: tuple[int, int, int] = (0, 255, 0)
LENGTH_DIRECTION: float = 5.0
WIDTH_DIRECTION: int = 1
FPS: int = 60

def generate_colorspace() -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    red: npt.NDArray = np.zeros(6*255)
    green: npt.NDArray = np.zeros(6*255)
    blue: npt.NDArray = np.zeros(6*255)
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

def main():

    # Initialisation
    pygame.init()
    display: pygame.Surface = pygame.display.set_mode(DISPLAY_SIZE)
    clock: pygame.time.Clock = pygame.time.Clock()
    red, green, blue = generate_colorspace()
    simulation: Iterator[tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]] = src.simulation.run_with_groups()
    running: bool = True
    iter: int = 0
    paused: bool = True

    # Visualization and simulation loop
    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if paused:
            time.sleep(0.1)
            continue

        # Get positions and headings of next kick
        u_x, u_y, phi, group = next(simulation)

        # Visualize state every ITERATIONS_BETWEEN_VISUAL_UPDATES kicks
        if iter % ITERATIONS_BETWEEN_VISUAL_UPDATES == 0:
            groups: npt.NDArray = np.unique(group)
            index_colorspace: npt.NDArray = np.linspace(0, 6*255, len(groups), endpoint=False).astype(int)
            group_to_index: npt.NDArray = np.full(groups.max()+1, -1)
            group_to_index[groups] = np.arange(len(groups))
            display.fill((0, 0, 0))
            for x, y, dx, dy, ix in zip(u_x, u_y, np.cos(phi), np.sin(phi), group_to_index[group]):
                offset_x, offset_y = DISPLAY_SIZE[0]//2, DISPLAY_SIZE[1]//2
                dx, dy = LENGTH_DIRECTION*dx, LENGTH_DIRECTION*dy
                x, y = x+offset_x, y+offset_y
                color: tuple[int, int, int] = (red[index_colorspace[ix]], green[index_colorspace[ix]], blue[index_colorspace[ix]])  # type: ignore
                pygame.draw.circle(display, color, (x, y), SIZE_POSITION)
                pygame.draw.line(display, color, (x, y), (x+dx, y+dy), WIDTH_DIRECTION)
            pygame.display.flip()
            clock.tick(FPS)

        iter+=1
    pygame.quit()

if __name__ == "__main__":
    main()
