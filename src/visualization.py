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

def main():

    # Initialisation
    pygame.init()
    display: pygame.Surface = pygame.display.set_mode(DISPLAY_SIZE)
    clock: pygame.time.Clock = pygame.time.Clock()
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
            display.fill((0, 0, 0))
            for x, y, dx, dy in zip(u_x, u_y, np.cos(phi), np.sin(phi)):
                offset_x, offset_y = DISPLAY_SIZE[0]//2, DISPLAY_SIZE[1]//2
                dx, dy = LENGTH_DIRECTION*dx, LENGTH_DIRECTION*dy
                x, y = x+offset_x, y+offset_y
                pygame.draw.circle(display, COLOR_POSITION, (x, y), SIZE_POSITION)
                pygame.draw.line(display, COLOR_DIRECTION, (x, y), (x+dx, y+dy), WIDTH_DIRECTION)
            pygame.display.flip()
            clock.tick(FPS)

        iter+=1
    pygame.quit()

if __name__ == "__main__":
    main()
