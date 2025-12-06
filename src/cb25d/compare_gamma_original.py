# pyright: strict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import cache

import cloudpickle  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from tqdm.auto import tqdm

from cb25d.batch import run_batch_simulation
from cb25d.simulation_impl_original import (
    SimulationImplOriginal,
    SimulationRecorderOriginal,
)

type CreateInitialState = Callable[
    [
        float,  # Attraction
        float,  # Alignment
        int,  # Seed
    ],
    SimulationImplOriginal,
]


@cache
def _uncloudpickle(create_initial_state_b: bytes) -> CreateInitialState:
    return cloudpickle.loads(create_initial_state_b)


def _run_single_simulation(
    value: tuple[int, float, float, bytes, int],
) -> tuple[float, float, float]:
    seed, att, ali, create_initial_state_b, steps_per_run = value
    run_batch_simulation(
        _uncloudpickle(create_initial_state_b)(att, ali, seed),
        rec := SimulationRecorderOriginal(skip_first_n=steps_per_run // 2),
        steps=steps_per_run,
    )
    return rec.dispersion, rec.polarization, rec.milling


def run_gamma_comparison_original(
    *,
    seed: int,
    att_vals: np.ndarray,
    ali_vals: np.ndarray,
    create_initial_state: CreateInitialState,
    runs_per_config: int,
    steps_per_run: int,
):
    seed *= len(att_vals) * len(ali_vals) * runs_per_config
    statistics = np.zeros((len(att_vals), len(ali_vals), 3))
    create_initial_state_b = cloudpickle.dumps(create_initial_state)  # pyright: ignore[reportUnknownMemberType]

    futures = {
        (seed + i, *args): ij
        for i, (ij, args) in enumerate(
            ((i_att, i_ali), (att, ali, create_initial_state_b, steps_per_run))
            for i_att, att in enumerate(att_vals)
            for i_ali, ali in enumerate(ali_vals)
            for _ in range(runs_per_config)
        )
    }

    with ProcessPoolExecutor() as ppx, tqdm(total=len(futures)) as progress:
        futures = {ppx.submit(_run_single_simulation, k): v for k, v in futures.items()}
        for future in as_completed(futures):
            i, j = futures[future]
            statistics[i, j, :] += future.result()
            progress.update()
    statistics /= runs_per_config
    return statistics
