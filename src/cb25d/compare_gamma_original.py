# pyright: strict
from collections.abc import Callable
from itertools import product
from typing import cast

import cloudpickle  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from tqdm.contrib.concurrent import (
    process_map,  # pyright: ignore[reportUnknownVariableType]
)

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


def _run_single_simulation(
    value: tuple[int, float, float, bytes, int],
) -> tuple[float, float]:
    seed, att, ali, create_initial_state_b, steps_per_run = value
    create_initial_state = cast(
        CreateInitialState, cloudpickle.loads(create_initial_state_b)
    )
    run_batch_simulation(
        create_initial_state(att, ali, seed),
        rec := SimulationRecorderOriginal(skip_first_n=steps_per_run // 2),
        steps=steps_per_run,
    )
    return rec.dispersion, rec.polarization


def run_gamma_comparison_original(
    *,
    seed: int,
    att_vals: np.ndarray,
    ali_vals: np.ndarray,
    create_initial_state: CreateInitialState,
    runs_per_config: int,
    steps_per_run: int,
):
    n_runs = len(att_vals) * len(ali_vals) * runs_per_config
    seed *= n_runs
    statistics = np.zeros((len(att_vals), len(ali_vals), 2))
    create_initial_state_b = cloudpickle.dumps(create_initial_state)  # pyright: ignore[reportUnknownMemberType]
    for stats, (i, j, _) in zip(
        cast(
            list[tuple[float, float]],
            process_map(
                _run_single_simulation,
                [
                    (seed + i, *args)
                    for i, args in enumerate(
                        (att, ali, create_initial_state_b, steps_per_run)
                        for att in att_vals
                        for ali in ali_vals
                        for _ in range(runs_per_config)
                    )
                ],
            ),
        ),
        product(range(len(att_vals)), range(len(ali_vals)), range(runs_per_config)),
    ):
        statistics[i, j, :] += stats
    statistics /= runs_per_config
    return statistics
