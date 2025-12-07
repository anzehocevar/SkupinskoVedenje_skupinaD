# pyright: strict
from collections.abc import Callable

import numpy as np

from cb25d.batch import run_batch_simulation, run_multiprocess_simulations
from cb25d.simulation_impl_original import (
    SimulationImplOriginal,
    SimulationRecorderOriginal,
)


def run_gamma_comparison_original(
    *,
    seed: int,
    att_vals: np.ndarray,
    ali_vals: np.ndarray,
    create_initial_state: Callable[
        [
            float,  # Attraction
            float,  # Alignment
            int,  # Seed
        ],
        SimulationImplOriginal,
    ],
    runs_per_config: int,
    steps_per_run: int,
):
    seed *= len(att_vals) * len(ali_vals) * runs_per_config
    statistics = np.zeros((len(att_vals), len(ali_vals), 3))

    def run(seed: int, att: float, ali: float):
        run_batch_simulation(
            create_initial_state(att, ali, seed),
            rec := SimulationRecorderOriginal(skip_first_n=steps_per_run // 2),
            steps=steps_per_run,
        )
        return rec.dispersion, rec.polarization, rec.milling

    for (_, i, j), result in run_multiprocess_simulations(
        fn=run,
        args={
            (i, *ij): (seed + i, *args)
            for i, (ij, args) in enumerate(
                ((i_att, i_ali), (att, ali))
                for i_att, att in enumerate(att_vals)
                for i_ali, ali in enumerate(ali_vals)
                for _ in range(runs_per_config)
            )
        },
    ).items():
        statistics[i, j, :] += result

    statistics /= runs_per_config
    return statistics
