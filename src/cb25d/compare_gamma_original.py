# pyright: strict
from collections.abc import Callable

import numpy as np

from cb25d.batch import run_batch_simulation
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
    n_runs = len(att_vals) * len(ali_vals) * runs_per_config
    seed *= n_runs
    statistics = np.zeros((len(att_vals), len(ali_vals), 2))
    for i_att, att in enumerate(att_vals):
        for i_ali, ali in enumerate(ali_vals):
            for _ in range(runs_per_config):
                run_batch_simulation(
                    create_initial_state(att, ali, seed),
                    rec := SimulationRecorderOriginal(skip_first_n=steps_per_run // 2),
                    steps=steps_per_run,
                )
                statistics[i_att, i_ali, 0] += rec.dispersion
                statistics[i_att, i_ali, 1] += rec.polarization
                print(f"{seed}/{n_runs}")
                seed += 1
    statistics /= runs_per_config
    return statistics
