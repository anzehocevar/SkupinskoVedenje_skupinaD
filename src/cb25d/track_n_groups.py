# pyright: strict
from collections.abc import Callable

import numpy as np

from cb25d.batch import run_batch_simulation, run_multiprocess_simulations
from cb25d.simulation_impl_original import (
    SimulationImplOriginal,
    SimulationRecorderOriginal,
)


def run_n_groups_original(
    *,
    seed: int,
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

    emergences: dict[str, tuple[float, float]] = {
        "Swarming": (0.6, 0.6),
        "Schooling": (0.22, 0.6),
        "Milling": (0.37, 0.2),
    }

    seed *= len(emergences) * runs_per_config
    statistics: dict[str, list[np.ndarray]] = {
        "Swarming": [],
        "Schooling": [],
        "Milling": [],
    }

    def run(seed: int, att: float, ali: float):
        state: SimulationImplOriginal = create_initial_state(att, ali, seed)
        run_batch_simulation(
            state,
            rec := SimulationRecorderOriginal(skip_first_n=0, use_groups=True),
            steps=steps_per_run,
        )
        if state.group_every_n_steps > 0 and rec.n_groups:
            return rec.n_groups[::state.group_every_n_steps]
        return rec.n_groups

    for (_, name), result in run_multiprocess_simulations(
        fn=run,
        args={
            (i, ij): (seed + i, *args)
            for i, (ij, args) in enumerate(
                (name, (att, ali))
                for name, (att, ali) in emergences.items()
                for _ in range(runs_per_config)
            )
        },
    ).items():
        statistics[name].append(result)

    return {k: np.array(v) for k, v in statistics.items()}

if __name__ == "__main__":
    from pathlib import Path
    from cb25d.simulation_impl_original import (
        SimulationImplOriginal,
        generate_initial_conditions,
    )
    def compute(k: int):
        statistics = run_n_groups_original(
            seed=0,
            create_initial_state=lambda att, ali, seed: SimulationImplOriginal(
                c_eta=0.8,
                c_gamma_ali=ali,
                c_gamma_att=att,
                c_gamma_rand=0.2,
                c_k=k,
                c_l_ali=3,
                c_tau_0=0.8,
                c_dist_critical=4*3,
                c_dist_merge=min(3, 3),
                **generate_initial_conditions(
                    seed=0,
                    n=100,
                    l_att=3,
                ),
                group_every_n_steps=100,
            ),
            runs_per_config=1000,
            steps_per_run=40000 * 100,
        )
        p_base = Path("results/original/n_groups")
        p = p_base / f"k={k}"
        p.mkdir(parents=True, exist_ok=True)
        for em, result in statistics.items():
            np.save(p / f"{em}.npy", result)
    compute(k=1)
    
