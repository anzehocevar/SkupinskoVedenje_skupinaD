from cb25d.simulation_framework import SimulationImpl, SimulationRecorder


def run_batch_simulation[T: SimulationImpl](
    impl: T,
    recorder: SimulationRecorder[T],
    *,
    time: float | None = None,
    steps: int | None = None,
) -> None:
    if time is None and steps is None:
        raise ValueError("attempted to run simulation with no stopping condition")

    current_step = 0
    current_state = impl

    while (time is None or current_state.time <= time) and (
        steps is None or current_step < steps
    ):
        recorder.record(current_state)
        current_step += 1
        current_state.step()
