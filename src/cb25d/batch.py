# pyright: strict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm.auto import tqdm

from cb25d.cloudpickled_value import CloudpickledValue
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


def _run_fn[**A, R](
    fn_cp: CloudpickledValue[Callable[A, R]],
    *args: A.args,
    **kwargs: A.kwargs,
) -> R:
    """Wrapper around CloudpickledValue with a callable.

    Used since multiprocessing uses regular pickle,
    and regular pickle can't pickle functions defined in a notebook.
    """
    return fn_cp.value(*args, **kwargs)


def run_multiprocess_simulations[CorrelationId, *A, R](
    *,
    fn: Callable[[*A], R],
    args: dict[CorrelationId, tuple[*A]],
) -> dict[CorrelationId, R]:
    fn_cp = CloudpickledValue(fn, True)
    ret: dict[CorrelationId, R] = {}

    with (
        ProcessPoolExecutor() as ppx,
        tqdm(total=len(args), smoothing=0) as progress,
    ):
        futures = {ppx.submit(_run_fn, fn_cp, *args_): c for c, args_ in args.items()}
        for future in as_completed(futures):
            ret[futures[future]] = future.result()
            progress.update()

    return ret
