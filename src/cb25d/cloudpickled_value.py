# pyright: strict
from dataclasses import dataclass
from typing import Any

import cloudpickle  # type: ignore

_CACHE: dict[bytes, Any] = {}


@dataclass(slots=True)
class CloudpickledValue[T]:
    value: T
    cache: bool

    def __getstate__(self) -> object:
        return (
            cloudpickle.dumps(self.value),  # pyright: ignore[reportUnknownMemberType]
            self.cache,
        )

    def __setstate__(self, value: tuple[bytes, bool]):
        if value[1]:
            self.cache = value[1]
            if value[0] in _CACHE:
                self.value = _CACHE[value[0]]
            else:
                ret = cloudpickle.loads(value[0])
                _CACHE[value[0]] = ret
                self.value = ret
        self.value = cloudpickle.loads(value[0])
