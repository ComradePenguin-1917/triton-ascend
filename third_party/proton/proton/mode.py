from dataclasses import dataclass, field
from typing import Optional, Union, List
from enum import Enum


class Optimize(Enum):
    TIMESHIFT = "time_shift"
    SCHED_STORES = "sched_stores"
    SCHED_BARRIERS = "sched_barriers"
    CLOCK32 = "clock32"

    def __str__(self):
        return self.value


optimizations = {
    "time_shift": Optimize.TIMESHIFT,
    "sched_stores": Optimize.SCHED_STORES,
    "sched_barriers": Optimize.SCHED_BARRIERS,
    "clock32": Optimize.CLOCK32,
}


@dataclass(frozen=True)
class BaseMode:
    name: str


@dataclass(frozen=True)
class InstrumentationMode(BaseMode):
    """Instrumentation mode with configurable buffer_size and optimizations.

    Args:
        buffer_size (int): Per-kernel profiling buffer size in bytes. 0 means
            default (4096). On Ascend, this is the only configurable option.
        optimizations (List[Optimize]): Optimization flags (time_shift, etc).
        sample_every_n (int): Profile every N-th block. 1 = all blocks
            (default), 2 = every 2nd block, 4 = every 4th block, etc.
            Reduces profiling memory and overhead for large grids.
    """
    buffer_size: int = 0
    optimizations: List[Optimize] = field(default_factory=list)
    sample_every_n: int = 1

    def __post_init__(self):
        values_str = getattr(self, "optimizations")
        if isinstance(values_str, str):
            values = [value.strip() for value in values_str.split(",")] if len(values_str) > 0 else []
            for value in values:
                if value not in optimizations:
                    raise ValueError(f"Unknown optimization: {value}")
            object.__setattr__(self, "optimizations", [optimizations[value] for value in values])
        if self.sample_every_n < 1:
            raise ValueError(f"sample_every_n must be >= 1, got {self.sample_every_n}")

    def __str__(self):
        optimizations_str = ",".join([str(opt) for opt in self.optimizations])
        return f"{self.name}:buffer_size={self.buffer_size}:optimizations={optimizations_str}:sample_every_n={self.sample_every_n}"


@dataclass(frozen=True)
class Default(InstrumentationMode):
    name: str = field(default="default", init=False)
