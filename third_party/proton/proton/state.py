"""
Global state for proton instrumentation compile-time parameters.

Mirrors NV's `triton.knobs.compilation.instrumentation_mode` approach:
proton.start() writes config here, JITFunction.run() reads it and injects
into kwargs → options → cache key. When profiling parameters change, the
cache key changes naturally, triggering recompilation without env vars.
"""

from typing import Dict

_compile_opts: Dict[str, int] = {}


def set_compile_option(key: str, value: int) -> None:
    _compile_opts[key] = value


def get_compile_options() -> Dict[str, int]:
    return dict(_compile_opts)


def clear_compile_options() -> None:
    _compile_opts.clear()
