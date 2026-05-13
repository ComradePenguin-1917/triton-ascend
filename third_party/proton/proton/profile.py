import functools
import os
import math
from pathlib import Path
from typing import Optional, Union

import triton

from triton._C.libproton import proton as libproton

from .flags import is_command_line, set_profiling_off, set_profiling_on
from .hook import register_triton_hook, unregister_triton_hook
from .mode import BaseMode, InstrumentationMode
from . import state

DEFAULT_PROFILE_NAME = "proton"

DEFAULT_DATA_SEGMENT_BYTES = 4096
_current_data_segment_bytes = DEFAULT_DATA_SEGMENT_BYTES
_current_sample_every_n = 1


def get_data_segment_bytes() -> int:
    return _current_data_segment_bytes


def get_sample_every_n() -> int:
    return _current_sample_every_n


def _select_backend() -> str:
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend == "cuda":
        return "cupti"
    if backend == "hip":
        return "roctracer"
    if backend in ("ascend", "npu"):
        return "npu"
    raise ValueError("No backend is available for the current target.")


def _get_backend_default_path(backend: str) -> str:
    if backend == "cupti":
        return str(Path(__file__).resolve().parents[2] / "backends" / "nvidia" / "lib" / "cupti")
    return ""


def _resolve_backend(backend: str) -> tuple[str, str, str]:
    if backend in ("npu", "ascend"):
        return "instrumentation", "", "npu"
    if backend == "npu-native":
        return "npu", "", ""
    if backend == "instrumentation":
        return "instrumentation", "", ""
    return backend, _get_backend_default_path(backend), ""


def _check_env(backend: str) -> None:
    if backend == "roctracer":
        hip_device_envs = ["HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
        for env in hip_device_envs:
            if os.getenv(env) is not None:
                raise ValueError(
                    f"Proton does not work when the environment variable {env} is set on AMD GPUs. Please unset it and use `ROCR_VISIBLE_DEVICES` instead"
                )


def start(
    name: Optional[str] = None,
    *,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    hook: Optional[str] = None,
    mode: Optional[Union[str, BaseMode]] = None,
):
    if is_command_line():
        return

    if name is None:
        name = DEFAULT_PROFILE_NAME

    if backend is None:
        backend = _select_backend()

    _check_env(backend)

    profiler_name, profiler_path, default_mode = _resolve_backend(backend)

    data_segment_bytes = None
    if mode is not None:
        if isinstance(mode, BaseMode):
            opts = []
            if isinstance(mode, InstrumentationMode):
                data_segment_bytes = mode.buffer_size if mode.buffer_size > 0 else DEFAULT_DATA_SEGMENT_BYTES
                opts.append(f"buffer_size={mode.buffer_size}")
                optimizations_str = ",".join([str(opt) for opt in mode.optimizations])
                opts.append(f"optimizations={optimizations_str}")
                opts.append(f"sample_every_n={mode.sample_every_n}")
            mode_str = f"{default_mode}:{':'.join(opts)}" if opts else default_mode
        else:
            mode_str = mode
    else:
        mode_str = default_mode

    if data_segment_bytes is not None:
        global _current_data_segment_bytes
        _current_data_segment_bytes = data_segment_bytes

    if mode is not None and isinstance(mode, InstrumentationMode):
        global _current_sample_every_n
        _current_sample_every_n = mode.sample_every_n

    state.set_compile_option("proton_data_segment_bytes", _current_data_segment_bytes)
    state.set_compile_option("proton_sample_every_n", _current_sample_every_n)

    set_profiling_on()
    if hook == "triton":
        register_triton_hook()
    return libproton.start(name, context, data, profiler_name, profiler_path, mode_str)


def activate(session: Optional[int] = 0) -> None:
    if is_command_line() and session != 0:
        raise ValueError("Only one session can be activated when running from the command line.")
    libproton.activate(session)


def deactivate(session: Optional[int] = 0) -> None:
    if is_command_line() and session != 0:
        raise ValueError("Only one session can be deactivated when running from the command line.")
    libproton.deactivate(session)


def finalize(session: Optional[int] = None, output_format: str = "hatchet") -> None:
    if session is None:
        set_profiling_off()
        state.clear_compile_options()
        libproton.finalize_all(output_format)
        unregister_triton_hook()
        return

    if is_command_line() and session != 0:
        raise ValueError("Only one session can be finalized when running from the command line.")
    libproton.finalize(session, output_format)


def _profiling(
    func,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    hook: Optional[str] = None,
    mode: Optional[Union[str, BaseMode]] = None,
):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        session = start(name, context=context, data=data, backend=backend, hook=hook, mode=mode)
        ret = func(*args, **kwargs)
        deactivate(session)
        return ret

    return wrapper


def profile(
    func=None,
    *,
    name: Optional[str] = None,
    context: Optional[str] = "shadow",
    data: Optional[str] = "tree",
    backend: Optional[str] = None,
    hook: Optional[str] = None,
    mode: Optional[Union[str, BaseMode]] = None,
):
    if func is None:
        def decorator(f):
            return _profiling(f, name=name, context=context, data=data, backend=backend, hook=hook, mode=mode)

        return decorator
    return _profiling(func, name=name, context=context, data=data, backend=backend, hook=hook, mode=mode)
