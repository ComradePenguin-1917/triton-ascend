# flake8: noqa
from .scope import scope, enter_scope, exit_scope
from .profile import (
    start,
    activate,
    deactivate,
    finalize,
    profile,
    DEFAULT_PROFILE_NAME,
    get_data_segment_bytes,
)
from .mode import Default, InstrumentationMode, BaseMode, Optimize
