try:
    from . import libtriton
except ImportError as e:
    import sys
    print(f"Warning: Failed to import libtriton: {e}", file=sys.stderr)

try:
    from . import libproton
except ImportError:
    pass

__all__ = ["libtriton", "libproton"]
