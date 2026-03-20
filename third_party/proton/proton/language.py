from triton._C.libtriton import proton as triton_proton

from .flags import get_profiling_on


def _create_record(builder, is_start, name):
    if not get_profiling_on():
        return
    triton_proton.create_proton_record(builder.get_op_builder_capsule(), is_start, name)


def _extract_scope_name(context_expr):
    if not context_expr.args:
        raise ValueError("pl.scope requires a scope name")
    scope_name = context_expr.args[0]
    if getattr(scope_name, "value", None) is None or not isinstance(scope_name.value, str):
        raise ValueError("pl.scope currently requires a string literal name")
    return scope_name.value


def _handle_scope_with(generator, node):
    scope_name = _extract_scope_name(node.items[0].context_expr)
    _create_record(generator.builder, True, scope_name)
    generator.visit_compound_statement(node.body)
    _create_record(generator.builder, False, scope_name)
    return None


class scope:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


try:
    from triton.compiler.code_generator import WITH_DISPATCH

    WITH_DISPATCH[scope] = _handle_scope_with
except Exception:
    pass
