"""Compatibility shim for Python's stdlib ``types`` module.

This file used to hold MSA-specific data classes. Having a project file named
``types.py`` can shadow Python's standard library when scripts are launched from
``src/``. Export stdlib ``types`` symbols first, then expose the old MSA names
for backward compatibility.
"""

from __future__ import annotations

import sys as _sys


_stdlib_types_path = (
    f"{_sys.base_prefix}/lib/python{_sys.version_info.major}.{_sys.version_info.minor}/types.py"
)
_stdlib_namespace: dict[str, object] = {}
with open(_stdlib_types_path, "r", encoding="utf-8") as _stdlib_fh:
    exec(compile(_stdlib_fh.read(), _stdlib_types_path, "exec"), _stdlib_namespace)

for _name, _value in _stdlib_namespace.items():
    if _name.startswith("__") and _name not in {"__all__", "__doc__"}:
        continue
    globals()[_name] = _value

__all__ = list(_stdlib_namespace.get("__all__", []))

try:
    from src.msa_types import Document, ProtocolConstants
except Exception:  # pragma: no cover - supports importing from inside src/
    from msa_types import Document, ProtocolConstants

__all__.extend(["Document", "ProtocolConstants"])
