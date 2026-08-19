"""
tests/unit/_agent_safety_check.py

Shared helper: checks a module's ACTUAL imports and attribute/call usage
for network/socket/subprocess capability, via AST parsing -- not a naive
substring search, which would false-positive on docstrings that mention
these words while explaining their absence (as several src/agents/
modules deliberately do).
"""

import ast

FORBIDDEN_IMPORT_MODULES = {"socket", "subprocess", "urllib", "urllib2", "requests", "http.client", "ftplib", "telnetlib", "paramiko"}
FORBIDDEN_ATTR_CALLS = {("os", "system"), ("os", "popen"), ("os", "exec"), ("os", "spawn")}


def assert_no_io_capability(module_path: str) -> None:
    with open(module_path) as f:
        tree = ast.parse(f.read(), filename=module_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in FORBIDDEN_IMPORT_MODULES, f"{module_path}: forbidden import '{alias.name}'"
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in FORBIDDEN_IMPORT_MODULES, f"{module_path}: forbidden import-from '{node.module}'"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                pair = (node.func.value.id, node.func.attr)
                assert pair not in FORBIDDEN_ATTR_CALLS, f"{module_path}: forbidden call '{pair[0]}.{pair[1]}'"
        if isinstance(node, ast.Name) and node.id in ("eval", "exec"):
            assert False, f"{module_path}: forbidden builtin usage '{node.id}'"
