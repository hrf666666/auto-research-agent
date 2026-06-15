"""Shared AST scanner for PyTorch model structure analysis — single source.

Previously, the "walk ast → find nn.Module subclasses → collect self.X
assignments → estimate params" idiom was copy-pasted 15+ times across
verifier.py, experiment_evaluator.py, simulation_sandbox.py, and
model_analyzer.py, with subtle divergences (e.g. one copy handles Conv3d,
another doesn't; the param estimator ignores keyword arguments, returning 0
for modern PyTorch style `Conv2d(out_channels=64, in_channels=3)`).

This module provides ONE scanner. All consumers import from here.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModuleInfo:
    """Structure of one nn.Module subclass discovered by the scanner."""
    name: str
    init_assigns: dict[str, str] = field(default_factory=dict)  # attr → call name
    forward_refs: set[str] = field(default_factory=set)         # self.X referenced in forward
    estimated_params: int = 0


@dataclass
class ModelStructure:
    """Full scan result for a model file."""
    modules: list[ModuleInfo] = field(default_factory=list)
    total_estimated_params: int = 0
    parse_error: Optional[str] = None
    param_estimation_reliable: bool = True  # False when AST estimate returns 0


def scan_model_file(source: str) -> ModelStructure:
    """Parse Python source and return the structure of all nn.Module subclasses.

    This is the SINGLE implementation replacing 15+ copies.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return ModelStructure(parse_error=str(e))

    modules = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not _is_nn_module(node):
            continue
        info = ModuleInfo(name=node.name)
        # Collect self.X = SomeCall() assignments from __init__
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                for stmt in ast.walk(item):
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == "self"):
                                attr_name = target.attr
                                call_name = _extract_call_name(stmt.value)
                                if call_name:
                                    info.init_assigns[attr_name] = call_name
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                for stmt in ast.walk(item):
                    if (isinstance(stmt, ast.Attribute)
                            and isinstance(stmt.value, ast.Name)
                            and stmt.value.id == "self"):
                        info.forward_refs.add(stmt.attr)

        info.estimated_params = _estimate_params(node)
        modules.append(info)

    total = sum(m.estimated_params for m in modules)
    return ModelStructure(
        modules=modules,
        total_estimated_params=total,
        param_estimation_reliable=total > 0,
    )


def _is_nn_module(class_node: ast.ClassDef) -> bool:
    """True if the class inherits from nn.Module."""
    for base in class_node.bases:
        if isinstance(base, ast.Attribute) and base.attr == "Module":
            return True
        if isinstance(base, ast.Name) and base.id == "Module":
            return True
    return False


def _extract_call_name(node) -> Optional[str]:
    """Extract the function/class name from an Assign value (e.g. Conv2d, Linear)."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        if isinstance(node.func, ast.Name):
            return node.func.id
    return None


def _get_arg(node: ast.Call, index: int, keyword: str):
    """Get a positional or keyword argument from a Call node.

    Handles BOTH positional (`Conv2d(3, 64, 3)`) and keyword
    (`Conv2d(in_channels=3, out_channels=64, kernel_size=3)`) styles.
    This fixes the bug where the old estimator ignored kwargs entirely,
    returning 0 params for modern PyTorch code.
    """
    # Try positional first
    if index < len(node.args):
        return node.args[index]
    # Then try keyword
    for kw in node.keywords:
        if kw.arg == keyword:
            return kw.value
    return None


def _const_value(node, default=0):
    """Extract an integer from an ast.Constant, else default."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return default


def _estimate_params(class_node: ast.ClassDef) -> int:
    """Estimate parameter count from __init__ layer definitions.

    Handles Conv2d/Conv3d/ConvTranspose2d/Linear/BatchNorm2d/GroupNorm/Embedding.
    Reads BOTH positional and keyword args (fixes the kwargs bug).
    """
    total = 0
    for item in class_node.body:
        if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
            continue
        for stmt in ast.walk(item):
            if not (isinstance(stmt, ast.Call)
                    and isinstance(stmt.func, ast.Attribute)):
                continue
            attr = stmt.func.attr

            if attr in ("Conv2d", "Conv3d", "ConvTranspose2d"):
                # Conv: in_ch, out_ch, kernel
                in_ch = _const_value(_get_arg(stmt, 0, "in_channels"))
                out_ch = _const_value(_get_arg(stmt, 1, "out_channels"))
                kernel = _const_value(_get_arg(stmt, 2, "kernel_size"), 3)
                if in_ch and out_ch:
                    total += in_ch * out_ch * kernel * kernel

            elif attr == "Linear":
                # Linear: in_features, out_features
                in_f = _const_value(_get_arg(stmt, 0, "in_features"))
                out_f = _const_value(_get_arg(stmt, 1, "out_features"))
                if in_f and out_f:
                    total += in_f * out_f + out_f  # weight + bias

            elif attr in ("BatchNorm2d", "BatchNorm3d", "LayerNorm"):
                num = _const_value(_get_arg(stmt, 0, "num_features"))
                if num:
                    total += num * 2  # weight + bias

            elif attr == "GroupNorm":
                num = _const_value(_get_arg(stmt, 1, "num_channels"))
                if num:
                    total += num * 2

            elif attr == "Embedding":
                num_emb = _const_value(_get_arg(stmt, 0, "num_embeddings"))
                dim = _const_value(_get_arg(stmt, 1, "embedding_dim"))
                if num_emb and dim:
                    total += num_emb * dim

    return total


def find_dead_branches(structure: ModelStructure) -> list[str]:
    """Find init assignments never referenced in forward (potential dead branches).

    Returns a list of attribute names that are assigned in __init__ but
    never used in forward() — they receive no gradients.
    """
    dead = []
    for module in structure.modules:
        for attr in module.init_assigns:
            if attr not in module.forward_refs:
                dead.append(f"{module.name}.{attr}")
    return dead
