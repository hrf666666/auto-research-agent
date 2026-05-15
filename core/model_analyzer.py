"""
AutoResearcher Model Analyzer Mixin

Extracted from ToolRegistry for maintainability.
Contains all model architecture analysis methods:
- AST-based structural analysis (branches, channels, parameters)
- Data flow graph tracing
- Information bottleneck detection
- Gradient path analysis
- Structural soundness scoring
- Domain assumption analysis
- Idea-architecture alignment
- Decoder adequacy analysis
- Runtime model probe (generate & run diagnostic scripts)
- Ablation experiment design
"""

import ast
import re
import sys
import json
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger("autoresearcher.model_analyzer")


class ModelAnalyzerMixin:
    """Model analysis methods mixin for ToolRegistry.

    All methods assume they have access to:
    - self.workspace: Path to the workspace directory
    - self._resolve_workspace_path(path): resolve relative paths
    """

    def _analyze_model_ast(self, tree, content: str) -> dict:
        """Analyze model structure from AST."""
        result = {
            "model_classes": [],
            "total_params_estimate": 0,
            "branch_analysis": {},
            "warnings": [],
        }

        # Find nn.Module subclasses
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == "Module":
                        branch_info = self._extract_branch_info(node, tree)
                        if branch_info:
                            result["model_classes"].append(node.name)
                            result["branch_analysis"] = branch_info
                        break

        # Estimate total parameters
        result["total_params_estimate"] = self._estimate_params_from_ast(tree)

        # Check branch balance
        if result["branch_analysis"]:
            total_ch = sum(b.get("output_channels", 0)
                          for b in result["branch_analysis"].values())
            for name, info in result["branch_analysis"].items():
                ch = info.get("output_channels", 0)
                ratio = ch / max(total_ch, 1)
                info["channel_ratio"] = round(ratio, 3)
                if ratio < 0.10:
                    result["warnings"].append(
                        f"Branch '{name}' has only {ratio*100:.0f}% of fusion channels — "
                        f"it will be gradient-drowned by other branches"
                    )
                elif ratio < 0.20:
                    result["warnings"].append(
                        f"Branch '{name}' has {ratio*100:.0f}% of fusion channels — "
                        f"may need verification via ablation"
                    )

        return result

    def _extract_branch_info(self, class_node, tree=None) -> dict:
        """Extract branch information from a model class AST node.

        Enhanced to handle custom class instantiation (e.g., self.fft_branch = AngularFFTBranch(out_channels=32))
        by tracing into the custom class definition to find the final output channel count.
        """
        import ast

        branches = {}
        # Build a lookup of class definitions in the same file for tracing custom classes
        class_defs = {}
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_defs[node.name] = node

        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        name = target.attr
                        if any(kw in name.lower() for kw in ("conv", "stream", "branch", "encoder", "path", "head", "center")):
                            # Try to estimate output channels
                            if isinstance(node.value, ast.Call):
                                ch = self._extract_channels_from_call(node.value)
                                if ch:
                                    branches[name] = {"output_channels": ch}
                                else:
                                    # Try custom class: look for out_channels kwarg or trace into class def
                                    ch = self._extract_channels_from_custom_class(node.value, class_defs)
                                    if ch:
                                        branches[name] = {"output_channels": ch}
        return branches

    def _extract_channels_from_custom_class(self, call_node, class_defs: dict) -> int:
        """Extract output channels from a custom class instantiation.

        Strategies:
        1. Check for out_channels/out_ch keyword argument in the constructor call
        2. Find the class definition and look for the LAST Conv2d's out_channels
        """
        import ast

        # Strategy 1: keyword argument in constructor call
        if isinstance(call_node, ast.Call):
            for kw in call_node.keywords:
                if kw.arg in ("out_channels", "out_ch", "channels", "ch"):
                    if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                        return kw.value.value

            # Strategy 2: trace into class definition
            class_name = ""
            if isinstance(call_node.func, ast.Name):
                class_name = call_node.func.id
            elif isinstance(call_node.func, ast.Attribute):
                class_name = call_node.func.attr

            if class_name and class_name in class_defs:
                class_def = class_defs[class_name]
                # Find the LAST Conv2d in this class definition
                last_conv_out = 0
                for child in ast.walk(class_def):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        if child.func.attr == "Conv2d" and len(child.args) >= 2:
                            out_ch = child.args[1]
                            if isinstance(out_ch, ast.Constant) and isinstance(out_ch.value, int):
                                last_conv_out = out_ch.value
                if last_conv_out:
                    return last_conv_out

        return 0

    def _extract_channels_from_call(self, call_node) -> int:
        """Try to extract output channels from a Conv2d or Sequential call."""
        import ast
        if isinstance(call_node, ast.Call):
            if isinstance(call_node.func, ast.Attribute):
                if call_node.func.attr == "Conv2d" and call_node.args:
                    # Conv2d(in_ch, out_ch, ...)
                    if len(call_node.args) >= 2:
                        out_ch = call_node.args[1]
                        if isinstance(out_ch, ast.Constant):
                            return out_ch.value
            elif isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "Sequential":
                # Sequential contains nested modules
                for arg in call_node.args:
                    ch = self._extract_channels_from_call(arg)
                    if ch:
                        return ch
        return 0

    def _estimate_params_from_ast(self, tree) -> int:
        """Rough parameter estimate from AST."""
        import ast
        total = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "Conv2d" and len(node.args) >= 2:
                    in_ch = node.args[0]
                    out_ch = node.args[1]
                    kernel = node.args[2] if len(node.args) > 2 else ast.Constant(3)
                    if isinstance(in_ch, ast.Constant) and isinstance(out_ch, ast.Constant):
                        ic, oc = in_ch.value, out_ch.value
                        if isinstance(kernel, ast.Constant):
                            k = kernel.value if isinstance(kernel.value, int) else 3
                        else:
                            k = 3
                        total += ic * oc * k * k
                elif node.func.attr in ("Linear", "BatchNorm2d", "GroupNorm"):
                    if node.args:
                        ch = node.args[0]
                        if isinstance(ch, ast.Constant):
                            total += ch.value * 2  # rough estimate
        return total

    def _parse_target_size(self, size_str: str) -> tuple:
        try:
            parts = size_str.lower().split("x")
            return int(parts[0]), int(parts[1])
        except:
            return 256, 256

    def _estimate_gpu_memory(self, analysis: dict, h: int, w: int) -> dict:
        """Estimate GPU memory usage."""
        total_ch = sum(b.get("output_channels", 0)
                      for b in analysis.get("branch_analysis", {}).values())
        # Rough: each feature map = ch * h * w * 4 bytes (float32)
        # Plus gradients, optimizer states (~3x)
        feature_maps_mb = total_ch * h * w * 4 / (1024 * 1024)
        total_mb = feature_maps_mb * 4  # forward + backward + optimizer + activation cache
        return {
            "feature_maps_mb": round(feature_maps_mb, 1),
            "estimated_total_mb": round(total_mb, 1),
            "target_size": f"{h}x{w}",
        }

    def _analyze_data_feasibility(self, manifest_path: Path, analysis: dict) -> dict:
        """Check if available data can support the model complexity."""
        import json
        manifest = json.loads(manifest_path.read_text())
        datasets = manifest.get("datasets", {})
        total_train = 0
        total_val = 0
        domain_counts = {}
        for ds_name, ds_info in datasets.items():
            scenes = ds_info.get("scenes", {})
            for sname, sdata in scenes.items():
                split = sdata.get("split", "unknown")
                if split == "train":
                    total_train += 1
                elif split == "val":
                    total_val += 1
                # Dynamic domain grouping from manifest
                group = ds_info.get("type", "")
                if not group:
                    name_lower = ds_name.lower()
                    # Generic heuristic: common CV domain naming patterns
                    if "non-lambertian" in name_lower or "specular" in name_lower:
                        group = "Non-Lambertian"
                    elif "lambertian" in name_lower or "diffuse" in name_lower:
                        group = "Lambertian"
                    elif "urban" in name_lower or "mixed" in name_lower:
                        group = "Mixed"
                    else:
                        group = ds_name[:30]
                domain_counts[group] = domain_counts.get(group, 0) + 1

        params = analysis.get("total_params_estimate", 0)
        ratio = total_train / max(params / 1000, 1)  # samples per K params

        warnings = []
        if ratio < 10:
            warnings.append(
                f"CRITICAL: Only {ratio:.1f} training samples per K parameters. "
                f"Severe overfitting risk. Consider reducing model size."
            )
        elif ratio < 50:
            warnings.append(
                f"WARNING: {ratio:.1f} samples per K parameters. "
                f"Monitor for overfitting."
            )

        for domain, count in domain_counts.items():
            if count < 10:
                warnings.append(
                    f"Domain '{domain}' has only {count} scenes — "
                    f"metrics for this domain will be unreliable"
                )

        return {
            "total_train_scenes": total_train,
            "total_val_scenes": total_val,
            "domain_counts": domain_counts,
            "samples_per_k_params": round(ratio, 1),
            "warnings": warnings,
        }

    # ── Deep Model Analysis Methods ──

    def _analyze_data_flow(self, tree, content: str) -> dict:
        """Analyze data flow through the model: input → intermediate → output.

        Traces how tensor dimensions change through the forward pass by analyzing:
        - Module assignments in __init__ (what layers exist)
        - Forward method operations (how data flows through layers)
        - Concatenation/addition points (where branches merge)
        - Dimension changes at each stage

        Returns a data flow graph with nodes (modules) and edges (tensor connections).
        """
        import ast

        data_flow = {
            "input_modules": [],       # Modules that process raw input
            "processing_stages": [],    # Ordered processing stages in forward()
            "fusion_points": [],        # Where branches are combined (cat/add)
            "output_modules": [],       # Final processing before output
            "flow_description": [],     # Human-readable flow description
            "warnings": [],
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Check if this is an nn.Module subclass
            is_module = False
            for base in node.bases:
                if isinstance(base, ast.Attribute) and base.attr == "Module":
                    is_module = True
                    break
            if not is_module:
                continue

            # Analyze __init__ to find module declarations
            init_modules = {}  # name → type
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for stmt in ast.walk(item):
                        if isinstance(stmt, ast.Assign):
                            for target in stmt.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                    if target.value.id == "self":
                                        mod_type = self._get_call_name(stmt.value)
                                        if mod_type:
                                            init_modules[target.attr] = mod_type

                # Analyze forward() to trace data flow
                if isinstance(item, ast.FunctionDef) and item.name == "forward":
                    self._trace_forward_flow(item, init_modules, data_flow)

            break  # Only analyze the first nn.Module class

        return data_flow

    def _get_call_name(self, node) -> str:
        """Extract the class/function name from a Call AST node."""
        import ast
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                return node.func.attr
            elif isinstance(node.func, ast.Name):
                return node.func.id
        return ""

    def _trace_forward_flow(self, forward_func: ast.FunctionDef, init_modules: dict, data_flow: dict):
        """Trace data flow through the forward() method."""
        import ast

        # Track variable assignments to understand data transformations
        var_types = {}  # variable_name → what operation created it
        stage_counter = 0

        for stmt in ast.walk(forward_func):
            # Detect tensor operations: x = self.module(x), x = torch.cat([...]), x = x + y
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
                if not targets or not isinstance(targets[0], ast.Name):
                    continue
                var_name = targets[0].id

                # Case 1: self.module(x) — calling a declared module
                if isinstance(stmt.value, ast.Call):
                    call = stmt.value
                    if isinstance(call.func, ast.Attribute):
                        if isinstance(call.func.value, ast.Name) and call.func.value.id == "self":
                            attr_name = call.func.attr
                            mod_type = init_modules.get(attr_name, "Unknown")
                            stage_counter += 1
                            stage = {
                                "stage": stage_counter,
                                "variable": var_name,
                                "module": attr_name,
                                "module_type": mod_type,
                            }
                            # Check for dimension info in args
                            data_flow["processing_stages"].append(stage)
                            var_types[var_name] = f"self.{attr_name}({mod_type})"

                        # Detect torch.cat — fusion point
                        elif call.func.attr == "cat":
                            stage_counter += 1
                            cat_sources = self._extract_cat_sources(call)
                            fusion = {
                                "stage": stage_counter,
                                "variable": var_name,
                                "operation": "concatenation",
                                "sources": cat_sources,
                            }
                            data_flow["fusion_points"].append(fusion)
                            data_flow["processing_stages"].append(fusion)
                            var_types[var_name] = f"torch.cat({', '.join(cat_sources)})"

            # Detect addition: z = x + y (fusion via addition)
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.op, ast.Add) and isinstance(stmt.target, ast.Name):
                    # x += y — this is a skip connection or addition fusion
                    right_var = self._extract_var_name(stmt.value)
                    if right_var:
                        stage_counter += 1
                        fusion = {
                            "stage": stage_counter,
                            "variable": stmt.target.id,
                            "operation": "addition",
                            "sources": [stmt.target.id, right_var],
                        }
                        data_flow["fusion_points"].append(fusion)

        # Generate human-readable flow description
        if data_flow["processing_stages"]:
            data_flow["flow_description"].append(
                f"Model has {len(data_flow['processing_stages'])} processing stages, "
                f"{len(data_flow['fusion_points'])} fusion points."
            )
        if data_flow["fusion_points"]:
            for fp in data_flow["fusion_points"]:
                op = fp.get("operation", "unknown")
                sources = fp.get("sources", [])
                data_flow["flow_description"].append(
                    f"Fusion via {op}: {', '.join(sources)} → {fp.get('variable', '?')}"
                )

        # Warn if no fusion points but multiple branches in __init__
        branch_modules = [name for name, mtype in init_modules.items()
                         if any(kw in name.lower() for kw in ("branch", "stream", "encoder", "path"))]
        if len(branch_modules) > 1 and not data_flow["fusion_points"]:
            data_flow["warnings"].append(
                f"DETECTED {len(branch_modules)} branch-like modules ({', '.join(branch_modules)}) "
                f"but NO fusion points found in forward(). Branches may not be properly combined."
            )

    def _extract_cat_sources(self, call_node) -> list:
        """Extract source variable names from a torch.cat() call."""
        import ast
        sources = []
        if call_node.args and isinstance(call_node.args[0], ast.List):
            for elt in call_node.args[0].elts:
                name = self._extract_var_name(elt)
                if name:
                    sources.append(name)
        return sources

    def _extract_var_name(self, node) -> str:
        """Extract variable name from an AST node."""
        import ast
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._extract_var_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return self._extract_var_name(node.value)
        return ""

    def _detect_information_bottlenecks(self, tree, data_flow: dict) -> dict:
        """Detect information bottlenecks where channels compress too aggressively.

        A bottleneck occurs when:
        1. A branch has many input channels but outputs very few (compression ratio > 8:1)
        2. Multiple branches merge but one dominates the channel count
        3. An intermediate layer has fewer channels than needed to represent the information
        """
        import ast

        bottlenecks = {
            "detected": [],
            "warnings": [],
        }

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            is_module = False
            for base in node.bases:
                if isinstance(base, ast.Attribute) and base.attr == "Module":
                    is_module = True
                    break
            if not is_module:
                continue

            # Collect all Conv2d/Linear channel pairs
            # Walk ALL Conv2d calls including those inside nn.Sequential
            channel_changes = []
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if child.func.attr == "Conv2d" and len(child.args) >= 2:
                        in_ch = self._get_constant_value(child.args[0])
                        out_ch = self._get_constant_value(child.args[1])
                        if in_ch and out_ch:
                            ratio = in_ch / max(out_ch, 1)
                            channel_changes.append({
                                "type": "Conv2d",
                                "in_channels": in_ch,
                                "out_channels": out_ch,
                                "compression_ratio": round(ratio, 1),
                            })
                            if ratio > 8:
                                bottlenecks["detected"].append({
                                    "type": "aggressive_compression",
                                    "in_channels": in_ch,
                                    "out_channels": out_ch,
                                    "compression_ratio": round(ratio, 1),
                                    "severity": "high" if ratio > 16 else "medium",
                                    "description": (
                                        f"Conv2d compresses {in_ch} → {out_ch} channels "
                                        f"(ratio {ratio:.0f}:1). Information loss likely."
                                    ),
                                })
                    elif child.func.attr == "Linear" and len(child.args) >= 2:
                        in_f = self._get_constant_value(child.args[0])
                        out_f = self._get_constant_value(child.args[1])
                        if in_f and out_f:
                            ratio = in_f / max(out_f, 1)
                            if ratio > 8:
                                bottlenecks["detected"].append({
                                    "type": "aggressive_compression",
                                    "in_features": in_f,
                                    "out_features": out_f,
                                    "compression_ratio": round(ratio, 1),
                                    "severity": "high" if ratio > 16 else "medium",
                                    "description": (
                                        f"Linear compresses {in_f} → {out_f} features "
                                        f"(ratio {ratio:.0f}:1). Information loss likely."
                                    ),
                                })

            # Check fusion channel balance: detect when one branch dominates
            # by analyzing __init__ module channel outputs vs fusion input channels
            fusion_channels = {}  # module_name → estimated output channels
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                    if child.func.attr == "Conv2d" and len(child.args) >= 2:
                        # Track the LAST Conv2d in each Sequential (output channels)
                        out_ch = self._get_constant_value(child.args[1])
                        if out_ch:
                            # Try to find which self.xxx this belongs to
                            fusion_channels[id(child)] = out_ch

            # Analyze per-branch output channels from the legacy branch analysis
            branch_analysis = self._extract_branch_info(node, tree)
            if branch_analysis and len(branch_analysis) >= 2:
                total_branch_ch = sum(b.get("output_channels", 0) for b in branch_analysis.values())
                for name, info in branch_analysis.items():
                    ch = info.get("output_channels", 0)
                    if total_branch_ch > 0:
                        ratio = ch / total_branch_ch
                        if ratio < 0.10:
                            bottlenecks["detected"].append({
                                "type": "branch_dominance",
                                "branch": name,
                                "channels": ch,
                                "total_fusion_channels": total_branch_ch,
                                "ratio": round(ratio, 3),
                                "severity": "high" if ratio < 0.05 else "medium",
                                "description": (
                                    f"Branch '{name}' has {ch} channels ({ratio*100:.0f}% of fusion). "
                                    f"It will be gradient-drowned by other branches "
                                    f"(total fusion: {total_branch_ch} channels)."
                                ),
                            })

            # Check fusion points for channel imbalance
            for fp in data_flow.get("fusion_points", []):
                if fp.get("operation") == "concatenation":
                    sources = fp.get("sources", [])
                    if len(sources) >= 2:
                        bottlenecks["detected"].append({
                            "type": "fusion_imbalance",
                            "sources": sources,
                            "severity": "medium",
                            "description": (
                                f"Fusion concatenates {len(sources)} branches: "
                                f"{', '.join(sources)}. If channel counts are imbalanced, "
                                f"smaller branches will be gradient-drowned."
                            ),
                        })

            break  # Only first nn.Module

        if not bottlenecks["detected"]:
            bottlenecks["warnings"].append("No significant information bottlenecks detected.")
        else:
            high_count = sum(1 for b in bottlenecks["detected"] if b.get("severity") == "high")
            if high_count > 0:
                bottlenecks["warnings"].append(
                    f"CRITICAL: {high_count} severe bottleneck(s) detected. "
                    f"These will cause information loss and may explain poor model performance."
                )

        return bottlenecks

    def _get_constant_value(self, node) -> int | None:
        """Safely extract an integer constant from an AST node."""
        import ast
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        return None

    def _analyze_gradient_paths(self, tree, data_flow: dict) -> dict:
        """Analyze gradient flow paths through the model.

        Checks whether all branches receive meaningful gradients by analyzing:
        1. Are there skip connections (addition) that bypass branches?
        2. Are there branches that feed into low-channel bottlenecks?
        3. Is there a main path that dominates gradient flow?
        """
        gradient_paths = {
            "skip_connections": [],
            "potential_dead_branches": [],
            "gradient_dominance_warnings": [],
            "recommendations": [],
        }

        # Analyze fusion points for gradient issues
        for fp in data_flow.get("fusion_points", []):
            op = fp.get("operation", "")
            sources = fp.get("sources", [])

            if op == "addition" and len(sources) >= 2:
                gradient_paths["skip_connections"].append({
                    "type": "residual_connection",
                    "sources": sources,
                    "note": "Addition-based skip connection — gradients flow equally to both paths.",
                })
            elif op == "concatenation" and len(sources) >= 2:
                gradient_paths["recommendations"].append(
                    f"Fusion point '{fp.get('variable', '?')}' uses concatenation of {len(sources)} sources. "
                    f"If source channel counts differ greatly, the smaller source gets fewer gradients. "
                    f"Consider equalizing channel counts before fusion."
                )

        # Check for branches that never fuse (dead branches)
        all_stages = data_flow.get("processing_stages", [])
        stage_vars = set()
        for stage in all_stages:
            if isinstance(stage, dict) and "variable" in stage:
                stage_vars.add(stage["variable"])

        for fp in data_flow.get("fusion_points", []):
            for src in fp.get("sources", []):
                # Strip subscripts like features[0] → features
                base_var = src.split("[")[0].split(".")[0]
                if base_var in stage_vars:
                    stage_vars.discard(base_var)

        # Remaining stage_vars may be unused (dead branches)
        if stage_vars:
            gradient_paths["potential_dead_branches"] = list(stage_vars)
            gradient_paths["gradient_dominance_warnings"].append(
                f"Variables {stage_vars} are computed but may not contribute to the final output. "
                f"This suggests dead branches that waste parameters and computation."
            )

        return gradient_paths

    def _analyze_structural_soundness(self, tree, data_flow: dict, bottlenecks: dict) -> dict:
        """Evaluate structural soundness of the model architecture.

        Checks:
        1. Branch balance — are all branches meaningful?
        2. Redundancy — are there duplicate modules doing the same thing?
        3. Depth vs width — is the model too deep or too wide for the data?
        4. Overall structural score
        """
        import ast

        soundness = {
            "score": 10,  # Start at 10, deduct for issues
            "issues": [],
            "strengths": [],
        }

        # Check 1: Branch balance (from legacy analysis + new data flow)
        fusion_count = len(data_flow.get("fusion_points", []))
        if fusion_count == 0:
            # No multi-branch architecture — could be fine or could mean dead branches
            pass
        elif fusion_count == 1:
            soundness["strengths"].append("Single fusion point — clear information aggregation.")
        else:
            soundness["issues"].append(
                f"Multiple fusion points ({fusion_count}) — ensure they don't conflict."
            )
            soundness["score"] -= 1

        # Check 2: Bottleneck severity
        high_bottlenecks = [b for b in bottlenecks.get("detected", []) if b.get("severity") == "high"]
        if high_bottlenecks:
            soundness["issues"].append(
                f"{len(high_bottlenecks)} severe bottleneck(s) will cause information loss."
            )
            soundness["score"] -= len(high_bottlenecks) * 2

        # Check 3: Forward pass complexity
        stage_count = len(data_flow.get("processing_stages", []))
        if stage_count > 30:
            soundness["issues"].append(
                f"Very deep forward pass ({stage_count} stages) — gradient vanishing risk."
            )
            soundness["score"] -= 2
        elif stage_count < 3:
            soundness["issues"].append(
                f"Very shallow forward pass ({stage_count} stages) — may lack representational power."
            )
            soundness["score"] -= 1
        else:
            soundness["strengths"].append(f"Reasonable forward pass depth ({stage_count} stages).")

        # Check 4: Dead branches
        dead_branches = []  # Would need gradient path analysis
        for gp in data_flow.get("processing_stages", []):
            if isinstance(gp, dict) and gp.get("module_type", "").startswith("Dead"):
                dead_branches.append(gp.get("module", "unknown"))

        # Clamp score
        soundness["score"] = max(0, min(10, soundness["score"]))

        return soundness

    def _analyze_domain_assumptions(self, content: str, analysis: dict) -> dict:
        """Analyze what physical/domain assumptions the model architecture encodes.

        This is critical for understanding WHY a model might fail on certain domains.
        Uses a knowledge base of common CV method assumptions (EPI, FFT, Conv3D, etc.)
        to detect potential assumption violations. This is generic scientific knowledge,
        not project-specific logic.
        """
        assumptions = {
            "detected_assumptions": [],
            "risk_domains": [],
            "recommendations": [],
        }

        content_lower = content.lower()

        # Pattern-based assumption detection
        assumption_patterns = [
            (r"epi|epipolar", "EPI/Epipolar analysis",
             "Assumes Lambertian surfaces (constant radiance across views). "
             "Violated by specular, reflective, or scattering surfaces.",
             "Non-Lambertian"),
            (r"fft|fourier|freq", "Frequency domain analysis",
             "Assumes frequency components carry domain-relevant information. "
             "High-frequency noise or aliasing may corrupt features.",
             "Depends on data quality"),
            (r"angular.*weight|view.*weight|attention.*angular",
             "Angular/view weighting",
             "Assumes some views are more informative than others. "
             "May not generalize if view importance varies by scene type.",
             "Cross-domain"),
            (r"resnet|residual.*block|pretrained|frozen",
             "Pre-trained backbone features",
             "Assumes backbone features transfer to the target domain. "
             "May fail if target domain is very different from ImageNet.",
             "Domain gap"),
            (r"sigmoid", "Sigmoid output activation",
             "Sigmoid converges to ~0.5 without strong gradients. "
             "If loss doesn't produce strong gradients, output collapses.",
             "All domains (training stability)"),
            (r"mean.*pool|avg.*pool|adaptive_avg_pool",
             "Mean pooling aggregation",
             "Averaging erases inter-view/inter-channel differences. "
             "May lose critical domain-specific information.",
             "Heterogeneous domains"),
            (r"conv3d|3d.*conv", "3D convolution",
             "Captures spatio-angular correlations. "
             "Works well when the input structure is regular. "
             "May overfit to structural patterns absent in some domains.",
             "Irregular data domains"),
        ]

        for pattern, name, implication, risk_domain in assumption_patterns:
            if re.search(pattern, content_lower):
                assumptions["detected_assumptions"].append({
                    "name": name,
                    "pattern": pattern,
                    "implication": implication,
                    "risk_domain": risk_domain,
                })
                if risk_domain not in assumptions["risk_domains"]:
                    assumptions["risk_domains"].append(risk_domain)

        # Generate recommendations based on detected assumptions
        for asm in assumptions["detected_assumptions"]:
            risk = asm["risk_domain"]
            if "Lambertian" in risk:
                assumptions["recommendations"].append(
                    f"Architecture uses {asm['name']} which makes specific surface property "
                    f"assumptions. For domains that violate these assumptions, consider adding "
                    f"a domain-aware branch that doesn't rely on this assumption, or use "
                    f"domain-specific loss weighting."
                )
            elif "stability" in risk.lower():
                assumptions["recommendations"].append(
                    f"Architecture uses {asm['name']} which can cause training instability. "
                    f"Consider replacing with a more stable alternative (e.g., clamped linear output)."
                )
            elif "gap" in risk.lower():
                assumptions["recommendations"].append(
                    f"Architecture uses {asm['name']} — verify that pre-trained features "
                    f"transfer to the target domain. Consider fine-tuning the backbone."
                )

        if not assumptions["detected_assumptions"]:
            assumptions["recommendations"].append(
                "No strong domain assumptions detected — architecture is relatively domain-agnostic."
            )

        return assumptions

    def _diagnose_results_vs_architecture(self, results: dict, analysis: dict, content: str) -> dict:
        """Diagnose WHY training results are poor by tracing back to architectural causes.

        Given training metrics (e.g., domain-specific MAEs) and the architecture analysis,
        produces a causal chain: poor metric → architectural component → root cause → fix.
        """
        diagnosis = {
            "domain_findings": [],
            "root_causes": [],
            "recommended_fixes": [],
            "confidence": "medium",
        }

        domain_assumptions = analysis.get("domain_assumptions", {})
        bottlenecks = analysis.get("information_bottlenecks", {})
        structural = analysis.get("structural_soundness", {})
        gradient_paths = analysis.get("gradient_paths", {})

        # Analyze each domain's metrics
        for key, value in results.items():
            if not key.startswith("MAE_") and not key.startswith("mse_"):
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue

            domain_name = key.replace("MAE_", "").replace("mse_", "").replace("_", " ")

            finding = {
                "metric": key,
                "value": val,
                "severity": "high" if val > 0.35 else ("medium" if val > 0.20 else "low"),
            }

            # Link to domain assumptions
            matching_assumptions = []
            for asm in domain_assumptions.get("detected_assumptions", []):
                risk = asm.get("risk_domain", "")
                if domain_name.lower() in risk.lower() or "all" in risk.lower():
                    matching_assumptions.append(asm["name"])

            if matching_assumptions and val > 0.20:
                finding["likely_architectural_cause"] = matching_assumptions
                diagnosis["root_causes"].append(
                    f"{key}={val:.4f}: Architecture uses {', '.join(matching_assumptions)} "
                    f"which may not work for {domain_name} domain."
                )
                diagnosis["recommended_fixes"].append(
                    f"For {domain_name} domain: Replace or augment {', '.join(matching_assumptions)} "
                    f"with a domain-robust alternative."
                )

            # Link to bottlenecks
            if val > 0.30 and bottlenecks.get("detected"):
                high_bn = [b for b in bottlenecks["detected"] if b.get("severity") == "high"]
                if high_bn:
                    finding["bottleneck_contribution"] = [
                        b["description"] for b in high_bn
                    ]
                    diagnosis["root_causes"].append(
                        f"{key}={val:.4f}: Severe information bottleneck may be limiting "
                        f"the model's ability to represent {domain_name} features."
                    )

            diagnosis["domain_findings"].append(finding)

        # Add gradient path warnings
        if gradient_paths.get("potential_dead_branches"):
            diagnosis["root_causes"].append(
                f"Dead branches detected: {gradient_paths['potential_dead_branches']}. "
                f"These waste parameters and may indicate design flaws."
            )

        # Add structural score
        if structural.get("score", 10) < 6:
            diagnosis["root_causes"].append(
                f"Structural soundness score: {structural['score']}/10. "
                f"Multiple architectural issues detected."
            )

        if not diagnosis["root_causes"]:
            diagnosis["root_causes"].append(
                "No clear architectural root causes found for the observed metrics. "
                "Consider investigating training dynamics (learning rate, data augmentation, loss function)."
            )

        return diagnosis

    # ── Layer 9: Idea-Architecture Alignment ──

    def _analyze_idea_architecture_alignment(self, analysis: dict, content: str) -> dict:
        """Compare model architecture against PROJECT_BRIEF to verify idea is faithfully implemented.

        This is the key analysis that answers: "Does this model actually implement the research idea?"

        It performs:
        1. Parses PROJECT_BRIEF.md to extract core hypotheses, key innovations, and domain targets
        2. Identifies which model branches/modules correspond to which idea components
        3. Computes per-branch channel allocation vs. idea importance (the "FFT is only 9%" problem)
        4. Detects structural gaps — missing skip connections, absent attention mechanisms, etc.
        5. Checks decoder adequacy for the task complexity described in the idea
        6. Generates specific, actionable improvement suggestions tied to the idea
        """

        brief_path = self.workspace / "PROJECT_BRIEF.md"
        if not brief_path.exists():
            return {"skipped": "PROJECT_BRIEF.md not found in workspace"}

        brief_text = brief_path.read_text()
        brief_lower = brief_text.lower()

        result = {
            "core_idea_summary": "",
            "idea_component_to_branch": [],
            "channel_allocation_analysis": [],
            "structural_gap_analysis": [],
            "decoder_adequacy": {},
            "alignment_score": 10,  # Start at 10, deduct for misalignments
            "critical_findings": [],
            "improvement_suggestions": [],
        }

        # ── Step 1: Extract core idea from PROJECT_BRIEF ──
        # Look for key innovation patterns
        idea_patterns = [
            # (regex_pattern, idea_component_name, importance_weight)
            (r"角度频率|angular.*freq|fft|fourier|k-space|频域",
             "angular_frequency_analysis", 1.0),
            (r"双掩码|dual.*mask|介质掩码|角度.*掩码|medium.*mask|angular.*mask",
             "dual_mask_modeling", 1.0),
            (r"epi|epipolar|epi.*slope|epi.*branch",
             "EPI_branch", 0.7),
            (r"brdf|反射类型|reflection.*type|材质.*分类|material.*class",
             "BRDF_reflection_analysis", 0.9),
            (r"分量感知|component.*aware|domain.*aware|分量.*深度",
             "component_aware_depth", 1.0),
            (r"非朗伯|non.?lambertian|specular|散射|scatter",
             "non_lambertian_handling", 1.0),
            (r"skip.*connection|跳跃连接|残差|residual|u.?net",
             "skip_connection_residual", 0.6),
            (r"multi.?scale|多尺度|金字塔|pyramid|fpn",
             "multi_scale_processing", 0.5),
            (r"attention|注意力|attention.*mechanism|通道.*加权",
             "attention_mechanism", 0.6),
            (r"fusion.*weight|融合.*权重|自适应.*融合|adaptive.*fusion",
             "adaptive_fusion", 0.8),
        ]

        detected_idea_components = {}
        for pattern, component_name, importance in idea_patterns:
            matches = re.findall(pattern, brief_lower)
            if matches:
                detected_idea_components[component_name] = {
                    "brief_mentions": len(matches),
                    "importance": importance,
                    "patterns_matched": matches[:3],
                }

        if not detected_idea_components:
            result["core_idea_summary"] = "Could not extract specific idea components from PROJECT_BRIEF"
            return result

        # Summarize the core idea
        top_components = sorted(detected_idea_components.items(),
                                key=lambda x: x[1]["importance"] * x[1]["brief_mentions"], reverse=True)
        result["core_idea_summary"] = (
            f"Detected {len(top_components)} key idea components in PROJECT_BRIEF. "
            f"Most important: {', '.join(c[0] for c in top_components[:4])}."
        )

        # ── Step 2: Map idea components to model branches/modules ──
        # Extract model branch info from analysis
        branch_analysis = analysis.get("branch_analysis", {})
        data_flow = analysis.get("data_flow", {})
        content_lower = content.lower()

        # Map model constructs to idea components
        model_component_patterns = {
            "angular_frequency_analysis": [
                r"fft", r"fourier", r"freq.*branch", r"angular.*fft", r"angle.*fft",
                r"fft_branch", r"freq_branch", r"spectrum",
            ],
            "dual_mask_modeling": [
                r"dual.*mask", r"medium.*mask", r"angular.*mask", r"mask.*branch",
                r"介质掩码", r"角度掩码", r"med_mask", r"ang_mask",
            ],
            "EPI_branch": [
                r"epi", r"epipolar", r"stream_h", r"stream_v", r"stream_d",
                r"direction.*stream",
            ],
            "BRDF_reflection_analysis": [
                r"brdf", r"reflect.*type", r"material.*class", r"specular.*detect",
                r"reflection.*branch",
            ],
            "component_aware_depth": [
                r"component.*aware", r"domain.*aware", r"分量.*感知", r"weighted.*depth",
                r"domain.*weight",
            ],
            "non_lambertian_handling": [
                r"non.?lambertian", r"specular.*branch", r"scatter.*branch",
                r"non_lambert.*branch", r"spec_branch",
            ],
            "attention_mechanism": [
                r"attention", r"se_block", r"cbam", r"channel.*attention",
                r"spatial.*attention", r"self_atten",
            ],
            "adaptive_fusion": [
                r"adaptive.*fusion", r"fusion.*weight", r"learned.*fusion",
                r"gate.*fusion", r"softmax.*fusion",
            ],
        }

        # Detect which idea components exist in the model code
        model_implements = {}
        for idea_comp, patterns in model_component_patterns.items():
            for pat in patterns:
                if re.search(pat, content_lower):
                    model_implements[idea_comp] = pat
                    break

        # Build the mapping table
        for comp_name, comp_info in detected_idea_components.items():
            mapping = {
                "idea_component": comp_name,
                "importance": comp_info["importance"],
                "brief_mentions": comp_info["brief_mentions"],
                "implemented_in_model": comp_name in model_implements,
            }
            if comp_name in model_implements:
                mapping["detected_pattern"] = model_implements[comp_name]

            # Try to find corresponding branch channels
            for branch_name, branch_info in branch_analysis.items():
                branch_lower = branch_name.lower()
                comp_patterns = model_component_patterns.get(comp_name, [])
                if any(re.search(p, branch_lower) for p in comp_patterns):
                    mapping["corresponding_branch"] = branch_name
                    mapping["branch_output_channels"] = branch_info.get("output_channels", 0)
                    break

            result["idea_component_to_branch"].append(mapping)

            # Flag critical gaps: high-importance idea not implemented
            if not mapping["implemented_in_model"] and comp_info["importance"] >= 0.9:
                result["critical_findings"].append(
                    f"CRITICAL: Idea component '{comp_name}' (importance={comp_info['importance']:.1f}, "
                    f"mentioned {comp_info['brief_mentions']}x in brief) is NOT implemented in the model. "
                    f"This is a core part of the research idea that is missing."
                )
                result["alignment_score"] -= 3
            elif not mapping["implemented_in_model"] and comp_info["importance"] >= 0.7:
                result["critical_findings"].append(
                    f"WARNING: Idea component '{comp_name}' (importance={comp_info['importance']:.1f}) "
                    f"is not explicitly implemented. It may be implicitly handled or missing."
                )
                result["alignment_score"] -= 1

        # ── Step 3: Channel allocation analysis (the "FFT is only 9%" check) ──
        if branch_analysis:
            total_ch = sum(b.get("output_channels", 0) for b in branch_analysis.values())
            if total_ch > 0:
                for branch_name, branch_info in branch_analysis.items():
                    ch = branch_info.get("output_channels", 0)
                    ratio = ch / total_ch
                    entry = {
                        "branch": branch_name,
                        "channels": ch,
                        "ratio_pct": round(ratio * 100, 1),
                    }

                    # Check if this branch corresponds to a HIGH-importance idea component
                    branch_lower = branch_name.lower()
                    is_key_innovation = False
                    for comp_name, comp_info in detected_idea_components.items():
                        if comp_info["importance"] >= 0.9:
                            comp_patterns = model_component_patterns.get(comp_name, [])
                            if any(re.search(p, branch_lower) for p in comp_patterns):
                                is_key_innovation = True
                                entry["corresponds_to_idea"] = comp_name
                                entry["idea_importance"] = comp_info["importance"]
                                break

                    # Flag if a KEY innovation branch gets too few channels
                    if is_key_innovation and ratio < 0.15:
                        entry["alert"] = (
                            f"KEY INNOVATION BRANCH '{branch_name}' gets only {ratio*100:.0f}% "
                            f"of fusion channels ({ch}/{total_ch}). This branch is central to the "
                            f"research idea but is architecturally under-represented. Its contribution "
                            f"will likely be drowned out by other branches during both forward pass "
                            f"and gradient backpropagation."
                        )
                        result["critical_findings"].append(entry["alert"])
                        result["alignment_score"] -= 2

                        # Generate improvement suggestion
                        suggested_ch = max(int(total_ch * 0.25), ch * 2)
                        result["improvement_suggestions"].append(
                            f"INCREASE '{branch_name}' channels from {ch} to ~{suggested_ch} "
                            f"(from {ratio*100:.0f}% to ~{suggested_ch/(total_ch-ch+suggested_ch)*100:.0f}% "
                            f"of fusion). This branch implements a core idea component and needs more "
                            f"representational capacity."
                        )
                    elif is_key_innovation and ratio < 0.25:
                        entry["note"] = (
                            f"Key innovation branch '{branch_name}' gets {ratio*100:.0f}% — "
                            f"adequate but could benefit from more capacity."
                        )

                    result["channel_allocation_analysis"].append(entry)

                # Sort by ratio descending for readability
                result["channel_allocation_analysis"].sort(
                    key=lambda x: x["ratio_pct"], reverse=True
                )

        # ── Step 4: Decoder adequacy analysis ──
        decoder_analysis = self._analyze_decoder_adequacy(content, detected_idea_components)
        result["decoder_adequacy"] = decoder_analysis
        if decoder_analysis.get("issues"):
            for issue in decoder_analysis["issues"]:
                if "CRITICAL" in issue or "high" in issue.lower():
                    result["alignment_score"] -= 2
                else:
                    result["alignment_score"] -= 1

        # ── Step 5: Structural gap detection ──
        # Check for common architectural patterns that the idea implies but the model lacks
        implied_patterns = {
            "skip_connection_residual": {
                "regex": [r"skip.*conn", r"\+=.*identity", r"residual.*add", r"\+ .*identity",
                          r"shortcut", r"resblock", r"ResBlock"],
                "why_important": "Skip connections preserve low-level spatial details that are critical for dense prediction tasks.",
                "default_impact": "Without skip connections, the decoder must reconstruct fine details from compressed features alone — this often causes blurry or over-smoothed outputs.",
            },
            "multi_scale_processing": {
                "regex": [r"pool.*pool", r"stride.*2", r"downsample", r"upsample", r"interpolate",
                          r"FPN", r"pyramid", r"MultiScale", r"multi_scale"],
                "why_important": "Multi-scale processing captures both local and global context, essential for depth estimation where depth cues exist at multiple scales.",
                "default_impact": "Single-scale processing may miss global depth structure or fine-grained details.",
            },
            "attention_mechanism": {
                "regex": [r"attention", r"SE.*block", r"CBAM", r"gate", r"squeeze.*excite",
                          r"self_attention", r"cross_attention"],
                "why_important": "When fusing multiple branches (EPI + FFT + center), attention allows the model to learn which branch to trust for each pixel/domain.",
                "default_impact": "Without attention, all branches contribute equally via concatenation — the model cannot adaptively weight branches for different domains.",
            },
        }

        for pattern_name, pattern_info in implied_patterns.items():
            has_pattern = any(re.search(p, content_lower) for p in pattern_info["regex"])
            entry = {
                "pattern": pattern_name,
                "present": has_pattern,
            }
            if has_pattern:
                entry["status"] = "present"
            else:
                # Only flag as gap if the idea implies it would be useful
                idea_components = detected_idea_components.keys()
                is_relevant = False
                if pattern_name == "skip_connection_residual":
                    is_relevant = True  # Almost always relevant for dense prediction tasks
                elif pattern_name == "multi_scale_processing":
                    is_relevant = "component_aware_depth" in idea_components or "non_lambertian_handling" in idea_components or len(list(idea_components)) >= 2
                elif pattern_name == "attention_mechanism":
                    is_relevant = len([b for b in branch_analysis.values()]) >= 3  # Multi-branch fusion

                if is_relevant:
                    entry["status"] = "MISSING"
                    entry["why_important"] = pattern_info["why_important"]
                    entry["impact"] = pattern_info["default_impact"]
                    result["critical_findings"].append(
                        f"MISSING PATTERN: '{pattern_name}' is absent from the model. "
                        f"{pattern_info['default_impact']}"
                    )
                    result["improvement_suggestions"].append(
                        f"ADD {pattern_name}: {pattern_info['why_important']}"
                    )
                    result["alignment_score"] -= 1
                else:
                    entry["status"] = "not_applicable"

            result["structural_gap_analysis"].append(entry)

        # ── Step 6: Final summary ──
        result["alignment_score"] = max(0, min(10, result["alignment_score"]))

        if result["alignment_score"] >= 8:
            result["overall_assessment"] = (
                f"Model architecture aligns well with the research idea (score: {result['alignment_score']}/10). "
                f"Minor improvements may help but the core idea is faithfully implemented."
            )
        elif result["alignment_score"] >= 5:
            result["overall_assessment"] = (
                f"Model architecture PARTIALLY implements the research idea (score: {result['alignment_score']}/10). "
                f"Key components are present but under-represented or structurally weak. "
                f"Review the critical findings above for specific fixes."
            )
        else:
            result["overall_assessment"] = (
                f"Model architecture POORLY implements the research idea (score: {result['alignment_score']}/10). "
                f"Core idea components are missing or severely under-represented. "
                f"A significant architectural revision is likely needed."
            )

        return result

    def _analyze_decoder_adequacy(self, content: str, idea_components: dict) -> dict:
        """Analyze whether the decoder is adequate for the task complexity described in the idea.

        Checks:
        1. Decoder depth (number of layers)
        2. Presence of skip connections in decoder
        3. Decoder receptive field vs. input size
        4. Whether the decoder can handle multi-domain outputs implied by the idea
        """

        content_lower = content.lower()

        result = {
            "decoder_depth": 0,
            "has_skip_connections": False,
            "has_multi_scale_output": False,
            "issues": [],
            "recommendations": [],
        }

        # Find decoder block using AST for reliable multi-line parsing
        decoder_conv_count = 0
        try:
            file_tree = ast.parse(content)
            for node in ast.walk(file_tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                is_module = False
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == "Module":
                        is_module = True
                        break
                if not is_module:
                    continue

                for item in node.body:
                    if not isinstance(item, ast.FunctionDef) or item.name != "__init__":
                        continue
                    for stmt in ast.walk(item):
                        if not isinstance(stmt, ast.Assign):
                            continue
                        for target in stmt.targets:
                            if not isinstance(target, ast.Attribute):
                                continue
                            if not isinstance(target.value, ast.Name) or target.value.id != "self":
                                continue
                            attr_name = target.attr.lower()
                            if any(kw in attr_name for kw in ("decoder", "head", "output", "final", "out")):
                                # Count Conv2d in this assignment's value
                                for child in ast.walk(stmt):
                                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                                        if child.func.attr == "Conv2d":
                                            decoder_conv_count += 1
            result["decoder_depth"] = decoder_conv_count
        except Exception:
            # Fallback: regex
            decoder_match = re.search(
                r"self\.(?:decoder|head|output|final|out)\s*=\s*nn\.Sequential\((.+?)\)",
                content, re.DOTALL
            )
            if decoder_match:
                decoder_block = decoder_match.group(1)
                result["decoder_depth"] = len(re.findall(r"Conv2d", decoder_block))

        # Check for skip connections anywhere in model
        result["has_skip_connections"] = bool(
            re.search(r"\+= .*identity|shortcut|skip.*conn|residual.*add", content_lower)
        )

        # Check for multi-scale output
        result["has_multi_scale_output"] = bool(
            re.search(r"multi.*scale|pyramid|fpn|upsample.*\+.*downsample|interpolate.*\+.*conv",
                      content_lower)
        )

        # Assess adequacy
        fusion_ch_match = re.search(r"fusion_ch\s*=\s*(\d+)", content)
        fusion_channels = int(fusion_ch_match.group(1)) if fusion_ch_match else 0

        if result["decoder_depth"] <= 3 and fusion_channels > 200:
            result["issues"].append(
                f"high: Very shallow decoder ({result['decoder_depth']} Conv layers) for a large "
                f"fusion input ({fusion_channels} channels). The decoder must compress a lot of "
                f"information in few layers — this risks losing fine-grained spatial details, "
                f"especially for dense prediction tasks where spatial precision matters."
            )
            result["recommendations"].append(
                f"Widen the decoder: Add at least 1-2 more Conv layers and consider skip connections "
                f"from branch intermediate features (e.g., branch features at half resolution)."
            )

        if not result["has_skip_connections"] and fusion_channels > 100:
            result["issues"].append(
                f"medium: No skip connections in the model. For a {fusion_channels}-channel fusion, "
                f"the decoder must reconstruct spatial details purely from compressed features. "
                f"This typically leads to blurry edge predictions."
            )
            result["recommendations"].append(
                "Add U-Net style skip connections: connect intermediate features from branches "
                "(before fusion) to corresponding decoder layers."
            )

        # Check if idea requires multi-domain handling but decoder is single-output
        needs_domain_awareness = any(
            comp in idea_components
            for comp in ["non_lambertian_handling", "component_aware_depth", "dual_mask_modeling",
                         "domain_adaptive", "multi_domain"]
        )
        if needs_domain_awareness:
            has_domain_branch = bool(re.search(r"domain.*weight|mask.*predict|domain.*class|confidence.*map", content_lower))
            if not has_domain_branch:
                result["issues"].append(
                    "medium: The idea requires domain-aware/component-aware processing, "
                    "but the model has no explicit domain prediction or confidence branch. "
                    "The model outputs a single prediction without domain adaptation."
                )
                result["recommendations"].append(
                    "Add a parallel domain/classification head that predicts per-pixel domain "
                    "weights (e.g., domain probability). Use these weights "
                    "to modulate the main output or loss weighting."
                )

        return result

    # ── Runtime Model Probe ──

    def _infer_input_shape(self, model_path: str) -> str:
        """Infer input shape from model's first Conv/Linear layer or __init__ params.

        Returns a default shape string if inference fails.
        """
        try:
            abs_path = self._resolve_workspace_path(model_path)
            content = abs_path.read_text()
            # Look for common shape patterns in the model code
            # Pattern 1: explicit shape in __init__ or forward docstring
            m = re.search(r"input.*shape.*?(\[[\d,\s]+\])", content)
            if m:
                return m.group(1).replace(" ", "")
            # Pattern 2: look for Conv3d/Conv2d in_channels hint
            m = re.search(r"Conv3d\((\d+)", content)
            if m:
                in_ch = int(m.group(1))
                # Heuristic: if in_ch > 10, likely multi-view (e.g., 81, 49, 25)
                return f"[1, {in_ch}, 3, 64, 64]"
            m = re.search(r"Conv2d\((\d+)", content)
            if m:
                return f"[1, {m.group(1)}, 64, 64]"
        except Exception:
            pass
        # Generic fallback — most common CV input
        return "[1, 3, 64, 64]"

    def _exec_probe_model(
        self,
        model_path: str,
        model_class: str,
        input_shape: str = "",
        checkpoint_path: str = "",
    ) -> str:
        """Runtime model diagnostic: instantiate, forward, backward, capture real tensor stats.

        This generates a Python probe script, runs it via subprocess, and returns
        structured results including per-module activation statistics and gradient norms.
        """
        # Build the probe script
        probe_script = self._build_probe_script(model_path, model_class, input_shape, checkpoint_path)

        # Write probe script to workspace temp file
        probe_path = self.workspace / "_model_probe_tmp.py"
        try:
            probe_path.write_text(probe_script)

            result = subprocess.run(
                [sys.executable, str(probe_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.workspace),
            )

            if result.returncode != 0:
                return json.dumps({
                    "error": f"Probe failed with returncode {result.returncode}",
                    "stderr": result.stderr[-2000:],
                    "stdout": result.stdout[-500:],
                })

            # Parse the JSON output from the probe script
            output = result.stdout.strip()
            if output.startswith("{"):
                try:
                    return json.dumps(json.loads(output), ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass

            return json.dumps({
                "raw_output": output[-3000:],
                "stderr": result.stderr[-500:],
            })

        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Probe timed out after 60s — model may be too large"})
        except Exception as e:
            return json.dumps({"error": f"Probe execution failed: {e}"})
        finally:
            # Clean up temp file
            try:
                probe_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _build_probe_script(
        self, model_path: str, model_class: str, input_shape: str, checkpoint_path: str
    ) -> str:
        """Build a self-contained Python script that probes the model."""
        # Parse input_shape
        if input_shape:
            try:
                shape = json.loads(input_shape)
                shape_str = str(shape)
            except json.JSONDecodeError:
                shape_str = self._infer_input_shape(model_path)
        else:
            shape_str = self._infer_input_shape(model_path)

        checkpoint_code = ""
        if checkpoint_path:
            abs_ckpt = self._resolve_workspace_path(checkpoint_path)
            checkpoint_code = f"""
    # Load trained weights
    import sys as _sys
    _ckpt = torch.load('{abs_ckpt}', map_location='cpu', weights_only=False)
    if isinstance(_ckpt, dict) and 'model_state_dict' in _ckpt:
        model.load_state_dict(_ckpt['model_state_dict'])
    elif isinstance(_ckpt, dict) and 'state_dict' in _ckpt:
        model.load_state_dict(_ckpt['state_dict'])
    else:
        model.load_state_dict(_ckpt)
    print(f"Loaded checkpoint from {checkpoint_path}", file=_sys.stderr)
"""

        abs_model = self._resolve_workspace_path(model_path)
        model_dir = abs_model.parent
        model_file = abs_model.stem

        return f'''"""Auto-generated model probe script"""
import sys
import json
import traceback

# Add model directory to path
sys.path.insert(0, "{model_dir}")
sys.path.insert(0, "{self.workspace}")
# Also add project root (workspace may be project_dir/workspace or project_dir itself)
from pathlib import Path as _P
_ws = _P("{self.workspace}")
sys.path.insert(0, str(_ws.parent))  # project_dir if workspace = project_dir/workspace
sys.path.insert(0, str(_ws))          # workspace itself

import torch
import torch.nn as nn

# ── Activation Hooks ──
activations = {{}}
gradients = {{}}

def hook_fn(name):
    def forward_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            activations[name] = output.detach()
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            activations[name] = output[0].detach()
    return forward_hook

def grad_hook_fn(name):
    def backward_hook(module, grad_input, grad_output):
        if isinstance(grad_output, (tuple,)) and len(grad_output) > 0 and grad_output[0] is not None:
            gradients[name] = grad_output[0].detach()
    return backward_hook

try:
    # ── Import model ──
    from {model_file} import {model_class}

    # ── Instantiate ──
    model = {model_class}()
    model.eval()
{checkpoint_code}
    # ── Register hooks ──
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.Conv3d)):
            module.register_forward_hook(hook_fn(name))
            module.register_full_backward_hook(grad_hook_fn(name))

    # ── Create dummy input ──
    dummy_input = torch.randn({shape_str})

    # ── Forward pass ──
    with torch.no_grad():
        output = model(dummy_input)

    # ── Collect forward statistics ──
    results = {{
        "model_class": "{model_class}",
        "input_shape": {shape_str},
        "output_shape": list(output.shape) if isinstance(output, torch.Tensor) else "unknown",
        "modules_probed": len(activations),
    }}

    # ── Per-module activation analysis ──
    module_stats = []
    for name, act in activations.items():
        flat = act.flatten().float()
        stats = {{
            "name": name,
            "shape": list(act.shape),
            "mean": round(flat.mean().item(), 6),
            "std": round(flat.std().item(), 6),
            "min": round(flat.min().item(), 6),
            "max": round(flat.max().item(), 6),
            "sparsity": round((flat == 0).float().mean().item(), 4),
            "dead_ratio": round((flat.abs() < 1e-6).float().mean().item(), 4),
        }}
        # Check for collapse
        if stats["std"] < 1e-5:
            stats["warning"] = "DEAD: activation collapsed to near-constant"
        elif stats["std"] < 1e-3:
            stats["warning"] = "WEAK: very low activation variance"
        elif stats["dead_ratio"] > 0.5:
            stats["warning"] = f"SPARSE: {{stats['dead_ratio']*100:.0f}}% values are near-zero"

        module_stats.append(stats)

    results["activation_stats"] = module_stats

    # ── Gradient analysis (requires backward pass) ──
    model.train()
    activations.clear()
    gradients.clear()

    # Re-register hooks (clear old)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.Conv3d)):
            module.register_forward_hook(hook_fn(name))
            module.register_full_backward_hook(grad_hook_fn(name))

    dummy_input2 = torch.randn({shape_str})
    output2 = model(dummy_input2)

    if isinstance(output2, torch.Tensor):
        loss = output2.mean()
        loss.backward()

        grad_stats = []
        for name, grad in gradients.items():
            flat = grad.flatten().float()
            g_stats = {{
                "name": name,
                "grad_mean": round(flat.mean().item(), 8),
                "grad_std": round(flat.std().item(), 8),
                "grad_norm": round(flat.norm().item(), 6),
                "grad_max": round(flat.abs().max().item(), 6),
            }}
            if g_stats["grad_norm"] < 1e-7:
                g_stats["warning"] = "GRADIENT-DEAD: this module receives no meaningful gradients"
            elif g_stats["grad_norm"] < 1e-4:
                g_stats["warning"] = "GRADIENT-WEAK: very small gradients, may be drowned by other branches"
            grad_stats.append(g_stats)

        results["gradient_stats"] = grad_stats

        # ── Branch gradient balance analysis ──
        # Find pairs of branches with vastly different gradient norms
        if len(grad_stats) >= 2:
            norms = [(g["name"], g["grad_norm"]) for g in grad_stats if g["grad_norm"] > 0]
            if norms:
                max_name, max_norm = max(norms, key=lambda x: x[1])
                min_name, min_norm = min(norms, key=lambda x: x[1])
                if max_norm > 0 and min_norm > 0:
                    ratio = max_norm / min_norm
                    results["gradient_balance"] = {{
                        "max_gradient_module": max_name,
                        "max_gradient_norm": round(max_norm, 6),
                        "min_gradient_module": min_name,
                        "min_gradient_norm": round(min_norm, 6),
                        "imbalance_ratio": round(ratio, 1),
                    }}
                    if ratio > 100:
                        results["gradient_balance"]["warning"] = (
                            f"SEVERE IMBALANCE: {{max_name}} gets {{ratio:.0f}}x more gradient "
                            f"than {{min_name}}. The weaker module is effectively dead."
                        )
                    elif ratio > 10:
                        results["gradient_balance"]["warning"] = (
                            f"MODERATE IMBALANCE: {{max_name}} gets {{ratio:.0f}}x more gradient "
                            f"than {{min_name}}. The weaker module may not learn effectively."
                        )

    # ── Parameter statistics ──
    param_stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            flat = param.data.flatten().float()
            p_stats = {{
                "name": name,
                "shape": list(param.shape),
                "mean": round(flat.mean().item(), 6),
                "std": round(flat.std().item(), 6),
                "num_params": param.numel(),
            }}
            param_stats.append(p_stats)
    results["parameter_stats"] = param_stats
    results["total_parameters"] = sum(p.numel() for p in model.parameters())

    # ── Multi-pattern activation comparison ──
    # Test if the model produces different outputs for different input patterns
    model.eval()
    patterns = {{
        "random": torch.randn({shape_str}),
        "uniform_05": torch.ones({shape_str}) * 0.5,
        "zeros_edge": torch.randn({shape_str}) * 0.01,  # near-zero input
    }}

    pattern_outputs = {{}}
    for pname, pinput in patterns.items():
        with torch.no_grad():
            pout = model(pinput)
        if isinstance(pout, torch.Tensor):
            pattern_outputs[pname] = {{
                "output_mean": round(pout.mean().item(), 6),
                "output_std": round(pout.std().item(), 6),
                "output_min": round(pout.min().item(), 6),
                "output_max": round(pout.max().item(), 6),
            }}

    results["input_sensitivity"] = pattern_outputs

    # Check if all patterns produce identical output → model ignores input
    if len(pattern_outputs) >= 2:
        means = [v["output_mean"] for v in pattern_outputs.values()]
        stds = [v["output_std"] for v in pattern_outputs.values()]
        mean_range = max(means) - min(means)
        std_range = max(stds) - min(stds)
        if mean_range < 1e-5 and std_range < 1e-5:
            results["input_sensitivity"]["warning"] = (
                "CRITICAL: Model produces IDENTICAL output regardless of input. "
                "The model is not learning — it has collapsed to a constant function."
            )

    # ── Branch Feature Redundancy Analysis ──
    # Check if different branches produce REDUNDANT (highly correlated) features.
    # If branch outputs are >95% correlated, one branch is wasted.
    model.eval()
    activations.clear()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv3d)):
            module.register_forward_hook(hook_fn(name))

    with torch.no_grad():
        _ = model(torch.randn({shape_str}))

    # Group activations by branch (heuristic: modules sharing a prefix like 'stream_h', 'fft_branch')
    branch_groups = {{}}
    for name, act in activations.items():
        parts = name.split('.')
        prefix = parts[0] if len(parts) > 1 else 'unknown'
        if prefix not in branch_groups:
            branch_groups[prefix] = []
        flat = act.flatten().float()
        if flat.numel() > 0:
            branch_groups[prefix].append(flat)

    branch_redundancy = {{}}
    branch_names = [k for k in branch_groups if len(branch_groups[k]) > 0]
    for i, bn1 in enumerate(branch_names):
        for bn2 in branch_names[i+1:]:
            # Compare last activation of each branch (output features)
            act1 = branch_groups[bn1][-1] if branch_groups[bn1] else None
            act2 = branch_groups[bn2][-1] if branch_groups[bn2] else None
            if act1 is not None and act2 is not None and act1.numel() > 1 and act2.numel() > 1:
                # Flatten to 1D and compute cosine similarity
                min_len = min(act1.numel(), act2.numel())
                a1 = act1[:min_len]
                a2 = act2[:min_len]
                cos_sim = torch.nn.functional.cosine_similarity(
                    a1.unsqueeze(0), a2.unsqueeze(0)
                ).item()
                corr = round(abs(cos_sim), 4)
                key = f"{{bn1}}_vs_{{bn2}}"
                branch_redundancy[key] = {{
                    "cosine_similarity": corr,
                    "redundant": corr > 0.95,
                }}
                if corr > 0.95:
                    branch_redundancy[key]["warning"] = (
                        f"Branches {{bn1}} and {{bn2}} produce nearly identical features "
                        f"(cosine_similarity={{corr}}). One is redundant."
                    )

    if branch_redundancy:
        results["branch_redundancy"] = branch_redundancy

    # ── Feature Rank Analysis ──
    # For the final fused features, check how many dimensions carry information.
    # If 90% of variance is in <10% of dimensions, there's severe information compression.
    if activations:
        last_key = list(activations.keys())[-1]
        last_act = activations[last_key].flatten().float()
        if last_act.numel() > 10:
            # Simple rank estimation via ratio of top-k values to total
            sorted_abs, _ = torch.sort(last_act.abs(), descending=True)
            total_energy = sorted_abs.sum().item()
            for top_k_pct in [0.5, 0.9, 0.95]:
                threshold = total_energy * top_k_pct
                cumsum = torch.cumsum(sorted_abs, dim=0)
                idx = (cumsum >= threshold).nonzero(as_tuple=False)
                if len(idx) > 0:
                    rank_needed = idx[0].item() + 1
                    results.setdefault("feature_rank", {{}})[f"top_{{int(top_k_pct*100)}}pct_energy"] = {{
                        "dimensions_needed": rank_needed,
                        "total_dimensions": last_act.numel(),
                        "compression_ratio": round(last_act.numel() / max(rank_needed, 1), 1),
                    }}

    print(json.dumps(results, ensure_ascii=False))

except Exception as e:
    error_result = {{
        "error": str(e),
        "traceback": traceback.format_exc()[-2000:],
    }}
    print(json.dumps(error_result, ensure_ascii=False))
'''

    def _exec_generate_diagnostic(
        self,
        question: str,
        model_path: str,
        model_class: str,
        data_path: str = "",
        checkpoint_path: str = "",
    ) -> str:
        """Generate a domain-specific diagnostic script based on the question asked.

        Unlike probe_model (random data), this creates a targeted script that
        tests specific properties like per-domain feature distribution, assumption
        validity, or branch behavior on different input types.
        """
        resolved_model = self._resolve_workspace_path(model_path)
        if not resolved_model.exists():
            return json.dumps({"error": f"Model file not found: {model_path}"})

        # Determine diagnostic type from the question
        question_lower = question.lower()
        diag_type = "general"
        if any(kw in question_lower for kw in ["epi", "slope", "epipolar", "domain", "assumption"]):
            diag_type = "domain_analysis"
        elif any(kw in question_lower for kw in ["fft", "frequency", "branch", "dead"]):
            diag_type = "branch_analysis"
        elif any(kw in question_lower for kw in ["gradient", "backward", "vanishing", "exploding"]):
            diag_type = "gradient_analysis"
        elif any(kw in question_lower for kw in ["attention", "weight", "importance"]):
            diag_type = "attention_analysis"

        # Generate the diagnostic script
        script = self._build_diagnostic_script(
            question=question,
            diag_type=diag_type,
            model_path=str(resolved_model),
            model_class=model_class,
            data_path=data_path,
            checkpoint_path=checkpoint_path,
        )

        # Save to workspace
        script_name = f"_diagnostic_{diag_type}_tmp.py"
        script_path = self.workspace / script_name
        try:
            script_path.write_text(script)

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self.workspace),
            )

            if result.returncode != 0:
                return json.dumps({
                    "error": f"Diagnostic script failed (returncode {result.returncode})",
                    "stderr": result.stderr[-2000:],
                    "question": question,
                    "diagnostic_type": diag_type,
                })

            output = result.stdout.strip()
            if output.startswith("{"):
                try:
                    parsed = json.loads(output)
                    parsed["_meta"] = {
                        "question": question,
                        "diagnostic_type": diag_type,
                        "script_path": script_name,
                    }
                    return json.dumps(parsed, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass

            return json.dumps({
                "output": output[-3000:],
                "_meta": {"question": question, "diagnostic_type": diag_type},
            })

        except subprocess.TimeoutExpired:
            return json.dumps({"error": "Diagnostic timed out after 60s"})
        except Exception as e:
            return json.dumps({"error": f"Diagnostic execution failed: {e}"})
        finally:
            try:
                script_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _build_diagnostic_script(
        self, question: str, diag_type: str,
        model_path: str, model_class: str,
        data_path: str, checkpoint_path: str
    ) -> str:
        """Build a targeted diagnostic script based on the question type."""

        checkpoint_code = ""
        if checkpoint_path:
            abs_ckpt = self._resolve_workspace_path(checkpoint_path)
            abs_ckpt_str = str(abs_ckpt).replace("\\", "\\\\")
            checkpoint_code = f"""
    # Load trained weights
    ckpt = torch.load(r'{abs_ckpt}', map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    print(f"Loaded checkpoint from {abs_ckpt_str}, keys: {{len(ckpt.get('model_state_dict', ckpt))}}")
"""

        data_load_code = ""
        if data_path:
            abs_data = self._resolve_workspace_path(data_path)
            data_load_code = f"""
    # Try to load real data
    try:
        import numpy as np
        data_files = list(Path(r'{abs_data}').glob('*.npy')) + list(Path(r'{abs_data}').glob('*.npz'))
        if data_files:
            sample = np.load(str(data_files[0]), allow_pickle=True)
            if hasattr(sample, 'keys'):
                sample = sample[list(sample.keys())[0]]
            print(f"Loaded real data: shape={{sample.shape}}")
    except Exception as e:
        print(f"Data loading failed: {{e}}")
"""

        if diag_type == "domain_analysis":
            question_repr = repr(question)
            specific_code = """
    # Domain-specific analysis
    # Generate different input patterns to simulate domain differences
    _shape = %s
    patterns = {
        "smooth": torch.randn(*_shape) * 0.3 + 0.5,  # Low-variance, centered
        "high_freq": torch.randn(*_shape),             # Full noise
        "constant": torch.ones(*_shape) * 0.5,         # Constant output
        "sparse_peak": torch.zeros(*_shape),            # Sparse activation pattern
    }
    patterns["sparse_peak"][:, :, :, 10:20, 10:20] = 1.0  # Bright spot

    results = {}
    for name, pattern in patterns.items():
        activations = {}
        def hook_fn(n):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activations[n] = output.detach()
            return hook

        hooks = []
        for n, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv3d)):
                hooks.append(m.register_forward_hook(hook_fn(n)))

        with torch.no_grad():
            out = model(pattern)

        for h in hooks:
            h.remove()

        stats = {}
        for n, act in activations.items():
            flat = act.flatten().float()
            stats[n] = {
                "mean": round(flat.mean().item(), 4),
                "std": round(flat.std().item(), 4),
                "dead_ratio": round((flat.abs() < 1e-6).float().mean().item(), 4),
                "max": round(flat.max().item(), 4),
            }
        if isinstance(out, torch.Tensor):
            flat_out = out.flatten()
            stats["OUTPUT"] = {
                "mean": round(flat_out.mean().item(), 4),
                "std": round(flat_out.std().item(), 4),
                "range": round((flat_out.max() - flat_out.min()).item(), 4),
            }

        results[name] = stats

    # Compute domain sensitivity: how much do activations change across patterns
    all_layers = set()
    for pattern_stats in results.values():
        all_layers.update(pattern_stats.keys())

    sensitivity = {}
    for layer in all_layers:
        stds = [results[p].get(layer, {}).get("std", 0) for p in results]
        means = [results[p].get(layer, {}).get("mean", 0) for p in results]
        sensitivity[layer] = {
            "std_variance": round(max(stds) - min(stds), 4),
            "mean_range": round(max(means) - min(means), 4),
        }

    print(json.dumps({
        "diagnostic": "domain_analysis",
        "question": %s,
        "per_pattern_stats": results,
        "domain_sensitivity": sensitivity,
    }, ensure_ascii=False, indent=2))
""" % (shape_str, question_repr)

        elif diag_type == "branch_analysis":
            specific_code = """
    # Branch analysis: compare activations across branches
    x = torch.randn(*%s)
    branch_outputs = {}

    def branch_hook(branch_name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                branch_outputs[branch_name] = output.detach()
        return hook

    # Register hooks on top-level modules (branches)
    hooks = []
    for name, module in model.named_children():
        if hasattr(module, 'forward'):
            hooks.append(module.register_forward_hook(branch_hook(name)))

    with torch.no_grad():
        out = model(x)

    for h in hooks:
        h.remove()

    results = {"branches": {}}
    branch_names = list(branch_outputs.keys())
    for name in branch_names:
        flat = branch_outputs[name].flatten().float()
        results["branches"][name] = {
            "shape": list(branch_outputs[name].shape),
            "mean": round(flat.mean().item(), 4),
            "std": round(flat.std().item(), 4),
            "dead_ratio": round((flat.abs() < 1e-6).float().mean().item(), 4),
            "l2_norm": round(flat.norm().item(), 4),
        }

    # Pairwise cosine similarity between branches
    if len(branch_names) >= 2:
        results["redundancy"] = {}
        for i in range(len(branch_names)):
            for j in range(i + 1, len(branch_names)):
                a = branch_outputs[branch_names[i]].flatten()
                b = branch_outputs[branch_names[j]].flatten()
                if a.shape == b.shape:
                    cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                    results["redundancy"][f"{branch_names[i]}_vs_{branch_names[j]}"] = round(cos_sim, 4)

    print(json.dumps({
        "diagnostic": "branch_analysis",
        "question": %s,
        **results,
    }, ensure_ascii=False, indent=2))
""" % (shape_str, repr(question))

        elif diag_type == "gradient_analysis":
            specific_code = """
    # Gradient analysis: check gradient flow through all layers
    x = torch.randn(*%s, requires_grad=True)
    grad_stats = {}

    def grad_hook(layer_name):
        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                go = grad_output[0]
            else:
                go = grad_output
            if isinstance(go, torch.Tensor):
                grad_stats[layer_name] = {
                    "grad_norm": round(go.norm().item(), 6),
                    "grad_mean": round(go.mean().item(), 6),
                    "grad_std": round(go.std().item(), 6),
                }
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv3d)):
            hooks.append(module.register_full_backward_hook(grad_hook(name)))

    output = model(x)
    if isinstance(output, torch.Tensor):
        loss = output.mean()
        loss.backward()

    for h in hooks:
        h.remove()

    # Identify gradient bottlenecks
    grad_norms = {n: s["grad_norm"] for n, s in grad_stats.items() if s["grad_norm"] > 0}
    if grad_norms:
        max_norm = max(grad_norms.values())
        min_norm = min(grad_norms.values())
        bottleneck_layers = [n for n, g in grad_norms.items() if g < max_norm * 0.01]
    else:
        max_norm = min_norm = 0
        bottleneck_layers = []

    print(json.dumps({
        "diagnostic": "gradient_analysis",
        "question": %s,
        "per_layer_gradients": grad_stats,
        "gradient_balance": round(max_norm / max(min_norm, 1e-10), 2),
        "bottleneck_layers": bottleneck_layers[:5],
        "zero_gradient_layers": [n for n, s in grad_stats.items() if s["grad_norm"] == 0][:5],
    }, ensure_ascii=False, indent=2))
""" % (shape_str, repr(question))

        else:  # general
            specific_code = """
    # General diagnostic
    x = torch.randn(*%s)
    with torch.no_grad():
        out = model(x)

    results = {
        "diagnostic": "general",
        "question": %s,
    }

    if isinstance(out, torch.Tensor):
        flat = out.flatten()
        results["output_stats"] = {
            "shape": list(out.shape),
            "mean": round(flat.mean().item(), 4),
            "std": round(flat.std().item(), 4),
            "min": round(flat.min().item(), 4),
            "max": round(flat.max().item(), 4),
        }

    results["model_parameters"] = sum(p.numel() for p in model.parameters())
    results["model_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(json.dumps(results, ensure_ascii=False, indent=2))
""" % (shape_str, repr(question))

        script = f'''
import sys
import json
import traceback
from pathlib import Path

import torch
import torch.nn as nn

try:
    # Add project to path
    sys.path.insert(0, str(Path(r"{model_path}").parent))
    sys.path.insert(0, '.')

    # Import model
    import importlib.util
    spec = importlib.util.spec_from_file_location("model_module", r"{model_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ModelClass = getattr(mod, "{model_class}")
    model = ModelClass()
    model.eval()
{checkpoint_code}
{data_load_code}
{specific_code}

except Exception as e:
    print(json.dumps({{"error": str(e), "traceback": traceback.format_exc()[-2000:]}}, ensure_ascii=False))
'''
        return script

    def _exec_design_ablation(
        self,
        model_path: str,
        model_class: str,
        target_metrics: str = "",
        baseline_metrics: str = "",
    ) -> str:
        """Design ablation experiments by systematically analyzing model components.

        Uses AST analysis to identify major components, then generates an
        ablation plan with prioritized experiments.
        """
        resolved_model = self._resolve_workspace_path(model_path)
        if not resolved_model.exists():
            return json.dumps({"error": f"Model file not found: {model_path}"})

        try:
            source = resolved_model.read_text()
            tree = ast.parse(source)
        except Exception as e:
            return json.dumps({"error": f"Failed to parse model: {e}"})

        # Find the model class
        model_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == model_class:
                model_node = node
                break

        if not model_node:
            return json.dumps({"error": f"Class {model_class} not found in {model_path}"})

        # Extract components from __init__
        components = []
        for node in ast.walk(model_node):
            if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
                continue
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                            if target.value.id == "self":
                                attr_name = target.attr
                                # Check if it's a module (has nn.XXX in value)
                                if isinstance(stmt.value, ast.Call):
                                    func = stmt.value.func
                                    if isinstance(func, ast.Attribute):
                                        module_type = func.attr
                                        if module_type in (
                                            "Conv2d", "Linear", "Conv3d", "Sequential",
                                            "ModuleList", "ModuleDict", "BatchNorm2d",
                                            "ReLU", "LeakyReLU", "Dropout", "MaxPool2d",
                                            "AvgPool2d", "AdaptiveAvgPool2d",
                                            "MultiheadAttention", "TransformerEncoder",
                                            "TransformerEncoderLayer",
                                        ):
                                            # Extract channel/size info
                                            info = {"type": module_type}
                                            for kw in (stmt.value.keywords or []):
                                                if kw.arg in ("out_channels", "out_features", "embed_dim"):
                                                    if isinstance(kw.value, ast.Constant):
                                                        info[kw.arg] = kw.value.value
                                            components.append({
                                                "name": attr_name,
                                                "module_type": module_type,
                                                "info": info,
                                            })

        if not components:
            return json.dumps({
                "error": "No identifiable components found for ablation",
                "model_class": model_class,
            })

        # Group components by functional category
        categories = {
            "backbone": [],
            "branch": [],
            "head": [],
            "fusion": [],
            "normalization": [],
            "activation": [],
            "other": [],
        }

        for comp in components:
            name = comp["name"].lower()
            if any(kw in name for kw in ["backbone", "resnet", "encoder", "feature"]):
                categories["backbone"].append(comp)
            elif any(kw in name for kw in ["branch", "stream", "path", "dir"]):
                categories["branch"].append(comp)
            elif any(kw in name for kw in ["head", "output", "fc", "predictor"]):
                categories["head"].append(comp)
            elif any(kw in name for kw in ["cat", "fuse", "merge", "combine", "fusion"]):
                categories["fusion"].append(comp)
            elif any(kw in name for kw in ["norm", "bn", "batch", "layer", "group"]):
                categories["normalization"].append(comp)
            elif any(kw in name for kw in ["relu", "sigmoid", "act", "leaky", "gelu"]):
                categories["activation"].append(comp)
            else:
                categories["other"].append(comp)

        # Design ablation experiments
        ablations = []
        ablation_id = 0

        # 1. Remove each branch one at a time
        for comp in categories["branch"]:
            ablation_id += 1
            ablations.append({
                "id": ablation_id,
                "type": "component_removal",
                "target": comp["name"],
                "description": f"Remove {comp['name']} ({comp['module_type']})",
                "expected_insight": f"How much does {comp['name']} contribute to overall performance?",
                "priority": "high" if "branch" in comp["name"].lower() else "medium",
                "risk": "low" if len(categories["branch"]) > 2 else "high",
            })

        # 2. Replace fusion with simple mean
        if categories["fusion"]:
            ablation_id += 1
            ablations.append({
                "id": ablation_id,
                "type": "fusion_replacement",
                "target": "all_fusion",
                "description": "Replace all learned fusion with simple mean",
                "expected_insight": "Is learned fusion better than naive averaging?",
                "priority": "high",
                "risk": "medium",
            })

        # 3. Freeze backbone
        if categories["backbone"]:
            ablation_id += 1
            ablations.append({
                "id": ablation_id,
                "type": "freeze_component",
                "target": categories["backbone"][0]["name"],
                "description": f"Freeze {categories['backbone'][0]['name']} (no gradient)",
                "expected_insight": "Is the backbone actually learning domain-specific features?",
                "priority": "high",
                "risk": "low",
            })

        # 4. Remove normalization layers
        if categories["normalization"]:
            ablation_id += 1
            ablations.append({
                "id": ablation_id,
                "type": "component_removal",
                "target": "all_normalization",
                "description": "Remove all normalization layers",
                "expected_insight": "Are normalization layers helping or hurting?",
                "priority": "medium",
                "risk": "high",
            })

        # 5. Single-branch model (keep only best branch)
        if len(categories["branch"]) > 1:
            ablation_id += 1
            ablations.append({
                "id": ablation_id,
                "type": "single_branch",
                "target": "keep_only_best_branch",
                "description": "Keep only the best-performing branch, remove all others",
                "expected_insight": "Are multiple branches actually better than one?",
                "priority": "medium",
                "risk": "medium",
            })

        # Parse baseline metrics for value estimation
        baseline = {}
        if baseline_metrics:
            try:
                baseline = json.loads(baseline_metrics)
            except (json.JSONDecodeError, TypeError):
                pass

        # Estimate information value of each ablation
        for abl in ablations:
            if baseline:
                # Higher priority for ablations that target the worst domain
                abl["estimated_value"] = "high" if abl["priority"] == "high" else "medium"
            else:
                abl["estimated_value"] = "medium" if abl["priority"] == "high" else "low"

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        ablations.sort(key=lambda a: priority_order.get(a["priority"], 3))

        return json.dumps({
            "model": model_class,
            "components_found": len(components),
            "categories": {k: len(v) for k, v in categories.items() if v},
            "total_ablations": len(ablations),
            "recommended_order": ablations,
            "pilot_suggestion": (
                "Run the top 2 ablations as pilot experiments (2-3 epochs each) "
                "before committing to full training. This gives maximum information "
                "with minimum GPU cost."
            ),
        }, ensure_ascii=False, indent=2)
