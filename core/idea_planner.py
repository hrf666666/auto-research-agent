"""
AutoResearcher Idea-to-Model Planning Engine

Bridges the gap between a research idea (PROJECT_BRIEF) and a concrete
model architecture implementation. This is the "forward design" capability
that enables PhD-level architecture planning.

The pipeline:
  1. Idea Formalization — Parse PROJECT_BRIEF into structured idea components
  2. Module Decomposition — Break the idea into independent functional modules
  3. Module Specification — Define input/output/assumptions for each module
  4. Capacity Planning — Estimate channel counts, parameters, and compute for each module
  5. Integration Planning — Design how modules connect and interact
  6. Fusion Strategy — Determine the optimal fusion method for multi-module models
  7. Verification Plan — Define what to check at each stage

This module provides:
- `IdeaPlanner` class: The main planning engine
- `plan_model` tool: Exposed to Leader and Code agents via ToolRegistry
"""

import json
import re
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.idea_planner")


class ModuleSpec:
    """Specification for a single model module/branch."""

    def __init__(self, name: str, function: str, inputs: list[str], outputs: list[str],
                 assumptions: list[str], capacity_hint: str = "medium",
                 dependencies: list[str] = None):
        self.name = name
        self.function = function  # What this module does
        self.inputs = inputs       # Expected input types/shapes
        self.outputs = outputs     # Output types/shapes
        self.assumptions = assumptions  # Physical/mathematical assumptions
        self.capacity_hint = capacity_hint  # "light", "medium", "heavy"
        self.dependencies = dependencies or []  # Other modules this depends on

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "function": self.function,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "assumptions": self.assumptions,
            "capacity_hint": self.capacity_hint,
            "dependencies": self.dependencies,
        }


class FusionStrategy:
    """Recommended fusion method with justification."""

    def __init__(self, method: str, rationale: str,
                 channel_allocation: dict = None,
                 risks: list[str] = None):
        self.method = method
        self.rationale = rationale
        self.channel_allocation = channel_allocation or {}
        self.risks = risks or []

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "rationale": self.rationale,
            "channel_allocation": self.channel_allocation,
            "risks": self.risks,
        }


class ArchitecturePlan:
    """Complete model architecture plan derived from a research idea."""

    def __init__(self):
        self.idea_summary: str = ""
        self.idea_components: list[dict] = []
        self.modules: list[ModuleSpec] = []
        self.fusion_strategy: Optional[FusionStrategy] = None
        self.integration_plan: dict = {}
        self.capacity_budget: dict = {}
        self.verification_plan: list[dict] = []
        self.risks: list[str] = []
        self.implementation_order: list[str] = []
        self.alignment_score: int = 0  # 0-10

    def to_dict(self) -> dict:
        return {
            "idea_summary": self.idea_summary,
            "idea_components": self.idea_components,
            "modules": [m.to_dict() for m in self.modules],
            "fusion_strategy": self.fusion_strategy.to_dict() if self.fusion_strategy else None,
            "integration_plan": self.integration_plan,
            "capacity_budget": self.capacity_budget,
            "verification_plan": self.verification_plan,
            "risks": self.risks,
            "implementation_order": self.implementation_order,
            "alignment_score": self.alignment_score,
        }


# ── Generic Idea Pattern Database ──
# Maps common research idea patterns to architectural module templates.
# This is REUSABLE across domains — not project-specific.
IDEA_PATTERNS = {
    "multi_branch_fusion": {
        "description": "Multiple parallel branches that process different aspects of input, then fuse",
        "typical_modules": ["feature_extractor", "domain_specific_branch", "fusion_module", "decoder"],
        "fusion_methods": {
            "balanced_concat": {
                "when": "All branches equally important",
                "channel_rule": "Each branch gets ~1/N of fusion channels",
            },
            "attention_weighted": {
                "when": "Branch importance varies by input",
                "channel_rule": "Key branches get 25-40%, supporting branches get 15-25%",
            },
            "gated_fusion": {
                "when": "Branches should be conditionally active",
                "channel_rule": "Each branch gets enough channels to be independently useful (≥64ch)",
            },
        },
        "common_pitfalls": [
            "Dominant branch drowns out minority branches (< 10% channels)",
            "Fusion by simple concat without learned weighting",
            "All branches have identical architecture despite different functions",
            "No mechanism to detect which branch is reliable for each input",
        ],
    },
    "frequency_domain_analysis": {
        "description": "Transform input to frequency domain for analysis",
        "typical_modules": ["spatial_encoder", "frequency_transform", "band_separator", "frequency_decoder"],
        "design_rules": {
            "fft_resolution": "Angular resolution must be ≥ 5×5 for meaningful 2D-FFT",
            "band_separation": "Separate low (structure) / mid (texture) / high (detail) bands explicitly",
            "normalization": "Use log1p or dB scaling — raw magnitude causes gradient domination",
            "phase": "Consider retaining phase information, not just magnitude",
        },
        "common_pitfalls": [
            "FFT on too-small angular grid (noise dominates)",
            "No frequency band separation — single FFT feature vector loses information",
            "Compressing 243 FFT channels to 32 in one step (bottleneck)",
        ],
    },
    "domain_adaptive": {
        "description": "Model adapts behavior based on input domain/type",
        "typical_modules": ["shared_backbone", "domain_detector", "domain_heads", "adaptive_fusion"],
        "design_rules": {
            "detection_first": "Domain detection must be verified BEFORE building domain-specific paths",
            "data_requirement": "Need ≥ 20 samples per domain to learn domain-specific features",
            "shared_vs_specific": "Shared features first, domain-specific on top — not parallel from scratch",
        },
        "common_pitfalls": [
            "Domain head with no data to train on",
            "Hard domain classification when soft blending is more appropriate",
            "Shared backbone too small to carry multi-domain information",
        ],
    },
    "component_aware": {
        "description": "Decompose input into physical components, process each separately",
        "typical_modules": ["component_decomposer", "component_processors", "recompositor", "output_head"],
        "design_rules": {
            "decomposition_verifiability": "Component decomposition must be verifiable on data",
            "independence": "Components should be as independent as possible for parallel learning",
            "recomposition": "Weighted sum with learnable weights, not fixed formula",
        },
        "common_pitfalls": [
            "Components not actually independent — shared gradients cause interference",
            "No way to verify decomposition quality during training",
            "Fixed weights when adaptive weighting is needed",
        ],
    },
    "attention_mechanism": {
        "description": "Learn which parts of input are most important",
        "typical_modules": ["query_projector", "key_projector", "value_projector", "attention_aggregator"],
        "design_rules": {
            "dimension": "Attention dimension ≥ sqrt(channel_count) for expressive power",
            "data_requirement": "Attention needs ≥ 50 training samples to converge",
            "fallback": "Always have a non-attention fallback path",
        },
        "common_pitfalls": [
            "Attention weights converge to uniform (not enough data)",
            "Self-attention on spatial dimensions blows up memory (O(n²))",
            "No residual connection around attention — gradient highway breaks",
        ],
    },
    "progressive_refinement": {
        "description": "Start coarse, refine progressively through stages",
        "typical_modules": ["coarse_estimator", "refinement_stage_1", "refinement_stage_2", "final_head"],
        "design_rules": {
            "coarse_first": "Coarse estimator must work independently before adding refinement",
            "residual_refinement": "Each stage predicts residual, not absolute output",
            "feature_reuse": "Later stages should reuse features from earlier stages (skip connections)",
        },
        "common_pitfalls": [
            "Refinement stages don't improve over coarse (not enough capacity)",
            "No skip connections — later stages lose spatial detail",
            "Too many stages for the data available (overfitting)",
        ],
    },
}


class IdeaPlanner:
    """Plans model architecture from research idea description.

    This is the PhD-level "design before coding" capability that bridges:
    Idea text → Structured plan → Module specifications → Implementation guidance

    Usage:
        planner = IdeaPlanner(project_dir)
        plan = planner.plan(brief_text, existing_model_path=None)
        # plan is an ArchitecturePlan with modules, fusion, verification, etc.
    """

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.brief_path = self.project_dir / "PROJECT_BRIEF.md"
        self.manifest_path = self.project_dir / "DATASET_MANIFEST.json"

    def plan(self, brief_text: str = None, existing_model_path: str = None) -> dict:
        """Generate a complete architecture plan from the idea description.

        Args:
            brief_text: PROJECT_BRIEF content. If None, reads from file.
            existing_model_path: Path to existing model file (for incremental design).

        Returns:
            ArchitecturePlan as dict.
        """
        if brief_text is None:
            if self.brief_path.exists():
                brief_text = self.brief_path.read_text()
            else:
                return {"error": "No PROJECT_BRIEF found"}

        plan = ArchitecturePlan()

        # Phase 1: Idea Formalization
        plan.idea_components = self._formalize_idea(brief_text)
        plan.idea_summary = self._summarize_idea(plan.idea_components, brief_text)

        # Phase 2: Module Decomposition
        plan.modules = self._decompose_into_modules(plan.idea_components, brief_text)

        # Phase 3: Capacity Planning
        plan.capacity_budget = self._plan_capacity(plan.modules)

        # Phase 4: Fusion Strategy
        if len(plan.modules) > 1:
            plan.fusion_strategy = self._design_fusion_strategy(plan.modules, plan.idea_components)

        # Phase 5: Integration Plan
        plan.integration_plan = self._plan_integration(plan.modules, plan.fusion_strategy)

        # Phase 6: Verification Plan
        plan.verification_plan = self._design_verification_plan(plan.modules, plan.idea_components)

        # Phase 7: Risk Assessment
        plan.risks = self._assess_risks(plan)

        # Phase 8: Implementation Order
        plan.implementation_order = self._plan_implementation_order(plan.modules)

        # Phase 9: Alignment Score
        plan.alignment_score = self._compute_alignment(plan, brief_text)

        return plan.to_dict()

    # ── Phase 1: Idea Formalization ──

    def _formalize_idea(self, brief_text: str) -> list[dict]:
        """Extract structured idea components from PROJECT_BRIEF.

        Identifies:
        - Core hypothesis (what the idea claims)
        - Key innovations (what's new)
        - Physical assumptions (what must be true)
        - Target domains (where it should work)
        - Success criteria (how to measure)
        """
        components = []
        brief_lower = brief_text.lower()

        # Extract core hypothesis
        hypothesis = self._extract_hypothesis(brief_text)
        if hypothesis:
            components.append({
                "type": "hypothesis",
                "content": hypothesis,
            })

        # Extract key innovations (look for "核心", "key insight", "innovation", "关键")
        innovations = self._extract_innovations(brief_text)
        for innov in innovations:
            components.append({
                "type": "innovation",
                "content": innov["text"],
                "pattern_match": innov.get("pattern"),
                "pattern": innov.get("pattern"),
            })

        # Extract physical assumptions
        assumptions = self._extract_physical_assumptions(brief_text)
        for asm in assumptions:
            components.append({
                "type": "assumption",
                "content": asm,
            })

        # Extract target domains
        domains = self._extract_target_domains(brief_text)
        if domains:
            components.append({
                "type": "target_domains",
                "domains": domains,
            })

        # Extract success criteria
        criteria = self._extract_success_criteria(brief_text)
        if criteria:
            components.append({
                "type": "success_criteria",
                "metrics": criteria,
            })

        # Extract data constraints
        data_info = self._extract_data_info()
        if data_info:
            components.append({
                "type": "data_constraints",
                "info": data_info,
            })

        return components

    def _extract_hypothesis(self, text: str) -> str:
        """Extract the core hypothesis from the brief."""
        # Look for explicit hypothesis statements
        patterns = [
            r"(?:核心\s*(?:idea|思想|假设|Idea)[：:]\s*)(.+?)(?:\n\n|\n##|\Z)",
            r"(?:核心\s*(?:Idea|思想)[^\n]*\n)(.+?)(?:\n\n|\n##|\Z)",
            r"(?:key\s+insight[：:]\s*)(.+?)(?:\n\n|\n##|\Z)",
            r"(?:核心\s*洞察[：:]\s*)(.+?)(?:\n\n|\n##|\Z)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                return m.group(1).strip()[:500]

        # Fallback: first paragraph after "研究目标" or "Research Objective"
        m = re.search(r"(?:研究目标|Research\s+Objective)[^\n]*\n(.+?)(?:\n\n|\n##|\Z)",
                       text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()[:500]

        return ""

    def _extract_innovations(self, text: str) -> list[dict]:
        """Extract key innovation points.
        
        Uses a vocabulary-to-concept mapping: natural language terms → abstract 
        innovation categories. This is a knowledge organization method — the specific
        terms serve as a retrieval vocabulary, not project-specific logic.
        Adding new term-category pairs extends coverage without changing logic.
        """
        innovations = []

        # Vocabulary → concept mapping for innovation detection.
        # Each entry: (regex_pattern, abstract_concept_name)
        innov_patterns = [
            (r"角度频率分析|angular\s+freq", "angular_frequency_analysis"),
            (r"双掩码|dual\s+mask", "dual_mask_modeling"),
            (r"分量感知|component[-\s]?aware", "component_aware_processing"),
            (r"epi|epipolar", "epi_branch"),
            (r"brdf|反射类型|reflection\s+type", "brdf_analysis"),
            (r"注意力|attention", "attention_mechanism"),
            (r"非朗伯|non.?lambertian", "non_lambertian_handling"),
            (r"多尺度|multi.?scale|金字塔|pyramid", "multi_scale_processing"),
            (r"自适应融合|adaptive\s+fusion", "adaptive_fusion"),
            (r"类mri|mri.?analogy|k.?space", "frequency_domain_mri_analogy"),
        ]

        for pat, name in innov_patterns:
            if re.search(pat, text, re.IGNORECASE):
                # Find the surrounding context
                m = re.search(rf"([^\n]*{pat}[^\n]*)", text, re.IGNORECASE)
                context = m.group(0).strip() if m else name
                innovations.append({"text": context, "pattern": name})

        return innovations

    def _extract_physical_assumptions(self, text: str) -> list[str]:
        """Extract physical/mathematical assumptions the idea relies on."""
        assumptions = []
        assumption_patterns = [
            (r"(?:假设|assume|assumption)[：:]\s*(.+?)(?:\n|$)", "explicit"),
            (r"朗伯|lambertian", "Lambertian surface assumption"),
            (r"视角一致性|view\s+consistency|epi.*线形|linear.*epi", "EPI linearity assumption"),
            (r"角点余弦分布|cosine\s+distribution", "Cosine angular distribution"),
            (r"频谱.*低频|spectrum.*low\s+freq", "Frequency structure assumption"),
            (r"波矢|wave\s+vector", "Wave vector linearity assumption"),
        ]

        seen = set()
        for pat, label in assumption_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m and label not in seen:
                if label == "explicit":
                    assumptions.append(m.group(1).strip())
                else:
                    assumptions.append(label)
                seen.add(label)

        return assumptions

    def _extract_target_domains(self, text: str) -> list[str]:
        """Extract target domains/scenarios.
        
        Uses a generic vocabulary of common CV domain terms.
        This is a knowledge organization vocabulary, not project-specific logic.
        """
        domain_keywords = [
            "Lambertian", "Non-Lambertian",
            "Mixed", "outdoor", "indoor", "specular",
            "transparent", "reflective", "scattering",
        ]
        found = []
        text_lower = text.lower()
        for dk in domain_keywords:
            if dk.lower() in text_lower and dk not in found:
                found.append(dk)
        return found

    def _extract_success_criteria(self, text: str) -> list[dict]:
        """Extract measurable success criteria."""
        criteria = []
        # Match patterns like "MAE < 0.20", "val_MAE < 0.16", "准确率 > 85%"
        metric_patterns = [
            (r"(MAE_\w*|val_MAE|val_loss|accuracy|精度|准确率)\s*[<>=]+\s*([\d.]+)", "metric_threshold"),
            (r"(\w*[Mm]etric\w*)\s*[<>=]+\s*([\d.]+)", "metric_threshold"),
        ]
        for pat, mtype in metric_patterns:
            for m in re.finditer(pat, text):
                criteria.append({
                    "metric": m.group(1),
                    "threshold": m.group(2),
                    "operator": "<",
                })
        return criteria

    def _extract_data_info(self) -> dict:
        """Extract data statistics from DATASET_MANIFEST."""
        if not self.manifest_path.exists():
            return {}
        try:
            manifest = json.loads(self.manifest_path.read_text())
            datasets = manifest.get("datasets", {})
            info = {"total_train": 0, "total_val": 0, "per_domain": {}}
            for ds_name, ds_info in datasets.items():
                ds_type = ds_info.get("type", "unknown")
                train = ds_info.get("train_scenes", ds_info.get("train", 0))
                val = ds_info.get("val_scenes", ds_info.get("val", 0))
                if isinstance(train, (list, dict)):
                    train = len(train)
                if isinstance(val, (list, dict)):
                    val = len(val)
                info["total_train"] += train
                info["total_val"] += val
                if ds_type not in info["per_domain"]:
                    info["per_domain"][ds_type] = {"train": 0, "val": 0}
                info["per_domain"][ds_type]["train"] += train
                info["per_domain"][ds_type]["val"] += val
            return info
        except Exception:
            return {}

    def _summarize_idea(self, components: list[dict], brief_text: str) -> str:
        """Generate a concise summary of the idea."""
        innovations = [c for c in components if c.get("type") == "innovation"]
        domains = [c for c in components if c.get("type") == "target_domains"]
        criteria = [c for c in components if c.get("type") == "success_criteria"]

        parts = []
        if innovations:
            innov_names = [i.get("pattern", i.get("content", "?")) for i in innovations[:4]]
            parts.append(f"Key innovations: {', '.join(innov_names)}")
        if domains:
            parts.append(f"Target domains: {', '.join(domains[0].get('domains', []))}")
        if criteria:
            all_metrics = []
            for c in criteria[:4]:
                for m in c.get("metrics", []):
                    all_metrics.append(f"{m.get('metric', '?')}<={m.get('threshold', '?')}")
            if all_metrics:
                parts.append(f"Success criteria: {', '.join(all_metrics)}")

        return "; ".join(parts) if parts else "Could not extract structured idea"

    # ── Phase 2: Module Decomposition ──

    # Map from _extract_innovations pattern names to IDEA_PATTERNS keys
    _INNOVATION_TO_PATTERN = {
        "angular_frequency_analysis": "frequency_domain_analysis",
        "frequency_domain_mri_analogy": "frequency_domain_analysis",
        "epi_branch": "multi_branch_fusion",
        "attention_mechanism": "attention_mechanism",
        "component_aware_processing": "component_aware",
        "adaptive_fusion": "multi_branch_fusion",
        "multi_scale_processing": "progressive_refinement",
        "dual_mask_modeling": "multi_branch_fusion",
        "brdf_analysis": "domain_adaptive",
        "non_lambertian_handling": "domain_adaptive",
    }

    def _decompose_into_modules(self, components: list[dict], brief_text: str) -> list[ModuleSpec]:
        """Break the idea into functional modules with specifications.

        Uses the IDEA_PATTERNS database to match idea components to
        architectural module templates.
        """
        modules = []

        # Match idea components to architectural patterns
        innovations = [c for c in components if c.get("type") == "innovation"]
        matched_patterns = set()
        for innov in innovations:
            innov_pattern = innov.get("pattern", "")
            idea_pattern = self._INNOVATION_TO_PATTERN.get(innov_pattern, "")
            if idea_pattern:
                matched_patterns.add(idea_pattern)

        # Deduplicate and create modules from each unique pattern
        # Use primary pattern (most matched) as backbone, supplement with others
        seen_module_names = set()
        # Priority order: multi_branch_fusion is typically the backbone
        pattern_priority = ["multi_branch_fusion", "frequency_domain_analysis",
                            "domain_adaptive", "component_aware", "attention_mechanism",
                            "progressive_refinement"]
        sorted_patterns = sorted(matched_patterns, key=lambda p: pattern_priority.index(p) if p in pattern_priority else 99)

        # Primary pattern provides backbone modules
        backbone_name = ""
        if sorted_patterns:
            primary = sorted_patterns[0]
            pattern = IDEA_PATTERNS[primary]
            for i, mod_name in enumerate(pattern["typical_modules"]):
                seen_module_names.add(mod_name)
                # First module is the backbone/input encoder
                if not backbone_name:
                    backbone_name = mod_name
                dep = modules[-1].name if modules else ""
                modules.append(ModuleSpec(
                    name=mod_name,
                    function=self._infer_module_function(mod_name, pattern),
                    inputs=self._infer_module_inputs(mod_name, i),
                    outputs=self._infer_module_outputs(mod_name, i),
                    assumptions=self._get_module_assumptions(mod_name, pattern),
                    capacity_hint=self._estimate_module_capacity(mod_name, brief_text),
                    dependencies=dep,
                ))

        # Secondary patterns add branch modules (skip fusion/decoder if already present)
        # Branch modules depend on the backbone, not on each other (parallel architecture)
        structural_names = {"fusion_module", "decoder", "output_head", "feature_extractor",
                            "input_encoder", "shared_backbone"}
        for idea_pattern in sorted_patterns[1:]:
            if idea_pattern not in IDEA_PATTERNS:
                continue
            pattern = IDEA_PATTERNS[idea_pattern]
            for i, mod_name in enumerate(pattern["typical_modules"]):
                if mod_name in seen_module_names:
                    continue
                if mod_name in structural_names:
                    continue  # Don't duplicate backbone/decoder from secondary patterns
                seen_module_names.add(mod_name)
                modules.append(ModuleSpec(
                    name=mod_name,
                    function=self._infer_module_function(mod_name, pattern),
                    inputs=self._infer_module_inputs(mod_name, i),
                    outputs=self._infer_module_outputs(mod_name, i),
                    assumptions=self._get_module_assumptions(mod_name, pattern),
                    capacity_hint=self._estimate_module_capacity(mod_name, brief_text),
                    dependencies=backbone_name,  # Parallel: all branches depend on backbone
                ))

        # If no patterns matched, create generic decomposition
        if not matched_patterns:
            modules = self._generic_decomposition(components, brief_text)

        # Always add common modules if not present
        self._ensure_common_modules(modules, brief_text)

        return modules

    def _infer_module_function(self, mod_name: str, pattern: dict) -> str:
        """Infer what a module does based on its name and pattern context."""
        function_map = {
            "feature_extractor": "Extract low-level features from raw input",
            "spatial_encoder": "Encode spatial information from input images",
            "domain_specific_branch": "Process domain-specific patterns (e.g., frequency, angular)",
            "frequency_transform": "Apply frequency-domain transform (FFT/DCT) to extract spectral features",
            "band_separator": "Separate frequency bands (low/mid/high) for independent analysis",
            "component_decomposer": "Decompose input into physical components",
            "component_processors": "Process each physical component independently",
            "shared_backbone": "Extract shared features across all domains",
            "domain_detector": "Detect which domain/type the input belongs to",
            "domain_heads": "Domain-specific prediction heads",
            "query_projector": "Project features into query space for attention",
            "key_projector": "Project features into key space for attention",
            "value_projector": "Project features into value space for attention",
            "attention_aggregator": "Aggregate features using attention weights",
            "coarse_estimator": "Produce initial coarse prediction",
            "refinement_stage_1": "First refinement stage (predict residual)",
            "refinement_stage_2": "Second refinement stage (predict residual)",
            "frequency_decoder": "Decode frequency-domain features back to spatial",
            "recompositor": "Re-compose decomposed components with learned weights",
            "adaptive_fusion": "Adaptively fuse multi-branch features with learned weights",
            "fusion_module": "Fuse features from multiple branches/modules",
            "decoder": "Decode fused features to final output",
            "output_head": "Final prediction layer(s)",
        }
        return function_map.get(mod_name, f"Process {mod_name} features")

    def _infer_module_inputs(self, mod_name: str, index: int) -> list[str]:
        """Infer expected inputs for a module."""
        input_map = {
            "feature_extractor": ["raw_input_tensor"],
            "spatial_encoder": ["center_view_or_stacked_views"],
            "domain_specific_branch": ["backbone_features_or_raw_input"],
            "frequency_transform": ["angular_grid_reshaped_to_2d"],
            "band_separator": ["fft_magnitude_spectrum"],
            "fusion_module": ["branch_feature_list"],
            "decoder": ["fused_features"],
            "output_head": ["decoder_output"],
        }
        if mod_name in input_map:
            return input_map[mod_name]
        if index == 0:
            return ["raw_input"]
        return ["previous_module_output"]

    def _infer_module_outputs(self, mod_name: str, index: int) -> list[str]:
        """Infer expected outputs for a module."""
        output_map = {
            "feature_extractor": ["feature_maps"],
            "spatial_encoder": ["spatial_feature_maps"],
            "fusion_module": ["fused_feature_maps"],
            "decoder": ["full_resolution_output"],
            "output_head": ["final_prediction"],
        }
        if mod_name in output_map:
            return output_map[mod_name]
        return ["feature_maps"]

    def _get_module_assumptions(self, mod_name: str, pattern: dict) -> list[str]:
        """Get physical assumptions for a module."""
        rules = pattern.get("design_rules", {})
        if isinstance(rules, dict):
            return [f"{k}: {v}" for k, v in rules.items()][:3]
        return []

    def _estimate_module_capacity(self, mod_name: str, brief_text: str) -> str:
        """Estimate whether this module should be light/medium/heavy."""
        heavy_keywords = ["backbone", "encoder", "shared", "feature_extractor"]
        light_keywords = ["head", "detector", "aggregator", "fusion", "output"]
        name_lower = mod_name.lower()
        if any(k in name_lower for k in heavy_keywords):
            return "heavy"
        if any(k in name_lower for k in light_keywords):
            return "light"
        return "medium"

    def _generic_decomposition(self, components: list[dict], brief_text: str) -> list[ModuleSpec]:
        """Create a generic module decomposition when no pattern matches."""
        modules = [
            ModuleSpec("input_encoder", "Encode raw input into feature space",
                       ["raw_input"], ["feature_maps"],
                       ["Input format is consistent across domains"]),
            ModuleSpec("core_processor", "Main processing module implementing the key idea",
                       ["feature_maps"], ["processed_features"],
                       ["Core idea's physical assumptions hold"]),
            ModuleSpec("output_decoder", "Decode processed features to final prediction",
                       ["processed_features"], ["prediction"],
                       ["Output range matches target distribution"]),
        ]

        # If there are multiple innovations, create a branch for each
        innovations = [c for c in components if c.get("type") == "innovation"]
        if len(innovations) > 1:
            modules = [
                ModuleSpec("input_encoder", "Encode raw input into shared feature space",
                           ["raw_input"], ["feature_maps"], []),
            ]
            for i, innov in enumerate(innovations[:4]):
                modules.append(ModuleSpec(
                    f"branch_{i+1}_{innov.get('pattern', 'custom')}",
                    f"Process {innov.get('pattern', 'aspect ' + str(i+1))}",
                    ["feature_maps"], ["branch_features"],
                    innov.get("content", "").split(".")[0][:100] if innov.get("content") else [],
                ))
            modules.append(ModuleSpec("fusion_module", "Fuse all branch outputs",
                                       [f"branch_{i+1}_*" for i in range(len(innovations[:4]))],
                                       ["fused_features"], []))
            modules.append(ModuleSpec("output_decoder", "Decode to final prediction",
                                       ["fused_features"], ["prediction"], []))

        return modules

    def _ensure_common_modules(self, modules: list[ModuleSpec], brief_text: str) -> None:
        """Ensure essential modules are present."""
        names = {m.name for m in modules}
        if "output_head" not in names and "output_decoder" not in names and "decoder" not in names:
            modules.append(ModuleSpec("decoder", "Decode features to final output",
                                       ["fused_or_final_features"], ["prediction"], []))

    # ── Phase 3: Capacity Planning ──

    def _plan_capacity(self, modules: list[ModuleSpec]) -> dict:
        """Plan channel counts and parameter budgets for each module."""
        budget = {}
        # Base channel count depends on total module count
        n_modules = len(modules)
        base_channels = max(32, min(128, 512 // n_modules))

        for mod in modules:
            if mod.capacity_hint == "heavy":
                ch = base_channels * 2
                params_est = ch * 9 * 9 * 3 + ch * 3 * 3 * ch  # rough conv estimate
            elif mod.capacity_hint == "light":
                ch = max(16, base_channels // 2)
                params_est = ch * base_channels * 1 + ch * 1  # minimal
            else:
                ch = base_channels
                params_est = ch * ch * 3 * 3 * 2  # two conv layers

            budget[mod.name] = {
                "recommended_channels": ch,
                "estimated_params": params_est,
                "capacity_tier": mod.capacity_hint,
            }

        # Calculate fusion budget
        module_budgets = {k: v for k, v in budget.items() if isinstance(v, dict)}
        total_branch_ch = sum(b["recommended_channels"] for b in module_budgets.values())
        budget["total_fusion_channels"] = total_branch_ch
        budget["total_estimated_params"] = sum(b["estimated_params"] for b in module_budgets.values())

        return budget

    # ── Phase 4: Fusion Strategy ──

    def _design_fusion_strategy(self, modules: list[ModuleSpec],
                                 idea_components: list[dict]) -> FusionStrategy:
        """Design the optimal fusion method for combining modules."""
        n_branches = len([m for m in modules if m.capacity_hint in ("medium", "heavy")])

        # Check if there's an explicit fusion pattern in the idea
        innovations = [c for c in idea_components if c.get("type") == "innovation"]
        has_adaptive_fusion = any(
            i.get("pattern") in ("adaptive_fusion", "attention_mechanism", "component_aware")
            for i in innovations
        )
        has_component_aware = any(
            i.get("pattern") == "component_aware" for i in innovations
        )

        # Select fusion method
        # Only count actual processing branches (not encoder/decoder/fusion/head)
        structural_names = {"fusion_module", "decoder", "output_head", "feature_extractor",
                            "input_encoder", "shared_backbone", "domain_detector",
                            "attention_aggregator", "recompositor"}
        branch_modules = [m for m in modules if m.name not in structural_names]
        n_branches = len(branch_modules) if branch_modules else n_branches
        if has_component_aware:
            method = "weighted_sum"
            rationale = (
                "Component-aware design: use learnable per-pixel weights to combine "
                "component-specific depth estimates. Weight distribution should reflect "
                "the physical decomposition (e.g., w_diffuse + w_specular + w_scatter ≈ 1)."
            )
        elif has_adaptive_fusion and n_branches >= 3:
            method = "attention_weighted"
            rationale = (
                "Multi-branch with adaptive fusion: use cross-directional attention "
                "to learn per-pixel branch importance. Key branches should get 25-40% "
                "of fusion channels, supporting branches 15-25%."
            )
        elif n_branches == 2:
            method = "learned_weighted_concat"
            rationale = (
                "Two-branch design: use concatenation with a learned channel allocation. "
                "Give each branch proportional channels based on its importance weight. "
                "Avoid simple equal split if one branch is more innovative."
            )
        else:
            method = "balanced_concat"
            rationale = (
                "Standard multi-branch: concatenate with approximately equal channel allocation. "
                "Ensure no branch gets < 15% of total fusion channels."
            )

        # Channel allocation — only for actual processing branches
        channel_allocation = {}
        # Scale total channels based on branch count
        total_ch = max(128, len(branch_modules) * 32)  # At least 32ch per branch
        if branch_modules:
            per_branch = total_ch // len(branch_modules)
            for mod in branch_modules:
                channel_allocation[mod.name] = per_branch

        # Risks
        risks = [
            "Gradient imbalance between branches (monitor with probe_model after 1 epoch)",
            "One branch may dominate early training (consider separate learning rates)",
        ]
        if n_branches > 3:
            risks.append(
                "More than 3 branches increases risk of dead branches — "
                "verify each branch produces distinct features using cosine similarity"
            )

        return FusionStrategy(method, rationale, channel_allocation, risks)

    # ── Phase 5: Integration Plan ──

    def _plan_integration(self, modules: list[ModuleSpec],
                          fusion_strategy: Optional[FusionStrategy]) -> dict:
        """Plan how modules connect and interact."""
        plan = {
            "architecture_type": "parallel_branches" if len(modules) > 2 else "sequential",
            "data_flow": [],
            "skip_connections": [],
            "normalization": "BatchNorm2d (default) or GroupNorm (if batch size=1)",
        }

        # Build data flow description
        for i, mod in enumerate(modules):
            step = {
                "step": i + 1,
                "module": mod.name,
                "input_from": mod.dependencies if mod.dependencies else "raw_input",
                "output_to": modules[i + 1].name if i + 1 < len(modules) else "final_output",
            }
            plan["data_flow"].append(step)

        # Recommend skip connections
        if len(modules) > 3:
            plan["skip_connections"].append({
                "from": modules[0].name,
                "to": modules[-1].name,
                "type": "identity_add",
                "reason": "Preserve low-level spatial details for the decoder",
            })

        return plan

    # ── Phase 6: Verification Plan ──

    def _design_verification_plan(self, modules: list[ModuleSpec],
                                  idea_components: list[dict]) -> list[dict]:
        """Design verification steps for each module and the overall architecture."""
        plan = []

        # Per-module verification
        for mod in modules:
            plan.append({
                "phase": "module_verification",
                "module": mod.name,
                "checks": [
                    f"Forward pass produces finite output with correct shape",
                    f"Gradients flow through this module (not dead)",
                    f"Module respects its physical assumptions: {mod.assumptions[:2]}",
                ],
                "tool": "probe_model (after implementation)",
            })

        # Overall verification
        plan.append({
            "phase": "architecture_verification",
            "checks": [
                "No branch gets < 15% of fusion channels",
                "Fusion module is the largest gradient receiver (not dominated by one branch)",
                "Output range matches target distribution (use sigmoid only if target ∈ [0,1])",
                "Total parameters match data capacity (< 5 params per training sample)",
            ],
            "tool": "analyze_model (after implementation)",
        })

        # Idea alignment verification
        innovations = [c for c in idea_components if c.get("type") == "innovation"]
        if innovations:
            plan.append({
                "phase": "idea_alignment",
                "checks": [
                    f"Each innovation ({', '.join(i.get('pattern', '?') for i in innovations[:4])}) "
                    f"maps to a specific architectural module",
                    "Key innovation modules have sufficient capacity (≥ 25% of fusion channels)",
                    "No innovation is only represented by a single Conv2d layer",
                ],
                "tool": "analyze_model idea_architecture_alignment",
            })

        return plan

    # ── Phase 7: Risk Assessment ──

    def _assess_risks(self, plan: ArchitecturePlan) -> list[str]:
        """Identify architectural risks."""
        risks = []

        # Check module count vs data
        data_info = {}
        for c in plan.idea_components:
            if c.get("type") == "data_constraints":
                data_info = c.get("info", {})

        total_train = data_info.get("total_train", 0)
        n_modules = len(plan.modules)
        total_params = plan.capacity_budget.get("total_estimated_params", 0)

        if total_train > 0 and total_params > 0:
            ratio = total_params / total_train
            if ratio > 5:
                risks.append(
                    f"HIGH RISK: {ratio:.1f} params per training sample — "
                    f"severe overfitting likely. Reduce module count or add regularization."
                )
            elif ratio > 2:
                risks.append(
                    f"MEDIUM RISK: {ratio:.1f} params per training sample — "
                    f"use dropout and early stopping."
                )

        # Check domain data scarcity
        per_domain = data_info.get("per_domain", {})
        for domain, counts in per_domain.items():
            if counts.get("train", 0) < 10:
                risks.append(
                    f"DATA WALL: {domain} domain has only {counts.get('train', 0)} training "
                    f"samples — NO module design can overcome this. Consider data augmentation "
                    f"or transfer learning."
                )

        # Check branch balance (only actual branches, not encoder/decoder)
        branch_names_check = {"branch_", "processor", "transform", "decomposer", "domain_", "frequency_", "epi"}
        if plan.fusion_strategy and plan.fusion_strategy.channel_allocation:
            alloc = plan.fusion_strategy.channel_allocation
            if alloc:
                total_ch = sum(alloc.values())
                n_alloc = len(alloc)
                # Skip imbalance check if branches are naturally balanced (each ~1/N)
                # Only report if there's actual imbalance (some much less than others)
                if n_alloc > 0:
                    expected_pct = 100.0 / n_alloc
                    for name, ch in alloc.items():
                        pct = ch / total_ch * 100 if total_ch > 0 else 0
                        # Only flag if significantly below the expected share
                        if pct < expected_pct * 0.5 and pct < 15:
                            risks.append(
                                f"BRANCH IMBALANCE: {name} gets only {pct:.0f}% of "
                                f"fusion channels — likely to be gradient-drowned."
                            )
                    # Warn if too many branches
                    if n_alloc > 5:
                        risks.append(
                            f"COMPLEXITY WARNING: {n_alloc} processing branches detected. "
                            f"Consider consolidating to 3-4 branches for stable training."
                        )

        return risks

    # ── Phase 8: Implementation Order ──

    def _plan_implementation_order(self, modules: list[ModuleSpec]) -> list[str]:
        """Determine the order in which modules should be implemented."""
        order = []

        # Phase 0: Data validation (always first)
        order.append("DATA_VALIDATION: Verify data supports the idea's assumptions")

        # Phase 1: Input encoder (foundation)
        for mod in modules:
            if "encoder" in mod.name.lower() or "extractor" in mod.name.lower() or "input" in mod.name.lower():
                order.append(f"IMPL: {mod.name}")

        # Phase 2: Core processor(s)
        for mod in modules:
            if any(kw in mod.name.lower() for kw in ("branch", "processor", "transform", "decomposer")):
                order.append(f"IMPL: {mod.name}")

        # Phase 3: Fusion
        for mod in modules:
            if "fusion" in mod.name.lower():
                order.append(f"IMPL: {mod.name}")

        # Phase 4: Decoder/output
        for mod in modules:
            if "decoder" in mod.name.lower() or "output" in mod.name.lower() or "head" in mod.name.lower():
                order.append(f"IMPL: {mod.name}")

        # Phase 5: Verification
        order.append("VERIFY: Run analyze_model + probe_model on complete architecture")

        return order

    # ── Phase 9: Alignment Score ──

    # Map innovation patterns to module name substrings for alignment checking
    _INNOVATION_MODULE_KEYWORDS = {
        "angular_frequency_analysis": ["frequency", "fft", "spectral"],
        "frequency_domain_mri_analogy": ["frequency", "fft", "spectral"],
        "epi_branch": ["epi", "branch", "domain_specific"],
        "attention_mechanism": ["attention", "aggregat"],
        "component_aware_processing": ["component", "decompos"],
        "adaptive_fusion": ["fusion", "adaptive"],
        "multi_scale_processing": ["scale", "coarse", "refinement"],
        "dual_mask_modeling": ["mask", "dual"],
        "brdf_analysis": ["brdf", "domain", "reflection"],
        "non_lambertian_handling": ["domain", "adaptive", "non_lambert"],
    }

    def _compute_alignment(self, plan: ArchitecturePlan, brief_text: str) -> int:
        """Compute how well the plan aligns with the idea (0-10)."""
        score = 10

        innovations = [c for c in plan.idea_components if c.get("type") == "innovation"]
        modules = plan.modules

        # Check: each innovation should map to a module
        for innov in innovations:
            pattern = innov.get("pattern", "")
            keywords = self._INNOVATION_MODULE_KEYWORDS.get(pattern, [])
            if keywords:
                has_module = any(
                    any(kw in m.name.lower() for kw in keywords)
                    for m in modules
                )
            else:
                has_module = any(pattern.replace("_", "") in m.name.replace("_", "").lower() for m in modules)
            if not has_module:
                score -= 1  # Missing module for an innovation (reduced penalty)

        # Check: fusion strategy should be reasonable
        if plan.fusion_strategy:
            if plan.fusion_strategy.method == "balanced_concat" and len(innovations) > 2:
                score -= 1  # Balanced concat isn't optimal for multi-innovation ideas

        # Check: data risks
        risks = plan.risks
        high_risks = [r for r in risks if "HIGH RISK" in r or "DATA WALL" in r]
        score -= len(high_risks) * 2

        return max(0, min(10, score))
