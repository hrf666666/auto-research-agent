"""
AutoResearcher Domain Knowledge Builder

Extracted from ResearchLoop for maintainability.
Contains domain-specific knowledge injection methods:
- _build_domain_knowledge: method-hypothesis-domain compatibility from PROJECT_BRIEF
- _build_cross_experiment_insights: meta-patterns across experiments

IMPORTANT: Domain knowledge is NOT hardcoded. It is extracted dynamically from:
1. PROJECT_BRIEF.md — methods, assumptions, and constraints mentioned in the brief
2. DATASET_MANIFEST.json — data characteristics and limitations
3. Code analysis — what methods are actually implemented in the codebase
4. A generic method properties database (method_properties) — reusable across domains

The only domain-specific content is the GENERIC method properties database,
which catalogs known method assumptions (e.g., "EPI assumes Lambertian").
This is scientific knowledge, not hardcoded project logic.
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("autoresearcher.domain_knowledge")


class DomainKnowledgeMixin:
    """Domain knowledge methods mixin for ResearchLoop.

    All methods assume they have access to:
    - self.project_dir: Path to the project directory
    - self.memory: MemoryManager instance
    """

    # ── Generic Method Properties Database ──
    # This is REUSABLE SCIENTIFIC KNOWLEDGE about common ML/CV methods,
    # NOT project-specific hardcoded logic. Each entry describes:
    # - What assumption a method makes
    # - When that assumption is violated
    # - What failure symptoms look like
    # - What alternatives exist
    #
    # Methods are matched dynamically against PROJECT_BRIEF content.
    # Adding a new method here automatically enables detection for all projects.
    METHOD_PROPERTIES = {
        "epi": {
            "name": "Epipolar Plane Image (EPI)",
            "patterns": ["epi", "epipolar", "epinet", "epi slope", "epi branch"],
            "assumption": (
                "EPI assumes radiance is CONSTANT across views for the same 3D point "
                "(Lambertian condition). Under this assumption, EPI slope is proportional "
                "to scene depth."
            ),
            "violated_when": [
                "Specular/reflective surfaces — radiance changes with view angle",
                "Transparent/refractive materials — multiple depths per pixel",
                "Occlusion boundaries — EPI line is discontinuous",
                "Non-diffuse scattering — radiance varies with viewing direction",
            ],
            "failure_symptoms": [
                "Domain with non-Lambertian surfaces has MAE >> Lambertian MAE",
                "Adding more EPI directions does NOT improve non-Lambertian domains",
                "EPI features have high activation variance across non-Lambertian scenes",
            ],
            "alternatives": [
                "Per-view encoding (no angular structure assumption)",
                "Angular attention (learns which views matter)",
                "Domain-specific feature extraction",
            ],
        },
        "fft": {
            "name": "Angular FFT (2D Fourier Transform on angular grid)",
            "patterns": ["fft", "fourier", "frequency", "angular_freq", "k-space", "频域"],
            "assumption": (
                "Angular frequency patterns are STABLE and carry geometric information. "
                "Low frequencies encode coarse depth, high frequencies encode fine details."
            ),
            "violated_when": [
                "High-frequency noise (sensor noise, compression artifacts)",
                "Irregular angular sampling (missing views, non-uniform grid)",
                "Very few angular views (e.g., < 5x5) — FFT resolution too low",
            ],
            "failure_symptoms": [
                "FFT branch activation collapses to near-constant (std < 1e-3)",
                "FFT branch gradient is 10x+ smaller than other branches",
            ],
            "alternatives": [
                "Learned angular features (Conv3D or attention)",
                "DCT (Discrete Cosine Transform) — more robust to noise",
            ],
        },
        "attention": {
            "name": "Attention Mechanism (angular/spatial/channel)",
            "patterns": ["attention", "self-attention", "angular attention", "cross-attention", "注意力"],
            "assumption": (
                "Not all views/pixels/channels are equally important. "
                "Attention can learn to weight them adaptively."
            ),
            "violated_when": [
                "Insufficient training data — attention weights don't converge",
                "All views actually ARE equally important — attention adds noise",
                "Attention dimension too small to capture complexity",
            ],
            "failure_symptoms": [
                "Attention weights converge to uniform (1/N for all)",
                "Adding attention doesn't change MAE",
            ],
            "alternatives": [
                "Fixed weighting based on domain knowledge",
                "Simple mean/std pooling as non-parametric alternative",
            ],
        },
        "resnet": {
            "name": "ResNet backbone (pre-trained)",
            "patterns": ["resnet", "backbone", "pretrained", "pre-trained", "imagenet"],
            "assumption": (
                "ImageNet pre-trained features transfer to the target domain. "
                "Residual connections allow training deeper networks."
            ),
            "violated_when": [
                "Target domain is very different from natural images (e.g., EPI stacks)",
                "Input is multi-view stacked, not single 2D image",
                "Target task needs very different spatial resolution",
            ],
            "failure_symptoms": [
                "Fine-tuning makes performance WORSE",
                "Frozen vs fine-tuned results are similar",
            ],
            "alternatives": [
                "Train backbone from scratch with domain-specific initialization",
                "Use lighter backbone if data is limited",
            ],
        },
        "sigmoid": {
            "name": "Sigmoid output activation",
            "patterns": ["sigmoid"],
            "assumption": (
                "Output should be in [0, 1] and the gradient is meaningful "
                "throughout the range."
            ),
            "violated_when": [
                "Most outputs cluster near 0.5 (sigmoid saturation)",
                "Loss function doesn't produce strong enough gradients",
                "Ground truth has values near 0 or 1 (sigmoid gradient vanishes)",
            ],
            "failure_symptoms": [
                "Model output mean ~ 0.5 with very low std",
                "probe_model shows output collapse regardless of input",
            ],
            "alternatives": [
                "Softplus + clamping (no saturation)",
                "Direct regression with L1 loss (no activation)",
                "Tanh + scaling to [0, 1]",
            ],
        },
        "conv3d": {
            "name": "3D Convolution",
            "patterns": ["conv3d", "3d conv", "3d convolution"],
            "assumption": (
                "Spatio-angular correlations follow regular patterns. "
                "Conv3D can learn local angular-spatial joint features."
            ),
            "violated_when": [
                "Angular patterns are irregular (non-Lambertian surfaces)",
                "Very few angular views — Conv3D kernel doesn't fit",
                "Angular dimension is not spatially aligned",
            ],
            "failure_symptoms": [
                "Conv3D branch overfits (train loss low, val loss high)",
                "Adding Conv3D doesn't improve non-Lambertian domains",
            ],
            "alternatives": [
                "Separable Conv2D on each view + angular aggregation",
                "Pseudo-4D convolution (factorized)",
            ],
        },
        "contrastive": {
            "name": "Contrastive Learning",
            "patterns": ["contrastive", "infonce", "nt-xent", "simclr", "moco"],
            "assumption": (
                "Positive pairs share the same semantic content; negative pairs don't. "
                "Enough negative samples exist for the representation to be discriminative."
            ),
            "violated_when": [
                "Hard negatives are actually semantically similar",
                "Too few negative samples for the embedding dimension",
                "Augmentation destroys task-relevant information",
            ],
            "failure_symptoms": [
                "Loss decreases but downstream metrics don't improve",
                "Representation collapses to a single point (uniformity loss)",
            ],
            "alternatives": [
                "Non-contrastive methods (BYOL, VICReg)",
                "Supervised contrastive with class labels",
            ],
        },
    }

    def _build_domain_knowledge(self) -> dict:
        """Dynamically extract method-hypothesis-domain knowledge from PROJECT_BRIEF.

        This is the DOMAIN KNOWLEDGE INJECTION that bridges the gap between
        "I know this method uses X" and "I know X assumes Y which fails on Z".

        Instead of hardcoding project-specific knowledge, this method:
        1. Reads PROJECT_BRIEF.md to find which methods are relevant
        2. Matches methods against the generic METHOD_PROPERTIES database
        3. Extracts domain-specific constraints from the brief itself
        4. Detects implemented methods by scanning the codebase
        """
        brief_path = self.project_dir / "PROJECT_BRIEF.md"
        if not brief_path.exists():
            return {}

        try:
            brief_text = brief_path.read_text()
            brief_lower = brief_text.lower()
        except Exception:
            return {}

        kb = {
            "methods_found": [],
            "critical_assumptions": [],
            "method_domain_compatibility": {},
            "data_constraints": [],
            "implemented_methods": [],
        }

        # ── Step 1: Detect methods mentioned in PROJECT_BRIEF ──
        for key, props in self.METHOD_PROPERTIES.items():
            patterns = props.get("patterns", [key])
            matched = False
            for pat in patterns:
                if pat in brief_lower:
                    matched = True
                    break
            if not matched:
                continue

            kb["methods_found"].append(key)
            kb["critical_assumptions"].append({
                "method": props["name"],
                "assumption": props["assumption"],
                "violated_when": props["violated_when"],
                "failure_symptoms": props["failure_symptoms"],
                "alternative": props["alternatives"],
            })

            # ── Step 2: Extract domain compatibility from brief text ──
            # Instead of hardcoding "epi is strong for Lambertian", infer from brief
            domain_compat = self._infer_domain_compatibility(
                key, props, brief_text
            )
            if domain_compat:
                kb["method_domain_compatibility"][key] = domain_compat

        # ── Step 3: Extract data constraints from brief ──
        kb["data_constraints"] = self._extract_data_constraints(brief_text)

        # ── Step 4: Detect implemented methods from codebase ──
        kb["implemented_methods"] = self._detect_implemented_methods()

        # If we found methods, add a mandatory reasoning prompt
        if kb["critical_assumptions"]:
            kb["reasoning_instruction"] = (
                "MANDATORY: Before proposing ANY architectural change, you MUST:\n"
                "1. Check which methods your current architecture uses\n"
                "2. Check the ASSUMPTION of each method\n"
                "3. Check whether that assumption holds for the domain you're trying to improve\n"
                "4. If the assumption is VIOLATED, do NOT add more of the same method.\n"
                "   Instead, use an ALTERNATIVE that doesn't depend on the violated assumption.\n"
                "5. Your experiment hypothesis MUST be falsifiable: 'If we remove X and Y improves, "
                "then X was the cause of poor Y performance.'\n"
                "6. Check data_constraints — if a domain has < 5 training scenes, NO architecture "
                "change can help. Consider data augmentation or transfer learning instead."
            )

        return kb

    def _infer_domain_compatibility(
        self, method_key: str, props: dict, brief_text: str
    ) -> dict:
        """Infer method-domain compatibility from PROJECT_BRIEF content.

        Instead of hardcoding which methods work for which domains,
        we extract domain information from the brief and use the method's
        assumption to determine compatibility.
        """
        brief_lower = brief_text.lower()

        # Extract domain names from brief (look for common patterns)
        domain_names = self._extract_domain_names(brief_text)

        # Use the method's violated_when list to infer weak domains
        weak_domains = []
        strong_domains = []
        violated_conditions = props.get("violated_when", [])

        for domain in domain_names:
            domain_lower = domain.lower()
            # Check if any violation condition mentions this domain's characteristics
            domain_keywords = self._get_domain_keywords(domain)
            is_weak = False
            for condition in violated_conditions:
                cond_lower = condition.lower()
                for dk in domain_keywords:
                    if dk in cond_lower:
                        is_weak = True
                        break
                if is_weak:
                    break

            if is_weak:
                weak_domains.append(domain)
            else:
                strong_domains.append(domain)

        if not strong_domains and not weak_domains:
            return {}

        reason = props.get("assumption", "")
        # Extract first sentence as reason
        reason = reason.split(".")[0] + "."

        return {
            "strong": strong_domains,
            "weak": weak_domains,
            "reason": reason,
        }

    def _extract_domain_names(self, brief_text: str) -> list[str]:
        """Extract domain/scene type names from PROJECT_BRIEF.

        Looks for patterns like "Lambertian", "Non-Lambertian", "Mixed",
        "outdoor", "indoor", etc.
        """
        # Common domain patterns in CV papers
        domain_patterns = [
            r"Lambertian",
            r"Non[- ]Lambertian",
            r"Mixed",
            r"outdoor",
            r"indoor",
            r"urban",
            r"rural",
            r"synthetic",
            r"real[- ]world",
            r"daytime",
            r"nighttime",
            r"specular",
            r"transparent",
            r"reflective",
        ]
        found = []
        for pat in domain_patterns:
            matches = re.findall(pat, brief_text, re.IGNORECASE)
            for m in matches:
                normalized = m.strip().title()
                if normalized not in found:
                    found.append(normalized)
        return found

    def _get_domain_keywords(self, domain_name: str) -> list[str]:
        """Get characteristic keywords for a domain to match against violation conditions."""
        domain_lower = domain_name.lower()
        mapping = {
            "lambertian": ["lambertian", "diffuse", "constant radiance", "cosine"],
            "non-lambertian": ["specular", "reflective", "non-lambertian", "mirror",
                               "transparent", "scattering"],
            "mixed": ["mixed", "urban", "complex", "combined"],
            "outdoor": ["outdoor", "sunlight", "natural"],
            "indoor": ["indoor", "artificial light"],
            "synthetic": ["synthetic", "rendered", "simulated"],
        }
        for key, keywords in mapping.items():
            if key in domain_lower:
                return keywords
        return [domain_lower]

    def _extract_data_constraints(self, brief_text: str) -> list[dict]:
        """Extract data-related constraints from PROJECT_BRIEF.

        Looks for training scene counts, dataset sizes, etc.
        """
        constraints = []
        brief_lower = brief_text.lower()

        # Look for dataset size patterns
        # Match patterns like "4 train", "20 train scenes", "only 3", "data scarcity"
        scarcity_patterns = [
            (r"(\d+)\s+train(?:ing)?\s*(?:scene|sample|image)", "train_count"),
            (r"only\s+(\d+)\s+(?:train|scene|sample)", "scarcity_warning"),
            (r"data\s+scarci?ty", "data_scarcity"),
            (r"data\s+imbalan[cs]e", "data_imbalance"),
        ]
        for pat, ctype in scarcity_patterns:
            matches = re.finditer(pat, brief_lower)
            for m in matches:
                if ctype in ("train_count", "scarcity_warning"):
                    count = int(m.group(1))
                    if count < 10:
                        constraints.append({
                            "type": "data_scarcity",
                            "detail": f"Only {count} training samples detected — "
                                      f"insufficient for learning complex domain-specific features",
                            "recommendation": (
                                "Consider: (1) data augmentation, (2) transfer learning, "
                                "(3) few-shot techniques, (4) not expecting domain-specific improvements"
                            ),
                        })
                elif ctype == "data_scarcity":
                    constraints.append({
                        "type": "data_scarcity",
                        "detail": "Data scarcity mentioned in brief",
                        "recommendation": "Architecture changes alone cannot overcome data limitations",
                    })
                elif ctype == "data_imbalance":
                    constraints.append({
                        "type": "data_imbalance",
                        "detail": "Data imbalance mentioned in brief",
                        "recommendation": "Use domain-balanced sampling and per-domain metrics",
                    })

        return constraints

    def _detect_implemented_methods(self) -> list[str]:
        """Scan the codebase to detect which methods are actually implemented."""
        methods = []
        models_dir = self.project_dir / "models"
        if not models_dir.exists():
            return methods

        try:
            for py_file in models_dir.glob("*.py"):
                try:
                    content = py_file.read_text().lower()
                    for key, props in self.METHOD_PROPERTIES.items():
                        for pat in props.get("patterns", [key]):
                            if pat in content:
                                if key not in methods:
                                    methods.append(key)
                                break
                except Exception:
                    continue
        except Exception:
            pass

        return methods

    def _build_cross_experiment_insights(self) -> dict:
        """Integrate knowledge across multiple experiments to find meta-patterns.

        Uses STRUCTURED SQLite queries instead of regex on markdown text.
        Falls back to regex only if database is empty.
        """
        insights = {
            "method_effect_matrix": {},
            "stuck_domains": [],
            "meta_patterns": [],
        }

        # ── PRIMARY: Structured SQLite queries ──
        try:
            effect_matrix = self.memory.get_method_domain_effect_matrix()
            if effect_matrix:
                insights["method_effect_matrix"] = effect_matrix

                # Find dominated methods (never best for any domain)
                all_domains = set()
                for m, domains in effect_matrix.items():
                    all_domains.update(domains.keys())

                best_per_domain = {}
                for domain in all_domains:
                    best_method = min(
                        ((m, effect_matrix[m][domain]["best_mae"])
                         for m in effect_matrix if domain in effect_matrix[m]),
                        key=lambda x: x[1],
                        default=(None, float('inf'))
                    )
                    if best_method[0]:
                        best_per_domain[domain] = best_method

                # Check if any method is best for ALL domains (breakthrough)
                if best_per_domain:
                    method_wins = {}
                    for domain, (method, mae) in best_per_domain.items():
                        method_wins[method] = method_wins.get(method, 0) + 1
                    dominant = [m for m, wins in method_wins.items() if wins == len(all_domains)]
                    if dominant:
                        insights["meta_patterns"].append(
                            f"DOMINANT METHOD: {dominant[0]} is best for ALL domains. "
                            f"If further improvements are needed, consider architectural "
                            f"modifications to this method rather than trying new methods."
                        )

            # Structured stuck domain detection
            stuck_threshold = 0.30  # default
            if hasattr(self, 'adaptive_thresholds'):
                try:
                    stuck_threshold = self.adaptive_thresholds.get_thresholds().get("domain_gap_critical", 0.30)
                except Exception:
                    pass
            stuck = self.memory.get_stuck_domains_structured(threshold=stuck_threshold)
            if stuck:
                insights["stuck_domains"] = stuck
                stuck_names = [d["domain"] for d in stuck]
                insights["meta_patterns"].append(
                    f"DOMAINS STUCK ({', '.join(stuck_names)}): These domains resist ALL methods tried. "
                    f"This is NOT a hyperparameter problem — it's a method assumption violation. "
                    f"The next experiment MUST use a method whose assumptions are satisfied by "
                    f"these domains. Consider: per-view encoding (no angular assumption), "
                    f"domain-specific feature extraction (not prediction heads), or "
                    f"non-parametric angular aggregation."
                )

            # Calibration feedback
            calibration = self.memory.get_experiment_calibration()
            if calibration.get("total_hypotheses", 0) >= 3:
                acc = calibration["accuracy"]
                if acc < 0.3:
                    insights["meta_patterns"].append(
                        f"LOW HYPOTHESIS ACCURACY ({acc:.0%}): Only {acc:.0%} of your hypotheses "
                        f"have been correct. Your mental model of what works may be WRONG. "
                        f"Consider: (1) re-reading the literature for missed assumptions, "
                        f"(2) using analyze_model/probe_model BEFORE proposing changes, "
                        f"(3) running pilot experiments (2 epochs) before full training."
                    )

        except Exception as e:
            logger.debug(f"Structured meta-pattern query failed, falling back to regex: {e}")
            # ── FALLBACK: regex-based detection (old behavior) ──
            try:
                log_text = self.memory.get_log()
            except Exception:
                return {}

            method_keywords = self.memory.method_keywords

            if hasattr(self, '_best_domain_metrics') and self._best_domain_metrics:
                for domain_key, best_val in self._best_domain_metrics.items():
                    domain_name = domain_key.replace("MAE_", "")
                    if best_val > stuck_threshold:
                        insights["stuck_domains"].append({
                            "domain": domain_name,
                            "best_mae": round(best_val, 4),
                            "status": "STUCK",
                            "implication": "Core method assumption likely violated.",
                        })

            dead_end_methods = set()
            for method_key, keywords in method_keywords.items():
                for kw in keywords:
                    if re.search(rf"\bdead.{0,5}end\b.*\b{kw}\b", log_text, re.IGNORECASE) or \
                       re.search(rf"\b{kw}\b.*\bdead.{0,5}end\b", log_text, re.IGNORECASE):
                        dead_end_methods.add(method_key)

            if dead_end_methods:
                insights["meta_patterns"].append(
                    f"EXHAUSTED METHODS: {', '.join(dead_end_methods)} have been tried and failed."
                )

        # ── v14: Architecture-level dead end synthesis ──
        self._synthesize_architecture_dead_ends(insights)

        return insights if insights["meta_patterns"] or insights["stuck_domains"] else {}

    # ── v14: Architecture-level dead end synthesis ──
    # Known architecture patterns for clustering dead ends.
    _ARCH_PATTERNS = {
        "epi": ["epi", "epinet", "epipolar", "epi slope", "epi branch"],
        "unet": ["unet", "u-net", "u_net"],
        "transformer": ["transformer", "vit", "self_attention"],
        "cnn": ["resnet", "vgg", "mobilenet", "efficientnet"],
        "graph": ["gnn", "graph", "gcn", "gat"],
        "lfnet": ["lfnet", "lf_net"],
        "oacc": ["oacc", "occlusion_aware"],
        "mvsnet": ["mvsnet", "multi_view_stereo"],
    }

    def _synthesize_architecture_dead_ends(self, insights: dict) -> None:
        """Cluster dead ends by architecture and detect architecture bottlenecks (v14).

        When 5+ dead ends cluster around the same architecture, this indicates
        the architecture itself is the bottleneck, not individual approaches.
        Injects a strong meta-pattern into insights.
        """
        try:
            all_dead_ends = self.memory.get_dead_ends_full()
            if len(all_dead_ends) < 5:
                return

            # Cluster dead ends by architecture
            arch_clusters: dict[str, list[str]] = {}
            unclustered = []
            for de_text in all_dead_ends:
                de_lower = de_text.lower()
                matched = False
                for arch_key, patterns in self._ARCH_PATTERNS.items():
                    for pat in patterns:
                        if pat in de_lower:
                            arch_clusters.setdefault(arch_key, []).append(de_text)
                            matched = True
                            break
                    if matched:
                        break
                if not matched:
                    unclustered.append(de_text)

            # Detect architecture bottlenecks (5+ dead ends for same architecture)
            for arch_key, des in arch_clusters.items():
                if len(des) >= 5:
                    # Further analyze: what components were attempted?
                    component_keywords = {
                        "loss": 0, "attention": 0, "conv": 0, "stream": 0,
                        "branch": 0, "augment": 0, "pretrain": 0, "norm": 0,
                        "head": 0, "feature": 0, "fusion": 0, "skip": 0,
                    }
                    for de in des:
                        de_lower = de.lower()
                        for kw in component_keywords:
                            if kw in de_lower:
                                component_keywords[kw] += 1

                    # Find the most-patched components
                    attempted_components = [
                        f"{kw} ({count} attempts)"
                        for kw, count in sorted(component_keywords.items(), key=lambda x: -x[1])
                        if count > 0
                    ][:5]

                    insights["meta_patterns"].append(
                        f"ARCHITECTURE BOTTLENECK DETECTED (v14): "
                        f"'{arch_key}' architecture has {len(des)} dead ends across "
                        f"{len(all_dead_ends)} total dead ends ({len(des)*100//len(all_dead_ends)}% of all failures). "
                        f"Components attempted: {', '.join(attempted_components)}. "
                        f"DIAGNOSIS: The architecture itself is the bottleneck — "
                        f"no amount of component-level patching will fix this. "
                        f"A fundamentally different architecture is required."
                    )
                    insights["architecture_bottleneck"] = {
                        "architecture": arch_key,
                        "dead_end_count": len(des),
                        "total_dead_ends": len(all_dead_ends),
                        "attempted_components": attempted_components,
                    }
        except Exception as e:
            logger.debug(f"Architecture dead end synthesis failed: {e}")
