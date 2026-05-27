"""
AutoResearcher Visual Analyzer — Inference & Visual Analysis Module

When training results are consistently poor (consecutive failures),
this module triggers a visual analysis pipeline:
  1. Run inference to generate prediction outputs (depth maps, etc.)
  2. Send prediction images to multimodal LLM via MCP for visual analysis
  3. Analyze WHY the model is failing (not just numeric metrics)
  4. Return structured diagnosis for the REFLECT phase

Design principle:
- Three separate vision providers with distinct keys/endpoints:
  - GLM Coding Plan: MCP (zai-mcp-server) → glm-5v-turbo → glm-4.6v
  - Ali Token Plan: qwen3.6-plus (with cross-provider backup to DashScope for qwen3.5-plus)
  - Ali DashScope: qwen3.6-plus → qwen3.5-plus
- Primary LLM (GLM-5.1) is text-only; vision models (glm-5v-turbo, glm-4.6v) are separate
- MCP server (zai-mcp-server) runs locally via npx stdio — images sent as local paths or base64 data URLs
"""

import os
import json
import base64
import logging
import subprocess
import threading
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("autoresearcher.visual_analyzer")


@dataclass
class VisualAnalysisResult:
    """Result of visual analysis on inference outputs."""
    triggered: bool = False
    inference_run: bool = False
    images_analyzed: int = 0
    diagnosis: list[str] = field(default_factory=list)
    severity: str = "info"  # info | warning | critical
    recommended_actions: list[str] = field(default_factory=list)
    raw_response: str = ""
    mcp_used: bool = False
    fallback_used: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "triggered": self.triggered,
            "inference_run": self.inference_run,
            "images_analyzed": self.images_analyzed,
            "diagnosis": self.diagnosis,
            "severity": self.severity,
            "recommended_actions": self.recommended_actions,
            "raw_response": self.raw_response[:500],
            "mcp_used": self.mcp_used,
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


class VisualAnalyzer:
    """Visual analysis module for diagnosing model failures through inference outputs.

    Trigger condition: consecutive poor training results (default: 5 cycles).

    Pipeline:
    1. Find best checkpoint and run inference on validation scenes
    2. Collect output images (depth maps, predictions, visualizations)
    3. Send images to multimodal LLM for visual analysis
    4. Parse analysis into structured diagnosis for REFLECT phase
    """

    def __init__(
        self,
        project_dir: Path,
        workspace: Path,
        config: dict = None,
        tools_registry=None,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.workspace = Path(workspace)
        self._tools_registry = tools_registry  # ToolRegistry for reusing MCP sessions

        # Configuration with defaults
        va_config = (config or {}).get("visual_analysis", {})
        self.trigger_threshold: int = va_config.get("trigger_threshold", 5)
        self.mcp_timeout: int = va_config.get("mcp_timeout", 60)  # seconds
        self.max_images_per_analysis: int = va_config.get("max_images", 6)
        self.enabled: bool = va_config.get("enabled", True)

        # MCP configuration (requires GLM key)
        mcp_config = (config or {}).get("multimodal_mcp", {})
        self.mcp_enabled: bool = mcp_config.get("enabled", True)
        self.mcp_env_key: str = mcp_config.get("api_key_env", "GLM_CODING_PLAN_API_KEY")
        self.mcp_mode: str = mcp_config.get("mode", "ZHIPU")

        # ── Visual Provider Detection ──
        # Separate keys with different endpoints, NO mixing between providers.
        self._detect_visual_providers()

    def _detect_visual_providers(self):
        """Detect available visual analysis providers and their fallback chains.

        Each provider has its own API key and endpoint — NO mixing.
        Priority: GLM Coding Plan > Ali Token Plan > Ali DashScope

        GLM Coding Plan (GLM_CODING_PLAN_API_KEY):
            endpoint: https://open.bigmodel.cn/api/coding/paas/v4
            chain: MCP zai-mcp-server → glm-5v-turbo → glm-4.6v

        Ali Token Plan (ALI_TOKEN_PLAN_API_KEY):
            endpoint: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
            chain: qwen3.6-plus (only; qwen3.5-plus not on token plan)
            Fallback: try ALI_API_KEY on dashscope endpoint for qwen3.5-plus

        Ali DashScope (ALI_API_KEY):
            endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1
            chain: qwen3.6-plus → qwen3.5-plus
        """
        VISION_PROVIDERS = {
            "glm_coding_plan": {
                "env_key": "GLM_CODING_PLAN_API_KEY",
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
                "models": ["glm-5v-turbo", "glm-4.6v"],
                "mcp_enabled": True,
                "label": "GLM (Coding Plan)",
            },
            "ali_token_plan": {
                "env_key": "ALI_TOKEN_PLAN_API_KEY",
                "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                "models": ["qwen3.6-plus", "qwen3.6-flash"],  # Vision-capable models
                "mcp_enabled": False,
                "label": "Ali (Token Plan)",
            },
            "ali_dashscope": {
                "env_key": "ALI_API_KEY",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "models": ["qwen3.6-plus", "qwen3.5-plus"],
                "mcp_enabled": False,
                "label": "Ali (DashScope)",
            },
        }

        self._provider_config = None
        self._backup_provider = None  # For cross-provider fallback (e.g., token plan → dashscope)

        for name, cfg in VISION_PROVIDERS.items():
            key = os.environ.get(cfg["env_key"], "")
            if key:
                self._provider_config = {
                    **cfg,
                    "name": name,
                    "api_key": key,
                }
                # For ali_token_plan, set dashscope as cross-provider backup (for qwen3.5-plus)
                if name == "ali_token_plan":
                    dash_key = os.environ.get("ALI_API_KEY", "")
                    if dash_key:
                        self._backup_provider = {
                            **VISION_PROVIDERS["ali_dashscope"],
                            "name": "ali_dashscope",
                            "api_key": dash_key,
                            "models": ["qwen3.5-plus"],  # Only use dashscope for the missing model
                        }
                break  # First available wins

        if self._provider_config:
            self._api_base = self._provider_config["base_url"]
            self._api_key = self._provider_config["api_key"]
            self._models = self._provider_config["models"]
            self.mcp_enabled = self.mcp_enabled and self._provider_config.get("mcp_enabled", False)
            logger.info(
                f"VisualAnalyzer: provider={self._provider_config['label']} "
                f"endpoint={self._api_base} mcp={self.mcp_enabled} "
                f"models={self._models}"
            )
            if self._backup_provider:
                logger.info(
                    f"VisualAnalyzer: cross-provider backup={self._backup_provider['label']} "
                    f"models={self._backup_provider['models']}"
                )
        else:
            self._api_base = ""
            self._api_key = ""
            self._models = []
            self.mcp_enabled = False
            logger.warning(
                "VisualAnalyzer: NO API key found. "
                "Set GLM_CODING_PLAN_API_KEY, ALI_TOKEN_PLAN_API_KEY, or ALI_API_KEY."
            )

    def should_trigger(self, no_progress_streak: int) -> bool:
        """Check if visual analysis should be triggered."""
        if not self.enabled:
            return False
        return no_progress_streak >= self.trigger_threshold

    def analyze(
        self,
        no_progress_streak: int,
        best_checkpoint_path: Optional[str] = None,
        experiment_info: Optional[dict] = None,
    ) -> VisualAnalysisResult:
        """Run full visual analysis pipeline.

        Args:
            no_progress_streak: Number of consecutive poor training cycles
            best_checkpoint_path: Path to best model checkpoint (auto-detected if None)
            experiment_info: Dict with experiment context (model name, dataset, etc.)

        Returns:
            VisualAnalysisResult with structured diagnosis
        """
        result = VisualAnalysisResult(triggered=True)

        if not self.should_trigger(no_progress_streak):
            result.triggered = False
            return result

        logger.info(
            f"VISUAL ANALYSIS TRIGGERED: {no_progress_streak} consecutive poor cycles. "
            f"Running inference + visual diagnosis..."
        )

        # Step 1: Run inference to generate outputs
        try:
            inference_outputs = self._run_inference(best_checkpoint_path, experiment_info)
            if not inference_outputs:
                result.error = "Inference produced no outputs — cannot analyze visually"
                result.severity = "warning"
                return result
            result.inference_run = True
        except Exception as e:
            result.error = f"Inference failed: {str(e)[:300]}"
            result.severity = "critical"
            logger.error(f"Visual analysis inference failed: {e}")
            return result

        # Step 2: Collect images for analysis
        images = self._collect_analysis_images(inference_outputs, self.max_images_per_analysis)
        if not images:
            result.error = "No analyzable images found in inference outputs"
            result.severity = "warning"
            return result

        result.images_analyzed = len(images)
        logger.info(f"Collected {len(images)} images for visual analysis")

        # Step 3: Send to multimodal LLM for analysis
        try:
            analysis_result = self._analyze_images(images, experiment_info)
            result.diagnosis = analysis_result.get("diagnosis", [])
            result.recommended_actions = analysis_result.get("recommended_actions", [])
            result.severity = analysis_result.get("severity", "info")
            result.raw_response = analysis_result.get("raw_response", "")
            result.mcp_used = analysis_result.get("mcp_used", False)
            result.fallback_used = analysis_result.get("fallback_used", False)

            logger.info(
                f"Visual analysis complete: severity={result.severity}, "
                f"{len(result.diagnosis)} findings, mcp={result.mcp_used}"
            )
        except Exception as e:
            result.error = f"Multimodal analysis failed: {str(e)[:300]}"
            result.severity = "warning"
            logger.error(f"Visual analysis failed: {e}")

        return result

    def _run_inference(
        self,
        checkpoint_path: Optional[str],
        experiment_info: Optional[dict],
    ) -> list[Path]:
        """Run inference script to generate prediction outputs.

        Strategy:
        1. If checkpoint_path provided, use it directly
        2. Otherwise, find best checkpoint from outputs/ directory
        3. Look for existing inference scripts (scripts/inference_*.py)
        4. If no script exists, create a minimal one based on project structure

        Returns:
            List of paths to generated output files (images, npy, etc.)
        """
        # Find best checkpoint if not provided
        if not checkpoint_path:
            checkpoint_path = self._find_best_checkpoint()
        if not checkpoint_path or not Path(checkpoint_path).exists():
            logger.warning(f"No valid checkpoint found: {checkpoint_path}")
            return []

        # Look for existing inference script
        scripts_dir = self.project_dir / "scripts"
        inference_script = None
        if scripts_dir.exists():
            for script in scripts_dir.glob("inference*.py"):
                inference_script = script
                break

        if not inference_script:
            # Create minimal inference script based on detected project structure
            logger.info("No inference script found, attempting auto-detection...")
            inference_script = self._detect_or_create_inference_script()

        if not inference_script or not inference_script.exists():
            logger.warning("Cannot create inference script for this project type")
            return []

        # Run inference
        output_dir = self.workspace / "visual_analysis_outputs"
        output_dir.mkdir(exist_ok=True)

        try:
            cmd = [
                "python",
                str(inference_script),
                "--checkpoint", str(checkpoint_path),
                "--output_dir", str(output_dir),
            ]

            # Try to limit inference to a few samples for quick analysis
            cmd.extend(["--max_scenes", str(self.max_images_per_analysis)])

            logger.info(f"Running inference: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min max for inference
                cwd=str(self.project_dir),
            )

            if result.returncode != 0:
                logger.warning(f"Inference returned non-zero: {result.stderr[:500]}")
            else:
                logger.info(f"Inference completed successfully")

        except subprocess.TimeoutExpired:
            logger.warning("Inference timed out after 10 minutes")
        except FileNotFoundError:
            logger.warning(f"Inference script not found: {inference_script}")
            return []
        except Exception as e:
            logger.error(f"Inference execution error: {e}")
            return []

        # Collect all generated outputs
        outputs = []
        if output_dir.exists():
            # Prioritize image files
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp"):
                outputs.extend(output_dir.rglob(ext))
            # Also include .npy files for numeric inspection
            outputs.extend(output_dir.rglob("*.npy"))

        # Also check common output locations
        for pattern in [
            self.project_dir / "outputs" / "**" / "*_depth.png",
            self.project_dir / "outputs" / "**" / "*prediction*",
        ]:
            for p in self.project_dir.glob(str(pattern)):
                if p not in outputs:
                    outputs.append(p)

        return outputs

    def _find_best_checkpoint(self) -> Optional[str]:
        """Find the best model checkpoint from outputs/ directory."""
        candidates = [
            self.project_dir / "outputs" / exp / "best_checkpoint.pt"
            for exp in sorted((self.project_dir / "outputs").glob("exp_*"))
        ]
        # Also check direct outputs path
        candidates.append(self.project_dir / "outputs" / "best_checkpoint.pt")

        # Pick the most recently modified one
        valid_candidates = [c for c in candidates if c.exists()]
        if valid_candidates:
            best = max(valid_candidates, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found best checkpoint: {best} ({best.stat().st_mtime})")
            return str(best)
        return None

    def _detect_or_create_inference_script(self) -> Optional[Path]:
        """Detect project type and locate/create appropriate inference script."""
        # Check if this looks like a standard ML project
        models_dir = self.project_dir / "models"
        datasets_dir = self.project_dir / "datasets"

        if models_dir.exists() and datasets_dir.exists():
            # Check for existing inference scripts
            scripts_dir = self.project_dir / "scripts"
            if scripts_dir.exists():
                for script in scripts_dir.glob("*.py"):
                    content = script.read_text(errors="ignore")
                    if "inference" in content.lower() and "def main" in content:
                        return script

        return None

    def _collect_analysis_images(
        self, outputs: list[Path], max_count: int
    ) -> list[dict]:
        """Collect image data for multimodal analysis.

        Returns:
            List of dicts with keys: path, b64_data, filename, description
        """
        images = []
        seen_basenames = set()

        for output_path in outputs:
            if len(images) >= max_count:
                break

            # Skip duplicate basenames (same scene, different format)
            basename = output_path.stem
            if basename in seen_basenames:
                continue

            # Only process image files for multimodal analysis
            if output_path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
                continue

            try:
                b64_data = base64.b64encode(output_path.read_bytes()).decode("utf-8")
                images.append({
                    "path": str(output_path),
                    "b64_data": b64_data,
                    "filename": output_path.name,
                    "description": self._generate_image_description(output_path),
                })
                seen_basenames.add(basename)
            except Exception as e:
                logger.warning(f"Failed to read image {output_path}: {e}")

        # Sort: prefer depth maps first, then other visualizations
        priority_keywords = ["depth", "pred", "output", "result"]
        images.sort(key=lambda x: (
            0 if any(k in x["filename"].lower() for k in priority_keywords) else 1,
            x["filename"]
        ))

        return images[:max_count]

    def _generate_image_description(self, img_path: Path) -> str:
        """Generate a description of what the image represents."""
        name_lower = img_path.name.lower()
        parent = img_path.parent.name if img_path.parent else ""

        desc_parts = []
        if "depth" in name_lower:
            desc_parts.append("predicted depth map")
        elif "gt" in name_lower or "ground_truth" in name_lower:
            desc_parts.append("ground truth depth map")
        elif "input" in name_lower or "rgb" in name_lower:
            desc_parts.append("input RGB image")
        elif "error" in name_lower:
            desc_parts.append("error visualization")
        elif "disp" in name_lower:
            desc_parts.append("disparity map")
        else:
            desc_parts.append("model output visualization")

        if parent and parent != "visual_analysis_outputs":
            desc_parts.append(f"from experiment '{parent}'")

        return " ".join(desc_parts)

    def _analyze_images(
        self,
        images: list[dict],
        experiment_info: Optional[dict] = None,
    ) -> dict:
        """Send images to multimodal LLM for visual analysis.

        Fallback chain:
        - GLM key available: MCP zai-mcp-server → GLM-5V-Turbo → GLM-4.6V
        - Qwen key available: Qwen3.6-plus → Qwen3.5-plus (native multimodal)
        """
        prompt = self._build_analysis_prompt(images, experiment_info)

        # Prepare image data for multimodal API call
        image_contents = []
        for img in images:
            mime_type = self._guess_mime_type(img["filename"])
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{img['b64_data']}",
                    "detail": "high",
                },
            })

        result = {
            "diagnosis": [],
            "recommended_actions": [],
            "severity": "info",
            "raw_response": "",
            "mcp_used": False,
            "fallback_used": False,
        }

        if not self._api_key or not self._models:
            result["error"] = "No API key or models configured"
            result["severity"] = "warning"
            return result

        # ── MCP first (GLM only) ──
        if self.mcp_enabled and self._provider_config["name"] == "glm_coding_plan":
            try:
                mcp_result = self._call_zai_mcp_vision(images, experiment_info)
                if mcp_result:
                    parsed = self._parse_analysis_response(mcp_result)
                    result.update(parsed)
                    result["mcp_used"] = True
                    return result
            except Exception as e:
                logger.warning(f"MCP zai-mcp-server failed: {e}, falling back...")

        # ── Try each model in order ──
        for i, model_name in enumerate(self._models):
            try:
                api_result = self._call_multimodal_model(
                    model_name, prompt, image_contents,
                    base_url=self._api_base,
                )
                if api_result:
                    parsed = self._parse_analysis_response(api_result)
                    result.update(parsed)
                    if not result.get("mcp_used"):
                        result["fallback_used"] = True
                    return result
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")

        # ── Cross-provider fallback (e.g., Ali Token Plan → DashScope for qwen3.5-plus) ──
        if self._backup_provider:
            for model_name in self._backup_provider["models"]:
                try:
                    logger.info(f"Trying cross-provider: {self._backup_provider['label']}/{model_name}")
                    api_result = self._call_multimodal_model(
                        model_name, prompt, image_contents,
                        base_url=self._backup_provider["base_url"],
                        api_key=self._backup_provider["api_key"],
                    )
                    if api_result:
                        parsed = self._parse_analysis_response(api_result)
                        result.update(parsed)
                        result["fallback_used"] = True
                        return result
                except Exception as e:
                    logger.warning(f"Cross-provider {model_name} failed: {e}")

        result["error"] = f"All models failed: {self._models}"
        result["severity"] = "warning"
        return result

    def _call_zai_mcp_vision(self, images: list[dict],
                             experiment_info: Optional[dict] = None) -> Optional[str]:
        """Call zai-mcp-server vision tools via ToolRegistry's MCP stdio transport.

        Reuses the ToolRegistry's cached stdio session instead of spawning a new
        subprocess each time. Falls back to direct spawn if ToolRegistry unavailable.
        """
        import time

        api_key = os.getenv(self.mcp_env_key)
        if not api_key:
            logger.warning(f"MCP API key not found: env var {self.mcp_env_key}")
            return None

        # Build analysis prompt — emphasize real analysis, not generic description
        combined_prompt = (
            "You are analyzing images from a deep learning experiment. "
            "Provide CONCRETE, SPECIFIC observations — do NOT give generic descriptions.\n\n"
            "For each image:\n"
            "1. Describe the ACTUAL pixel content you see (colors, patterns, shapes, spatial structure)\n"
            "2. If this is a depth/prediction map: Is it uniform/blank? Are there visible gradients? "
            "Does it show meaningful spatial structure or just noise?\n"
            "3. If comparing prediction vs ground truth: What are the visible DIFFERENCES? "
            "Is the prediction blurrier, more uniform, offset, or distorted compared to GT?\n"
            "4. Quantify where possible (e.g., 'mostly uniform blue with a small bright region in center', "
            "'prediction shows similar structure to GT but with loss of fine detail at edges')\n"
            "5. Likely root cause if the output appears problematic"
        )
        if experiment_info:
            info_parts = [f"{k}: {v}" for k, v in experiment_info.items()]
            combined_prompt += f"\n\nExperiment context: {'; '.join(info_parts)}"

        # Use ToolRegistry's MCP session if available (reuse subprocess)
        if self._tools_registry and hasattr(self._tools_registry, '_mcp_sessions'):
            return self._call_zai_via_tools_registry(images, combined_prompt)

        # No ToolRegistry available — cannot call MCP
        logger.warning("ToolRegistry not available for MCP vision call")
        return None

    def _call_zai_via_tools_registry(self, images: list[dict],
                                      combined_prompt: str) -> Optional[str]:
        """Call zai-mcp-server via ToolRegistry's cached stdio session."""
        import time

        # Ensure zai_vision session exists
        if "zai_vision" not in self._tools_registry.mcp_available:
            logger.info("zai_vision not in ToolRegistry, cannot call MCP vision")
            return None

        session = self._tools_registry._mcp_sessions.get("zai_vision")
        if not session or session.get("transport") != "stdio":
            # Try to initialize
            session = self._tools_registry._mcp_stdio_initialize("zai_vision")
            if not session:
                return None
            self._tools_registry._mcp_sessions["zai_vision"] = session

        all_results = []
        for img in images:
            filename_lower = img["filename"].lower()
            if "error" in filename_lower:
                tool_name = "diagnose_error_screenshot"
            elif any(k in filename_lower for k in ("chart", "plot", "curve", "loss", "metric")):
                tool_name = "analyze_data_visualization"
            else:
                tool_name = "analyze_image"

            # Use local path (stdio server can read local files)
            image_source = img.get("path", "")
            if not image_source or not Path(image_source).exists():
                mime_type = self._guess_mime_type(img["filename"])
                image_source = f"data:{mime_type};base64,{img['b64_data']}"

            result = self._tools_registry._mcp_stdio_call(
                "zai_vision", tool_name,
                {"image_source": image_source, "prompt": combined_prompt},
                timeout=self.mcp_timeout,
            )
            if result:
                all_results.append(f"[{img['filename']}] ({tool_name}):\n{result}")

        if not all_results:
            return None

        combined = "\n\n".join(all_results)
        return json.dumps({
            "diagnosis": [{
                "category": "visual_mcp_analysis",
                "confidence": "high",
                "description": combined[:4000],
                "image_evidence": [img["filename"] for img in images],
            }],
            "recommended_actions": [],
            "severity": "info",
            "summary": f"MCP vision analysis of {len(images)} images via zai-mcp-server (stdio, reused session)",
        })

    @staticmethod
    def _extract_mcp_text(result: dict) -> Optional[str]:
        """Extract text content from an MCP JSON-RPC result dict."""
        content_blocks = result.get("content", [])
        if result.get("isError"):
            return None
        text_parts = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts) if text_parts else None


    def _call_multimodal_model(self, model_name: str, prompt: str,
                              image_contents: list,
                              base_url: str = None, api_key: str = None) -> Optional[str]:
        """Call a multimodal model via OpenAI-compatible API.

        Uses self._api_base/_api_key by default. Pass base_url/api_key to override
        for cross-provider fallback calls.
        """
        import openai

        url = base_url or self._api_base
        key = api_key or self._api_key

        if not key or not url:
            logger.warning("_call_multimodal_model: no API key or base URL")
            return None

        client = openai.OpenAI(api_key=key, base_url=url)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *image_contents,
            ],
        }]

        logger.info(f"Calling multimodal: {model_name} ({url})")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=4096,
            timeout=self.mcp_timeout * 2,
        )
        return response.choices[0].message.content if response.choices else None

    def _build_analysis_prompt(self, images: list[dict], experiment_info: Optional[dict]) -> str:
        """Build the analysis prompt for multimodal LLM."""
        info_str = ""
        if experiment_info:
            parts = []
            for k, v in experiment_info.items():
                parts.append(f"- {k}: {v}")
            info_str = "\n".join(parts)

        image_list = "\n".join(
            f"  [{i+1}] {img['filename']}: {img['description']}"
            for i, img in enumerate(images)
        )

        return f"""You are an expert deep learning researcher analyzing why a model is failing.

## Task
Analyze the following {len(images)} model output images and diagnose WHY the model predictions are wrong.

## Experiment Context
{info_str or 'Context not provided'}

## Images to Analyze
{image_list}

## Analysis Requirements

For EACH image, you MUST examine:
1. **Visual quality**: Is the output structurally correct? Or is it uniform/garbage?
   - Describe ACTUAL pixel content you see (colors, gradients, shapes, patterns)
   - Do NOT give generic descriptions like "shows a depth map" — say what colors/values are where
2. **Value range**: Are values in a reasonable range? (e.g., depth should vary spatially)
3. **Artifact detection**: Any obvious patterns indicating specific failure modes?
4. **Prediction vs GT comparison**: If both prediction and ground truth images are present,
   compare them directly — is the prediction blurrier? More uniform? Offset? Missing fine details?
   What specifically differs about the spatial structure?

Then provide a consolidated diagnosis:

### Diagnosis Categories (pick ALL that apply):
- **Data problem**: Training data doesn't match test distribution, insufficient samples, wrong normalization
- **Architecture problem**: Model capacity insufficient, missing components, wrong inductive bias
- **Training problem**: Loss function mismatched, learning rate issues, underfitting/overfitting
- **Domain gap**: Model trained on one domain but evaluated on another (e.g., domain A vs domain B with different data distributions)
- **GT/label problem**: Ground truth may be incorrect or missing for some samples
- **Resolution/input problem**: Input preprocessing losing information, resize artifacts

## Response Format (STRICT JSON)
```json
{{
  "diagnosis": [
    {{
      "category": "domain_gap|data_problem|architecture_problem|training_problem|gt_problem|input_problem",
      "confidence": "high|medium|low",
      "description": "Detailed explanation of what's wrong and evidence from the images",
      "image_evidence": ["list of image filenames that support this finding"]
    }}
  ],
  "recommended_actions": [
    "Specific, actionable fix #1",
    "Specific, actionable fix #2"
  ],
  "severity": "critical|warning|info",
  "summary": "One-paragraph plain-language summary for a human researcher"
}}
```

CRITICAL: Base your analysis ONLY on what you can SEE in these images. Do not guess."""

    @staticmethod
    def _guess_mime_type(filename: str) -> str:
        """Guess MIME type from file extension."""
        ext = Path(filename).suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        return mime_map.get(ext, "image/png")

    @staticmethod
    def _parse_analysis_response(response_text: str) -> dict:
        """Parse structured analysis response from LLM."""
        result = {
            "diagnosis": [],
            "recommended_actions": [],
            "severity": "info",
            "raw_response": response_text,
        }

        if not response_text:
            return result

        # Try to extract JSON from response
        try:
            # Find JSON block in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)

                result["diagnosis"] = data.get("diagnosis", [])
                result["recommended_actions"] = data.get("recommended_actions", [])
                result["severity"] = data.get("severity", "info")

                # Add summary as first diagnosis item if present
                summary = data.get("summary", "")
                if summary:
                    result["diagnosis"].insert(0, {
                        "category": "overall",
                        "confidence": "high",
                        "description": summary,
                        "image_evidence": [],
                    })

                return result
        except json.JSONDecodeError:
            pass

        # Fallback: parse text into structured format
        lines = response_text.strip().split("\n")
        result["diagnosis"] = [{
            "category": "text_analysis",
            "confidence": "medium",
            "description": response_text[:1000],
            "image_evidence": [],
        }]
        result["severity"] = "warning"

        # Extract action items
        for line in lines:
            lower = line.lower()
            if any(kw in lower for kw in ("recommend", "should", "fix", "try", "action")):
                result["recommended_actions"].append(line.strip())

        return result
