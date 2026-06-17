"""
AutoResearcher Tool Registry

Each agent gets a minimal tool set (3-5 tools) instead of all tools.
This reduces token overhead per API call significantly.
"""

import ast
import os
import re
import sys
import time
import subprocess
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from .mcp_client import MCPClientMixin
from .model_analyzer import ModelAnalyzerMixin

logger = logging.getLogger("autoresearcher.tools")


class ToolRegistry(MCPClientMixin, ModelAnalyzerMixin):
    """Manages tools available to agents.

    Design principle: minimal tool sets per agent.
    - Leader: log_memory, write_file, read_file (3 tools)
    - Idea Agent: search_papers, get_paper, write_file, read_file (4 tools)
    - Code Agent: run_shell, launch_experiment, write_file, read_file, list_files (5 tools)
    - Researcher: search_papers, web_search, web_fetch, write_file, read_file, list_files (6 tools)
    - Writing Agent: write_file, read_file, list_files (3 tools)

    Fewer tools = fewer tokens in each API call = lower cost.

    Literature search fallback chain:
    - search_papers: MCP web search only
    - get_paper: MCP web_reader only
    - web_search: MCP web search only
    - web_fetch: MCP web_reader → urllib direct fetch
    """

    def __init__(self, workspace: Path, memory=None, config: dict = None):
        self.workspace = Path(workspace).resolve()
        self._memory = memory  # Optional MemoryManager reference for log_memory tool
        # Phase 2: safety config (tool-level contracts)
        self._mandatory_dry_run = (config or {}).get("safety", {}).get("mandatory_dry_run", False)

        # Protected files and directories — always initialized
        self._protected_files = {
            "state.json", "MEMORY_LOG.md", "PROJECT_BRIEF.md", ".lock",
            "DIRECTIVE.md", "HUMAN_DIRECTIVE.md", "AGENT_STUCK.md",
            "config.yaml",  # Project config
            "autoresearcher.log",  # Agent's own log (code agent must not delete this)
        }
        # Patterns for directories that should never be written by agents.
        # When workspace == project root, agents must not overwrite core code.
        self._protected_dirs = {"models", "datasets", "data", "scripts"}
        # Blocked Python patterns for run_python safety
        self._blocked_py_patterns = [
            r"\bos\.system\b",
            r"\bsubprocess\b",
            r"\bos\.remove\b",
            r"\bos\.unlink\b",
            r"\bshutil\.rmtree\b",
            r"\bos\.environ\b",
            r"\bos\.getenv\b",
            r"\bos\.popen\b",
            r"\bos\.listdir\b",
            r"\b__import__\b",
            r"\bimportlib\b",
            r"\bgetattr\s*\(\s*__builtins__",
            # Block getattr on os/module to prevent bypass via string concatenation
            # e.g. getattr(os, 'env'+'iron'), getattr(__import__('os'), 'system')
            r"\bgetattr\s*\(.+os\b",
            r"\bgetattr\s*\(.+__import__",
            r"\bgetattr\s*\(.+subprocess",
            r"\bctypes\b",
            r"\bcompile\s*\(",
            r"\beval\s*\(",
            r"\bexec\s*\(",
            r"\bbreakpoint\s*\(",
            r"\bopen\s*\(.+[\"']w[b]?\b[\"']",
            r"\.write_text\s*\(",
            r"\.write_bytes\s*\(",
            # Block string concatenation bypass: 'o'+'s' used to evade keyword detection
            r"[\"']o[\"']\s*\+\s*[\"']s[\"']",
            r"[\"']sy[\"']\s*\+\s*[\"']stem[\"']",
            # Block os access via chr() or bytes trickery
            r"\bchr\s*\(\s*\d+\s*\)\s*\+",
        ]

        # ── MCP service availability detection & session management ──
        self._mcp_init_fields()

    def shutdown(self):
        """Clean up all MCP sessions: close SSE connections, kill stdio subprocesses."""
        self.shutdown_mcp()


# MCP methods inherited from MCPClientMixin (see mcp_client.py)

    def get_tools_for(self, agent_type: str) -> list[dict]:
        """Get tool definitions for a specific agent type."""
        tool_map = {
            "leader": [self._tool_log_memory, self._tool_query_memory, self._tool_write_file, self._tool_read_file, self._tool_list_files],
            "idea": [self._tool_search_papers, self._tool_get_paper, self._tool_write_file, self._tool_read_file],
            "researcher": [
                self._tool_search_papers,
                self._tool_web_search,
                self._tool_web_fetch,
                self._tool_analyze_image,
                self._tool_write_file,
                self._tool_read_file,
                self._tool_list_files,
                self._tool_analyze_model,
            ],
            "code": [
                self._tool_run_shell,
                self._tool_run_python,
                self._tool_launch_experiment,
                self._tool_diagnose_error,
                self._tool_write_file,
                self._tool_read_file,
                self._tool_list_files,
                self._tool_analyze_model,
                self._tool_probe_model,
                self._tool_generate_diagnostic,
                self._tool_design_ablation,
                self._tool_code_review,
            ],
            "writing": [self._tool_write_file, self._tool_read_file, self._tool_list_files],
        }
        return tool_map.get(agent_type, [])

    def execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name and return the result.

        Includes safety guards for common LLM argument mistakes:
        - args as list instead of dict → wrap as dict
        - extra/unknown arguments → strip them silently
        """
        # Guard: LLM sometimes sends args as a list instead of dict
        if isinstance(args, list):
            logger.warning(f"Tool {name} received list args instead of dict: {str(args)[:200]}")
            if len(args) == 0:
                args = {}
            elif isinstance(args[0], dict):
                args = args[0]
            else:
                return json.dumps({"error": f"Invalid arguments format for {name}. Expected a JSON object."})

        if not isinstance(args, dict):
            logger.warning(f"Tool {name} received non-dict args: {type(args)}")
            return json.dumps({"error": f"Invalid arguments format for {name}. Expected a JSON object."})

        handlers = {
            "run_shell": self._exec_run_shell,
            "run_python": self._exec_run_python,
            "launch_experiment": self._exec_launch_experiment,
            "write_file": self._exec_write_file,
            "read_file": self._exec_read_file,
            "list_files": self._exec_list_files,
            "search_papers": self._exec_search_papers,
            "get_paper": self._exec_get_paper,
            "web_search": self._exec_web_search,
            "web_fetch": self._exec_web_fetch,
            "log_memory": self._exec_log_memory,
            "query_memory": self._exec_query_memory,
            "analyze_image": self._exec_analyze_image,
            "diagnose_error": self._exec_diagnose_error,
            "analyze_model": self._exec_analyze_model,
            "probe_model": self._exec_probe_model,
            "generate_diagnostic": self._exec_generate_diagnostic,
            "design_ablation": self._exec_design_ablation,            "code_review": self._exec_code_review,
        }

        handler = handlers.get(name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {name}"})

        try:
            return handler(**args)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return json.dumps({"error": str(e)})

    # --- Tool Definitions (for API schema) ---

    @property
    def _tool_run_shell(self) -> dict:
        return {
            "name": "run_shell",
            "description": "Run a shell command and return output. Use for quick checks, file ops, git commands. For long-running training, use launch_experiment instead.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120)", "default": 120},
                },
                "required": ["command"],
            },
        }

    @property
    def _tool_run_python(self) -> dict:
        return {
            "name": "run_python",
            "description": (
                "Execute a Python code snippet safely. Sandbox: no os/subprocess/file-writes/eval/exec. "
                "Use for quick shape checks, import tests, data validation, tensor math. "
                "For complex scripts, use run_shell or write_file + launch_experiment instead."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute. Print output is captured. No file I/O allowed."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)", "default": 30},
                },
                "required": ["code"],
            },
        }

    @property
    def _tool_launch_experiment(self) -> dict:
        return {
            "name": "launch_experiment",
            "description": "Launch a long-running experiment via nohup. Returns PID for monitoring. Use this for training runs, not run_shell.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Training command to run"},
                    "log_file": {"type": "string", "description": "Path for stdout/stderr log"},
                    "gpu": {"type": "string", "description": "CUDA_VISIBLE_DEVICES value"},
                },
                "required": ["command", "log_file"],
            },
        }

    @property
    def _tool_write_file(self) -> dict:
        return {
            "name": "write_file",
            "description": "Write content to a file. Cannot overwrite protected files (state.json, MEMORY_LOG.md, PROJECT_BRIEF.md).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        }

    @property
    def _tool_read_file(self) -> dict:
        return {
            "name": "read_file",
            "description": "Read a file's contents. Supports optional offset/limit for large files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace"},
                    "offset": {"type": "integer", "description": "Line number to start reading from (0-based, default: 0)", "default": 0},
                    "limit": {"type": "integer", "description": "Max number of lines to read (default: all)", "default": 0},
                },
                "required": ["path"],
            },
        }

    @property
    def _tool_list_files(self) -> dict:
        return {
            "name": "list_files",
            "description": "List files in a directory.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path relative to workspace", "default": "."},
                },
            },
        }

    @property
    def _tool_search_papers(self) -> dict:
        return {
            "name": "search_papers",
            "description": "Search for academic papers via MCP web search.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Max results (default: 10)", "default": 10},
                    "year": {"type": "string", "description": "Year filter, e.g. '2024-2026'"},
                },
                "required": ["query"],
            },
        }

    @property
    def _tool_web_search(self) -> dict:
        return {
            "name": "web_search",
            "description": "Perform a web search to find papers, project pages, or technical information. Use for retrieving current information that MCP may not find.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results (default: 5)", "default": 5},
                },
                "required": ["query"],
            },
        }

    @property
    def _tool_web_fetch(self) -> dict:
        return {
            "name": "web_fetch",
            "description": "Fetch a URL and extract structured information. Use to verify paper details from arXiv, conference pages, or project pages.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "fetch_info": {"type": "string", "description": "What information to extract from the page"},
                },
                "required": ["url", "fetch_info"],
            },
        }

    @property
    def _tool_log_memory(self) -> dict:
        return {
            "name": "log_memory",
            "description": "Log an entry to the memory system. Use 'milestone' for key results, 'decision' for routine decisions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["milestone", "decision"]},
                    "entry": {"type": "string", "description": "Content to log"},
                },
                "required": ["type", "entry"],
            },
        }

    @property
    def _tool_get_paper(self) -> dict:
        return {
            "name": "get_paper",
            "description": "Fetch details for a specific paper by arXiv ID or URL. Returns title, abstract, authors, year, citation count, and URL.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "arXiv ID (e.g. '649def34f8be52c8b66281af98ae884c09aef38b') or arXiv ID (e.g. 'arXiv:2401.12345')"},
                },
                "required": ["paper_id"],
            },
        }

    @property
    def _tool_analyze_image(self) -> dict:
        return {
            "name": "analyze_image",
            "description": (
                "Analyze an image using vision AI. Supports multiple analysis types:\n"
                "- 'data_viz': Analyze charts/dashboards/heatmaps for trends and anomalies\n"
                "- 'diagram': Understand architecture diagrams, flowcharts, UML, ER diagrams\n"
                "- 'ocr': Extract text from screenshots (code, terminal, documents)\n"
                "- 'error': Diagnose error popups, stack traces, and log screenshots\n"
                "- 'general': General-purpose image understanding\n"
                "- 'depth_map': Analyze depth/disparity map quality and detect artifacts"
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the image file"},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["data_viz", "diagram", "ocr", "error", "general", "depth_map"],
                        "default": "general",
                        "description": "Type of visual analysis to perform",
                    },
                    "prompt": {"type": "string", "description": "Additional context or specific question about the image"},
                },
                "required": ["image_path"],
            },
        }

    @property
    def _tool_diagnose_error(self) -> dict:
        return {
            "name": "diagnose_error",
            "description": (
                "Diagnose errors from screenshot or text. Can analyze error popups, "
                "stack traces, training log screenshots, and terminal output images. "
                "Returns structured diagnosis with likely root cause and fix suggestions."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to the error screenshot image"},
                    "error_text": {"type": "string", "description": "Error text/stack trace (alternative to image_path)"},
                    "context": {"type": "string", "description": "Additional context (e.g., what command was running)"},
                },
            },
        }

    @property
    def _tool_analyze_model(self) -> dict:
        return {
            "name": "analyze_model",
            "description": (
                "Deep analysis of a model's architecture: data flow, information bottlenecks, "
                "gradient paths, structural soundness, data feasibility, and IDEA ALIGNMENT. "
                "Returns: (1) parameter counts & per-branch feature ratios, (2) data flow graph — "
                "how information flows from input to output through each module, (3) information "
                "bottleneck detection — where channels compress too aggressively, (4) gradient "
                "path analysis — whether all branches receive meaningful gradients, (5) structural "
                "soundness checks — redundant/dominant/dead branches, (6) domain assumption analysis — "
                "what physical assumptions the architecture makes, (7) GPU memory & data feasibility, "
                "(8) result-to-architecture diagnosis (if training_results provided), "
                "(9) IDEA-ARCHITECTURE ALIGNMENT — compares model structure against PROJECT_BRIEF.md "
                "to verify the core research idea is faithfully implemented. Checks: per-branch channel "
                "allocation vs idea importance, missing architectural patterns implied by the idea, "
                "decoder adequacy, and provides an alignment score (0-10) with specific improvement suggestions. "
                "Use BEFORE training to catch design flaws, or AFTER training to diagnose WHY a model failed. "
                "MANDATORY after creating or significantly modifying a model architecture."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the model Python file (e.g., 'models/angular_freq_depth_net_v2.py')",
                    },
                    "dataset_manifest": {
                        "type": "string",
                        "description": "Path to DATASET_MANIFEST.json for data feasibility analysis (default: 'DATASET_MANIFEST.json')",
                    },
                    "target_size": {
                        "type": "string",
                        "description": "Target spatial size for GPU memory estimate (default: '256x256')",
                    },
                    "training_results": {
                        "type": "string",
                        "description": "Optional JSON string of training results (metrics) to enable result-to-architecture diagnosis",
                    },
                },
                "required": ["model_path"],
            },
        }

    @property
    def _tool_probe_model(self) -> dict:
        return {
            "name": "probe_model",
            "description": (
                "RUNTIME model diagnostic: instantiates the model, runs forward+backward "
                "with dummy data, and captures ACTUAL tensor statistics. This is NOT static "
                "analysis — it runs the real code to answer questions that AST cannot:\n"
                "- Do branches produce distinct features or collapse to the same values?\n"
                "- Does each branch receive meaningful gradients (or is it gradient-dead)?\n"
                "- What is the actual output value distribution (collapsed to mean? or spread?)\n"
                "- What happens to intermediate features at each processing stage?\n"
                "- How do activations change with different input patterns?\n"
                "Use AFTER analyze_model for deep diagnosis. Requires GPU/CPU with PyTorch."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the model Python file",
                    },
                    "model_class": {
                        "type": "string",
                        "description": "Name of the model class to instantiate (e.g., 'MyModelV2')",
                    },
                    "input_shape": {
                        "type": "string",
                        "description": "Input tensor shape as JSON list, e.g. '[1, 3, 64, 64]'. Auto-detected if omitted.",
                    },
                    "checkpoint_path": {
                        "type": "string",
                        "description": "Optional path to a .pth checkpoint to load trained weights (enables trained-model diagnosis)",
                    },
                },
                "required": ["model_path", "model_class"],
            },
        }

    @property
    def _tool_generate_diagnostic(self) -> dict:
        return {
            "name": "generate_diagnostic",
            "description": (
                "Generate a domain-specific or hypothesis-specific diagnostic script. "
                "Unlike probe_model (which uses random data), this generates a script "
                "that tests specific questions about the data/model:\n"
                "- 'Is a specific branch producing distinct features per domain?'\n"
                "- 'Does the model produce different outputs for different inputs?'\n"
                "Returns the generated script path and a summary of what it tests."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to answer (e.g., 'Is branch X dead for domain Y?')",
                    },
                    "model_path": {
                        "type": "string",
                        "description": "Path to the model Python file",
                    },
                    "model_class": {
                        "type": "string",
                        "description": "Name of the model class",
                    },
                    "data_path": {
                        "type": "string",
                        "description": "Path to real validation data (optional, uses random data if omitted)",
                    },
                    "checkpoint_path": {
                        "type": "string",
                        "description": "Path to trained model checkpoint (optional)",
                    },
                },
                "required": ["question", "model_path", "model_class"],
            },
        }

    @property
    def _tool_design_ablation(self) -> dict:
        return {
            "name": "design_ablation",
            "description": (
                "Generate an ablation experiment plan by systematically removing or "
                "replacing model components. Given a model architecture, it:\n"
                "1. Identifies all major architectural components (branches, modules, layers)\n"
                "2. Generates ablation variants (remove each component one at a time)\n"
                "3. Estimates the information value of each ablation\n"
                "4. Returns a prioritized list of ablation experiments\n"
                "Use when the model's performance is unclear and you need to understand "
                "which components actually contribute to performance."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Path to the model Python file",
                    },
                    "model_class": {
                        "type": "string",
                        "description": "Name of the model class",
                    },
                    "target_metrics": {
                        "type": "string",
                        "description": "JSON list of metrics to track (e.g., '[\"val_MAE\", \"val_loss\"]')",
                    },
                    "baseline_metrics": {
                        "type": "string",
                        "description": "JSON string of current baseline metrics for value estimation",
                    },
                },
                "required": ["model_path", "model_class"],
            },
        }

    # --- Tool Implementations ---

    def _resolve_workspace_path(self, path: str) -> Path:
        """Resolve a user-supplied path and keep it inside the workspace.

        Handles common LLM mistakes:
        - Absolute paths: auto-converts to relative if inside workspace
        - Workspace-prefixed paths: strips the prefix
        """
        if path is None or not str(path).strip():
            raise ValueError("Path cannot be empty")

        requested = Path(path)

        # If absolute path, try to make it relative to workspace
        if requested.is_absolute():
            try:
                rel = requested.relative_to(self.workspace)
                requested = rel
            except ValueError:
                # Path is outside workspace — try stripping common prefixes
                # that LLMs often add (e.g., /workspace/...)
                parts = requested.parts
                if len(parts) > 1:
                    # Try using just the last N components
                    for i in range(1, len(parts)):
                        candidate = Path(*parts[i:])
                        resolved_candidate = (self.workspace / candidate).resolve(strict=False)
                        try:
                            resolved_candidate.relative_to(self.workspace)
                            if resolved_candidate.exists():
                                return resolved_candidate
                        except ValueError:
                            continue
                raise ValueError(
                    f"Path must be relative to workspace. "
                    f"Got absolute path: {path}. "
                    f"Workspace is: {self.workspace}"
                )

        resolved = (self.workspace / requested).resolve(strict=False)

        try:
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise ValueError(f"Path escapes workspace: {path}") from exc

        return resolved

    def _validate_command(self, command: str) -> str:
        """Validate a shell command for safety before execution with shell=True.

        Blocks dangerous commands while allowing shell operators (&&, |, etc.)
        and builtins (cd, echo, etc.).
        """
        if not command or not command.strip():
            raise ValueError("Command cannot be empty")

        # Strip leading/trailing whitespace
        cmd = command.strip()

        # Block patterns that could escape or cause damage
        blocked_patterns = [
            (r"\brm\s+(-[rfRF]+\s+)?/\b", "remove root filesystem"),
            (r"\brm\s+(-[rfRF]+\s+)?\*?\.\.\b", "remove parent directory"),
            (r"\bsudo\b", "sudo"),
            (r"\bsu\b", "su command"),
            (r"\bmkfs\b", "mkfs"),
            (r"\bshutdown\b", "shutdown"),
            (r"\breboot\b", "reboot"),
            (r"\bpoweroff\b", "poweroff"),
            (r"\bhalt\b", "halt"),
            (r"\bdd\s+.*of=/dev/", "dd to device"),
            (r">\s*/dev/(?!null\b|zero\b|tty\b|pts/|fd/|stdin\b|stdout\b|stderr\b)", "redirect to unsafe device"),
            (r"\bchmod\s+777\b", "chmod 777"),
            # Prevent pipe-to-shell (remote code execution)
            (r"\|\s*(ba)?sh\b", "pipe to shell"),
            (r"\|\s*bash\b", "pipe to bash"),
            (r"\b(tee|wget|curl)\s+.*\|\s*(ba)?sh\b", "download and pipe to shell"),
            (r"`[^`]*`", "backtick command substitution"),
            # Block dangerous $() usages but allow safe ones (date, pwd, basename, dirname, echo)
            (r"\$\((?!(?:date|pwd|basename|dirname|echo|readlink|realpath|dirname|seq)\b)[^)]*\)", "dollar-paren command substitution"),
            # Prevent reverse shells
            (r"\bnc\s+.*-e\b", "netcat reverse shell"),
            (r"\bsocat\b", "socat"),
            (r"\bmkfifo\b", "mkfifo"),
            # Prevent arbitrary python execution bypasses
            (r"\bpython[23]?\s+-c\b.*\bos\b", "python -c os bypass"),
            (r"\bpython[23]?\s+-c\b.*\bsubprocess\b", "python -c subprocess bypass"),
            # Prevent data exfiltration via curl/wget
            (r"\bcurl\s+.*(-d|-F|--data|--form|-T|--upload-file)\s+@", "curl data upload"),
            # Prevent deleting agent's own log
            (r"\brm\s+.*autoresearcher\.log", "delete agent log"),
            (r"\bwget\s+.*(--post-file|--post-data)", "wget data upload"),
            # Block script interpreters that bypass command restrictions
            (r"\bperl\s+-e\b", "perl eval bypass"),
            (r"\bruby\s+-e\b", "ruby eval bypass"),
            (r"\bnode\s+-e\b", "node eval bypass"),
            # Block base64 decode pipe to shell (command obfuscation)
            (r"\|\s*base64\s+-d\s*\|\s*(ba)?sh", "base64 decode pipe to shell"),
            # Block env/shell variable manipulation for privilege escalation
            (r"\bexport\s+PATH\b", "PATH manipulation"),
            (r"\bexport\s+LD_PRELOAD\b", "LD_PRELOAD manipulation"),
            # Block writing to /etc, systemd, cron (persistence mechanisms)
            (r">\s*/etc/", "write to /etc"),
            (r"\bcrontab\b", "crontab manipulation"),
            (r"\bsystemctl\b", "systemctl"),
        ]

        for pattern, reason in blocked_patterns:
            if re.search(pattern, cmd):
                raise ValueError(f"Blocked: {reason}")

        return cmd

    def _exec_run_shell(self, command: str, timeout: int = 120) -> str:
        """Execute a shell command with timeout. Uses shell=True to support
        pipes, redirects, &&, cd, and other shell operators."""
        try:
            validated_cmd = self._validate_command(command)
            result = subprocess.run(
                validated_cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                cwd=str(self.workspace),
            )
            return json.dumps({
                "stdout": result.stdout[-2000:],  # Cap output
                "stderr": result.stderr[-500:],
                "returncode": result.returncode,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"Command timed out after {timeout}s"})

    def _exec_run_python(self, code: str, timeout: int = 30) -> str:
        """Execute a Python code snippet in a sandboxed subprocess.

        Security model:
        - Blocked: os.system, subprocess, os.remove, shutil.rmtree, os.environ,
          __import__, eval(), exec(), open(..., 'w')
        - Allowed: torch, numpy, math, json, sys (read-only), project imports
        - Runs as a subprocess with workspace as cwd, killed after timeout
        """
        if not code or not code.strip():
            return json.dumps({"error": "Code cannot be empty"})

        # Security check: block dangerous patterns
        # De-obfuscate common string concatenation tricks before checking
        deobfuscated = re.sub(r"[\"']\s*\+\s*[\"']", "", code)  # 'o'+'s' -> 'os'
        deobfuscated = re.sub(r"\\x[0-9a-fA-F]{2}", "", deobfuscated)  # \x00 escapes

        for pattern in self._blocked_py_patterns:
            # Check both original and deobfuscated code
            match = re.search(pattern, code) or re.search(pattern, deobfuscated)
            if match:
                return json.dumps({
                    "error": (
                        f"Blocked pattern in code: '{match.group()}'. "
                        f"run_python is for read-only computation (shapes, imports, math). "
                        f"Use run_shell or write_file for file I/O and system operations."
                    )
                })

        # Wrap code to capture stdout and handle errors
        wrapped = (
            "import sys, io, traceback;\n"
            "_stdout_capture = io.StringIO();\n"
            "sys.stdout = _stdout_capture;\n"
            "sys.stderr = _stdout_capture;\n"
            "try:\n"
        )
        for line in code.split("\n"):
            wrapped += f"    {line}\n"
        wrapped += (
            "except Exception as _e:\n"
            "    traceback.print_exc();\n"
            "finally:\n"
            "    sys.stdout = sys.__stdout__;\n"
            "    sys.stderr = sys.__stderr__;\n"
            "    print(_stdout_capture.getvalue(), end='')\n"
        )

        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapped],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
            )
            return json.dumps({
                "stdout": result.stdout[-3000:],
                "stderr": result.stderr[-500:],
                "returncode": result.returncode,
            })
        except subprocess.TimeoutExpired:
            return json.dumps({"error": f"Python code timed out after {timeout}s"})
        except FileNotFoundError:
            return json.dumps({"error": "Python interpreter not found"})

    # Commands that must NEVER be launched as experiments (infinite loops, smoke tests, etc.)
    _BLOCKED_COMMANDS = [
        "audit_smoke_loop",
        "audit_smoke_test",
        "smoke_loop",
        "while True",
        "while true",
        "while 1",
        "while(1",
        "while (1",
    ]

    def _exec_launch_experiment(self, command: str, log_file: str, gpu: str = None) -> str:
        """Launch experiment via nohup. Uses shell=True to support
        environment variables, cd, pipes, and other shell operators."""
        # Block dangerous / infinite-loop commands
        cmd_lower = command.lower()
        for blocked in self._BLOCKED_COMMANDS:
            if blocked.lower() in cmd_lower:
                return json.dumps({
                    "error": f"Command blocked: contains '{blocked}'. "
                             f"This looks like a smoke test or infinite loop, not a real training run. "
                             f"Use train_v11.py or a proper training script instead."
                })

        # Phase 2 change 2: mandatory dry-run check (P1: safety in the tool)
        # When config safety.mandatory_dry_run is true, require that a dry-run
        # of this script was recently performed (within 10 minutes).
        # Default: OFF (does not change existing behavior).
        if getattr(self, '_mandatory_dry_run', False):
            script_name = self._extract_script_name(command)
            if script_name and not self._has_recent_dry_run(script_name):
                return json.dumps({
                    "error": (
                        f"Dry-run required before launching '{script_name}'. "
                        f"Run: python {script_name} --dry_run first, then retry. "
                        f"This prevents wasting GPU on scripts with import/shape errors."
                    )
                })

        # P1: Check dead-end constraints before launching (safety in the tool)
        if self._memory:
            try:
                from .constraint_engine import StrategyConstraintEngine
                engine = StrategyConstraintEngine.__new__(StrategyConstraintEngine)
                engine._rules = []
                engine._rules_loaded = False
                engine.workspace = self.workspace
                engine.generate_rules_from_history(self._memory)
                violations = engine.check_constraints({"task": command}, self._memory)
                if engine.has_forbidden_violation(violations):
                    return json.dumps({
                        "error": "This approach has been recorded as a dead end (failed 5+ times). "
                                 "Choose a different approach. Dead-end details: " +
                                 "; ".join(violations[:3])
                    })
            except Exception:
                pass  # Don't block launch if constraint check fails

        env = os.environ.copy()
        if gpu:
            env["CUDA_VISIBLE_DEVICES"] = gpu

        validated_cmd = self._validate_command(command)
        log_path = self._resolve_workspace_path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, "w") as f:
            proc = subprocess.Popen(
                validated_cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                shell=True,
                start_new_session=True,
                cwd=str(self.workspace),
            )

        # Write a structured manifest alongside the log. This is the single
        # source of truth for "an experiment was launched" — replacing the
        # brittle regex that used to sniff run_shell command text for
        # `python train.py`. The dispatcher reads pid/status from the tool
        # return value (truth), and VERIFY/REFLECT can read this manifest
        # for the full record (script, gpu, timestamp).
        manifest = {
            "pid": proc.pid,
            "command": command,
            "log_file": str(log_path),
            "gpu": gpu,
            "workspace": str(self.workspace),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "launched",
        }
        manifest_path = log_path.parent / "experiment_manifest.json"
        try:
            with open(manifest_path, "w") as mf:
                json.dump(manifest, mf, indent=2)
        except OSError:
            logger.warning(f"Could not write experiment_manifest.json at {manifest_path}")

        return json.dumps({"pid": proc.pid, "log_file": str(log_path), "status": "launched"})

    def _exec_write_file(self, path: str, content: str) -> str:
        """Write file with protection check."""
        # Use _resolve_workspace_path FIRST to ensure path is within workspace
        try:
            resolved = self._resolve_workspace_path(path)
        except ValueError as e:
            return json.dumps({"error": str(e)})

        # Check filename-based protection against the BASENAME
        if resolved.name in self._protected_files:
            return json.dumps({"error": f"Cannot overwrite protected file: {path}"})

        # Block writes to .git/, __pycache__, and other system directories
        resolved_str = str(resolved)
        for blocked_dir in (".git", "__pycache__", ".git/hooks"):
            if f"/{blocked_dir}/" in resolved_str or resolved_str.endswith(f"/{blocked_dir}"):
                return json.dumps({"error": f"Cannot write to system directory: {blocked_dir}"})

        # Protect core source directories — agents should not overwrite
        # project code (models/, datasets/, scripts/, data/).
        try:
            rel = resolved.relative_to(self.workspace)
            parts = rel.parts
        except ValueError:
            return json.dumps({"error": f"Path escapes workspace: {path}"})

        if parts and parts[0] in self._protected_dirs:
            # Allow writing to scripts/*.py (experiment scripts)
            if parts[0] == "scripts" and resolved.suffix == ".py":
                pass  # allowed
            # Allow datasets/__init__.py and datasets/unified_lf_dataset.py (registration)
            elif parts[0] == "datasets" and len(parts) == 2 and parts[1] in ("__init__.py", "unified_lf_dataset.py"):
                pass  # allowed
            else:
                return json.dumps({
                    "error": f"Cannot write to protected directory '{parts[0]}/'. "
                             f"Write new files to the project root or logs/ directory."
                })

        # Phase 2 change 1: naming convention enforcement (P1: safety in the tool)
        # Python files must go in scripts/ or tools/, not the workspace root.
        # Training scripts (train_*.py) must be in scripts/.
        # Diagnostic scripts (debug_*/diag_*/_check_*) must be in tools/.
        if resolved.suffix == ".py":
            top_dir = parts[0] if len(parts) > 1 else ""
            name = resolved.name
            if len(parts) == 1:
                # .py file in workspace root
                return json.dumps({
                    "error": (
                        f"Python files must not be written to the workspace root. "
                        f"Use scripts/ for training scripts or tools/ for utilities. "
                        f"Got: {path}. Suggested: scripts/{name}"
                    )
                })
            if name.startswith("train_") and top_dir != "scripts":
                return json.dumps({
                    "error": (
                        f"Training scripts (train_*.py) must be in scripts/. "
                        f"Got: {path}. Suggested: scripts/{name}"
                    )
                })
            if (name.startswith(("debug_", "diag_", "_check_", "dryrun_", "dry_run_"))
                    and top_dir != "tools"):
                return json.dumps({
                    "error": (
                        f"Diagnostic scripts must be in tools/. "
                        f"Got: {path}. Suggested: tools/{name}"
                    )
                })

        file_path = resolved
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Limit file size to 10MB
        MAX_FILE_SIZE = 10 * 1024 * 1024
        if len(content) > MAX_FILE_SIZE:
            return json.dumps({"error": f"Content too large: {len(content)} bytes (max {MAX_FILE_SIZE})"})

        file_path.write_text(content)

        # ── Scan for synthetic data patterns in training scripts ──
        if parts and parts[0] == "scripts" and Path(path).suffix == ".py":
            for pattern, desc in [
                (r"np\.random\.rand\(", "np.random.rand() — random noise"),
                (r"torch\.rand\(", "torch.rand() — random tensor"),
                (r"SyntheticLF\w*", "SyntheticLF — synthetic dataset"),
                (r"RandomDataset", "RandomDataset — random data"),
            ]:
                if re.search(pattern, content):
                    return json.dumps({
                        "status": "written_with_warning",
                        "path": str(file_path),
                        "bytes": len(content),
                        "warning": (
                            f"SUSPECTED SYNTHETIC DATA: Found {desc} in training script. "
                            f"VERIFY will block this experiment. Use the project's real dataset instead."
                        ),
                    })

        return json.dumps({"status": "written", "path": str(file_path), "bytes": len(content)})

    def _exec_read_file(self, path: str, offset: int = 0, limit: int = 0) -> str:
        """Read file contents with optional line offset/limit."""
        file_path = self._resolve_workspace_path(path)
        if not file_path.exists():
            return json.dumps({"error": f"File not found: {path}"})
        try:
            content = file_path.read_text()
        except UnicodeDecodeError:
            return json.dumps({"error": f"File is binary, cannot read as text: {path}"})

        if offset > 0 or limit > 0:
            lines = content.split("\n")
            if offset > 0:
                lines = lines[offset:]
            if limit > 0:
                lines = lines[:limit]
            content = "\n".join(lines)

        return content[:10000]  # Cap at 10K chars

    def _extract_script_name(self, command: str) -> str:
        """Extract the python script name from a launch command."""
        import re
        # Match python [flags] script.py
        m = re.search(r'(?:python|python3)\s+(?:[^\s]*\s+)*?([a-zA-Z_][\w/]*\.py)', command)
        return m.group(1) if m else ""

    def _has_recent_dry_run(self, script_name: str, max_age_seconds: int = 600) -> bool:
        """Check if a dry-run of this script was performed recently.

        Looks for experiment_manifest.json entries with a dry-run marker
        within the last max_age_seconds (default 10 minutes).
        """
        import time
        # Check outputs/*/experiment_manifest.json for dry-run records
        outputs_dir = self.workspace / "outputs"
        if not outputs_dir.exists():
            return False
        script_basename = Path(script_name).name
        now = time.time()
        for manifest_path in outputs_dir.glob("*/experiment_manifest.json"):
            try:
                data = json.loads(manifest_path.read_text())
                cmd = data.get("command", "")
                ts = data.get("timestamp", "")
                # Parse timestamp and check age
                from datetime import datetime
                dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S") if ts else None
                if dt:
                    age = now - dt.timestamp()
                    if age > max_age_seconds:
                        continue
                # Check if this is a dry-run of the same script
                if script_basename in cmd and ("--dry" in cmd or "dry_run" in cmd or "dryrun" in cmd):
                    return True
            except Exception:
                continue
        return False

    def _exec_list_files(self, path: str = ".") -> str:
        """List directory contents."""
        dir_path = self._resolve_workspace_path(path)
        if not dir_path.is_dir():
            return json.dumps({"error": f"Not a directory: {path}"})
        files = sorted([f.name for f in dir_path.iterdir()])
        return json.dumps({"files": files[:100]})  # Cap at 100 entries

    def _exec_search_papers(self, query: str, limit: int = 10, year: str = None) -> str:
        """Search for academic papers via MCP."""
        mcp_result = self._mcp_web_search(
            f"{query} academic paper" + (f" {year}" if year else ""),
            max_results=limit,
        )
        if mcp_result:
            try:
                parsed = json.loads(mcp_result)
                if parsed.get("results") and len(parsed["results"]) > 0:
                    valid = [r for r in parsed["results"] if r.get("title")]
                    if valid:
                        parsed["results"] = valid[:limit]
                        parsed["source"] = "mcp"
                        return json.dumps(parsed, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, TypeError):
                pass
        return json.dumps({"error": "MCP search returned no results."})


    def _exec_get_paper(self, paper_id: str) -> str:
        """Fetch paper details via MCP web_reader."""
        url = None
        if paper_id.startswith("arXiv:"):
            url = f"https://arxiv.org/abs/{paper_id[6:]}"
        elif paper_id.startswith("http"):
            url = paper_id
        if url:
            result = self._mcp_web_fetch(url)
            if result:
                return result
        return json.dumps({"error": "Could not fetch paper. Use web_fetch with the URL."})


    def _parse_arxiv_page(self, content: str, paper_id: str) -> dict | None:
        """Parse arXiv page content to extract paper metadata."""

        title = ""
        # Try to find title in markdown headings or bold text
        title_match = re.search(r"#+\s*(.+?)(?:\n|$)", content)
        if title_match:
            title = title_match.group(1).strip()
        if not title:
            title_match = re.search(r"\*\*(.+?)\*\*", content)
            if title_match:
                title = title_match.group(1).strip()

        # Extract abstract
        abstract = ""
        abs_match = re.search(r"Abstract[:\s]*(.*?)(?:\n\n|\n#|$)", content, re.DOTALL)
        if abs_match:
            abstract = abs_match.group(1).strip()[:1000]

        # Extract authors
        authors = []
        author_match = re.search(r"Authors?[:\s]*(.*?)(?:\n|$)", content)
        if author_match:
            authors = [{"name": a.strip()} for a in author_match.group(1).split(",") if a.strip()][:10]

        # Extract year from arXiv ID
        year = None
        year_match = re.match(r"(?:arXiv:)?(\d{2})(\d{2})\.", paper_id)
        if year_match:
            year = int("20" + year_match.group(1))

        if not title and not abstract:
            return None

        return {
            "title": title or "Unknown",
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "url": f"https://arxiv.org/abs/{paper_id.replace('arXiv:', '')}",
            "paperId": paper_id,
        }

    def _exec_web_search(self, query: str, max_results: int = 5) -> str:
        """Web search via MCP."""
        mcp_result = self._mcp_web_search(query, max_results)
        if mcp_result:
            try:
                parsed = json.loads(mcp_result)
                if parsed.get("results") and len(parsed["results"]) > 0:
                    valid = [r for r in parsed["results"] if r.get("title")]
                    if valid:
                        parsed["results"] = valid[:max_results]
                        parsed["source"] = "mcp"
                        return json.dumps(parsed, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, TypeError):
                pass
        return json.dumps({"error": "MCP web search returned no results."})


    def _exec_web_fetch(self, url: str, fetch_info: str) -> str:
        """Fetch a URL with fallback chain: MCP web_reader → urllib direct."""
        # SSRF protection: block internal/private URLs
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return json.dumps({"error": f"Blocked: unsupported URL scheme '{parsed.scheme}'"})
            hostname = parsed.hostname or ""
            _blocked_host_patterns = [
                r"^(localhost)$",
                r"^127\.",
                r"^0\.0\.0\.0$",
                r"^10\.",
                r"^172\.(1[6-9]|2\d|3[01])\.",
                r"^192\.168\.",
                r"^169\.254\.",
                r"^\[::1\]$",
                r"^\[::ffff:",
            ]
            for pat in _blocked_host_patterns:
                if re.search(pat, hostname):
                    return json.dumps({"error": "Blocked: internal/private URL not allowed"})
            # Resolve hostname and check IP to catch DNS rebinding / hex IPs
            import socket
            import ipaddress
            try:
                resolved_ips = socket.getaddrinfo(hostname, parsed.port, proto=socket.IPPROTO_TCP)
                for _, _, _, _, addr in resolved_ips:
                    ip = ipaddress.ip_address(addr[0])
                    if ip.is_private or ip.is_loopback or ip.is_reserved or ip.is_link_local:
                        return json.dumps({"error": "Blocked: internal/private URL not allowed"})
            except (socket.gaierror, ValueError):
                pass  # DNS resolution failure — let urllib handle it
        except Exception:
            return json.dumps({"error": "Invalid URL"})

        # ── Level 1: Try MCP web_reader ──
        mcp_result = self._mcp_web_fetch(url, fetch_info)
        if mcp_result:
            try:
                parsed = json.loads(mcp_result)
                content = parsed.get("content_snippet", "")
                if content and len(content) > 30:
                    parsed["source"] = "mcp_web_reader"
                    return json.dumps(parsed, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, KeyError):
                pass
            logger.info("web_fetch: MCP result invalid, falling back to urllib")

        # ── Level 2: Direct urllib fetch ──
        return self._web_fetch_urllib(url, fetch_info)

    def _web_fetch_urllib(self, url: str, fetch_info: str) -> str:
        """Direct urllib fetch with HTML parsing. Always returns a result."""
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AutoResearcher/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else "Unknown"

            desc_match = re.search(
                r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE,
            )
            if not desc_match:
                desc_match = re.search(
                    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']description["\']',
                    html, re.IGNORECASE,
                )
            description = desc_match.group(1).strip() if desc_match else ""

            body_match = re.search(r"<body[^>]*>(.*)", html, re.IGNORECASE | re.DOTALL)
            body = body_match.group(1) if body_match else ""
            body = re.sub(r"<script[^>]*>.*?</script>", "", body, flags=re.DOTALL | re.IGNORECASE)
            body = re.sub(r"<style[^>]*>.*?</style>", "", body, flags=re.DOTALL | re.IGNORECASE)
            body = re.sub(r"<[^>]+>", " ", body)
            body = re.sub(r"\s+", " ", body).strip()

            result = {
                "source": "urllib_direct",
                "url": url,
                "title": title,
                "description": description[:500],
                "content_snippet": body[:1000],
                "fetched_for": fetch_info,
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"Web fetch failed: {str(e)}", "url": url})


    def _tool_query_memory(self) -> dict:
        """Schema for query_memory tool."""
        return {
            "name": "query_memory",
            "description": "Query your experiment history. Use this to check past results, "
                           "dead ends, causal relationships, and best metrics.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["best_metrics", "dead_ends", "causal_chain",
                                 "experiment_history", "lessons",
                                 "failed_launches"],
                        "description": "Type of memory query"
                    },
                    "limit": {"type": "integer", "description": "Max results (default 5)"}
                },
                "required": ["query_type"]
            }
        }

    def _exec_query_memory(self, query_type: str, limit: int = 5) -> str:
        """Query experiment memory — lets LLM actively look up history."""
        import json
        if not self._memory:
            return json.dumps({"error": "Memory not available"})
        try:
            if query_type == "best_metrics":
                stats = self._memory.get_summary_stats()
                return json.dumps(stats, default=str)
            elif query_type == "dead_ends":
                ends = self._memory.get_dead_ends_full()[:limit]
                return json.dumps(ends, default=str)
            elif query_type == "causal_chain":
                chain = self._memory.get_causal_history(limit=limit)
                return json.dumps(chain, default=str)
            elif query_type == "experiment_history":
                rows = self._memory.get_experiment_history(limit=limit) \
                    if hasattr(self._memory, 'get_experiment_history') else []
                return json.dumps(rows, default=str)
            elif query_type == "failed_launches":
                # Read from ResearchLoop's counter (stored in state.json)
                import json
                state_path = self.workspace / "state.json"
                if state_path.exists():
                    try:
                        state = json.loads(state_path.read_text())
                        return json.dumps({"consecutive_failed_launches": state.get("consecutive_failed_launches", 0)})
                    except Exception:
                        pass
                return json.dumps({"consecutive_failed_launches": 0})
            elif query_type == "lessons":
                lessons = self._memory.get_code_review_lessons(limit=limit) \
                    if hasattr(self._memory, 'get_code_review_lessons') else []
                return json.dumps(lessons, default=str)
            else:
                return json.dumps({"error": f"Unknown query_type: {query_type}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _exec_log_memory(self, type: str, entry: str) -> str:
        """Log to memory (delegated to MemoryManager if available)."""
        if self._memory:
            if type == "milestone":
                self._memory.log_milestone(entry)
            elif type == "decision":
                self._memory.log_decision(entry)
            else:
                self._memory.log_decision(f"[{type}] {entry}")
        return json.dumps({"status": "logged", "type": type, "entry": entry[:200]})

    def _exec_analyze_image(self, image_path: str, analysis_type: str = "general",
                            prompt: str = "") -> str:
        """Analyze an image using MCP vision tools with API fallback.

        Routes to the appropriate specialized tool based on analysis_type.
        Falls back to direct multimodal API call if MCP is unavailable.
        """
        # Resolve path
        try:
            abs_path = self._resolve_workspace_path(image_path)
            if not abs_path.exists():
                return json.dumps({"error": f"Image not found: {image_path}"})
            path_str = str(abs_path)
        except ValueError as e:
            return json.dumps({"error": str(e)})

        # Attempt 1: MCP zai-mcp-server specialized tools
        if "zai_vision" in self.mcp_available:
            tool_router = {
                "data_viz": ("analyze_data_visualization", lambda: self._mcp_analyze_data_viz(path_str, prompt)),
                "diagram": ("understand_technical_diagram", lambda: self._mcp_understand_diagram(path_str, prompt)),
                "ocr": ("extract_text_from_screenshot", lambda: self._mcp_extract_text(path_str)),
                "error": ("diagnose_error_screenshot", lambda: self._mcp_diagnose_error_screenshot(path_str, prompt)),
                "general": ("image_analysis", lambda: self._mcp_image_analysis(path_str, prompt)),
                "depth_map": ("image_analysis", lambda: self._mcp_image_analysis(
                    path_str,
                    (prompt or "") + (
                        "\nAnalyze this depth/disparity map output from a neural network. "
                        "Check: 1) Is it uniform/blank? 2) Value range reasonable? "
                        "3) Artifacts? 4) Spatial structure? 5) Likely failure cause?"
                        if not prompt else ""
                    ),
                )),
            }

            handler = tool_router.get(analysis_type, tool_router["general"])
            tool_name, caller = handler

            result = caller()
            if not result and analysis_type != "general":
                logger.info(f"analyze_image: {tool_name} failed, falling back to image_analysis")
                result = self._mcp_image_analysis(path_str, prompt)
                tool_name = "image_analysis"

            if result:
                return json.dumps({
                    "status": "analyzed",
                    "tool_used": tool_name,
                    "analysis_type": analysis_type,
                    "image_path": str(abs_path),
                    "result": result[:5000],
                }, ensure_ascii=False, indent=2)

        # Attempt 2: Direct multimodal API fallback (GLM-5V-Turbo / Qwen)
        logger.info("analyze_image: MCP unavailable, using direct API fallback")
        return self._analyze_image_api_fallback(abs_path, analysis_type, prompt)

    def _analyze_image_api_fallback(self, abs_path: Path, analysis_type: str,
                                     prompt: str) -> str:
        """Fallback: analyze image via direct multimodal API call.

        Uses GLM or Qwen endpoint based on available API keys.
        """
        import base64

        # Read and encode image
        try:
            img_bytes = abs_path.read_bytes()
            b64_data = base64.b64encode(img_bytes).decode("utf-8")
        except Exception as e:
            return json.dumps({"error": f"Cannot read image: {e}"})

        ext = abs_path.suffix.lower()
        mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webp": "image/webp", ".gif": "image/gif"}
        mime_type = mime_map.get(ext, "image/png")

        image_content = [{
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_data}", "detail": "high"},
        }]

        # Build analysis prompt
        type_prompts = {
            "data_viz": "Analyze this chart/dashboard/visualization. Identify trends, anomalies, and key insights.",
            "diagram": "Describe and interpret this technical diagram (architecture/flow/UML/ER). Explain the structure.",
            "ocr": "Extract all text from this screenshot. Preserve formatting and structure.",
            "error": "Diagnose the error shown in this screenshot. Identify the root cause and suggest fixes.",
            "depth_map": "Analyze this depth map output. Check: uniform/blank? Value range? Artifacts? Likely failure cause?",
            "general": "Describe and analyze this image in detail.",
        }
        analysis_prompt = type_prompts.get(analysis_type, type_prompts["general"])
        if prompt:
            analysis_prompt += f"\n\nAdditional context: {prompt}"

        # Try providers in order: GLM → Ali Token Plan → Ali DashScope
        providers = []
        glm_key = os.environ.get("GLM_CODING_PLAN_API_KEY", "")
        if glm_key:
            providers.append({
                "api_key": glm_key,
                "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
                "models": ["glm-5v-turbo", "glm-4.6v"],
                "label": "GLM",
            })
        ali_tp_key = os.environ.get("ALI_TOKEN_PLAN_API_KEY", "")
        if ali_tp_key:
            providers.append({
                "api_key": ali_tp_key,
                "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
                "models": ["qwen3.6-plus"],  # qwen3.5-plus not on token plan
                "label": "Ali(TokenPlan)",
            })
        ali_key = os.environ.get("ALI_API_KEY", "")
        if ali_key:
            providers.append({
                "api_key": ali_key,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "models": ["qwen3.6-plus", "qwen3.5-plus"],
                "label": "Qwen",
            })

        for provider in providers:
            for model in provider["models"]:
                try:
                    import openai
                    client = openai.OpenAI(api_key=provider["api_key"], base_url=provider["base_url"])
                    messages = [{"role": "user", "content": [{"type": "text", "text": analysis_prompt}, *image_content]}]
                    response = client.chat.completions.create(model=model, messages=messages, max_tokens=4096, timeout=60)
                    content = response.choices[0].message.content if response.choices else None
                    if content:
                        return json.dumps({
                            "status": "analyzed",
                            "tool_used": f"{provider['label']}/{model}",
                            "analysis_type": analysis_type,
                            "image_path": str(abs_path),
                            "result": content[:5000],
                        }, ensure_ascii=False, indent=2)
                except Exception as e:
                    logger.warning(f"analyze_image fallback {provider['label']}/{model} failed: {e}")

        return json.dumps({
            "error": f"All analysis methods failed (MCP + API fallback) for {abs_path}",
            "analysis_type": analysis_type,
        })

    def _exec_diagnose_error(self, image_path: str = None, error_text: str = None,
                             context: str = "") -> str:
        """Diagnose errors from screenshot or text.

        If image_path provided: tries MCP diagnose_error_screenshot, then API fallback.
        If error_text provided: returns the text for LLM to analyze.
        """
        if image_path:
            try:
                abs_path = self._resolve_workspace_path(image_path)
                if not abs_path.exists():
                    return json.dumps({"error": f"Error screenshot not found: {image_path}"})

                # Try MCP first
                if "zai_vision" in self.mcp_available:
                    result = self._mcp_diagnose_error_screenshot(str(abs_path), context)
                    if result:
                        return json.dumps({
                            "status": "diagnosed",
                            "method": "mcp_diagnose_error_screenshot",
                            "image_path": str(abs_path),
                            "result": result[:5000],
                        }, ensure_ascii=False, indent=2)

                # Fallback: use analyze_image API path
                logger.info("diagnose_error: MCP unavailable, using API fallback")
                return self._analyze_image_api_fallback(abs_path, "error", context or "Diagnose this error screenshot")

            except ValueError as e:
                return json.dumps({"error": str(e)})

        if error_text:
            return json.dumps({
                "status": "received",
                "method": "text_analysis",
                "error_text": error_text[:3000],
                "context": context,
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "error": "Provide either image_path or error_text for diagnosis",
            "usage": "diagnose_error(image_path='path/to/error.png') or diagnose_error(error_text='...')",
        })

    def _exec_analyze_model(self, model_path: str, dataset_manifest: str = "DATASET_MANIFEST.json",
                           target_size: str = "256x256", training_results: str = "") -> str:
        """Deep model architecture analysis: data flow, bottlenecks, gradient paths, structural soundness.

        This tool performs multi-layer analysis of a model's Python source:
        Layer 1: Surface analysis — parameter counts, channel ratios (legacy)
        Layer 2: Data flow graph — how information flows input → output through modules
        Layer 3: Information bottleneck detection — where channels compress too aggressively
        Layer 4: Gradient path analysis — whether all branches receive meaningful gradients
        Layer 5: Structural soundness — redundant/dominant/dead branches
        Layer 6: Domain assumption analysis — physical assumptions the architecture encodes
        Layer 7: Data feasibility + GPU memory estimate
        Layer 8: Result-to-architecture diagnosis (if training_results provided)
        """
        try:
            abs_path = self._resolve_workspace_path(model_path)
            if not abs_path.exists():
                return json.dumps({"error": f"Model file not found: {model_path}"})

            content = abs_path.read_text()
            import ast
            tree = ast.parse(content)

            # Layer 1: Legacy surface analysis
            analysis = self._analyze_model_ast(tree, content)

            # Layer 2: Data flow graph
            data_flow = self._analyze_data_flow(tree, content)
            analysis["data_flow"] = data_flow

            # Layer 3: Information bottleneck detection
            bottlenecks = self._detect_information_bottlenecks(tree, data_flow)
            analysis["information_bottlenecks"] = bottlenecks

            # Layer 4: Gradient path analysis
            gradient_paths = self._analyze_gradient_paths(tree, data_flow)
            analysis["gradient_paths"] = gradient_paths

            # Layer 5: Structural soundness
            structural = self._analyze_structural_soundness(tree, data_flow, bottlenecks)
            analysis["structural_soundness"] = structural

            # Layer 6: Domain assumption analysis
            domain_assumptions = self._analyze_domain_assumptions(content, analysis)
            analysis["domain_assumptions"] = domain_assumptions

            # Layer 7: Data feasibility + GPU memory
            manifest_path = self.workspace / dataset_manifest
            if manifest_path.exists():
                analysis["data_feasibility"] = self._analyze_data_feasibility(manifest_path, analysis)

            h, w = self._parse_target_size(target_size)
            analysis["gpu_estimate"] = self._estimate_gpu_memory(analysis, h, w)

            # Layer 8: Result-to-architecture diagnosis
            if training_results:
                try:
                    results_data = json.loads(training_results) if isinstance(training_results, str) else {}
                    analysis["result_diagnosis"] = self._diagnose_results_vs_architecture(
                        results_data, analysis, content
                    )
                except (json.JSONDecodeError, TypeError):
                    analysis["result_diagnosis"] = {"error": "Could not parse training_results JSON"}

            # Layer 9: Idea-Architecture alignment analysis
            idea_alignment = self._analyze_idea_architecture_alignment(analysis, content)
            if idea_alignment:
                analysis["idea_architecture_alignment"] = idea_alignment

            return json.dumps(analysis, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Model analysis failed: {e}", exc_info=True)
            return json.dumps({"error": str(e)})


    # Model analysis methods inherited from ModelAnalyzerMixin (see model_analyzer.py)
    # Includes: _analyze_model_ast, _extract_branch_info, _extract_channels_from_*,
    # _estimate_params_from_ast, _parse_target_size, _estimate_gpu_memory,
    # _analyze_data_feasibility, _analyze_data_flow, _get_call_name,
    # _trace_forward_flow, _extract_cat_sources, _extract_var_name,
    # _detect_information_bottlenecks, _get_constant_value, _analyze_gradient_paths,
    # _analyze_structural_soundness, _analyze_domain_assumptions,
    # _diagnose_results_vs_architecture, _analyze_idea_architecture_alignment,
    # _analyze_decoder_adequacy, _exec_probe_model, _build_probe_script,
    # _exec_generate_diagnostic, _build_diagnostic_script, _exec_design_ablation

    # ─────────────────────────────────────────────────
    # Code Review Tool (Knowledge-Base Enhanced)
    # ─────────────────────────────────────────────────

    @property
    def _tool_code_review(self) -> dict:
        return {
            "name": "code_review",
            "description": (
                "Knowledge-base enhanced code review. Checks code against past mistakes "
                "stored in the code_review_lessons table. Returns: (1) relevant past lessons "
                "that match the code, (2) pattern-based warnings, (3) suggestions. "
                "Use BEFORE and AFTER modifying model/training code to catch known anti-patterns."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to review (relative to workspace)",
                    },
                    "focus": {
                        "type": "string",
                        "description": "Review focus: 'all' (default), 'architecture', 'training', 'data'",
                        "default": "all",
                    },
                },
                "required": ["file_path"],
            },
        }

    def _exec_code_review(self, file_path: str, focus: str = "all") -> str:
        """Execute knowledge-base enhanced code review.

        Reads the file, searches for relevant past lessons, runs pattern checks,
        and returns actionable findings.
        """
        try:
            resolved = self._resolve_workspace_path(file_path)
            if not resolved.exists():
                return json.dumps({"error": f"File not found: {file_path}"})
            content = resolved.read_text()
        except ValueError as e:
            return json.dumps({"error": str(e)})

        findings = []

        # ── 1. Search knowledge base for relevant lessons ──
        relevant_lessons = []
        if self._memory:
            relevant_lessons = self._memory.search_relevant_lessons(content, limit=10)

        if relevant_lessons:
            for lesson in relevant_lessons:
                findings.append({
                    "type": "past_mistake",
                    "severity": lesson.get("severity", "MEDIUM"),
                    "pattern": lesson.get("pattern", ""),
                    "description": lesson.get("description", ""),
                    "fix": lesson.get("fix_suggestion", ""),
                    "hit_count": lesson.get("hit_count", 1),
                })

        # ── 2. Pattern-based checks ──
        content_lower = content.lower()

        # Architecture patterns
        if focus in ("all", "architecture"):
            # Softmax without temperature
            if "softmax" in content_lower and "temperature" not in content_lower:
                findings.append({
                    "type": "pattern",
                    "severity": "MEDIUM",
                    "pattern": "softmax_without_temperature",
                    "description": "Softmax used without learnable temperature. May collapse to uniform.",
                    "fix": "Add nn.Parameter(torch.ones(1)*T) and divide logits by temperature.",
                })
            # Conv2d input channel mismatch detection
            conv_inputs = re.findall(r"Conv2d\((\d+),", content)
            if len(conv_inputs) >= 2:
                inputs_int = [int(c) for c in conv_inputs if int(c) > 1]
                if inputs_int and max(inputs_int) / min(inputs_int) > 8:
                    findings.append({
                        "type": "pattern",
                        "severity": "MEDIUM",
                        "pattern": "channel_asymmetry",
                        "description": f"Channel asymmetry: {min(inputs_int)}-{max(inputs_int)} range. Low-channel branch may lack capacity.",
                        "fix": "Balance channel allocation or use FiLM conditioning instead of raw concat.",
                    })

        # Training patterns
        if focus in ("all", "training"):
            if "aux_weight" in content_lower:
                aux_match = re.search(r"aux_weight\s*[=:]\s*([0-9.]+)", content, re.IGNORECASE)
                if aux_match and float(aux_match.group(1)) < 0.05:
                    findings.append({
                        "type": "pattern",
                        "severity": "HIGH",
                        "pattern": "low_aux_weight",
                        "description": f"aux_weight={aux_match.group(1)} is too low (< 0.05). Routing/gate modules won't learn.",
                        "fix": "Increase aux_weight to >= 0.1, ideally 0.2-0.3.",
                    })

        result = {
            "file": file_path,
            "total_findings": len(findings),
            "high_severity": sum(1 for f in findings if f["severity"] == "HIGH"),
            "findings": findings[:15],
            "kb_lessons_available": len(relevant_lessons),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
