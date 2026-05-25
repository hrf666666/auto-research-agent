"""
AutoResearcher Agent Dispatcher

Leader-Worker architecture for efficient token usage:
- Leader: Central decision-maker, persistent conversation within a cycle
- Workers: Specialized agents (idea/code/writing), spawned on demand

Only ONE worker runs at a time. Others idle at zero token cost.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.agents")


# Agent definitions directory
AGENTS_DIR = Path(__file__).parent.parent / "agents"


class ToolCallRecord:
    """Immutable record of a single tool call during an LLM session.

    This is the ANTI-DECEPTION primitive. When the LLM calls a tool,
    we record the tool name, arguments, and the ACTUAL system-returned
    result — not the LLM's summary of what happened.
    """

    __slots__ = ("name", "arguments", "result", "timestamp")

    def __init__(self, name: str, arguments: dict, result: str):
        self.name = name
        self.arguments = arguments
        self.result = result
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "tool": self.name,
            "arguments": self.arguments,
            "result_preview": self.result[:500],
            "timestamp": self.timestamp,
        }


class ToolTrace:
    """Complete record of all tool calls made during one LLM session.

    This is what makes the system resistant to LLM deception:
    - If the LLM says "I launched PID 12345" but launch_experiment
      was never called, the trace will be empty → the claim is false.
    - If launch_experiment was called but returned an error, the trace
      will show the error → the claim is false even if LLM says success.
    - Key facts (PID, log_file, exit codes) are extracted from tool
      results, not from LLM text.
    """

    def __init__(self):
        self.calls: list[ToolCallRecord] = []

    def record(self, name: str, arguments: dict, result: str):
        self.calls.append(ToolCallRecord(name, arguments, result))

    @property
    def tool_names(self) -> list[str]:
        return [c.name for c in self.calls]

    def get_tool_results(self, tool_name: str) -> list[str]:
        """Get all results for a specific tool name."""
        return [c.result for c in self.calls if c.name == tool_name]

    def get_first_result(self, tool_name: str) -> Optional[str]:
        """Get the first result for a specific tool name, or None."""
        for c in self.calls:
            if c.name == tool_name:
                return c.result
        return None

    def was_tool_called(self, tool_name: str) -> bool:
        return any(c.name == tool_name for c in self.calls)

    def extract_launch_facts(self) -> dict:
        """Extract verified facts from launch_experiment tool calls.

        Returns a dict with keys extracted from ACTUAL tool return values,
        not from LLM narrative text. This is the single source of truth
        for whether an experiment was really launched.
        """
        facts = {}
        for call in self.calls:
            if call.name == "launch_experiment":
                try:
                    result_data = json.loads(call.result)
                    if "pid" in result_data:
                        facts["pid"] = result_data["pid"]
                    if "log_file" in result_data:
                        facts["log_file"] = result_data["log_file"]
                    if "status" in result_data:
                        facts["launch_status"] = result_data["status"]
                    # If launch_experiment returned an error, record it
                    if "error" in result_data:
                        facts["launch_error"] = result_data["error"]
                except (json.JSONDecodeError, TypeError):
                    pass
        return facts

    def extract_shell_facts(self) -> list[dict]:
        """Extract facts from run_shell tool calls (exit codes, output)."""
        facts = []
        for call in self.calls:
            if call.name == "run_shell":
                try:
                    result_data = json.loads(call.result)
                    facts.append({
                        "command": call.arguments.get("command", ""),
                        "returncode": result_data.get("returncode"),
                        "stdout_preview": result_data.get("stdout", "")[:200],
                        "stderr_preview": result_data.get("stderr", "")[:200],
                        "had_error": result_data.get("returncode", 0) != 0,
                    })
                except (json.JSONDecodeError, TypeError):
                    pass
        return facts

    def to_dict(self) -> dict:
        return {
            "total_calls": len(self.calls),
            "tool_names": self.tool_names,
            "calls": [c.to_dict() for c in self.calls],
            "launch_facts": self.extract_launch_facts(),
        }


# Token Plan provider configurations
# Each token_plan is a cost-optimized subscription for code agents,
# using OpenAI-compatible protocol with different base URLs and API keys.
#
# Tiered model strategy:
#   - strong_model: For upstream complex tasks (Leader THINK/REFLECT, idea, researcher)
#   - fast_model:  For downstream simple tasks (code, writing, Leader within-cycle)
#
# Failover: If the primary provider fails (errors, timeouts), auto-switches
# to the next available provider in TOKEN_PLAN_FAILOVER_ORDER.
TOKEN_PLAN_PROVIDERS = {
    "glm_token_plan": {
        "base_url": "https://open.bigmodel.cn/api/coding/paas/v4",
        "env_key": "GLM_CODING_PLAN_API_KEY",
        "strong_model": "glm-5.1",          # Best GLM for complex reasoning
        "fast_model": "glm-5",              # Fast GLM for routine tasks
        "models": [
            "glm-4.5",              # GLM 4.5
            "glm-4.5-air",          # GLM 4.5 Air (lightweight)
            "glm-4.6",              # GLM 4.6
            "glm-4.7",              # GLM 4.7
            "glm-5",                # GLM 5
            "glm-5-turbo",          # GLM 5 Turbo (fast)
            "glm-5.1",              # GLM 5.1 (strongest)
        ],
    },
    "ali_token_plan": {
        "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "env_key": "ALI_TOKEN_PLAN_API_KEY",
        "strong_model": "qwen3.6-plus",    # Best reasoning model on Ali token plan
        "fast_model": "qwen3.6-plus",      # Fallback: same model for all tasks (no lighter tier available)
        "models": [
            "qwen3.6-plus",         # Qwen: reasoning, vision, text generation
            "qwen-image-2.0",       # Qwen: image generation
            "qwen-image-2.0-pro",   # Qwen: image generation (pro)
            "wan2.7-image",         # Wanxiang: image generation
            "wan2.7-image-pro",     # Wanxiang: image generation (pro)
            "deepseek-v3.2",        # DeepSeek: reasoning, text generation
            "glm-5",                # Zhipu AI: text generation
            "MiniMax-M2.5",         # MiniMax: reasoning, text generation
        ],
    },
}

# Failover order: GLM first, ALI as backup.
# Routing strategy: all tasks use the PRIMARY provider's models (GLM 5.1 strong / GLM 5 fast).
# Only if the primary provider fails (error/timeout) do we switch ENTIRELY to the backup
# provider (ALI: qwen3.6-plus for all tasks). We never mix models across providers.
TOKEN_PLAN_FAILOVER_ORDER = ["glm_token_plan", "ali_token_plan"]

# Tasks that require the strong model (complex reasoning / planning).
# Everything else uses the fast model.
STRONG_MODEL_TASKS = {"think", "reflect", "idea", "researcher", "code"}


class AgentDispatcher:
    """Dispatches tasks to specialized agents.

    The Leader agent decides what to do, then dispatches to workers:
    - idea_agent: Literature search, hypothesis formation
    - code_agent: Experiment implementation and execution
    - writing_agent: Report generation and paper writing

    Each worker has a minimal tool set (3-5 tools) to reduce token overhead.

    ANTI-DECEPTION DESIGN:
    All _call_llm variants return a ToolTrace alongside the text response.
    The tool trace records every tool call made during the LLM session,
    including the actual return values. This ensures that key facts
    (PIDs, file paths, exit codes) come from system-level execution,
    NOT from the LLM's narrative text.
    """

    WORKER_CONFIGS = {
        "idea": {
            "prompt_file": "idea_agent.md",
            "max_turns": 12,
            "tools": ["search_papers", "get_paper", "write_file", "read_file"],
        },
        "code": {
            "prompt_file": "code_agent.md",
            "max_turns": 40,
            "tools": ["run_shell", "run_python", "launch_experiment", "diagnose_error",
                      "write_file", "read_file", "list_files", "analyze_model", "probe_model",
                      "generate_diagnostic", "design_ablation"],
        },
        "writing": {
            "prompt_file": "writing_agent.md",
            "max_turns": 30,
            "tools": ["write_file", "read_file", "list_files"],
        },
        "researcher": {
            "prompt_file": "researcher_agent.md",
            "max_turns": 30,
            "tools": ["search_papers", "web_search", "web_fetch", "analyze_image",
                      "write_file", "read_file", "list_files", "analyze_model"],
        },
    }

    # Provider health tracking for auto-failover
    _provider_health: dict[str, dict] = {}  # class-level: shared across instances

    def __init__(self, model: str = "claude-sonnet-4-6", provider: str = "anthropic", max_steps: int = 3, tools=None):
        self.model = model
        self.provider = provider  # "anthropic", "openai", "qwen", "ali_token_plan", "glm_token_plan"
        self.max_steps = max_steps
        self._leader_history = []
        self.tools = tools  # ToolRegistry instance for executing tools

        # Resolve token_plan provider config
        self._token_plan_config = TOKEN_PLAN_PROVIDERS.get(self.provider)
        if self._token_plan_config:
            logger.info(
                f"Token Plan detected: {self.provider} "
                f"(strong={self._token_plan_config['strong_model']}, "
                f"fast={self._token_plan_config['fast_model']})"
            )

        # Initialize provider health tracking
        if provider in TOKEN_PLAN_PROVIDERS and provider not in AgentDispatcher._provider_health:
            AgentDispatcher._provider_health[provider] = {
                "consecutive_failures": 0,
                "last_failure_time": 0,
                "total_calls": 0,
                "total_failures": 0,
            }

        logger.info(f"AgentDispatcher initialized: provider='{self.provider}', model='{self.model}'")

    def _execute_tool_with_trace(self, func_name: str, func_args: dict, trace: ToolTrace = None) -> str:
        """Execute a tool and record the result in the trace.

        This is the ONLY way tool execution results should be captured.
        By centralizing here, every _call_* method automatically gets
        anti-deception tracing without code duplication.
        """
        tool_result = self.tools.execute_tool(func_name, func_args)
        if trace is not None:
            trace.record(func_name, func_args, str(tool_result))
        return tool_result

    def dispatch_leader(self, task: str, context: dict) -> dict:
        """Send a task to the Leader agent.

        The Leader maintains conversation history within a cycle for
        coherent multi-step reasoning. History is cleared between cycles.

        Args:
            task: "think" or "reflect"
            context: Current state (brief, memory, results, etc.)

        Returns:
            Leader's decision as a dict
        """
        system_prompt = self._load_prompt("leader.md")

        messages = list(self._leader_history)
        messages.append({
            "role": "user",
            "content": self._format_leader_input(task, context),
        })

        # ── Give REFLECT phase tools for deep investigation ──
        # THINK phase: no tools (pure reasoning, fast)
        # REFLECT phase: read_file + list_files for cross-validation
        reflect_tools = None
        if task == "reflect" and self.tools:
            # Use public get_tools_for() to avoid fragile private method references
            researcher_tools = self.tools.get_tools_for("researcher")
            reflect_tools = [
                t for t in researcher_tools
                if t.get("name") in ("read_file", "list_files")
            ]

        # Leader tasks (think/reflect) always use strong model
        # REFLECT with tools needs more turns for cross-validation reading
        effective_max_turns = 20 if (task == "reflect" and reflect_tools) else 10
        response_text, _trace = self._call_llm(
            system=system_prompt,
            messages=messages,
            tools=reflect_tools,
            max_turns=effective_max_turns,
            task_tier=task,  # "think" or "reflect" → strong model
        )

        # Persist conversation for within-cycle coherence
        self._leader_history = messages + [{"role": "assistant", "content": response_text}]

        return self._parse_leader_response(response_text)

    def dispatch_worker(self, agent_type: str, task: str, tools: list, max_turns_override: int = None) -> dict:
        """Dispatch a task to a worker agent.

        Workers are stateless — each dispatch is independent.
        This keeps token costs predictable.

        ANTI-DECEPTION: Returns both the LLM text response AND the
        tool execution trace. The trace records what tools were actually
        called and what they returned — not what the LLM claims happened.

        Args:
            agent_type: "idea", "code", "researcher", or "writing"
            task: Task description from the Leader
            tools: Tool definitions to provide
            max_turns_override: Override max_turns for this dispatch

        Returns:
            Worker's result as a dict, including 'tool_trace' key
        """
        if agent_type not in self.WORKER_CONFIGS:
            raise ValueError(f"Unknown agent type: {agent_type}")

        config = self.WORKER_CONFIGS[agent_type]
        system_prompt = self._load_prompt(config["prompt_file"])
        effective_max_turns = max_turns_override or config["max_turns"]

        logger.info(f"Dispatching {agent_type} agent: {task[:100]}...")

        response_text, trace = self._call_llm(
            system=system_prompt,
            messages=[{"role": "user", "content": task}],
            tools=tools,
            max_turns=effective_max_turns,
            task_tier=agent_type,  # "idea"/"researcher" → strong, "code"/"writing" → fast
        )

        result = self._parse_worker_response(response_text, agent_type, trace)
        logger.info(f"Worker {agent_type} completed: {str(result)[:200]}")
        return result

    def reset_leader_history(self):
        """Clear leader conversation history between cycles."""
        self._leader_history = []

    def _call_llm(self, system: str, messages: list, tools: list = None, max_turns: int = 10, task_tier: str = None) -> tuple[str, ToolTrace]:
        """Call the LLM API with tool execution support.

        Routing strategy (per user requirement):
        - GLM is the PRIMARY provider: glm-5.1 for strong tasks, glm-5 for fast tasks
        - If GLM fails (error/timeout/rate-limit), ENTIRELY switch to ALI fallback
        - ALI fallback: qwen3.6-plus for all tasks (no cross-provider model mixing)
        - Provider health tracking with cooldown prevents flapping

        Args:
            system: System prompt
            messages: Conversation messages
            tools: Tool definitions
            max_turns: Max tool-call turns
            task_tier: Task type for model selection ("think", "reflect", "idea",
                       "researcher" → strong; "code", "writing" → fast).

        Returns:
            (response_text, tool_trace)
        """
        trace = ToolTrace()

        # ── Build the list of providers to try (primary + failover) ──
        providers_to_try = self._build_provider_queue()

        last_error = None
        missing_keys = []
        for provider_key, provider_config in providers_to_try:
            api_key = os.getenv(provider_config["env_key"])
            if not api_key:
                missing_keys.append(provider_config["env_key"])
                logger.debug(f"Skipping {provider_key}: API key not set ({provider_config['env_key']})")
                # Don't overwrite last_error — a previous provider may have had a real API failure
                # that's more useful than "key not set" for a fallback provider.
                if last_error is None:
                    last_error = RuntimeError(
                        f"API key not set: {provider_config['env_key']}. "
                        f"Run: export {provider_config['env_key']}=\"your-key\""
                    )
                continue

            # Resolve model FOR THIS specific provider (each provider has its own models)
            model = self._resolve_model_for_provider(provider_key, provider_config, task_tier)

            try:
                text = self._call_openai_compatible(
                    system=system, messages=messages, tools=tools,
                    max_turns=max_turns, trace=trace,
                    base_url=provider_config["base_url"],
                    api_key=api_key,
                    provider_label=f"token_plan[{provider_key}]",
                    model=model,
                    task_tier=task_tier,
                )

                # Check if the response indicates an API error (not a tool result)
                # Use JSON parsing to avoid false positives from normal text containing {"error"
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict) and parsed.get("error") and "API" in str(parsed.get("error", "")):
                        raise RuntimeError(f"API returned error: {text[:200]}")
                except (json.JSONDecodeError, TypeError):
                    pass  # Not JSON — likely a normal text response

                # Success — reset failure counter
                self._record_provider_success(provider_key)
                return text, trace

            except Exception as e:
                last_error = e
                self._record_provider_failure(provider_key, str(e))
                logger.warning(
                    f"Provider {provider_key} (model={model}) failed: {e}. "
                    f"Trying next provider..."
                )
                continue

        # ── All token_plan providers failed — try legacy providers ──
        if missing_keys and last_error and "not set" in str(last_error):
            logger.error(
                f"All token_plan providers skipped — missing API keys: {missing_keys}. "
                f"Set at least one: export {missing_keys[0]}=\"your-key\""
            )
        else:
            logger.error(f"All token_plan providers failed. Last error: {last_error}")

        if self.provider == "anthropic":
            text = self._call_anthropic(system, messages, tools, max_turns, trace, task_tier=task_tier)
        elif self.provider == "openai":
            text = self._call_openai_compatible(
                system=system, messages=messages, tools=tools,
                max_turns=max_turns, trace=trace,
                base_url=None, api_key=None, provider_label="openai",
                model=self.model,
                task_tier=task_tier,
            )
        else:
            raise RuntimeError(
                f"All providers failed for provider='{self.provider}'. "
                f"Last error: {last_error}. "
                f"Set a valid API key or configure a failover provider."
            )

        return text, trace

    def _resolve_model_for_provider(self, provider_key: str, provider_config: dict, task_tier: str = None) -> str:
        """Resolve which model to use for a SPECIFIC provider.

        Each provider has its own strong/fast models. This ensures we never
        send an ALI model name to the GLM API (or vice versa).

        Routing rules:
        - GLM primary: strong=glm-5.1, fast=glm-5
        - ALI fallback: strong=qwen3.6-plus, fast=qwen3.6-plus
        - If user set a specific model (not auto/default), use it only if
          it's available on that provider.
        """
        # Check if the user explicitly chose a model (not auto/default)
        if self.model not in ("default", "auto"):
            # User chose a specific model — use it if available on this provider
            if self.model in provider_config.get("models", []):
                return self.model
            # Model not available on this provider, fall through to tier logic

        # Tiered selection per provider
        if task_tier in STRONG_MODEL_TASKS:
            return provider_config["strong_model"]
        else:
            return provider_config["fast_model"]

    def _build_provider_queue(self) -> list[tuple[str, dict]]:
        """Build ordered list of (provider_key, config) to try.

        Primary provider first, then failover candidates.
        Skips providers that are in cooldown (too many recent failures).
        """
        if not self._token_plan_config:
            return []

        queue = []
        # Always try primary provider first
        primary = self.provider
        if primary in TOKEN_PLAN_PROVIDERS:
            queue.append((primary, TOKEN_PLAN_PROVIDERS[primary]))

        # Then failover providers
        for key in TOKEN_PLAN_FAILOVER_ORDER:
            if key != primary and key in TOKEN_PLAN_PROVIDERS:
                health = AgentDispatcher._provider_health.get(key, {})
                # Skip if in cooldown (3+ consecutive failures within last 5 min)
                if health.get("consecutive_failures", 0) >= 3:
                    last_fail = health.get("last_failure_time", 0)
                    if time.time() - last_fail < 300:  # 5 min cooldown
                        logger.debug(f"Skipping {key}: in cooldown (3+ recent failures)")
                        continue
                    # Cooldown expired, give it another chance
                queue.append((key, TOKEN_PLAN_PROVIDERS[key]))

        return queue

    def _record_provider_success(self, provider_key: str):
        """Record a successful API call, resetting failure counter."""
        health = AgentDispatcher._provider_health.get(provider_key)
        if health:
            health["consecutive_failures"] = 0
            health["total_calls"] += 1

    def _record_provider_failure(self, provider_key: str, error: str):
        """Record a failed API call, incrementing failure counter."""
        if provider_key not in AgentDispatcher._provider_health:
            AgentDispatcher._provider_health[provider_key] = {
                "consecutive_failures": 0, "last_failure_time": 0,
                "total_calls": 0, "total_failures": 0,
            }
        health = AgentDispatcher._provider_health[provider_key]
        health["consecutive_failures"] += 1
        health["last_failure_time"] = time.time()
        health["total_calls"] += 1
        health["total_failures"] += 1

    # ─────────────────────────────────────────────────
    # Shared OpenAI-compatible provider (used by token_plan, qwen, openai)
    # ─────────────────────────────────────────────────

    def _call_openai_compatible(
        self,
        system: str,
        messages: list,
        tools: list = None,
        max_turns: int = 10,
        trace: ToolTrace = None,
        base_url: str = None,
        api_key: str = None,
        provider_label: str = "openai_compatible",
        model: str = None,
        task_tier: str = None,
    ) -> str:
        """Call an OpenAI-compatible API with tool execution support.

        This is the unified implementation for all OpenAI-protocol providers
        (token_plan, qwen, openai). Eliminates the previous 3x code duplication.

        Args:
            model: Override model name. If None, uses self.model with mapping.
        """
        # Resolve model: explicit param > self.model
        effective_model = model or self.model

        # Dynamic max_tokens: code agent needs more output space than leader
        # Leader tasks produce short JSON (~2K), code agent produces long tool args
        # This prevents write_file content truncation for code agent tasks.
        _MAX_TOKENS_MAP = {
            "code": 16384,      # Code agent: complex reasoning + long file writes (upgraded from 8192)
            "writing": 16384,   # Writing agent: long report generation
            "researcher": 16384, # Researcher: paper analysis, web fetch results
            "idea": 16384,      # Idea agent: moderate output
            "think": 16384,     # Leader think: structured JSON decision
            "reflect": 16384,   # Leader reflect: deep cross-validation + root cause analysis
        }
        effective_max_tokens = _MAX_TOKENS_MAP.get(task_tier, 4096)

        logger.info(
            f"Calling {provider_label} API: model={effective_model}, "
            f"messages={len(messages)}, tools={bool(tools)}"
        )
        try:
            import openai

            if not api_key:
                logger.error(f"API key not configured for {provider_label}")
                return json.dumps({"error": f"API key not configured for {provider_label}"})

            kwargs = {
                "timeout": 120.0,  # 2 min total (was 5 min — causes long hangs)
                "max_retries": 1,  # 1 retry (vs default 2)
            }
            if base_url:
                kwargs["base_url"] = base_url
            client = openai.OpenAI(api_key=api_key, **kwargs)

            # Build messages with system prompt
            # GLM coding plan requires non-empty system content (returns 400 otherwise)
            system_content = system or "You are a helpful AI research assistant."
            api_messages = [{"role": "system", "content": system_content}]
            for msg in messages:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            # GLM coding plan requires tool_calls to produce text responses.
            # If no tools are provided, inject a dummy "respond" tool so the
            # API returns structured output instead of empty content.
            effective_tools = tools
            dummy_tool = False
            if not effective_tools and "coding" in (base_url or ""):
                effective_tools = [{
                    "name": "respond",
                    "description": "Return your analysis and decision as structured JSON.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "response": {
                                "type": "string",
                                "description": "Your full response text"
                            }
                        },
                        "required": ["response"],
                    },
                }]
                dummy_tool = True

            # Tool execution loop
            if effective_tools:
                tool_map = {t["name"]: t for t in effective_tools}
                available_tools = {
                    t["name"]: {"type": "function", "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}})
                    }}
                    for t in effective_tools
                }

                consecutive_list_files = 0  # Track consecutive list_files calls

                for turn in range(max_turns):
                    response = client.chat.completions.create(
                        model=effective_model,
                        max_tokens=effective_max_tokens,
                        messages=api_messages,
                        tools=list(available_tools.values()) if available_tools else None,
                        tool_choice="auto",
                    )

                    choice = response.choices[0]
                    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                        # ── Collect ALL tool_calls into ONE assistant message ──
                        # OpenAI protocol requires exactly one assistant message with
                        # all tool_calls, followed by individual tool result messages.
                        # Appending per-tool assistant messages causes consecutive
                        # assistant messages which violates the API contract.
                        assistant_tool_calls = []
                        pending_results = []  # [(tool_call_id, func_name, result_content)]

                        for tool_call in choice.message.tool_calls:
                            func_name = tool_call.function.name
                            assistant_tool_calls.append({
                                "id": tool_call.id,
                                "type": "function",
                                "function": {"name": func_name, "arguments": tool_call.function.arguments}
                            })

                            # Parse tool arguments with truncation recovery
                            raw_args = tool_call.function.arguments
                            repaired = False
                            try:
                                func_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                func_args = self._repair_json_args(raw_args)
                                if not func_args:
                                    logger.warning(f"Skipping {func_name}: JSON args irrecoverable")
                                    pending_results.append((
                                        tool_call.id, func_name,
                                        json.dumps({
                                            "error": "JSON arguments were truncated and could not be recovered. "
                                                     "Please retry with shorter/simpler arguments."
                                        })
                                    ))
                                    continue
                                repaired = True
                                logger.warning(f"Recovered truncated JSON args for {func_name}: {str(func_args)[:200]}")

                            # Ensure func_args is always a dict
                            if not isinstance(func_args, dict):
                                logger.warning(f"Tool args for {func_name} is {type(func_args).__name__}, wrapping in dict")
                                func_args = {"raw": func_args}

                            # For write_file with repaired args, reject if content appears truncated
                            if repaired and func_name == "write_file" and isinstance(func_args, dict) and "content" in func_args:
                                content = func_args["content"]
                                if content and not content.rstrip().endswith(('\n', '}', ']', ')', '"""', "'''", '`', '.')):
                                    logger.warning(
                                        f"Rejecting write_file to {func_args.get('path','?')}: "
                                        f"content appears truncated (ends with: ...{content[-50:]})"
                                    )
                                    pending_results.append((
                                        tool_call.id, func_name,
                                        json.dumps({
                                            "error": "File content was TRUNCATED by max_tokens. "
                                                     "The file was NOT written to avoid corruption. "
                                                     "Please split into smaller chunks or write only the changed sections."
                                        })
                                    ))
                                    continue

                            logger.info(f"Executing tool: {func_name}")

                            # Handle dummy "respond" tool — extract text and return immediately
                            if dummy_tool and func_name == "respond":
                                if isinstance(func_args, dict):
                                    return func_args.get("response", "")
                                elif isinstance(func_args, str):
                                    return func_args
                                else:
                                    return str(func_args)

                            if func_name in tool_map:
                                # ── Rate-limit consecutive list_files calls ──
                                if func_name == "list_files":
                                    consecutive_list_files += 1
                                    if consecutive_list_files > 3:
                                        tool_result = json.dumps({
                                            "error": (
                                                "Too many consecutive list_files calls. "
                                                "You already know the directory structure. "
                                                "Focus on the PRIMARY task."
                                            )
                                        })
                                        consecutive_list_files = 0
                                    else:
                                        tool_result = self._execute_tool_with_trace(func_name, func_args, trace)
                                else:
                                    consecutive_list_files = 0
                                    tool_result = self._execute_tool_with_trace(func_name, func_args, trace)
                                pending_results.append((tool_call.id, func_name, str(tool_result)))
                            else:
                                pending_results.append((
                                    tool_call.id, func_name,
                                    json.dumps({"error": f"Unknown tool: {func_name}"})
                                ))

                        # ── Append ONE assistant message with ALL tool_calls ──
                        api_messages.append({
                            "role": "assistant",
                            "tool_calls": assistant_tool_calls,
                        })

                        # ── Append individual tool result messages ──
                        budget_msg = (
                            f"\n[SYSTEM] Turn {turn+1}/{max_turns}. "
                            f"Remaining: {max_turns - turn - 1}."
                        )
                        if max_turns - turn - 1 <= int(max_turns * 0.2):
                            budget_msg += (
                                " CRITICAL: Almost out of turns. "
                                "You MUST finish your primary task NOW or report failure."
                            )
                        elif max_turns - turn - 1 <= int(max_turns * 0.4):
                            budget_msg += (
                                " WARNING: Past 60% of budget. "
                                "Stop exploring and focus on the PRIMARY task."
                            )

                        for tc_id, _fname, result_text in pending_results:
                            # Smart truncation: try to keep metric-related lines
                            if len(result_text) > 7900:
                                lines = result_text.split('\n')
                                metric_lines = [l for l in lines if any(
                                    kw in l.lower() for kw in [
                                        'mae', 'mse', 'loss', 'epoch', 'val_',
                                        'best', 'metric', 'score', 'accuracy',
                                        'routing_w', 'train_', 'final',
                                    ]
                                )]
                                if metric_lines:
                                    head = '\n'.join(lines[:20])
                                    tail = '\n'.join(metric_lines[-30:])
                                    truncated = f"{head}\n... [TRUNCATED] key metrics:\n{tail}"
                                    tool_content = truncated[:7900]
                                else:
                                    tool_content = result_text[:7900]
                            else:
                                tool_content = result_text
                            tool_content += budget_msg
                            api_messages.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": tool_content[:8000],
                            })
                        continue
                    else:
                        # No more tool calls, return the response
                        return choice.message.content if choice.message.content else ""

                # Max turns reached — find last assistant message for best text
                return self._find_last_assistant_text(api_messages, "Max turns reached")
            else:
                # No tools, simple call
                response = client.chat.completions.create(
                    model=effective_model,
                    max_tokens=effective_max_tokens,
                    messages=api_messages,
                )
                return response.choices[0].message.content if response.choices else ""

        except ImportError:
            logger.warning("openai package not installed. Using mock response.")
            return json.dumps({"action": "wait", "reason": "LLM not available"})
        except Exception as e:
            logger.error(f"{provider_label} API call failed: {e}")
            # Re-raise so _call_llm can try failover
            raise

    # ─────────────────────────────────────────────────
    # Anthropic provider (different protocol)
    # ─────────────────────────────────────────────────

    def _call_anthropic(self, system: str, messages: list, tools: list = None, max_turns: int = 10, trace: ToolTrace = None, task_tier: str = None) -> str:
        """Call Anthropic Claude API with tool execution support."""
        # Dynamic max_tokens for Anthropic
        _MAX_TOKENS_MAP = {
            "code": 8192, "writing": 16384, "researcher": 16384,
            "idea": 16384, "think": 16384, "reflect": 16384,
        }
        effective_max_tokens = _MAX_TOKENS_MAP.get(task_tier, 16384)

        logger.info(f"Calling Anthropic Claude API: model={self.model}, messages={len(messages)}, tools={bool(tools)}")
        try:
            import anthropic

            client = anthropic.Anthropic()

            api_messages = []
            for msg in messages:
                api_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

            kwargs = {
                "model": self.model,
                "max_tokens": effective_max_tokens,
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": api_messages,
            }

            if tools:
                kwargs["tools"] = [{"name": t["name"], "description": t.get("description", ""), "input_schema": t.get("input_schema", {"type": "object", "properties": {}})} for t in tools]
                tool_map = {t["name"]: t for t in tools}

                for turn in range(max_turns):
                    response = client.messages.create(**kwargs)
                    content = response.content

                    # Check if ANY block is tool_use (not just the first one)
                    has_tool_use = any(
                        hasattr(block, "type") and block.type == "tool_use"
                        for block in (content or [])
                    )

                    if has_tool_use:
                        # Collect text blocks for the conversation history
                        text_parts = []
                        for block in content:
                            if hasattr(block, "type") and block.type == "text":
                                text_parts.append(block.text)
                            elif hasattr(block, "type") and block.type == "tool_use":
                                func_name = block.name
                                func_args = block.input
                                logger.info(f"Executing tool: {func_name}")

                                if func_name in tool_map:
                                    tool_result = self._execute_tool_with_trace(func_name, func_args, trace)
                                    api_messages.append({"role": "user", "content": [{
                                        "type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": str(tool_result)[:8000]
                                    }]})
                                else:
                                    api_messages.append({"role": "user", "content": [{
                                        "type": "tool_result",
                                        "tool_use_id": block.id,
                                        "content": json.dumps({"error": f"Unknown tool: {func_name}"})
                                    }]})

                        # If there were text blocks alongside tool_use, log them
                        if text_parts:
                            logger.debug(f"Anthropic text alongside tool_use: {' '.join(text_parts)[:200]}")
                        continue
                    else:
                        # No tool calls — extract text from content blocks
                        text_parts = []
                        for block in (content or []):
                            if hasattr(block, "type") and block.type == "text":
                                text_parts.append(block.text)
                        return "\n".join(text_parts) if text_parts else ""

                # Max turns reached — find last assistant text
                return self._find_last_assistant_text(api_messages, "Max turns reached")
            else:
                response = client.messages.create(**kwargs)
                return response.content[0].text if response.content else ""

        except ImportError:
            logger.warning("anthropic package not installed. Trying openai fallback.")
            return self._call_openai_compatible(
                system=system, messages=messages, tools=tools,
                max_turns=max_turns, trace=trace,
                base_url=None, api_key=None, provider_label="openai_fallback",
            )

    # ─────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────

    @staticmethod
    def _repair_json_args(raw: str) -> dict:
        """Attempt to recover a dict from truncated JSON tool arguments.

        LLM output can be cut off mid-string (e.g. by max_tokens), producing
        invalid JSON like: {"path": "src/model.py", "content": "def foo():\n  r
        This tries several recovery strategies.
        """
        if not raw:
            return {}

        # Strategy 1: close all open braces/brackets
        for suffix in ['"}', '"}]', '"]}', '"}]}']:
            try:
                return json.loads(raw + suffix)
            except json.JSONDecodeError:
                pass

        # Strategy 1.5: handle single-quote JSON (LLMs sometimes use single quotes)
        if "'" in raw and '"' not in raw:
            try:
                return json.loads(raw.replace("'", '"'))
            except json.JSONDecodeError:
                pass

        # Strategy 2: strip trailing incomplete key-value pair
        # Find last complete key-value and truncate there
        s = raw.rstrip()
        attempts = 0
        while s and not s.endswith('}') and attempts < 5:
            attempts += 1
            last_comma = s.rfind('",')
            if last_comma < 0:
                last_comma = s.rfind("',")
            if last_comma > 0:
                s = s[:last_comma + 1] + '}'
            else:
                break
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                continue

        # Strategy 3: return empty dict — let the tool fail gracefully
        return {}

    @staticmethod
    def _find_last_assistant_text(api_messages: list, fallback: str) -> str:
        """Find the last assistant message with text content.

        When max turns is reached, the last message in api_messages
        may be a tool result (role="tool"), not an assistant message.
        This helper searches backwards for the last assistant text.
        Also checks tool_calls messages that may have text alongside tool calls.
        """
        for msg in reversed(api_messages):
            if msg.get("role") != "assistant":
                continue
            # Direct text content
            if msg.get("content"):
                return msg["content"]
            # Tool calls message — extract text from tool arguments as last resort
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in reversed(tool_calls):
                    try:
                        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
                        # For respond tool, return its response field
                        if isinstance(args, dict) and "response" in args:
                            return args["response"]
                    except (json.JSONDecodeError, TypeError):
                        pass
        return fallback

    def _load_prompt(self, filename: str) -> str:
        """Load agent prompt from agents/ directory."""
        prompt_path = AGENTS_DIR / filename
        if prompt_path.exists():
            return prompt_path.read_text()
        logger.warning(f"Prompt file not found: {prompt_path}")
        return f"You are the {filename.replace('.md', '')} agent."

    # Reasoning principles injected into every leader dispatch
    _REASONING_REMINDER = (
        "## Reasoning Checklist (MANDATORY)\n"
        "Before deciding, you MUST address:\n"
        "1. **Assumptions**: What am I assuming? Write them out.\n"
        "2. **Alternatives**: Is there a simpler way? Am I changing too many variables?\n"
        "3. **Success criteria**: Concrete, measurable — not 'improve' but 'MAE < 0.35'.\n"
        "4. **Surgical**: Every code change must trace to this experiment's hypothesis.\n"
        "5. **Honesty**: If results don't meet criteria, say so — don't spin.\n"
        "6. **Verify-first**: If VERIFY found module failures, address those BEFORE judging the experiment.\n"
        "7. **Data integrity**: ALWAYS specify which dataset class to use. NEVER allow synthetic data.\n"
    )

    def _generate_project_knowledge(self, context: dict) -> str:
        """Generate a brief project knowledge summary from available context.

        This provides the Leader with key facts about the project structure,
        dataset classes, and data paths that it needs to correctly instruct
        the Code Agent. Without this, the Leader might forget or hallucinate
        data sources.
        """
        workspace_dir = context.get("workspace_dir", "")
        if not workspace_dir:
            return ""

        knowledge_parts = []
        workspace = Path(workspace_dir)

        # Check for key files and extract relevant info
        dataset_init = workspace / "datasets" / "__init__.py"
        if dataset_init.exists():
            try:
                content = dataset_init.read_text()
                # Extract exported class names
                imports = re.findall(r"(?:from|import)\s+(\w+)", content)
                classes = re.findall(r"class\s+(\w+)", content)
                if classes or imports:
                    knowledge_parts.append(
                        f"- Dataset classes available: {', '.join(set(classes + imports))}"
                    )
            except Exception:
                pass

        models_init = workspace / "models" / "__init__.py"
        if models_init.exists():
            try:
                content = models_init.read_text()
                classes = re.findall(r"class\s+(\w+)", content)
                functions = re.findall(r"def\s+(\w+)", content)
                if classes or functions:
                    knowledge_parts.append(
                        f"- Model classes: {', '.join(classes[:5])}"
                    )
            except Exception:
                pass

        # Check data directory
        data_dir = workspace / "data"
        if data_dir.exists():
            subdirs = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if subdirs:
                knowledge_parts.append(
                    f"- Data directories: {', '.join(subdirs[:10])}"
                )

        # Check DATASET_MANIFEST
        manifest_path = workspace / "DATASET_MANIFEST.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                datasets = manifest.get("datasets", {})
                total_scenes = 0
                for ds in (datasets.values() if isinstance(datasets, dict) else datasets):
                    if not isinstance(ds, dict):
                        continue
                    ds_count = 0
                    scenes = ds.get("scenes")
                    if isinstance(scenes, (dict, list)):
                        ds_count = len(scenes)
                    if ds_count == 0:
                        ds_count = ds.get("total_valid_scenes", 0) or 0
                    total_scenes += ds_count
                knowledge_parts.append(
                    f"- DATASET_MANIFEST: {len(datasets)} datasets, {total_scenes} total scenes"
                )
            except Exception:
                pass

        if knowledge_parts:
            return (
                "Key facts about this project (auto-detected):\n"
                + "\n".join(knowledge_parts)
                + "\n\nWhen dispatching to Code Agent, ALWAYS specify the dataset class "
                "and remind it to use real data (not synthetic)."
            )
        return ""

    def _format_leader_input(self, task: str, context: dict) -> str:
        """Format context into a structured input for the Leader."""
        parts = [f"## Task: {task.upper()}\n"]

        # Inject reasoning principles reminder for every dispatch
        parts.append(self._REASONING_REMINDER)

        # For REFLECT phase: remind Leader about available tools for cross-validation
        if task == "reflect":
            parts.append(
                "## REFLECT Phase Tools Available\n"
                "You have `read_file` and `list_files` tools available during REFLECT.\n"
                "Use them to perform Step 3.5 cross-validation:\n"
                "- Read model source code (models/*.py) to verify architecture matches expectations\n"
                "- Read DATASET_MANIFEST.json to check data splits and scene counts\n"
                "- Read training logs to verify loss curves and metric trends\n"
                "- List files in outputs/ to check what artifacts were produced\n"
                "This lets you trace visual analysis findings back to concrete code/data causes.\n\n"
            )

        # Inject working directory so LLM uses correct paths
        if context.get("workspace_dir"):
            parts.append(
                f"## Working Directory (CRITICAL)\n"
                f"The code agent's working directory is: `{context['workspace_dir']}`\n"
                f"All file paths and commands must be relative to this directory.\n"
                f"Do NOT use /workspace or any other hardcoded path.\n\n"
            )

        # Inject project knowledge summary (auto-generated)
        project_knowledge = self._generate_project_knowledge(context)
        if project_knowledge:
            parts.append(f"## Project Knowledge\n{project_knowledge}\n")

        if context.get("directive"):
            parts.append(f"## Human Directive (HIGHEST PRIORITY)\n{context['directive']}\n")

        parts.append(f"## Project Brief\n{context.get('brief', 'N/A')}\n")
        parts.append(f"## Memory Log\n{context.get('memory_log', 'N/A')}\n")
        parts.append(f"## Cycle: {context.get('cycle', 'N/A')}\n")

        # ── Inject session statistics from SQLite (if available) ──
        stats = context.get("session_stats")
        if stats and stats.get("total_cycles", 0) > 0:
            parts.append("## Session Statistics\n")
            parts.append(f"- Total cycles: {stats['total_cycles']}\n")
            parts.append(f"- Experiments launched: {stats['experiments_launched']} ({stats['launch_rate']*100:.0f}%)\n")
            parts.append(f"- Dead ends accumulated: {stats['dead_ends_count']}\n")
            recent_failures = context.get("recent_failures", [])
            if recent_failures:
                parts.append("\n### Recent Failure Patterns:\n")
                for f in recent_failures[:3]:
                    diag = f.get("verify_diagnosis", "") or f.get("active_problem", "")
                    parts.append(f"- Cycle {f.get('cycle', '?')}: {diag[:150]}\n")

        # ── Inject code review lessons from knowledge base ──
        # These are past mistakes the agent has learned from.
        code_review_lessons = context.get("code_review_lessons")
        if code_review_lessons:
            parts.append(f"\n{code_review_lessons}\n")
        relevant_lessons = context.get("relevant_code_review_lessons")
        if relevant_lessons:
            parts.append(f"\n{relevant_lessons}\n")

        if context.get("experiment_result"):
            # Cap experiment result to prevent context overflow
            result_str = json.dumps(context['experiment_result'], indent=2)
            if len(result_str) > 4000:
                result_str = result_str[:4000] + "\n... (truncated for brevity)"
            parts.append(f"## Experiment Result\n{result_str}\n")

        # Inject VERIFY report diagnosis for REFLECT tasks
        if context.get("verify_diagnosis"):
            parts.append("## VERIFY Report — Module Diagnosis\n")
            parts.append("**CRITICAL: You MUST address these verification failures before drawing conclusions.**\n")
            parts.append("The following modules did NOT function correctly:\n")
            for diag in context["verify_diagnosis"]:
                parts.append(f"- {diag}\n")
            if context.get("verify_failed_modules"):
                parts.append(f"\nFailed modules: {', '.join(context['verify_failed_modules'])}\n")
            parts.append(
                "\n**Action required**: Before deciding this experiment 'failed' or 'succeeded', "
                "determine whether the failure is in the experiment logic or in a broken module. "
                "If a module is broken, the experiment results are UNRELIABLE — fix the module first.\n"
            )

        # ── ANTI-DECEPTION: Inject fabrication warning if detected ──
        if context.get("llm_fabrication_detected"):
            parts.append("## LLM FABRICATION DETECTED \n")
            parts.append("**The Code agent CLAIMED to perform actions that it did NOT actually perform.**\n\n")
            parts.append("Evidence:\n")
            for detail in context.get("fabrication_details", []):
                parts.append(f"- {detail}\n")
            parts.append(
                "\n**MANDATORY ACTIONS:**\n"
                "1. Do NOT trust any claims from the previous EXECUTE phase\n"
                "2. Mark this cycle's results as UNRELIABLE\n"
                "3. Record `module_failure` for 'llm_honesty' — the Code agent must be re-instructed\n"
                "4. The next THINK must explicitly verify tool trace before trusting any output\n"
            )

        # ── VISUAL ANALYSIS: Inject multimodal diagnosis when available ──
        va = context.get("visual_analysis")
        if va and va.get("triggered"):
            severity_emoji = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}
            sev = severity_emoji.get(va.get("severity", "info"), "INFO")
            parts.append(f"## {sev}: VISUAL ANALYSIS DIAGNOSIS\n")
            parts.append(
                f"The agent has run **inference + multimodal image analysis** after "
                f"{context.get('visual_analysis', {}).get('images_analyzed', '?')} images.\n"
                f"This analysis LOOKS at model predictions (not just numbers) to find failures.\n\n"
            )
            va_diags = context.get("visual_analysis_diagnosis", [])
            if va_diags:
                parts.append("### Visual Findings:\n")
                for d in va_diags[:5]:
                    parts.append(f"- {d}\n")
            va_actions = context.get("visual_analysis_actions", [])
            if va_actions:
                parts.append("\n### Recommended Actions (from visual analysis):\n")
                for a in va_actions[:5]:
                    parts.append(f"1. {a}\n")
                parts.append(
                    "\n**IMPORTANT**: These recommendations come from actual visual inspection of "
                    "model outputs. They reveal problems invisible to numeric metrics alone. "
                    "You SHOULD incorporate them into your next experiment plan.\n"
                )

        return "\n".join(parts)

    def _parse_leader_response(self, response: str) -> dict:
        """Parse Leader's response into structured action."""
        try:
            # Try to find JSON in response (supports nested braces)
            depth = 0
            start = None
            for i, ch in enumerate(response):
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidate = response[start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            # Validate it's a decision JSON with an "action" field
                            if isinstance(parsed, dict) and "action" in parsed:
                                return parsed
                            # Not a decision — keep searching for the actual decision JSON
                        except json.JSONDecodeError:
                            # This brace pair wasn't valid JSON; keep searching
                            start = None
        except (AttributeError, IndexError):
            pass

        # Fallback: extract action from text
        response_lower = response.lower()
        if "wait" in response_lower or "no experiment" in response_lower:
            return {"action": "wait", "reason": response[:200]}

        return {
            "action": "experiment",
            "agent": "code",
            "task": response,
        }

    def _parse_worker_response(self, response: str, agent_type: str, trace: ToolTrace = None) -> dict:
        """Parse worker response into structured result.

        ANTI-DECEPTION DESIGN:
        Key facts (PID, log_file, exit codes) are extracted from the
        ToolTrace (system-level tool return values), NOT from the LLM's
        narrative text. The LLM might claim "experiment launched with PID
        12345" but if launch_experiment was never called, or returned an
        error, the trace will reveal the truth.

        The LLM text is still included as 'response' for REFLECT context,
        but all machine-actionable fields come from tool results.
        """
        result = {"agent": agent_type, "response": response}

        # Attach tool trace for downstream verification
        if trace is not None:
            result["tool_trace"] = trace.to_dict()

        # ── ANTI-DECEPTION: Extract facts from TOOL RESULTS, not LLM text ──
        if agent_type == "code" and trace is not None:
            launch_facts = trace.extract_launch_facts()

            if launch_facts:
                # launch_experiment WAS called — get facts from actual tool result
                result["experiment_launched"] = True
                if "pid" in launch_facts:
                    # Normalize PID to int for consistent downstream usage
                    try:
                        result["pid"] = int(launch_facts["pid"])
                    except (ValueError, TypeError):
                        result["pid"] = launch_facts["pid"]
                if "log_file" in launch_facts:
                    result["log_file"] = launch_facts["log_file"]
                if "launch_error" in launch_facts:
                    # Tool returned an error — experiment did NOT actually launch
                    result["experiment_launched"] = False
                    result["launch_error"] = launch_facts["launch_error"]
            else:
                # launch_experiment was NEVER called — but maybe the code agent
                # launched training via run_shell (nohup/python train.py).
                # This is a common pattern when LLMs don't follow instructions.
                shell_facts = trace.extract_shell_facts()
                training_via_shell = False
                for sf in shell_facts:
                    cmd = sf.get("command", "")
                    # Detect training commands launched via run_shell
                    # Exclude --help, --dry_run, and other non-training invocations
                    is_help = bool(re.search(r'\b--help\b|\b-h$', cmd))
                    is_inspect = bool(re.search(r'\bpython\s+-c\b', cmd) and not re.search(r'train|epoch', cmd, re.IGNORECASE))
                    if not sf.get("had_error", False) and not is_help and not is_inspect and re.search(
                        r'\bnohup\s+.*python.*train'
                        r'|\bpython\s+.*train.*\.py\s'
                        r'|\bpython\s+.*train.*\.py$',
                        cmd, re.IGNORECASE,
                    ):
                        training_via_shell = True
                        result["experiment_launched"] = True
                        result["launch_via_shell"] = True
                        logger.info(
                            f"Detected training via run_shell (not launch_experiment): "
                            f"{cmd[:100]}"
                        )
                        # Try to extract PID from nohup output
                        stdout = sf.get("stdout_preview", "")
                        pid_match = re.search(r'\bPID[=:]\s*(\d+)', stdout, re.IGNORECASE)
                        if not pid_match:
                            pid_match = re.search(r'\[(\d+)\]', stdout)
                        if pid_match:
                            try:
                                result["pid"] = int(pid_match.group(1))
                            except ValueError:
                                pass
                        break

                if not training_via_shell:
                    # Check if the LLM falsely claimed to have launched
                    llm_claims_launch = bool(re.search(
                        r'\bexperiment\s+(?:was\s+)?launched\b'
                        r'|\bPID\s*[=:]\s*\d+'
                        r'|\blaunched\s+(?:the\s+)?experiment\b'
                        r'|\btraining\s+(?:has\s+)?started\b',
                        response, re.IGNORECASE,
                    ))
                    if llm_claims_launch:
                        logger.warning(
                            "ANTI-DECEPTION: LLM claims experiment launched, "
                            "but launch_experiment tool was never called! "
                            "Rejecting the claim."
                        )
                        result["experiment_launched"] = False
                        result["deception_detected"] = True
                        result["deception_detail"] = (
                            "LLM text claims experiment launched, but launch_experiment "
                            "tool was never called. The claim is FABRICATED."
                        )
                    else:
                        result["experiment_launched"] = False

            # Extract dry-run / pre-flight evidence from shell results
            shell_facts = trace.extract_shell_facts()
            if shell_facts:
                result["shell_commands_run"] = len(shell_facts)
                result["shell_commands_ok"] = sum(
                    1 for s in shell_facts if not s.get("had_error", False)
                )
                # Check if dry-run was actually performed
                for sf in shell_facts:
                    cmd = sf.get("command", "")
                    if re.search(r'\b(dry[\s_-]?run|--dry)\b', cmd, re.IGNORECASE) or \
                       re.search(r'\bmax.steps[\s=]+2\b', cmd):
                        result["dry_run_performed"] = True
                        result["dry_run_passed"] = not sf.get("had_error", False)
                        break

        elif agent_type == "code" and trace is None:
            # Fallback for backward compatibility (no trace available)
            # This is the OLD, deception-vulnerable path
            logger.warning("No tool trace available — falling back to text-based parsing (deception-vulnerable)")
            if "PID" in response or "launched" in response.lower():
                result["experiment_launched"] = True
                pid_match = re.search(r"PID[=:\s]+(\d+)", response)
                if pid_match:
                    result["pid"] = int(pid_match.group(1))

        return result
