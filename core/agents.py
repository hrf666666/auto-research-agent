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
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("autoresearcher.agents")


class _LLMCallTimeout(Exception):
    """Raised when a single LLM API call exceeds its wall-clock budget.

    GLM's streaming mode (for chunk in response_stream) is NOT protected by
    the socket timeout — if the server stops sending chunks, the client blocks
    forever (verified: process stuck in do_poll for 4+ min). This exception
    is raised by SIGALRM to break out of that block so failover can kick in.
    """


def _llm_timeout_handler(signum, frame):
    raise _LLMCallTimeout("LLM call exceeded wall-clock timeout (stream hung)")


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
                    # F1 fix: once we have a successful pid, stop scanning so a
                    # later failed call (or an earlier stale launch_error)
                    # can't overwrite/contradict it. Clear any stale error.
                    if "pid" in result_data:
                        facts["pid"] = result_data["pid"]
                        facts.pop("launch_error", None)
                        if "log_file" in result_data:
                            facts["log_file"] = result_data["log_file"]
                        if "status" in result_data:
                            facts["launch_status"] = result_data["status"]
                        break  # successful launch found — done
                    if "log_file" in result_data:
                        facts["log_file"] = result_data["log_file"]
                    if "status" in result_data:
                        facts["launch_status"] = result_data["status"]
                    # If launch_experiment returned an error, record it
                    if "error" in result_data:
                        facts["launch_error"] = result_data["error"]
                except (json.JSONDecodeError, TypeError):
                    # F2 fix: previously `pass` made the anti-deception layer
                    # blind to non-JSON tool results (plain error strings,
                    # tracebacks). Record the raw text so a genuine launch
                    # failure is visible rather than indistinguishable from
                    # "launch never called".
                    if "launch_error" not in facts:
                        facts["launch_error"] = (call.result or "")[:500]
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
        # glm-5.2 works on coding/paas/v4 endpoint (verified 2026-06-15)
        # Thinking (reasoning_content) auto-enabled via base_url.
        "strong_model": "glm-5.2",            # Best GLM (thinking auto-enabled)
        "fast_model": "glm-5",              # Fast GLM for routine tasks
        # Model-level failover chains: if primary model fails, try next in list
        "strong_model_chain": [
            "glm-5.2",              # GLM 5.2 (strongest, thinking auto-enabled)
            "glm-5.1",              # GLM 5.1 (thinking)
            "glm-5",                # GLM 5
            "glm-5-turbo",          # GLM 5 Turbo (fast)
            "glm-4.7",              # GLM 4.7
        ],
        "fast_model_chain": [
            "glm-5",                # GLM 5
            "glm-5-turbo",          # GLM 5 Turbo
            "glm-4.7",              # GLM 4.7
            "glm-4.6",              # GLM 4.6
        ],
        "models": [
            "glm-4.5",              # GLM 4.5
            "glm-4.5-air",          # GLM 4.5 Air (lightweight)
            "glm-4.6",              # GLM 4.6
            "glm-4.7",              # GLM 4.7
            "glm-5",                # GLM 5
            "glm-5-turbo",          # GLM 5 Turbo (fast)
            "glm-5.1",              # GLM 5.1 (thinking)
            "glm-5.2",              # GLM 5.2 (thinking) — strongest, coding plan
        ],
    },
    "ali_token_plan": {
        "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "env_key": "ALI_TOKEN_PLAN_API_KEY",
        "strong_model": "qwen3.7-max",      # Best reasoning model on Ali token plan
        "fast_model": "qwen3.6-plus",       # Fast tier: vision + reasoning + text
        # Model-level failover chains: if primary model fails, try next in list
        "strong_model_chain": [
            "qwen3.7-max",          # Qwen: reasoning, text generation (strongest)
            "deepseek-v4-pro",      # DeepSeek: reasoning, text generation
            "glm-5.1",              # Zhipu AI: text generation (thinking)
            "qwen3.6-plus",         # Qwen: fallback to fast tier
        ],
        "fast_model_chain": [
            "qwen3.6-plus",         # Qwen: reasoning, vision, text generation
            "qwen3.6-flash",        # Qwen: fastest tier
            "deepseek-v4-flash",    # DeepSeek: fast reasoning
            "glm-5",                # Zhipu AI: text generation
        ],
        "models": [
            "qwen3.7-max",          # Qwen: reasoning, text generation (strong)
            "qwen3.6-plus",         # Qwen: reasoning, vision, text generation (fast)
            "qwen3.6-flash",        # Qwen: reasoning, vision, text generation (fastest)
            "qwen-image-2.0",       # Qwen: image generation
            "qwen-image-2.0-pro",   # Qwen: image generation (pro)
            "wan2.7-image",         # Wanxiang: image generation
            "wan2.7-image-pro",     # Wanxiang: image generation (pro)
            "deepseek-v4-pro",      # DeepSeek: reasoning, text generation
            "deepseek-v4-flash",    # DeepSeek: reasoning, text generation (fast)
            "deepseek-v3.2",        # DeepSeek: reasoning, text generation
            "glm-5.2",              # Zhipu AI: thinking (403 on standard GLM plan; valid on some Ali tiers)
            "glm-5.1",              # Zhipu AI: text generation
            "glm-5",                # Zhipu AI: text generation
            "MiniMax-M2.5",         # MiniMax: reasoning, text generation
        ],
    },
}

# Failover order: GLM first, ALI as backup.
#
# Two-level failover:
#   Level 1 (model-level): Within a provider, if the primary model fails (e.g. qwen3.7-max),
#     try the next model in strong_model_chain / fast_model_chain before giving up on the provider.
#   Level 2 (provider-level): If ALL models in a provider's chain fail, switch to the next
#     provider in TOKEN_PLAN_FAILOVER_ORDER and try its model chain.
#
TOKEN_PLAN_FAILOVER_ORDER = ["glm_token_plan", "ali_token_plan"]

# Tasks that require the strong model (complex reasoning / planning).
# Everything else uses the fast model.
STRONG_MODEL_TASKS = {"think", "reflect", "idea", "researcher", "code"}

# Exploration-only tools that a code agent is blocked from calling once it
# passes 60% of its turn budget (Phase 2 convergence gate). Tools NOT in this
# set (write_file, launch_experiment, run_shell, diagnose_error, analyze_model,
# probe_model) remain available so the agent can finalize and launch.
_CODE_EXPLORE_TOOLS = frozenset({
    "read_file", "list_files", "web_search", "web_fetch",
    "search_papers", "get_paper",
})


_QUOTA_CODE_PATTERNS = {"1308", "1220"}  # GLM quota codes observed in production
_QUOTA_KEYWORDS = ("使用上限", "配额", "quota", "exhausted", "limit reached",
                   "will reset", "将在", "重置")
# Parse a reset timestamp like "2026-06-15 19:42:06" from the error message.
_RESET_TIME_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})")


def _extract_error_body(exc: Exception) -> dict:
    """Best-effort extraction of the structured error body from an SDK exception.

    openai/zai APIStatusError variants expose the parsed JSON body in different
    ways (``.body``, ``.error``, ``.response``). We probe each and also fall
    back to scanning the string representation for a JSON object.
    """
    for attr in ("body", "error"):
        val = getattr(exc, attr, None)
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
    # Fall back to scanning str(exc) for embedded JSON.
    msg = str(exc)
    m = re.search(r'\{.*\}', msg)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            pass
    return {}


def _classify_429(exc: Exception) -> dict:
    """Classify a 429 error as rate-limit vs quota-window-exhausted.

    Returns ``{"type": "rate_limit"|"quota_exhausted", "reset_time": datetime|None}``.
    A quota-exhausted 429 carries the reset timestamp so the caller can cool
    the provider until the window actually resets, instead of a fixed 300s.

    Detection, in priority order:
      1. Error body code in _QUOTA_CODE_PATTERNS (GLM 1308/1220).
      2. Error body/message contains a quota keyword.
    Falls back to ``rate_limit`` (transient) when no quota signal is found.
    """
    msg = str(exc)
    body = _extract_error_body(exc)
    # Drill into {"error": {"code": ..., "message": ...}} nesting.
    err_obj = body.get("error", body) if isinstance(body, dict) else {}
    code = str(err_obj.get("code", "")) if isinstance(err_obj, dict) else ""
    inner_msg = str(err_obj.get("message", "")) if isinstance(err_obj, dict) else ""

    # Scan BOTH the raw exception text and the nested error message for a
    # reset timestamp (GLM puts it in error.message, not in the HTTP reason).
    reset_time = None
    for haystack in (inner_msg, msg):
        m = _RESET_TIME_RE.search(haystack)
        if m:
            try:
                reset_time = datetime.strptime(m.group(1).replace("T", " "),
                                               "%Y-%m-%d %H:%M:%S")
                break
            except ValueError:
                continue

    combined = f"{code} {inner_msg} {msg}".lower()
    if code in _QUOTA_CODE_PATTERNS:
        return {"type": "quota_exhausted", "reset_time": reset_time}
    for kw in _QUOTA_KEYWORDS:
        if kw.lower() in combined:
            return {"type": "quota_exhausted", "reset_time": reset_time}
    return {"type": "rate_limit", "reset_time": None}


def _is_permanent_error(exc: Exception) -> bool:
    """Classify an exception as a PERMANENT provider error.

    Permanent errors (auth failure, bad model name, forbidden) must not be
    retried across the whole model×provider matrix — they will fail every
    time and just burn rate-limit budget.

    Quota-window-exhausted 429s are ALSO permanent-until-reset: the whole
    API key shares one quota window, so every model in the chain fails
    identically until the window resets. A per-minute rate-limit 429 stays
    transient (retry the next model after a short backoff).

    Uses the SDK exception's structured ``status_code`` attribute rather than
    string-matching error messages (which is brittle: " 400" matches "400
    samples" in a normal response). The openai and anthropic SDKs both expose
    ``APIStatusError`` subclasses whose ``status_code`` reflects the real HTTP
    status, so this is precise. Degrades to "transient" (returns False) when
    the exception carries no status info, so unknown errors still get retried.
    """
    # 0. Quota-window 429 — permanent-until-reset (breaks the model chain).
    status = getattr(exc, "status_code", None)
    if status == 429:
        return _classify_429(exc)["type"] == "quota_exhausted"

    # 1. Structured status_code attribute (openai/anthropic APIStatusError).
    if status is not None:
        # 4xx client errors are permanent (bad request/auth/forbidden/not found).
        # 5xx server errors are transient.
        if isinstance(status, int) and 400 <= status < 500:
            return True

    # 2. SDK type-based fallback (covers cases where status_code wasn't set
    # but the exception type is unambiguous). Lazy-import so missing SDKs
    # don't break the module.
    permanent_types = ()
    for mod_name in ("openai", "anthropic"):
        try:
            mod = __import__(mod_name)
            permanent_types += (
                mod.AuthenticationError,
                mod.PermissionDeniedError,
                mod.BadRequestError,
                mod.NotFoundError,
            )
        except (ImportError, AttributeError):
            pass
    if permanent_types and isinstance(exc, permanent_types):
        return True

    return False


# ── Lightweight synthetic response objects for GLM streaming mode ──
# Defined at module level to avoid repeated class creation inside the tool loop.
class _SyntheticMessage:
    """Mimics openai.types.chat.ChatCompletionMessage for streaming assembly."""
    __slots__ = ("content", "tool_calls")


class _SyntheticChoice:
    """Mimics openai.types.chat.Choice for streaming assembly."""
    __slots__ = ("finish_reason", "message")


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
            # NOTE: "tools" field removed (Fix D) — it was never read (always
            # overridden by ToolRegistry.get_tools_for(agent_type)). Keeping it
            # created a divergent second source of truth.
        },
        "code": {
            "prompt_file": "code_agent.md",
            "max_turns": 40,
        },
        "writing": {
            "prompt_file": "writing_agent.md",
            "max_turns": 30,
        },
        "researcher": {
            "prompt_file": "researcher_agent.md",
            "max_turns": 30,
        },
    }

    # Provider health tracking for auto-failover
    _provider_health: dict[str, dict] = {}  # class-level: shared across instances

    # Max output tokens per task tier. Shared by _call_openai_compatible and
    # _call_anthropic so the two paths never drift (a previous bug gave the
    # Anthropic 'code' tier only 8192 while OpenAI-compat got 16384, causing
    # disproportionate write_file truncation on Claude).
    _MAX_TOKENS_MAP = {
        "code": 16384,       # Code agent: complex reasoning + long file writes
        "writing": 16384,    # Writing agent: long report generation
        "researcher": 16384, # Researcher: paper analysis, web fetch results
        "idea": 16384,       # Idea agent: moderate output
        "think": 16384,      # Leader think: structured JSON decision
        "reflect": 16384,    # Leader reflect: deep cross-validation + root cause
    }
    _MAX_TOKENS_DEFAULT = 8192  # fallback when task_tier is None or unrecognized

    # Lock guarding _provider_health (R3: class-level dict mutated without sync)
    _provider_health_lock = __import__("threading").Lock()

    def __init__(self, model: str = "auto", provider: str = "glm_token_plan", max_steps: int = 3, tools=None):
        self.model = model
        self.provider = provider  # "glm_token_plan", "ali_token_plan", "openai", "qwen"
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
                "cooldown_until": 0,
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
        response_text, trace = self._call_llm(
            system=system_prompt,
            messages=messages,
            tools=reflect_tools,
            max_turns=effective_max_turns,
            task_tier=task,  # "think" or "reflect" → strong model
        )

        # Persist conversation for within-cycle coherence
        self._leader_history = messages + [{"role": "assistant", "content": response_text}]

        result = self._parse_leader_response(response_text, task=task)

        # Phase 2a (Reform v21): THINK parse-failure retry with feedback.
        # Phase 0 probe showed GLM usually outputs valid JSON (often in ```json
        # fences) but occasionally outputs prose. A single feedback retry
        # recovers most of these. REFLECT failures are handled by the fact
        # spine (Phase 2b), so we only retry THINK here.
        if (
            task == "think"
            and result.get("action") == "wait"
            and "Unparseable" in result.get("reason", "")
        ):
            retry_messages = list(self._leader_history) + [{
                "role": "user",
                "content": (
                    "Your previous response could not be parsed as a decision JSON. "
                    "Please output your decision as a ```json code block containing "
                    "a JSON object with at least these fields: "
                    '"action", "task", "hypothesis", "success_criteria". '
                    "Do not write prose before or after the JSON block."
                ),
            }]
            try:
                retry_response, retry_trace = self._call_llm(
                    system=system_prompt,
                    messages=retry_messages,
                    tools=None,
                    max_turns=10,
                    task_tier=task,
                )
                retry_result = self._parse_leader_response(retry_response, task=task)
                # Only accept retry if it actually parsed (not another default wait)
                if "Unparseable" not in retry_result.get("reason", ""):
                    logger.info(
                        f"THINK retry succeeded after parse failure. "
                        f"action={retry_result.get('action')}"
                    )
                    self._leader_history = retry_messages + [
                        {"role": "assistant", "content": retry_response}
                    ]
                    if retry_trace is not None and retry_trace.calls:
                        retry_result["leader_trace"] = retry_trace.to_dict()
                    return retry_result
                else:
                    logger.warning("THINK retry also failed to parse. Defaulting to wait.")
            except Exception as e:
                logger.warning(f"THINK retry call failed: {e}")

        # F23 fix: previously the REFLECT trace was discarded (_trace). During
        # REFLECT the Leader uses read_file/list_files for cross-validation,
        # so the trace records what it actually inspected — valuable both for
        # anti-deception checks and for downstream verification. Attach it.
        if trace is not None and trace.calls:
            result["leader_trace"] = trace.to_dict()
        return result

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
        # F24 fix: `or` swallows a legitimate override of 0. Use explicit None
        # check so 0 is honored (though 0 is unusual, it shouldn't silently
        # become the config default).
        effective_max_turns = (
            max_turns_override if max_turns_override is not None
            else config["max_turns"]
        )

        logger.info(f"Dispatching {agent_type} agent: {task[:100]}...")

        response_text, trace = self._call_llm(
            system=system_prompt,
            messages=[{"role": "user", "content": task}],
            tools=tools,
            max_turns=effective_max_turns,
            task_tier=agent_type,  # "idea"/"researcher" → strong, "code"/"writing" → fast
        )

        result = self._parse_worker_response(response_text, agent_type, trace,
                                             task=task)

        if result.get("convergence_failed"):
            logger.warning(
                "Code dispatch failed to converge: task mentioned experiment/train "
                "but launch_experiment was never called."
            )

        logger.info(f"Worker {agent_type} completed: {str(result)[:200]}")
        return result

    def reset_leader_history(self):
        """Clear leader conversation history between cycles."""
        self._leader_history = []

    def _call_llm(self, system: str, messages: list, tools: list = None, max_turns: int = 10, task_tier: str = None) -> tuple[str, ToolTrace]:
        """Call the LLM API with tool execution support.

        Two-level failover:
        - Level 1 (model-level): Try models in strong/fast_model_chain within provider
        - Level 2 (provider-level): Switch to next provider in TOKEN_PLAN_FAILOVER_ORDER

        Args:
            system: System prompt
            messages: Conversation messages
            tools: Tool definitions
            max_turns: Max tool-call turns
            task_tier: Task type for model selection

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
                if last_error is None:
                    last_error = RuntimeError(
                        f"API key not set: {provider_config['env_key']}. "
                        f"Run: export {provider_config['env_key']}=\"your-key\""
                    )
                continue

            # ── Level 1: Model-level failover chain ──
            model_chain = self._resolve_model_chain(provider_config, task_tier)
            quota_cooled_this_provider = False

            for model in model_chain:
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

                    # B2 sentinel check moved to the end of _call_llm. The old
                    # "API" text-sniff here (R4) was removed: it discarded
                    # legitimate responses whose JSON happened to mention "API"
                    # in an error field (e.g. a code agent writing
                    # {"error":"third-party API down"}), forcing false failover.

                    # Success — reset failure counter and return
                    self._record_provider_success(provider_key)
                    return text, trace

                except Exception as e:
                    last_error = e
                    # ── Quota-window 429: break the model chain immediately ──
                    # The whole API key shares one quota window, so every model
                    # in this provider's chain will 429 identically. Cooling
                    # this ONE provider until reset and trying the NEXT provider
                    # (which has its own quota) is correct. Do NOT burn the
                    # remaining 5 models, and do NOT re-add this provider.
                    if getattr(e, "status_code", None) == 429 and \
                            _classify_429(e)["type"] == "quota_exhausted":
                        info = _classify_429(e)
                        reset_time = info.get("reset_time")
                        self._record_provider_failure(
                            provider_key, str(e), cooldown_until=reset_time,
                        )
                        logger.warning(
                            f"Provider {provider_key} QUOTA EXHAUSTED"
                            f"{f' until {reset_time}' if reset_time else ''}. "
                            f"Cooling whole provider (not retrying its model chain). "
                            f"Trying next provider..."
                        )
                        # Mark so the post-loop soft-failure record (which
                        # clears cooldown_until) is skipped below.
                        quota_cooled_this_provider = True
                        break  # exit model loop → next provider in outer loop

                    # ── Other permanent errors: abort the whole matrix ──
                    if _is_permanent_error(e):
                        logger.error(
                            f"Provider {provider_key} model {model} returned a "
                            f"PERMANENT error ({type(e).__name__}): {e}. "
                            f"Aborting failover chain — fix the config/key."
                        )
                        self._record_provider_failure(provider_key, str(e))
                        raise
                    logger.warning(
                        f"Provider {provider_key} model {model} failed "
                        f"(transient, {type(e).__name__}): {e}. "
                        f"Trying next model in chain..."
                    )
                    continue

            # All models in this provider's chain failed.
            # If the chain was broken by a quota 429, the cooldown was already
            # recorded with its reset deadline — skip this soft-failure record
            # (it would clear cooldown_until, defeating the quota cooldown).
            if not quota_cooled_this_provider:
                self._record_provider_failure(provider_key, str(last_error))
            logger.warning(
                f"All models failed for provider {provider_key}. "
                f"Trying next provider..."
            )

        # ── All token_plan providers failed — try legacy providers ──
        if missing_keys and last_error and "not set" in str(last_error):
            logger.error(
                f"All token_plan providers skipped — missing API keys: {missing_keys}. "
                f"Set at least one: export {missing_keys[0]}=\"your-key\""
            )
        else:
            logger.error(f"All token_plan providers failed. Last error: {last_error}")

        if self.provider == "openai":
            # OPENAI_API_KEY resolved here — _call_openai_compatible now raises
            # on missing api_key (B2 rework), so we must supply a real key.
            _openai_key = os.environ.get("OPENAI_API_KEY")
            if not _openai_key:
                raise RuntimeError(
                    "All token_plan providers failed and OPENAI_API_KEY is not set. "
                    "Set at least one provider key."
                )
            text = self._call_openai_compatible(
                system=system, messages=messages, tools=tools,
                max_turns=max_turns, trace=trace,
                base_url=None, api_key=_openai_key, provider_label="openai",
                model=self.model,
                task_tier=task_tier,
            )
        else:
            raise RuntimeError(
                f"All providers failed for provider='{self.provider}'. "
                f"Last error: {last_error}. "
                f"Set a valid API key or configure a failover provider."
            )

        # B2 rework: the old sentinel that sniffed for {"error"...} in the
        # returned text is gone. The root cause — _call_openai_compatible
        # returning an error JSON string instead of raising — is fixed at
        # source (api_key check now raises). All provider paths now either
        # return genuine LLM output or raise; no downstream guessing needed.
        return text, trace

    def _resolve_model_chain(self, provider_config: dict, task_tier: str = None) -> list[str]:
        """Resolve the model failover chain for a specific provider and task tier.

        Returns an ordered list of models to try. The first is the primary model,
        subsequent entries are fallbacks within the same provider.

        Falls back to [strong_model] or [fast_model] if no chain is defined.
        """
        # Check if user explicitly chose a specific model (not auto/default)
        if self.model not in ("default", "auto"):
            if self.model in provider_config.get("models", []):
                return [self.model]  # User's choice only, no chain

        # Tiered chain selection
        if task_tier in STRONG_MODEL_TASKS:
            chain = provider_config.get("strong_model_chain")
            if chain:
                return list(chain)
            return [provider_config["strong_model"]]
        else:
            chain = provider_config.get("fast_model_chain")
            if chain:
                return list(chain)
            return [provider_config["fast_model"]]

    def _build_provider_queue(self) -> list[tuple[str, dict]]:
        """Build ordered list of (provider_key, config) to try.

        Primary provider first, then failover candidates. Skips providers that
        are in cooldown. A provider is in cooldown if EITHER:
          - it has a ``cooldown_until`` epoch set (quota-window exhaustion) and
            ``time.time() < cooldown_until``, OR
          - it has 3+ consecutive failures within the last 5 minutes.

        Quota-cooled providers are NEVER re-added as a "last resort" — re-adding
        them just burns doomed calls against an exhausted quota window. Only
        providers cooled by the softer 3-strike rule can be re-added if the
        queue would otherwise be empty.
        """
        if not self._token_plan_config:
            return []

        def _in_cooldown(key: str) -> tuple[bool, bool]:
            """Return (in_cooldown, is_quota_cooldown)."""
            health = AgentDispatcher._provider_health.get(key, {})
            # Quota-window cooldown: absolute deadline, never bypassed.
            cooldown_until = health.get("cooldown_until", 0)
            if cooldown_until and time.time() < cooldown_until:
                return True, True
            # Soft 3-strike cooldown: 3+ consecutive failures within 5 min.
            if health.get("consecutive_failures", 0) >= 3:
                last_fail = health.get("last_failure_time", 0)
                if time.time() - last_fail < 300:
                    return True, False
            return False, False

        queue = []
        primary = self.provider
        primary_skipped = False
        # Try primary first, but respect cooldown. A quota-exhausted primary
        # must not block the queue every cycle.
        if primary in TOKEN_PLAN_PROVIDERS:
            cooled, is_quota = _in_cooldown(primary)
            if cooled:
                logger.debug(
                    f"Primary {primary} in cooldown"
                    f"{' (quota-exhausted)' if is_quota else ''}; trying failovers first."
                )
                primary_skipped = True
            else:
                queue.append((primary, TOKEN_PLAN_PROVIDERS[primary]))

        # Failover providers (also respect cooldown)
        for key in TOKEN_PLAN_FAILOVER_ORDER:
            if key != primary and key in TOKEN_PLAN_PROVIDERS:
                cooled, is_quota = _in_cooldown(key)
                if cooled:
                    logger.debug(f"Skipping {key}: in cooldown"
                                 f"{' (quota)' if is_quota else ' (3+ failures)'}")
                    continue
                queue.append((key, TOKEN_PLAN_PROVIDERS[key]))

        # Last resort: if the queue is empty, re-add the primary ONLY IF it is
        # not quota-cooled. A quota-exhausted provider must never be retried
        # before its window resets — doing so just burns more doomed calls.
        if not queue and primary in TOKEN_PLAN_PROVIDERS:
            _, is_quota = _in_cooldown(primary)
            if is_quota:
                logger.warning(
                    f"All providers unavailable and primary {primary} is "
                    f"quota-exhausted — refusing to retry it. Dispatch will fail "
                    f"until a quota window resets."
                )
            else:
                logger.debug(f"All providers in soft-cooldown; retrying primary {primary}.")
                queue.append((primary, TOKEN_PLAN_PROVIDERS[primary]))

        return queue

    def _record_provider_success(self, provider_key: str):
        """Record a successful API call, resetting failure counter.

        B4 fix: lazily initialize the health dict so the success counter and
        failure-reset actually take effect for providers that weren't pre-seeded
        in __init__ (previously this no-op'd silently when health was missing).
        A success also clears any quota-window cooldown (the window evidently
        reset, or a different key is in use).
        """
        with AgentDispatcher._provider_health_lock:
            health = AgentDispatcher._provider_health.get(provider_key)
            if health is None:
                health = {
                    "consecutive_failures": 0, "last_failure_time": 0,
                    "total_calls": 0, "total_failures": 0,
                    "cooldown_until": 0,
                }
                AgentDispatcher._provider_health[provider_key] = health
            health["consecutive_failures"] = 0
            health["cooldown_until"] = 0
            health["total_calls"] += 1

    def _record_provider_failure(self, provider_key: str, error: str,
                                 cooldown_until=None):
        """Record a failed API call, incrementing failure counter.

        R3 fix: all mutations of the class-level _provider_health dict are now
        guarded by _provider_health_lock to prevent lost-update races when
        multiple dispatchers run concurrently.

        ``cooldown_until``: if given (a datetime), sets an absolute epoch
        deadline after which the provider may be retried. Used for
        quota-window exhaustion where the soft 3-strike/5-min rule is far too
        short (GLM windows are ~5 hours). If parsing the reset time failed, a
        conservative 1-hour fallback is applied.
        """
        with AgentDispatcher._provider_health_lock:
            if provider_key not in AgentDispatcher._provider_health:
                AgentDispatcher._provider_health[provider_key] = {
                    "consecutive_failures": 0, "last_failure_time": 0,
                    "total_calls": 0, "total_failures": 0,
                    "cooldown_until": 0,
                }
            health = AgentDispatcher._provider_health[provider_key]
            health["consecutive_failures"] += 1
            health["last_failure_time"] = time.time()
            health["total_calls"] += 1
            health["total_failures"] += 1
            if cooldown_until is not None:
                # Convert datetime → epoch. Fallback: 1 hour from now if the
                # parsed reset time is in the past or unparseable.
                try:
                    deadline = cooldown_until.timestamp()
                    if deadline <= time.time():
                        deadline = time.time() + 3600  # conservative 1h
                except (OSError, ValueError):
                    deadline = time.time() + 3600
                health["cooldown_until"] = deadline
            else:
                # A normal (non-quota) failure does not set an absolute deadline;
                # the soft 3-strike rule governs. Keep cooldown_until cleared so
                # a prior quota window doesn't linger after it expires.
                health["cooldown_until"] = 0

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
        effective_max_tokens = self._MAX_TOKENS_MAP.get(task_tier, self._MAX_TOKENS_DEFAULT)

        logger.info(
            f"Calling {provider_label} API: model={effective_model}, "
            f"messages={len(messages)}, tools={bool(tools)}"
        )
        try:
            import openai

            if not api_key:
                # Raise instead of returning an error JSON string. The old
                # behavior (return {"error": ...}) was a contract violation:
                # callers treated the returned string as a valid LLM response,
                # and _parse_leader_response would turn it into a "run an
                # experiment" decision with the error message as the task.
                # Raising lets _call_llm's failover handle it properly.
                raise RuntimeError(f"API key not configured for {provider_label}")

            # ── Detect GLM provider ──
            # Use zai.ZhipuAiClient for GLM (supports thinking, tool_stream)
            # Use openai.OpenAI for all other providers
            is_glm = "bigmodel.cn" in (base_url or "")

            if is_glm:
                from zai import ZhipuAiClient
                # CRITICAL: ZhipuAiClient() with no base_url defaults to the
                # standard PAAS endpoint (.../api/paas/v4). Coding Plan keys
                # MUST hit .../api/coding/paas/v4 or billing will silently go
                # to the wrong quota (verified: client.base_url reflects the
                # passed value, and a wrong endpoint returns 404). Guard so a
                # future config change can't silently reroute billing.
                if not base_url or "/coding/" not in base_url:
                    logger.error(
                        f"GLM provider selected but base_url is not a Coding "
                        f"Plan endpoint (got {base_url!r}). Coding Plan keys "
                        f"require '/api/coding/paas/v4'. Aborting to avoid "
                        f"billing the wrong quota."
                    )
                    return json.dumps({
                        "error": (
                            "GLM Coding Plan endpoint misconfigured: base_url must "
                            "contain '/coding/'. Check TOKEN_PLAN_PROVIDERS."
                        )
                    })
                # CRITICAL: pass an explicit timeout. Without it, ZhipuAiClient
                # (httpx under the hood) has NO read timeout — if the server
                # half-closes the socket (CLOSE-WAIT), the client polls forever
                # and the whole agent hangs indefinitely (verified: process
                # stuck in do_poll for 4+ min, no log, no crash). 120s matches
                # the openai.OpenAI path below; thinking-mode calls can be slow
                # so keep it generous, but it MUST be bounded.
                client = ZhipuAiClient(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=120.0,
                    max_retries=1,
                )
            else:
                kwargs = {
                    "timeout": 120.0,
                    "max_retries": 1,
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

            # ── GLM streaming + thinking support ──
            # zai.ZhipuAiClient supports: thinking={"type":"enabled"/"disabled"},
            # stream=True, tool_stream=True.
            # GLM-5.2/5.1/5 have built-in thinking by default (reasoning_content auto-returned).
            # For fast/cheap tasks (writing), disable thinking to save tokens.

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
                    # Build API call kwargs
                    create_kwargs = dict(
                        model=effective_model,
                        max_tokens=effective_max_tokens,
                        messages=api_messages,
                        tools=list(available_tools.values()) if available_tools else None,
                        tool_choice="auto",
                    )
                    if is_glm:
                        create_kwargs["stream"] = True
                        create_kwargs["tool_stream"] = True
                        # Disable thinking for fast-tier tasks (writing) to save tokens
                        if task_tier in ("writing",):
                            create_kwargs["thinking"] = {"type": "disabled"}
                            enable_thinking = False
                        else:
                            enable_thinking = True

                    if is_glm:
                        # ── GLM Streaming Mode ──
                        # Wall-clock protection: streaming reads are NOT covered
                        # by socket timeout. If the server stops sending chunks,
                        # the for-loop blocks forever. SIGALRM breaks it so
                        # failover can kick in. 120s = generous for thinking mode
                        # (normal calls take 10-30s), but bounded.
                        _LLM_WALL_CLOCK = 120
                        signal.signal(signal.SIGALRM, _llm_timeout_handler)
                        signal.alarm(_LLM_WALL_CLOCK)
                        response_stream = client.chat.completions.create(**create_kwargs)
                        reasoning_parts = []
                        content_parts = []
                        final_tool_calls = {}  # idx -> accumulated tool_call
                        finish_reason = None

                        try:
                            for chunk in response_stream:
                                if not chunk.choices:
                                    continue
                                delta = chunk.choices[0].delta
                                # Accumulate reasoning
                                rc = getattr(delta, 'reasoning_content', None)
                                if rc:
                                    reasoning_parts.append(rc)
                                # Accumulate content
                                dc = getattr(delta, 'content', None)
                                if dc:
                                    content_parts.append(dc)
                                # Accumulate tool calls
                                # NOTE: stream deltas are incremental. The first
                                # fragment carries id + function.name + start of
                                # arguments; continuation fragments carry only
                                # index + function.arguments (and function may be
                                # None on some fragments). Guard accordingly.
                                tcs = getattr(delta, 'tool_calls', None)
                                if tcs:
                                    for tc in tcs:
                                        idx = getattr(tc, 'index', None)
                                        if idx is None:
                                            continue
                                        fn = getattr(tc, 'function', None)
                                        if idx not in final_tool_calls:
                                            # First fragment for this call
                                            tc_id = getattr(tc, 'id', None) or f"synthetic_tc_{idx}_{turn}"
                                            fn_name = getattr(fn, 'name', None) if fn else None
                                            fn_args = (getattr(fn, 'arguments', None) or "") if fn else ""
                                            final_tool_calls[idx] = tc
                                            tc.id = tc_id
                                            if fn is not None:
                                                if not getattr(fn, 'name', None):
                                                    fn.name = fn_name
                                                fn.arguments = fn_args
                                        else:
                                            # Continuation fragment: append args only
                                            if fn is not None and getattr(fn, 'arguments', None):
                                                base_fn = getattr(final_tool_calls[idx], 'function', None)
                                                if base_fn is not None:
                                                    base_fn.arguments = (getattr(base_fn, 'arguments', None) or "") + fn.arguments

                                # Capture finish_reason from last chunk
                                fr = getattr(chunk.choices[0], "finish_reason", None)
                                if fr:
                                    finish_reason = fr
                            signal.alarm(0)  # cancel wall-clock timer — stream completed
                        except _LLMCallTimeout:
                            signal.alarm(0)
                            partial_content = "".join(content_parts)
                            logger.warning(
                                f"GLM stream hung >{_LLM_WALL_CLOCK}s (wall-clock timeout). "
                                f"Partial: {partial_content[:150]!r}... Triggering failover."
                            )
                            raise  # Re-raise for failover
                        except Exception as stream_err:
                            signal.alarm(0)
                            partial_content = "".join(content_parts)
                            logger.warning(
                                f"GLM stream interrupted: {stream_err}. "
                                f"Partial content: {partial_content[:200]}..."
                            )
                            raise  # Re-raise for failover

                        reasoning_content = "".join(reasoning_parts)
                        stream_content = "".join(content_parts)

                        # Log reasoning if present (or warn if thinking enabled but empty)
                        if reasoning_content:
                            logger.info(f"[{effective_model} thinking] {reasoning_content[:200]}...")
                        elif enable_thinking:
                            logger.debug(f"[{effective_model}] thinking enabled but no reasoning_content returned")

                        # Warn on completely empty response
                        if not stream_content and not final_tool_calls:
                            logger.warning(f"GLM stream returned empty: no content, no tool_calls")

                        # Build a synthetic response-like object for downstream logic
                        syn_msg = _SyntheticMessage()
                        syn_msg.content = stream_content or None
                        syn_msg.tool_calls = list(final_tool_calls.values()) if final_tool_calls else None

                        syn_choice = _SyntheticChoice()
                        # Use tool_calls presence as primary signal, not finish_reason
                        syn_choice.finish_reason = ("tool_calls" if final_tool_calls
                                                    else (finish_reason or "stop"))
                        syn_choice.message = syn_msg

                        # Adapt to non-streaming logic below
                        choice = syn_choice
                    else:
                        # ── Non-GLM: standard synchronous call ──
                        response = client.chat.completions.create(**create_kwargs)
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

                            # B7: For write_file, repaired args that LOST the
                            # content key (Strategy 2 truncation can drop the
                            # whole trailing field) must be rejected outright —
                            # otherwise write_file would silently write an empty
                            # file. The previous guard below only fired when
                            # "content" was present, so a missing key slipped
                            # through and wrote garbage.
                            if repaired and func_name == "write_file":
                                if not isinstance(func_args, dict) or "content" not in func_args or not func_args.get("content"):
                                    logger.warning(
                                        f"Rejecting write_file: repaired args lost the content "
                                        f"key (truncation during max_tokens). Path="
                                        f"{func_args.get('path', '?') if isinstance(func_args, dict) else '?'}."
                                    )
                                    pending_results.append((
                                        tool_call.id, func_name,
                                        json.dumps({
                                            "error": "File content was LOST to max_tokens truncation. "
                                                     "The file was NOT written. Please split into smaller chunks."
                                        })
                                    ))
                                    continue

                            # For write_file with repaired args, reject if content appears truncated
                            if repaired and func_name == "write_file" and isinstance(func_args, dict) and "content" in func_args:
                                file_content = func_args["content"]
                                if file_content and not file_content.rstrip().endswith(('\n', '}', ']', ')', '"""', "'''", '`', '.')):
                                    logger.warning(
                                        f"Rejecting write_file to {func_args.get('path','?')}: "
                                        f"content appears truncated (ends with: ...{file_content[-50:]})"
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
                                # ── Hard convergence gate (Phase 2) ──
                                # Once a code agent passes 60% of its turn budget,
                                # block exploration-only tools so the remaining
                                # turns are spent converging (write_file,
                                # launch_experiment, diagnose_error). This fixes
                                # the observed pathology where 59.7% of tool
                                # calls were read_file/list_files and 0% were
                                # launch_experiment — the agent explored until
                                # the budget ran out without ever launching.
                                if task_tier == "code" and \
                                        turn >= max_turns * 0.6 and \
                                        func_name in _CODE_EXPLORE_TOOLS:
                                    tool_result = json.dumps({
                                        "error": (
                                            f"Turn {turn+1}/{max_turns}: past 60% budget. "
                                            f"'{func_name}' is blocked — stop exploring. "
                                            f"You MUST now converge: use write_file to finalize "
                                            f"the script, then launch_experiment to run it. "
                                            f"Only write_file / launch_experiment / diagnose_error "
                                            f"are allowed past this point."
                                        )
                                    })
                                    pending_results.append(
                                        (tool_call.id, func_name, str(tool_result))
                                    )
                                    continue
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
                            "content": None,
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
                create_kwargs = dict(
                    model=effective_model,
                    max_tokens=effective_max_tokens,
                    messages=api_messages,
                )
                if is_glm:
                    create_kwargs["stream"] = True
                    if task_tier in ("writing",):
                        create_kwargs["thinking"] = {"type": "disabled"}

                if is_glm:
                    # ── GLM Streaming (no tools) ──
                    response_stream = client.chat.completions.create(**create_kwargs)
                    reasoning_parts = []
                    content_parts = []
                    try:
                        for chunk in response_stream:
                            if not chunk.choices:
                                continue
                            delta = chunk.choices[0].delta
                            rc = getattr(delta, 'reasoning_content', None)
                            if rc:
                                reasoning_parts.append(rc)
                            dc = getattr(delta, 'content', None)
                            if dc:
                                content_parts.append(dc)
                    except Exception as stream_err:
                        logger.warning(f"GLM stream interrupted (no-tools path): {stream_err}")
                        raise
                    reasoning_content = "".join(reasoning_parts)
                    stream_content = "".join(content_parts)
                    if reasoning_content:
                        logger.info(f"[{effective_model} thinking] {reasoning_content[:200]}...")
                    if not stream_content:
                        # Empty stream (no content, no error) — likely a server
                        # hiccup. Returning "" would be misread downstream as a
                        # valid empty leader decision. Raise so _call_llm can
                        # fail over to the next provider/model.
                        raise RuntimeError(
                            f"GLM {effective_model} returned an empty stream "
                            f"(no content). Triggering failover."
                        )
                    return stream_content
                else:
                    response = client.chat.completions.create(**create_kwargs)
                    return response.choices[0].message.content if response.choices else ""

        except ImportError:
            # R7 fix: previously returned a mock {"action":"wait"} string that
            # _parse_leader_response mistook for a legitimate "wait" decision,
            # making an unattended agent hang silently forever. Raise so the
            # caller sees a real error and can surface it.
            raise RuntimeError(
                "openai package not installed. Install with: pip install openai"
            )
        except Exception as e:
            logger.error(f"{provider_label} API call failed: {e}")
            # Re-raise so _call_llm can try failover
            raise

    # ─────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────

    @staticmethod
    def _repair_json_args(raw: str) -> dict:
        """Recover a dict from (possibly truncated) LLM tool arguments.

        LLM output can be cut off mid-string by max_tokens, producing invalid
        JSON like ``{"path": "src/model.py", "content": "def foo():\\n  r``.

        Strategy, in order of safety:
        1. ``raw_decode`` — parse a complete leading object, ignoring trailing
           junk. Handles "complete JSON + prose" and "multiple objects, take
           first" with zero heuristics.
        2. Close obvious truncation — append common closers and try json.loads.
        3. Truncate-back — scan for the last structural ``,``, drop everything
           after it, close the object. The scan is string-aware (tracks
           in-string + escapes) so a ``,`` *inside* a string value (e.g. code
           containing ``print("a", b)``) is never mistaken for a structural
           separator. This replaces the old ``rfind('",')`` which corrupted
           file contents by matching inside strings.
        4. Give up → empty dict (tool fails with a clear error).

        Returns {} on failure; callers (write_file guard) reject repaired args
        that lost expected keys.
        """
        if not raw:
            return {}

        raw = raw.strip()
        decoder = json.JSONDecoder()

        # Strategy 1: raw_decode handles "complete object + trailing text"
        # and "first of several objects" cleanly, no guessing.
        try:
            obj, _end = decoder.raw_decode(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Strategy 1.5: single-quote JSON (some LLMs emit {'k': 'v'})
        if "'" in raw and '"' not in raw:
            try:
                return json.loads(raw.replace("'", '"'))
            except json.JSONDecodeError:
                pass

        # Strategy 2: append common closers for simple truncation. Covers two
        # truncation shapes: (a) cut inside a string value — needs a closing
        # quote before the brace ('"' + closer); (b) cut right after a complete
        # value, just missing the object/array close (bare closer).
        for suffix in ('"}', '"}]', '"]}', '"}]}', '}', ']', ']}'):
            try:
                obj = json.loads(raw + suffix)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        # Strategy 3: truncate back to the last structural comma (string-aware).
        pos = AgentDispatcher._last_structural_comma_pos(raw)
        if pos > 0:
            candidate = raw[:pos] + '}'
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

        return {}

    @staticmethod
    def _last_structural_comma_pos(s: str) -> int:
        """Index of the last structural ``,`` in JSON-ish ``s`` (string-aware).

        Returns the position to truncate *after* (i.e. the ``,`` index), or -1.
        A structural comma is one at JSON nesting level 1 (directly inside the
        top object) and NOT inside a string value. String-awareness is the
        whole point: ``rfind('",')`` matched inside ``"print(a, b)"`` and
        truncated file contents.
        """
        in_string = False
        escape = False
        depth = 0
        last_comma = -1
        for i, ch in enumerate(s):
            if escape:
                escape = False
                continue
            if in_string:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == '{' or ch == '[':
                depth += 1
            elif ch == '}' or ch == ']':
                depth -= 1
            elif ch == ',' and depth == 1:
                last_comma = i
        return last_comma

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
        """Format context into a structured input for the Leader.

        Register-driven: iterates the ContextKey registry, calling each key's
        serializer. This is the SINGLE serialization path — adding a key to
        the registry (context_keys.py) automatically makes it appear in the
        prompt. Previously this was a 170-line hardcoded if-block chain that
        silently dropped 37 of 48 injected context keys.
        """
        from .context_keys import serialize_context

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

        # Inject project knowledge summary (auto-generated, not in registry)
        project_knowledge = self._generate_project_knowledge(context)
        if project_knowledge:
            parts.append(f"## Project Knowledge\n{project_knowledge}\n")

        # ── Register-driven serialization of all context keys ──
        # This replaces ~150 lines of hardcoded if-blocks. Every ContextKey
        # in the registry whose serializer returns a non-None string is
        # included. Previously-dropped keys (domain_knowledge,
        # architecture_plan_summary, training_curve_analysis, etc.) now
        # reach the Leader automatically.
        phase = "reflect" if task == "reflect" else "think"
        parts.append(serialize_context(context, phase))

        return "\n".join(parts)

    def _parse_leader_response(self, response: str, task: str = "think") -> dict:
        """Parse Leader's response into structured action.

        task selects the schema key used to accept a parsed JSON object:
          - "think"   requires an ``action`` key (decision schema)
          - "reflect" requires a ``milestone`` or ``decision`` key (reflection
            schema, which legitimately has NO ``action`` key — see leader.md)

        Reform v21 root-cause fix: previously this method always demanded an
        ``action`` key, but REFLECT's schema (milestone/decision/dead_end/
        active_problem/causal_link/lesson) contains no ``action`` key. So every
        valid REFLECT JSON was rejected as "Unparseable", REFLECT was reported as
        "100% failing", and the wrong root cause ("LLM writes prose") was logged.
        Black-box testing (Reform v21 angle-3 probe) proved GLM returns valid
        REFLECT JSON 3/3 — the parser was the bug, not the model.
        """
        parsed = self._extract_first_decision_json(response, task=task)
        if parsed is not None:
            return parsed


        # Parse failure must NEVER auto-trigger an experiment. A confused,
        # truncated, or empty leader response used as a task description would
        # send the code agent off to modify code on garbage instructions. Wait
        # is the safe default — the next cycle can retry with fresh context.
        logger.warning(
            f"Leader response unparseable (no decision JSON found). "
            f"Defaulting to wait. Response head: {response[:200]!r}"
        )
        return {
            "action": "wait",
            "reason": f"Unparseable leader response: {response[:200]}",
        }

    @staticmethod
    def _extract_first_decision_json(response: str, task: str = "think") -> Optional[dict]:
        """Find the first balanced {...} in `response` that parses to a dict
        matching the expected schema for `task`.

        B6 fix: the old walker counted braces without tracking string context,
        so a brace inside a JSON string value (e.g. ``"task": "apply {x:1}"``)
        made depth hit 0 at the wrong offset and the real JSON was never
        extracted. This version is string-aware: it tracks ``in_string`` and
        handles ``\\`` escapes, and it also strips ``` ```json ``` fences first.

        Schema selection (Reform v21 root-cause fix): a parsed JSON object is
        accepted only if it carries a key that identifies its schema.
          - task="think"   → must contain ``action`` (decision schema)
          - task="reflect" → must contain ``milestone`` or ``decision`` (the
            reflection schema has no ``action`` key; demanding one rejected
            every valid REFLECT output — see _parse_leader_response docstring).
        Returns None if no matching JSON is found.
        """
        if not response:
            return None

        # Which key(s) identify the expected schema for this task.
        # A parsed dict must contain at least one of these to be accepted.
        if task == "reflect":
            schema_keys = ("milestone", "decision")
        else:
            schema_keys = ("action",)

        # Strip markdown code fences so fenced JSON is parsed directly.
        # Keep this conservative: only strip a leading fence and a trailing fence.
        stripped = response.strip()
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            if first_nl != -1:
                inner = stripped[first_nl + 1:]
                if inner.rstrip().endswith("```"):
                    stripped = inner.rstrip()[:-3]

        depth = 0
        start = None
        in_string = False
        escape = False
        for i, ch in enumerate(stripped):
            if escape:
                escape = False
                continue
            if in_string:
                if ch == '\\':
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            # Not inside a string
            if ch == '"':
                in_string = True
            elif ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidate = stripped[start:i + 1]
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict) and any(
                                k in parsed for k in schema_keys
                            ):
                                return parsed
                            # Balanced & valid JSON but wrong schema — keep scanning.
                        except json.JSONDecodeError:
                            pass
                        start = None
        return None

    def _parse_worker_response(self, response: str, agent_type: str, trace: ToolTrace = None, task: str = "") -> dict:
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
                # launch_experiment was NEVER called. Check whether the agent
                # tried to launch training via run_shell instead — this is a
                # contract violation: training MUST go through launch_experiment
                # (which writes the structured manifest and returns a real PID).
                # The old code accepted shell-launched training via a brittle
                # 4-branch regex that silently missed renamed scripts, `python
                # -m`, torchrun, etc. — causing experiment_launched=False and
                # no-progress miscounting. Now we DETECT the violation and flag
                # it explicitly rather than guessing from free text.
                shell_facts = trace.extract_shell_facts()
                training_via_shell = False
                for sf in shell_facts:
                    cmd = sf.get("command", "")
                    # Broad detection: does this look like it's launching a
                    # training process (python + train, torchrun, accelerate)?
                    # We want to CATCH the violation, not miss it.
                    looks_like_training = bool(re.search(
                        r'\bpython\b[^|;&]*\w*train\w*\.py\b'
                        r'|\btorchrun\b'
                        r'|\baccelerate\b[^|;&]*launch'
                        r'|\bpython\s+-m\s+\S*train\w*\b'
                        r'|\bnohup\b[^|;&]*\bpython\b',
                        cmd, re.IGNORECASE,
                    ))
                    if looks_like_training:
                        training_via_shell = True
                        result["experiment_launched"] = False
                        result["launch_error"] = (
                            "Training was launched via run_shell instead of "
                            "launch_experiment. This is forbidden — launch_experiment "
                            "writes the structured manifest and returns a trackable "
                            "PID. Re-launch using launch_experiment(command=..., "
                            f"log_file=...). Detected command: {cmd[:120]}"
                        )
                        logger.warning(
                            f"FORBIDDEN launch path: code agent ran training via "
                            f"run_shell instead of launch_experiment: {cmd[:100]}"
                        )
                        break

                if not training_via_shell:
                    # No training attempt at all. Check if the LLM falsely
                    # claimed to have launched (anti-deception).
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

        # Phase 2 convergence flag: a code dispatch asked to run an experiment
        # that never launched (and wasn't blocked by a tool error or flagged
        # as deception) failed to converge. This gives the loop (Phase 4) a
        # clean signal to force a re-dispatch. Only flagged when the task
        # explicitly mentions experiment/train/launch so pure-analysis code
        # dispatches aren't mis-flagged.
        if agent_type == "code" \
                and not result.get("experiment_launched") \
                and not result.get("launch_error") \
                and not result.get("deception_detected") \
                and re.search(r'\b(experiment|train|launch)',
                              task or "", re.IGNORECASE):
            result["convergence_failed"] = True

        return result
