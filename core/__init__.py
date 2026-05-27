"""AutoResearcher Core - Autonomous ML Experiment Agent Framework."""

from .loop import ResearchLoop
from .memory import MemoryManager
from .monitor import ExperimentMonitor
from .tools import ToolRegistry
from .verifier import ExperimentVerifier, VerifyReport, VerifyCheck
from .agents import AgentDispatcher, ToolTrace, ToolCallRecord
from .constraint_engine import (
    StrategyConstraintEngine,
    ContextPruner,
)
from .simulation_sandbox import SimulationSandbox

__version__ = "0.4.0"
__all__ = [
    "ResearchLoop", "MemoryManager", "ExperimentMonitor",
    "AgentDispatcher", "ToolRegistry",
    "ExperimentVerifier", "VerifyReport", "VerifyCheck",
    "ToolTrace", "ToolCallRecord",
    "StrategyConstraintEngine", "ContextPruner",
    "SimulationSandbox",
]
