"""Agent modules for The Analyst platform."""

from src.agents.arabic_nlp import ArabicNLPAgent
from src.agents.base import AgentContext, AgentOption, AgentResult, BaseAgent
from src.agents.insights import InsightsAgent
from src.agents.modeling import ModelingAgent
from src.agents.report import ReportAgent
from src.agents.retrieval import RetrievalAgent
from src.agents.statistical import StatisticalAgent
from src.agents.transform import TransformAgent
from src.agents.visualization import VisualizationAgent

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentResult",
    "AgentContext",
    "AgentOption",
    # Data layer agents
    "RetrievalAgent",
    "TransformAgent",
    # Analysis layer agents
    "StatisticalAgent",
    "InsightsAgent",
    "ArabicNLPAgent",
    "ModelingAgent",
    # Output layer agents
    "VisualizationAgent",
    "ReportAgent",
]
