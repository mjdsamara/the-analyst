"""
Intent routing for The Analyst orchestrator.

Parses user requests and determines which agents and workflows to invoke.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentType(str, Enum):
    """Types of user intents."""

    # Data operations
    LOAD_DATA = "load_data"
    EXPLORE_DATA = "explore_data"
    TRANSFORM_DATA = "transform_data"

    # Analysis operations
    STATISTICAL_ANALYSIS = "statistical_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CORRELATION_ANALYSIS = "correlation_analysis"
    TREND_ANALYSIS = "trend_analysis"

    # Predictive operations
    FORECAST = "forecast"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    # Output operations
    VISUALIZE = "visualize"
    GENERATE_REPORT = "generate_report"
    EXPORT = "export"

    # Meta operations
    HELP = "help"
    STATUS = "status"
    CLARIFICATION_NEEDED = "clarification_needed"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """Parsed user intent with extracted parameters."""

    type: IntentType
    confidence: float
    parameters: dict[str, Any] = field(default_factory=dict)
    agents_required: list[str] = field(default_factory=list)
    high_stakes: bool = False
    high_stakes_reasons: list[str] = field(default_factory=list)
    clarifications_needed: list[str] = field(default_factory=list)

    @property
    def requires_approval(self) -> bool:
        """Check if this intent requires user approval."""
        return self.high_stakes or self.type in (
            IntentType.FORECAST,
            IntentType.CLASSIFICATION,
            IntentType.REGRESSION,
            IntentType.GENERATE_REPORT,
            IntentType.EXPORT,
        )


# Pattern matching rules for intent detection
INTENT_PATTERNS: dict[IntentType, list[str]] = {
    IntentType.LOAD_DATA: [
        r"load\s+(the\s+)?data",
        r"import\s+(the\s+)?data",
        r"read\s+(the\s+)?file",
        r"open\s+(the\s+)?file",
        r"ingest",
        r"from\s+file",
        r"\.csv|\.xlsx?|\.json|\.parquet",
    ],
    IntentType.EXPLORE_DATA: [
        r"explore\s+(the\s+)?data",
        r"show\s+(me\s+)?(the\s+)?data",
        r"what\s+(does|is)\s+the\s+data",
        r"describe\s+(the\s+)?data",
        r"profile\s+(the\s+)?data",
        r"summary\s+of",
        r"overview\s+of",
    ],
    IntentType.TRANSFORM_DATA: [
        r"clean\s+(the\s+)?data",
        r"transform\s+(the\s+)?data",
        r"prepare\s+(the\s+)?data",
        r"filter\s+(the\s+)?data",
        r"remove\s+(duplicates|missing|null)",
        r"fill\s+(missing|null)",
        r"convert\s+type",
    ],
    IntentType.STATISTICAL_ANALYSIS: [
        r"analyze\s+(the\s+)?data",
        r"statistical\s+analysis",
        r"descriptive\s+statistics",
        r"distribution",
        r"mean|median|standard\s+deviation",
        r"percentile",
        r"hypothesis\s+test",
    ],
    IntentType.SENTIMENT_ANALYSIS: [
        r"sentiment\s+analysis",
        r"sentiment\s+of",
        r"analyze\s+sentiment",
        r"positive\s+or\s+negative",
        r"opinion\s+analysis",
        r"arabic\s+text",
    ],
    IntentType.CORRELATION_ANALYSIS: [
        r"correlation",
        r"relationship\s+between",
        r"correlated",
        r"associated\s+with",
    ],
    IntentType.TREND_ANALYSIS: [
        r"trend",
        r"over\s+time",
        r"time\s+series",
        r"pattern\s+over",
        r"growth",
        r"decline",
    ],
    IntentType.FORECAST: [
        r"forecast",
        r"predict\s+(future|next)",
        r"projection",
        r"project\s+(next|future|\d+)",
        r"what\s+will",
        r"expected\s+value",
    ],
    IntentType.CLASSIFICATION: [
        r"classify",
        r"classification",
        r"categorize",
        r"which\s+category",
    ],
    IntentType.REGRESSION: [
        r"regression",
        r"predict\s+(a\s+)?value",
        r"estimate",
    ],
    IntentType.VISUALIZE: [
        r"visualize",
        r"chart",
        r"graph",
        r"plot",
        r"show\s+(me\s+)?(a\s+)?chart",
        r"create\s+(a\s+)?visual",
    ],
    IntentType.GENERATE_REPORT: [
        r"generate\s+(a\s+)?(\w+\s+)?report",
        r"create\s+(a\s+)?(\w+\s+)?report",
        r"write\s+(a\s+)?(\w+\s+)?report",
        r"summary\s+report",
        r"executive\s+summary",
        r"(powerpoint|pptx)\s+presentation",
        r"create\s+(a\s+)?presentation",
    ],
    IntentType.EXPORT: [
        r"export",
        r"save\s+(as|to)",
        r"download",
        r"send\s+to",
        r"share",
    ],
    IntentType.HELP: [
        r"help",
        r"how\s+do\s+i",
        r"what\s+can\s+you\s+do",
        r"capabilities",
    ],
    IntentType.STATUS: [
        r"status",
        r"progress",
        r"where\s+are\s+we",
    ],
}

# Agent mapping for each intent
INTENT_AGENTS: dict[IntentType, list[str]] = {
    IntentType.LOAD_DATA: ["retrieval"],
    IntentType.EXPLORE_DATA: ["retrieval"],
    IntentType.TRANSFORM_DATA: ["transform"],
    IntentType.STATISTICAL_ANALYSIS: ["statistical"],
    IntentType.SENTIMENT_ANALYSIS: ["arabic_nlp"],
    IntentType.CORRELATION_ANALYSIS: ["statistical"],
    IntentType.TREND_ANALYSIS: ["statistical"],
    IntentType.FORECAST: ["modeling"],
    IntentType.CLASSIFICATION: ["modeling"],
    IntentType.REGRESSION: ["modeling"],
    IntentType.VISUALIZE: ["visualization"],
    IntentType.GENERATE_REPORT: ["insights", "visualization", "report"],
    IntentType.EXPORT: ["report"],
    IntentType.HELP: [],
    IntentType.STATUS: [],
    IntentType.CLARIFICATION_NEEDED: [],
    IntentType.UNKNOWN: [],
}

# High-stakes keywords - these require confirmation before proceeding
# Keywords are matched with word boundaries to avoid false positives
# e.g., "model" won't match in "data model" but will match "build a model"
HIGH_STAKES_KEYWORDS: list[str] = [
    "delete",
    "remove",
    "drop",
    "send",
    "share",
    "export to",
    "production",
    "stakeholder",
    "executive",
    "forecast",
    "predict",
]

# Context-sensitive keywords that need additional checks
# "model" is high-stakes only when used as a verb (build/train/create a model)
HIGH_STAKES_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "model": ["build", "train", "create", "deploy", "run", "fit", "predictive"],
}


class IntentRouter:
    """Routes user requests to appropriate agents based on intent."""

    def __init__(self) -> None:
        """Initialize the router."""
        self._compiled_patterns: dict[IntentType, list[re.Pattern[str]]] = {}
        for intent_type, patterns in INTENT_PATTERNS.items():
            self._compiled_patterns[intent_type] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def parse_intent(self, text: str) -> Intent:
        """
        Parse user text to determine intent.

        Args:
            text: User's input text

        Returns:
            Parsed Intent object
        """
        # Check for high-stakes keywords FIRST (applies to all intents including UNKNOWN)
        # Use word boundary matching to avoid false positives
        high_stakes_reasons = []
        text_lower = text.lower()
        for keyword in HIGH_STAKES_KEYWORDS:
            # Use word boundary regex for accurate matching
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, text_lower):
                high_stakes_reasons.append(f"Contains high-stakes keyword: '{keyword}'")

        # Check context-sensitive keywords (e.g., "model" only when used as action)
        for keyword, context_words in HIGH_STAKES_CONTEXT_KEYWORDS.items():
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, text_lower):
                # Check if any context word is present
                for context in context_words:
                    if re.search(rf"\b{re.escape(context)}\b", text_lower):
                        high_stakes_reasons.append(
                            f"Contains high-stakes keyword: '{keyword}' with action context"
                        )
                        break

        # Score each intent type
        scores: dict[IntentType, float] = {}
        for intent_type, patterns in self._compiled_patterns.items():
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                scores[intent_type] = matches / len(patterns)

        # Determine primary intent
        if not scores:
            # Extract parameters even for UNKNOWN intents
            parameters = self._extract_parameters(text, IntentType.UNKNOWN)
            return Intent(
                type=IntentType.UNKNOWN,
                confidence=0.0,
                parameters=parameters,
                high_stakes=len(high_stakes_reasons) > 0,
                high_stakes_reasons=high_stakes_reasons,
                clarifications_needed=[
                    "I couldn't understand your request. Could you please rephrase?"
                ],
            )

        best_intent = max(scores, key=lambda k: scores[k])
        confidence = scores[best_intent]

        # Extract parameters
        parameters = self._extract_parameters(text, best_intent)

        # Get required agents
        agents = INTENT_AGENTS.get(best_intent, [])

        # Check if clarification is needed
        clarifications = self._check_clarifications(text, best_intent, parameters)

        return Intent(
            type=best_intent,
            confidence=confidence,
            parameters=parameters,
            agents_required=agents,
            high_stakes=len(high_stakes_reasons) > 0,
            high_stakes_reasons=high_stakes_reasons,
            clarifications_needed=clarifications,
        )

    def _extract_parameters(self, text: str, intent_type: IntentType) -> dict[str, Any]:
        """Extract relevant parameters from text based on intent."""
        params: dict[str, Any] = {}

        # Extract file paths - use specific filename characters to avoid greedy matching
        file_pattern = r'["\']?([\w\-./\\]+\.(csv|xlsx?|json|parquet))["\']?'
        file_matches = re.findall(file_pattern, text, re.IGNORECASE)
        if file_matches:
            params["files"] = [m[0] for m in file_matches]

        # Extract date ranges
        date_pattern = r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})"
        date_match = re.search(date_pattern, text)
        if date_match:
            params["date_range"] = {"start": date_match.group(1), "end": date_match.group(2)}

        # Extract column names (in quotes or after "column")
        col_pattern = r'column\s+["\']?(\w+)["\']?|["\'](\w+)["\']'
        col_matches = re.findall(col_pattern, text, re.IGNORECASE)
        if col_matches:
            params["columns"] = [c[0] or c[1] for c in col_matches if c[0] or c[1]]

        # Extract numeric values
        num_pattern = r"\b(\d+(?:\.\d+)?)\b"
        num_matches = re.findall(num_pattern, text)
        if num_matches:
            params["numeric_values"] = [float(n) for n in num_matches]

        # Intent-specific extractions
        if intent_type == IntentType.FORECAST:
            # Extract forecast horizon
            horizon_pattern = r"next\s+(\d+)\s+(days?|weeks?|months?|years?)"
            horizon_match = re.search(horizon_pattern, text, re.IGNORECASE)
            if horizon_match:
                params["forecast_horizon"] = {
                    "value": int(horizon_match.group(1)),
                    "unit": horizon_match.group(2).lower().rstrip("s"),
                }

        if intent_type == IntentType.GENERATE_REPORT:
            # Extract output format
            format_pattern = r"(pdf|powerpoint|pptx|html|markdown|md)"
            format_match = re.search(format_pattern, text, re.IGNORECASE)
            if format_match:
                params["output_format"] = format_match.group(1).lower()

        return params

    def _check_clarifications(
        self, text: str, intent_type: IntentType, parameters: dict[str, Any]
    ) -> list[str]:
        """Check if clarifications are needed based on intent and parameters."""
        clarifications = []

        if intent_type == IntentType.LOAD_DATA and "files" not in parameters:
            clarifications.append("Which file would you like to load?")

        if intent_type == IntentType.FORECAST and "forecast_horizon" not in parameters:
            clarifications.append("How far ahead would you like to forecast?")

        if intent_type == IntentType.GENERATE_REPORT and "output_format" not in parameters:
            clarifications.append(
                "What format would you like for the report (PDF, PowerPoint, HTML, Markdown)?"
            )

        return clarifications

    def get_workflow_agents(self, intent: Intent) -> list[str]:
        """
        Get the ordered list of agents for a workflow based on intent.

        Args:
            intent: Parsed intent

        Returns:
            Ordered list of agent names
        """
        # Standard workflow order
        full_workflow = [
            "retrieval",
            "transform",
            "statistical",
            "arabic_nlp",
            "modeling",
            "insights",
            "visualization",
            "report",
        ]

        # Filter to only include required agents
        required = set(intent.agents_required)
        return [a for a in full_workflow if a in required]
