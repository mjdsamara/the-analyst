"""
Main Orchestrator for The Analyst platform.

The orchestrator is the primary entry point that:
- Parses user intents
- Presents analysis options
- Coordinates specialized agents
- Ensures human-in-the-loop approval
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam

from src.agents.arabic_nlp import ArabicNLPAgent
from src.agents.base import AgentContext, AgentOption
from src.agents.insights import InsightsAgent
from src.agents.modeling import ModelingAgent, ModelType, TaskType
from src.agents.report import ReportAgent
from src.agents.retrieval import RetrievalAgent
from src.agents.statistical import AnalysisType, StatisticalAgent
from src.agents.transform import TransformAgent
from src.agents.visualization import VisualizationAgent
from src.config import get_settings
from src.orchestrator.router import Intent, IntentRouter, IntentType
from src.orchestrator.state import (
    ConversationState,
    StateManager,
    WorkflowPhase,
    WorkflowState,
)
from src.prompts.orchestrator import ORCHESTRATOR_PROMPT
from src.utils.notifications import Notifier
from src.utils.obsidian import ObsidianVault

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Main orchestrator agent for The Analyst platform.

    Coordinates all sub-agents and manages the analysis workflow.
    """

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        settings = get_settings()
        self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.orchestrator_model
        self.router = IntentRouter()
        self.state_manager = StateManager()

        # Initialize states
        self.conversation = ConversationState()
        self.workflow: WorkflowState | None = None

        # Agent registry (lazy loaded)
        self._agents: dict[str, Any] = {}

        # Shared context for agents
        self._agent_context = AgentContext(session_id="orchestrator")

        # Initialize notification system
        self.notifier = Notifier()
        self.obsidian = ObsidianVault() if settings.obsidian_vault_path else None

        logger.info("Orchestrator initialized")

    def _get_agent(self, agent_name: str) -> Any:
        """
        Get or create an agent instance.

        Args:
            agent_name: Name of the agent to get

        Returns:
            Agent instance
        """
        if agent_name not in self._agents:
            if agent_name == "retrieval":
                self._agents[agent_name] = RetrievalAgent(context=self._agent_context)
            elif agent_name == "transform":
                self._agents[agent_name] = TransformAgent(context=self._agent_context)
            elif agent_name == "statistical":
                self._agents[agent_name] = StatisticalAgent(context=self._agent_context)
            elif agent_name == "insights":
                self._agents[agent_name] = InsightsAgent(context=self._agent_context)
            elif agent_name == "arabic_nlp":
                self._agents[agent_name] = ArabicNLPAgent(context=self._agent_context)
            elif agent_name == "modeling":
                self._agents[agent_name] = ModelingAgent(context=self._agent_context)
            elif agent_name == "visualization":
                self._agents[agent_name] = VisualizationAgent(context=self._agent_context)
            elif agent_name == "report":
                self._agents[agent_name] = ReportAgent(context=self._agent_context)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")

        return self._agents[agent_name]

    @property
    def system_prompt(self) -> str:
        """Get the orchestrator's system prompt."""
        return ORCHESTRATOR_PROMPT

    async def process_message(self, user_message: str) -> str:
        """
        Process a user message and return a response.

        This is the main entry point for user interaction.

        Args:
            user_message: The user's input message

        Returns:
            The orchestrator's response
        """
        # Add user message to conversation
        self.conversation.add_message("user", user_message)

        # Check for pending approval
        if self.conversation.pending_approval:
            return await self._handle_approval_response(user_message)

        # Parse intent
        intent = self.router.parse_intent(user_message)
        self.conversation.current_intent = intent.type.value

        logger.info(f"Parsed intent: {intent.type.value} (confidence: {intent.confidence:.2f})")

        # Handle different intent types
        if intent.type == IntentType.HELP:
            response = self._generate_help_response()
        elif intent.type == IntentType.STATUS:
            response = self._generate_status_response()
        elif intent.type == IntentType.UNKNOWN or intent.clarifications_needed:
            response = await self._request_clarification(intent)
        else:
            response = await self._handle_analysis_intent(intent)

        # Add response to conversation
        self.conversation.add_message("assistant", response)

        # Save state
        self.state_manager.save_conversation(self.conversation)
        if self.workflow:
            self.state_manager.save_workflow(self.workflow)

        return response

    async def _handle_analysis_intent(self, intent: Intent) -> str:
        """Handle an analysis-related intent."""
        # Check for high-stakes
        if intent.high_stakes:
            return await self._request_high_stakes_approval(intent)

        # Create workflow
        self.workflow = WorkflowState()
        self.workflow.advance_phase(WorkflowPhase.OPTION_PRESENTATION)

        # Generate options based on intent
        options = await self._generate_options(intent)

        # Present options
        return self._format_options_response(intent, options)

    async def _generate_options(self, intent: Intent) -> list[AgentOption]:
        """Generate analysis options based on intent."""
        # Use LLM to generate contextual options
        messages = [
            {
                "role": "user",
                "content": f"""Based on this user request, generate 2-3 analysis approaches.

User request: {self.conversation.messages[-1].content if self.conversation.messages else "No message"}

Intent detected: {intent.type.value}
Parameters extracted: {intent.parameters}

For each approach, provide:
1. A short title
2. Description of what this approach does
3. 2-3 pros
4. 2-3 cons
5. Estimated complexity (low/medium/high)

Recommend one option and explain why.""",
            }
        ]

        _response = await self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=0.7,
            system=self.system_prompt,
            messages=cast(list[MessageParam], messages),
        )

        # Response parsing is delegated to _get_default_options which provides
        # curated options based on intent type. LLM response available in _response
        # for future enhancement of dynamic option generation.
        return self._get_default_options(intent)

    def _get_default_options(self, intent: Intent) -> list[AgentOption]:
        """Get default options for common intents."""
        if intent.type == IntentType.STATISTICAL_ANALYSIS:
            return [
                AgentOption(
                    id="comprehensive",
                    title="Comprehensive EDA",
                    description="Full exploratory data analysis with all statistical tests",
                    recommended=True,
                    pros=["Thorough coverage", "Identifies all patterns", "Publication-ready"],
                    cons=["Takes longer", "May surface irrelevant findings"],
                    estimated_complexity="medium",
                ),
                AgentOption(
                    id="quick",
                    title="Quick Summary",
                    description="Basic descriptive statistics and key metrics",
                    pros=["Fast results", "Easy to interpret", "Good for initial look"],
                    cons=["May miss subtle patterns", "No hypothesis testing"],
                    estimated_complexity="low",
                ),
                AgentOption(
                    id="targeted",
                    title="Targeted Analysis",
                    description="Focus on specific columns or relationships",
                    pros=["Efficient", "Answers specific questions"],
                    cons=["Requires knowing what to look for"],
                    estimated_complexity="low",
                ),
            ]
        elif intent.type == IntentType.SENTIMENT_ANALYSIS:
            return [
                AgentOption(
                    id="marbert",
                    title="MARBERT Analysis",
                    description="Arabic-specific sentiment analysis using MARBERT model",
                    recommended=True,
                    pros=["Best for Arabic text", "Handles dialects", "High accuracy"],
                    cons=["Requires Arabic text", "Slower processing"],
                    estimated_complexity="medium",
                ),
                AgentOption(
                    id="multi",
                    title="Multi-lingual Analysis",
                    description="Handle mixed Arabic and English text",
                    pros=["Handles code-switching", "Comprehensive"],
                    cons=["Complex pipeline", "May need manual review"],
                    estimated_complexity="high",
                ),
            ]
        elif intent.type == IntentType.FORECAST:
            return [
                AgentOption(
                    id="prophet",
                    title="Prophet Forecasting",
                    description="Facebook Prophet for time series with seasonality",
                    recommended=True,
                    pros=["Handles seasonality", "Robust to missing data", "Confidence intervals"],
                    cons=["Requires date column", "May overfit short series"],
                    estimated_complexity="medium",
                ),
                AgentOption(
                    id="arima",
                    title="ARIMA Model",
                    description="Classical ARIMA for stationary time series",
                    pros=["Well-understood", "Good for short-term"],
                    cons=["Requires stationarity", "Parameter tuning needed"],
                    estimated_complexity="high",
                ),
                AgentOption(
                    id="simple",
                    title="Simple Methods",
                    description="Moving average and exponential smoothing",
                    pros=["Fast", "Easy to interpret", "Good baseline"],
                    cons=["May miss complex patterns"],
                    estimated_complexity="low",
                ),
            ]
        else:
            return [
                AgentOption(
                    id="default",
                    title="Standard Approach",
                    description="Default analysis workflow",
                    recommended=True,
                    pros=["Proven methodology", "Comprehensive"],
                    cons=["May not be optimized for specific needs"],
                    estimated_complexity="medium",
                ),
            ]

    def _format_options_response(self, intent: Intent, options: list[AgentOption]) -> str:
        """Format options into a user-friendly response."""
        lines = [
            f"Based on your request for **{intent.type.value.replace('_', ' ')}**, I've identified {len(options)} approaches:\n"
        ]

        for i, opt in enumerate(options, 1):
            rec = " (Recommended)" if opt.recommended else ""
            lines.append(f"**Option {i}: {opt.title}**{rec}")
            lines.append(f"- Description: {opt.description}")
            lines.append(f"- Pros: {', '.join(opt.pros)}")
            lines.append(f"- Cons: {', '.join(opt.cons)}")
            lines.append(f"- Complexity: {opt.estimated_complexity}")
            lines.append("")

        recommended = next((o for o in options if o.recommended), options[0])
        lines.append(
            f"**My Recommendation**: I recommend **{recommended.title}** because it provides the best balance of thoroughness and efficiency for your use case."
        )
        lines.append("")
        lines.append("Which approach would you like to proceed with? (Enter option number or name)")

        # Set pending approval
        self.conversation.pending_approval = {
            "type": "option_selection",
            "options": [o.id for o in options],
            "intent": intent.type.value,
        }

        return "\n".join(lines)

    async def _handle_approval_response(self, response: str) -> str:
        """Handle a response to a pending approval request."""
        pending = self.conversation.pending_approval
        if not pending:
            return "No pending approval request."

        approval_type = pending.get("type")

        if approval_type == "option_selection":
            # Parse selection
            response_lower = response.lower().strip()
            options = pending.get("options", [])

            # Try to match by number or name
            selected = None
            if response_lower.isdigit():
                idx = int(response_lower) - 1
                if 0 <= idx < len(options):
                    selected = options[idx]
            else:
                for opt in options:
                    if opt.lower() in response_lower or response_lower in opt.lower():
                        selected = opt
                        break

            if selected:
                self.conversation.record_approval(
                    action=f"Selected option: {selected}",
                    approved=True,
                    reason=pending.get("intent", ""),
                )
                self.conversation.clear_pending()
                # Set selected_option AFTER clear_pending (which clears it)
                self.conversation.selected_option = selected

                # Proceed with selected approach
                return await self._execute_selected_option(selected, pending.get("intent"))
            else:
                return (
                    f"I didn't understand your selection. Please choose from: {', '.join(options)}"
                )

        elif approval_type == "high_stakes":
            if any(word in response.lower() for word in ["yes", "approve", "proceed", "confirm"]):
                self.conversation.record_approval(
                    action=pending.get("action", ""),
                    approved=True,
                )
                self.conversation.clear_pending()
                return "Approved. Proceeding with the operation."
            else:
                self.conversation.record_approval(
                    action=pending.get("action", ""),
                    approved=False,
                    reason=response,
                )
                self.conversation.clear_pending()
                return "Operation cancelled. How else can I help you?"

        return "Unexpected approval state. Please start over."

    async def _execute_selected_option(self, option: str, intent: str | None) -> str:
        """Execute the selected analysis option."""
        if not self.workflow:
            self.workflow = WorkflowState()

        self.workflow.advance_phase(WorkflowPhase.DATA_RETRIEVAL)

        # Store the selected option and intent for the workflow
        self._agent_context.set_data("selected_option", option)
        self._agent_context.set_data("selected_intent", intent)

        # Check if data is already loaded
        loaded_data = self._agent_context.get_data("loaded_data")

        if loaded_data is not None:
            return await self._continue_analysis_with_data(option, intent)

        return f"""Excellent choice! Starting **{option}** analysis.

I'll now:
1. Load and validate your data
2. Apply necessary transformations
3. Perform the analysis
4. Generate insights

Please provide the data file path or let me know if the data is already loaded."""

    async def _continue_analysis_with_data(self, option: str, intent: str | None) -> str:
        """Continue analysis with already loaded data."""
        loaded_data = self._agent_context.get_data("loaded_data")

        if intent == IntentType.STATISTICAL_ANALYSIS.value:
            return await self._run_statistical_analysis(loaded_data, option)
        elif intent == IntentType.SENTIMENT_ANALYSIS.value:
            return await self._run_sentiment_analysis(loaded_data)
        elif intent == IntentType.FORECAST.value:
            return await self._run_forecast(loaded_data, option)
        else:
            # Default to statistical analysis
            return await self._run_statistical_analysis(loaded_data, option)

    async def _run_statistical_analysis(self, data: Any, option: str) -> str:
        """Run statistical analysis with the selected option."""
        if self.workflow is None:
            self.workflow = WorkflowState()
        self.workflow.advance_phase(WorkflowPhase.ANALYSIS)
        self.workflow.agents_used.append("statistical")

        statistical_agent = self._get_agent("statistical")

        # Map option to analysis type
        analysis_type_map = {
            "comprehensive": AnalysisType.COMPREHENSIVE,
            "quick": AnalysisType.DESCRIPTIVE,
            "targeted": AnalysisType.CORRELATION,
            "descriptive": AnalysisType.DESCRIPTIVE,
            "correlation": AnalysisType.CORRELATION,
            "distribution": AnalysisType.DISTRIBUTION,
        }
        analysis_type = analysis_type_map.get(option.lower(), AnalysisType.COMPREHENSIVE)

        result = await statistical_agent.run(data=data, analysis_type=analysis_type)

        if result.success and result.data:
            # Store results for insights generation
            self._agent_context.set_data("statistical_results", result.data.to_dict())

            # Format and return output
            formatted = statistical_agent.format_output(result.data)

            # Send notification
            self.notifier.notify_analysis_complete(
                analysis_type="Statistical Analysis",
                summary=f"Completed {option} analysis with {result.data.row_count if hasattr(result.data, 'row_count') else 'N/A'} rows",
            )

            # Create Obsidian note if available
            if self.obsidian and self.obsidian.is_available():
                self.obsidian.create_analysis_note(
                    {
                        "title": f"Statistical Analysis - {option.title()}",
                        "summary": f"Completed {option} statistical analysis",
                        "analysis_type": "statistical",
                        "methodology": option,
                    }
                )

            return f"## Statistical Analysis Complete\n\n{formatted}\n\nWould you like me to generate insights from these findings?"
        else:
            self.notifier.notify_error(
                error=result.error or "Unknown error",
                context="Statistical Analysis",
            )
            return f"Analysis encountered an issue: {result.error}"

    async def _run_sentiment_analysis(self, data: Any) -> str:
        """Run Arabic sentiment analysis."""
        if self.workflow is None:
            self.workflow = WorkflowState()
        self.workflow.advance_phase(WorkflowPhase.ANALYSIS)
        self.workflow.agents_used.append("arabic_nlp")

        arabic_agent = self._get_agent("arabic_nlp")

        # Check for text columns
        if hasattr(data, "columns"):
            text_cols = data.select_dtypes(include=["object"]).columns.tolist()
            if text_cols:
                # Use first text column
                text_data = data[text_cols[0]].dropna().tolist()
                result = await arabic_agent.run(text=text_data)

                if result.success and result.data:
                    formatted = arabic_agent.format_output(result.data)

                    # Send notification
                    self.notifier.notify_analysis_complete(
                        analysis_type="Arabic Sentiment Analysis",
                        summary=f"Analyzed {len(text_data)} text entries",
                    )

                    # Create Obsidian note if available
                    if self.obsidian and self.obsidian.is_available():
                        self.obsidian.create_analysis_note(
                            {
                                "title": "Arabic Sentiment Analysis",
                                "summary": f"Analyzed sentiment of {len(text_data)} Arabic text entries",
                                "analysis_type": "sentiment",
                                "methodology": "MARBERT",
                            }
                        )

                    return f"## Arabic NLP Analysis Complete\n\n{formatted}"
                else:
                    self.notifier.notify_error(
                        error=result.error or "Unknown error",
                        context="Arabic Sentiment Analysis",
                    )
                    return f"Analysis encountered an issue: {result.error}"

        return "No text columns found for sentiment analysis. Please specify a text column."

    async def _run_forecast(
        self,
        data: Any,
        option: str,
        target_column: str | None = None,
        date_column: str | None = None,
        periods: int = 30,
    ) -> str:
        """Run time series forecasting."""
        if self.workflow is None:
            self.workflow = WorkflowState()
        self.workflow.advance_phase(WorkflowPhase.ANALYSIS)
        self.workflow.agents_used.append("modeling")

        modeling_agent = self._get_agent("modeling")

        # Try to infer target and date columns if not specified
        if hasattr(data, "columns"):
            numeric_cols = data.select_dtypes(include=["number"]).columns.tolist()
            datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()

            # Check for object columns that might be dates
            if not datetime_cols:
                for col in data.select_dtypes(include=["object"]).columns:
                    try:
                        import pandas as pd

                        pd.to_datetime(data[col].dropna().head())
                        datetime_cols.append(col)
                    except (ValueError, TypeError):
                        pass

            if target_column is None and numeric_cols:
                target_column = numeric_cols[0]

            if date_column is None and datetime_cols:
                date_column = datetime_cols[0]

        if target_column is None:
            return (
                "Could not identify a numeric column for forecasting. Please specify target_column."
            )

        # Map option to model type
        model_type_map = {
            "prophet": ModelType.PROPHET,
            "arima": ModelType.ARIMA,
            "simple": ModelType.EXPONENTIAL_SMOOTHING,
            "exponential_smoothing": ModelType.EXPONENTIAL_SMOOTHING,
        }
        model_type = model_type_map.get(option.lower(), ModelType.EXPONENTIAL_SMOOTHING)

        result = await modeling_agent.run(
            data=data,
            target_column=target_column,
            date_column=date_column,
            task_type=TaskType.TIME_SERIES_FORECAST,
            model_type=model_type,
            periods=periods,
        )

        if result.success and result.data:
            # Store results for visualization
            self._agent_context.set_data("modeling_results", result.data.to_dict())

            formatted = modeling_agent.format_output(result.data)

            # Send notification
            self.notifier.notify_analysis_complete(
                analysis_type="Time Series Forecasting",
                summary=f"Generated {periods}-period forecast using {option}",
            )

            # Create Obsidian note if available
            if self.obsidian and self.obsidian.is_available():
                self.obsidian.create_analysis_note(
                    {
                        "title": f"Forecast - {target_column}",
                        "summary": f"Generated {periods}-period forecast for {target_column}",
                        "analysis_type": "forecast",
                        "methodology": option,
                    }
                )

            return f"## Forecasting Complete\n\n{formatted}\n\nWould you like to visualize these forecasts?"
        else:
            self.notifier.notify_error(
                error=result.error or "Unknown error",
                context="Time Series Forecasting",
            )
            return f"Forecasting encountered an issue: {result.error}"

    async def run_classification(
        self,
        target_column: str,
        feature_columns: list[str] | None = None,
        model_type: str | None = None,
    ) -> str:
        """Run classification modeling."""
        loaded_data = self._agent_context.get_data("loaded_data")

        if loaded_data is None:
            return "No data loaded. Please load data first."

        if not self.workflow:
            self.workflow = WorkflowState()

        self.workflow.advance_phase(WorkflowPhase.ANALYSIS)
        self.workflow.agents_used.append("modeling")

        modeling_agent = self._get_agent("modeling")

        # Parse model type
        model_type_enum = None
        if model_type:
            model_type_map = {
                "logistic": ModelType.LOGISTIC_REGRESSION,
                "logistic_regression": ModelType.LOGISTIC_REGRESSION,
                "random_forest": ModelType.RANDOM_FOREST_CLASSIFIER,
                "gradient_boosting": ModelType.GRADIENT_BOOSTING_CLASSIFIER,
            }
            model_type_enum = model_type_map.get(model_type.lower())

        result = await modeling_agent.run(
            data=loaded_data,
            target_column=target_column,
            feature_columns=feature_columns,
            task_type=TaskType.CLASSIFICATION,
            model_type=model_type_enum,
        )

        if result.success and result.data:
            self._agent_context.set_data("modeling_results", result.data.to_dict())

            formatted = modeling_agent.format_output(result.data)

            # Send notification
            self.notifier.notify_analysis_complete(
                analysis_type="Classification Modeling",
                summary=f"Built classification model for {target_column}",
            )

            # Create Obsidian note if available
            if self.obsidian and self.obsidian.is_available():
                self.obsidian.create_analysis_note(
                    {
                        "title": f"Classification Model - {target_column}",
                        "summary": f"Built classification model to predict {target_column}",
                        "analysis_type": "classification",
                        "methodology": model_type or "auto-selected",
                    }
                )

            return f"## Classification Complete\n\n{formatted}"
        else:
            self.notifier.notify_error(
                error=result.error or "Unknown error",
                context="Classification Modeling",
            )
            return f"Classification encountered an issue: {result.error}"

    async def run_regression(
        self,
        target_column: str,
        feature_columns: list[str] | None = None,
        model_type: str | None = None,
    ) -> str:
        """Run regression modeling."""
        loaded_data = self._agent_context.get_data("loaded_data")

        if loaded_data is None:
            return "No data loaded. Please load data first."

        if not self.workflow:
            self.workflow = WorkflowState()

        self.workflow.advance_phase(WorkflowPhase.ANALYSIS)
        self.workflow.agents_used.append("modeling")

        modeling_agent = self._get_agent("modeling")

        # Parse model type
        model_type_enum = None
        if model_type:
            model_type_map = {
                "linear": ModelType.LINEAR_REGRESSION,
                "linear_regression": ModelType.LINEAR_REGRESSION,
                "ridge": ModelType.RIDGE_REGRESSION,
                "lasso": ModelType.LASSO_REGRESSION,
                "random_forest": ModelType.RANDOM_FOREST_REGRESSOR,
                "gradient_boosting": ModelType.GRADIENT_BOOSTING_REGRESSOR,
            }
            model_type_enum = model_type_map.get(model_type.lower())

        result = await modeling_agent.run(
            data=loaded_data,
            target_column=target_column,
            feature_columns=feature_columns,
            task_type=TaskType.REGRESSION,
            model_type=model_type_enum,
        )

        if result.success and result.data:
            self._agent_context.set_data("modeling_results", result.data.to_dict())

            formatted = modeling_agent.format_output(result.data)

            # Send notification
            self.notifier.notify_analysis_complete(
                analysis_type="Regression Modeling",
                summary=f"Built regression model for {target_column}",
            )

            # Create Obsidian note if available
            if self.obsidian and self.obsidian.is_available():
                self.obsidian.create_analysis_note(
                    {
                        "title": f"Regression Model - {target_column}",
                        "summary": f"Built regression model to predict {target_column}",
                        "analysis_type": "regression",
                        "methodology": model_type or "auto-selected",
                    }
                )

            return f"## Regression Complete\n\n{formatted}"
        else:
            self.notifier.notify_error(
                error=result.error or "Unknown error",
                context="Regression Modeling",
            )
            return f"Regression encountered an issue: {result.error}"

    async def load_data(self, file_path: str) -> str:
        """
        Load data from a file.

        Args:
            file_path: Path to the data file

        Returns:
            Status message
        """
        retrieval_agent = self._get_agent("retrieval")
        result = await retrieval_agent.run(file_path=file_path)

        if result.success and result.data:
            # Store loaded data
            self._agent_context.set_data("loaded_data", result.data.data)
            self._agent_context.set_data("data_profile", result.data.profile.to_dict())
            self._agent_context.set_data("data_quality", result.data.quality.to_dict())

            if self.workflow:
                self.workflow.data_checksums[file_path] = result.data.profile.checksum
                self.workflow.agents_used.append("retrieval")

            # Format output
            profile_output = retrieval_agent.format_profile_output(result.data.profile)
            quality_output = retrieval_agent.format_quality_output(result.data.quality)

            return f"{profile_output}\n\n{quality_output}"
        else:
            return f"Failed to load data: {result.error}"

    async def generate_insights(self) -> str:
        """Generate insights from analysis results."""
        statistical_results = self._agent_context.get_data("statistical_results")
        data_profile = self._agent_context.get_data("data_profile")

        if not statistical_results and not data_profile:
            return "No analysis results available. Please run an analysis first."

        insights_agent = self._get_agent("insights")
        result = await insights_agent.run(
            analysis_results=statistical_results,
            data_profile=data_profile,
        )

        if result.success and result.data:
            if self.workflow:
                self.workflow.agents_used.append("insights")

            # Store insights for visualization and report generation
            self._agent_context.set_data("insights_results", result.data.to_dict())

            formatted = insights_agent.format_output(result.data)
            return f"{formatted}"
        else:
            return f"Failed to generate insights: {result.error}"

    async def generate_visualizations(
        self,
        chart_requests: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Generate visualizations from analysis results.

        Args:
            chart_requests: Optional specific chart requests

        Returns:
            Status message with visualization summary
        """
        loaded_data = self._agent_context.get_data("loaded_data")
        statistical_results = self._agent_context.get_data("statistical_results")

        if loaded_data is None and not statistical_results:
            return (
                "No data or analysis results available. Please load data or run an analysis first."
            )

        visualization_agent = self._get_agent("visualization")
        result = await visualization_agent.run(
            data=loaded_data,
            analysis_results=statistical_results,
            chart_requests=chart_requests,
        )

        if result.success and result.data:
            if self.workflow:
                self.workflow.agents_used.append("visualization")
                self.workflow.advance_phase(WorkflowPhase.OUTPUT_GENERATION)

            # Store visualizations for report generation
            self._agent_context.set_data("visualization_results", result.data.to_dict())

            formatted = visualization_agent.format_output(result.data)
            return f"## Visualizations Generated\n\n{formatted}\n\nWould you like to generate a report with these visualizations?"
        else:
            return f"Failed to generate visualizations: {result.error}"

    async def generate_report(
        self,
        format: str = "markdown",
        audience: str = "general",
        title: str = "Analytics Report",
        output_path: str | None = None,
        draft_only: bool = False,
    ) -> str:
        """
        Generate a formatted report.

        Args:
            format: Output format (pdf, pptx, html, markdown)
            audience: Target audience (executive, technical, general)
            title: Report title
            output_path: Path to save the report
            draft_only: If True, return draft structure for approval

        Returns:
            Status message with report details
        """
        insights_results = self._agent_context.get_data("insights_results")
        visualization_results = self._agent_context.get_data("visualization_results")
        statistical_results = self._agent_context.get_data("statistical_results")

        if not insights_results and not visualization_results and not statistical_results:
            return (
                "No content available for report. Please run analysis and generate insights first."
            )

        report_agent = self._get_agent("report")
        result = await report_agent.run(
            insights=insights_results,
            visualizations=visualization_results,
            analysis_results=statistical_results,
            format=format,
            audience=audience,
            title=title,
            output_path=output_path,
            draft_only=draft_only,
        )

        if result.success and result.data:
            if self.workflow:
                self.workflow.agents_used.append("report")
                if not draft_only:
                    self.workflow.advance_phase(WorkflowPhase.COMPLETED)

            formatted = report_agent.format_output(result.data)

            if draft_only:
                # Set pending approval for draft
                self.conversation.pending_approval = {
                    "type": "report_draft",
                    "draft": result.data.to_dict(),
                    "format": format,
                    "title": title,
                }
                self.notifier.notify_approval_required(
                    action="Report Generation",
                    reason="Draft report ready for review",
                )
                return f"## Report Draft\n\n{formatted}\n\nDo you approve this report structure? (yes/no or provide feedback)"
            else:
                # Send notification for completed report
                self.notifier.notify_analysis_complete(
                    analysis_type="Report Generation",
                    summary=f"Generated {format.upper()} report: {title}",
                    output_path=output_path,
                )

                # Create Obsidian note if available
                if self.obsidian and self.obsidian.is_available():
                    self.obsidian.create_analysis_note(
                        {
                            "title": f"Report: {title}",
                            "summary": f"Generated {format.upper()} report for {audience} audience",
                            "analysis_type": "report",
                            "methodology": f"{format} format, {audience} audience",
                        }
                    )

                return f"## Report Generated\n\n{formatted}"
        else:
            self.notifier.notify_error(
                error=result.error or "Unknown error",
                context="Report Generation",
            )
            return f"Failed to generate report: {result.error}"

    async def get_visualization_options(self) -> str:
        """Get available visualization options."""
        visualization_agent = self._get_agent("visualization")
        loaded_data = self._agent_context.get_data("loaded_data")
        options = visualization_agent.get_chart_type_options(loaded_data)

        return self._format_options_response(
            Intent(
                type=IntentType.VISUALIZE,
                confidence=1.0,
                parameters={},
                agents_required=["visualization"],
            ),
            options,
        )

    async def get_report_options(self) -> str:
        """Get available report format options."""
        report_agent = self._get_agent("report")
        format_options = report_agent.get_format_options()
        audience_options = report_agent.get_audience_options()

        lines = [
            "## Report Generation Options",
            "",
            "### Format Options",
            "",
        ]

        for opt in format_options:
            rec = " (Recommended)" if opt.recommended else ""
            lines.append(f"**{opt.title}**{rec}")
            lines.append(f"- Description: {opt.description}")
            lines.append(f"- Pros: {', '.join(opt.pros)}")
            lines.append(f"- Cons: {', '.join(opt.cons)}")
            lines.append("")

        lines.extend(
            [
                "### Audience Options",
                "",
            ]
        )

        for opt in audience_options:
            rec = " (Recommended)" if opt.recommended else ""
            lines.append(f"**{opt.title}**{rec}")
            lines.append(f"- Description: {opt.description}")
            lines.append("")

        lines.append("Which format and audience would you like? (e.g., 'pdf for executives')")

        return "\n".join(lines)

    async def _request_clarification(self, intent: Intent) -> str:
        """Request clarification from the user."""
        if intent.clarifications_needed:
            return "I need a bit more information to proceed:\n\n" + "\n".join(
                f"- {c}" for c in intent.clarifications_needed
            )
        return "I'm not sure I understood your request. Could you please rephrase or provide more details?"

    async def _request_high_stakes_approval(self, intent: Intent) -> str:
        """Request approval for high-stakes operations."""
        self.conversation.pending_approval = {
            "type": "high_stakes",
            "action": intent.type.value,
            "reasons": intent.high_stakes_reasons,
        }

        lines = [
            "**High-Stakes Operation Detected**",
            "",
            f"Your request involves: {intent.type.value.replace('_', ' ')}",
            "",
            "Reasons for confirmation:",
        ]
        for reason in intent.high_stakes_reasons:
            lines.append(f"- {reason}")

        lines.extend(
            [
                "",
                "Do you want to proceed? (yes/no)",
            ]
        )

        return "\n".join(lines)

    def _generate_help_response(self) -> str:
        """Generate a help response."""
        return """# The Analyst - Help

I'm your AI-powered analytics assistant. Here's what I can do:

## Data Operations
- **Load data**: "Load the data from sales.csv"
- **Explore data**: "Show me the data profile"
- **Transform data**: "Clean the data and handle missing values"

## Analysis
- **Statistical analysis**: "Analyze the distribution of revenue"
- **Sentiment analysis**: "Analyze the sentiment of Arabic comments"
- **Correlations**: "What's the correlation between views and engagement?"
- **Trends**: "Show me the trend over time"

## Predictions
- **Forecasting**: "Forecast next 30 days of traffic"
- **Classification**: "Classify articles by topic"

## Outputs
- **Visualizations**: "Create a chart of monthly revenue"
- **Reports**: "Generate an executive report in PDF"

## Tips
- I'll always present options and wait for your approval
- I'll explain my methodology and reasoning
- Ask for clarification anytime!

What would you like to analyze today?"""

    def _generate_status_response(self) -> str:
        """Generate a status response."""
        if not self.workflow:
            return "No active workflow. Start by telling me what you'd like to analyze."

        return f"""# Current Status

**Workflow ID**: {self.workflow.workflow_id}
**Phase**: {self.workflow.phase.value}
**Agents Used**: {', '.join(self.workflow.agents_used) or 'None yet'}

**Data Files**: {len(self.workflow.data_checksums)} loaded
**Transformations**: {len(self.workflow.transformations)} applied
**Errors**: {len(self.workflow.errors)}

What would you like to do next?"""


async def main() -> None:
    """Main entry point for the orchestrator."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    orchestrator = Orchestrator()
    print("The Analyst - AI-Powered Analytics Platform")
    print("Type 'help' for available commands, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            response = await orchestrator.process_message(user_input)
            print(f"\nAnalyst: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            logger.exception("Error processing message")


if __name__ == "__main__":
    asyncio.run(main())
