"""
Command-line interface for The Analyst platform.

Provides commands for running analyses, generating reports,
and interactive exploration.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

app = typer.Typer(
    name="analyst",
    help="AI-powered analytics platform for media companies",
    add_completion=False,
)
console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    # Set level directly on root logger to work even if already configured
    logging.root.setLevel(level)
    # Only configure if not already set up
    if not logging.root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )


def print_banner() -> None:
    """Print the application banner."""
    banner = r"""
 _____ _            _                _           _
|_   _| |__   ___  / \   _ __   __ _| |_   _ ___| |_
  | | | '_ \ / _ \/ _ \ | '_ \ / _` | | | | / __| __|
  | | | | | |  __/ ___ \| | | | (_| | | |_| \__ \ |_
  |_| |_| |_|\___/_/   \_\_| |_|\__,_|_|\__, |___/\__|
                                        |___/
    """
    console.print(banner, style="bold blue")
    console.print("AI-Powered Analytics Platform", style="italic")
    console.print()


@app.command()
def analyze(
    file: Path = typer.Argument(
        ...,
        help="Path to the data file to analyze",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("report.md"),
        "--output",
        "-o",
        help="Output file path for the report",
    ),
    analysis_type: str = typer.Option(
        "comprehensive",
        "--type",
        "-t",
        help="Type of analysis: comprehensive, descriptive, correlation, distribution",
    ),
    target: str | None = typer.Option(
        None,
        "--target",
        help="Target column for analysis",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Run full analysis workflow on a data file.

    Loads the data, performs analysis, generates insights, and creates a report.
    """
    setup_logging(verbose)
    print_banner()

    console.print(f"[bold]Analyzing:[/bold] {file}")
    console.print(f"[bold]Analysis type:[/bold] {analysis_type}")
    console.print(f"[bold]Output:[/bold] {output}")
    console.print()

    async def run_analysis() -> None:
        from src.orchestrator.main import Orchestrator

        orchestrator = Orchestrator()

        # Load data
        with console.status("[bold green]Loading data..."):
            result = await orchestrator.load_data(str(file))
            console.print(Markdown(result))

        # Run analysis
        with console.status(f"[bold green]Running {analysis_type} analysis..."):
            # Simulate selecting the analysis option
            orchestrator.conversation.selected_option = analysis_type
            orchestrator._agent_context.set_data("selected_option", analysis_type)

            loaded_data = orchestrator._agent_context.get_data("loaded_data")
            if loaded_data is not None:
                result = await orchestrator._run_statistical_analysis(loaded_data, analysis_type)
                console.print(Markdown(result))

        # Generate insights
        with console.status("[bold green]Generating insights..."):
            result = await orchestrator.generate_insights()
            console.print(Markdown(result))

        # Save to output
        insights_results = orchestrator._agent_context.get_data("insights_results")
        statistical_results = orchestrator._agent_context.get_data("statistical_results")

        report_content = ["# Analysis Report", ""]

        if statistical_results:
            report_content.append("## Statistical Analysis")
            report_content.append(str(statistical_results))
            report_content.append("")

        if insights_results:
            report_content.append("## Insights")
            report_content.append(str(insights_results))

        output.write_text("\n".join(report_content))
        console.print(f"\n[bold green]Report saved to:[/bold green] {output}")

    asyncio.run(run_analysis())


@app.command()
def forecast(
    file: Path = typer.Argument(
        ...,
        help="Path to the time series data file",
        exists=True,
        readable=True,
    ),
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Column to forecast",
    ),
    periods: int = typer.Option(
        30,
        "--periods",
        "-p",
        help="Number of periods to forecast",
    ),
    date_column: str | None = typer.Option(
        None,
        "--date",
        "-d",
        help="Date column name",
    ),
    method: str = typer.Option(
        "exponential_smoothing",
        "--method",
        "-m",
        help="Forecasting method: exponential_smoothing, prophet, arima",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Generate forecast for time series data.

    Supports exponential smoothing, Prophet, and ARIMA methods.
    """
    setup_logging(verbose)
    print_banner()

    console.print(f"[bold]Forecasting:[/bold] {file}")
    console.print(f"[bold]Target column:[/bold] {target}")
    console.print(f"[bold]Periods:[/bold] {periods}")
    console.print(f"[bold]Method:[/bold] {method}")
    console.print()

    async def run_forecast() -> None:
        from src.orchestrator.main import Orchestrator

        orchestrator = Orchestrator()

        # Load data
        with console.status("[bold green]Loading data..."):
            result = await orchestrator.load_data(str(file))
            console.print(Markdown(result))

        # Run forecast
        with console.status(f"[bold green]Running {method} forecast..."):
            loaded_data = orchestrator._agent_context.get_data("loaded_data")
            if loaded_data is not None:
                result = await orchestrator._run_forecast(
                    loaded_data,
                    method,
                    target_column=target,
                    date_column=date_column,
                    periods=periods,
                )
                console.print(Markdown(result))

    asyncio.run(run_forecast())


@app.command()
def sentiment(
    file: Path = typer.Argument(
        ...,
        help="Path to the data file with Arabic text",
        exists=True,
        readable=True,
    ),
    text_column: str = typer.Option(
        ...,
        "--column",
        "-c",
        help="Column containing Arabic text",
    ),
    output: Path = typer.Option(
        Path("sentiment_results.md"),
        "--output",
        "-o",
        help="Output file path",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Run Arabic sentiment analysis on text data.

    Uses MARBERT for Arabic-specific sentiment classification.
    """
    setup_logging(verbose)
    print_banner()

    console.print(f"[bold]Analyzing:[/bold] {file}")
    console.print(f"[bold]Text column:[/bold] {text_column}")
    console.print()

    async def run_sentiment() -> None:
        from src.orchestrator.main import Orchestrator

        orchestrator = Orchestrator()

        # Load data
        with console.status("[bold green]Loading data..."):
            result = await orchestrator.load_data(str(file))
            console.print(Markdown(result))

        # Run sentiment analysis
        with console.status("[bold green]Running Arabic sentiment analysis..."):
            loaded_data = orchestrator._agent_context.get_data("loaded_data")
            if loaded_data is not None:
                result = await orchestrator._run_sentiment_analysis(loaded_data)
                console.print(Markdown(result))

                output.write_text(result)
                console.print(f"\n[bold green]Results saved to:[/bold green] {output}")

    asyncio.run(run_sentiment())


@app.command()
def report(
    format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown, html, pdf, pptx",
    ),
    audience: str = typer.Option(
        "general",
        "--audience",
        "-a",
        help="Target audience: executive, technical, general",
    ),
    title: str = typer.Option(
        "Analytics Report",
        "--title",
        "-t",
        help="Report title",
    ),
    output: Path = typer.Option(
        Path("report.md"),
        "--output",
        "-o",
        help="Output file path",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
) -> None:
    """
    Generate a formatted report from analysis results.

    Requires previous analysis to have been run.
    """
    setup_logging(verbose)

    console.print(f"[bold]Generating report:[/bold] {title}")
    console.print(f"[bold]Format:[/bold] {format}")
    console.print(f"[bold]Audience:[/bold] {audience}")
    console.print()

    async def run_report() -> None:
        from src.orchestrator.main import Orchestrator

        orchestrator = Orchestrator()

        with console.status("[bold green]Generating report..."):
            result = await orchestrator.generate_report(
                format=format,
                audience=audience,
                title=title,
                output_path=str(output),
            )
            console.print(Markdown(result))

    asyncio.run(run_report())


@app.command()
def interactive() -> None:
    """
    Start interactive REPL mode.

    Allows conversational interaction with The Analyst.
    """
    print_banner()

    console.print("[bold]Interactive Mode[/bold]")
    console.print("Type 'help' for available commands, 'quit' to exit.")
    console.print()

    async def run_interactive() -> None:
        from src.orchestrator.main import Orchestrator

        orchestrator = Orchestrator()

        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]")

                if not user_input.strip():
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    console.print("[bold]Goodbye![/bold]")
                    break

                with console.status("[bold green]Thinking..."):
                    response = await orchestrator.process_message(user_input)

                console.print()
                console.print(
                    Panel(
                        Markdown(response),
                        title="[bold]Analyst[/bold]",
                        border_style="green",
                    )
                )
                console.print()

            except KeyboardInterrupt:
                console.print("\n[bold]Goodbye![/bold]")
                break
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {e}")

    asyncio.run(run_interactive())


@app.command()
def version() -> None:
    """Show version information."""
    console.print("[bold]The Analyst[/bold] v0.1.0")
    console.print("AI-powered analytics orchestration system")


@app.callback()
def main() -> None:
    """
    The Analyst - AI-Powered Analytics Platform

    An opinionated, multi-agent orchestration system for media company analytics.
    """
    pass


def cli() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
