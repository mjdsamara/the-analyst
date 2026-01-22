# The Analyst

AI-powered analytics orchestration platform for media companies.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Tests: pytest](https://img.shields.io/badge/tests-pytest-green.svg)](https://pytest.org/)
[![codecov](https://codecov.io/gh/mjdsamara/the-analyst/branch/main/graph/badge.svg)](https://codecov.io/gh/mjdsamara/the-analyst)

## Overview

The Analyst is an opinionated, AI-powered analytics orchestration system designed for media company analytics. It features a main orchestrator agent coordinating 8 specialized sub-agents for data retrieval, transformation, statistical analysis, Arabic NLP, predictive modeling, visualization, and report generation.

**Core Philosophy**: Advisory mode by default. Always present options with reasoning, wait for human approval on significant decisions.

## Features

- **Multi-agent Architecture** - 8 specialized agents with single responsibilities
- **Human-in-Loop** - Advisory mode ensures transparency and control
- **Statistical Rigor** - All analyses include methodology, confidence intervals, limitations
- **Arabic NLP Support** - MARBERT + CAMeL Tools for Arabic text analysis
- **Predictive Modeling** - Time series forecasting, classification, regression
- **Interactive Visualizations** - Plotly-based charts with export options
- **Multi-format Reports** - PDF, PPTX, HTML, Markdown output
- **Claude Code Integration** - Slash commands, hooks, and agent definitions

## Quick Start

```bash
# Clone repository
git clone <repository-url>
cd the-analyst

# Setup environment
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and DATABASE_URL

# Verify installation
pytest
./scripts/verify-boris-compliance.sh

# Run
python -m src.orchestrator.main
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## Slash Commands

| Command | Description |
|---------|-------------|
| `/analyze` | Start analysis workflow (statistical, trend, correlation, full) |
| `/ingest` | Load data from file or API source |
| `/report` | Generate formatted report (PDF, PPTX, HTML, Markdown) |
| `/sentiment` | Arabic sentiment analysis with dialect detection |
| `/forecast` | Predictive modeling and time series forecasting |

## Project Structure

```
the-analyst/
├── CLAUDE.md              # Project context for Claude Code
├── SETUP.md               # Detailed setup instructions
├── README.md              # This file
├── src/
│   ├── config.py          # Configuration and constraints
│   ├── cli.py             # Typer CLI interface
│   ├── orchestrator/      # Main orchestrator agent
│   │   ├── main.py        # Orchestrator implementation
│   │   ├── router.py      # Intent routing
│   │   └── state.py       # Workflow state management
│   ├── agents/            # Specialized sub-agents
│   │   ├── base.py        # BaseAgent class
│   │   ├── retrieval.py   # Data ingestion
│   │   ├── transform.py   # Data cleaning
│   │   ├── statistical.py # Statistical analysis
│   │   ├── arabic_nlp.py  # Arabic NLP
│   │   ├── modeling.py    # Predictive models
│   │   ├── insights.py    # Finding synthesis
│   │   ├── visualization.py # Chart generation
│   │   └── report.py      # Report generation
│   ├── prompts/           # System prompts
│   ├── tools/             # Shared utilities
│   ├── models/            # Pydantic schemas
│   ├── database/          # Async PostgreSQL layer
│   └── utils/             # Logging, notifications
├── .claude/
│   ├── settings.json      # Claude Code permissions
│   ├── agents/            # Agent definitions
│   ├── commands/          # Slash commands
│   └── hooks/             # Pre/Post tool hooks
├── tests/                 # Test suite (830+ tests)
├── scripts/               # Verification scripts
├── data/                  # Data directories
│   ├── raw/               # Source data (read-only)
│   ├── processed/         # Transformed data
│   └── outputs/           # Analysis outputs
└── memory/                # Agent memory and context
```

## Agent Architecture

| Layer | Agent | Job | Autonomy |
|-------|-------|-----|----------|
| Orchestration | Orchestrator | Route requests, coordinate agents | Advisory |
| Data | Retrieval | Ingest data from files/APIs | Supervised |
| Data | Transform | Clean, reshape, prepare data | Supervised |
| Analysis | Statistical | Statistical analysis and EDA | Advisory |
| Analysis | Arabic NLP | Arabic sentiment, NER, topics | Supervised |
| Analysis | Modeling | Predictive models (forecast, classify, regress) | Advisory |
| Analysis | Insights | Synthesize findings into actionable insights | Advisory |
| Output | Visualization | Create interactive charts | Supervised |
| Output | Report | Generate formatted outputs | Advisory |

**Autonomy Levels:**
- **Advisory**: Presents options with reasoning, waits for approval
- **Supervised**: Proceeds with task, reports actions taken

## Development

```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest                              # Full suite
pytest --cov=src                    # With coverage
pytest tests/test_agents -v         # Agents only

# Type checking
mypy src/

# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Pre-commit hooks
pre-commit install
pre-commit run --all-files

# Boris compliance check
./scripts/verify-boris-compliance.sh
```

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Project philosophy, constraints, Claude Code reference |
| [SETUP.md](SETUP.md) | Detailed installation and configuration guide |
| [.env.example](.env.example) | Environment variable reference |
| [memory/context.md](memory/context.md) | Project history and decisions |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ (async) |
| LLM | Claude (Anthropic) |
| Database | PostgreSQL + SQLAlchemy (async) |
| Arabic NLP | MARBERT + CAMeL Tools |
| Statistics | SciPy, Statsmodels, Scikit-learn |
| Visualization | Plotly + Kaleido |
| Reports | WeasyPrint (PDF), python-pptx |
| CLI | Typer |
| Testing | pytest + pytest-asyncio |

## License

MIT
