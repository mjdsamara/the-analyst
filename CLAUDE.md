# The Analyst - AI-Powered Analytics Platform

> **Owner**: Head of Product & Analytics
> **Domain**: Media companies (IMI, Sky News Arabia, CNN, The National, etc.)
> **Architecture**: Multi-agent orchestration system

## Project Identity

The Analyst is an opinionated, AI-powered analytics orchestration system for media company analytics. It features a main orchestrator agent coordinating specialized sub-agents for data retrieval, analysis, Arabic NLP, predictive modeling, and report generation.

**Core Philosophy**: Advisory mode by default. Always present options with reasoning, wait for human approval on significant decisions.

---

## Agent Architecture

### Orchestrator Agent
- **Single Job**: Route requests, coordinate agents, ensure human approval
- **Model**: Claude Opus 4.5
- **Autonomy**: Advisory (always presents options)

> **Note**: All agents use Claude Opus 4.5.

### Data Layer Agents
| Agent | Single Job | Autonomy |
|-------|------------|----------|
| Retrieval | Ingest data from files/APIs | Supervised |
| Transform | Clean, reshape, prepare data | Supervised |

### Analysis Layer Agents
| Agent | Single Job | Autonomy |
|-------|------------|----------|
| Statistical | Statistical analysis and EDA | Advisory |
| Arabic NLP | Arabic sentiment, NER, topics | Supervised |
| Modeling | Predictive models | Advisory |
| Insights | Synthesize findings | Advisory |

### Output Layer Agents
| Agent | Single Job | Autonomy |
|-------|------------|----------|
| Visualization | Create charts | Supervised |
| Report | Generate formatted outputs | Advisory |

### Utility Layer Agents
| Agent | Single Job | Autonomy |
|-------|------------|----------|
| Workspace Janitor | Recommend cleanup after sessions | Advisory |

---

## NEVER-DO List (Hard Constraints)

These actions are **strictly prohibited**:

1. **Modify source data files** - Read-only access always
2. **Share user data with external services** - No data exfiltration
3. **Make analytical decisions without showing reasoning** - Transparency required
4. **Execute cost-incurring API calls without approval** - Budget protection
5. **Proceed with analysis without presenting options first** - Human-in-loop
6. **Skip verification steps for statistical claims** - Rigor required
7. **Output results without confidence intervals where applicable** - Uncertainty quantification

---

## High-Stakes Triggers

These keywords/patterns require explicit confirmation before proceeding:

```python
HIGH_STAKES_KEYWORDS = [
    "delete", "remove", "drop",           # Data operations
    "send", "share", "export to",         # External sharing
    "production", "stakeholder", "executive",  # High-visibility
    "forecast", "predict",                # Predictive claims
]

# Context-sensitive keywords (require action context)
HIGH_STAKES_CONTEXT_KEYWORDS = {
    "model": ["build", "train", "create", "deploy", "run", "fit"],
}
```

> **Note**: Keywords are matched with word boundaries to avoid false positives. Context-sensitive keywords like "model" only trigger when used with action verbs (e.g., "build a model" triggers, but "data model" does not).

Also trigger confirmation for:
- API calls with cost > $5 (configurable via `COST_ALERT_THRESHOLD`)
- Operations affecting > 5,000 rows (configurable via `MAX_ROWS_WITHOUT_CONFIRM`)
- External API calls to new endpoints

---

## Hook System

Claude Code hooks enforce BORIS compliance and protect data integrity. These are configured in `.claude/settings.json`.

### Pre-Tool Hooks

**`check-dangerous-commands.sh`** - Blocks dangerous shell operations:
- `rm -rf` commands
- `sudo` commands
- Any modifications to `data/raw/` directory

**`validate-path.sh`** - Validates file paths before write operations:
- Blocks writes to source data directories (`data/raw/`)
- Blocks writes to system directories
- Enforces output directory structure

### Post-Tool Hooks

**`verify-output.sh`** - Validates analysis outputs contain required elements:
- Checks for methodology section
- Verifies confidence intervals are included
- Ensures limitations are documented

### Hook Configuration

Hooks are configured in `.claude/settings.json` using `PreToolUse` and `PostToolUse` matchers. See the settings file for the full configuration.

---

## Common Mistakes to Avoid

### Data Handling
- **WRONG**: Loading entire large files into memory
- **RIGHT**: Use chunked reading for files > 100MB

### Arabic NLP
- **WRONG**: Using general-purpose models for Arabic sentiment
- **RIGHT**: Use MARBERT/CAMeL Tools for Arabic-specific tasks
- **WRONG**: Ignoring dialect variations (MSA vs. Gulf vs. Egyptian)
- **RIGHT**: Detect and handle dialect appropriately

### Statistical Analysis
- **WRONG**: Reporting p-values without effect sizes
- **RIGHT**: Always include effect sizes and confidence intervals
- **WRONG**: Drawing causal conclusions from correlational data
- **RIGHT**: Use appropriate causal language; correlation ≠ causation

### Visualization
- **WRONG**: Creating charts without titles, labels, or sources
- **RIGHT**: Every chart needs title, axis labels, and data source
- **WRONG**: Using misleading scales or truncated axes
- **RIGHT**: Start axes at zero unless there's explicit justification

### Reports
- **WRONG**: Presenting findings without methodology
- **RIGHT**: Every report includes methodology section
- **WRONG**: Generating final reports without draft approval
- **RIGHT**: Show draft structure, get approval, then finalize

---

## Verification Requirements

Every analysis output MUST include:

1. **Methodology**: Which statistical tests/methods and why
2. **Confidence**: Intervals, p-values, effect sizes
3. **Baseline**: Historical context or benchmarks
4. **Limitations**: Sample size concerns, data quality issues
5. **Reproducibility**: Parameters used, random seeds

---

## Directory Structure

```
the-analyst/
├── CLAUDE.md              # This file - project context
├── src/
│   ├── config.py          # Environment & constraints
│   ├── cli.py             # Main CLI module
│   ├── orchestrator/      # Main orchestrator agent
│   ├── agents/            # Specialized sub-agents
│   ├── prompts/           # System prompts for agents
│   ├── tools/             # Shared tools and utilities
│   ├── models/            # Data models and schemas
│   ├── middleware/        # BORIS compliance middleware
│   │   ├── audit.py       # Audit logging
│   │   ├── cost_tracking.py  # Cost monitoring
│   │   └── autonomy.py    # Autonomy enforcement
│   ├── database/          # Database layer
│   │   ├── client.py      # Database client
│   │   └── models.py      # ORM models
│   └── utils/             # Logging, notifications
├── .claude/
│   ├── settings.json      # Tool permissions
│   ├── commands/          # Slash commands
│   ├── agents/            # Agent definition markdown files
│   └── hooks/             # Verification hooks
├── .github/
│   └── workflows/         # CI/CD pipeline
├── scripts/               # Verification scripts
├── memory/                # Agent memory and context
├── notebooks/             # Jupyter notebooks
├── data/
│   ├── raw/               # Source data (read-only)
│   ├── processed/         # Transformed data
│   └── outputs/           # Analysis outputs
└── tests/
    ├── test_agents/       # Agent unit tests
    ├── test_orchestrator/ # Orchestrator tests
    ├── test_middleware/   # Middleware tests
    ├── test_database/     # Database tests
    ├── test_models/       # Data model tests
    ├── test_tools/        # Tool tests
    ├── test_utils/        # Utility tests
    ├── test_integration/  # End-to-end tests
    └── test_cli.py        # CLI tests
```

---

## Workflow: Standard Analysis

1. **User Request** → Orchestrator parses intent
2. **Present Options** → "I recommend A, B, or C because..."
3. **User Approves** → Selected approach proceeds
4. **Retrieval** → Load and validate data
5. **Transform** → Clean and prepare
6. **Analysis** → Statistical, NLP, Modeling (parallel where possible)
7. **Insights** → Synthesize findings
8. **Present Findings** → Ask for output format
9. **Generate Output** → Visualization + Report
10. **Notify** → Terminal, Desktop, Obsidian

---

## Technology Stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.11+ (async) |
| LLM | Claude Opus 4.5 (Anthropic) |
| Database | PostgreSQL |
| Arabic NLP | MARBERT + CAMeL Tools |
| Visualization | Plotly |
| Reports | WeasyPrint (PDF), python-pptx |

---

## Commands Reference

- `/analyze` - Start analysis workflow
- `/ingest` - Load data from file/source
- `/report` - Generate formatted report
- `/sentiment` - Arabic sentiment analysis
- `/forecast` - Predictive modeling

---

## Environment Variables

Required in `.env`:
```
ANTHROPIC_API_KEY=        # Claude API access
HUGGINGFACE_TOKEN=        # Arabic model inference
DATABASE_URL=             # PostgreSQL connection
```

Optional:
```
LOG_LEVEL=INFO
COST_ALERT_THRESHOLD=5.0              # Cost threshold in USD (default: 5.0)
MAX_ROWS_WITHOUT_CONFIRM=5000         # Row count threshold (default: 5000)
ORCHESTRATOR_MODEL=claude-opus-4-5-20251101
ANALYSIS_MODEL=claude-opus-4-5-20251101
UTILITY_MODEL=claude-opus-4-5-20251101

# Path configuration
DATA_PATH=./data                      # Base data directory
OUTPUT_PATH=./data/outputs            # Output directory

# Notification settings
ENABLE_DESKTOP_NOTIFICATIONS=true     # Toggle desktop notifications
OBSIDIAN_VAULT_PATH=                  # Optional Obsidian vault integration
```

---

## Quick Start Reference

### First Time Setup
```bash
# 1. Clone and enter project
git clone <repository>
cd the-analyst

# 2. Create environment
uv venv .venv
source .venv/bin/activate

# 3. Install dependencies
uv pip install -e ".[dev]"

# 4. Copy and configure environment
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 5. Verify setup
python -c "from src.config import get_settings; print('Config OK')"
pytest --co -q  # List tests without running
```

### Daily Development
```bash
# Activate environment
source .venv/bin/activate

# Run tests
pytest                        # All tests
pytest tests/test_agents      # Agent tests only
pytest -x --tb=short          # Stop on first failure

# Type checking
mypy src/

# Format code
black src/ tests/
ruff check src/ tests/ --fix

# Verify Boris compliance
./scripts/verify-boris-compliance.sh
```

---

## Testing Commands

### Running Tests
```bash
# Full test suite with coverage
pytest

# Specific test file
pytest tests/test_agents/test_statistical.py

# Specific test function
pytest tests/test_agents/test_statistical.py::test_correlation_analysis -v

# Run with verbose output
pytest -v

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run async tests only
pytest -m asyncio
```

### Test Organization
```
tests/
├── conftest.py              # Shared fixtures (mock clients, sample data)
├── test_cli.py              # CLI command tests
├── test_agents/             # Unit tests for each agent
│   ├── test_retrieval.py
│   ├── test_transform.py
│   ├── test_statistical.py
│   ├── test_arabic_nlp.py
│   ├── test_modeling.py
│   ├── test_insights.py
│   ├── test_visualization.py
│   └── test_report.py
├── test_orchestrator/       # Orchestrator unit tests
├── test_middleware/         # Middleware tests
├── test_database/           # Database integration tests
├── test_models/             # Data model tests
├── test_tools/              # Tool/utility tests
├── test_utils/              # Notification, logging tests
├── test_integration/        # End-to-end workflow tests
└── fixtures/                # Test data files
```

### Writing Tests
- Use descriptive names: `test_<function>_<scenario>_<expected>`
- Include docstrings explaining test purpose
- Use fixtures from `conftest.py` for common setup
- Always test error conditions and edge cases
- Mark async tests with `@pytest.mark.asyncio`

---

## Common Operations Guide

### Loading Data
```python
from src.agents.retrieval import RetrievalAgent
from src.agents.base import AgentContext

context = AgentContext(session_id="my-session")
agent = RetrievalAgent("retrieval", context)
result = await agent.run(file_path="data/raw/sample.csv")
```

### Running Analysis via CLI

#### CLI Command Reference

| Command | Arguments | Key Options |
|---------|-----------|-------------|
| `analyze` | `file` | `--output`, `--type` (comprehensive\|descriptive\|correlation\|distribution), `--target`, `--verbose` |
| `forecast` | `file` | `--target`, `--periods`, `--date`, `--method` (exponential_smoothing\|prophet\|arima), `--verbose` |
| `sentiment` | `file` | `--column`, `--output`, `--verbose` |
| `report` | (none) | `--format` (markdown\|html\|pdf\|pptx), `--audience`, `--title`, `--output`, `--verbose` |
| `interactive` | (none) | (none) - Conversational REPL mode |
| `version` | (none) | (none) - Show version info |

#### Example Commands
```bash
# Statistical analysis (comprehensive by default)
python -m src.cli analyze data/raw/viewership.csv --type comprehensive

# Correlation analysis on specific target
python -m src.cli analyze data/raw/metrics.csv --type correlation --target revenue

# Time series forecast with Prophet
python -m src.cli forecast data/raw/traffic.csv --target views --periods 30 --method prophet

# Arabic sentiment analysis
python -m src.cli sentiment data/raw/comments.csv --column content

# Generate PDF report
python -m src.cli report --format pdf --output reports/analysis.pdf --title "Q4 Analysis"

# Interactive mode
python -m src.cli interactive
```

### Adding a New Agent

1. **Create agent file**: `src/agents/new_agent.py`
   ```python
   from src.agents.base import BaseAgent, AgentResult

   class NewAgent(BaseAgent):
       """Single job: <describe the one thing this agent does>"""

       async def execute(self, **kwargs) -> AgentResult:
           # Implementation
           pass
   ```

2. **Add to config**: Update `AGENT_CONFIG` in `src/config.py`

3. **Create Claude definition**: `.claude/agents/new-agent.md`

4. **Add tests**: `tests/test_agents/test_new_agent.py`

5. **Update router**: Add routing logic in `src/orchestrator/router.py`

6. **Run verification**: `./scripts/verify-boris-compliance.sh`

---

## Troubleshooting

### Common Issues

**Issue: `ModuleNotFoundError: No module named 'src'`**
```bash
# Ensure installed in editable mode
uv pip install -e "."
# Or set PYTHONPATH
export PYTHONPATH="."
```

**Issue: `ANTHROPIC_API_KEY not found`**
```bash
# Check .env file exists
ls -la .env
# Copy from example if missing
cp .env.example .env
# Edit with your key
```

**Issue: Tests failing with database errors**
```bash
# Start PostgreSQL with Docker
docker-compose up -d db

# Wait for it to be ready
docker-compose logs db

# Or use SQLite for tests (check conftest.py)
```

**Issue: Pre-commit hooks failing**
```bash
# Run formatters manually first
black src/ tests/
ruff check src/ tests/ --fix
mypy src/

# Then retry
pre-commit run --all-files
```

**Issue: Arabic NLP models not loading**
```bash
# Install Arabic dependencies
uv pip install -e ".[arabic]"

# Check HuggingFace token
echo $HUGGINGFACE_TOKEN
```

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG python -m src.orchestrator.main

# Run specific agent with tracing
python -c "
import logging
import asyncio
logging.basicConfig(level=logging.DEBUG)

from src.agents.statistical import StatisticalAgent
from src.agents.base import AgentContext

async def debug():
    ctx = AgentContext(session_id='debug')
    agent = StatisticalAgent('statistical', ctx)
    result = await agent.run(data_path='data/raw/test_sample.csv')
    print(result)

asyncio.run(debug())
"
```

### Getting Help
- Check `memory/context.md` for domain context and preferences
- Review agent definitions in `.claude/agents/`
- Run Boris compliance: `./scripts/verify-boris-compliance.sh`
- See test examples in `tests/` for usage patterns
